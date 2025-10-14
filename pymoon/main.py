from typing import Union, Sequence
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from .config import *
from .geometry import get_normals, render_moon_face
from .photometry import _MODEL_FUNCS
from .shadows import _visible_fraction_from_elevation, compute_shadow
from .utils import resize_image

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

Vector = Union[np.ndarray, Sequence[float]]


def get_moon_mask(
    N,
    s_vec,
    *,
    pointlike_sun: bool = False,
    shadows: bool = False,
    model: str = "binary",
    obs_vec: Vector = (0, 0, 1),
    R_S: float = R_SUN,  # Sun radius [m]
    d_SM: float = D_MOON_SUN,  # Sun–Moon distance [m]
    theta_S: Union[float, None] = None,
    R_M: float = R_MOON,
    dem: Union[str, np.ndarray, None] = None,
    moonmap: Union[str, np.ndarray, None] = None,
    default_res: tuple = (720, 1440),
    # photometry parameters
    k: float = 0.4,  # LS/Lambert blend
    B0: float = 0.45,  # opposition bump amplitude
    sigma_deg: float = 5.0,  # opposition width (deg)
    shadow_max_steps: int = 10,
    gamma_corr=1.0, 
):
    
    # Load DEM
    if dem is None:
        # default to 0 elevation
        dem_arr = np.zeros(default_res, dtype=float)
    elif isinstance(dem, str):
        dem_arr = rasterio.open(dem).read(1).astype(float)
    else:
        dem_arr = np.asarray(dem, float)

    # Compute surface normals
    n = get_normals(dem_arr, R_M=R_M)  # shape (3, h, w)
    n = np.moveaxis(n, 0, -1)  # shape (h, w, 3)

    # Normalize sun vector
    s = np.asarray(s_vec, float)
    s /= np.linalg.norm(s)

    # Normalize observer vector
    v = np.asarray(obs_vec, float)
    v /= np.linalg.norm(v)

    # Cosines
    mu0 = np.dot(n, s)  # incidence cosine
    mu = np.dot(n, v)  # emission cosine
    mu0 = np.maximum(mu0, 0.0)
    mu = np.maximum(mu, 0.0)

    # Visibility fraction
    if not pointlike_sun:
        # solar angular radius
        theta_S = theta_S if theta_S is not None else np.arctan(R_S / d_SM)
        # solar elevation
        epsilon = np.arcsin(np.clip(np.dot(n, s), -1.0, 1.0))
        f = _visible_fraction_from_elevation(epsilon, theta_S)
    else:
        f = (mu0 > 0).astype(np.float64)

    # Photometry
    if model not in _MODEL_FUNCS:
        raise ValueError(f"Unknown photometry model '{model}'")
    mask = _MODEL_FUNCS[model](
        mu0, mu, f, k=k, eps=1e-3, s_vec=s, B0=B0, sigma_deg=sigma_deg
    )
    mask /= np.nanmax(mask) if np.nanmax(mask) > 0 else 1.0
    moon_img = np.ones_like(mask)

    if shadows:
        mask = compute_shadow(
            mask, 
            dem_arr,
            s_vec,
            R_M=R_M,
            theta_S=theta_S,
            pointlike_sun=pointlike_sun,
            max_steps=shadow_max_steps,
        )
    else:
        if dem is not None:
            dem_flat = np.zeros(default_res, dtype=float)
            n_flat = get_normals(dem_flat, R_M=R_M)  # shape (3, h, w)
            n_flat = np.moveaxis(n_flat, 0, -1)  # shape (h, w, 3)
            mu0_flat = np.dot(n_flat, s)
            dark = mu0_flat <= 0.0
            mask[dark] *= mask[dark] 

    # Apply gamma correction
    if gamma_corr != 1.0:
        mask = np.clip(mask, 0.0, 1.0) ** (1.0 / gamma_corr)

    if moonmap is not None:
        # Load moon texture
        if isinstance(moonmap, str):
            moon_img = rasterio.open(moonmap).read(1).astype(float)
        elif isinstance(moonmap, np.ndarray):
            moon_img = np.asarray(moonmap, float)
        else:
            raise ValueError("moonmap must be a file path or a numpy array.")

        if moon_img.shape != dem_arr.shape:
            # Resample moonmap to DEM shape
            moon_img = resize_image(moon_img, dem_arr.shape)
        
        
    moonface = render_moon_face(moon_img*mask, obs_vec, out_px=N)

    return moonface


    
        




        
