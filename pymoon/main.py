from typing import Union, Sequence, Optional
import warnings
import colorsys

from PIL import Image
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
    gamma_corr: Union[float, None] = 1.0,
    mask_mode: str = "multiply",
    outside_color: Optional[Union[str, Sequence[float]]] = None,
    radius: float = 1.0,
    return_mask=False, 
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
    moon_tex = None

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
        # this part is necessary to avoid supurious white pixels when not computing shadows in the case of non flat DEM
        if dem is not None:
            dem_flat = np.zeros(default_res, dtype=float)
            n_flat = get_normals(dem_flat, R_M=R_M)  # shape (3, h, w)
            n_flat = np.moveaxis(n_flat, 0, -1)  # shape (h, w, 3)
            mu0_flat = np.dot(n_flat, s)
            dark = mu0_flat <= 0.0
            mask[dark] *= mu0_flat[dark]


    # Apply gamma correction
    if gamma_corr is not None:
        mask = np.clip(mask, 0.0, 1.0) ** (1.0 / gamma_corr)


    if return_mask:
        return mask

    if moonmap is not None:
        # Load moon texture
        if isinstance(moonmap, str):
            moon_tex = rasterio.open(moonmap).read([1, 2, 3]).astype(float)
            moon_tex = np.moveaxis(moon_tex, 0, -1)  # shape (h, w, 3)
        elif isinstance(moonmap, np.ndarray):
            moon_tex = np.asarray(moonmap, float)
        else:
            raise ValueError("moonmap must be a file path or a numpy array.")

        if moon_tex.shape != dem_arr.shape:
            # Resample moonmap to DEM shape
            moon_tex = resize_image(moon_tex, dem_arr.shape)

    moonface = render_moon_face(moon_tex, mask, obs_vec, out_px=N, mask_mode=mask_mode, outside_color=outside_color, radius=radius)

    return moonface

    
def tint_moon_image(
    moon_rgba: Union[np.ndarray, Image.Image],
    color: Union[Sequence[float], float],
    *,
    color_space: str = "rgb",
) -> np.ndarray:
    """
    Apply a user-selected color tint to a grayscale moon image while preserving
    the original luminance contrast.

    Parameters
    ----------
    moon_rgba : np.ndarray or PIL.Image.Image
        RGBA disc array returned by :func:`get_moon_mask`. RGB channels are
        assumed to be identical grayscale values.
    color : sequence or float
        Target color specification, interpreted according to ``color_space``.
        - ``"rgb"``: (R, G, B) components in 0-1 or 0-255 range, or a hex string
          like ``"#RRGGBB"``.
        - ``"cmyk"``: (C, M, Y, K) components in 0-1 or 0-100.
        - ``"hsb"`` (HSV): (H, S, B) where H may be 0-1 or 0-360 degrees, S/B in
          0-1 or 0-100.
        - ``"wavelength"``: single float wavelength in nanometers (380-780 nm).
    color_space : str, optional
        Name of the color space used for ``color``.

    Returns
    -------
    np.ndarray
        RGBA array with same shape as ``moon_rgba`` tinted in the requested hue.
    """

    try:
        from PIL import Image  # type: ignore

        pil_image_cls = Image.Image
    except Exception:  # pragma: no cover
        pil_image_cls = None

    if pil_image_cls and isinstance(moon_rgba, pil_image_cls):
        moon_rgba = np.asarray(moon_rgba)
    else:
        moon_rgba = np.asarray(moon_rgba)

    if moon_rgba.ndim != 3 or moon_rgba.shape[-1] != 4:
        raise ValueError("moon_rgba must be an (H, W, 4) RGBA image.")

    base_rgb = _color_to_rgb(color, color_space).reshape((1, 1, 3))

    rgb = moon_rgba[..., :3].astype(np.float32) / 255.0
    intensity = rgb[..., :1]
    tinted_rgb = np.clip(intensity * base_rgb, 0.0, 1.0)
    tinted_rgb = (tinted_rgb * 255.0).round().astype(np.uint8)

    alpha = moon_rgba[..., 3:4]
    return np.concatenate([tinted_rgb, alpha], axis=-1)


# def add_glow_to_image(
#     moon_rgba: Union[np.ndarray, "Image.Image"],
#     *,
#     strength: float = 0.5,
#     sigma: Union[float, int] = 0.04,
#     alpha_boost: float = 0.4,
#     color: Union[Sequence[float], float, str, None] = None,
#     color_space: str = "rgb",
# ) -> np.ndarray:
#     """
#     Add a soft glow around the rendered moon disc.

#     Parameters
#     ----------
#     moon_rgba : np.ndarray or PIL.Image.Image
#         RGBA disc image (as returned by :func:`get_moon_mask`).
#     strength : float
#         Scales the brightness of the glow colour addition.
#     sigma : float or int
#         Gaussian blur radius for the halo. Values <= 1 are interpreted as a
#         fraction of the output size; larger values are treated as pixels.
#     alpha_boost : float
#         Additional opacity contributed by the halo.
#     color : sequence, float or str, optional
#         Colour for the glow. Uses the same formats as :func:`tint_moon_image`.
#     color_space : str
#         Colour space for the glow colour (default ``"rgb"``).
#     """

#     try:
#         from PIL import Image  # type: ignore
#         pil_image_cls = Image.Image
#     except Exception:  # pragma: no cover
#         pil_image_cls = None

#     if pil_image_cls and isinstance(moon_rgba, pil_image_cls):
#         arr = np.asarray(moon_rgba)
#     else:
#         arr = np.asarray(moon_rgba)

#     if arr.ndim != 3 or arr.shape[-1] != 4:
#         raise ValueError("moon_rgba must be an (H, W, 4) RGBA image.")

#     from scipy.ndimage import gaussian_filter

#     arr_f = arr.astype(np.float32) / 255.0
#     alpha = arr_f[..., 3]

#     if isinstance(sigma, (float, int)) and sigma <= 1.0:
#         sigma_px = max(arr.shape[0], arr.shape[1]) * float(sigma)
#     else:
#         sigma_px = float(sigma)

#     blurred = gaussian_filter(alpha, sigma=sigma_px)
#     halo = np.clip(blurred - alpha, 0.0, None)

#     if color is None:
#         color_vec = np.array([1.0, 1.0, 1.0], dtype=float)
#     else:
#         color_vec = _color_to_rgb(color, color_space)

#     rgb = arr_f[..., :3] + halo[..., None] * color_vec * strength
#     rgb = np.clip(rgb, 0.0, 1.0)

#     alpha_new = np.clip(alpha + halo * alpha_boost, 0.0, 1.0)

#     out = np.empty_like(arr, dtype=np.uint8)
#     out[..., :3] = (rgb * 255.0).round().astype(np.uint8)
#     out[..., 3] = (alpha_new * 255.0).round().astype(np.uint8)
#     return out


def _color_to_rgb(
    color: Union[Sequence[float], float],
    color_space: str,
) -> np.ndarray:
    color_space = color_space.lower()

    if color_space == "rgb":
        if isinstance(color, str):
            color = color.lstrip("#")
            if len(color) != 6:
                raise ValueError(
                    "Hex RGB string must be 6 characters (e.g. '#FFA500')."
                )
            arr = np.array([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)
        else:
            arr = np.asarray(color, dtype=float)
        if arr.shape != (3,):
            raise ValueError("RGB color must be a sequence of three components.")
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    if color_space == "cmyk":
        arr = np.asarray(color, dtype=float)
        if arr.shape != (4,):
            raise ValueError("CMYK color must be a sequence of four components.")
        if arr.max() > 1.0:
            arr = arr / 100.0
        c, m, y, k = np.clip(arr, 0.0, 1.0)
        r = (1.0 - c) * (1.0 - k)
        g = (1.0 - m) * (1.0 - k)
        b = (1.0 - y) * (1.0 - k)
        return np.array([r, g, b], dtype=float)

    if color_space in {"hsb", "hsv"}:
        arr = np.asarray(color, dtype=float)
        if arr.shape != (3,):
            raise ValueError("HSB color must be a sequence of three components.")
        h, s, v = arr
        if h > 1.0:
            h = h / 360.0
        if s > 1.0:
            s = s / 100.0
        if v > 1.0:
            v = v / 100.0
        h = h % 1.0
        s = np.clip(s, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=float)

    if color_space == "wavelength":
        wavelength = float(color)
        return np.array(_wavelength_to_rgb(wavelength), dtype=float)

    raise ValueError(f"Unsupported color_space '{color_space}'.")


def _wavelength_to_rgb(wavelength: float) -> Sequence[float]:
    """
    Convert a wavelength in nanometers to an approximate sRGB triple in [0, 1].
    Based on the approximation by Dan Bruton.
    """
    gamma = 0.8
    wavelength = float(wavelength)
    if wavelength < 380 or wavelength > 780:
        return (0.0, 0.0, 0.0)

    if wavelength < 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        r = ((440 - wavelength) / (440 - 380)) * attenuation
        g = 0.0
        b = 1.0 * attenuation
    elif wavelength < 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif wavelength < 510:
        r = 0.0
        g = 1.0
        b = (510 - wavelength) / (510 - 490)
    elif wavelength < 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wavelength < 645:
        r = 1.0
        g = (645 - wavelength) / (645 - 580)
        b = 0.0
    else:
        attenuation = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)
        r = 1.0 * attenuation
        g = 0.0
        b = 0.0

    r = np.clip(r, 0.0, 1.0) ** gamma
    g = np.clip(g, 0.0, 1.0) ** gamma
    b = np.clip(b, 0.0, 1.0) ** gamma
    return (r, g, b)
