from typing import Optional, Sequence, Union

import numpy as np

from .config import *

Vector = Union[np.ndarray, Sequence[float]]


def get_3d_positions(r):
    """
    Convert an absolute radius grid into Cartesian coordinates.

    Parameters
    ----------
    r : np.ndarray
        Array of shape (h, w) containing the absolute radius in kilometers for each
        latitude/longitude sample on the Moon.
    """

    h, w = r.shape

    # Coordinate grids - standard lat/lon
    lon_rad = np.deg2rad((np.arange(w) / w) * 360 - 180)
    lat_rad = np.deg2rad(90 - (np.arange(h) / h) * 180)

    # Precompute trig
    cos_lat = np.cos(lat_rad[:, None])
    sin_lat = np.sin(lat_rad[:, None])
    cos_lon = np.cos(lon_rad[None, :])
    sin_lon = np.sin(lon_rad[None, :])

    # 3D positions
    x3d = r * cos_lat * sin_lon
    y3d = r * sin_lat
    z3d = r * cos_lat * cos_lon

    return x3d, y3d, z3d


def get_normals(dem: np.ndarray, R_M: float = R_MOON):
    """
    Compute the surface normals of a digital elevation model (DEM) of the Moon.
    Parameters:
    -----------
    dem : ndarray (h, w)
        Digital elevation model in kilometers
    R_M : float
        Moon radius in km (default 1737.4)
    """

    # Absolute radius
    R_filled = R_M + dem

    # 3D positions
    x3d, y3d, z3d = get_3d_positions(R_filled)

    # Shift positions to compute finite differences
    x_dlat = np.zeros_like(x3d)
    y_dlat = np.zeros_like(y3d)
    z_dlat = np.zeros_like(z3d)

    x_dlon = np.zeros_like(x3d)
    y_dlon = np.zeros_like(y3d)
    z_dlon = np.zeros_like(z3d)

    # Latitude direction (north-south)
    x_dlat[1:-1, :] = x3d[2:, :] - x3d[:-2, :]
    y_dlat[1:-1, :] = y3d[2:, :] - y3d[:-2, :]
    z_dlat[1:-1, :] = z3d[2:, :] - z3d[:-2, :]

    x_dlat[0, :] = x3d[1, :] - x3d[0, :]
    y_dlat[0, :] = y3d[1, :] - y3d[0, :]
    z_dlat[0, :] = z3d[1, :] - z3d[0, :]

    x_dlat[-1, :] = x3d[-1, :] - x3d[-2, :]
    y_dlat[-1, :] = y3d[-1, :] - y3d[-2, :]
    z_dlat[-1, :] = z3d[-1, :] - z3d[-2, :]

    # Longitude direction (east-west, periodic)
    x_dlon[:, 1:-1] = x3d[:, 2:] - x3d[:, :-2]
    y_dlon[:, 1:-1] = y3d[:, 2:] - y3d[:, :-2]
    z_dlon[:, 1:-1] = z3d[:, 2:] - z3d[:, :-2]

    x_dlon[:, 0] = x3d[:, 1] - x3d[:, -1]
    y_dlon[:, 0] = y3d[:, 1] - y3d[:, -1]
    z_dlon[:, 0] = z3d[:, 1] - z3d[:, -1]

    x_dlon[:, -1] = x3d[:, 0] - x3d[:, -2]
    y_dlon[:, -1] = y3d[:, 0] - y3d[:, -2]
    z_dlon[:, -1] = z3d[:, 0] - z3d[:, -2]

    # Cross product to get normal
    nx = y_dlat * z_dlon - z_dlat * y_dlon
    ny = z_dlat * x_dlon - x_dlat * z_dlon
    nz = x_dlat * y_dlon - y_dlat * x_dlon

    # Normalize
    n_norm = np.sqrt(nx**2 + ny**2 + nz**2)
    n_norm = np.where(n_norm > 0, n_norm, 1.0)
    nx /= n_norm
    ny /= n_norm
    nz /= n_norm

    return np.array([nx, ny, nz])


def _parse_color(color: Union[str, Sequence[float]]) -> np.ndarray:
    """
    Convert a color specification to an RGBA uint8 array.
    Accepts hex strings (#RRGGBB or #RRGGBBAA) or sequences of length 3 or 4
    with range either 0-1 or 0-255.
    """
    if isinstance(color, str):
        color = color.strip()
        if color.startswith("#"):
            color = color[1:]
        if len(color) not in (6, 8):
            raise ValueError("Hex color must be #RRGGBB or #RRGGBBAA.")
        values = [int(color[i : i + 2], 16) for i in range(0, len(color), 2)]
        if len(values) == 3:
            values.append(255)
        return np.array(values, dtype=np.uint8)

    arr = np.asarray(color, dtype=float)
    if arr.ndim != 1 or arr.size not in (3, 4):
        raise ValueError("Color must be a sequence of 3 (RGB) or 4 (RGBA) values.")
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0.0, 255.0)
    if arr.size == 3:
        arr = np.concatenate([arr, [255.0]])
    return arr.astype(np.uint8)


def render_moon_face(
    tex: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    obs_vec: Vector,
    out_px: int = 800,
    mask_mode: str = "multiply",
    outside_color: Optional[Union[str, Sequence[float]]] = None,
    radius: float = 1.0,
):
    """
    Render a disc image of a sphere textured with an equirectangular map for an
    arbitrary observer direction.

    Parameters
    ----------
    tex : np.ndarray or None
        Input texture (H, W, C) or (H, W) in equirectangular layout. Grayscale
        inputs are expanded to RGB. Floating point arrays are assumed to be in
        [0, 1]. If ``None``, a neutral white texture is used.
    mask : np.ndarray or None
        Scalar mask of shape (H, W). When ``mask_mode='multiply'`` this modulates
        the texture intensity; when ``'alpha'`` it drives the alpha channel;
        ``'both'`` multiplies intensity and alpha simultaneously.
    obs_vec : array-like of length 3
        Observer direction in world coordinates. Must be non-zero. The vector is
        normalized internally and (0, 0, 1) reproduces the azimuth=0 view.
    out_px : int
        Output image size (square), in pixels.
    mask_mode : {'multiply', 'alpha', 'both'}
        Defines how the mask is applied.
    outside_color : color-like, optional
        When provided, the exterior of the disc is filled with this color and
        the output is fully opaque. Accepts hex strings or RGB/RGBA sequences.
        Not compatible with ``mask_mode`` values that modify alpha.
    radius : float, optional
        Relative radius of the rendered disc (1.0 fills the frame, 0.5 renders a
        disc half the width/height). Must be in (0, 1].

    Returns
    -------
    np.ndarray
        RGBA image array of shape (out_px, out_px, 4) with transparent background
        outside the visible disc.
    """

    if tex is None and mask is None:
        raise ValueError("Either tex or mask must be provided.")

    tex_arr = None
    if tex is not None:
        tex_arr = np.asarray(tex)
        if tex_arr.ndim == 2:
            tex_arr = np.stack([tex_arr] * 3, axis=-1)
        elif tex_arr.ndim == 3 and tex_arr.shape[-1] == 4:
            tex_arr = tex_arr[..., :3]
        if np.issubdtype(tex_arr.dtype, np.floating):
            tex_max = float(np.nanmax(tex_arr)) if tex_arr.size else 1.0
            if tex_max <= 1.0 + 1e-6:
                tex_arr = np.clip(tex_arr, 0.0, 1.0)
            else:
                tex_arr = np.clip(tex_arr, 0.0, 255.0) / 255.0
        else:
            tex_arr = tex_arr.astype(np.float32) / 255.0
        tex_arr = tex_arr.astype(np.float32)

    mask_arr = None
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask_arr = mask_arr.astype(np.float32)
        mask_max = float(np.nanmax(mask_arr)) if mask_arr.size else 1.0
        if mask_max > 1.0 + 1e-6:
            mask_arr /= 255.0
        mask_arr = np.clip(mask_arr, 0.0, 1.0)

    if tex_arr is None:
        tex_arr = np.ones(mask_arr.shape + (3,), dtype=np.float32)

    if mask_arr is None:
        mask_arr = np.ones(tex_arr.shape[:2], dtype=np.float32)

    if tex_arr.shape[:2] != mask_arr.shape:
        raise ValueError("Texture and mask must share the same spatial dimensions.")

    mask_mode = mask_mode.lower()
    if mask_mode not in {"multiply", "alpha", "both"}:
        raise ValueError("mask_mode must be 'multiply', 'alpha', or 'both'.")
    if outside_color is not None and mask_mode != "multiply":
        raise ValueError("outside_color requires mask_mode='multiply'.")

    h, w = tex_arr.shape[:2]

    obs_vec = np.asarray(obs_vec, dtype=float)
    if obs_vec.shape != (3,):
        raise ValueError("obs_vec must be an array-like of length 3.")

    norm = np.linalg.norm(obs_vec)
    if norm == 0:
        raise ValueError("obs_vec must have a non-zero length.")
    forward = obs_vec / norm

    # Build an orthonormal camera basis while keeping lunar north up when possible.
    north = np.array([0.0, 1.0, 0.0])
    if np.abs(np.dot(forward, north)) > 0.999:
        north = np.array([0.0, 0.0, 1.0])

    right = np.cross(north, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        north = np.array([1.0, 0.0, 0.0])
        right = np.cross(north, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            raise ValueError("Failed to construct a camera frame from obs_vec.")
    right /= right_norm
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    # Build orthographic grid (x to the right, y up), unit disc
    N = out_px
    radius = float(radius)
    if not (0.0 < radius <= 1.0):
        raise ValueError("radius must be in the range (0, 1].")

    scale = 1.0 / radius
    y, x = np.linspace(scale, -scale, N), np.linspace(-scale, scale, N)
    xx, yy = np.meshgrid(x, y)
    rr2 = xx**2 + yy**2
    visible = rr2 <= 1.0
    zz = np.zeros_like(xx)
    zz[visible] = np.sqrt(1.0 - rr2[visible])

    # Directions in world coordinates for visible samples
    dirs = (
        xx[..., None] * right[None, None, :]
        + yy[..., None] * up[None, None, :]
        + zz[..., None] * forward[None, None, :]
    )
    dirs_x = dirs[..., 0]
    dirs_y = dirs[..., 1]
    dirs_z = dirs[..., 2]

    lat = np.zeros_like(xx)
    lon = np.zeros_like(xx)
    lat[visible] = np.arcsin(np.clip(dirs_y[visible], -1.0, 1.0))
    lon[visible] = np.arctan2(dirs_x[visible], dirs_z[visible])

    # Map lon/lat to texture coordinates (equirectangular)
    u = (lon + np.pi) / (2 * np.pi)
    v = (np.pi / 2 - lat) / np.pi

    ui = (u * w).astype(np.int64) % w
    vi = np.clip((v * h).astype(np.int64), 0, h - 1)

    sampled_tex = np.zeros((N, N, 3), dtype=np.float32)
    sampled_tex[visible] = tex_arr[vi[visible], ui[visible]]

    sampled_mask = np.zeros((N, N), dtype=np.float32)
    sampled_mask[visible] = mask_arr[vi[visible], ui[visible]]
    sampled_mask = np.clip(sampled_mask, 0.0, 1.0)

    if mask_mode in {"multiply", "both"}:
        sampled_tex *= sampled_mask[..., None]

    alpha = np.zeros((N, N), dtype=np.float32)
    if mask_mode in {"alpha", "both"}:
        alpha[visible] = sampled_mask[visible]
    else:
        alpha[visible] = 1.0

    rgb_uint8 = np.clip(sampled_tex * 255.0, 0.0, 255.0).round().astype(np.uint8)
    alpha_uint8 = np.clip(alpha * 255.0, 0.0, 255.0).round().astype(np.uint8)

    out = np.zeros((N, N, 4), dtype=np.uint8)

    if outside_color is None:
        out[..., :3] = rgb_uint8
        out[..., 3] = alpha_uint8
    else:
        bg_rgba = _parse_color(outside_color)
        out[:] = bg_rgba
        out_rgb = out[..., :3]
        out_alpha = out[..., 3]
        out_rgb[visible] = rgb_uint8[visible]
        out_alpha[visible] = alpha_uint8[visible]
        out_alpha[~visible] = 255  # ensure background is fully opaque

    return out
