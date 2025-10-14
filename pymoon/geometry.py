from typing import Union, Sequence

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


def render_moon_face(tex: np.ndarray, obs_vec: Vector, out_px=800):
    """
    Render a disc image of a sphere textured with an equirectangular map for an
    arbitrary observer direction.

    Parameters
    ----------
    tex : np.ndarray
        Input texture (H, W, C) in equirectangular layout. Grayscale inputs are
        expanded to RGB. Floating point arrays are assumed to be in [0, 1].
    obs_vec : array-like of length 3
        Observer direction in world coordinates. Must be non-zero. The vector is
        normalized internally and (0, 0, 1) reproduces the azimuth=0 view.
    out_px : int
        Output image size (square), in pixels.

    Returns
    -------
    np.ndarray
        RGBA image array of shape (out_px, out_px, 4) with transparent background
        outside the visible disc.
    """

    if tex.ndim == 2:
        tex = np.stack([tex] * 3, axis=-1)  # grayscale to RGB

    if np.issubdtype(tex.dtype, np.floating):
        tex_min = float(np.nanmin(tex))
        tex_max = float(np.nanmax(tex))
        if tex_max <= 1.0 + 1e-6:
            tex = np.clip(tex, 0.0, 1.0)
            tex = (tex * 255.0).round().astype(np.uint8)
        else:
            tex = np.clip(tex, 0.0, 255.0).round().astype(np.uint8)

    h, w, _ = tex.shape

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
    y, x = np.linspace(1, -1, N), np.linspace(-1, 1, N)
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

    out = np.zeros((N, N, 4), dtype=np.uint8)
    out[..., 3] = 0
    out_rgb = out[..., :3]
    out_rgb[visible] = tex[vi[visible], ui[visible]]
    out[..., 3][visible] = 255

    return out

