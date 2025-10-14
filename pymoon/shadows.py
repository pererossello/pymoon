import numpy as np
from tqdm import tqdm

from .config import *
from .geometry import get_normals, get_3d_positions

def compute_shadow(
    base_mask,
    dem,
    s_vec,
    R_M=R_MOON,
    theta_S=THETA_SUN,
    pointlike_sun=False,
    max_steps=10,
):

    # Sun direction (parallax is negligible)
    s_vec = np.array(s_vec) / np.linalg.norm(s_vec)

    sx = s_vec[0]
    sy = s_vec[1]
    sz = s_vec[2]

    h, w = dem.shape
    R_filled = R_M + dem  # Absolute radius

    nx, ny, nz = get_normals(dem, R_M=R_M)
    x3d, y3d, z3d = get_3d_positions(R_filled)

    # Incidence cosine and solar elevation
    mu0 = sx * nx + sy * ny + sz * nz
    epsilon_sun = np.arcsin(np.clip(mu0, -1.0, 1.0))

    visible_mask = base_mask > 0

    if not visible_mask.any():  # no illuminated pixels
        return base_mask

    # Shadow computation
    max_elevation = np.zeros((h, w), dtype=np.float64)
    occluded_any = np.zeros((h, w), dtype=bool)
    base_step = R_M * np.pi / max(h, w) * 0.5
    active_mask = visible_mask.copy()

    for step in tqdm(range(1, max_steps), desc="Computing Shadows"):
        if not active_mask.any():
            break

        step_dist = base_step * step

        # Ray positions
        x_ray = x3d + sx * step_dist
        y_ray = y3d + sy * step_dist
        z_ray = z3d + sz * step_dist

        r_ray = np.sqrt(x_ray**2 + y_ray**2 + z_ray**2)

        # Convert to lat/lon
        lat_ray = np.arcsin(np.clip(y_ray / r_ray, -1, 1))
        lon_ray = np.arctan2(x_ray, z_ray)

        # Pixel indices
        lat_idx = (np.pi / 2 - lat_ray) / np.pi * h
        lon_idx = ((lon_ray + np.pi) / (2 * np.pi) * w) % w

        lat_i = np.clip(np.floor(lat_idx).astype(int), 0, h - 1)
        lon_i = np.floor(lon_idx).astype(int) % w

        lat_frac = lat_idx - lat_i
        lon_frac = lon_idx - lon_i

        lat_i1 = np.clip(lat_i + 1, 0, h - 1)
        lon_i1 = (lon_i + 1) % w

        # Bilinear interpolation
        R00 = R_filled[lat_i, lon_i]
        R01 = R_filled[lat_i, lon_i1]
        R10 = R_filled[lat_i1, lon_i]
        R11 = R_filled[lat_i1, lon_i1]

        R_terrain = (
            R00 * (1 - lat_frac) * (1 - lon_frac)
            + R01 * (1 - lat_frac) * lon_frac
            + R10 * lat_frac * (1 - lon_frac)
            + R11 * lat_frac * lon_frac
        )

        height_diff = R_terrain - r_ray
        occluded = (height_diff > 0) & active_mask

        if occluded.any():
            cos_lat_ray = np.cos(lat_ray)
            sin_lat_ray = np.sin(lat_ray)
            cos_lon_ray = np.cos(lon_ray)
            sin_lon_ray = np.sin(lon_ray)

            x_occ = R_terrain * cos_lat_ray * sin_lon_ray
            y_occ = R_terrain * sin_lat_ray
            z_occ = R_terrain * cos_lat_ray * cos_lon_ray

            occ_vec_x = x_occ - x3d
            occ_vec_y = y_occ - y3d
            occ_vec_z = z_occ - z3d

            parallel = occ_vec_x * sx + occ_vec_y * sy + occ_vec_z * sz
            forward = parallel > 0

            distance = np.sqrt(occ_vec_x**2 + occ_vec_y**2 + occ_vec_z**2)
            distance = np.where(distance > 0, distance, np.finfo(np.float64).eps)

            occ_unit_x = occ_vec_x / distance
            occ_unit_y = occ_vec_y / distance
            occ_unit_z = occ_vec_z / distance

            dot_normal = occ_unit_x * nx + occ_unit_y * ny + occ_unit_z * nz
            dot_normal = np.clip(dot_normal, -1.0, 1.0)
            epsilon_occ = np.arcsin(dot_normal)

            occ_mask = occluded & forward
            if occ_mask.any():
                max_elevation[occ_mask] = np.maximum(
                    max_elevation[occ_mask], epsilon_occ[occ_mask]
                )
                occluded_any |= occ_mask

                if pointlike_sun:
                    active_mask[occ_mask] = False
                else:
                    fully_shadowed = occ_mask & (epsilon_occ >= (epsilon_sun + theta_S))
                    active_mask[fully_shadowed] = False

        too_far = (r_ray - R_M) > R_M * 0.3
        active_mask = active_mask & ~too_far

    if pointlike_sun:
        visibility_fraction = np.where(occluded_any, 0.0, 1.0)
    else:
        delta = epsilon_sun - max_elevation
        visibility_fraction = _visible_fraction_from_elevation(delta, theta_S)

    return base_mask * visibility_fraction


def _visible_fraction_from_elevation(epsilon, theta_S=THETA_SUN):
    """
    epsilon: solar elevation [rad] per pixel
    theta_S: solar angular radius [rad]
    returns: fraction of solar disk visible (0..1)
    """
    y = epsilon / theta_S
    f = np.zeros_like(y, dtype=np.float64)

    # Day / night
    f[y >= 1.0] = 1.0
    f[y <= -1.0] = 0.0

    # Penumbra
    m = (y > -1.0) & (y < 1.0)
    yc = np.clip(y[m], -1.0, 1.0)
    f[m] = (np.arccos(-yc) + yc * np.sqrt(1.0 - yc * yc)) / np.pi
    return f


