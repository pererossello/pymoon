import numpy as np


# ---------------------------
# Geometry helpers
# ---------------------------


def _grid_normals(N, radius=1.0):
    """
    Orthographic projection of a sphere of given 'radius' onto an NxN grid.
    Returns unit surface normals (nx, ny, nz) and 'inside' mask.
    """
    u = np.linspace(-radius, radius, N, dtype=float)
    X, Y = np.meshgrid(u, u)
    R2 = X * X + Y * Y
    inside = R2 <= radius * radius
    Z = np.sqrt(np.clip(radius * radius - R2, 0.0, None))

    # Unit normals (since (X, Y, Z) lies on sphere of radius 'radius')
    inv_r = 1.0 / radius
    n = np.array([X * inv_r, Y * inv_r, Z * inv_r])  # shape (3, N, N)
    return n, inside


# ---------------------------
# Finite-Sun visibility
# ---------------------------


def _visible_fraction_from_elevation(epsilon, alpha):
    """
    epsilon: solar elevation [rad] per pixel
    alpha:   solar angular radius [rad]
    returns: fraction of solar disk visible (0..1)
    """
    y = epsilon / alpha
    f = np.zeros_like(y, dtype=np.float64)

    # Day / night
    f[y >= 1.0] = 1.0
    f[y <= -1.0] = 0.0

    # Penumbra
    m = (y > -1.0) & (y < 1.0)
    yc = np.clip(y[m], -1.0, 1.0)
    f[m] = (np.arccos(-yc) + yc * np.sqrt(1.0 - yc * yc)) / np.pi
    return f


# ---------------------------
# Photometric models
# ---------------------------


def _model_visibility(mu0, mu, f, **kw):
    return f


def _model_lambert(mu0, mu, f, **kw):
    return f * mu0


def _model_ls(mu0, mu, f, eps=1e-8, k=0.0, **kw):
    # Lommel–Seeliger: good for airless bodies like the Moon
    return f * (mu * mu0 / (mu0 + mu + eps))


def _model_ls_lambert(mu0, mu, f, k=0.4, eps=1e-8, **kw):
    # Blend of LS and Lambert; k in [0..1]
    return f * ((1 - k) * (mu * mu0 / (mu0 + mu + eps)) + k * mu0)


def _opposition_bump(s_vec, v_vec=(0, 0, 1), B0=0.5, sigma_deg=5.0):
    # Simple global opposition surge (shadow hiding / coherent backscatter)
    s = np.asarray(s_vec, float)
    s /= np.linalg.norm(s)
    v = np.asarray(v_vec, float)
    v /= np.linalg.norm(v_vec)
    g = np.arccos(np.clip(np.dot(s, v), -1.0, 1.0))  # phase angle (rad)
    return 1.0 + B0 * np.exp(-((np.degrees(g) / sigma_deg) ** 2))


def _model_ls_opposition(mu0, mu, f, k=0.4, eps=1e-3, s_vec=None, **kw):
    I = _model_ls_lambert(mu0, mu, f, k=k, eps=eps)
    if s_vec is not None:
        I *= _opposition_bump(
            s_vec, B0=kw.get("B0", 0.45), sigma_deg=kw.get("sigma_deg", 5.0)
        )
    return I


_MODEL_FUNCS = {
    "visibility": _model_visibility,
    "lambert": _model_lambert,
    "ls": _model_ls,
    "ls_lambert": _model_ls_lambert,
    "ls_opposition": _model_ls_opposition,  # recommended default for Moon
}

# ---------------------------
# Master function
# ---------------------------


def get_moon_mask(
    N,
    s_vec,
    *,
    radius=1.0,
    model="ls_opposition",
    penumbra=True,
    # Sun geometry (only used if penumbra=True)
    R_S=6.9634e8,  # Sun radius [m]
    d_SM=1.496e11,  # Sun–Moon distance [m]
    alpha=None,  # optional: directly pass solar angular radius [rad]
    # Model params
    k=0.4,  # LS/Lambert blend
    eps=1e-3,  # numeric floor for LS
    B0=0.45,  # opposition bump amplitude
    sigma_deg=5.0,  # opposition width (deg)
    observer_vec=(0, 0, 1),
    dtype=np.float32,
):
    """
    Returns an NxN float mask in [0,1].

    N:          image size
    s_vec:      Sun direction (3,), will be normalized
    radius:     sphere radius in screen units (fills [-radius, radius])
    model:      'visibility' | 'lambert' | 'ls' | 'ls_lambert' | 'ls_opposition'
    penumbra:   include finite-sun visibility ramp
    alpha:      override solar angular radius in radians (skips R_S/d_SM)
    """

    # Grid and normals
    n, inside = _grid_normals(N, radius)
    normals = np.moveaxis(n, 0, -1)  # shape (N, N, 3)
    s = np.asarray(s_vec, float)
    s /= np.linalg.norm(s)
    v = np.asarray(observer_vec, float)
    # v[-1] *= -1  # flip z for orthographic observer
    v /= np.linalg.norm(v)

    # Cosines
    mu0 = np.dot(normals, s)  # incidence cosine
    mu = np.dot(normals, v)  # emission cosine

    # Visibility fraction f
    if penumbra:
        a = alpha if alpha is not None else np.arctan(R_S / d_SM)
        epsilon = np.arcsin(np.clip(np.dot(normals, s), -1.0, 1.0))  # solar elevation
        f = _visible_fraction_from_elevation(epsilon, a)
    else:
        f = (mu0 > 0).astype(np.float64)

    mu0 = np.maximum(mu0, 0.0)
    mu = np.maximum(mu, 0.0)

    # Photometry
    if model not in _MODEL_FUNCS:
        raise ValueError(f"Unknown model '{model}'")
    I = _MODEL_FUNCS[model](
        mu0, mu, f, k=k, eps=eps, s_vec=s, B0=B0, sigma_deg=sigma_deg
    )

    # Outside disk → 0
    I = np.where(inside, I, 0.0).astype(dtype)
    I /= np.nanmax(I) if np.nanmax(I) > 0 else 1.0
    I = np.clip(I, 0.0, 1.0)

    return I


def get_disk_mask(N, radius=1):
    u = np.linspace(-1, 1, N, dtype=float)
    X, Y = np.meshgrid(u, u)
    RHO_SQ = X * X + Y * Y

    inside = RHO_SQ <= radius

    MASK = inside.astype(np.uint8)

    return MASK


# ---------------------------
# Terminator function
# ---------------------------


def rho_terminator(phi, s_vec):
    """
    Calculate rho at the terminator for given phi, and s
    """
    s1, s2, s3 = s_vec

    num = np.abs(s3)
    den = np.sqrt(s3**2 + (s1 * np.cos(phi) + s2 * np.sin(phi)) ** 2)

    return num / den


def get_phi_star(s_vec):
    return np.arctan2(s_vec[1], s_vec[0]) + np.pi / 2


# ---------------------------
# Shadows
# ---------------------------

def compute_cellsize(h, w, R_M=1737.4):

    lat_rad = np.deg2rad(90 - (np.arange(h) / h) * 180)
    
    # Angular resolution in radians
    delta_lon = 2 * np.pi / w  # radians per pixel in longitude
    delta_lat = np.pi / h      # radians per pixel in latitude
    
    # Cellsize in y-direction (latitude) - constant everywhere
    cellsize_y = R_M * delta_lat  # km per pixel
    cellsize_y = np.full(h, cellsize_y)[:, None]  # shape (h, 1)
    
    # Cellsize in x-direction (longitude) - varies with latitude
    cellsize_x = R_M * delta_lon * np.cos(lat_rad)  # km per pixel, shape (h,)
    cellsize_x = cellsize_x[:, None]  # shape (h, 1)
    
    return cellsize_x, cellsize_y


# def compute_shadow(dem, s_vec, R_M=1737.4, d_MS=1.496e8, R_S=6.9634e5):
#     """Compute the shadow map for a given DEM and light source vector.

#     Args:
#         dem (ndarray): Digital Elevation Model (DEM) array.
#         s_vec (ndarray): Sun direction vector (from Moon center to Sun)
#         R_M (float, optional): Radius of the Moon in km. Defaults to 1737.4.
#         d_MS (float, optional): Distance from Moon to Sun in km. Defaults to 1.496e8.
#     """

#     R = R_M + dem
#     h, w = dem.shape

#     lon_rad = np.deg2rad((np.arange(w) / w) * 360 - 180)
#     lat_rad = np.deg2rad(90 - (np.arange(h) / h) * 180)

#     x3d = R * np.cos(lat_rad[:, None]) * np.sin(lon_rad[None, :])
#     y3d = R * np.sin(lat_rad[:, None])
#     z3d = R * np.cos(lat_rad[:, None]) * np.cos(lon_rad[None, :])

#     # Compute cellsize arrays
#     cellsize_x, cellsize_y = compute_cellsize(h, w, R_M)

def compute_shadow(dem, s_vec, R_M=1737.4, d_MS=1.496e8, R_S=6.9634e5, max_steps=10):

    s_vec = np.array(s_vec) / np.linalg.norm(s_vec)
    
    R = R_M + dem
    h, w = dem.shape

    lon_rad = np.deg2rad((np.arange(w) / w) * 360 - 180)
    lat_rad = np.deg2rad(90 - (np.arange(h) / h) * 180)

    x3d = R * np.cos(lat_rad[:, None]) * np.sin(lon_rad[None, :])
    y3d = R * np.sin(lat_rad[:, None])
    z3d = R * np.cos(lat_rad[:, None]) * np.cos(lon_rad[None, :])

    # Compute cellsize arrays
    cellsize_x, cellsize_y = compute_cellsize(h, w, R_M)
    
    # Solar angular radius as seen from Moon
    alpha_sun = np.arctan(R_S / d_MS)  # radians
    
    # Initialize visibility map
    mask = np.ones((h, w), dtype=np.float32)
    
    # Mask out back-facing hemisphere
    mask = np.where(z3d < 0, 0., mask)
    
    # Vectorized shadow computation using DEM in 2D projection
    # Convert s_vec to local tangent plane direction
    s_vec_local = d_MS * s_vec[:,None,None] - np.array([x3d, y3d, z3d])
    # Local ray direction from each surface point to sun
    # Shape: (3, h, w)
    s_vec_local = d_MS * s_vec[:, None, None] - np.array([x3d, y3d, z3d])
    
    # Normalize local directions
    s_norm = np.sqrt(s_vec_local[0]**2 + s_vec_local[1]**2 + s_vec_local[2]**2)
    dx = s_vec_local[0] / s_norm  # shape (h, w)
    dy = s_vec_local[1] / s_norm
    dz = s_vec_local[2] / s_norm

      
    # Fill NaNs in DEM
    dem_filled = np.nan_to_num(dem, nan=np.nanmin(dem))
    
    # Maximum occlusion angle for each pixel
    max_occlusion = np.zeros((h, w), dtype=np.float32)
    
    # Ray march for each pixel
    for i in range(h):
        for j in range(w):
            if z3d[i, j] < 0:
                continue
            
            # Current surface point
            x0, y0, z0 = x3d[i, j], y3d[i, j], z3d[i, j]
            
            # Local ray direction for this pixel
            ray_dx, ray_dy, ray_dz = dx[i, j], dy[i, j], dz[i, j]
            
            # Step along ray
            for step in range(1, max_steps):
                # Next point in 3D
                x_ray = x0 + ray_dx * cellsize_y[i, 0] * step
                y_ray = y0 + ray_dy * cellsize_y[i, 0] * step
                z_ray = z0 + ray_dz * cellsize_y[i, 0] * step
                
                r_norm = np.sqrt(x_ray**2 + y_ray**2 + z_ray**2)
                
                # Convert to lat/lon
                lat_ray = np.arcsin(np.clip(y_ray / r_norm, -1, 1))
                lon_ray = np.arctan2(x_ray, z_ray)
                
                # Convert to pixel indices
                lat_idx = int(np.round((np.pi/2 - lat_ray) / np.pi * h))
                lon_idx = int(np.round((lon_ray + np.pi) / (2*np.pi) * w)) % w
                
                if lat_idx < 0 or lat_idx >= h:
                    break
                
                # Check occlusion
                R_terrain = R_M + dem_filled[lat_idx, lon_idx]
                R_ray = r_norm
                
                if R_terrain > R_ray:
                    height_diff = R_terrain - R_ray
                    horizontal_dist = cellsize_y[i, 0] * step
                    angular_occl = np.arctan2(height_diff, horizontal_dist)
                    
                    max_occlusion[i, j] = max(max_occlusion[i, j], angular_occl)
                    
                    if angular_occl > alpha_sun:
                        break  # Fully shadowed
    
    # Compute visibility with penumbra
    # visibility = 1 means fully lit, 0 means fully shadowed
    visibility = np.clip(1.0 - max_occlusion / alpha_sun, 0.0, 1.0)
    mask = mask * visibility
    
    return mask


import numpy as np

import numpy as np

def compute_shadow_horizon(
    dem,
    s_vec,
    R_M=1737.4,           # km
    d_MS=1.496e8,         # km
    R_S=6.9634e5,         # km
    sigma_max_deg=12.0,   # angular search along sunward great circle
    step_factor=1.25,     # ~1.25× pixel angular res
    camera_vec=(0.0, 0.0, 1.0)
):
    """
    Penumbra-aware visibility on a spherical body with topography.
    DEM units must match R_M (km here). Returns float32 visibility in [0,1].
    """
    dem = np.asarray(dem, dtype=np.float64)
    h, w = dem.shape

    # Normalize directions
    s_vec = np.asarray(s_vec, dtype=np.float64)
    s_vec = s_vec / np.linalg.norm(s_vec)
    cam = np.asarray(camera_vec, dtype=np.float64)
    cam = cam / np.linalg.norm(cam)

    # Build true (h,w) lat/lon grids
    lon_1d = np.deg2rad((np.arange(w) / w) * 360.0 - 180.0)   # [-pi, pi)
    lat_1d = np.deg2rad(90.0 - (np.arange(h) / h) * 180.0)    # [ pi/2..-pi/2]
    lat2d, lon2d = np.meshgrid(lat_1d, lon_1d, indexing="ij") # both (h,w)

    # Planetocentric radius and position
    Rp = R_M + dem                                           # (h,w)
    cosφ = np.cos(lat2d); sinφ = np.sin(lat2d)
    cosλ = np.cos(lon2d); sinλ = np.sin(lon2d)

    x = Rp * cosφ * sinλ                                     # (h,w)
    y = Rp * sinφ
    z = Rp * cosφ * cosλ
    P = np.stack([x, y, z], axis=-1)                         # (h,w,3)
    n_hat = P / np.linalg.norm(P, axis=-1, keepdims=True)    # (h,w,3)

    # Camera-facing and daylight masks
    cam_dot = (n_hat @ cam).astype(np.float64)               # (h,w)
    camera_mask = cam_dot > 0.0

    alpha_sun = np.arctan(R_S / d_MS)                        # radians

    mu0 = (n_hat @ s_vec).astype(np.float64)                 # (h,w)
    daylight = mu0 > 0.0

    # Project sun into local tangent to get azimuth
    t = s_vec - mu0[..., None] * n_hat                       # (h,w,3)
    t_norm = np.linalg.norm(t, axis=-1)
    t_hat = np.zeros_like(t)
    ok = t_norm > 0
    t_hat[ok] = t[ok] / t_norm[ok, None]

    # Tangent basis (east, north), both (h,w,3)
    east = np.stack([-sinλ, np.zeros_like(lon2d), cosλ], axis=-1)
    north = np.stack(
        [-sinφ * cosλ,  np.cos(lat2d), -sinφ * sinλ],
        axis=-1
    )
    # Azimuth from north toward east
    A = np.arctan2(np.sum(t_hat * east, axis=-1), np.sum(t_hat * north, axis=-1))  # (h,w)
    sinA = np.sin(A); cosA = np.cos(A)

    # Pixel angular resolution and sampling schedule
    dlat = np.pi / max(h - 1, 1)
    dlon = 2.0 * np.pi / max(w, 1)
    delta_sigma = step_factor * min(dlat, dlon)
    sigma_max = np.deg2rad(sigma_max_deg)
    n_steps = int(np.ceil(sigma_max / max(delta_sigma, 1e-9)))
    n_steps = max(n_steps, 1)
    sigmas = (np.arange(1, n_steps + 1, dtype=np.float64)) * delta_sigma  # (k,)

    # Bilinear sampler for DEM at (lat, lon) arrays (both (h,w))
    def bilinear_sample_dem(lat_arr, lon_arr):
        # Map lat,lon -> image coords u in [0,w-1], v in [0,h-1]
        u = (lon_arr + np.pi) * (w - 1) / (2.0 * np.pi)
        v = (np.pi / 2.0 - lat_arr) * (h - 1) / np.pi

        u0 = np.floor(u).astype(np.int64)
        v0 = np.floor(v).astype(np.int64)
        u1 = (u0 + 1) % w
        v1 = np.clip(v0 + 1, 0, h - 1)

        fu = u - u0
        fv = v - v0

        Q11 = dem[v0, u0]
        Q21 = dem[v0, u1]
        Q12 = dem[v1, u0]
        Q22 = dem[v1, u1]

        top = (1 - fu) * Q11 + fu * Q21
        bot = (1 - fu) * Q12 + fu * Q22
        return (1 - fv) * top + fv * bot

    # Horizon accumulation
    theta_max = np.full((h, w), -1e9, dtype=np.float64)
    active = daylight & camera_mask & np.isfinite(dem)

    for sigma in sigmas:
        if not np.any(active):
            break

        cosσ = np.cos(sigma)
        sinσ = np.sin(sigma)

        # Great-circle forward: new lat φ2 and lon λ2 along azimuth A by σ
        sinφ2 = sinφ * cosσ + np.cos(lat2d) * sinσ * cosA
        sinφ2 = np.clip(sinφ2, -1.0, 1.0)
        φ2 = np.arcsin(sinφ2)

        y_term = sinA * sinσ * np.cos(lat2d)
        x_term = cosσ - sinφ * sinφ2
        Δλ = np.arctan2(y_term, x_term)
        λ2 = lon2d + Δλ

        # Sample terrain height along the sunward great circle
        h_samp = bilinear_sample_dem(φ2, λ2)
        valid = np.isfinite(h_samp) & active
        if not np.any(valid):
            continue

        Rt = R_M + h_samp
        # theta = atan2(Rt - Rp*cosσ, Rp*sinσ)
        num = Rt - Rp * cosσ
        den = Rp * sinσ
        # avoid 0/0 at sigma→0
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)

        theta = np.full((h, w), -1e9, dtype=np.float64)
        theta[valid] = np.arctan2(num[valid], den[valid])

        theta_max = np.maximum(theta_max, theta)

        # Early-out pixels that are fully shadowed
        active &= (theta_max < alpha_sun)

    # Convert horizon angle to visibility
    vis = 1.0 - (theta_max / alpha_sun)
    vis = np.clip(vis, 0.0, 1.0)

    vis[~daylight] = 0.0
    vis[~camera_mask] = 0.0
    vis[~np.isfinite(dem)] = 0.0

    return vis.astype(np.float32)


def compute_shadow_fast(dem, s_vec, R_M=1737.4, d_MS=1.496e8, R_S=6.9634e5, max_steps=100):
    """
    Compute shadows on lunar surface with penumbra effects.
    
    Parameters:
    -----------
    dem : ndarray (h, w)
        Digital elevation model in kilometers
    s_vec : array-like (3,)
        Sun direction vector (will be normalized)
    R_M : float
        Moon radius in km (default 1737.4)
    d_MS : float
        Moon-Sun distance in km (default 1.496e8)
    R_S : float
        Sun radius in km (default 6.9634e5)
    max_steps : int
        Maximum ray marching steps
        
    Returns:
    --------
    mask : ndarray (h, w)
        Shadow mask with values 0 (full shadow) to 1 (full light)
    """
    
    s_vec = np.array(s_vec) / np.linalg.norm(s_vec)
    
    h, w = dem.shape
    
    # Fill NaNs
    dem_filled = np.nan_to_num(dem, nan=np.nanmin(dem))
    R_filled = R_M + dem_filled
    
    # Coordinate grids - standard lat/lon
    lon_rad = np.deg2rad((np.arange(w) / w) * 360 - 180)
    lat_rad = np.deg2rad(90 - (np.arange(h) / h) * 180)
    
    # Precompute trig
    cos_lat = np.cos(lat_rad[:, None])
    sin_lat = np.sin(lat_rad[:, None])
    cos_lon = np.cos(lon_rad[None, :])
    sin_lon = np.sin(lon_rad[None, :])
    
    # 3D positions
    x3d = R_filled * cos_lat * sin_lon
    y3d = R_filled * sin_lat
    z3d = R_filled * cos_lat * cos_lon
    
    # Solar angular radius
    alpha_sun = np.arctan(R_S / d_MS)
    
    # Sun direction (approximately parallel for distant sun)
    # Just use the constant direction - parallax is negligible
    dx = s_vec[0]
    dy = s_vec[1]
    dz = s_vec[2]
    
    # Compute surface normals using simple finite differences
    # This is more robust than analytical derivatives
    dlat = np.pi / h
    dlon = 2 * np.pi / w
    
    # Compute position differences to get tangent vectors
    # We'll compute normals by finite differences in 3D space
    
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
    
    # Ensure normals point outward (dot product with position vector should be positive)
    radial_dot = nx * x3d + ny * y3d + nz * z3d
    flip = radial_dot < 0
    nx = np.where(flip, -nx, nx)
    ny = np.where(flip, -ny, ny)
    nz = np.where(flip, -nz, nz)
    
    # Cosine of incidence angle
    cos_incident = dx * nx + dy * ny + dz * nz
    
    # Initialize mask: front-facing hemisphere and illuminated slopes
    mask = np.where((z3d > 0) & (cos_incident > 0), 1.0, 0.0)
    
    visible_mask = mask > 0
    
    if not visible_mask.any():
        return mask
    
    # Shadow computation
    max_occlusion = np.zeros((h, w), dtype=np.float32)
    base_step = R_M * np.pi / max(h, w) * 0.5
    active_mask = visible_mask.copy()
    
    for step in range(1, max_steps):
        if not active_mask.any():
            break
        
        step_dist = base_step * step
        
        # Ray positions
        x_ray = x3d + dx * step_dist
        y_ray = y3d + dy * step_dist
        z_ray = z3d + dz * step_dist
        
        r_ray = np.sqrt(x_ray**2 + y_ray**2 + z_ray**2)
        
        # Convert to lat/lon
        lat_ray = np.arcsin(np.clip(y_ray / r_ray, -1, 1))
        lon_ray = np.arctan2(x_ray, z_ray)
        
        # Pixel indices
        lat_idx = ((np.pi/2 - lat_ray) / np.pi * h)
        lon_idx = ((lon_ray + np.pi) / (2*np.pi) * w) % w
        
        lat_i = np.clip(np.floor(lat_idx).astype(int), 0, h-1)
        lon_i = np.floor(lon_idx).astype(int) % w
        
        lat_frac = lat_idx - lat_i
        lon_frac = lon_idx - lon_i
        
        lat_i1 = np.clip(lat_i + 1, 0, h-1)
        lon_i1 = (lon_i + 1) % w
        
        # Bilinear interpolation
        R00 = R_filled[lat_i, lon_i]
        R01 = R_filled[lat_i, lon_i1]
        R10 = R_filled[lat_i1, lon_i]
        R11 = R_filled[lat_i1, lon_i1]
        
        R_terrain = (R00 * (1-lat_frac) * (1-lon_frac) +
                     R01 * (1-lat_frac) * lon_frac +
                     R10 * lat_frac * (1-lon_frac) +
                     R11 * lat_frac * lon_frac)
        
        height_diff = R_terrain - r_ray
        occluded = (height_diff > 0) & active_mask
        
        if occluded.any():
            angular_occl = np.arctan2(np.maximum(height_diff, 0), step_dist)
            max_occlusion = np.maximum(max_occlusion, angular_occl)
            
            fully_shadowed = angular_occl >= alpha_sun
            active_mask = active_mask & ~fully_shadowed
        
        too_far = (r_ray - R_M) > R_M * 0.3
        active_mask = active_mask & ~too_far
    
    # Penumbra
    visibility = np.clip(1.0 - max_occlusion / alpha_sun, 0.0, 1.0)
    
    # Final mask
    mask = mask * visibility * np.maximum(cos_incident, 0)
    
    return mask


# def compute_shadow_fast(dem, s_vec, R_M=1737.4, d_MS=1.496e8, R_S=6.9634e5, max_steps=50):

#     s_vec = np.array(s_vec) / np.linalg.norm(s_vec)
    
#     R = R_M + dem
#     h, w = dem.shape

#     lon_rad = np.deg2rad((np.arange(w) / w) * 360 - 180)
#     lat_rad = np.deg2rad(90 - (np.arange(h) / h) * 180)

#     x3d = R * np.cos(lat_rad[:, None]) * np.sin(lon_rad[None, :])
#     y3d = R * np.sin(lat_rad[:, None])
#     z3d = R * np.cos(lat_rad[:, None]) * np.cos(lon_rad[None, :])

#     # Compute cellsize arrays
#     cellsize_x, cellsize_y = compute_cellsize(h, w, R_M)
    
#     # Solar angular radius as seen from Moon
#     alpha_sun = np.arctan(R_S / d_MS)
    
#     # Local ray direction from each surface point to sun
#     # Shape: (3, h, w)
#     sun_pos = d_MS * s_vec[:, None, None]
#     s_vec_local = sun_pos - np.array([x3d, y3d, z3d])
    
#     # Normalize local directions
#     s_norm = np.sqrt(s_vec_local[0]**2 + s_vec_local[1]**2 + s_vec_local[2]**2)
#     dx = s_vec_local[0] / s_norm  # shape (h, w)
#     dy = s_vec_local[1] / s_norm
#     dz = s_vec_local[2] / s_norm
    
#     # Initialize mask
#     mask = np.ones((h, w), dtype=np.float32)
#     mask = np.where(z3d < 0, 0., mask)
    
#     # Fill NaNs
#     dem_filled = np.nan_to_num(dem, nan=np.nanmin(dem))
#     R_filled = R_M + dem_filled
    
#     # Maximum occlusion angle
#     max_occlusion = np.zeros((h, w), dtype=np.float32)
    
#     # Use average cellsize for stepping
#     cellsize_step = np.mean(cellsize_y)
    
#     # Vectorized ray marching for all visible pixels at once
#     visible_mask = z3d > 0
    
#     for step in range(1, max_steps):
#         # Compute next points for ALL pixels at once
#         x_ray = x3d + dx * cellsize_step * step
#         y_ray = y3d + dy * cellsize_step * step
#         z_ray = z3d + dz * cellsize_step * step
        
#         r_norm = np.sqrt(x_ray**2 + y_ray**2 + z_ray**2)
        
#         # Convert to lat/lon
#         lat_ray = np.arcsin(np.clip(y_ray / r_norm, -1, 1))
#         lon_ray = np.arctan2(x_ray, z_ray)
        
#         # Convert to pixel indices
#         lat_idx = np.round((np.pi/2 - lat_ray) / np.pi * h).astype(int)
#         lon_idx = np.round((lon_ray + np.pi) / (2*np.pi) * w).astype(int) % w
        
#         # Bounds check
#         valid = (lat_idx >= 0) & (lat_idx < h) & visible_mask
        
#         # Get terrain heights at ray positions (for valid pixels)
#         R_terrain = np.full((h, w), -np.inf)
#         R_terrain[valid] = R_filled[lat_idx[valid], lon_idx[valid]]
        
#         # Check occlusion
#         height_diff = R_terrain - r_norm
#         occluded = (height_diff > 0) & valid
        
#         if occluded.any():
#             # Compute angular occlusion
#             horizontal_dist = cellsize_step * step
#             angular_occl = np.arctan2(np.maximum(height_diff, 0), horizontal_dist)
            
#             # Update max occlusion
#             max_occlusion = np.maximum(max_occlusion, angular_occl)
            
#             # Stop tracking fully shadowed pixels
#             fully_shadowed = angular_occl > alpha_sun
#             visible_mask = visible_mask & ~fully_shadowed
        
#         # Early exit if no visible pixels left
#         if not visible_mask.any():
#             break
    
#     # Compute visibility with penumbra
#     visibility = np.clip(1.0 - max_occlusion / alpha_sun, 0.0, 1.0)
#     mask = mask * visibility
    
#     return mask