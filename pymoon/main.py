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
    return f * (mu0 / (mu0 + mu + eps))


def _model_ls_lambert(mu0, mu, f, k=0.4, eps=1e-8, **kw):
    # Blend of LS and Lambert; k in [0..1]
    return f * ((1 - k) * (mu0 / (mu0 + mu + eps)) + k * mu0)


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

    if model != "visibility":
        I *= mu

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
# Rotation matrices
# ---------------------------


def get_R_x(theta):
    """Rotation matrix around x axis by angle theta (radians)"""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def get_R_y(theta):
    """Rotation matrix around y axis by angle theta (radians)"""
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def get_R_z(theta):
    """Rotation matrix around z axis by angle theta (radians)"""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
