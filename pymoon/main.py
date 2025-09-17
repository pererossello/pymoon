import numpy as np

def get_disk_mask(N, radius=1):
    u = np.linspace(-1, 1, N, dtype=float)
    X, Y = np.meshgrid(u, u)  
    RHO_SQ = X*X + Y*Y

    inside = RHO_SQ <= radius  

    MASK = inside.astype(np.uint8)

    return MASK

def get_moon_mask(N, s_vec, radius=1):

    u = np.linspace(-1, 1, N, dtype=float)
    X, Y = np.meshgrid(u, u)  
    RHO_SQ = X*X + Y*Y
    Z = np.sqrt(np.clip(1.0 - RHO_SQ, 0.0, 1.0))

    inside = RHO_SQ <= radius  

    mu = X*s_vec[0] + Y*s_vec[1] + Z*s_vec[2]

    MASK = (inside & (mu > 0.0)).astype(np.uint8)

    return MASK


def rho_terminator(phi, s_vec):
    """
    Calculate rho at the terminator for given phi, and s
    """
    s1, s2, s3 = s_vec

    num = np.abs(s3)
    den = np.sqrt(
        s3 ** 2
        + (s1 * np.cos(phi) + s2 * np.sin(phi)) ** 2
    )

    return num / den


def get_phi_star(s_vec):
    return np.arctan2(s_vec[1], s_vec[0]) + np.pi/2


############################
# ROTATION MATRICES
############################

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