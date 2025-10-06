import numpy as np
from PIL import Image



# ---------------------------
# Projection
# ---------------------------

def render_moon_face(tex, azimuth_deg=0, out_px=800):
    """
    Render a disc image of a sphere textured with an equirectangular map.
    
    Parameters
    ----------
    tex : PIL.Image.Image or np.ndarray
        The input texture (equirectangular RGB/RGBA image).
    azimuth_deg : float
        Central longitude to face the camera, in degrees.
        0 shows longitude 0 at center; 90 rotates view eastward, etc.
    out_px : int
        Output image size (square), in pixels.
        
    Returns
    -------
    PIL.Image.Image
        The rendered disc image (RGBA, transparent outside the disc).
    """

    # Load texture
    if isinstance(tex, np.ndarray):
        tex_np = tex
    else:
        tex_np = np.asarray(tex)

    h, w, _ = tex_np.shape

    # Build orthographic grid (x to the right, y up), unit disc
    N = out_px
    y, x = np.linspace(1, -1, N), np.linspace(-1, 1, N)  # image coords: top-left origin
    xx, yy = np.meshgrid(x, y)
    rr2 = xx**2 + yy**2
    visible = rr2 <= 1.0  # inside the disc = visible hemisphere
    zz = np.zeros_like(xx)
    zz[visible] = np.sqrt(1.0 - rr2[visible])  # z towards the viewer

    # Camera pointing at latitude 0, longitude lon0
    lon0 = np.deg2rad(azimuth_deg)

    # Inverse orthographic projection (phi1 = 0 simplifies nicely)
    # lat = arcsin(y)
    # lon = lon0 + atan2(x, z)
    lat = np.zeros_like(xx)
    lon = np.zeros_like(xx)
    lat[visible] = np.arcsin(yy[visible])
    lon[visible] = lon0 + np.arctan2(xx[visible], zz[visible])

    # Map lon/lat to texture coordinates (equirectangular)
    # u in [0, 1) across longitudes [-pi, pi]; v in [0, 1] across latitudes [-pi/2, pi/2]
    u = (lon + np.pi) / (2 * np.pi)
    v = (np.pi / 2 - lat) / np.pi

    # Convert to pixel indices
    ui = (u * w).astype(np.int64) % w
    vi = np.clip((v * h).astype(np.int64), 0, h - 1)

    # Sample the texture
    out = np.zeros((N, N, 4), dtype=np.uint8)
    out[..., 3] = 0  # transparent background
    out_rgb = out[..., :3]
    out_rgb[visible] = tex_np[vi[visible], ui[visible]]
    out[..., 3][visible] = 255

    return Image.fromarray(out, mode="RGBA")


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
