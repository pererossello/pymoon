import numpy as np
from scipy.ndimage import map_coordinates


def resize_image(img, new_shape, order=1):
    """
    Resize a numpy image array to ``new_shape`` using spline interpolation while
    enforcing periodicity along the longitude (width) axis.

    Parameters
    ----------
    img : np.ndarray
        Input array of shape (H, W) or (H, W, C).
    new_shape : tuple[int, int]
        Desired (height, width).
    order : int, optional
        Interpolation order for ``scipy.ndimage.zoom`` (0=nearest, 1=linear, ...).
    """

    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError("img must be a 2D or 3D array.")
    if len(new_shape) != 2:
        raise ValueError("new_shape must be a (height, width) tuple.")
    if any(n <= 0 for n in new_shape):
        raise ValueError("new_shape values must be positive integers.")

    h, w = arr.shape[:2]
    new_h, new_w = new_shape

    scale_y = h / new_h
    scale_x = w / new_w
    y_coords = (np.arange(new_h) + 0.5) * scale_y - 0.5
    x_coords = (np.arange(new_w) + 0.5) * scale_x - 0.5
    coords_y, coords_x = np.meshgrid(y_coords, x_coords, indexing="ij")
    coords_x = np.mod(coords_x, w)
    sample_coords = np.stack([coords_y, coords_x])

    prefilter = order > 1

    if arr.ndim == 2:
        result = map_coordinates(
            arr,
            sample_coords,
            order=order,
            mode="nearest",
            prefilter=prefilter,
        )
    else:
        channels = []
        for c in range(arr.shape[2]):
            chan = map_coordinates(
                arr[..., c],
                sample_coords,
                order=order,
                mode="nearest",
                prefilter=prefilter,
            )
            channels.append(chan)
        result = np.stack(channels, axis=-1)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        result = np.clip(np.rint(result), info.min, info.max).astype(arr.dtype)
    elif np.issubdtype(arr.dtype, np.bool_):
        result = result > 0.5
    else:
        result = result.astype(arr.dtype, copy=False)

    return result.astype(np.uint8)
