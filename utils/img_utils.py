import cv2
import numpy as np


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    Convert a depth map to an 8-bit color visualization.

    Args:
        depth: Depth map with shape (H, W). Zero and non-finite values are
            treated as invalid background when minmax is not provided.
        minmax: Optional (min, max) depth range.
        cmap: OpenCV colormap id. Pass None to return a grayscale RGB image.
    """
    x = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if minmax is None:
        valid = x > 0
        if not np.any(valid):
            mi, ma = 0.0, 1.0
        else:
            mi = float(np.min(x[valid]))
            ma = float(np.max(x[valid]))
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    x = (255 * x).astype(np.uint8)

    if cmap is None:
        x = np.repeat(x[..., None], 3, axis=-1)
    else:
        x = cv2.applyColorMap(x, cmap)

    return x, [mi, ma]


def draw_3d_box_on_img(vertices, img, color=(255, 128, 128), thickness=1):
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1, idx2 in [
                ((0, k, l), (1, k, l)),
                ((k, 0, l), (k, 1, l)),
                ((k, l, 0), (k, l, 1)),
            ]:
                cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)

    for idx1, idx2 in [((1, 0, 0), (1, 1, 1)), ((1, 1, 0), (1, 0, 1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)
