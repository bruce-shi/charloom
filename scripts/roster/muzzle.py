"""Locate the lower lip beneath an overhanging muzzle and seat the mouth inside it."""

import numpy as np
from scipy.ndimage import median_filter


def bite_plane(x, y, z, front):
    """Use the lower side of a resolved overbite, in the calibrated mouth view."""
    width = np.ptp(x)
    mid = len(x) // 2
    projection = z[mid] - np.mean(z[[0, -1]])
    if projection < width * .45:
        return None
    offsets = np.linspace(-.065, .065, 49) * width
    heights = y[:, None] + offsets
    depths = front(np.repeat(x, len(offsets)), heights.ravel()).reshape(heights.shape)
    step = np.diff(depths, axis=1)
    lip = step.argmax(axis=1)
    rows = np.arange(len(x))
    ledge = step[rows, lip] > .025 * width
    if not ledge[mid - 2:mid + 3].any():
        return None
    # Stay on the chin side of the shelf. The upper muzzle can be several
    # millimetres farther forward at the same pixel height.
    lower = np.where(ledge, heights[rows, np.maximum(lip - 1, 0)], y)
    lower = median_filter(lower, size=3, mode="nearest")
    depth = front(x, lower)
    reach = depth[mid] - np.mean(depth[[0, -1]])
    if reach <= 0:
        return None
    pitch = (np.mean(lower[[0, -1]]) - lower[mid]) / reach
    return lower, depth, float(np.clip(pitch, 0., 1.))


def fit_oral_mesh(teeth, tongue, mouth, pitch, ray_distance):
    """Fit both arches and tongue into the measured width and thickness at bite height."""
    xs, top, bottom, depth = mouth
    cx = (xs[0] + xs[-1]) / 2
    width = np.ptp(xs)
    cz = np.interp(cx, xs, depth)
    cy = np.interp(cx, xs, (top + bottom) * .5)
    points = np.vstack([teeth, tongue])
    x, y, z = points.T
    # Measure a common lateral scale for all oral parts on the original shell.
    tx, _, tz = teeth.T
    dz = cz - tz
    origins = np.c_[np.full(len(tx), cx), cy + pitch * dz, tz]
    side_dirs = np.zeros_like(origins)
    side_dirs[:, 0] = np.where(tx < cx, -1, 1)
    room = ray_distance(origins, side_dirs)
    ratio = (room - width * .055) / np.maximum(np.abs(tx - cx), width * .005)
    scale_x = min(1., float(ratio.min()))
    if scale_x <= 0:
        raise ValueError("Measured muzzle has no room for dental arches")
    x = cx + (x - cx) * scale_x
    bite_y = cy + pitch * (cz - z)
    origins = np.c_[x, bite_y, z]
    vertical = np.zeros_like(origins)
    vertical[:, 1] = np.where(y >= cy, 1, -1)
    height = ray_distance(origins, vertical)
    # Compress the whole arch by the same factor to retain tooth/gum contours.
    # Use a shared lower-arch scale for the tongue as well.
    tooth_y = teeth[:, 1]
    available = height[:len(teeth)]
    scales = []
    for sign in (1, -1):
        mask = (tooth_y - cy) * sign > .0001
        room = available[mask] * .70 / np.abs(tooth_y[mask] - cy)
        scales.append(float(room.min(initial=1.)))
    offset = np.minimum(np.abs(y - cy) * np.where(y >= cy, scales[0], scales[1]), height * .70)
    fitted = np.asarray(np.c_[x, bite_y + vertical[:, 1] * offset, z], dtype=points.dtype)
    return fitted[:len(teeth)], fitted[len(teeth):]
