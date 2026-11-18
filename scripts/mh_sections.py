"""Measure both silhouette edges on the saved outline stations in their original cameras."""
import numpy as np


def edges(mask, first, last):
    """Endpoints of the connected foreground interval containing the station midpoint."""
    first, last = np.asarray(first, float), np.asarray(last, float)
    length = np.linalg.norm(last - first)
    if length < 1:
        return None
    direction = (last - first) / length
    center = (first + last) / 2
    h, w = mask.shape
    reach = int(np.ceil(np.hypot(w, h)))
    offsets = np.arange(-reach, reach + 1)
    coords = np.rint(center + offsets[:, None] * direction).astype(int)
    x, y = coords.T
    inside_frame = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    hits = np.zeros(len(offsets), bool)
    hits[inside_frame] = mask[y[inside_frame], x[inside_frame]]
    if not hits[reach]:
        return None
    start = np.flatnonzero(~hits[:reach])[-1] + 1 if not hits[:reach].all() else 0
    end = reach + np.flatnonzero(~hits[reach:])[0] - 1
    if start == 0 or not inside_frame[start - 1] or not inside_frame[end + 1]:
        return None
    return center + offsets[[start, end], None] * direction


def neck_transition(mask, landmarks):
    """Frontal silhouette between the lateral jaw and the shoulders, including the collar."""
    needed = ("neck", "shoulder_l", "shoulder_r")
    if any(key not in landmarks for key in needed):
        return {}
    cx, neck_y = landmarks["neck"]
    shoulder_y = np.mean([landmarks[k][1] for k in needed[1:]])
    rows = np.flatnonzero(mask[:, int(round(cx))])
    above = rows[rows < neck_y]
    if not len(above) or shoulder_y <= neck_y:
        return {}
    first = round(neck_y - .55 * (neck_y - above[0]))
    # Stop before deltoids and arms dominate the section. These are projected
    # widths of the whole visible neck-to-shoulder transition, not internal neck girth.
    stop = round(neck_y + .30 * (shoulder_y - neck_y))
    sections = {}
    for y in range(first, stop + 1, 5):
        pair = edges(mask, [cx - 1, y], [cx + 1, y])
        if pair is not None:
            sections[f"neck_transition_y{y:03d}_w"] = pair
    return sections


def measure(mask, stations, mm_per_px):
    records = {}
    for name, (p, q) in stations.items():
        ref = np.asarray([p, q], float)
        observed = edges(mask, p, q)
        length = np.linalg.norm(ref[1] - ref[0])
        entry = {"reference_px": ref.tolist(), "reference_mm": round(length * mm_per_px, 2)}
        if observed is None:
            entry["status"] = "no_contiguous_section"
        else:
            width = np.linalg.norm(observed[1] - observed[0])
            entry.update(status="measured", model_px=observed.round(2).tolist(),
                         model_mm=round(width * mm_per_px, 2),
                         error_mm=round((width - length) * mm_per_px, 2),
                         endpoint_errors_mm=(np.linalg.norm(observed - ref, axis=1) * mm_per_px).round(2).tolist())
        records[name] = entry
    return records
