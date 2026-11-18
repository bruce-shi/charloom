"""Standing frontal heights from registered Y-down pixels, measured above the supporting sole."""
import numpy as np


# Same joint centres as pose::LANDMARK_JOINTS.
JOINTS = {"neck": "neck01"}
for part, bone in (("shoulder", "upperarm01"), ("elbow", "lowerarm01"), ("wrist", "wrist"),
                   ("hip", "upperleg01"), ("knee", "lowerleg01"), ("ankle", "foot")):
    for side in ("l", "r"):
        JOINTS[f"{part}_{side}"] = f"{bone}.{side.upper()}"


def column_edge(mask, point, edge, tolerance):
    """Edge of the vertical foreground run containing the landmark (within tolerance)."""
    x, y = point
    hits = []
    for col in range(round(x) - 2, round(x) + 3):
        if not 0 <= col < mask.shape[1]:
            continue
        steps = np.diff(np.r_[0, mask[:, col], 0].astype(np.int8))
        runs = zip(np.flatnonzero(steps == 1), np.flatnonzero(steps == -1) - 1)
        nearby = [(max(0, a - y, y - b), a, b) for a, b in runs]
        if nearby:
            distance, top, bottom = min(nearby)
            row = top if edge == "top" else bottom
            if distance <= tolerance and row not in (0, mask.shape[0] - 1):
                hits.append(row)
    if len(hits) >= 3:
        return [float(x), float(np.median(hits))]


def measure(mask, landmarks, meters_per_pixel):
    """Surface heights and landmark heights stay distinct, including both shoulders."""
    h, w = mask.shape
    marks = {k: p for k, p in landmarks.items() if k in JOINTS and
             0 <= p[0] < w and 0 <= p[1] < h}
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return {"reason": "Empty silhouette", "stations": {}}
    span = float(rows.max() - rows.min())
    radius = max(2, round(span * 0.04))
    soles = []
    for side in ("l", "r"):
        ankle = marks.get(f"ankle_{side}")
        if ankle is None:
            continue
        x, y = map(round, ankle)
        # Only the foot connected to the ankle in this band supplies a floor.
        band = mask[y:, max(0, x-radius):min(w, x+radius+1)].any(axis=1)
        occupied = np.flatnonzero(band)
        if len(occupied) and occupied[0] <= radius:
            gaps = np.flatnonzero(np.diff(occupied) > 1)
            soles.append(int(y + occupied[gaps[0] if len(gaps) else -1]))
    if not soles and not marks and rows.min() > 0 and rows.max() < h - 1:
        # Legacy diagnostics can contain a body mesh without a rig or ankle marks.
        soles = [int(rows.max())]
    if not soles or max(soles) == h - 1:
        return {"reason": "No visible supporting sole", "stations": {}}
    floor = max(soles)
    top = int(rows.min())
    stature = floor - top
    if stature <= 0:
        return {"reason": "Invalid standing height", "stations": {}}
    points = dict(marks)
    if top > 0:
        points["stature"] = [float(np.median(cols[rows == top])), top]
    for side in ("l", "r"):
        shoulder = marks.get(f"shoulder_{side}")
        if shoulder is None:
            continue
        point = column_edge(mask, shoulder, "top", span * 0.02)
        # A merged head or ear outline cannot identify the shoulder surface.
        if point and abs(shoulder[1] - point[1]) <= span * 0.12:
            points[f"shoulder_surface_{side}"] = point
    if "hip_l" in marks and "hip_r" in marks:
        pelvis = np.mean([marks["hip_l"], marks["hip_r"]], axis=0)
        point = column_edge(mask, pelvis, "bottom", span * 0.015)
        knees = [marks[k][1] for k in ("knee_l", "knee_r") if k in marks]
        # No measurable crotch when touching legs hide the gap.
        if point and knees and pelvis[1] <= point[1] < min(knees):
            points["crotch"] = point
    stations = {}
    for name, (x, y) in points.items():
        distance = floor - y
        stations[name] = {"kind": "joint" if name in JOINTS else "surface",
                          "px": [round(float(x), 2), round(float(y), 2)],
                          "height_px": round(float(distance), 2),
                          "height_mm": round(float(distance * meters_per_pixel * 1000), 2),
                          "fraction_of_stature": round(float(distance / stature), 5) if top > 0 else None}
    return {"floor_y_px": floor, "meters_per_pixel": meters_per_pixel, "stations": stations}
