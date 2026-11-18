"""Silhouette contours of the bodysuit edits, for the body fit.

    out/tools/cv/.venv/bin/python scripts/mh-suit-contours.py out/examples/mh-guide-NAME [...]

`scripts/mh-bodysuit.py` has the image model repaint the front and side
views with the body in a skin-tight suit and the head shaved bald, and
registers the edits onto the views (`views/bodysuit_<view>-reg.png`). Its
outline rows sample that silhouette every centimetre; this writes the
silhouette itself: every closed contour of the figure in each registered
edit (outer outline and enclosed gaps, between an arm and the trunk),
smoothed along the curve (Gaussian, SIGMA px) and resampled every STEP px,
in the view's pixels, to `views/suit_contours.json`:

    {"views": {"body_y000": [[[x, y], ...], ...], "body_y090": [...]}}

The fit pulls the body's own silhouette in each view onto the nearest
point of these curves, so the body follows the suit's whole outline, not
rows between which it was free to ripple.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SIGMA = 2.0
STEP = 1.5
MIN_POINTS = 40


def figure_mask(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    c = np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], dtype=float)
    bg = np.median(c, axis=0)
    m = (np.abs(img.astype(float) - bg).sum(axis=2) > 40).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m


def smooth_closed(pts: np.ndarray):
    """Gaussian along a closed curve, then resampled every STEP px."""
    n = len(pts)
    r = int(3 * SIGMA)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / SIGMA) ** 2)
    k /= k.sum()
    ext = np.concatenate([pts[-r:], pts, pts[:r]])
    sm = np.stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)], axis=1)[:n]
    seg = np.linalg.norm(np.diff(np.vstack([sm, sm[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < STEP * MIN_POINTS:
        return None
    t = np.arange(0.0, total, STEP)
    closed = np.vstack([sm, sm[:1]])
    return np.stack([np.interp(t, s, closed[:, i]) for i in range(2)], axis=1)


def main(folder: Path):
    g = json.loads((folder / "guidance.json").read_text())
    front = g["front_view"]
    yaw = int(front.split("_y")[1])
    side = f"body_y{(yaw + 90) % 360:03d}"
    out = {"views": {}, "sigma_px": SIGMA, "step_px": STEP}
    for view in (front, side):
        path = folder / "views" / f"bodysuit_{view}-reg.png"
        if not path.exists():
            continue
        m = figure_mask(path)
        contours, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        curves = []
        for c in contours:
            pts = c[:, 0, :].astype(float)
            if len(pts) < MIN_POINTS:
                continue
            sm = smooth_closed(pts)
            if sm is not None:
                curves.append(np.round(sm, 2).tolist())
        out["views"][view] = curves
    (folder / "views" / "suit_contours.json").write_text(json.dumps(out))
    print(folder.name, {v: [len(c) for c in cs] for v, cs in out["views"].items()})


if __name__ == "__main__":
    for f in sys.argv[1:]:
        main(Path(f))
