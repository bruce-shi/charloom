"""Body outlines from bodysuit edits of the guidance views.

    out/tools/cv/.venv/bin/python scripts/mh-bodysuit.py out/examples/mh-guide-NAME [--edit] [--write]

(OpenCV and numpy: the vision environment, see scripts/mh-cv-guide.py.)

GPT Image 2.5 through fal and genmedia (FAL_KEY or established genmedia configuration) repaints the front and side body views, side by
side in ONE sheet so both show the same body, with every garment replaced
by a plain, skin-tight green suit and the head shaved bald: the body's own
shape under the clothes and the skull under the hair. The sheet is split
back into the views (`views/bodysuit_<view>.png`). The suit's silhouette is
then measured at the outline stations (chest, waist, hips, thighs, knees,
shins, and the skull), from the guidance's joint marks, and written as
`outline_*` marks (`--write`), so the body's size under the clothes is
measured, not guessed.

  --edit   reuse an identical sheet or request it once; --refresh requests a fresh sample
  --back   add the back view to the sheet (three panels, lower resolution)
  --write  write the outline marks into guidance.json (a backup is kept)

A bad edit is rejected, not trusted: the hands (bare in every edit) must
stay within a few pixels of the original's, and every station must be
found. Writes views/bodysuit.json with the measures and the checks.
"""
from measurement_images import sheets
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import mh_heights

MODEL = "openai/gpt-image-2.5/sunburst/edit"
SHEET = (
    "This sheet shows the SAME 3D character in {n} panels side by side: {panels}. "
    "Edit every panel the same way so they show one consistent body. "
)
PROMPT = (
    "Edit this 3D character render. Replace ALL clothing, armour, capes, coats, robes, "
    "skirts, sashes, belts, straps, shoes, socks, accessories and props with one plain, "
    "skin-tight, matte, solid bright green full-body suit (like a motion-capture suit) "
    "covering the body from the neck to the wrists and ankles, so the true body shape "
    "under the clothes shows: realistic human anatomy and proportions for this character "
    "(the same build: slim, average, muscular or heavy as the clothes suggest). Keep the "
    "face exactly as it is. Bare feet. Hands bare, the same size and in the same place "
    "at the end of each arm, with the same number of fingers, but relaxed and natural: "
    "fingers gently open, slightly curved and separated, the thumb apart from them; "
    "fix any fused, bent-back or malformed fingers. In profile, keep a vertical palm "
    "against the outside of the thigh, fingertips pointing down and clear of the "
    "front waistline, wherever the original arm and wrist positions allow it. "
    "Preserve the source shoulder, elbow and wrist positions. Remove ALL head hair, and any hat, "
    "helmet, hood, hair ornament or beard: a completely bald, smooth scalp, so the true "
    "shape of the skull shows. Keep the eyebrows, eyelashes and ears, and any horns or "
    "animal ears. Keep the exact same pose, position, size, camera and framing in every "
    "panel: do not move, turn or rescale anything. Keep the plain flat grey background. "
    "No text, no panel borders, no shadows on the background."
)
W, H = 768, 1024
SUIT = (40, 200, 40)
STATIONS = ("chest", "waist", "hips", "thigh", "knee", "shin")


# Sheet sizes: views side by side, 768x1024 each; the request size keeps
# the sheet's aspect (edges multiples of 16, about 2K on the long edge).
SHEET_SIZE = {2: (2048, 1360), 3: (2560, 1136)}


def edit_sheet(folder: Path, views, names, refresh=False):
    """One edit of the views side by side (`names` say which is which in the
    prompt); split back into views/bodysuit_<view>.png (768x1024) and
    bodysuit_<view>-2k.png (the panel at the returned size)."""
    n = len(views)
    w, h = SHEET_SIZE[n]
    panels = ", ".join(f"{name} ({where})" for name, where in
                       zip(names, ["left", "middle", "right"] if n == 3 else ["left", "right"]))
    prompt = SHEET.format(n=n, panels=panels) + PROMPT.replace("this 3D character render", "the sheet")
    with tempfile.TemporaryDirectory() as tmp:
        big = Path(tmp) / "in.png"
        subprocess.run(["magick", *[str(folder / "views" / f"{v}.png") for v in views], "+append",
                        "-resize", f"{w}x{h}!", str(big)], check=True)
        images, receipt = sheets(folder, big, dict(model=MODEL, prompt=prompt, size=f"{w}x{h}", quality="medium", n="1"), refresh)
    sheet = folder / "views" / "bodysuit_sheet-2k.png"
    sheet.write_bytes(images[0])
    for i, v in enumerate(views):
        dst = folder / "views" / f"bodysuit_{v}.png"
        subprocess.run(["magick", str(sheet), "-crop", f"{w // n}x{h}+{i * (w // n)}+0", "+repage",
                        str(dst.with_name(dst.stem + "-2k.png"))], check=True)
        subprocess.run(["magick", str(sheet), "-resize", f"{W * n}x{H}!", "-crop", f"{W}x{H}+{i * W}+0",
                        "+repage", str(dst)], check=True)
    return {**receipt, "prompt": prompt, "size": f"{w}x{h}"}


def image(path: Path):
    """BGR image at the view's size (OpenCV)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img.shape[1] != W or img.shape[0] != H:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


def corner_colour(path: Path):
    """The background colour (RGB): median of the four corners."""
    img = image(path)
    c = np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], dtype=float)
    b, g, r = np.median(c, axis=0)
    return int(r), int(g), int(b)


def clean(m):
    """A mask's largest connected component, seams closed, specks gone."""
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == keep


def calibrated_mask(img):
    """Silhouette in a calibrated PNG rendered against a flat background."""
    corners = img[[0, 0, -1, -1], [0, -1, 0, -1]]
    background = np.median(corners, axis=0)
    return clean(np.any(img != background, axis=2))


def mask(path: Path, suit_only=False):
    """The figure (anything off the background), or with `suit_only` the
    suit's green (HSV hue 35-90, shaded edges included), as a bool array."""
    img = image(path)
    if suit_only:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (35, 70, 50), (90, 255, 255)) > 0
    else:
        bg = np.array(corner_colour(path)[::-1], dtype=int)
        m = np.abs(img.astype(int) - bg).sum(axis=2) > 40
    return clean(m)


def collar_mask(path: Path, below_y: int):
    """The suit alone, without green skin: pixels near the suit's own colour,
    its median hue and saturation on the torso (the green of
    `mask(suit_only=True)` below `below_y`, the neck mark). A fixed bright
    green missed the dimlight's darker suit; the broad suit hue range took
    the troll's olive skin (hue 33-36, saturation 100-125; its suit 52, 221)."""
    hsv = cv2.cvtColor(image(path), cv2.COLOR_BGR2HSV)
    green = mask(path, suit_only=True)
    green[:max(0, below_y)] = False
    if green.sum() < 100:
        return green
    h = float(np.median(hsv[..., 0][green]))
    sat = float(np.median(hsv[..., 1][green]))
    m = (np.abs(hsv[..., 0].astype(float) - h) <= 8) & (hsv[..., 1] >= 0.7 * sat) & (hsv[..., 2] >= 40)
    return clean(m)


def runs(row):
    """Runs of True in a mask row, as (first, last) column pairs."""
    r = np.asarray(row, dtype=np.int8)
    d = np.diff(np.concatenate(([0], r, [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def run_at(row, x):
    if x is None:
        return None
    for a, b in runs(row):
        if a - 2 <= x <= b + 2:
            return a, b
    return None


# Body proportions as shares of the suit figure's height: (name, human
# range, extreme beyond). Set from the human test characters (legs 0.51-0.56,
# arms 0.27-0.30, upper arm 0.046-0.075, thigh 0.107-0.163, trunk depth
# 0.133-0.231) and the cartoons (orc legs 0.39, goblin arms 0.35, troll upper
# arm 0.115, brute thigh 0.36 and trunk depth 0.55). The head is left out:
# it is fitted on its own, and the fox's big head on a human body is no
# reason to free its body.
PROPORTION_BANDS = {
    "leg_length": ((0.48, 0.60), (0.44, 0.64)),
    "arm_length": ((0.25, 0.31), (0.22, 0.34)),
    "upper_arm_width": ((0.035, 0.085), (0.025, 0.10)),
    "thigh_width": ((0.095, 0.175), (0.08, 0.20)),
    "trunk_depth": ((0.11, 0.245), (0.09, 0.28)),
}


def measure_proportions(fm, fmask, out):
    """`human`, `stylized` or `extreme` from the suit: each proportion inside
    its human range, past it, or past the extreme bound."""
    rows = np.flatnonzero(fmask.any(axis=1))
    if not len(rows):
        return {}
    top, bottom = int(rows.min()), int(rows.max())
    height = max(1, bottom - top)

    def widest(prefix, kind):
        best = 0.0
        for k, (_, p) in out.items():
            if k.startswith(prefix) and k.endswith(f"_{kind}1"):
                q = out[k[:-1] + "2"][1]
                best = max(best, ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5)
        return best or None
    m = {}
    if all(f"hip_{s}" in fm for s in "lr"):
        m["leg_length"] = (bottom - (fm["hip_l"][1] + fm["hip_r"][1]) / 2) / height
    arms = [((fm[f"shoulder_{s}"][0] - fm[f"wrist_{s}"][0]) ** 2 + (fm[f"shoulder_{s}"][1] - fm[f"wrist_{s}"][1]) ** 2) ** 0.5
            for s in "lr" if f"shoulder_{s}" in fm and f"wrist_{s}" in fm]
    if arms:
        m["arm_length"] = sum(arms) / len(arms) / height
    for name, prefix, kind in (("upper_arm_width", "outline_arm_", "w"), ("thigh_width", "outline_leg_", "w"),
                               ("trunk_depth", "outline_trunk", "d")):
        w = widest(prefix, kind)
        if w:
            m[name] = w / height
    reasons, level = [], 0
    for name, value in m.items():
        (lo, hi), (xlo, xhi) = PROPORTION_BANDS[name]
        if value < xlo or value > xhi:
            reasons.append(f"{name} {value:.3f} beyond {xlo}-{xhi}")
            level = 2
        elif value < lo or value > hi:
            reasons.append(f"{name} {value:.3f} outside {lo}-{hi}")
            level = max(level, 1)
    return {"verdict": ["human", "stylized", "extreme"][level] if m else None,
            "measures": {k: round(v, 3) for k, v in m.items()}, "reasons": reasons}


def marks(g, view):
    """Joint marks in `view` (px), by name."""
    out = {}
    for name, obs in g["landmarks"].items():
        for o in obs:
            if o["view"] == view:
                out[name] = o["px"]
    return out


def parallel_mark(g, cameras, view, name):
    """Project a marked pixel between parallel calibrated orthographic views.

    Close-ups retain depth-independent image coordinates along the same right/up
    axes. An oblique observation cannot be projected without its surface depth.
    """
    target = cameras[view]
    candidates = []
    for observation in g["landmarks"].get(name, []):
        camera = cameras.get(observation["view"])
        if camera is None or not all(np.allclose(camera[k], target[k], atol=1e-6)
                                     for k in ("right", "up")):
            continue
        x, y = observation["px"]
        point = (np.asarray(camera["center"])
                 + np.asarray(camera["right"]) * (x - camera["width"] / 2) * camera["meters_per_pixel"]
                 + np.asarray(camera["up"]) * (camera["height"] / 2 - y) * camera["meters_per_pixel"])
        delta = point - target["center"]
        pixel = [target["width"] / 2 + np.dot(delta, target["right"]) / target["meters_per_pixel"],
                 target["height"] / 2 - np.dot(delta, target["up"]) / target["meters_per_pixel"]]
        candidates.append((camera["meters_per_pixel"], pixel))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


def registration_patches(g, cameras, front):
    """Use the unchanged lower face and palm surfaces, avoiding repainted cuffs."""
    patches = {}
    if front not in cameras:
        return patches
    for side in ("l", "r"):
        palm = parallel_mark(g, cameras, front, "palm_" + side)
        if palm is not None:
            patches["palm_" + side] = (*palm, 34, 28, 20)
    nose, left, right = [parallel_mark(g, cameras, front, name)
                         for name in ("nose_tip", "mouth_l", "mouth_r")]
    if all(point is not None for point in (nose, left, right)):
        mouth = (np.asarray(left) + right) / 2
        width = max(16, int(abs(left[0] - right[0]) * 1.25))
        height = max(16, int(abs(mouth[1] - nose[1]) * 1.5))
        center = (np.asarray(nose) + mouth) / 2
        eyes = [parallel_mark(g, cameras, front, "eye_" + side) for side in ("l", "r")]
        if all(point is not None for point in eyes):
            span = abs(eyes[0][0] - eyes[1][0])
            top = min(point[1] for point in eyes) - span * 0.25
            bottom = max(left[1], right[1]) + span * 0.2
            width = max(width, int(span * 1.6))
            height = max(height, int(bottom - top))
            center = [(eyes[0][0] + eyes[1][0]) / 2, (top + bottom) / 2]
        patches["face"] = (*center, width, height, 40)
    return patches


def match_patch(orig, edit, cx, cy, w, h, radius):
    """Bound every candidate patch to the requested search rectangle."""
    tx, ty = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    template = orig[ty:ty + h, tx:tx + w].astype(np.float32)
    sx, sy = max(0, tx - radius), max(0, ty - radius)
    search = edit[sy:ty + h + radius, sx:tx + w + radius].astype(np.float32)
    if template.size == 0 or any(search.shape[i] < template.shape[i] for i in (0, 1)):
        return None
    scores = cv2.matchTemplate(search, template, cv2.TM_SQDIFF)
    score, _, location, _ = cv2.minMaxLoc(scores)
    th, tw = template.shape[:2]
    return ((tx + tw / 2, ty + th / 2),
            (sx + location[0] + tw / 2, sy + location[1] + th / 2),
            float(np.sqrt(max(0, score) / template.size)) / 255)


def main(folder: Path, do_edit: bool, do_write: bool, back: bool = False, refresh: bool = False):
    g = json.loads((folder / "guidance.json").read_text())
    front = g["front_view"]
    yaw = int(front.split("_y")[1])
    side = f"body_y{(yaw + 90) % 360:03d}"
    report = {"model": MODEL, "views": {}, "stations": {}, "checks": {}, "accepted": True}
    edited = {v: folder / "views" / f"bodysuit_{v}.png" for v in (front, side)}
    if not do_edit and not all(p.exists() for p in edited.values()):
        raise FileNotFoundError("Missing measurement panels; use --edit to generate the sheet")
    if do_edit:
        views, names = [front, side], ["the front view", "the side view (profile)"]
        if back:
            views.append(f"body_y{(yaw + 180) % 360:03d}")
            names.append("the back view")
        report["sheet"] = edit_sheet(folder, views, names, refresh)
    fm = marks(g, front)
    need = ["shoulder_l", "shoulder_r", "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r"]
    if any(n not in fm for n in need):
        raise SystemExit(f"front view lacks joint marks: {[n for n in need if n not in fm]}")
    ys = lambda a, b: (fm[a][1] + fm[b][1]) / 2
    shoulder, hip = ys("shoulder_l", "shoulder_r"), ys("hip_l", "hip_r")
    knee, ankle = ys("knee_l", "knee_r"), ys("ankle_l", "ankle_r")
    rows = {
        "chest": shoulder + 0.30 * (hip - shoulder),
        "waist": shoulder + 0.62 * (hip - shoulder),
        "hips": hip + 0.05 * (knee - hip),
        "thigh": hip + 0.50 * (knee - hip),
        "knee": knee,
        "shin": knee + 0.40 * (ankle - knee),
    }
    centre = (fm["hip_l"][0] + fm["hip_r"][0]) / 2
    omask = calibrated_mask(image(folder / "views" / f"{front}.png"))
    # Registration: the bare hands are unchanged by the edit, but the edit
    # may shift or rescale the figure by a few percent (the anime
    # characters' by 11-22 px). The fingertips (lowest point of each
    # hand) map the edit back onto the view by a similarity, applied to
    # the edited image before it is measured.
    # Each hand's image patch (around its wrist mark) matched in the edit:
    # two translations give the similarity (scale from the hands' spread).
    # Anchors: the face (kept as it is by the edit) and the back of each
    # hand at its wrist mark. A hand
    # counts only when it matches well (normalised RMSE < 0.08): the
    # anime mage's smooth hands matched off by 8 px and gave a false 14 %
    # scale. Scale and shift by least squares (no rotation); the face
    # alone gives the shift at scale 1.
    cameras = {c["id"]: c for c in json.loads((folder / "views.json").read_text())["cameras"]}
    original_front, edited_front = image(folder / "views" / f"{front}.png"), image(edited[front])
    anchors = {name: match_patch(original_front, edited_front, *patch)
               for name, patch in registration_patches(g, cameras, front).items()}
    report["checks"]["anchors"] = {k: (None if v is None else [round(v[1][0] - v[0][0], 1), round(v[1][1] - v[0][1], 1), round(v[2], 3)])
                                   for k, v in anchors.items()}
    good = [v for k, v in anchors.items() if v is not None and (k == "face" or v[2] < 0.08)]
    if not good:
        raise SystemExit("nothing matched for registration")
    to, te = [g[0] for g in good], [g[1] for g in good]
    mid_o = (sum(p[0] for p in to) / len(to), sum(p[1] for p in to) / len(to))
    mid_e = (sum(p[0] for p in te) / len(te), sum(p[1] for p in te) / len(te))
    if len(good) >= 2:
        num = sum((o[0] - mid_o[0]) * (e[0] - mid_e[0]) + (o[1] - mid_o[1]) * (e[1] - mid_e[1]) for o, e in zip(to, te))
        den = sum((e[0] - mid_e[0]) ** 2 + (e[1] - mid_e[1]) ** 2 for e in te)
        scale = num / max(1e-6, den)
    else:
        scale = 1.0
    report["checks"]["scale"] = round(scale, 4)
    report["checks"]["shift_px"] = [round(mid_o[0] - mid_e[0], 1), round(mid_o[1] - mid_e[1], 1)]
    if not 0.85 < scale < 1.15:
        report["accepted"] = False

    def register(src, dst, me, mo):
        bg = corner_colour(src)
        subprocess.run(["magick", str(src), "-virtual-pixel", "background", "-background",
                        "rgb(%d,%d,%d)" % bg, "-distort", "SRT",
                        f"{me[0]},{me[1]} {scale} 0 {mo[0]},{mo[1]}", str(dst)], check=True)
        return dst
    reg_front = register(edited[front], edited[front].with_name(edited[front].stem + "-reg.png"), mid_e, mid_o)
    fmask = mask(reg_front)
    # The side view: painted in the same sheet as the front, so the front's
    # transform carries over (the edit shifts the whole sheet). The feet
    # (lowest rows; shoes become bare feet, a sole's few px) only check it;
    # a side panel off by more than 10 px is placed by its feet instead.
    def feet(m):
        rows = np.flatnonzero(m.any(axis=1))
        bottom = int(rows.max())
        xs = np.flatnonzero(m[bottom - 30:bottom + 1].any(axis=0))
        return (float(xs.mean()), bottom)
    s_orig = calibrated_mask(image(folder / "views" / f"{side}.png"))
    side_reg = edited[side].with_name(edited[side].stem + "-reg.png")
    reg_side = register(edited[side], side_reg, mid_e, mid_o)
    fo, fr = feet(s_orig), feet(mask(reg_side))
    report["checks"]["side_feet_px"] = [round(fr[0] - fo[0], 1), fr[1] - fo[1]]
    if abs(fr[0] - fo[0]) > 10 or abs(fr[1] - fo[1]) > 10:
        report["checks"]["side_placed_by"] = "feet"
        reg_side = register(edited[side], side_reg, feet(mask(edited[side])), fo)
    out = {}
    width = abs(fm["shoulder_l"][0] - fm["shoulder_r"][0])
    smask = mask(reg_side)
    # The front view is measured on the suit's green: a hand beside a thigh
    # or a hip is skin, not body width; the edges are where the suit ends.
    fgreen = mask(reg_front, suit_only=True)
    sm = marks(g, side)
    # Continuous outlines: a row every STEP px (about 1 cm) down the trunk
    # and each leg and across each arm, both edges, so the body follows the
    # suit's whole outline, not a few stations between which it was guessed.
    # Names: outline_{trunk,leg_l,leg_r,arm_l,arm_r}NN_{w,d}{1,2}.
    STEP = 10
    # A leg's line (hip, knee, ankle marks) at row y.
    def leg_x(sd, y):
        pts = [fm[f"{j}_{sd}"] for j in ("hip", "knee", "ankle")]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if y <= y1 or (x1, y1) == tuple(pts[-1]):
                t = (y - y0) / (y1 - y0) if abs(y1 - y0) > 1 else 0.0
                return x0 + (x1 - x0) * min(1.0, max(0.0, t))

    def spans_legs(r, y):
        """A trunk row's run covers both legs' lines (below the crotch the
        run through the hips' middle column is one thigh: the hiker's last
        trunk rows read one leg, shifted 87 mm, and grew a shelf)."""
        return r is not None and r[0] <= leg_x("r", y) and r[1] >= leg_x("l", y)
    crotch = next((y for y in range(int(hip), int(knee)) if not spans_legs(run_at(fgreen[y], centre), y)),
                  int(hip + 0.15 * (knee - hip)))
    report["stations"]["crotch_y"] = crotch

    def side_x(y, a, b):
        """The side figure's column at row y on the line between the side
        view's marks a and b (None without them)."""
        if a not in sm or b not in sm:
            return None
        (x0, y0), (x1, y1) = sm[a], sm[b]
        t = (y - y0) / (y1 - y0) if abs(y1 - y0) > 1 else 0.0
        return x0 + (x1 - x0) * min(1.0, max(0.0, t))

    def side_run(m, y, x):
        r = run_at(m[y], x) if x is not None else None
        if r is None:
            rs = runs(m[y])
            r = max(rs, key=lambda q: q[1] - q[0]) if rs else None
        return r
    counts = {}
    # Raw edges per part and kind, smoothed along the part before writing:
    # key -> [(index, view, geometry, a, b)], geometry = ("row", y) or
    # ("normal", cx, cy, nx, ny) with a, b the edge offsets along it.
    raw = {}

    def put(key, i, view, geom, a, b):
        raw.setdefault(key, []).append((i, view, geom, a, b))

    def group_run(m, y):
        """The largest run with its neighbours across gaps under 6 % of the
        shoulder width (a sleeve's seam splits the side view's trunk)."""
        rs = runs(m[y])
        if not rs:
            return None
        k = max(range(len(rs)), key=lambda j: rs[j][1] - rs[j][0])
        a, b = rs[k]
        gap = 0.06 * width
        for q in rs[k + 1:]:
            if q[0] - b <= gap:
                b = q[1]
        for q in reversed(rs[:k]):
            if a - q[1] <= gap:
                a = q[0]
        return a, b
    # Trunk: from below the shoulders (the arms hang free there) to the crotch.
    # From just below the neck mark: the side view's arm hangs inside the
    # trunk's depth, so the upper chest and shoulders are measured too (the
    # front view's rows there merge with the arms and are left out below).
    top = int(fm["neck"][1] + 4) if "neck" in fm else int(shoulder + 0.15 * (hip - shoulder))
    for i, y in enumerate(range(top, crotch - 3, STEP)):
        r = run_at(fgreen[y], centre)
        # A row whose run reaches an arm's line (shoulder to elbow) has the
        # arm in it: left out, not clipped (clipped to 1.15 shoulder widths
        # the slim herbalist's upper chest widened into spikes).
        def arm_x(sd):
            (x0, y0), (x1, y1) = fm[f"shoulder_{sd}"], fm[f"elbow_{sd}"]
            return x0 + (x1 - x0) * min(1.0, max(0.0, (y - y0) / (y1 - y0))) if abs(y1 - y0) > 1 else x0
        if spans_legs(r, y) and not (r[0] <= arm_x("r") or r[1] >= arm_x("l")):
            put(("trunk", "w"), i, front, ("row", y), r[0], r[1])
        # Depth on the whole silhouette: the A-pose arm hangs inside the
        # trunk's depth, and on the suit's green alone the orc's bare hand
        # over its hip split the run (the hips read 104 px of their ~200).
        r = group_run(smask, y)
        if r is not None:
            put(("trunk", "d"), i, side, ("row", y), r[0], r[1])
        counts["trunk"] = i + 1
    # Down to 97 % of the way to the ankle (the cut counts vertices a fifth
    # on the leg's bones, `CUT_MEMBER`, so the foot's do not hide the
    # ankle): stopped at 85 %, the orc's thick lower leg was held down to
    # its ankle and met its foot in a ring.
    for i, y in enumerate(range(crotch + 3, int(knee + 0.97 * (ankle - knee)), STEP)):
        for sd in ("l", "r"):
            r = run_at(fgreen[y], leg_x(sd, y))
            if r is not None and r[1] - r[0] < 0.9 * width:
                put((f"leg_{sd}", "w"), i, front, ("row", y), r[0], r[1])
        # The side view sees the near leg; the far one stands behind it at
        # the same depth (its rays pass through both).
        r = side_run(smask, y, side_x(y, "knee_l", "hip_l") if y < sm.get("knee_l", [0, 0])[1] else side_x(y, "knee_l", "ankle_l"))
        if r is not None and r[1] - r[0] < 0.9 * width:
            for sd in ("l", "r"):
                put((f"leg_{sd}", "d"), i, side, ("row", y), r[0], r[1])
        counts["leg"] = i + 1
    # Arms: across the arm, perpendicular to it, every STEP px along the
    # shoulder-elbow-wrist line; where the arm's inner side runs into the
    # trunk its outer half is mirrored about the line.
    for sd in ("l", "r"):
        js = [fm.get(f"{j}_{sd}") for j in ("shoulder", "elbow", "wrist")]
        if None in js:
            continue
        segs = list(zip(js, js[1:]))
        lens = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in segs]
        total = sum(lens)
        i = 0
        dist = 0.15 * lens[0]
        while dist < total - 0.06 * lens[1]:
            k = 0 if dist < lens[0] else 1
            a, b = segs[k]
            f = (dist - (0 if k == 0 else lens[0])) / lens[k]
            cx, cy = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            nx, ny = -(b[1] - a[1]) / lens[k], (b[0] - a[0]) / lens[k]
            inside = lambda t: (0 <= int(cy + ny * t) < H and 0 <= int(cx + nx * t) < W
                                and fgreen[int(cy + ny * t)][int(cx + nx * t)])
            # MediaPipe's arm line can run along the sleeve's edge or just
            # off it: start from the nearest green pixel across the arm.
            start = next((t for d in range(0, 25) for t in (d, -d) if inside(t)), None)
            if start is not None:
                t0 = t1 = start
                while inside(t0 - 1) and t0 > -200:
                    t0 -= 1
                while inside(t1 + 1) and t1 < 200:
                    t1 += 1
                # Measured edge to edge, never mirrored: mirrored about the
                # off-centre line the herbalist's and stocky woman's
                # forearms read twice their short side. A row running into
                # the trunk is dropped as an outlier below (wider than 1.6x
                # the arm's median).
                if 4 <= t1 - t0 < 199:
                    put((f"arm_{sd}", "w"), i, front, ("normal", cx, cy, nx, ny), t0, t1)
            i += 1
            dist += STEP
        counts[f"arm_{sd}"] = i
    # Feet: their true size, length heel to toe in the side view's bottom
    # rows (both feet overlap there; the far one stands behind the near
    # one), width per foot in the front view's bottom rows. Measured on the
    # original views when the character wears no footwear (a garment named
    # shoe, boot, sandal or footwear): a bare or wrapped foot is the real
    # one there, while the image model redrew the troll's, goblin's and
    # brute's feet smaller. In footwear, on the suit's bare foot (grown to
    # its shoe the orc's foot was 1.5x MakeHuman's).
    shod = any(re.search(r"shoe|boot|sandal|footwear|clog|slipper", x.get("name", ""), re.I)
               for x in g.get("garments", []))
    report["stations"]["feet_from"] = "suit" if shod else "original"
    feet_front = fmask if shod else omask
    feet_side = smask if shod else s_orig
    srows = np.flatnonzero(feet_side.any(axis=1))
    frows = np.flatnonzero(feet_front.any(axis=1))
    if len(srows) and len(frows):
        fig = int(srows.max() - srows.min())
        sb = int(srows.max())
        band = feet_side[sb - max(3, int(0.02 * fig)):sb + 1]
        cols = np.flatnonzero(band.any(axis=0))
        if len(cols):
            y = sb - 2
            for sd in ("l", "r"):
                out[f"outline_foot_{sd}_d1"], out[f"outline_foot_{sd}_d2"] = (side, [int(cols.min()), y]), (side, [int(cols.max()), y])
            report["stations"]["foot_length_px"] = int(cols.max() - cols.min())
        fb = int(frows.max())
        for sd in ("l", "r"):
            ax = fm.get(f"ankle_{sd}")
            if ax is None:
                continue
            # A little above the sole (2-5 % of the height): the body's
            # foot stands inside its shoe, above the suit's bare sole, and a
            # cut at the floor missed it.
            best = None
            for y in range(fb - int(0.05 * fig), fb - int(0.02 * fig)):
                r = run_at(feet_front[y], ax[0])
                if r is not None and r[1] - r[0] < 0.6 * width and (best is None or r[1] - r[0] > best[1] - best[0]):
                    best = (r[0], r[1], y)
            if best:
                out[f"outline_foot_{sd}_w1"], out[f"outline_foot_{sd}_w2"] = (front, [best[0], best[2]]), (front, [best[1], best[2]])
                report["stations"][f"foot_{sd}_width_px"] = best[1] - best[0]
    # Neck: across its own axis, from the neck's base to the jaw, on the
    # whole silhouette (the suit ends at the neck's base; above it is skin).
    # The axis runs from the base to the skull's middle: a cartoon's neck
    # leans forward, and cut in horizontal rows the orc's read 19-32 %
    # too deep and ran into its chin after three rows. Each view: the base
    # is the silhouette's middle at the neck mark's row, the head's middle
    # the run's middle a quarter of the way down from the crown. The jaw:
    # where the cut widens past 1.15x the neck's narrowest so far.
    def head_axis(m, ny, x_hint):
        r0 = run_at(m[int(ny)], x_hint)
        if r0 is None:
            return None
        bx = (r0[0] + r0[1]) / 2
        y = int(ny)
        while y > 0 and run_at(m[y - 1], bx) is not None:
            y -= 1
        crown = y
        if ny - crown < 0.08 * H:
            return None
        sy = int(crown + 0.25 * (ny - crown))
        r1 = run_at(m[sy], bx)
        if r1 is None:
            return None
        return (bx, float(ny)), ((r1[0] + r1[1]) / 2, float(sy))
    if "neck" in fm:
        ny0 = fm["neck"][1]
        rs = runs(smask[int(ny0) + 2])
        sx = (lambda q: (q[0] + q[1]) / 2)(max(rs, key=lambda q: q[1] - q[0])) if rs else None
        suits = {front: collar_mask(reg_front, int(ny0)), side: collar_mask(reg_side, int(ny0))}
        for kind, view, m, x_hint in (("w", front, fmask, fm["neck"][0]), ("d", side, smask, sx)):
            if x_hint is None:
                continue
            # The neck starts where the suit ends: up the silhouette's
            # middle from the neck mark to the first row the suit no longer
            # fills. Measured from the neck mark (at the clavicles), the
            # cuts ran over the trapezius and the upper back: the troll's
            # neck read 290 px against a 219 px shoulder span (the orc's
            # 176 against 177, a hiker's 68 against 176) and the fit grew a
            # hood of neck over the shoulders.
            suit = suits[view]
            x, collar = x_hint, None
            for y in range(int(ny0), 0, -1):
                r = run_at(m[y], x)
                if r is None:
                    break
                x = (r[0] + r[1]) / 2
                third = max(1, (r[1] - r[0]) // 3)
                core = suit[y][r[0] + third:r[1] - third + 1]
                if core.size and core.mean() < 0.5:
                    collar = y
                    break
            if collar is None:
                continue
            ax = head_axis(m, collar, x)
            if ax is None:
                continue
            (bx, by), (hx, hy) = ax
            ln = ((hx - bx) ** 2 + (hy - by) ** 2) ** 0.5
            ux, uy = (hx - bx) / ln, (hy - by) / ln
            nx, ny = -uy, ux
            # Every cut from the collar halfway up to the skull's middle;
            # the neck is the narrowest stretch of them, the rows within
            # 15 % of the narrowest and contiguous with it. Seen from the
            # front a hunched body's shoulders rise to its collar (the
            # troll's first cuts were 268 px against a 219 px shoulder
            # span); above the narrowest the jaw widens out.
            # Each cut is the shortest through its point within 30° of the
            # axis' normal: the neck's own cross-section. Square to an axis
            # aimed at the skull's middle, the dimlight's side cuts ran from
            # the chin's underside to the nape (85 px deep, 60 wide).
            def cut(cx, cy, ex, ey):
                inside = lambda q: (0 <= int(cy + ey * q) < H and 0 <= int(cx + ex * q) < W
                                    and m[int(cy + ey * q)][int(cx + ex * q)])
                if not inside(0):
                    return None
                t0 = t1 = 0
                while inside(t0 - 1) and t0 > -300:
                    t0 -= 1
                while inside(t1 + 1) and t1 < 300:
                    t1 += 1
                return t0, t1
            cuts = []
            for t in np.arange(0.0, 0.5 * ln, STEP / 2):
                cx, cy = bx + ux * t, by + uy * t
                best = None
                for deg in range(-30, 31, 5):
                    c, sn = np.cos(np.radians(deg)), np.sin(np.radians(deg))
                    ex, ey = nx * c - ny * sn, nx * sn + ny * c
                    r = cut(cx, cy, ex, ey)
                    if r and (best is None or r[1] - r[0] < best[1] - best[0]):
                        best = (r[0], r[1], ex, ey)
                if best is None:
                    break
                cuts.append((cx, cy) + best)
            if not cuts:
                continue
            widths = [c[3] - c[2] for c in cuts]
            k = int(np.argmin(widths))
            lo = hi = k
            while lo > 0 and widths[lo - 1] <= 1.15 * widths[k]:
                lo -= 1
            while hi + 1 < len(cuts) and widths[hi + 1] <= 1.15 * widths[k]:
                hi += 1
            for i, (cx, cy, t0, t1, ex, ey) in enumerate(cuts[lo:hi + 1]):
                put(("neck", kind), i, view, ("normal", cx, cy, ex, ey), t0, t1)
                counts[f"neck_{kind}"] = i + 1
    # Smooth each edge along its part: a 5-row median, values off it by
    # more than 4 px or 6 % of the width replaced by it, then a 3-row mean.
    # Row to row the suit's edges jumped (the orc's trunk depth 433, 239,
    # 266, 423 mm where a sleeve split it) and the body took them in rolls.
    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
    replaced = 0
    for (part, kind), items in raw.items():
        items.sort(key=lambda it: it[0])
        for e in (3, 4):
            vals = [it[e] for it in items]
            med = [median(vals[max(0, j - 2):j + 3]) for j in range(len(vals))]
            span = [abs(it[4] - it[3]) for it in items]
            fixed = []
            for j, x in enumerate(vals):
                if abs(x - med[j]) > max(4.0, 0.06 * span[j]):
                    fixed.append(med[j])
                    replaced += 1
                else:
                    fixed.append(x)
            mean = [sum(fixed[max(0, j - 1):j + 2]) / len(fixed[max(0, j - 1):j + 2]) for j in range(len(fixed))]
            items[:] = [it[:e] + (mean[j],) + it[e + 1:] for j, it in enumerate(items)]
        # A row far wider than its part's median is the part merged with
        # another (an arm with the chest, the legs together): dropped.
        typical = median([abs(it[4] - it[3]) for it in items])
        dropped = [it for it in items if abs(it[4] - it[3]) > 1.6 * typical]
        report["stations"].setdefault("dropped", {})[f"{part}_{kind}"] = len(dropped)
        items = [it for it in items if abs(it[4] - it[3]) <= 1.6 * typical]
        for i, view, geom, a, b in items:
            if geom[0] == "row":
                p0, p1 = [round(a, 1), geom[1]], [round(b, 1), geom[1]]
            else:
                _, cx, cy, nx, ny = geom
                p0 = [round(cx + nx * a, 1), round(cy + ny * a, 1)]
                p1 = [round(cx + nx * b, 1), round(cy + ny * b, 1)]
            out[f"outline_{part}{i:02d}_{kind}1"], out[f"outline_{part}{i:02d}_{kind}2"] = (view, p0), (view, p1)
    report["stations"]["smoothing_replaced"] = replaced
    cameras = json.loads((folder / "views.json").read_text())["cameras"]
    front_camera = next(c for c in cameras if c["id"] == front)
    report["front_heights"] = mh_heights.measure(fmask, fm, front_camera["meters_per_pixel"])
    report["proportions"] = measure_proportions(fm, fmask, out)
    report["stations"]["rows"] = counts
    report["stations"]["pairs"] = sum(1 for k in out if k.endswith("1"))
    # Skull: the bald head above the ears. From the neck mark up, the run
    # through the neck's column, to the crown (last row with head); the
    # skull row is a quarter of the way from the crown to the neck base
    # (the ears start lower, about two fifths of the way).
    neck = fm.get("neck")
    if neck:
        nx, ny = int(round(neck[0])), int(round(neck[1]))
        y, crown = ny, None
        while y > 0 and run_at(fmask[y - 1], nx) is not None:
            y -= 1
        crown = y
        head = ny - crown
        if 0.08 * H < head < 0.5 * H:
            sy = int(round(crown + 0.25 * head))
            r = run_at(fmask[sy], nx)
            if r is not None and r[1] - r[0] < 0.9 * head:
                out["outline_skull_w1"], out["outline_skull_w2"] = (front, [r[0], sy]), (front, [r[1], sy])
                report["stations"]["skull_w_px"] = r[1] - r[0]
                rs = runs(smask[sy])
                # The side view's head: the run nearest the side figure's
                # neck column (the topmost rows' centre).
                top = int(np.flatnonzero(smask.any(axis=1)).min())
                cols = np.flatnonzero(smask[top + 3]).tolist() or [W // 2]
                cx = sum(cols) / len(cols)
                near = [q for q in rs if q[0] - 40 <= cx <= q[1] + 40]
                if near:
                    a, b = max(near, key=lambda q: q[1] - q[0])
                    if b - a < 1.2 * head:
                        out["outline_skull_d1"], out["outline_skull_d2"] = (side, [a, sy]), (side, [b, sy])
                        report["stations"]["skull_d_px"] = b - a
            report["stations"]["crown_y"] = crown
            report["stations"]["head_px"] = head
            # A human skull is longer than wide (depth / width 1.15-1.35);
            # outside 1.0-1.6 the run is ears, fur or a muzzle (the fox's
            # 126 px wide against 116 deep): no skull marks.
            w = report["stations"].get("skull_w_px")
            d = report["stations"].get("skull_d_px")
            if w and (not d or not 1.0 <= d / w <= 1.6):
                for k in ("w1", "w2", "d1", "d2"):
                    out.pop(f"outline_skull_{k}", None)
                report["checks"]["skull_rejected"] = [w, d]
    report["marks"] = {k: v for k, v in out.items()}
    (folder / "views" / "bodysuit.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("accepted", "checks", "stations")}, indent=1))
    if do_write and report["accepted"]:
        backup = folder / "guidance.before-bodysuit.json"
        if not backup.exists():
            backup.write_text(json.dumps(g, indent=1))
        g["landmarks"] = {k: v for k, v in g["landmarks"].items() if not k.startswith("outline_")}
        for name, (view, px) in out.items():
            g["landmarks"][name] = [{"view": view, "px": px, "confidence": 0.8}]
        # The body's proportions measured on the suit decide how far the
        # fit may leave MakeHuman's shapes, not the vision agent's guess.
        verdict = report["proportions"].get("verdict")
        if verdict:
            g["subject"]["proportions"] = verdict
        (folder / "guidance.json").write_text(json.dumps(g, indent=1))
        print("wrote", len(out), "outline marks")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0]), "--edit" in args or "--refresh" in args, "--write" in args, "--back" in args, "--refresh" in args)
