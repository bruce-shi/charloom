"""Bodysuit ensemble: several suit edits of the front, side and 3/4 views,
one consensus silhouette per view, each point weighted by how far the
edits agree there.

    out/tools/cv/.venv/bin/python scripts/mh-suit-ensemble.py out/examples/mh-guide-NAME [...] [--runs 3] [--edit]

`scripts/mh-bodysuit.py` has the image model paint ONE edit of the front
and side views, and the fit trusts its silhouette everywhere: where the
model guessed (under a skirt, a robe, a brute's belly) the body followed
the guess. This asks for RUNS edits of a 2x2 sheet (front, side and the
two front 3/4 views, `charloom mh-guide-views --yaw`, rendered when
missing), registers every panel onto its view as mh-bodysuit.py does
(the front by the face and hands, the others by the front's transform,
checked and if need be placed by the feet), and per view averages the
members' signed distance fields (the existing front/side edit counts as one
more member). The consensus outline's points get a weight in 0-1:

  * agreement: 1 / (1 + (spread / SPREAD_PX)^2), the spread the standard
    deviation of the members' signed distances at the point;
  * the clothes: a point more than OUTSIDE_PX outside the scan's own
    figure in that view (the body is inside the clothes) is cut to
    OUTSIDE_WEIGHT.

Each view's weights are then divided by their median (capped at 1), so the
pull keeps the single edit's strength overall and only the points the
edits disagree on more than usual are weakened: absolute weights cut the
whole pull where every edit wobbles a little (the brute, mean 0.4) and its
stretch rose 14.5 -> 22 (trial x102, 2026-11-08).

Gate: when the edits disagree by more than GATE_PX (median spread, front
and side views averaged) the image model is guessing everywhere, and the
ensemble's weak, uneven pull did more harm than the one edit: the brute
(4.1 px) and the troll (3.2 px, green skin under a green suit) got 7 % and
11 % left/right limb differences in the rig, while the other 14 characters
(at most 2.4 px) kept theirs and gained in garment fit (worst garment at
rest, sum over 16: 103 -> 56 %; trial x103, 2026-11-08). Past the gate the
file holds the single edit's contours (`suit_contours.json`, unweighted).

Writes views/suit_contours.ensemble.json, the front and side views (the
fit reads it with config `suit_contours_file`), and
suit_contours.ensemble-all.json with the 3/4 views too: in the trial they
helped no character consistently (brute, herbalist, kr-swordswoman,
stocky-woman), so they are not the default. Also the members and overlays in
views/suit-ensemble/ (members thin grey, consensus blue where trusted, red
where not), and views/suit-ensemble/ensemble.json with the checks.

  --runs N   edits per sheet request (default 3)
  --edit     reuse a matching cached sheet or request it once
  --refresh  explicitly request new measurement sheets
  --dry      no image model call even without sheets (the single edit alone)
  --mirror   front-view weights the lower of each point's and its mirror
             partner's (tried in x103: it cut the garment gains and barely
             helped the rig's limb symmetry; off by default)
  --name S   write suit_contours.S.json / S-all.json (default `ensemble`)
  --head-source RUN --head-review DIR
             register existing members' heads against RUN's visible scan skin;
             write head_views beside unchanged body views. No image API calls.
  --head-back --head-source RUN
             register existing suit-back sheets, tying scale to each same-sheet
             profile. Write only new views/suit-back/registration/ files.
"""
from measurement_images import sheets
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
mhb = import_module("mh-bodysuit")
contours_mod = import_module("mh-suit-contours")

ROOT = Path(__file__).resolve().parents[1]
GUIDE_TOOL = ROOT / "scripts/scan-guide.py"
W, H = 768, 1024
SHEET_W, SHEET_H = 2 * W, 2 * H
SPREAD_PX = 3.0
OUTSIDE_PX = 3.0
OUTSIDE_WEIGHT = 0.15
# A member whose figure overlaps the others' consensus less than this is
# dropped (a turned, rescaled or redrawn panel).
MIN_IOU = 0.88
MIRROR_PX = 6.0
GATE_PX = 3.0
GRID = (
    "This sheet shows the SAME 3D character in 4 panels in a 2x2 grid: the front view "
    "(top left), the side view (profile, top right), and two three-quarter views "
    "(bottom left and bottom right). Edit every panel the same way so they show one "
    "consistent body seen from four directions. "
)


def views_of(g):
    yaw = int(g["front_view"].split("_y")[1])
    f = lambda d: f"body_y{(yaw + d) % 360:03d}"
    # Grid order: top left, top right, bottom left, bottom right.
    return [f(0), f(90), f(45), f(315)]


def ensure_views(folder: Path, views):
    missing = [v for v in views if not (folder / "views" / f"{v}.png").exists()]
    if missing:
        yaws = [str(int(v.split("_y")[1])) for v in missing]
        source = json.loads((folder / "views.json").read_text())["source"]
        subprocess.run([sys.executable, str(GUIDE_TOOL), source, "--views", str(folder / "guidance.json"), "--yaw", *yaws],
                       check=True, capture_output=True)


def edit_grid(folder: Path, views, runs: int, out_dir: Path, refresh=False):
    prompt = GRID + mhb.PROMPT.replace("this 3D character render", "the sheet")
    tiles = [str(folder / "views" / f"{v}.png") for v in views]
    sheet_in = out_dir / "input_sheet.png"
    subprocess.run(["magick", "(", tiles[0], tiles[1], "+append", ")", "(", tiles[2], tiles[3], "+append", ")",
                    "-append", "-resize", f"{SHEET_W}x{SHEET_H}!", str(sheet_in)], check=True)
    images, receipt = sheets(folder, sheet_in, dict(model=mhb.MODEL, prompt=prompt, size=f"{SHEET_W}x{SHEET_H}", quality="medium", n=str(runs)), refresh)
    for k, raw in enumerate(images, 1):
        (out_dir / f"e{k}_sheet.png").write_bytes(raw)
    for old in out_dir.glob("e*_sheet.png"):
        if int(old.stem[1:].split("_")[0]) > len(images):
            old.unlink()
    return {**receipt, "members": len(images)}


def split(sheet: Path, views, out_dir: Path, k: int):
    img = cv2.imread(str(sheet), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SHEET_W, SHEET_H), interpolation=cv2.INTER_AREA)
    paths = {}
    for i, v in enumerate(views):
        x, y = (i % 2) * W, (i // 2) * H
        p = out_dir / f"e{k}_{v}.png"
        cv2.imwrite(str(p), img[y:y + H, x:x + W])
        paths[v] = p
    return paths


def bg_of(img):
    c = np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], dtype=float)
    return np.median(c, axis=0)


def figure(img):
    m = (np.abs(img.astype(float) - bg_of(img)).sum(axis=2) > 40)
    return mhb.clean(m)


def match(orig, edit, cx, cy, w, h, r):
    """Patch (w x h) of the original about (cx, cy) found in the edit within
    r px: (original centre, edit centre, normalised RMSE)."""
    tx, ty = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    t = orig[ty:ty + h, tx:tx + w].astype(np.float32)
    sx, sy = max(0, tx - r), max(0, ty - r)
    s = edit[sy:ty + h + r, sx:tx + w + r].astype(np.float32)
    if t.size == 0 or s.shape[0] < t.shape[0] or s.shape[1] < t.shape[1]:
        return None
    res = cv2.matchTemplate(s, t, cv2.TM_SQDIFF)
    lo, _, loc, _ = cv2.minMaxLoc(res)
    rmse = float(np.sqrt(lo / t.size)) / 255.0
    ex, ey = sx + loc[0], sy + loc[1]
    th, tw = t.shape[:2]
    return ((tx + tw / 2, ty + th / 2), (ex + tw / 2, ey + th / 2), rmse)


def similarity(g, front, orig, edit, cameras=None):
    """Scale and the two midpoints (edit, original) mapping the edit onto
    the view: the face and the backs of the hands (mh-bodysuit.py)."""
    cameras = cameras or {}
    anchors = {name: mhb.match_patch(orig, edit, *patch)
               for name, patch in mhb.registration_patches(g, cameras, front).items()}
    good = [v for k, v in anchors.items() if v is not None and (k == "face" or v[2] < 0.08)]
    if not good:
        return None
    to, te = [a[0] for a in good], [a[1] for a in good]
    mo = np.mean(to, axis=0)
    me = np.mean(te, axis=0)
    if len(good) >= 2:
        num = sum(np.dot(np.subtract(o, mo), np.subtract(e, me)) for o, e in zip(to, te))
        den = sum(np.dot(np.subtract(e, me), np.subtract(e, me)) for e in te)
        scale = float(num / max(1e-6, den))
    else:
        scale = 1.0
    return scale, me, mo


def warp(img, scale, me, mo):
    m = np.array([[scale, 0, mo[0] - scale * me[0]], [0, scale, mo[1] - scale * me[1]]], dtype=np.float32)
    bg = tuple(float(c) for c in bg_of(img))
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


def feet(m):
    rows = np.flatnonzero(m.any(axis=1))
    bottom = int(rows.max())
    xs = np.flatnonzero(m[bottom - 30:bottom + 1].any(axis=0))
    return np.array([float(xs.mean()), float(bottom)])


def register_member(g, views, panels, origs, omasks, cameras=None):
    """Registered images per view, and the checks; None when the front
    cannot be registered or is rescaled past 15 %."""
    front = views[0]
    edits = {v: cv2.imread(str(p), cv2.IMREAD_COLOR) for v, p in panels.items()}
    sim = similarity(g, front, origs[front], edits[front], cameras)
    if sim is None:
        return None, {"rejected": "nothing matched"}
    scale, me, mo = sim
    checks = {"scale": round(scale, 4), "shift_px": [round(float(mo[0] - me[0]), 1), round(float(mo[1] - me[1]), 1)]}
    if not 0.85 < scale < 1.15:
        return None, {**checks, "rejected": "scale"}
    out = {front: warp(edits[front], scale, me, mo)}
    for v in views[1:]:
        reg = warp(edits[v], scale, me, mo)
        fo, fr = feet(omasks[v]), feet(figure(reg))
        off = fr - fo
        checks[f"{v}_feet_px"] = [round(float(off[0]), 1), round(float(off[1]), 1)]
        if np.abs(off).max() > 10:
            checks[f"{v}_placed_by"] = "feet"
            reg = warp(edits[v], scale, feet(figure(edits[v])), fo)
        out[v] = reg
    return out, checks


def sdf(m):
    """Signed distance to the mask's edge, px: positive inside."""
    u = m.astype(np.uint8)
    inside = cv2.distanceTransform(u, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - u, cv2.DIST_L2, 5)
    return inside - outside


def sample(field, pts):
    """Bilinear samples of a field at (x, y) points."""
    x = np.clip(pts[:, 0], 0, field.shape[1] - 1.001)
    y = np.clip(pts[:, 1], 0, field.shape[0] - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    f = field
    return ((1 - fx) * (1 - fy) * f[y0, x0] + fx * (1 - fy) * f[y0, x0 + 1]
            + (1 - fx) * fy * f[y0 + 1, x0] + fx * fy * f[y0 + 1, x0 + 1])


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def main(folder: Path, runs: int, do_edit: bool, dry: bool = False, mirror: bool = False, out_name: str = "ensemble", refresh: bool = False):
    g = json.loads((folder / "guidance.json").read_text())
    views = views_of(g)
    ensure_views(folder, views)
    out_dir = folder / "views" / "suit-ensemble"
    out_dir.mkdir(exist_ok=True)
    report = {"views": views, "members": {}, "per_view": {}}
    cameras = {c["id"]: c for c in json.loads((folder / "views.json").read_text())["cameras"]}
    if not do_edit and not dry and not list(out_dir.glob("e*_sheet.png")):
        raise FileNotFoundError("Missing ensemble members; use --edit to generate or --dry for the existing single edit")
    if do_edit:
        report["edit"] = edit_grid(folder, views, runs, out_dir, refresh)
    origs = {v: cv2.imread(str(folder / "views" / f"{v}.png"), cv2.IMREAD_COLOR) for v in views}
    omasks = {v: mhb.calibrated_mask(origs[v]) for v in views}
    # Members: the grid edits, and the single front/side edit already there.
    members = {}
    for sheet in sorted(out_dir.glob("e*_sheet.png")):
        k = int(sheet.stem[1:].split("_")[0])
        reg, checks = register_member(g, views, split(sheet, views, out_dir, k), origs, omasks, cameras)
        report["members"][f"e{k}"] = checks
        if reg is not None:
            for v, img in reg.items():
                cv2.imwrite(str(out_dir / f"e{k}_{v}-reg.png"), img)
            members[f"e{k}"] = {v: figure(img) for v, img in reg.items()}
    single = {}
    for v in views[:2]:
        p = folder / "views" / f"bodysuit_{v}-reg.png"
        if p.exists():
            single[v] = figure(mhb.image(p))
    if single:
        members["single"] = single
        report["members"]["single"] = {"views": list(single)}
    curves_out, weights_out = {}, {}
    overlays = []
    for v in views:
        masks = {name: m[v] for name, m in members.items() if v in m}
        # Outliers: a member far off the others' mean (a redrawn panel).
        if len(masks) >= 3:
            for name in list(masks):
                rest = [sdf(m) for n2, m in masks.items() if n2 != name]
                cons = np.mean(rest, axis=0) > 0
                ov = iou(masks[name], cons)
                if ov < MIN_IOU:
                    report["members"].setdefault(name, {}).setdefault("dropped_views", {})[v] = round(ov, 3)
                    del masks[name]
        if not masks:
            continue
        fields = np.stack([sdf(m) for m in masks.values()])
        mean = fields.mean(axis=0)
        cons = mhb.clean(mean > 0)
        clothes = sdf(omasks[v])
        cs, _ = cv2.findContours(cons.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        curves, weights, spreads, outside = [], [], [], 0
        for c in cs:
            pts = c[:, 0, :].astype(float)
            if len(pts) < contours_mod.MIN_POINTS:
                continue
            sm = contours_mod.smooth_closed(pts)
            if sm is None:
                continue
            spread = sample(fields.std(axis=0), sm) if len(masks) > 1 else np.zeros(len(sm))
            w = 1.0 / (1.0 + (spread / SPREAD_PX) ** 2)
            out_of_clothes = sample(clothes, sm) < -OUTSIDE_PX
            w[out_of_clothes] = np.minimum(w[out_of_clothes], OUTSIDE_WEIGHT)
            outside += int(out_of_clothes.sum())
            curves.append(np.round(sm, 2).tolist())
            weights.append(np.round(w, 3).tolist())
            spreads.extend(spread.tolist())
        curves_out[v], weights_out[v] = curves, weights
        allw = np.concatenate([np.array(w) for w in weights]) if weights else np.zeros(0)
        report["per_view"][v] = {
            "members": len(masks),
            "points": int(len(allw)),
            "spread_px_p50": round(float(np.percentile(spreads, 50)), 2) if spreads else None,
            "spread_px_p90": round(float(np.percentile(spreads, 90)), 2) if spreads else None,
            "weight_mean": round(float(allw.mean()), 3) if len(allw) else None,
            "weak_share": round(float((allw < 0.5).mean()), 3) if len(allw) else None,
            "outside_clothes": outside,
        }
        # Overlay: the original view, members grey, consensus by weight.
        img = origs[v].copy()
        for m in masks.values():
            mc, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, mc, -1, (150, 150, 150), 1)
        for curve, w in zip(curves, weights):
            for (x, y), wi in zip(curve, w):
                col = (int(255 * wi), 40, int(255 * (1 - wi)))  # BGR: blue trusted, red weak
                cv2.circle(img, (int(round(x)), int(round(y))), 1, col, -1)
        cv2.putText(img, f"{v}  n={len(masks)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(out_dir / f"overlay_{v}.png"), img)
        overlays.append(img)
    if overlays:
        while len(overlays) < 4:
            overlays.append(np.full_like(overlays[0], 255))
        grid = np.vstack([np.hstack(overlays[:2]), np.hstack(overlays[2:4])])
        cv2.imwrite(str(out_dir / "overlay.png"), grid)
    fm = mhb.marks(g, views[0])
    if mirror and views[0] in curves_out and "hip_l" in fm and "hip_r" in fm:
        cx = (fm["hip_l"][0] + fm["hip_r"][0]) / 2
        pts = np.array([p for c in curves_out[views[0]] for p in c])
        w = np.concatenate([np.array(x) for x in weights_out[views[0]]])
        mirrored = np.stack([2 * cx - pts[:, 0], pts[:, 1]], axis=1)
        tree = cv2.flann_Index(pts.astype(np.float32), {"algorithm": 1, "trees": 4})
        idx, d2 = tree.knnSearch(mirrored.astype(np.float32), 1, params={})
        partner = np.sqrt(d2[:, 0]) <= MIRROR_PX
        w = np.where(partner, np.minimum(w, w[idx[:, 0]]), w)
        report["per_view"][views[0]]["mirror_paired"] = round(float(partner.mean()), 3)
        out, k = [], 0
        for c in weights_out[views[0]]:
            out.append(np.round(w[k:k + len(c)], 3).tolist())
            k += len(c)
        weights_out[views[0]] = out
    for v, ws in weights_out.items():
        med = float(np.median([x for w in ws for x in w])) if ws else 1.0
        weights_out[v] = [[round(min(1.0, x / max(med, 1e-6)), 3) for x in w] for w in ws]
    spreads = [report["per_view"][v]["spread_px_p50"] for v in views[:2]
               if report["per_view"].get(v, {}).get("spread_px_p50") is not None]
    disagreement = float(np.mean(spreads)) if spreads else None
    single_file = folder / "views" / "suit_contours.json"
    gated = disagreement is not None and disagreement > GATE_PX and single_file.exists()
    report["gate"] = {"disagreement_px": None if disagreement is None else round(disagreement, 2),
                      "limit_px": GATE_PX, "fallback_to_single": gated}
    for file, keep in ((f"suit_contours.{out_name}-all.json", views), (f"suit_contours.{out_name}.json", views[:2])):
        if gated and not file.endswith("-all.json"):
            single = json.loads(single_file.read_text())
            single["ensemble"] = {"gate": report["gate"]}
            (folder / "views" / file).write_text(json.dumps(single))
            continue
        (folder / "views" / file).write_text(json.dumps({
            "views": {v: c for v, c in curves_out.items() if v in keep},
            "weights": {v: w for v, w in weights_out.items() if v in keep},
            "sigma_px": contours_mod.SIGMA, "step_px": contours_mod.STEP,
            "ensemble": {v: r for v, r in report["per_view"].items() if v in keep}}))
    (out_dir / "ensemble.json").write_text(json.dumps(report, indent=1))
    print(folder.name, json.dumps({"members": report["members"], "per_view": report["per_view"], "gate": report["gate"]}, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    val = lambda flag, default: args[args.index(flag) + 1] if flag in args else default
    if "--head-back" in args:
        if {"--edit", "--refresh"}.intersection(args) or "--head-source" not in args:
            raise SystemExit("--head-back requires --head-source and forbids --edit")
        result = import_module("mh-back-registration").run(
            Path(args[0]), Path(val("--head-source", "")), sys.modules[__name__])
        print(json.dumps(result), flush=True)
        raise SystemExit(0)
    if "--head-3d-review" in args:
        import_module("mh-head-3d-review").run(Path(val("--head-3d-review", "")), Path(val("--head-review", "out/review/diagnostics/head-3d-registration")))
        raise SystemExit(0)
    if "--head-source" in args:
        if {"--edit", "--refresh"}.intersection(args):
            raise SystemExit("--head-source only reuses local images; --edit is incompatible")
        folder = Path(args[0])
        hr = import_module("mh-head-registration").run(
            folder, Path(val("--head-source", "")), Path(val("--head-review", str(folder / "views/head-registration"))),
            sys.modules[__name__])
        print(json.dumps({"folder": str(folder), "views": hr["views"]}), flush=True)
        raise SystemExit(0)
    runs = int(val("--runs", 3))
    folders = [Path(a) for i, a in enumerate(args)
               if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--runs", "--name"))]
    for f in folders:
        main(f, runs, "--edit" in args or "--refresh" in args, "--dry" in args, "--mirror" in args, val("--name", "ensemble"), "--refresh" in args)
