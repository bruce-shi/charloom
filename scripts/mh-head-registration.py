"""Local, skin-anchored registration of already generated bald-suit heads.

The scan segmentation is rasterized with every surface occluding. Only visible
skin silhouette segments and textured patches strictly inside visible skin can
anchor a similarity. Hair silhouettes never enter the registration objective.
Body contours are retained verbatim; head consumers explicitly read head_views.
"""
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

RESIDUAL_P95_PX = 3.0
ROTATION_DEG = 5.0
HELD_OUT = "e3"


def glb_meshes(path):
    data = Path(path).read_bytes()
    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + length])
    start = 28 + length
    def accessor(i):
        a = doc["accessors"][i]
        b = doc["bufferViews"][a["bufferView"]]
        dtype = np.dtype({5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[a["componentType"]])
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        return np.ndarray((a["count"], width), dtype=dtype, buffer=data,
                          offset=start + b.get("byteOffset", 0) + a.get("byteOffset", 0),
                          strides=(b.get("byteStride", width * dtype.itemsize), dtype.itemsize)).copy()
    # These exported diagnostic GLBs have world-space geometry and identity nodes.
    for node in doc.get("nodes", []):
        if "mesh" in node:
            assert node.get("translation", [0, 0, 0]) == [0, 0, 0]
            assert node.get("scale", [1, 1, 1]) == [1, 1, 1]
            assert node.get("rotation", [0, 0, 0, 1]) == [0, 0, 0, 1]
            assert "matrix" not in node or np.allclose(np.array(node["matrix"]).reshape(4, 4), np.eye(4))
    meshes = [({k: accessor(v) for k, v in p["attributes"].items()}, accessor(p["indices"]).reshape(-1, 3))
              for m in doc["meshes"] for p in m["primitives"]]
    return doc, meshes


def source_context(folder, source_run):
    report = json.loads((source_run / "report.json").read_text())
    matrix = np.array(report["working_frame"]["source_to_working"]).reshape(4, 4, order="F")
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    cameras = {}
    for c0 in json.loads((folder / "views.json").read_text())["cameras"]:
        c = c0.copy()
        c["center"] = matrix[:3, :3] @ c["center"] + matrix[:3, 3]
        for key in ("right", "up", "forward"):
            c[key] = matrix[:3, :3] @ c[key] / scale
        c["meters_per_pixel"] *= scale
        cameras[c["id"]] = c
    _, scan = glb_meshes(source_run / "1-segment/segmentation.glb")
    positions = np.concatenate([a["POSITION"] for a, _ in scan]).astype(float)
    offsets = np.cumsum([0] + [len(a["POSITION"]) for a, _ in scan])
    triangles = np.concatenate([t + offsets[i] for i, (_, t) in enumerate(scan)])
    colors = np.concatenate([a["COLOR_0"] for a, _ in scan])[:, :3]
    skin = np.all(abs(colors - [.95, .55, .35]) < .01, axis=1)
    labels = np.where(skin[triangles].sum(1) >= 2, 1, 2).astype(np.uint8)
    doc, prior = glb_meshes(Path(report["prior"]))
    attrs, _ = max(prior, key=lambda m: len(m[0]["POSITION"]))
    names = np.array([doc["nodes"][j].get("name", "").lower() for j in doc["skins"][0]["joints"]])
    j, w = attrs["JOINTS_0"], attrs["WEIGHTS_0"]
    neck = (w * np.char.startswith(names[j], "neck")).sum(1) > .5
    head = (w * (names[j] == "head")).sum(1) > .5
    _, body = glb_meshes(source_run / "2-body-rig/fit-diagnostic.glb")
    body = body[0][0]["POSITION"].astype(float)
    return cameras, positions, triangles, labels, body[head], body[neck]


def project(points, c):
    d = points - c["center"]
    return np.column_stack((d @ c["right"] / c["meters_per_pixel"] + c["width"] / 2,
                            c["height"] / 2 - d @ c["up"] / c["meters_per_pixel"]))


def skin_raster(context, view):
    cams, positions, triangles, labels, head, neck = context
    c = cams[view]
    hp, npix = project(head, c), project(neck, c)
    # The ramp spans the anatomical neck, not the hairline or GPT outline.
    ramp_top = float(np.percentile(npix[:, 1], 20))
    ramp_bottom = float(np.percentile(npix[:, 1], 90))
    ramp_bottom = max(ramp_top + 8, ramp_bottom)
    lo = np.floor(hp.min(0) - [35, 40]).astype(int)
    hi = np.ceil(hp.max(0) + [35, 25]).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(np.maximum(hi, [0, ramp_bottom + 3]), [c["width"], c["height"]]).astype(int)
    uv = project(positions, c)
    z = (positions - c["center"]) @ c["forward"]
    buf = np.full((hi[1] - lo[1], hi[0] - lo[0]), np.inf)
    out = np.zeros_like(buf, dtype=np.uint8)
    projected_tri = uv[triangles]
    selected = np.all(projected_tri.max(1) >= lo, axis=1) & np.all(projected_tri.min(1) < hi, axis=1)
    for k in np.flatnonzero(selected):
        t = triangles[k]
        p = uv[t] - lo
        low = np.maximum(np.floor(p.min(0)).astype(int), 0)
        high = np.minimum(np.ceil(p.max(0)).astype(int) + 1, out.shape[::-1])
        if np.any(low >= high):
            continue
        ax, ay = p[0] - p[2]
        bx, by = p[1] - p[2]
        det = ax * by - ay * bx
        if abs(det) < 1e-10:
            continue
        yy, xx = np.mgrid[low[1]:high[1], low[0]:high[0]]
        dx, dy = xx + .5 - p[2, 0], yy + .5 - p[2, 1]
        a, b = (dx * by - dy * bx) / det, (ax * dy - ay * dx) / det
        depth = a * z[t[0]] + b * z[t[1]] + (1 - a - b) * z[t[2]]
        target = buf[low[1]:high[1], low[0]:high[0]]
        keep = (a >= 0) & (b >= 0) & (a + b <= 1) & (depth < target)
        target[keep] = depth[keep]
        out[low[1]:high[1], low[0]:high[0]][keep] = labels[k]
    full = np.zeros((c["height"], c["width"]), np.uint8)
    full[lo[1]:hi[1], lo[0]:hi[0]] = out
    return full, (lo, hi), (ramp_top, ramp_bottom)


def affine(params, center):
    scale, angle, tx, ty = params
    co, si = np.cos(angle) * scale, np.sin(angle) * scale
    mat = np.array([[co, -si], [si, co]])
    return np.column_stack((mat, center + [tx, ty] - mat @ center))


def transform(points, mat):
    return points @ mat[:, :2].T + mat[:, 2]


def skin_anchors(orig, edit, labels, ramp):
    skin = labels == 1
    # Only skin adjacent to the true exterior/background is an invariant outline.
    # Skin/hair occlusion boundaries disappear in a bald edit and are excluded.
    exterior = cv2.dilate((labels == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    outline = skin & exterior
    outline[int(ramp[1]):] = False
    y, x = np.nonzero(outline)
    points = np.column_stack((x + .5, y + .5))
    interior = cv2.erode(skin.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    interior[int(ramp[0]):] = False
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 24, .03, 9, mask=interior.astype(np.uint8))
    templates = []
    if corners is not None:
        for x, y in corners[:, 0].astype(int):
            radius = 4
            patch = gray[y-radius:y+radius+1, x-radius:x+radius+1]
            if patch.shape != (9, 9) or patch.std() < 8:
                continue
            lo = np.maximum([x - 26, y - 26], 0)
            hi = np.minimum([x + 27, y + 27], gray.shape[::-1])
            search = cv2.cvtColor(edit[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_BGR2GRAY)
            corr = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(corr)
            if score < .78:
                continue
            px, py = np.array(loc) + lo + radius
            # Reciprocal match and unique peak keep changed/repainted features out.
            masked = corr.copy()
            cv2.circle(masked, loc, 4, -1, -1)
            if score - masked.max() < .04:
                continue
            eg = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
            back_patch = eg[py-radius:py+radius+1, px-radius:px+radius+1]
            back = cv2.matchTemplate(gray[lo[1]:hi[1], lo[0]:hi[0]], back_patch, cv2.TM_CCOEFF_NORMED)
            _, _, _, back_loc = cv2.minMaxLoc(back)
            if np.linalg.norm(np.array(back_loc) + lo + radius - [x, y]) > 2.5:
                continue
            templates.append(([float(x), float(y)], [float(px), float(py)], float(score)))
    return points, templates


def register(orig, edit, labels, box, ramp, figure):
    points, templates = skin_anchors(orig, edit, labels, ramp)
    lo, hi = box
    center = (lo + hi) / 2
    mask = figure(edit).astype(np.uint8)
    edge = mask.astype(bool) & ~(cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0)
    edge[:lo[1]] = False
    edge[int(ramp[1] + 25):] = False
    edge[:, :lo[0]] = False
    edge[:, hi[0]:] = False
    y, x = np.nonzero(edge)
    ep = np.column_stack((x + .5, y + .5))
    info = {"skin_outline_points": len(points), "texture_matches": len(templates),
            "residual_limit_px_p95": RESIDUAL_P95_PX, "rotation_limit_deg": ROTATION_DEG,
            "ramp_y_px": list(ramp), "crop": [*lo.tolist(), *hi.tolist()],
            "anchor_source": "z-buffered scan skin exterior + eroded-skin reciprocal texture patches"}
    info["minimum_correction_prior"] = {"scale_sigma": .05, "scale_weight": 2.0,
                                        "rotation_sigma_deg": 5.0,
                                        "reference": "existing body transform; no skull or hair anchors"}
    if len(points) < 8 and len(templates) < 3 or len(ep) < 8:
        return edit, {**info, "accepted": False, "reasons": ["insufficient_visible_skin_anchors"]}, None
    tree = cKDTree(ep)
    distance_field = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    src = np.array([t[0] for t in templates]).reshape(-1, 2)
    dst = np.array([t[1] for t in templates]).reshape(-1, 2)
    def residual(p):
        mat = affine(p, center)
        inverse = cv2.invertAffineTransform(mat)
        q = transform(points, inverse)
        # A bilinear chamfer field avoids the discontinuous tangential ICP
        # correspondences at equally near pixel centers.
        q = q - .5
        x = np.clip(q[:, 0], 0, edge.shape[1] - 1.001)
        y = np.clip(q[:, 1], 0, edge.shape[0] - 1.001)
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x-ix, y-iy
        outline = ((1-fx)*(1-fy)*distance_field[iy,ix] + fx*(1-fy)*distance_field[iy,ix+1]
                   + (1-fx)*fy*distance_field[iy+1,ix] + fx*fy*distance_field[iy+1,ix+1]) * p[0]
        texture = (transform(dst, mat) - src).ravel()
        # Match influence is comparable to 4 outline samples, never the skull.
        return np.r_[outline, texture * 2, 2 * (p[0]-1)/.05, p[1]/np.deg2rad(5)]
    def robust_loss(z):
        root = np.sqrt(1 + z)
        rho = np.vstack((2 * (root - 1), 1/root, -.5/root**3))
        # Keep the weak minimum-correction prior quadratic; it resolves
        # unobservable scale in short/occluded skin segments, without using hair.
        rho[0, -2:] = z[-2:]
        rho[1, -2:] = 1
        rho[2, -2:] = 0
        return rho
    initial = np.array([1., 0., 0., 0.])
    if len(templates):
        initial[2:] = np.median(src - dst, axis=0)
    initial[2:] = np.clip(initial[2:], -25, 25)
    result = least_squares(residual, initial, bounds=([.8, -np.deg2rad(5), -35, -35],
                                                    [1.25, np.deg2rad(5), 35, 35]),
                           loss=robust_loss, f_scale=1.5, max_nfev=180,
                           x_scale=[.05, .05, 10., 10.], diff_step=1e-4, xtol=1e-8, ftol=1e-8, gtol=1e-8)
    mat = affine(result.x, center)
    q = transform(points, cv2.invertAffineTransform(mat))
    distances = tree.query(q)[0] * result.x[0]
    td = np.linalg.norm(transform(dst, mat) - src, axis=1)
    p95 = float(np.percentile(distances, 95)) if len(distances) else None
    before_p95 = float(np.percentile(tree.query(points)[0], 95)) if len(points) else None
    texture_p95 = float(np.percentile(td, 95)) if len(td) else None
    reasons = []
    if p95 is not None and p95 > RESIDUAL_P95_PX:
        reasons.append("visible_skin_outline_p95_above_3px")
    if len(points) < 8 and len(td) >= 3 and texture_p95 > RESIDUAL_P95_PX:
        reasons.append("visible_skin_texture_p95_above_3px")
    yy, xx = np.indices(labels.shape, dtype=np.float32)
    inv = cv2.invertAffineTransform(mat)
    local_x = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
    local_y = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
    t = np.clip((ramp[1] - yy) / (ramp[1] - ramp[0]), 0, 1)
    t = t * t * (3 - 2 * t)
    out = cv2.remap(edit, (xx + t * (local_x - xx)).astype(np.float32), (yy + t * (local_y - yy)).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    assert np.array_equal(out[int(np.ceil(ramp[1])):], edit[int(np.ceil(ramp[1])):])
    return out, {**info, "accepted": not reasons, "reasons": reasons,
                 "similarity": {"scale": float(result.x[0]), "rotation_deg": float(np.rad2deg(result.x[1])),
                                "translation_px": result.x[2:].tolist(), "matrix": mat.tolist()},
                 "skin_outline_p95_px": p95, "skin_outline_median_px": float(np.median(distances)) if len(distances) else None,
                 "old_skin_outline_p95_px": before_p95,
                 "texture_p95_px": texture_p95, "texture_matches_detail": templates,
                 "texture_residual_warning": bool(len(td) and texture_p95 > RESIDUAL_P95_PX),
                 "body_pixels_unchanged_below_y": int(np.ceil(ramp[1]))}, points


def curves_for(mask, contours_mod):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    curves = []
    for c in cs:
        pts = c[:, 0].astype(float)
        if len(pts) < contours_mod.MIN_POINTS:
            continue
        sm = contours_mod.smooth_closed(pts)
        if sm is not None:
            curves.append(np.round(sm, 2).tolist())
    return curves


def run(folder, source_run, review_dir, ensemble):
    folder, source_run, review_dir = map(Path, (folder, source_run, review_dir))
    review_dir.mkdir(parents=True, exist_ok=True)
    context = source_context(folder, source_run)
    g = json.loads((folder / "guidance.json").read_text())
    views = ensemble.views_of(g)
    directory = folder / "views/suit-ensemble"
    report = json.loads((directory / "ensemble.json").read_text())
    # Keep the frozen body registration diagnostics independently of reruns.
    report["head_registration"] = {"method": "visible_skin_similarity_with_neck_ramp", "held_out_member": HELD_OUT,
        "residual_threshold_px_p95": RESIDUAL_P95_PX, "minimum_head_iou": ensemble.MIN_IOU,
        "source_run": str(source_run), "members": {}, "views": {},
        "rear_panel": "absent in saved GPT sheets; no synthetic rear evidence"}
    hr = report["head_registration"]
    head_views, held_views, head_weights = {}, {}, {}
    sheet_rows = []
    for v in views:
        labels, box, ramp = skin_raster(context, v)
        lo, hi = box
        orig = cv2.imread(str(folder / f"views/{v}.png"))
        cv2.imwrite(str(review_dir / f"{v}-scan-skin.png"), (labels == 1).astype(np.uint8) * 255)
        members = {p.name.split("_")[0]: p for p in sorted(directory.glob(f"e*_{v}-reg.png"))}
        single = folder / f"views/bodysuit_{v}-reg.png"
        if single.exists():
            members["single"] = single
        registered = {}
        pictures = []
        for name, path in members.items():
            old = cv2.imread(str(path))
            new, info, anchors = register(orig, old, labels, box, ramp, ensemble.figure)
            hr["members"].setdefault(name, {})[v] = info
            cv2.imwrite(str(directory / f"{name}_{v}-head-reg.png"), new)
            mask = ensemble.figure(new)
            registered[name] = mask
            if name == HELD_OUT:
                held_views[v] = curves_for(mask, ensemble.contours_mod)
            panels = []
            for title, img in [("old body registration", old), ("head-local registration", new)]:
                over = orig.copy()
                edges, _ = cv2.findContours(ensemble.figure(img).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(over, edges, -1, (220, 30, 220), 1)
                if anchors is not None:
                    for x, y in anchors:
                        cv2.circle(over, (round(x), round(y)), 1, (50, 215, 30), -1)
                crop = over[lo[1]:hi[1], lo[0]:hi[0]]
                crop = cv2.resize(crop, (350, round(crop.shape[0] * 350 / crop.shape[1])))
                panel = np.full((crop.shape[0] + 62, 350, 3), 255, np.uint8)
                panel[62:] = crop
                cv2.putText(panel, f"{name} {v}: {title}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .39, (0, 0, 0), 1)
                residual = info.get("old_skin_outline_p95_px" if title.startswith("old") else "skin_outline_p95_px")
                label = f"skin p95 {residual:.2f}px" if residual is not None else "no outline support"
                cv2.putText(panel, label + (" HELD OUT" if name == HELD_OUT else ""), (5, 38), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1)
                panels.append(panel)
            pictures.append(np.hstack(panels))
        candidates = {n: m for n, m in registered.items() if n != HELD_OUT and hr["members"][n][v]["accepted"]}
        # Screen simultaneously; the held-out member cannot influence rejection.
        dropped = []
        if len(candidates) >= 2:
            for name, mask in candidates.items():
                consensus = np.mean([ensemble.sdf(m) for n, m in candidates.items() if n != name], axis=0) > 0
                crop = np.s_[lo[1]:int(ramp[0]), lo[0]:hi[0]]
                score = ensemble.iou(mask[crop], consensus[crop])
                hr["members"][name][v]["head_iou_other_training_members"] = score
                if score < ensemble.MIN_IOU:
                    dropped.append(name)
        for name in dropped:
            hr["members"][name][v]["accepted"] = False
            hr["members"][name][v]["reasons"].append("head_crop_iou_below_MIN_IOU")
            del candidates[name]
        hr["views"][v] = {"accepted_training_members": list(candidates), "held_out_member": HELD_OUT,
                            "missing_consensus": not bool(candidates), "ramp_y_px": list(ramp)}
        if candidates:
            fields = np.stack([ensemble.sdf(m) for m in candidates.values()])
            consensus = fields.mean(0) > 0
            curves = curves_for(consensus, ensemble.contours_mod)
            head_views[v] = curves
            # Confidence depends on member agreement only; scan hair never weakens a GPT target.
            spread = fields.std(0)
            head_weights[v] = [np.round(1 / (1 + (ensemble.sample(spread, np.array(c)) / ensemble.SPREAD_PX)**2), 3).tolist() for c in curves]
            if HELD_OUT in registered:
                crop = np.s_[lo[1]:int(ramp[0]), lo[0]:hi[0]]
                hr["members"][HELD_OUT][v]["head_iou_training_consensus"] = ensemble.iou(registered[HELD_OUT][crop], consensus[crop])
        if pictures:
            row = np.vstack(pictures)
            cv2.imwrite(str(review_dir / f"{v}-registration.png"), row)
            sheet_rows.append(row)
    for file in ("suit_contours.ensemble.json", "suit_contours.ensemble-all.json"):
        path = folder / "views" / file
        data = json.loads(path.read_text())
        keep = views if "-all" in file else views[:2]
        data["head_views"] = {v: c for v, c in head_views.items() if v in keep}
        data["head_weights"] = {v: w for v, w in head_weights.items() if v in keep}
        data["head_registration"] = hr
        data["held_out_head_views"] = {v: c for v, c in held_views.items() if v in keep}
        path.write_text(json.dumps(data) + "\n")
    (directory / "ensemble.json").write_text(json.dumps(report, indent=2) + "\n")
    (review_dir / "registration.json").write_text(json.dumps(hr, indent=2) + "\n")
    if sheet_rows:
        height = max(i.shape[0] for i in sheet_rows)
        sheet = np.full((height, 700 * len(sheet_rows), 3), 255, np.uint8)
        for k, row in enumerate(sheet_rows):
            sheet[:row.shape[0], k*700:(k+1)*700] = row
        cv2.imwrite(str(review_dir / "before-after.png"), sheet)
    return hr
