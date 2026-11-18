"""Second match: the fitted body against the evidence it was fitted to.

    out/tools/cv/.venv/bin/python scripts/mh-match.py RUN_DIR GUIDE_DIR [RUN_DIR GUIDE_DIR ...]

Projects the run's fitted body (`2-body-rig/fit-diagnostic.glb`, else
`3-skin/body.glb`: the posed body in the working frame) through the
guidance cameras (front and side), fills its silhouette, and lays it over
the bodysuit edit and the original view: green = suit only (body too
small), magenta = fitted body only (too big), white = both. Per outline
station it prints the suit's width or depth and the fitted body's along
the same line, and the silhouettes' overlap (IoU). For the final neutral body it also measures
floor-to-surface and floor-to-joint heights in the frontal view, saving
`body-heights.json` and `renders/evidence-6-heights.png`. Writes
`renders/evidence-4-match.jpg`.
"""
import json
import cv2
import mh_heights
import mh_sections
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def glb_meshes(path, joints=None):
    """Every mesh primitive's (positions, triangles), with node transforms."""
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + n])
    off = 20 + n
    bn = struct.unpack_from("<I", data, off)[0]
    blob = data[off + 8:off + 8 + bn]

    def acc(i):
        a = doc["accessors"][i]
        bv = doc["bufferViews"][a["bufferView"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        comps = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[a["type"]]
        dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[a["componentType"]]
        stride = bv.get("byteStride")
        if stride and stride != comps * np.dtype(dt).itemsize:
            raw = np.frombuffer(blob, np.uint8, a["count"] * stride, start).reshape(-1, stride)
            return raw[:, :comps * np.dtype(dt).itemsize].copy().view(dt).reshape(-1, comps)
        return np.frombuffer(blob, dt, a["count"] * comps, start).reshape(-1, comps)

    def mat(node):
        if "matrix" in node:
            return np.array(node["matrix"]).reshape(4, 4).T
        m = np.eye(4)
        t, r, s = node.get("translation", [0, 0, 0]), node.get("rotation", [0, 0, 0, 1]), node.get("scale", [1, 1, 1])
        x, y, z, w = r
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m[:3, :3] = R * np.array(s)
        m[:3, 3] = t
        return m

    joint_ids = {i for skin in doc.get("skins", []) for i in skin["joints"]}
    out = []

    def walk(i, parent):
        node = doc["nodes"][i]
        m = parent @ mat(node)
        if joints is not None and i in joint_ids:
            joints[node.get("name", "")] = m[:3, 3]
        if "mesh" in node:
            for p in doc["meshes"][node["mesh"]]["primitives"]:
                pos = acc(p["attributes"]["POSITION"]).astype(float)
                # Skinned meshes ignore their node transform in glTF.
                mm = np.eye(4) if "skin" in node else m
                pos = pos @ mm[:3, :3].T + mm[:3, 3]
                tri = acc(p["indices"]).reshape(-1, 3) if "indices" in p else np.arange(len(pos)).reshape(-1, 3)
                out.append((doc["meshes"][node["mesh"]].get("name", ""), pos, tri))
        for c in node.get("children", []):
            walk(c, m)

    for i in doc["scenes"][doc.get("scene", 0)]["nodes"]:
        walk(i, np.eye(4))
    return out


def project(cam, pts):
    c, r, u = (np.array(cam[k]) for k in ("center", "right", "up"))
    d = pts - c
    px = d @ r / cam["meters_per_pixel"] + cam["width"] / 2
    py = cam["height"] / 2 - d @ u / cam["meters_per_pixel"]
    return np.stack([px, py], axis=-1)


def silhouette(cam, pts, tri):
    px, py = project(cam, pts).T
    img = Image.new("L", (cam["width"], cam["height"]), 0)
    dr = ImageDraw.Draw(img)
    for a, b, e in tri:
        dr.polygon([(px[a], py[a]), (px[b], py[b]), (px[e], py[e])], fill=255)
    return np.asarray(img) > 0


def mask(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    bg = np.median(np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]), axis=0)
    return np.abs(a - bg).sum(2) > 40


def span(m, p, q):
    """Extent of the silhouette `m` along the line p->q about its middle."""
    p, q = np.array(p, float), np.array(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    c = (p + q) / 2

    def inside(t):
        x, y = np.round(c + u * t).astype(int)
        return 0 <= x < m.shape[1] and 0 <= y < m.shape[0] and m[y, x]
    if not inside(0):
        return 0
    t0 = t1 = 0
    while inside(t1 + 1) and t1 < 400:
        t1 += 1
    while inside(t0 - 1) and t0 > -400:
        t0 -= 1
    return t1 - t0


def write_heights(run, guide, frame_dir, config, guidance, cam, matrix, inv, glb, body, rig):
    view = cam["id"]
    marks = {k: obs["px"] for k, values in guidance["landmarks"].items()
             for obs in values if obs["view"] == view and k in mh_heights.JOINTS}
    joint_marks = {k: project(cam, inv[:3, :3] @ rig[name] + inv[:3, 3])
                   for k, name in mh_heights.JOINTS.items() if name in rig}
    evidence = guide / "views"
    path = evidence / config.get("suit_contours_file", "suit_contours.ensemble.json")
    if not path.exists():
        path = evidence / "suit_contours.json"
    ref = np.zeros(body.shape, dtype=np.uint8)
    loops = json.loads(path.read_text())["views"].get(view) if path.exists() else None
    if loops:
        cv2.fillPoly(ref, [np.rint(loop).astype(np.int32) for loop in loops], 1)
    else:
        path = evidence / f"bodysuit_{view}-reg.png"
        from importlib import import_module
        ref = import_module("mh-suit-contours").figure_mask(path)
    scale = cam["meters_per_pixel"] * np.linalg.norm(matrix[:3, :3] @ cam["up"])
    reference = mh_heights.measure(ref, marks, scale)
    fitted = mh_heights.measure(body, joint_marks, scale)
    delta = {k: round(v["height_mm"] - reference["stations"][k]["height_mm"], 2)
             for k, v in fitted["stations"].items() if k in reference["stations"]}
    report = {"neutral_body": str(glb.resolve()), "guidance": str(guide.resolve()),
              "contour_reference": str(path.resolve()), "frame_report": str((frame_dir / "report.json").resolve()),
              "view": view, "units": "mm in exported model scale, above each silhouette's supporting sole",
              "reference": reference, "model": fitted, "model_minus_reference_mm": delta}
    (run / "body-heights.json").write_text(json.dumps(report, indent=2) + "\n")
    height_sheet(run, ref, body, reference, fitted, delta)


def height_sheet(run, reference_mask, body_mask, reference, model, delta):
    font = ImageFont.load_default(size=15)
    rows = [("stature", "Overall height"), ("neck", "Neck joint"),
            ("shoulder_surface", "Shoulder surface"), ("shoulder", "Shoulder joint"),
            ("elbow", "Elbow joint"), ("wrist", "Wrist joint"), ("hip", "Hip joint"),
            ("crotch", "Crotch surface"), ("knee", "Knee joint"), ("ankle", "Ankle joint")]
    sheet = Image.new("RGB", (1090, 620), "#f6f5f0")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 12), run.name + "   /   Standing heights", font=ImageFont.load_default(size=21), fill="#1c2d36")
    for i, (mask_image, measurements, title) in enumerate((
            (reference_mask, reference, "Reference"), (body_mask, model, "Final body"))):
        img = Image.new("RGB", (mask_image.shape[1], mask_image.shape[0]), "#e5e7e4")
        img.paste("#647878", mask=Image.fromarray(mask_image.astype(bool)))
        d = ImageDraw.Draw(img)
        for k, value in measurements["stations"].items():
            x, y = value["px"]
            colour = "#ffdf78" if value["kind"] == "surface" else "#64ebff"
            d.line((x-10,y,x+10,y), fill=colour, width=4)
            d.ellipse((x-4,y-4,x+4,y+4), fill=colour)
        if "floor_y_px" in measurements:
            y = measurements["floor_y_px"]
            d.line((0,y,img.width,y), fill="#ce7748", width=3)
        img.thumbnail((298,460))
        x = 18 + 309*i
        sheet.paste(img, (x, 89))
        draw.text((x, 57), title, font=font, fill="#253943")
    x, y = 639, 58
    for dx, label in ((0,"Landmark"),(176,"Ref mm"),(257,"Body mm"),(350,"Diff mm")):
        draw.text((x+dx,y),label,fill="#253943",font=font)
    for name, label in rows:
        for suffix in (("",) if name in ("stature","neck","crotch") else ("_l","_r")):
            k = name + suffix
            y += 27
            draw.text((x,y), label + (" " + suffix[-1].upper() if suffix else ""),fill="#253943",font=font)
            for dx, val in ((176,reference["stations"].get(k)),(257,model["stations"].get(k))):
                draw.text((x+dx,y),f'{val["height_mm"]:.0f}' if val else "—",fill="#253943",font=font)
            draw.text((x+350,y), f"{delta[k]:+.0f}" if k in delta else "—",fill="#754323",font=font)
    draw.text((18,548),"Yellow   silhouette surface     /     Cyan   joint centre",font=font,fill="#253943")
    draw.text((18,577),"Each height starts at the supporting sole. Difference = final body minus reference, at exported model scale.",font=font,fill="#253943")
    sheet.save(run / "renders/evidence-6-heights.png")


def main(pairs):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = None
    for run, guide in pairs:
        frame_dir = run
        rep = json.loads((run / "report.json").read_text())
        if "working_frame" not in rep and (run / "candidate-report.json").is_file():
            frame_dir = Path(json.loads((run / "candidate-report.json").read_text())["input"])
            rep = json.loads((frame_dir / "report.json").read_text())
        config_path = frame_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        m = np.array(rep["working_frame"]["source_to_working"]).reshape(4, 4).T
        inv = np.linalg.inv(m)
        glb = run / "2-body-rig/fit-diagnostic.glb"
        if not glb.exists():
            glb = run / "3-skin/body.glb"
        final_glb = run / "3-skin/body.glb"
        rig = {}
        if final_glb.exists():
            final_meshes = glb_meshes(final_glb, rig)
        else:
            final_glb, final_meshes = glb, glb_meshes(glb, rig)
        def body_parts(meshes):
            parts = [(pts @ inv[:3, :3].T + inv[:3, 3], tri) for name, pts, tri in meshes
                     if name in ("body", "fit-diagnostic") or name.startswith(("scan_hand_", "wrist_bridge_"))]
            if not parts:
                raise ValueError(f"No body mesh for {run}")
            return parts
        final_parts = body_parts(final_meshes)
        parts = final_parts if glb == final_glb else body_parts(glb_meshes(glb))
        g = json.loads((guide / "guidance.json").read_text())
        views = json.loads((guide / "views.json").read_text())
        if rep.get("source_hash") and any(d.get("source_hash", rep["source_hash"]) != rep["source_hash"] for d in (g, views)):
            raise ValueError(f"The guide and run refer to different source scans: {guide}, {run}")
        cams = {c["id"]: c for c in views["cameras"]}
        front = g["front_view"]
        side = f"body_y{(int(front.split('_y')[1]) + 90) % 360:03d}"
        marks = json.loads((guide / "views/bodysuit.json").read_text()).get("marks", {})
        tiles, lines = [], [f"== {run.name}  ({glb.name})"]
        (run / "renders").mkdir(exist_ok=True)
        sections = {"model": str(final_glb.resolve()), "guide": str(guide.resolve()), "views": {}}
        for view in (front, side):
            camera = cams[view]
            body = np.logical_or.reduce([silhouette(camera, pts, tri) for pts, tri in parts])
            final = body if glb == final_glb else np.logical_or.reduce(
                [silhouette(camera, pts, tri) for pts, tri in final_parts])
            stations = {k[8:-1]: (p, marks[k[:-1] + "2"][1]) for k, (vid, p) in marks.items()
                        if k.startswith("outline_") and k.endswith("1") and vid == view and k[:-1] + "2" in marks}
            mm_per_px = 1000 * camera["meters_per_pixel"] * np.linalg.norm(m[:3, :3] @ camera["right"])
            sections["views"][view] = {"mm_per_pixel": mm_per_px,
                                       "saved_stations": mh_sections.measure(final, stations, mm_per_px)}
            if view == front:
                from importlib import import_module
                suit_mask = import_module("mh-suit-contours").figure_mask(guide / "views" / f"bodysuit_{view}-reg.png")
                landmarks = {k: o["px"] for k, obs in g["landmarks"].items() for o in obs if o["view"] == view}
                extra = mh_sections.neck_transition(suit_mask, landmarks)
                sections["views"][view]["neck_transition_from_registered_image"] = mh_sections.measure(final, extra, mm_per_px)
                write_heights(run, guide, frame_dir, config, g, camera, m, inv, final_glb, final, rig)
            suit = mask(guide / "views" / f"bodysuit_{view}-reg.png")
            iou = (body & suit).sum() / max(1, (body | suit).sum())
            lines.append(f"  {view}: IoU body/suit {iou:.2f}")
            for src in (guide / "views" / f"bodysuit_{view}-reg.png", guide / "views" / f"{view}.png"):
                base = np.asarray(Image.open(src).convert("RGB")).astype(float) * 0.35
                ref = suit if "bodysuit" in src.name else mask(src)
                over = base.copy()
                over[ref & ~body] = [40, 220, 40]
                over[body & ~ref] = [230, 40, 230]
                over[body & ref] = base[body & ref] * 0.4 + 255 * 0.6
                im = Image.fromarray(over.astype(np.uint8))
                d = ImageDraw.Draw(im)
                for k, (v, p) in marks.items():
                    if v == view and k.endswith("1"):
                        q = marks[k[:-1] + "2"][1]
                        d.line([tuple(p), tuple(q)], fill=(255, 220, 0), width=1)
                        if "bodysuit" in src.name:
                            a, b = span(suit, p, q), span(body, p, q)
                            d.text((q[0] + 4, q[1] - 9), f"{k[8:-3]} {b}/{a}", fill=(255, 220, 0), font=font)
                            lines.append(f"    {k[8:-1]:14} suit {a:4d} px  fitted {b:4d} px  ({(b / a - 1) * 100 if a else 0:+.0f} %)")
                d.text((8, 6), f"{view} {'bodysuit' if 'bodysuit' in src.name else 'original'}  IoU {iou:.2f}"
                       f"  green=evidence only  magenta=fit only", fill=(255, 255, 255), font=font)
                tiles.append(im)
        out = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
        x = 0
        for t in tiles:
            out.paste(t, (x, 0))
            x += t.width
        out.save(run / "renders/evidence-4-match.jpg", quality=88)
        (run / "body-sections.json").write_text(json.dumps(sections, indent=2) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    a = [Path(p) for p in sys.argv[1:]]
    main(list(zip(a[0::2], a[1::2])))
