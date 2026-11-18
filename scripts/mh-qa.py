"""Quality checks for mh-transfer runs, one per known failure class.

    python3 scripts/mh-qa.py RUN_DIR... [--json OUT]

Reads each run's report.json and 3-skin/body.glb and flags:

  1 free-fit   creases the fit added to the body (dihedral > 0.8 rad sharper
               than MakeHuman's own shape, per region incl. hands, feet, legs),
               inverted triangles, edge stretch
  2 painted    eye-landmark residuals, face bumps/folds
  3 generator  hand / foot fits far from the scan only as information
  4 hidden     asymmetry of hidden and mixed regions, hidden body vs dressed
               girths (too thin / outside the clothes)
  5 body-cloth walk poke-through overall and per garment, at rest
  6 symmetry   left/right bone-length difference after the rig refit

Each check prints PASS / WARN / FAIL with the number behind it, so a run can
be judged without opening renders, and regressions show up across runs.
"""
import json
import math
import struct
import sys
from pathlib import Path



def read_glb(path):
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    doc = json.loads(b[20:20 + n])
    rest = b[20 + n:]
    blob = rest[8:8 + struct.unpack("<I", rest[:4])[0]] if rest else b""
    return doc, blob


COMP = {5126: ("f", 4), 5125: ("I", 4), 5123: ("H", 2), 5121: ("B", 1)}
NUM = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def accessor(doc, blob, i):
    a = doc["accessors"][i]
    bv = doc["bufferViews"][a["bufferView"]]
    fmt, size = COMP[a["componentType"]]
    n = NUM[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride") or size * n
    out = []
    for k in range(a["count"]):
        o = off + k * stride
        out.append(struct.unpack("<" + fmt * n, blob[o:o + size * n]))
    return out


def body_mesh(path):
    doc, blob = read_glb(path)
    mesh = next(m for m in doc["meshes"] if m["name"].startswith(("body", "base")))
    p = mesh["primitives"][0]
    pos = accessor(doc, blob, p["attributes"]["POSITION"])
    idx = [i[0] for i in accessor(doc, blob, p["indices"])]
    tris = [tuple(idx[k:k + 3]) for k in range(0, len(idx), 3)]
    joints = accessor(doc, blob, p["attributes"]["JOINTS_0"])
    weights = accessor(doc, blob, p["attributes"]["WEIGHTS_0"])
    node = next(n for n in doc["nodes"] if "skin" in n and doc["meshes"][n["mesh"]] is mesh)
    names = [doc["nodes"][j].get("name", "") for j in doc["skins"][node["skin"]]["joints"]]
    dom = [names[j[max(range(4), key=lambda k: w[k])]] for j, w in zip(joints, weights)]
    return pos, tris, dom


def region(name):
    n = name.lower()
    if any(k in n for k in ("finger", "metacarpal", "thumb", "hand", "wrist")):
        return "hand"
    if any(k in n for k in ("toe", "foot")):
        return "foot"
    if any(k in n for k in ("head", "jaw", "eye", "ear", "orbicularis", "levator", "oris", "risorius",
                              "temporalis", "nasal", "tongue", "teeth", "mentalis", "brow", "cheek", "lip", "zygomatic")):
        return "head"
    if any(k in n for k in ("thigh", "upperleg", "lowerleg", "calf", "shin", "knee", "upper_leg", "lower_leg")):
        return "leg"
    if any(k in n for k in ("upperarm", "lowerarm", "upper_arm", "lower_arm", "elbow", "forearm")):
        return "arm"
    return "torso"


def normal(p, t):
    a, b, c = (p[i] for i in t)
    u = [b[k] - a[k] for k in range(3)]
    v = [c[k] - a[k] for k in range(3)]
    n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    l = math.sqrt(sum(x * x for x in n)) or 1e-20
    return [x / l for x in n]


def weld(pos):
    key = {}
    return [key.setdefault((round(x, 6), round(y, 6), round(z, 6)), len(key)) for x, y, z in pos]


def new_folds(fitted, prior, tris, dom, wid):
    """Edges whose dihedral is > 0.8 rad sharper than in MakeHuman's shape."""
    edges = {}
    for t_i, t in enumerate(tris):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (min(wid[a], wid[b]), max(wid[a], wid[b]))
            edges.setdefault(k, []).append((t_i, a))
    nf = [normal(fitted, t) for t in tris]
    npr = [normal(prior, t) for t in tris]
    out = {}
    for faces in edges.values():
        if len(faces) != 2:
            continue
        (t0, a), (t1, _) = faces
        d = lambda n: math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(n[t0], n[t1])))))
        if d(nf) - d(npr) > 0.8:
            r = region(dom[a])
            out[r] = out.get(r, 0) + 1
    return out


def grade(value, warn, fail, higher_bad=True):
    if value is None:
        return "N/A "
    bad = (lambda x, y: x > y) if higher_bad else (lambda x, y: x < y)
    return "FAIL" if bad(value, fail) else "WARN" if bad(value, warn) else "PASS"


def check(run):
    run = Path(run)
    r = json.loads((run / "report.json").read_text())
    b = r["stage2_body_fit"]
    rows = []

    def row(cls, what, value, warn, fail, unit="", higher_bad=True):
        rows.append((grade(value, warn, fail, higher_bad), cls, what, value, unit))

    # 1 free fit: new creases by region against MakeHuman's own rest shape.
    try:
        pos, tris, dom = body_mesh(run / "3-skin/body.glb")
        # MakeHuman's own shape: the prior's base body next to the shape prior.
        prior = Path(r["prior"])
        base = prior.with_name("body.glb") if (prior.with_name("body.glb")).exists() else prior
        ppos, ptris, _ = body_mesh(base)
        if len(ppos) == len(pos) and ptris == tris:
            folds = new_folds(pos, ppos, tris, dom, weld(ppos))
            for reg in ("torso", "arm", "hand", "leg", "foot"):
                row("1 free-fit", f"new creases {reg}", folds.get(reg, 0), 15, 60)
        else:
            row("1 free-fit", "new creases (topology differs: digits cut?)", None, 0, 0)
    except Exception as e:
        row("1 free-fit", f"new creases (error {e})", None, 0, 0)
    row("1 free-fit", "inverted triangles", b["inverted_triangles"], 0, 3)
    row("1 free-fit", "edge stretch max", round(b["distortion_vs_prior_rest"]["edge_ratio"]["max"], 2), 4, 7)
    # 2 painted detail: eyes and face.
    f = b.get("face_landmark_residual_m") or {}
    eye = max([v for k, v in f.items() if "eye" in k] or [0])
    row("2 painted", "eye landmark residual max", round(eye * 1000, 1), 2.5, 5, "mm")
    row("2 painted", "face new folds", (b.get("face_new_fold_vertices") or {}).get("fitted"), 60, 150)
    row("2 painted", "face bumps >1 mm", (b.get("face_above_scan") or {}).get("vertices_over_1mm"), 150, 350)
    # 3 generator defects: region skin distance (information; hands/feet are
    # MakeHuman's on purpose).
    reg = b.get("scan_skin_to_body_by_region_m") or {}
    for k in ("Head", "Torso", "Arm", "Leg"):
        # A region with a handful of skin samples (a stray label) says nothing.
        if k in reg and reg[k].get("count", 0) >= 200:
            row("3 visible", f"skin p95 {k.lower()}", round(reg[k]["p95"] * 1000, 1), 12, 25, "mm")
    # 4 hidden body: asymmetry and size against the clothes.
    asym = (b.get("asymmetry") or {}).get("fitted", {}).get("by_class", {})
    for k in ("hidden", "mixed"):
        if k in asym:
            row("4 hidden", f"asymmetry p95 {k}", round(asym[k]["p95"] * 1000, 1), 8, 15, "mm")
    g = (r.get("stage2_body_size") or {}).get("girths_cm") or {}
    fit, dressed = g.get("fitted", {}), g.get("scan_dressed", {})
    for k in ("chest", "waist", "hips", "thigh", "upper_arm"):
        if k in fit and k in dressed and dressed[k]:
            ratio = fit[k] / dressed[k]
            # Dressed girths include loose robes and props: a low ratio is
            # only a hint (look at the render), a body outside its clothes
            # is a failure.
            row("4 hidden", f"{k} body/dressed", round(ratio, 2), 0.7, 0.0, "", higher_bad=False)
            if ratio > 1.0:
                rows.append(("FAIL", "4 hidden", f"{k} body outside clothes", round(ratio, 2), ""))
    # 5 body vs cloth: poke-through.
    c = r["stage5_checks"][0]
    row("5 body-cloth", "walk poke-through max", round(c["inside_fraction_over_frames"]["max"] * 100, 2), 1.0, 2.5, "%")
    # Fragments under 50 inside vertices (a strap, a quiver end) are noise.
    big = lambda g: ((g.get("inside_depth_m") or {}).get("count") or 0) >= 50 or g["inside_fraction"] < 0.5
    # Visible poke-through: outward-facing cloth inside the body (a lining
    # or tucked layer below the skin is hidden behind the outer face).
    seen = lambda g: g.get("inside_outward_fraction", g["inside_fraction"])
    worst = max((seen(x) for x in c["worst_frame"]["garments"] if big(x)), default=0)
    row("5 body-cloth", "worst garment in walk", round(worst * 100, 1), 5, 12, "%")
    rest = max((seen(x) for x in c["rest"]["garments"]), default=0)
    row("5 body-cloth", "worst garment at rest", round(rest * 100, 1), 1, 3, "%")
    # 6 symmetry: bone lengths after the rig refit.
    bones = (b.get("asymmetry") or {}).get("bones") or {}
    # Major limb bones only: millimetre face and finger-tip bones make
    # large ratios from tiny differences.
    major = ("clavicle", "upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist",
             "upperleg01", "upperleg02", "lowerleg01", "lowerleg02", "foot")
    for stage in ("pose", "rig"):
        limbs = (bones.get(stage) or {}).get("limbs") or {}
        rel = {k: v["relative"] for k, v in limbs.items() if k in major and "relative" in v}
        if rel:
            worst = max(rel, key=rel.get)
            row("6 symmetry", f"limb length L/R ({stage}, {worst})", round(rel[worst] * 100, 1), 2, 5, "%")
    return rows


def main(argv):
    out = None
    if "--json" in argv:
        i = argv.index("--json")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    allrows = {}
    for run in argv:
        rows = check(run)
        allrows[Path(run).name] = rows
        fails = sum(1 for x in rows if x[0] == "FAIL")
        warns = sum(1 for x in rows if x[0] == "WARN")
        print(f"== {Path(run).name}: {fails} FAIL, {warns} WARN")
        for g, cls, what, value, unit in rows:
            if g != "PASS":
                print(f"   {g} {cls:13} {what}: {value}{(' ' + unit) if unit else ''}")
    if out:
        Path(out).write_text(json.dumps({k: [list(x) for x in v] for k, v in allrows.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
