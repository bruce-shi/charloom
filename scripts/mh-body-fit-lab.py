"""Fit observed skin with the Python body surface solver and audit the exported mesh.

Existing files are immutable. Fits exposed skin, fairs hidden regions, and holds
head, hands, feet and mesh boundaries exactly. No rig/animation adoption.
"""

import argparse, hashlib, json, time
from pathlib import Path
import igl
import numpy as np
from roster.body_surface import solve_surface
from scipy.spatial import cKDTree

from roster import mesh as lab


from roster.mesh_output import add, normals

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--smoothness", type=float, default=20.0)
    ap.add_argument("--data-weight", type=float, default=10.0)
    args = ap.parse_args()
    run = args.run.resolve()
    out = args.out.resolve()
    if out.exists():
        raise ValueError("Output must be a new directory")
    report = json.loads((run / "report.json").read_text())
    g = lab.Glb(run / "3-skin/body.glb")
    p = next(m["primitives"][0] for m in g.doc["meshes"] if m.get("name") == "body")
    raw = g.array(p["attributes"]["POSITION"]).astype(float)
    uv = g.array(p["attributes"]["TEXCOORD_0"])
    rt = g.array(p["indices"]).reshape(-1, 3).astype(int)
    prior = lab.Glb(report["prior"])
    pp = prior.body()
    pv = prior.array(pp["attributes"]["POSITION"]).astype(float)
    puv = prior.array(pp["attributes"]["TEXCOORD_0"])
    rest, first, pw = np.unique(
        np.round(pv, 6), axis=0, return_index=True, return_inverse=True
    )
    rest = pv[first]
    distance, match = cKDTree(puv).query(uv)
    known = distance < 1e-7
    # UV identity is accepted only when all exact duplicates denote one control.
    for i in np.flatnonzero(known):
        same = np.flatnonzero(np.linalg.norm(puv - uv[i], axis=1) < 1e-7)
        if len(np.unique(pw[same])) != 1:
            raise ValueError("Ambiguous template UV identity")
    labels = np.where(known, pw[match], len(rest) + np.arange(len(raw)))
    _, first_raw, weld = np.unique(labels, return_index=True, return_inverse=True)
    base = raw[first_raw]
    t = weld[rt]
    ref = pv[match[first_raw]].copy()
    valid = known[first_raw]
    ref[~valid] = base[~valid]
    inconsistent = np.zeros(len(base), bool)
    inconsistent[weld[np.linalg.norm(raw - base[weld], axis=1) > 1e-5]] = True
    adj, L = lab.adjacency(t, len(base))
    degree = np.asarray(adj.sum(1)).ravel()
    names = [
        prior.doc["nodes"][i].get("name", "").lower()
        for i in prior.doc["skins"][0]["joints"]
    ]
    j = prior.array(pp["attributes"]["JOINTS_0"]).astype(int)
    weights = prior.array(pp["attributes"]["WEIGHTS_0"])
    protect = np.array(
        [n.startswith(("finger", "thumb", "hand", "foot", "toe")) for n in names]
    )
    protected = (weights * protect[j]).sum(1)[match[first_raw]] > 0.5
    free = (
        valid
        & ~protected
        & (ref[:, 1] < pv[:, 1].max() * 0.85)
        & (ref[:, 1] > 0.13)
        & (degree > 0)
    )
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
    e, c = np.unique(edges, axis=0, return_counts=True)
    boundary = np.zeros(len(base), bool)
    boundary[np.unique(e[c == 1])] = True
    free[boundary | inconsistent] = False
    # A fixed collar avoids changing the old face or custom hand seams.
    fixed = ~free
    for _ in range(1):
        fixed |= np.asarray(adj @ fixed.astype(int) > 0).ravel()
    free = ~fixed
    ids = np.flatnonzero(free)
    seg = lab.Glb(run / "1-segment/segmentation.glb")
    sv = []
    sf = []
    all_faces = []
    for mesh in seg.doc["meshes"]:
        for pr in mesh["primitives"]:
            v = seg.array(pr["attributes"]["POSITION"]).astype(float)
            f = seg.array(pr["indices"]).reshape(-1, 3).astype(int)
            color = seg.array(pr["attributes"]["COLOR_0"])
            skin = np.linalg.norm(color[:, :3] - [0.95, 0.55, 0.35], axis=1) < 1e-4
            off = sum(len(z) for z in sv)
            sf.append(f[skin[f].all(1)] + off)
            all_faces.append(f + off)
            sv.append(v)
    sv = np.concatenate(sv)
    sf = np.concatenate(sf)
    tree = igl.AABB()
    tree.init(sv, sf)
    start = time.monotonic()
    x, cohort, _ = solve_surface(base, t, free, scan=(sv, sf, np.concatenate(all_faces)),
                                 smoothness=args.smoothness, data_weight=args.data_weight, rounds=args.rounds)
    assert np.array_equal(x[fixed], base[fixed])
    out.mkdir(parents=True)

    def stats(q):
        dd, _, _ = tree.squared_distance(sv, sf, q[cohort])
        tri = q[t]
        a = np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        return {
            "exposed_cohort_count": len(cohort),
            "exposed_distance_p50_mm": float(np.median(np.sqrt(dd)) * 1000),
            "exposed_distance_p95_mm": float(np.percentile(np.sqrt(dd), 95) * 1000),
            "body_fold_edges": lab.fold_edges(q, ref, t, free),
            "degenerate_active_triangles": int(((a < 1e-12) & free[t].any(1)).sum()),
        }

    exports = []
    vn, _ = normals(x, t)
    for rel in ["3-skin/body.glb", "4-garments/character.glb"]:
        ex = lab.Glb(run / rel)
        body = next(m for m in ex.doc["meshes"] if m.get("name") == "body")
        ep = body["primitives"][0]
        eraw = ex.array(ep["attributes"]["POSITION"]).astype(float)
        euv = ex.array(ep["attributes"]["TEXCOORD_0"])
        dist, mp = cKDTree(np.c_[raw, uv]).query(np.c_[eraw, euv])
        if dist.max() > 1e-7:
            raise ValueError("Export body differs from reference")
        pos = x[weld[mp]].astype("<f4")
        norm = vn[weld[mp]].astype("<f4")
        unchanged = ~free[weld[mp]]
        pos[unchanged] = eraw[unchanged]
        norm[unchanged] = ex.array(ep["attributes"]["NORMAL"])[unchanged]
        ep["attributes"]["POSITION"] = add(ex, pos, "VEC3")
        ep["attributes"]["NORMAL"] = add(ex, norm, "VEC3")
        ep.pop("targets", None)
        body.pop("weights", None)
        for node in ex.doc["nodes"]:
            if "mesh" in node and ex.doc["meshes"][node["mesh"]].get("name") == "body":
                node.pop("weights", None)
        ex.doc.pop("animations", None)
        ex.doc["buffers"][0]["byteLength"] = len(ex.blob)
        path = out / rel
        path.parent.mkdir()
        ex.write(path)
        exports.append(
            {"file": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    np.savez_compressed(
        out / "surface.npz",
        before=base,
        positions=x,
        triangles=t,
        free=free,
        rest=ref,
        cohort=cohort,
        source_positions=sv,
        source_triangles=sf,
        export_to_control=weld,
    )
    result = {
        "input": str(run),
        "method": "body_surface_with_membrane_fairing",
        "status": "pending_independent_geometry_and_visual_review",
        "parameters": {
            "smoothness": args.smoothness,
            "data_weight": args.data_weight,
            "rounds": args.rounds,
        },
        "baseline_sha256": hashlib.sha256(
            (run / "3-skin/body.glb").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(
            Path(report["source"]).read_bytes()
        ).hexdigest(),
        "before": stats(base),
        "after": stats(x),
        "free_vertices": len(ids),
        "fixed_vertices_max_displacement_mm": 0.0,
        "max_displacement_mm": float(np.linalg.norm(x - base, axis=1).max() * 1000),
        "elapsed_seconds": time.monotonic() - start,
        "exports": exports,
        "neutral_only": True,
        "rig_refitted": False,
        "textures_rebaked": False,
        "rust_tests_run": False,
        "solver": "python_scipy",
        "user_approved": False,
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
