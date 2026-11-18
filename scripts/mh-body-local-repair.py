"""Repair an audited local body crossing patch; outputs require independent review.

The sparse solver fairs the selected controls with fixed boundary positions.
"""

import json, numpy as np, importlib.util, sys, hashlib
from pathlib import Path
from roster.body_surface import solve_surface

ROOT = Path(__file__).resolve().parent.parent
s = importlib.util.spec_from_file_location("m", ROOT / "scripts/mh-body-fit-lab.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
parent = Path(sys.argv[1])
out = Path(sys.argv[2])
rings = int(sys.argv[3]) if len(sys.argv) > 3 else 4
out.mkdir(exist_ok=False)
d = dict(np.load(parent / "surface.npz"))
r = json.loads((parent / "report.json").read_text())
audit = json.loads((parent / "geometry-audit.json").read_text())
x = d["positions"].copy()
base = x.copy()
t = d["triangles"]
free = d["free"]
adj, L = m.lab.adjacency(t, len(x))
g = m.lab.Glb(parent / "3-skin/body.glb")
p = next(z["primitives"][0] for z in g.doc["meshes"] if z.get("name") == "body")
rt = g.array(p["indices"]).reshape(-1, 3)
support = np.zeros(len(x), bool)
support[np.unique(d["export_to_control"][rt[np.unique(audit["after_pairs"])]])] = True
for _ in range(rings):
    support |= np.asarray(adj @ support.astype(int) > 0).ravel()
support &= free
ids = np.flatnonzero(support)
x, _, _ = solve_surface(x, t, support)
vn, _ = m.normals(x, t)
exports = []
for rel in ["3-skin/body.glb", "4-garments/character.glb"]:
    ex = m.lab.Glb(parent / rel)
    ep = next(z["primitives"][0] for z in ex.doc["meshes"] if z.get("name") == "body")
    old = ex.array(ep["attributes"]["POSITION"])
    uv = ex.array(ep["attributes"]["TEXCOORD_0"])
    du, mp = m.cKDTree(
        np.c_[
            g.array(p["attributes"]["POSITION"]), g.array(p["attributes"]["TEXCOORD_0"])
        ]
    ).query(np.c_[old, uv])
    assert du.max() < 1e-7
    w = d["export_to_control"][mp]
    v = old.copy()
    n = ex.array(ep["attributes"]["NORMAL"]).copy()
    v[support[w]] = x[w[support[w]]]
    n[support[w]] = vn[w[support[w]]]
    ep["attributes"]["POSITION"] = m.add(ex, v.astype("<f4"), "VEC3")
    ep["attributes"]["NORMAL"] = m.add(ex, n.astype("<f4"), "VEC3")
    ex.doc["buffers"][0]["byteLength"] = len(ex.blob)
    path = out / rel
    path.parent.mkdir()
    ex.write(path)
    exports.append(
        {"file": rel, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    )
d["positions"] = x
np.savez_compressed(out / "surface.npz", **d)
r.update(
    parent=str(parent),
    exports=exports,
    local_repair={
        "rings": rings,
        "support_vertices": len(ids),
        "max_move_mm": float(np.linalg.norm(x - base, axis=1).max() * 1000),
    },
    status="pending_geometry_and_visual_review",
)
r.pop("after", None)
(out / "report.json").write_text(json.dumps(r, indent=2) + "\n")
print(r["local_repair"])
