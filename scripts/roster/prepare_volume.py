"""Reconstruct a volumetric head target with retained functional eye/mouth cavities.

Visible source skin is the geometric observation. Missing skin under segmented
hair/garments is capped and inferred; source hair remains a separate final mesh.
"""

import json, sys
from pathlib import Path
import bpy, bmesh, numpy as np

ROOT = Path(__file__).resolve().parents[2]
args = sys.argv[sys.argv.index("--") + 1 :]
run = Path(args[0])
out = Path(args[1])
plane = float(args[2])
out.mkdir(parents=True, exist_ok=False)
base = json.load(open(run / "report.json"))
sys.path.insert(0, str(ROOT / "scripts"))
from roster.mesh import Glb

g = Glb(ROOT / Path(base["source"]))
node = next(n for n in g.doc["nodes"] if "mesh" in n)
pr = g.doc["meshes"][node["mesh"]]["primitives"][0]
T = (
    np.array(base["working_frame"]["source_to_working"]).reshape(4, 4).T
    @ np.array(node.get("matrix", np.eye(4).T.ravel())).reshape(4, 4).T
)
sv = g.array(pr["attributes"]["POSITION"]).astype(float) @ T[:3, :3].T + T[:3, 3]
sf = g.array(pr["indices"]).reshape(-1, 3).astype(int)
from mathutils import kdtree

_, canon = np.unique(sv, axis=0, return_inverse=True)
kd = kdtree.KDTree(len(sv))
for i, point in enumerate(sv):
    kd.insert(point, i)
kd.balance()
excluded = set()
hair_faces = set()
matched = 0
for path in (run / "1-segment/pieces").glob("*.glb"):
    if path.stem in ("ears", "tusks"):
        continue
    pg = Glb(path)
    pp = pg.doc["meshes"][0]["primitives"][0]
    pv = pg.array(pp["attributes"]["POSITION"])
    pf = pg.array(pp["indices"]).reshape(-1, 3)
    near = [kd.find(point) for point in pv]
    assert max(z[2] for z in near) < 2e-6, (path, "source mismatch")
    idx = canon[np.array([z[1] for z in near])]
    facekeys = {tuple(sorted(idx[face])) for face in pf}
    if path.stem in ["hair", "beard", "moustache"]:
        hair_faces.update(facekeys)
    else:
        excluded.update(facekeys)
    matched += len(pf)
candidate = sf[sv[sf, 1].max(1) > plane - 0.025]
keep = np.array([tuple(sorted(canon[face])) not in excluded for face in candidate])
print(
    "semantic exact triangles",
    len(candidate),
    "retained skin",
    int(keep.sum()),
    "excluded piece triangles",
    matched,
    flush=True,
)
faces = candidate[keep]
is_hair = np.array([tuple(sorted(canon[face])) in hair_faces for face in faces])
used = np.unique(faces)
inv = np.full(len(sv), -1, int)
inv[used] = np.arange(len(used))
p = sv[used]
faces = inv[faces]
_, first, weld = np.unique(
    np.round(p, 7), axis=0, return_index=True, return_inverse=True
)
p = p[first]
faces = weld[faces]
fixed = np.zeros(len(p), bool)
fixed[np.unique(faces[~is_hair])] = True
free = ~fixed
edges = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
edges = np.r_[edges, edges[:, ::-1]]
original = p.copy()
degree = np.bincount(edges[:, 0], minlength=len(p))
if "--prepare-only" in args:
    np.savez_compressed(
        out / "source-target-input.npz", positions=p, triangles=faces, free=free
    )
    sys.exit(0)
for iteration in range(30):
    total = np.zeros_like(p)
    np.add.at(total, edges[:, 0], p[edges[:, 1]])
    goal = total / np.maximum(degree, 1)[:, None]
    p[free] += 0.45 * (goal[free] - p[free])
print(
    "under-hair inferred vertices",
    int(free.sum()),
    "max smoothing mm",
    np.linalg.norm(p - original, axis=1).max() * 1000,
    flush=True,
)
bpy.ops.wm.read_factory_settings(use_empty=True)


def object_from(name, v, f):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(v).tolist(), [], np.asarray(f).tolist())
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


obj = object_from("resampled-skin", p, faces)
bm = bmesh.new()
bm.from_mesh(obj.data)
bmesh.ops.bisect_plane(
    bm,
    geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
    dist=1e-7,
    plane_co=(0, plane - 0.01, 0),
    plane_no=(0, 1, 0),
    clear_inner=True,
    clear_outer=False,
)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")


def components(bm):
    remaining = set(bm.faces)
    groups = []
    while remaining:
        face = remaining.pop()
        stack = [face]
        group = {face}
        while stack:
            face = stack.pop()
            for e in face.edges:
                for other in e.link_faces:
                    if other in remaining:
                        remaining.remove(other)
                        group.add(other)
                        stack.append(other)
        groups.append(group)
    return groups


groups = components(bm)
print(
    "source components",
    sorted([(len(z), sum(f.calc_area() for f in z)) for z in groups], reverse=True)[
        :12
    ],
    flush=True,
)
main = max(groups, key=lambda z: sum(f.calc_area() for f in z))
bmesh.ops.delete(
    bm, geom=[f for z in groups if z is not main for f in z], context="FACES"
)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
new = bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
print("inferred caps", len(new["faces"]), flush=True)
bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
bmesh.ops.triangulate(bm, faces=list(bm.faces))
print(
    "target closed status",
    len(bm.verts),
    len(bm.faces),
    "boundary",
    sum(e.is_boundary for e in bm.edges),
    "nonmanifold",
    sum(not e.is_manifold for e in bm.edges),
    "volume",
    bm.calc_volume(signed=True),
    flush=True,
)
bm.to_mesh(obj.data)
bm.free()
obj.data.calc_loop_triangles()
np.savez_compressed(
    out / "closed-target.npz",
    positions=np.array([z.co[:] for z in obj.data.vertices]),
    triangles=np.array([z.vertices[:] for z in obj.data.loop_triangles]),
)

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
mod = obj.modifiers.new("volume", "REMESH")
mod.mode = "VOXEL"
mod.voxel_size = 0.00165
mod.use_remove_disconnected = False
bpy.ops.object.modifier_apply(modifier=mod.name)
mod = obj.modifiers.new("fair", "SMOOTH")
mod.factor = 0.16
mod.iterations = 2
bpy.ops.object.modifier_apply(modifier=mod.name)
mod = obj.modifiers.new("surface budget", "DECIMATE")
mod.ratio = min(1, 40000 / max(1, len(obj.data.polygons) * 2))
mod.use_collapse_triangulate = True
bpy.ops.object.modifier_apply(modifier=mod.name)
obj.data.calc_loop_triangles()
np.savez_compressed(
    out / "raw-head.npz",
    positions=np.array([z.co[:] for z in obj.data.vertices]),
    triangles=np.array([z.vertices[:] for z in obj.data.loop_triangles]),
)
