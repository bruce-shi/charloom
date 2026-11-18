"""Assemble reviewed, separate torso and limb scan shells on the fitted rig."""

import igl
import manifold3d as md
import numpy as np
from common import add
from scipy.spatial import cKDTree


def joint(guide, name):
    return np.asarray(guide["joint_section_depth"][name]["point"])


def shell(points, faces, shoulder=None):
    used, local = np.unique(faces, return_inverse=True)
    vertices = points[used]
    triangles, _ = igl.bfs_orient(local.reshape(-1, 3))
    for loop in igl.boundary_loop_all(triangles):
        ring = np.asarray(loop)
        center = vertices[ring].mean(0)
        if (
            shoulder is None
            or np.linalg.norm(center - shoulder) > np.ptp(vertices[ring], axis=0).max()
        ):
            raise ValueError(
                "Open scan boundary outside a reviewed shoulder attachment"
            )
        triangles = np.vstack(
            [
                triangles,
                np.column_stack(
                    (np.roll(ring, -1), ring, np.full(len(ring), len(vertices)))
                ),
            ]
        )
        vertices = np.vstack([vertices, center])

    def make(f):
        return md.Manifold(
            md.Mesh(vertices.astype("f"), np.ascontiguousarray(f, dtype="I"))
        )

    if make(triangles).volume() < 0:
        triangles = triangles[:, ::-1].copy()
    # Some closed scan toes fold through one another. Extract the exterior
    # before binding so their hidden crossing faces cannot follow different bones.
    from igl.copyleft import cgal

    pairs = cgal.remesh_self_intersections(vertices, triangles, detect_only=True)[2]
    if len(pairs):
        vertices, triangles, _ = cgal.mesh_boolean(
            vertices, triangles, np.empty((0, 3)), np.empty((0, 3), dtype=int), "union"
        )
    solid = make(triangles)
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError("Reviewed body shell does not enclose a positive volume")
    return solid, len(pairs)


def prepare(config, source_hash, vertices, faces, guide, head):
    seeds = config["skin_seeds"]
    parts = {"torso", "arm_l", "arm_r", "leg_l", "leg_r"}
    if config["source_hash"] != source_hash or set(seeds) != parts:
        raise ValueError(
            "Body volume requires source-matched torso and limb seed triangles"
        )
    points, weld = np.unique(vertices, axis=0, return_inverse=True)
    triangles = weld[faces]
    _, labels = igl.facet_components(triangles)
    if any(i < 0 or i >= len(faces) for i in seeds.values()) or len(
        set(labels[list(seeds.values())])
    ) != len(parts):
        raise ValueError("Body volume seeds must identify distinct scan surfaces")
    surface = {key: np.flatnonzero(labels == labels[i]) for key, i in seeds.items()}
    clouds = {key: points[np.unique(triangles[ids])] for key, ids in surface.items()}
    solids = {
        key: shell(
            points,
            triangles[ids],
            joint(guide, "shoulder_" + key[-1]) if key.startswith("arm_") else None,
        )
        for key, ids in surface.items()
    }
    body = md.Manifold.batch_boolean(
        [solid for solid, _ in solids.values()], md.OpType.Add
    )
    fill_faces = 0
    if len(body.decompose()) > 1 and config.get("pelvis_fill") == "proximal_hull":
        # The torso and the two thighs can end under a single garment. Span
        # the missing interval using the first two edge rows of each surface.
        bands = []
        for key in ("torso", "leg_l", "leg_r"):
            q = clouds[key]
            tri = points[triangles[surface[key]]]
            step = np.median(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2))
            near = (
                q[:, 1] - q[:, 1].min() if key == "torso" else q[:, 1].max() - q[:, 1]
            )
            bands.append(q[near <= 2 * step])
        fill = md.Manifold.hull_points(np.vstack(bands))
        fill_faces = fill.num_tri()
        body += fill
    if len(body.decompose()) != 1 or (head ^ body).volume() <= 0:
        raise ValueError(
            "Observed body does not form a connected solid overlapping the head"
        )
    body = body.as_original()
    return (
        head + body,
        body.original_id(),
        surface,
        {
            "inferred_pelvis_hull_triangles": fill_faces,
            "source_crossings": {key: n for key, (_, n) in solids.items()},
        },
    )


def partition(mesh, body_id, keep):
    owner = np.repeat(mesh.run_original_id, np.diff(mesh.run_index) // 3)
    v = np.asarray(mesh.vert_properties[:, :3], dtype=float)
    f = np.asarray(mesh.tri_verts, dtype=int)

    def part(mask):
        used, local = np.unique(f[mask], return_inverse=True)
        return used, local.reshape(-1, 3)

    hi, hf = part(keep & (owner != body_id))
    bi, bf = part(owner == body_id)
    _, head_ring, body_ring = np.intersect1d(hi, bi, return_indices=True)
    if len(head_ring) < 3:
        raise ValueError("Head and body union has no shared skin boundary")
    return v[hi], hf, head_ring, v[bi], bf, body_ring


def body_weights(
    points,
    triangles,
    src,
    sf,
    surfaces,
    guide,
    donor,
    donor_joints,
    donor_weights,
    names,
    hands,
):
    """Use the anatomical side to select influences even near touching limbs."""
    distances = []
    keys = list(surfaces)
    for key in keys:
        d, _, _ = igl.point_mesh_squared_distance(points, src, sf[surfaces[key]])
        distances.append(d)
    labels = np.argmin(distances, axis=0)
    field = np.zeros((len(points), len(names)))

    def smooth(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    for owner, key in enumerate(keys):
        ids = np.flatnonzero(labels == owner)
        p = points[ids]
        if not len(ids):
            continue
        side = key[-1].upper()
        if key.startswith("arm_"):
            shoulder, elbow, wrist = [
                joint(guide, k + "_" + key[-1]) for k in ("shoulder", "elbow", "wrist")
            ]
            upper, lower = elbow - shoulder, wrist - elbow
            u = ((p - shoulder) @ upper) / (upper @ upper)
            t = ((p - elbow) @ lower) / (lower @ lower)
            # Partition the arm through the twist bones, with a smooth bend
            # across each hinge instead of a cut and a separate wrist strip.
            elbow_mix = smooth(0.5 + np.where(t > 0, t, u - 1.0) * 2.0)
            shoulder_mix = smooth(0.5 + u * 2.0)
            wrist_mix = smooth(0.5 + (t - 1.0) * 2.5)
            upper_twist, lower_twist = smooth(u), smooth(t)
            pairs = [
                ("spine01", 1.0 - shoulder_mix),
                (
                    "upperarm01." + side,
                    shoulder_mix * (1 - elbow_mix) * (1 - upper_twist),
                ),
                ("upperarm02." + side, shoulder_mix * (1 - elbow_mix) * upper_twist),
                (
                    "lowerarm01." + side,
                    shoulder_mix * elbow_mix * (1 - wrist_mix) * (1 - lower_twist),
                ),
                (
                    "lowerarm02." + side,
                    shoulder_mix * elbow_mix * (1 - wrist_mix) * lower_twist,
                ),
                ("wrist." + side, shoulder_mix * elbow_mix * wrist_mix),
            ]
            for name, w in pairs:
                field[ids, names.index(name)] = w
        else:
            allowed = np.asarray(
                [
                    name.startswith(
                        ("root", "spine", "pelvis", "breast", "clavicle", "neck")
                    )
                    if key == "torso"
                    else (
                        name.endswith("." + side)
                        and name.startswith(
                            ("pelvis", "upperleg", "lowerleg", "foot", "toe")
                        )
                    )
                    for name in names
                ]
            )
            relevant = allowed[donor_joints] * donor_weights
            mask = relevant.sum(1) > 0.5
            d, nn = cKDTree(donor[mask]).query(p, k=4)
            coeff = 1.0 / np.maximum(d, 1.0e-5) ** 2
            joints = donor_joints[mask][nn]
            weights = relevant[mask][nn] * coeff[:, :, None]
            np.add.at(field, (ids[:, None, None], joints), weights)
    # This route has no toe-specific observations. Transfer the common
    # ball-of-foot roll without imposing the template's five digit chains
    # on a different toe count.
    for side in ("L", "R"):
        toes = [
            i
            for i, name in enumerate(names)
            if name.startswith("toe") and name.endswith("." + side)
        ]
        if toes:
            rolled = field[:, toes].sum(axis=1)
            field[:, toes] = 0
            field[:, names.index("toe1-1." + side)] = rolled
    digit_domains = []
    for key, (hp, hj, hw, digits) in hands.items():
        side = key[-1].upper()
        ids = np.flatnonzero(labels == keys.index(key))
        p = points[ids]
        dist, nearest = cKDTree(hp).query(p)
        palm = field[ids, names.index("wrist." + side)].copy()
        for digit, native in enumerate(digits, 1):
            prefix = f"finger{digit}_{side.lower()}_"
            marks = guide["hand_points"]
            if not all(prefix + k in marks for k in ("base", "mid", "tip")):
                continue
            base, mid, tip = (
                np.asarray(marks[prefix + k]) for k in ("base", "mid", "tip")
            )
            axis = mid - base
            length = np.linalg.norm(axis)
            if length <= 1.0e-6:
                continue
            # At the measured digit base the palm starts to bend.
            # The scaffold's metacarpals can extend deep into a broad
            # three-finger palm; carrying their curl there folds it.
            bend = smooth(0.25 + (p - base) @ axis / (length * length))
            eligible = np.asarray(
                [
                    name.startswith(f"finger{native}-") and name.endswith("." + side)
                    for name in names
                ]
            )
            digit_domains.append(
                (ids, np.flatnonzero(eligible), bend, names.index("wrist." + side))
            )
            near_hand = 1.0 - smooth(dist / length)
            for ji, weight in zip(hj[nearest].T, hw[nearest].T):
                moved = weight * eligible[ji] * bend * palm * near_hand
                np.add.at(field, (ids, ji), moved)
                field[ids, names.index("wrist." + side)] -= moved
    # The union inserts new triangles between shells. Diffuse weights over
    # the connected surface around the shoulder and wrist so neighbouring
    # triangles on either side of those joins share a continuous bend.
    from scipy.sparse import diags, eye
    from scipy.sparse.linalg import spsolve

    field /= np.maximum(field.sum(axis=1, keepdims=True), 1.0e-12)
    sigma = np.zeros(len(points))
    for name in ("shoulder_l", "shoulder_r", "wrist_l", "wrist_r"):
        radius = guide["joint_section_depth"][name]["thickness_m"] / 2
        dist = np.linalg.norm(points - joint(guide, name), axis=1)
        sigma = np.maximum(sigma, radius * smooth(1.0 - dist / (3.0 * radius)))
    lap = igl.cotmatrix(points, triangles.astype(np.int64)).tocsr()
    lap.setdiag(0)
    lap.data = np.maximum(lap.data, 0.0)
    lap.setdiag(-np.asarray(lap.sum(axis=1)).ravel())
    mass = igl.massmatrix(
        points, triangles.astype(np.int64), igl.MASSMATRIX_TYPE_BARYCENTRIC
    ).diagonal()
    system = eye(len(points)) - diags(sigma**2 / np.maximum(mass, 1.0e-15)) @ lap
    field = np.maximum(spsolve(system.tocsc(), field), 0.0)
    # Diffusion joins palm weights across the wrist. Limit its distal bone
    # support to the measured finger: neighbouring palm vertices must not
    # choose different finger curls when the four influences are packed.
    for ids, joints, bend, wrist in digit_domains:
        removed = field[np.ix_(ids, joints)] * (1.0 - bend[:, None])
        field[np.ix_(ids, joints)] -= removed
        field[ids, wrist] += removed.sum(axis=1)
    # Actual boolean intersection vertices identify the shoulder socket on
    # both source shells. Keep the thin triangles around that curve together
    # while the upper arm rotates away from the torso.
    for side in ("l", "r"):
        shared = (
            np.maximum(
                distances[keys.index("torso")], distances[keys.index("arm_" + side)]
            )
            < 1.0e-12
        )
        ring = points[shared]
        if len(ring) < 3:
            continue
        radius = np.linalg.norm(ring - ring.mean(0), axis=1).mean()
        if radius <= 1.0e-6:
            continue
        retain = smooth(cKDTree(ring).query(points)[0] / radius)
        attachment = field[shared].mean(axis=0)
        field = field * retain[:, None] + attachment * (1.0 - retain[:, None])
    j = np.argsort(field, axis=1)[:, -4:][:, ::-1]
    w = np.take_along_axis(field, j, axis=1)
    if np.any(w.sum(1) <= 0):
        raise ValueError("Unbound observed body vertices")
    return j.astype("uint16"), w / w.sum(1)[:, None]


def bind_neck(head, body, head_ring, body_ring, names, hj, hw, bj, bw, blink, guide):
    """Bind both sides of the actual union boundary through a common neck band."""
    ring = head[head_ring]
    radius = np.ptp(ring[:, [0, 2]], axis=0).mean() / 2
    if radius <= 0:
        raise ValueError("Degenerate neck boundary")
    near = cKDTree(ring)
    neck = names.index("neck01")
    for pts, joints, weights in ((head, hj, hw), (body, bj, bw)):
        d = near.query(pts)[0]
        band = radius
        if pts is body:
            shoulder_y = np.mean(
                [joint(guide, k)[1] for k in ("shoulder_l", "shoulder_r")]
            )
            if ring[:, 1].min() > shoulder_y:
                band = min(band, ring[:, 1].min() - shoulder_y)
        ids = np.flatnonzero(d < band)
        u = np.clip(d[ids] / band, 0, 1)
        keep = u * u * (3.0 - 2.0 * u)
        field = np.zeros((len(ids), len(names)))
        np.add.at(
            field,
            (np.arange(len(ids))[:, None], joints[ids]),
            weights[ids] * keep[:, None],
        )
        field[:, neck] += 1.0 - keep
        j = np.argsort(field, axis=1)[:, -4:][:, ::-1]
        w = np.take_along_axis(field, j, axis=1)
        joints[ids], weights[ids] = j, w / w.sum(1)[:, None]
        if pts is head:
            blink[ids] *= keep[:, None]
    hj[head_ring], hw[head_ring] = bj[body_ring], bw[body_ring]


def align_pivots(g, guide):
    """Place hinges on measured axes, preserving local animation deltas."""
    skin = g.doc["skins"][0]
    nodes = skin["joints"]
    names = [g.doc["nodes"][n]["name"] for n in nodes]
    inv = g.array(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    rest = np.linalg.inv(inv.astype(float))
    new = rest[:, :3, 3].copy()
    for side in ("l", "r"):
        pivots = [joint(guide, k + "_" + side) for k in ("shoulder", "elbow", "wrist")]
        ids = [
            names.index(k + "." + side.upper())
            for k in ("upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist")
        ]
        for i in (0, 2):
            start, twist, end = ids[i : i + 3]
            axis = rest[end, :3, 3] - rest[start, :3, 3]
            fraction = np.clip(
                np.dot(rest[twist, :3, 3] - rest[start, :3, 3], axis) / (axis @ axis),
                0.0,
                1.0,
            )
            new[start], new[end] = pivots[i // 2 : i // 2 + 2]
            new[twist] = new[start] + fraction * (new[end] - new[start])
        toe_roots = [
            i
            for i, name in enumerate(names)
            if name.startswith("toe") and name.endswith("-1." + side.upper())
        ]
        if toe_roots:
            new[names.index("toe1-1." + side.upper())] = rest[toe_roots, :3, 3].mean(
                axis=0
            )
    new[names.index("neck01")] = joint(guide, "neck")
    parents = {
        child: i
        for i, node in enumerate(g.doc["nodes"])
        for child in node.get("children", [])
    }
    lookup = {node: i for i, node in enumerate(nodes)}
    shifts = {}
    for i, node in enumerate(nodes):
        parent = lookup.get(parents.get(node))
        delta = new[i] - rest[i, :3, 3]
        relative = (
            delta
            if parent is None
            else np.linalg.solve(
                rest[parent, :3, :3], delta - (new[parent] - rest[parent, :3, 3])
            )
        )
        if np.linalg.norm(relative) > 1.0e-10:
            part = g.doc["nodes"][node]
            if "matrix" in part:
                part["matrix"][12:15] = (
                    np.array(part["matrix"][12:15]) + relative
                ).tolist()
            else:
                part["translation"] = (
                    np.array(part.get("translation", [0.0, 0.0, 0.0])) + relative
                ).tolist()
            shifts[node] = relative
        if np.linalg.norm(delta) > 1.0e-10:
            world = rest[i].copy()
            world[:3, 3] = new[i]
            inv[i] = np.linalg.inv(world)
    for animation in g.doc.get("animations", []):
        for ch in animation["channels"]:
            target = ch["target"]
            if target["path"] == "translation" and target["node"] in shifts:
                sampler = dict(animation["samplers"][ch["sampler"]])
                track = g.array(sampler["output"]).copy()
                if sampler.get("interpolation") == "CUBICSPLINE":
                    track[1::3] += shifts[target["node"]]
                else:
                    track += shifts[target["node"]]
                sampler["output"] = add(g, track, "VEC3")
                view = g.doc["accessors"][sampler["output"]]["bufferView"]
                g.doc["bufferViews"][view].pop("target", None)
                ch["sampler"] = len(animation["samplers"])
                animation["samplers"].append(sampler)
