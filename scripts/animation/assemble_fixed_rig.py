"""Attach finished geometric layers to a character's existing skeleton in its bind pose."""

import copy

import numpy as np
from scipy.spatial.transform import Rotation

from animation.fixed_rig import copy_clip
from animation.surface import Surface
from roster.mesh import Glb
from roster.mesh_output import add, finish
from transfer.scan import transform_points


def layer_weights(part, scene, mapping, slots):
    weights = np.zeros((len(part.positions), len(slots)))
    # Several fine template joints can contribute to the same character joint.
    used = np.unique(part.palette[part.weights > 0])
    remap = np.zeros(len(part.bones), dtype=int)
    for i in used:
        name = scene.names[part.bones[i]]
        if name not in mapping or mapping[name] not in slots:
            raise ValueError(f"Missing skin joint mapping for {name}")
        remap[i] = slots[mapping[name]]
    rows = np.arange(len(weights))
    for joint, weight in zip(part.palette.T, part.weights.T):
        np.add.at(weights, (rows, remap[joint]), weight)
    indices = np.argsort(-weights, axis=1)[:, :4]
    values = np.take_along_axis(weights, indices, axis=1)
    if np.min(values.sum(axis=1)) < 1e-8:
        raise ValueError("Found an unbound vertex")
    return indices.astype("<u2"), (values / values.sum(axis=1)[:, None]).astype("<f4")


def add_layer(g, prim, part, rest, source_world, matrix, scene, mapping, slots):
    # Flatten the rest skin transform into positions and direction/morph attributes.
    if part.weights is None:
        raise ValueError(f"Expected a skinned layer at {scene.names[part.node]}")
    skin = source_world[part.bones] @ part.inverse
    linear = matrix[:3, :3] @ np.einsum(
        "nv,nvij->nij", part.weights, skin[part.palette, :3, :3]
    )
    attrs = prim["attributes"]
    attrs["POSITION"] = add(g, transform_points(rest, matrix).astype("<f4"), "VEC3")
    normal_matrix = np.linalg.inv(linear).transpose(0, 2, 1)
    normal_length = tangent_length = None
    for name, basis in [("NORMAL", normal_matrix), ("TANGENT", linear)]:
        if name in attrs:
            data = g.array(attrs[name]).copy()
            vec = np.einsum("nij,nj->ni", basis, data[:, :3])
            length = np.maximum(np.linalg.norm(vec, axis=1)[:, None], 1e-20)
            data[:, :3] = vec / length
            if name == "NORMAL":
                normal_length = length
            else:
                tangent_length = length
                data[:, 3] *= np.sign(np.linalg.det(linear))
            attrs[name] = add(g, data, "VEC3" if name == "NORMAL" else "VEC4")
    for target in prim.get("targets", []):
        for name, acc in list(target.items()):
            basis = normal_matrix if name == "NORMAL" else linear
            vec = np.einsum("nij,nj->ni", basis, g.array(acc))
            if name == "NORMAL":
                vec /= normal_length
            elif name == "TANGENT":
                vec /= tangent_length
            target[name] = add(g, vec.astype("<f4"), "VEC3")
    for attr in list(attrs):
        if attr.startswith(("JOINTS_", "WEIGHTS_")):
            del attrs[attr]
    indices, values = layer_weights(part, scene, mapping, slots)
    attrs["JOINTS_0"] = add(g, indices, "VEC4", 5123)
    attrs["WEIGHTS_0"] = add(g, values, "VEC4")


def rotation(matrix):
    basis = matrix[:3, :3]
    return Rotation.from_matrix(basis / np.linalg.norm(basis, axis=0))


def copy_expressions(g, source_scene, fixed, clip_name, node_map, pairs, matrix):
    """Retain morph curves and transfer only configured rigid facial motion."""
    original = next(
        a for a in source_scene.glb.doc["animations"] if a["name"] == clip_name
    )
    channels, samplers = [], []
    for c in original["channels"]:
        target = c["target"]
        if target["path"] == "weights":
            samplers.append(copy.deepcopy(original["samplers"][c["sampler"]]))
            channels.append(
                dict(
                    sampler=len(samplers) - 1,
                    target=dict(node=node_map[target["node"]], path="weights"),
                )
            )
    times = np.unique(
        np.concatenate([track[0] for track in source_scene.clips[clip_name].values()])
    )
    poses = [source_scene.world(source_scene.sample(clip_name, t)[0]) for t in times]
    old = source_scene.world(source_scene.rest)
    rest = fixed.world(fixed.rest)
    alignment = rotation(matrix)
    deltas = {}
    for dst_name, src_name in pairs.items():
        i, j = fixed.joint(dst_name), source_scene.joint(src_name)
        deltas[i] = [
            alignment * rotation(w[j]) * rotation(old[j]).inv() * alignment.inv()
            for w in poses
        ]
    for i, rotations in deltas.items():
        parent = fixed.parents.get(i)
        ancestor = parent
        while ancestor is not None and ancestor not in deltas:
            ancestor = fixed.parents.get(ancestor)
        parent_rest = (
            rotation(rest[parent]) if parent is not None else Rotation.identity()
        )
        result = []
        for k, delta in enumerate(rotations):
            parent_now = (
                deltas[ancestor][k] * parent_rest
                if ancestor is not None
                else parent_rest
            )
            value = (parent_now.inv() * delta * rotation(rest[i])).as_quat()
            result.append(-value if result and np.dot(value, result[-1]) < 0 else value)
        time_acc = add(g, times.astype("<f4"), "SCALAR")
        g.doc["accessors"][time_acc].update(
            min=[float(times[0])], max=[float(times[-1])]
        )
        rotation_acc = add(g, np.array(result, dtype="<f4"), "VEC4")
        for acc in [time_acc, rotation_acc]:
            g.doc["bufferViews"][g.doc["accessors"][acc]["bufferView"]].pop(
                "target", None
            )
        channels.append(
            dict(sampler=len(samplers), target=dict(node=i, path="rotation"))
        )
        samplers.append(
            dict(input=time_acc, output=rotation_acc, interpolation="LINEAR")
        )
    return dict(name=clip_name, channels=channels, samplers=samplers)


def assemble(asset, fixed_asset, config, output):
    fixed_surface, source = Surface(fixed_asset), Surface(asset)
    if len(fixed_surface.parts) != 1:
        raise ValueError("Expected the single bound scan produced by bind_fixed_rig")
    fixed = fixed_surface.scene
    part = fixed_surface.parts[0]
    # Skin accessors in Surface can be zero-copy views into its original buffer.
    g = Glb(asset)
    ref = fixed.glb
    src_nodes = g.doc["nodes"]
    nodes = copy.deepcopy(ref.doc["nodes"])
    # Keep the donor's joint indices stable but discard the donor surface.
    nodes[part.node].pop("mesh")
    nodes[part.node].pop("skin")
    scenes = copy.deepcopy(ref.doc["scenes"])
    roots = scenes[ref.doc.get("scene", 0)]["nodes"]
    source_world = source.scene.world(source.scene.rest)
    matrix = np.asarray(config["layer_to_rig_world"], dtype=float)
    slots = {fixed.names[n]: i for i, n in enumerate(part.bones)}
    node_map = {}
    mesh_ids = set()
    for q, rest in zip(source.parts, source.rest):
        if q.node not in node_map:
            src = src_nodes[q.node]
            if src["mesh"] in mesh_ids:
                raise ValueError(
                    "Instanced layer meshes must be made single-user before binding"
                )
            mesh_ids.add(src["mesh"])
            node_map[q.node] = len(nodes)
            roots.append(len(nodes))
            node = {
                k: copy.deepcopy(src[k])
                for k in ("name", "mesh", "extras", "weights")
                if k in src
            }
            node["skin"] = 0
            nodes.append(node)
        mesh = src_nodes[q.node]["mesh"]
        prim_id = next(
            i
            for i, p in enumerate(source.scene.glb.doc["meshes"][mesh]["primitives"])
            if p is q.prim
        )
        add_layer(
            g,
            g.doc["meshes"][mesh]["primitives"][prim_id],
            q,
            rest,
            source_world,
            matrix,
            source.scene,
            config["skin_joint_map"],
            slots,
        )
    animations = [copy_clip(ref, g, clip) for clip in ref.doc.get("animations", [])]
    for name, pairs in config.get("expressions", {}).items():
        animations.append(
            copy_expressions(g, source.scene, fixed, name, node_map, pairs, matrix)
        )
    inverse = np.linalg.inv(fixed.world(fixed.rest)[part.bones])
    skin = dict(
        joints=part.bones.tolist(),
        inverseBindMatrices=add(g, inverse.transpose(0, 2, 1).astype("<f4"), "MAT4"),
    )
    g.doc.update(
        nodes=nodes,
        skins=[skin],
        scenes=scenes,
        scene=ref.doc.get("scene", 0),
        animations=animations,
    )
    finish(g, output)
