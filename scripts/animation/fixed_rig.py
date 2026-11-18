"""Bind a static scan to an existing walking rig using calibrated joint positions."""

import copy
import hashlib

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from animation.surface import Surface
from roster.mesh_output import add, finish
from transfer.scan import transforms, transform_points


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_clip(src, dst, clip, offsets=None, edits=None):
    """Copy actual source keys; compensate translations for an adapted bind pose."""
    anim = copy.deepcopy(clip)
    edits = dict(edits or {})
    memo = {}
    samplers = []
    for channel in anim["channels"]:
        target = channel["target"]
        node, prop = target["node"], target["path"]
        name = src.doc["nodes"][node]["name"]
        sampler = dict(clip["samplers"][channel["sampler"]])
        for field in ("input", "output"):
            acc = sampler[field]
            offset = (
                (offsets or {}).get(node)
                if field == "output" and prop == "translation"
                else None
            )
            edit = (
                edits.pop(name, None)
                if field == "output" and prop == "rotation"
                else None
            )
            key = (acc, node if offset is not None or edit is not None else None, field)
            if key not in memo:
                values = src.array(acc).copy()
                if offset is not None:
                    if sampler.get("interpolation") == "CUBICSPLINE":
                        values[1::3] += offset
                    else:
                        values += offset
                if edit is not None:
                    if not np.array_equal(
                        src.array(
                            clip["samplers"][channel["sampler"]]["input"]
                        ).ravel(),
                        edit["times"],
                    ):
                        raise ValueError(
                            f"Rotation edit has different times for {name}"
                        )
                    updated = np.asarray(edit["values"], dtype="<f4")
                    if values.shape != updated.shape or not np.isfinite(updated).all():
                        raise ValueError(f"Invalid rotation edit for {name}")
                    values = updated
                index = add(
                    dst, values.astype("<f4"), src.doc["accessors"][acc]["type"]
                )
                view = dst.doc["accessors"][index]["bufferView"]
                dst.doc["bufferViews"][view].pop("target", None)
                if field == "input":
                    dst.doc["accessors"][index].update(
                        min=[float(values.min())], max=[float(values.max())]
                    )
                memo[key] = index
            sampler[field] = memo[key]
        channel["sampler"] = len(samplers)
        samplers.append(sampler)
    if edits:
        raise ValueError(f"Missing rotation tracks for {list(edits)}")
    anim["samplers"] = samplers
    return anim


def fit(scan, donor, config, output):
    for path, field in [(scan, "scan_sha256"), (donor, "rig_sha256")]:
        if digest(path) != config[field]:
            raise ValueError(f"Input hash mismatch for {field}")
    subject, reference = Surface(scan), Surface(donor)
    if len(subject.parts) != 1 or len(reference.parts) != 1:
        raise ValueError(
            "Use a single-surface static scan and a single-surface rig donor"
        )
    part, donor_part = subject.parts[0], reference.parts[0]
    if part.weights is not None or donor_part.weights is None:
        raise ValueError("Expected an unskinned scan and a skinned rig donor")
    g, ref = subject.scene.glb, reference.scene.glb
    xyz = transform_points(subject.rest[0], np.asarray(config["scan_to_rig_world"]))
    worlds, parents = transforms(ref.doc)
    joints = donor_part.bones
    nodes = copy.deepcopy(ref.doc["nodes"])
    targets = config["joint_positions_m"]
    goal = np.array([targets[nodes[i]["name"]] for i in joints])
    if goal.shape != (len(joints), 3) or not np.isfinite(goal).all():
        raise ValueError("Expected a finite three-dimensional position per joint")
    new = worlds.copy()
    new[joints, :3, 3] = goal
    # Intermediate helper nodes can sit between joints. Propagate each fitted
    # parent to these nodes before solving the next joint's local translation.
    for i in reference.scene.order:
        if (
            i not in joints
            and i in parents
            and not np.array_equal(new[parents[i]], worlds[parents[i]])
        ):
            parent = parents[i]
            new[i] = new[parent] @ np.linalg.inv(worlds[parent]) @ worlds[i]
    offsets = {}
    for i in joints:
        if "matrix" in nodes[i]:
            raise ValueError("Rig joints must use glTF TRS")
        parent = parents.get(i)
        local = np.linalg.inv(new[parent]) @ new[i] if parent is not None else new[i]
        offsets[i] = local[:3, 3] - nodes[i].get("translation", [0, 0, 0])
        nodes[i]["translation"] = local[:3, 3].tolist()
    # Warp only weight lookup coordinates into the donor's rest proportions.
    warp = RBFInterpolator(
        goal,
        worlds[joints, :3, 3] - goal,
        kernel="thin_plate_spline",
        smoothing=config["rbf_smoothing"],
    )
    if (
        not 1 <= config["neighbors"] <= len(reference.rest[0])
        or config["distance_floor_m"] <= 0
    ):
        raise ValueError(
            "Weight lookup requires a valid neighbor count and positive distance floor"
        )
    distances, nearest = cKDTree(reference.rest[0]).query(
        xyz + warp(xyz), k=config["neighbors"]
    )
    distances = distances.reshape(len(xyz), -1)
    nearest = nearest.reshape(len(xyz), -1)
    weights = (
        1
        / np.maximum(distances, config["distance_floor_m"]) ** config["distance_power"]
    )
    weights /= weights.sum(axis=1, keepdims=True)
    dense = np.zeros((len(donor_part.positions), len(joints)))
    for slots, values in zip(donor_part.palette.T, donor_part.weights.T):
        np.add.at(dense, (np.arange(len(dense)), slots), values)
    combined = np.einsum("ij,ijk->ik", weights, dense[nearest])
    for bias in config.get("height_weight_ramps", []):
        start, end = bias["range_m"]
        if end <= start:
            raise ValueError("A height ramp requires increasing limits")
        blend = np.clip((xyz[:, 1] - start) / (end - start), 0, 1)
        combined *= 1 - blend[:, None]
        combined[:, list(joints).index(reference.scene.joint(bias["joint"]))] += blend
    indices = np.argsort(-combined, axis=1)[:, :4]
    weights = np.take_along_axis(combined, indices, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    mesh_node = donor_part.node
    mesh_world = new[mesh_node]
    attrs = part.prim["attributes"]
    attrs["POSITION"] = add(
        g, transform_points(xyz, np.linalg.inv(mesh_world)).astype("<f4"), "VEC3"
    )
    # Apply the same change of coordinates to direction attributes.
    scan_world = subject.scene.world(subject.scene.rest)[part.node]
    linear = (np.linalg.inv(mesh_world) @ config["scan_to_rig_world"] @ scan_world)[
        :3, :3
    ]
    for label in ("NORMAL", "TANGENT"):
        if label in attrs:
            vec = g.array(attrs[label]).copy()
            matrix = np.linalg.inv(linear) if label == "NORMAL" else linear.T
            directions = vec[:, :3] @ matrix
            vec[:, :3] = directions / np.maximum(
                np.linalg.norm(directions, axis=1)[:, None], 1e-20
            )
            if label == "TANGENT":
                vec[:, 3] *= np.sign(np.linalg.det(linear))
            attrs[label] = add(g, vec, "VEC3" if label == "NORMAL" else "VEC4")
    attrs["JOINTS_0"] = add(g, indices.astype("<u2"), "VEC4", 5123)
    attrs["WEIGHTS_0"] = add(g, weights.astype("<f4"), "VEC4")
    ib = np.linalg.inv(new[joints]) @ mesh_world
    skin = dict(
        joints=joints.tolist(),
        inverseBindMatrices=add(g, ib.transpose(0, 2, 1).astype("<f4"), "MAT4"),
    )
    nodes[mesh_node].update(
        mesh=g.doc["nodes"][part.node]["mesh"], skin=0, name=config["mesh_name"]
    )
    g.doc.update(
        nodes=nodes,
        skins=[skin],
        scenes=copy.deepcopy(ref.doc["scenes"]),
        scene=ref.doc.get("scene", 0),
    )
    clip = next(
        a for a in ref.doc["animations"] if a.get("name") == config["source_clip"]
    )
    if any(c["target"]["path"] == "weights" for c in clip["channels"]):
        raise ValueError(
            "The rig source clip must animate the skeleton; bind scan morphs during assembly"
        )
    anim = copy_clip(ref, g, clip, offsets, config.get("rotation_track_edits"))
    anim["name"] = config["output_clip"]
    g.doc["animations"] = [anim]
    finish(g, output)
