"""Reference-pose rotation and root-motion transfer onto a target GLB."""

import re
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roster.mesh import Glb
from .scan import transforms


def key(name):
    token = re.sub(r"[^a-z0-9]", "", name.rsplit(":", 1)[-1].lower())
    if token.startswith(("hcleft", "hcright")):
        token = token[2:]
    token = token.replace("little", "pinky")
    if token in ("hips", "pelvis", "root"):
        return "pelvis"
    if token == "neck01":
        return "neck"
    side = (
        "left"
        if "left" in token or name.lower().endswith((".l", "_l"))
        else (
            "right" if "right" in token or name.lower().endswith((".r", "_r")) else ""
        )
    )
    if not side:
        return token
    stem = token.replace(side, "") if side in token else token[:-1]
    if stem.startswith("hand") and len(stem) > 4:
        stem = stem[4:]
    match = re.fullmatch(r"finger([1-5])([1-3])", stem)
    if match:
        stem = ("thumb", "index", "middle", "ring", "pinky")[int(match[1]) - 1] + match[
            2
        ]
    stem = re.sub(r"^(thumb|index|middle|ring|pinky)0?(\d+)(?:leaf)?$", r"\1\2", stem)
    aliases = {
        "upperleg01": "thigh",
        "upleg": "thigh",
        "lowerleg01": "shin",
        "leg": "shin",
        "calf": "shin",
        "upperarm01": "upperarm",
        "arm": "upperarm",
        "lowerarm01": "forearm",
        "lowerarm": "forearm",
        "wrist": "hand",
        "shoulder01": "clavicle",
        "shoulder": "clavicle",
        "toe11": "toe",
        "toebase": "toe",
        "ball": "toe",
        "ballleaf": "toeend",
        "toeend": "toeend",
    }
    return side + "_" + aliases.get(stem, stem)


def unit_rot(matrices):
    linear = matrices[..., :3, :3]
    return Rotation.from_matrix(
        linear / np.maximum(np.linalg.norm(linear, axis=-2, keepdims=True), 1e-20)
    )


def sample_motion(source, output, config, clip):
    fps = float(config.get("fps", 30))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if Path(source).suffix.lower() in (".glb", ".gltf"):
        from animation.scene import Scene

        scene = Scene(source)
        clip = clip or next(iter(scene.clips))
        lo, hi = scene.interval(clip)
        start, end = config.get("start",lo), config.get("end",hi)
        if not lo-1e-6<=start<=end<=hi+1e-6:
            raise ValueError("Motion trim exceeds the source clip")
        times = np.unique(np.r_[np.arange(start, end, 1 / fps), end].astype(np.float32))
        nodes = sorted(
            {i for skin in scene.glb.doc.get("skins", []) for i in skin["joints"]}
            or {i for i, name in enumerate(scene.names) if name}
        )
        np.savez(
            output,
            times=times - start,
            names=[scene.names[i] for i in nodes],
            rest=scene.world(scene.rest)[nodes],
            poses=np.stack(
                [scene.world(scene.sample(clip, t)[0])[nodes] for t in times]
            ),
        )
        return
    blender = (
        os.environ.get("BLENDER")
        or shutil.which("blender")
        or "/Applications/Blender.app/Contents/MacOS/Blender"
    )
    argv = [
        blender,
        "-b",
        "--factory-startup",
        "-t",
        "4",
        "--python-exit-code",
        "1",
        "--python",
        str(Path(__file__).with_name("motion_source.py")),
        "--",
        str(Path(source).resolve()),
        str(output),
        clip or "",
        str(config.get("fps", 30)),
        str(config.get("start", "")), str(config.get("end","")),
    ]
    import sys

    subprocess.run(argv, check=True, stdout=sys.stderr)


def map_joints(source_names, g, config):
    nodes = g.doc["nodes"]
    mapping = {}
    lookup = {key(n): i for i, n in enumerate(source_names)}
    target_joints = list(
        dict.fromkeys(i for skin in g.doc.get("skins", []) for i in skin["joints"])
    )
    if not target_joints:
        target_joints = [
            i for i, n in enumerate(nodes) if n.get("name") and "mesh" not in n
        ]
    overrides = config.get("joint_map", {})
    absent = set(overrides) - {nodes[i].get("name") for i in target_joints}
    if absent:
        raise ValueError(f"Unknown target joints in joint_map: {sorted(absent)}")
    src_spines = [
        i for i, n in enumerate(source_names) if re.fullmatch(r"spine\d*", key(n))
    ]
    target_spines = [
        i for i in target_joints if re.fullmatch(r"spine\d*", key(nodes[i]["name"]))
    ]
    for i in target_joints:
        n = nodes[i]["name"]
        if n in overrides:
            mapping[i] = source_names.index(overrides[n])
        elif n in source_names:
            mapping[i] = source_names.index(n)
        elif i in target_spines and src_spines:
            rank = target_spines.index(i)
            # MPFB has five spine segments. Extra segments inherit their parent.
            ranks = np.rint(
                np.linspace(0, len(target_spines) - 1, len(src_spines))
            ).astype(int)
            if rank in ranks:
                mapping[i] = src_spines[ranks.tolist().index(rank)]
        elif key(n) in lookup:
            mapping[i] = lookup[key(n)]
    scope = config.get("target_joint_scope")
    required = (
        set(target_joints)
        if scope is None
        else {i for i in target_joints if nodes[i].get("name") in scope}
    )
    if scope is not None and (not required or len(required) != len(scope)):
        raise ValueError("Each target_joint_scope entry must resolve to one joint")
    coverage = len(required.intersection(mapping)) / max(len(required), 1)
    minimum = config.get("min_target_coverage", 0.0)
    if not 0 <= minimum <= 1 or coverage < minimum:
        raise ValueError(f"Mapped coverage {coverage:.3f} below requirement {minimum}")
    if not mapping:
        raise ValueError("No compatible motion joints")
    return mapping, target_joints, coverage


def retarget(samples, model, name, config):
    from roster.write_face_clips import write_clips

    g = Glb(model)
    source_names = samples["names"].tolist()
    mapping, target_joints, coverage = map_joints(source_names, g, config)
    nodes = g.doc["nodes"]
    world, parents = transforms(g.doc)
    rest = samples["rest"]
    poses = samples["poses"]
    times = samples["times"]
    world_rot = {i: unit_rot(m) for i, m in enumerate(world)}
    alignment = Rotation.from_quat(config.get("alignment", [0, 0, 0, 1]))
    preserved = set(config.get("preserve_reference_joints", [])) | set(
        config.get("rest_local_reference_joints", [])
    )
    local_hold = set(config.get("preserve_local_pose_joints", []))
    for n in local_hold | preserved:
        if n not in {nodes[i].get("name") for i in target_joints}:
            raise ValueError(f"Unknown reference joint {n}")
    children_by_mapping = {i: [] for i in mapping}
    for i in mapping:
        parent = parents.get(i)
        while parent is not None and parent not in mapping:
            parent = parents.get(parent)
        if parent is not None:
            children_by_mapping[parent].append(i)
    delta = {}
    for i, j in mapping.items():
        if nodes[i].get("name") in local_hold:
            continue
        adjustment = Rotation.identity()
        if (
            config.get("align_reference_chains")
            and nodes[i].get("name") not in preserved
        ):
            children = children_by_mapping[i]
            if children:
                to = np.stack([world[k, :3, 3] - world[i, :3, 3] for k in children])
                src = alignment.apply(
                    np.stack(
                        [rest[mapping[k], :3, 3] - rest[j, :3, 3] for k in children]
                    )
                )
                valid = (np.linalg.norm(to, axis=1) > 1e-6) & (
                    np.linalg.norm(src, axis=1) > 1e-6
                )
                if valid.any():
                    adjustment, _ = Rotation.align_vectors(
                        src[valid] / np.linalg.norm(src[valid], axis=1)[:, None],
                        to[valid] / np.linalg.norm(to[valid], axis=1)[:, None],
                    )
        delta[i] = (
            alignment
            * unit_rot(poses[:, j])
            * unit_rot(rest[j]).inv()
            * alignment.inv()
            * adjustment
        )
    tracks = []
    for i in delta:
        j = mapping[i]
        parent = parents.get(i)
        ancestor = parent
        while ancestor is not None and ancestor not in delta:
            ancestor = parents.get(ancestor)
        # Local rotation = current parent world inverse times intended bone world rotation.
        pq = world_rot[parent] if parent is not None else Rotation.identity()
        dq = delta[ancestor] if ancestor is not None else Rotation.identity()
        rotation = ((dq * pq).inv() * delta[i] * world_rot[i]).as_quat()
        signs = np.r_[
            1.0,
            np.cumprod(
                np.where(np.sum(rotation[1:] * rotation[:-1], axis=1) < 0, -1.0, 1.0)
            ),
        ]
        rotation *= signs[:, None]
        tracks.append(
            dict(
                joint=i,
                property="rotation",
                times=times.tolist(),
                values=rotation.tolist(),
            )
        )
        if ancestor is None and config.get("root_motion", True):
            scale = config.get("root_scale")
            if scale is None:
                scale = np.ptp(world[list(mapping), 1, 3]) / max(
                    np.ptp(rest[list(mapping.values()), 1, 3]), 1e-8
                )
            move = alignment.apply(poses[:, j, :3, 3] - rest[j, :3, 3]) * scale
            # An in-place cycle removes net travel along selected world axes.
            # Retain each step's hip sway, bounce and authored starting offset.
            axes = config.get("in_place_axes", [])
            if not isinstance(axes, list) or any(axis not in ("x", "y", "z") for axis in axes):
                raise ValueError("in_place_axes must list world axes x, y or z")
            if axes and times[-1] > times[0]:
                columns = ["xyz".index(axis) for axis in set(axes)]
                phase = (times - times[0]) / (times[-1] - times[0])
                move[:, columns] -= phase[:, None] * (move[-1, columns] - move[0, columns])
            local = (
                np.linalg.solve(world[parent, :3, :3], move.T).T
                if parent is not None
                else move
            )
            translation = local + nodes[i].get("translation", [0.0, 0.0, 0.0])
            tracks.append(
                dict(
                    joint=i,
                    property="translation",
                    times=times.tolist(),
                    values=translation.tolist(),
                )
            )
    write_clips(model, [dict(name=name, tracks=tracks, morph_tracks=[])])
    peaks=[]
    if any(config.get(key) for key in ("limb_trajectories","wrist_twist_chains","distribute_wrist_twist","distribute_palm_twist")):
        from .trajectory import adapt
        scale = config.get("root_scale") or np.ptp(world[list(mapping),1,3])/max(np.ptp(rest[list(mapping.values()),1,3]),1e-8)
        peaks=adapt(model,name,samples,mapping,config,alignment,scale)
    if g.doc.get("extras", {}).get("charloomClothChains"):
        from .cloth import bake

        bake(model, name)
    return dict(
        name=name,
        coverage=len(mapping)/max(len(target_joints),1),
        scoped_coverage=coverage if config.get("target_joint_scope") is not None else None,
        max_trajectory_error_m=peaks,
        mapped_joints=len(mapping),
        samples=len(times),
        duration=float(times[-1]),
        unmapped_target=[nodes[i]["name"] for i in target_joints if i not in mapping],
    )
