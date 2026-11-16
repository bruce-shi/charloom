"""Attach rigid pieces and bind animated strips or panels to a calibrated garment."""

import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from animation.articulation import between, unit
from transfer.scan import transforms, transform_points

FABRICS = {
    "woven": (0.065, 0.65, 0.24),
    "silk": (0.085, 0.74, 0.30),
    "chiffon": (0.105, 0.84, 0.36),
    "tulle": (0.078, 0.57, 0.21),
    "organza": (0.066, 0.49, 0.17),
    "lace": (0.075, 0.64, 0.25),
    "satin": (0.084, 0.66, 0.26),
    "knit": (0.072, 0.58, 0.23),
    "denim": (0.052, 0.40, 0.14),
    "wool": (0.078, 0.45, 0.17),
    "velvet": (0.090, 0.51, 0.19),
    "leather": (0.045, 0.30, 0.10),
}


def fabric_motion(fabric):
    delay, gain, limit = (
        FABRICS[fabric]
        if isinstance(fabric, str)
        else (fabric["delay_s"], fabric["gain"], fabric["max_angle_rad"])
    )
    if (
        not np.isfinite([delay, gain, limit]).all()
        or delay <= 0
        or gain < 0
        or not 0 <= limit <= np.pi
    ):
        raise ValueError("Invalid fabric motion parameters")
    return delay, gain, limit


def preset_path(library, key):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", key):
        raise ValueError(f"Invalid garment preset id {key!r}")
    return library / (key + ".json")


def validate_rig(rig):
    mode = rig["mode"]
    if mode not in ("body", "rigid", "ribbon", "skirt", "panel", "cape", "veil"):
        raise ValueError(f"Unknown garment rig {mode}")
    axis = unit(rig.get("direction", [0.0, -1.0, 0.0]))
    if mode == "panel":
        width = rig.get("width_axis", [1.0, 0.0, 0.0])
        unit(width - axis * np.dot(width, axis))
    if mode == "skirt":
        cut = rig.get("cut", "flared")
        n = (
            {"flared": 4, "bell": 6, "pleated": 8}[cut]
            if isinstance(cut, str)
            else cut["panels"]
        )
        if not isinstance(n, int) or not 3 <= n <= 255:
            raise ValueError("Invalid skirt panel count")


def register_preset(library, data):
    key = data["id"]
    validate_rig(data["rig"])
    fabric_motion(data.get("fabric", "woven"))
    library.mkdir(parents=True, exist_ok=True)
    path = preset_path(library, key)
    with tempfile.NamedTemporaryFile(mode="w", dir=library, delete=False) as f:
        json.dump(data, f, indent=2)
    try:
        try:
            os.link(f.name, path)
        except FileExistsError:
            if json.loads(path.read_text()) != data:
                raise FileExistsError(f"A different garment preset already uses {key}")
    finally:
        os.unlink(f.name)
    return path


def rig_for(item, library):
    preset = {}
    if key := item.get("preset"):
        preset = json.loads(preset_path(library, key).read_text())
        if preset["id"] != key:
            raise ValueError("Garment preset id differs from its filename")
    rig = item.get("rig") or preset.get("rig")
    if not rig:
        name = item["name"].lower()
        if name == "helmet":
            rig = dict(mode="rigid", bone="head")
        elif item.get("rigid"):
            rig = dict(mode="rigid")
        elif name in ("skirt", "ribbon"):
            rig = dict(mode=name)
            if name == "ribbon" and item.get("label") == "hair":
                rig["bone"] = "head"
        else:
            rig = dict(mode="body")
    validate_rig(rig)
    fabric = item.get("fabric", preset.get("fabric", "woven"))
    fabric_motion(fabric)
    return rig, fabric


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def bind(g, primitives, skin_id, name, rig, fabric):
    from roster.mesh_output import add

    mode = rig["mode"]
    if mode == "body":
        return []
    skin = g.doc["skins"][skin_id]
    nodes = g.doc["nodes"]
    world, _ = transforms(g.doc)
    bone_names = [nodes[i]["name"] for i in skin["joints"]]
    attrs = [p["attributes"] for p in primitives]
    parts = [g.array(a["POSITION"]).copy() for a in attrs]
    all_points = np.vstack(parts)
    points, rows, remap = np.unique(
        all_points, axis=0, return_index=True, return_inverse=True
    )
    axis = unit(rig.get("direction", [0, -1, 0]))
    heights = points @ axis
    lo, span = heights.min(), np.ptp(heights)
    if mode != "rigid" and span < 1e-7:
        raise ValueError(f"Garment {name} has no length along the hanging axis")
    depth = (heights - lo) / max(span, 1e-9)
    mount_name = rig.get("bone") or {
        "cape": "spine01",
        "veil": "head",
        "skirt": "root",
    }.get(mode)
    if mount_name:
        mount = bone_names.index(mount_name)
    else:
        selected = rows if mode == "rigid" else rows[depth < 0.15]
        j = np.vstack([g.array(a["JOINTS_0"]) for a in attrs])[selected]
        w = np.vstack([g.array(a["WEIGHTS_0"]) for a in attrs])[selected]
        mount = int(np.bincount(j.ravel(), weights=w.ravel()).argmax())
    chain_info = []
    if mode == "rigid":
        joint = np.zeros((len(points), 4), dtype="<u2")
        joint[:, 0] = mount
        weights = np.zeros((len(points), 4), dtype="<f4")
        weights[:, 0] = 1.0
    else:
        center = points.mean(0)
        if mode == "skirt":
            cut = rig.get("cut", "flared")
            count = (
                {"flared": 4, "bell": 6, "pleated": 8}[cut]
                if isinstance(cut, str)
                else cut["panels"]
            )
            radial = points - center
            sectors = (
                np.mod(np.arctan2(radial[:, 2], radial[:, 0]), 2 * np.pi)
                * count
                / (2 * np.pi)
            )
        elif mode == "ribbon":
            count = 1
            sectors = np.zeros(len(points))
        else:
            count = 3
            width = np.asarray(rig.get("width_axis", [1.0, 0.0, 0.0]))
            col = points @ unit(width - axis * (width @ axis))
            if np.ptp(col) < 1e-7:
                raise ValueError(f"Garment {name} has no width")
            sectors = (col - col.min()) / np.ptp(col) * (count - 1)
        k = np.floor(sectors).astype(int) % count
        frac = sectors - np.floor(sectors)
        panel_weights = np.zeros((len(points), count))
        panel_weights[np.arange(len(points)), k] = 1 - frac
        panel_weights[np.arange(len(points)), (k + 1) % count] += frac
        matrix = world[skin["joints"][mount]]
        inverse = np.linalg.inv(matrix)
        matrices = list(g.array(skin["inverseBindMatrices"]).copy())
        mix = np.zeros((len(points), 1 + 2 * count))
        mobile = smooth((depth - 0.12) / 0.29)
        lower = smooth((depth - 0.5) / 0.32)
        mix[:, 0] = 1 - mobile
        slots = [mount]
        for panel in range(count):
            pivots = []
            for station in (0.16, 0.58, 1.0):
                factors = (
                    panel_weights[:, panel] / (0.09 + np.abs(depth - station)) ** 4
                )
                point = (
                    np.average(points, weights=factors, axis=0)
                    if factors.sum() > 0
                    else center.copy()
                )
                point += axis * (lo + station * span - point @ axis)
                pivots.append(transform_points(point, inverse))
            parent = skin["joints"][mount]
            ids = []
            for level, pivot in enumerate(pivots[:2]):
                ident = f"cloth_{name}_{panel}_{level}"
                if ident in bone_names:
                    raise ValueError(f"Duplicate cloth joint {ident}")
                node_id = len(nodes)
                nodes.append(
                    dict(
                        name=ident,
                        translation=(
                            pivot if level == 0 else pivot - pivots[0]
                        ).tolist(),
                    )
                )
                nodes[parent].setdefault("children", []).append(node_id)
                slots.append(len(skin["joints"]))
                skin["joints"].append(node_id)
                posed = matrix.copy()
                posed[:3, 3] = transform_points(pivot, matrix)
                matrices.append(np.linalg.inv(posed).T.ravel())
                ids.append(node_id)
                parent = node_id
            mix[:, 1 + 2 * panel] = mobile * (1 - lower) * panel_weights[:, panel]
            mix[:, 2 + 2 * panel] = mobile * lower * panel_weights[:, panel]
            chain_info.append(
                dict(
                    mount=skin["joints"][mount],
                    joints=ids,
                    pivots=np.array(pivots).tolist(),
                    fabric=fabric,
                )
            )
        skin["inverseBindMatrices"] = add(g, np.array(matrices, dtype="<f4"), "MAT4")
        top = np.argsort(-mix, axis=1, kind="stable")[:, :4]
        joint = np.asarray(slots, dtype="<u2")[top]
        weights = np.take_along_axis(mix, top, axis=1)
        weights /= weights.sum(1, keepdims=True)
        pad = 4 - weights.shape[1]
        if pad:
            weights = np.pad(weights, ((0, 0), (0, pad)))
            joint = np.pad(joint, ((0, 0), (0, pad)))
    start = 0
    for part, attr in zip(parts, attrs):
        selected = remap[start : start + len(part)]
        attr["JOINTS_0"] = add(g, joint[selected].astype("<u2"), "VEC4", 5123)
        attr["WEIGHTS_0"] = add(g, weights[selected].astype("<f4"), "VEC4")
        start += len(part)
    return chain_info


def bake(model, clip):
    from animation.scene import Scene
    from roster.write_face_clips import write_clips

    scene = Scene(model)
    chains = scene.glb.doc.get("extras", {}).get("charloomClothChains", [])
    if not chains:
        return
    start, end = scene.interval(clip)
    if end <= start:
        return
    times = np.linspace(start, end, 1 + int(np.ceil((end - start) * 30)))
    tracks = []
    for chain in chains:
        delay, gain, limit = fabric_motion(chain["fabric"])
        pivots = np.array(chain["pivots"])
        rows = [[], []]
        for t in times:
            now = scene.world(scene.sample(clip, t)[0])[chain["mount"]]
            upper = Rotation.identity()
            for level in (0, 1):
                prev = scene.world(
                    scene.sample(clip, max(start, t - delay * (level + 1)))[0]
                )[chain["mount"]]
                probe = pivots[level + 1]
                lag = transform_points(probe, np.linalg.solve(now, prev)) - probe
                direction = probe - pivots[level]
                rot = between(direction, direction + gain * lag).as_rotvec()
                fade = smooth((t - start) / delay) * smooth((end - t) / delay)
                q = Rotation.from_rotvec(
                    rot * min(1.0, limit / max(np.linalg.norm(rot), 1e-12)) * fade
                )
                rows[level].append((upper.inv() * q).as_quat())
                upper = q
        tracks.extend(
            dict(joint=j, property="rotation", times=times, values=row)
            for j, row in zip(chain["joints"], rows)
        )
    write_clips(model, [dict(name=clip, tracks=tracks, morph_tracks=[])], extend=True)
