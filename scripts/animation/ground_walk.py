"""Place a locomotion clip against its dressed contact surfaces.

Only the selected root translation is replaced. The exact same local root track
is written to every compatible layer; body shapes, rest poses, horizontal motion,
all joint rotations and other clips remain unchanged. This is not jump landing,
foot IK, or a fix for sliding contact feet.
"""

from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import numpy as np

from .surface import Surface
from roster.write_face_clips import write_clips


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _descends(scene, node, root):
    while node != root and node in scene.parents:
        node = scene.parents[node]
    return node == root


def _contact(surface, names, root):
    ids = [
        i for i, p in enumerate(surface.parts) if surface.scene.names[p.node] in names
    ]
    found = {surface.scene.names[surface.parts[i].node] for i in ids}
    if not names or found != set(names):
        raise ValueError(f"Contact mesh names not found: {set(names) - found}")
    for i in ids:
        p = surface.parts[i]
        if p.weights is None:
            if not _descends(surface.scene, p.node, root):
                raise ValueError(
                    "Unskinned contact mesh is outside chosen root hierarchy"
                )
        else:
            used = np.unique(p.bones[p.palette][p.weights > 1e-9])
            if any(not _descends(surface.scene, int(j), root) for j in used):
                raise ValueError(
                    "Contact mesh weights include joints outside chosen root"
                )
    return [(surface.parts[i], np.unique(surface.parts[i].faces)) for i in ids]


def _bottom(surface, ids, clip, time):
    pose, shapes = surface.scene.sample(clip, time)
    world = surface.scene.world(pose)
    return min(
        float(
            part.points(
                world,
                shapes.get(part.node, surface.scene.morph_rest(part.node)),
                vertices,
            )[:, 1].min()
        )
        for part, vertices in ids
    )


def ground_walk(
    reference,
    variants,
    out,
    *,
    contact_meshes,
    clip="walking",
    root="root",
    motion_kind,
    ground_y=0.0,
    sample_hz=120.0,
    tolerance_mm=0.5,
):
    """Write new reference/variant files under out, never mutate inputs.

    ``variants`` is {relative_output_path: source_path}, including the dressed
    reference itself. All variants must have the same root transform and root
    motion. Walking uses continuous ground contact. Running preserves airborne
    clearance and lifts the character only during ground penetration.
    """
    if motion_kind not in {"walking", "running"}:
        raise ValueError("Choose the reviewed locomotion type: walking or running")
    if (
        not np.isfinite([ground_y, sample_hz, tolerance_mm]).all()
        or sample_hz <= 0
        or tolerance_mm <= 0
    ):
        raise ValueError("Finite ground and positive sample_hz/tolerance_mm required")
    reference = Path(reference).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    surface = Surface(reference)
    scene = surface.scene
    root_id = scene.joint(root)
    if clip not in scene.clips:
        raise ValueError(f"Missing walking clip {clip}")
    ids = _contact(surface, contact_meshes, root_id)
    channels = scene.clips[clip]
    parent = scene.parents.get(root_id)
    ancestors = []
    n = parent
    while n is not None:
        ancestors.append(n)
        n = scene.parents.get(n)
    if any(node in ancestors and prop != "weights" for node, prop in channels):
        raise ValueError(
            "Animated root parent is unsupported; choose a stable common root"
        )
    track = channels.get((root_id, "translation"))
    if track is not None and track[2] != "LINEAR":
        raise ValueError(
            "Preserving horizontal motion requires LINEAR root translation"
        )
    rest_world = scene.world(scene.rest)
    parent_basis = np.eye(3) if parent is None else rest_world[parent, :3, :3]
    local_up = np.linalg.solve(parent_basis, np.array([0.0, 1.0, 0.0]))
    lo, hi = scene.interval(clip)
    if hi <= lo:
        raise ValueError("Walking clip must have positive duration")
    times = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) * sample_hz)) + 1))
    if track is not None:
        times = np.r_[times, track[0]]
    times = np.unique(times.astype("<f4")).astype(float)
    # Validate compatibility before creating any output. A shared track must not
    # silently place a layer's differently transformed skeleton elsewhere.
    prepared = []
    for relative, source in variants.items():
        relative = Path(relative)
        source = Path(source).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() != ".glb"
        ):
            raise ValueError("Output paths must stay under out")
        layer = Surface(source)
        ls = layer.scene
        lr = ls.joint(root)
        if clip not in ls.clips or not np.allclose(
            ls.interval(clip), [lo, hi], atol=1e-6
        ):
            raise ValueError("Layer clip interval differs")
        lp = ls.parents.get(lr)
        lw = ls.world(ls.rest)
        lb = np.eye(3) if lp is None else lw[lp, :3, :3]
        if not np.allclose(lb, parent_basis, atol=1e-6) or not np.allclose(
            ls.rest[lr], scene.rest[root_id], atol=1e-6
        ):
            raise ValueError("Layer root rest/parent differs")
        layer_channels = ls.clips[clip]
        layer_track = layer_channels.get((lr, "translation"))
        if (track is None) != (layer_track is None) or (
            track is not None
            and (
                track[2] != layer_track[2]
                or not np.array_equal(track[0], layer_track[0])
                or not np.array_equal(track[1], layer_track[1])
            )
        ):
            raise ValueError("Layer original root motion differs")
        if any(
            node != lr and prop != "weights" and _descends(ls, lr, node)
            for node, prop in layer_channels
        ):
            raise ValueError("Layer root parent is animated")
        # Every visible layer part must follow this root, including preserved head.
        for p in layer.parts:
            used = (
                [p.node]
                if p.weights is None
                else np.unique(p.bones[p.palette][p.weights > 1e-9])
            )
            if any(not _descends(ls, int(j), lr) for j in used):
                raise ValueError(f"Layer {relative} contains surface outside root")
        prepared.append((relative, source, layer, lr))
    if reference not in [p[1] for p in prepared]:
        raise ValueError("Variants must include dressed reference")
    cache = {}

    def bottom(t):
        t = float(t)
        if t not in cache:
            cache[t] = _bottom(surface, ids, clip, t)
        return cache[t]

    def target_offset(t):
        offset = ground_y - bottom(t)
        return max(0.0, offset) if motion_kind == "running" else offset

    for refinement in range(6):
        offsets = np.array([target_offset(t) for t in times])
        values = (
            np.array([scene.sample(clip, t)[0][root_id, :3] for t in times])
            + offsets[:, None] * local_up
        )
        # Check quarter/midpoints, not only baked keys. Add violating samples and
        # repeat rather than calling a key-only result grounded.
        probes = np.unique(
            np.r_[times, *(times[:-1] + f * np.diff(times) for f in [0.25, 0.5, 0.75])]
        )
        residual = np.array(
            [np.interp(t, times, offsets) - target_offset(t) for t in probes]
        )
        bad = np.abs(residual) > tolerance_mm / 1000
        if not bad.any():
            break
        times = np.unique(np.r_[times, probes[bad]].astype("<f4")).astype(float)
        if len(times) > 20000:
            raise ValueError("Ground track exceeded key budget")
    else:
        raise ValueError("Ground contact did not converge within requested tolerance")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staging = Path(tmp) / "result"
        staging.mkdir()
        outputs = []
        for relative, source, layer, lr in prepared:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            write_clips(
                target,
                [
                    dict(
                        name=clip,
                        tracks=[
                            dict(
                                joint=lr,
                                property="translation",
                                times=times.tolist(),
                                values=values.tolist(),
                            )
                        ],
                        morph_tracks=[],
                    )
                ],
                extend=True,
            )
            result = Surface(target)
            rs = result.scene
            preserved = True
            max_horizontal = 0.0
            max_root_difference = 0.0
            for cname, cchannels in layer.scene.clips.items():
                for key, original in cchannels.items():
                    if cname == clip and key == (lr, "translation"):
                        continue
                    now = rs.clips[cname].get(key)
                    if (
                        now is None
                        or original[2] != now[2]
                        or not np.array_equal(original[0], now[0])
                        or not np.array_equal(original[1], now[1])
                    ):
                        preserved = False
            if not preserved:
                raise RuntimeError("A non-root animation channel changed")
            for t in probes:
                before = layer.scene.world(layer.scene.sample(clip, t)[0])[lr, :3, 3]
                after = rs.world(rs.sample(clip, t)[0])[lr, :3, 3]
                max_horizontal = max(
                    max_horizontal, float(np.max(np.abs((after - before)[[0, 2]])))
                )
                expected = np.array(
                    [np.interp(t, times, values[:, axis]) for axis in range(3)]
                )
                max_root_difference = max(
                    max_root_difference,
                    float(np.max(np.abs(rs.sample(clip, t)[0][lr, :3] - expected))),
                )
            if max_horizontal > 1.0e-6 or max_root_difference > 1.0e-6:
                raise RuntimeError(
                    "Encoded root track differs across layers or changes horizontal motion"
                )
            record = dict(
                output=str(out / relative),
                input=str(source),
                input_sha256=_sha(source),
                sha256=_sha(target),
                non_root_channels_unchanged=preserved,
                max_horizontal_change_mm=max_horizontal * 1000,
                max_shared_root_track_difference_mm=max_root_difference * 1000,
            )
            if source == reference:
                rids = _contact(result, contact_meshes, lr)
                actual = np.array(
                    [_bottom(result, rids, clip, t) - ground_y for t in probes]
                )
                record.update(
                    contact_min_mm=float(actual.min() * 1000),
                    contact_max_mm=float(actual.max() * 1000),
                    contact_max_abs_mm=float(np.abs(actual).max() * 1000),
                )
                desired = np.array([bottom(t) + target_offset(t) - ground_y for t in probes])
                record["max_height_error_mm"] = float(np.abs(actual - desired).max() * 1000)
                if record["max_height_error_mm"] > tolerance_mm + 0.02:
                    raise RuntimeError("Encoded contact track exceeds tolerance")
            outputs.append(record)
        report = dict(
            operation="explicit_walking_root_ground" if motion_kind == "walking" else "running_ground_penetration_correction",
            motion_kind=motion_kind,
            clip=clip,
            root=root,
            contact_meshes=contact_meshes,
            ground_y_m=ground_y,
            sample_hz=sample_hz,
            tolerance_mm=tolerance_mm,
            keys=len(times),
            validation_samples=len(probes),
            refinement_rounds=refinement,
            duration_s=hi - lo,
            input_contact_range_mm=[
                float(min(cache.values()) * 1000),
                float(max(cache.values()) * 1000),
            ],
            offset_range_mm=[float(offsets.min() * 1000), float(offsets.max() * 1000)],
            outputs=outputs,
            limitations=[
                "Boot sole contact surface only; bare feet keep the identical world motion and may sit above ground by sole thickness.",
                "Preserves horizontal trajectory; does not solve foot sliding, leg IK or jumping.",
            ],
        )
        (staging / "ground-walk.json").write_text(json.dumps(report, indent=2))
        staging.rename(out)
        return report
