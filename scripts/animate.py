"""Inspect, sample, solve, retarget or bake a character animation from Python."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from animation.rig_identity import signature
from animation.keys import deduplicate
from animation.recipe import build
from animation.scene import Scene, SLICES
from animation.solve import solve
from roster.write_face_clips import write_clips
from transfer.motion import map_joints, retarget, sample_motion
from roster.mesh import Glb
from animation.surface import Surface


def inspect(scene):
    return {
        "nodes": scene.names,
        "skins": [len(s["joints"]) for s in scene.glb.doc.get("skins", [])],
        "clips": {name: scene.interval(name) for name in scene.clips},
    }


def validate(scene):
    for name, channels in scene.clips.items():
        for (idx, prop), (times, values, mode) in channels.items():
            if not (
                0 <= idx < len(scene.names)
                and prop in {*SLICES, "weights"}
                and mode in ("STEP", "LINEAR", "CUBICSPLINE")
                and np.isfinite(times).all()
                and np.all(np.diff(times) > 0)
                and np.isfinite(values).all()
            ):
                raise ValueError(f"Invalid channel {name}/{idx}/{prop}")
        lo, hi = scene.interval(name)
        for t in [lo, (hi + lo) / 2, hi]:
            if not np.isfinite(scene.world(scene.sample(name, t)[0])).all():
                raise ValueError(f"Non-finite pose in {name}")
    return {"clips_checked": list(scene.clips), "nodes": len(scene.names)}


def read_json(path, default):
    return json.loads(path.read_text()) if path is not None else default


def run(args):
    task = args.operation
    if task == "map":
        source = args.source
        if source.suffix.lower() == ".fbx":
            with tempfile.TemporaryDirectory() as tmp:
                take = Path(tmp) / "motion.npz"
                sample_motion(source, take, {"fps": 1}, None)
                with np.load(take) as result:
                    names = result["names"].tolist()
        else:
            names = Scene(source).names
        target = Glb(args.asset)
        mapping, joints, coverage = map_joints(
            names, target, read_json(args.config, {})
        )
        return dict(
            mapping={
                target.doc["nodes"][i]["name"]: names[j] for i, j in mapping.items()
            },
            coverage=len(mapping)/max(len(joints),1),
            scoped_coverage=coverage if args.config and read_json(args.config,{}).get("target_joint_scope") is not None else None,
            unmapped_target=[
                target.doc["nodes"][i]["name"] for i in joints if i not in mapping
            ],
        )
    if task in ("inspect", "validate", "sample"):
        if args.asset.suffix.lower() == ".fbx":
            with tempfile.TemporaryDirectory() as tmp:
                take = Path(tmp) / "motion.npz"
                sample_motion(
                    args.asset, take, {"fps": 30}, getattr(args, "clip", None)
                )
                with np.load(take) as data:
                    names, times = data["names"].tolist(), data["times"]
                    if not np.isfinite(data["poses"]).all():
                        raise ValueError("Non-finite FBX pose")
                    if task != "sample":
                        return dict(
                            nodes=names,
                            clips=data["clips"].tolist(),
                            sampled_clip=str(data["clip"]),
                            duration=float(times[-1]),
                        )
                    from animation.scene import decompose, blend_quats, matrices

                    i = int(
                        np.clip(
                            np.searchsorted(times, args.time, side="right") - 1,
                            0,
                            len(times) - 1,
                        )
                    )
                    j = min(i + 1, len(times) - 1)
                    u = (
                        np.clip((args.time - times[i]) / (times[j] - times[i]), 0, 1)
                        if j > i
                        else 0
                    )
                    rows = []
                    for ma, mb in zip(data["poses"][i], data["poses"][j]):
                        a, b = decompose(ma), decompose(mb)
                        pose = a + u * (b - a)
                        pose[3:7] = blend_quats(a[3:7], b[3:7], u)
                        rows.append(pose)
                    world = matrices(np.array(rows))
                    return dict(time=args.time, names=names, world=world.tolist())
        scene = Scene(args.asset)
        if task == "inspect":
            return inspect(scene)
        if task == "validate":
            return validate(scene)
        pose, shapes = scene.sample(args.clip, args.time)
        result = dict(
            time=args.time,
            names=scene.names,
            world=scene.world(pose).tolist(),
            morphs={str(n): w.tolist() for n, w in shapes.items()},
        )
        if args.surface:
            surface = Surface(args.asset)
            sampled = surface.evaluate(pose, shapes)
            meshes = []
            for part, points in zip(surface.parts, sampled):
                idx = np.unique(
                    np.linspace(0, len(points) - 1, min(32, len(points)), dtype=int)
                )
                mesh_id = scene.glb.doc["nodes"][part.node]["mesh"]
                prim_id = scene.glb.doc["meshes"][mesh_id]["primitives"].index(
                    part.prim
                )
                meshes.append(
                    dict(
                        node=part.node,
                        primitive=prim_id,
                        vertex_count=len(points),
                        samples=[
                            dict(index=int(i), position=points[i].tolist()) for i in idx
                        ],
                    )
                )
            result["meshes"] = meshes
        return result
    dest = args.out.resolve()
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        scratch = Path(tmp)
        folder = scratch / "result"
        folder.mkdir()
        model = folder / "character.glb"
        if task == "build":
            recipe = read_json(args.recipe, None)
            base = args.recipe.resolve().parent
            Glb(base / recipe["asset"]).write(model)
            result = build(model, recipe, base, scratch)
        elif task == "roundtrip":
            Glb(args.asset).write(model)
            result = inspect(Scene(model))
        elif task == "retarget":
            original = Glb(args.asset)
            fixed_rig = signature(original)
            original.write(model)
            cfg = read_json(args.config, {})
            samples = scratch / "source.npz"
            sample_motion(args.source, samples, cfg, args.clip)
            with np.load(samples) as take:
                result = retarget(take, model, cfg.get("name", args.source.stem), cfg)
        else:
            Glb(args.asset).write(model)
            scene = Scene(model)
            initial = scene.sample(args.clip, args.time)[0] if args.clip else scene.rest
            solved, reports = solve(
                scene,
                initial,
                read_json(args.effectors, []),
                read_json(args.config, {}),
            )
            tracks = [
                dict(joint=i, property=prop, times=[0.0], values=[solved[i, col]])
                for i in range(len(solved))
                for prop, col in SLICES.items()
                if not np.allclose(solved[i, col], scene.rest[i, col], atol=1e-9)
            ]
            if tracks:
                write_clips(
                    model, [dict(name="solved", tracks=tracks, morph_tracks=[])]
                )
            result = dict(effectors=reports, pose=solved.tolist())
        result["deduplicated_samplers"] = deduplicate(model)
        if task == "retarget":
            if signature(Glb(model)) != fixed_rig:
                raise ValueError("Retarget changed the target skeleton or skin binding")
            result["fixed_target_rig"] = fixed_rig
        result.update(file=str(dest / model.name), validation=validate(Scene(model)))
        (folder / "report.json").write_text(json.dumps(result, indent=2))
        folder.rename(dest)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for op in (
        "inspect",
        "validate",
        "sample",
        "solve",
        "map",
        "roundtrip",
        "retarget",
        "build",
    ):
        p = sub.add_parser(op)
        if op in ("retarget", "map"):
            p.add_argument("source", type=Path)
        p.add_argument("recipe" if op == "build" else "asset", type=Path)
        if op in ("sample", "solve", "retarget"):
            p.add_argument("--clip", required=op == "sample")
        if op == "sample":
            p.add_argument("--surface", action="store_true")
        if op in ("sample", "solve"):
            p.add_argument("--time", type=float, default=0.0)
        if op in ("solve", "retarget", "map"):
            p.add_argument("--config", type=Path)
        if op == "solve":
            p.add_argument("--effectors", type=Path, required=True)
        if op in ("solve", "retarget", "build", "roundtrip"):
            p.add_argument("--out", required=True, type=Path)
    print(json.dumps(run(parser.parse_args())))
