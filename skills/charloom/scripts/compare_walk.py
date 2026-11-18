"""Compare the named walk tracks on two GLBs without depending on node ordering."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from animation.scene import Scene


def read_walk(path, clip):
    scene = Scene(path)
    if clip not in scene.clips:
        raise ValueError(f"{path}: expected {clip!r}, found {list(scene.clips)}")
    tracks = {}
    rig = {}
    for (node, prop), track in scene.clips[clip].items():
        name = scene.names[node]
        key = (name, prop)
        if key in tracks or not name:
            raise ValueError(f"{path}: ambiguous track {key}")
        tracks[key] = track
        while node is not None:
            name = scene.names[node]
            if not name or scene.names.count(name) != 1:
                raise ValueError(
                    f"{path}: expected a unique name for animated ancestor {node}"
                )
            parent = scene.parents.get(node)
            rig[name] = (
                scene.names[parent] if parent is not None else None,
                scene.rest[node],
            )
            node = parent
    if not tracks:
        raise ValueError(f"{path}: empty clip {clip!r}")
    return tracks, rig


def compare(before, after, source_clip, result_clip):
    base, rig = read_walk(before, source_clip)
    test, other = read_walk(after, result_clip)
    changed = []
    for key in sorted(base.keys() & test.keys()):
        t, v, mode = base[key]
        u, w, other_mode = test[key]
        if mode != other_mode or not np.array_equal(t, u) or not np.array_equal(v, w):
            changed.append(key)
    rest = [
        name
        for name in sorted(rig.keys() & other.keys())
        if rig[name][0] != other[name][0]
        or not np.array_equal(rig[name][1], other[name][1])
    ]
    differences = dict(
        missing_tracks=sorted(base.keys() - test.keys()),
        extra_tracks=sorted(test.keys() - base.keys()),
        changed_tracks=changed,
        changed_rest_or_parents=rest,
        missing_rig_nodes=sorted(rig.keys() - other.keys()),
        extra_rig_nodes=sorted(other.keys() - rig.keys()),
    )
    return dict(
        exact_match=not any(differences.values()),
        baseline_tracks=len(base),
        result_tracks=len(test),
        **differences,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--source-clip", default="walking")
    parser.add_argument("--result-clip", default="walking")
    args = parser.parse_args()
    report = compare(args.baseline, args.result, args.source_clip, args.result_clip)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["exact_match"] else 1)
