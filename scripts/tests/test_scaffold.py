"""Exercise fitting across GLB buffer reallocations in a fresh process."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roster.mesh import Glb
from roster.mesh_output import add, idx, normals
from transfer.scaffold import fit_template


def run_fit():
    g = Glb.__new__(Glb)
    g.blob = bytearray()
    g.doc = dict(accessors=[], bufferViews=[], buffers=[{"byteLength": 0}])
    locations = {
        "neck01": [0, 0, 0],
        "eye.L": [0.05, 0.15, 0.04],
        "eye.R": [-0.05, 0.15, 0.04],
        "oris01": [0, 0.075, 0.08],
        "oris05": [0, 0.065, 0.08],
    }
    g.doc["nodes"] = [{"name": "root", "children": list(range(1, 8))}]
    g.doc["nodes"] += [
        {"name": name, "translation": pos} for name, pos in locations.items()
    ]
    inverse = np.repeat(np.eye(4)[None], 5, axis=0)
    inverse[:, :3, 3] = -np.array(list(locations.values()))
    g.doc["skins"] = [{
        "joints": list(range(1, 6)),
        "inverseBindMatrices": add(g, inverse.transpose(0, 2, 1).reshape(-1, 16).astype("<f4"), "MAT4"),
    }]
    g.doc["meshes"] = []
    rng = np.random.default_rng(12)
    for name, count in [("body", 30000), ("teeth", 300)]:
        positions = rng.normal(size=(count, 3)).astype("<f4") * 0.02
        faces = np.arange(count, dtype=np.uint32).reshape(-1, 3)
        primitive = {
            "attributes": {
                "POSITION": add(g, positions, "VEC3"),
                "NORMAL": add(g, normals(positions, faces)[0].astype("<f4"), "VEC3"),
            },
            "indices": idx(g, faces),
        }
        g.doc["nodes"].append({"name": name, "mesh": len(g.doc["meshes"])})
        g.doc["meshes"].append({"name": name, "primitives": [primitive]})
    resolved = {
        "joint_section_depth": {},
        "landmarks": {
            "neck": [0, 0, 0],
            "eye_l": locations["eye.L"],
            "eye_r": locations["eye.R"],
            "mouth_l": [-0.02, 0.07, 0.08],
            "mouth_r": [0.02, 0.07, 0.08],
        },
    }
    def relocating_add(glb, values, kind, ct=5126):
        # Model an allocator relocating and recycling the previous backing store.
        previous = glb.blob
        glb.blob = bytearray(previous)
        previous[:] = b"\xff" * len(previous)
        return add(glb, values, kind, ct)

    with patch("roster.mesh_output.add", relocating_add):
        fit_template(g, resolved)
    for mesh in g.doc["meshes"]:
        p = mesh["primitives"][0]
        positions = g.array(p["attributes"]["POSITION"])
        faces = g.array(p["indices"]).reshape(-1, 3)
        expected = normals(positions, faces)[0]
        np.testing.assert_allclose(g.array(p["attributes"]["NORMAL"]), expected, atol=1e-4)
    print(json.dumps({"meshes": len(g.doc["meshes"])}))


class ScaffoldBufferTest(unittest.TestCase):
    def test_normals_after_appending_positions(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--fit"],
                cwd=scratch, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["meshes"], 2)


if __name__ == "__main__":
    if "--fit" in sys.argv:
        run_fit()
    else:
        unittest.main()
