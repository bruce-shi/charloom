"""Check source-surface preservation and the split lip binding on a small planar mesh."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "roster"))
from mesh import Glb
from mesh_output import add, idx
from scan_surface_face import bind_face


class ScanSurfaceFaceTest(unittest.TestCase):
    def test_neutral_surface_and_lip_lining_follow_source(self):
        model = Glb.__new__(Glb)
        model.blob = bytearray()
        model.doc = dict(
            accessors=[],
            bufferViews=[],
            buffers=[dict(byteLength=0)],
            materials=[dict()],
            skins=[dict(joints=[1])],
            nodes=[dict(name="fitted_head", mesh=0, skin=0), dict(name="head")],
            scenes=[dict(nodes=[0, 1])],
            meshes=[],
        )
        vertices = np.array(
            [[-0.1, 1.5, 0.1], [0.1, 1.5, 0.1], [0.1, 1.75, 0.1], [-0.1, 1.75, 0.1]],
            dtype="<f4",
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype="<f4")
        # An artificial closure behind the cheek shares the facial bounding box.
        # It must stay in the head, outside the animated source-surface patch.
        source_vertices = vertices.copy()
        vertices = np.vstack((vertices, vertices[:3] - [0, 0, 0.02])).astype("<f4")
        uv = np.vstack((uv, uv[:3]))
        count = len(vertices)
        attributes = dict(
            POSITION=add(model, vertices, "VEC3"),
            NORMAL=add(model, np.tile([0.0, 0.0, 1.0], (count, 1)).astype("<f4"), "VEC3"),
            TEXCOORD_0=add(model, uv, "VEC2"),
            JOINTS_0=add(model, np.zeros((count, 4), dtype="<u2"), "VEC4", 5123),
            WEIGHTS_0=add(
                model, np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)).astype("<f4"), "VEC4"
            ),
        )
        model.doc["meshes"].append(
            dict(
                name="fitted_head",
                primitives=[
                    dict(attributes=attributes, indices=idx(model, np.vstack((faces, [4, 5, 6]))), material=0)
                ],
            )
        )
        mouth_x = np.linspace(-0.04, 0.04, 9)
        curves = dict(
            mouth=(mouth_x, np.full(9, 1.56), np.full(9, 1.56), np.full(9, 0.1))
        )
        for side, center in [("l", 0.045), ("r", -0.045)]:
            x = np.linspace(center - 0.025, center + 0.025, 9)
            arch = np.sin(np.linspace(0, np.pi, 9))
            curves[side] = (
                x,
                1.67 + 0.004 * arch,
                1.67 - 0.003 * arch,
                np.full(9, 0.1),
            )
        controls = bind_face(model, curves, source_vertices.astype(float), faces,
                             observed_faces=[True, True, False])
        retained_indices = model.doc["meshes"][0]["primitives"][0]["indices"]
        np.testing.assert_array_equal(model.array(retained_indices).ravel(), [4, 5, 6])
        face = next(m for m in model.doc["meshes"] if m["name"] == "FaceSurface")
        primitive = face["primitives"][0]
        positions = model.array(primitive["attributes"]["POSITION"])
        actual_uv = model.array(primitive["attributes"]["TEXCOORD_0"])
        np.testing.assert_allclose(positions[:, 2], 0.1, atol=1.0e-7)
        np.testing.assert_allclose(
            actual_uv[:, 0], (positions[:, 0] + 0.1) / 0.2, atol=1.0e-6
        )
        np.testing.assert_allclose(
            actual_uv[:, 1], (positions[:, 1] - 1.5) / 0.25, atol=1.0e-6
        )
        self.assertLess(controls["sourceSurface"]["neutral_max_distance_mm"], 0.001)

        def samples(bindings, values):
            return np.array(
                [np.asarray(b["weights"]) @ values[b["vertices"]] for b in bindings]
            )

        upper = samples(controls["mouth"]["upper"], positions)
        lower = samples(controls["mouth"]["lower"], positions)
        np.testing.assert_allclose(upper, lower, atol=1.0e-7)
        target_index = face["extras"]["targetNames"].index("cheekJawOpen")
        delta = model.array(primitive["targets"][target_index]["POSITION"])
        opened = samples(controls["mouth"]["upper"], positions + delta) - samples(
            controls["mouth"]["lower"], positions + delta
        )
        self.assertGreater(opened[:, 1].max(), 0.02)
        cavity = next(m for m in model.doc["meshes"] if m["name"] == "MouthInterior")[
            "primitives"
        ][0]
        lining = model.array(cavity["attributes"]["POSITION"]).reshape(65, 6, 3)
        self.assertTrue(np.all(lining[:, :, 2] < upper[:, None, 2]))
        lining_delta = model.array(cavity["targets"][target_index]["POSITION"]).reshape(
            65, 6, 3
        )
        np.testing.assert_allclose(
            lining_delta[:, 0], samples(controls["mouth"]["upper"], delta), atol=1.0e-7
        )
        np.testing.assert_allclose(
            lining_delta[:, -1], samples(controls["mouth"]["lower"], delta), atol=1.0e-7
        )


if __name__ == "__main__":
    unittest.main()
