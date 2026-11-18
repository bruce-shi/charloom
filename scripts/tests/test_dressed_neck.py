"""Check interpolated UVs and geometry when clipping an observed collar triangle."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "roster"))
from dressed_neck import lower_surface


class DressedNeckTest(unittest.TestCase):
    def test_clipped_quad_preserves_area_and_uvs(self):
        vertices = np.array([[0., -1., 0.], [2., -1., 0.], [0., 1., 0.]])
        uv = (vertices[:, :2] + [0., 1.]) / 2
        points, faces, normals, tex = lower_surface(
            vertices, np.array([[0, 1, 2]]), np.array([0., 1., 0.]), 0.,
            np.tile([0., 0., 1.], (3, 1)), uv)
        self.assertEqual(len(faces), 2)
        self.assertLessEqual(float(points[:, 1].max()), 0.)
        np.testing.assert_allclose(tex, (points[:, :2] + [0., 1.]) / 2)
        tri = points[faces]
        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
        self.assertAlmostEqual(area, 1.5)
        np.testing.assert_allclose(normals, np.tile([0., 0., 1.], (len(points), 1)))
