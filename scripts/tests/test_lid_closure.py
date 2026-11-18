"""Check closure on sparse lid triangles using the full measured margins."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "roster"))
from face_atlas import close_authored_lids


class LidClosureTest(unittest.TestCase):
    def test_sparse_triangle_interpolation_closes_both_margins(self):
        vertices = np.array([
            [-.012, 0., .1], [0., .006, .102], [.012, 0., .1],
            [-.012, 0., .1], [0., -.002, .102], [.012, 0., .1],
            [0., .03, .1],
        ])
        samples = np.linspace(.1, .9, 17)
        barycentric = np.column_stack((1 - samples, samples, np.zeros(len(samples))))
        weights = np.repeat(barycentric[:, None, :], 2, axis=1)
        corners = np.tile([[0, 1, 2], [3, 4, 5]], (len(samples), 1, 1))
        points = (weights[..., None] * vertices[corners]).sum(axis=2)
        contact = dict(weights=weights, corners=corners, points=points, x=points[:, 0, 0])
        arc = (vertices[:3, 0], vertices[:3, 1], vertices[3:6, 1], vertices[:3, 2])
        delta = np.zeros_like(vertices)
        delta[:6, 1] = [-.001, -.009, -.001, .0001, .001, .0001]
        close_authored_lids(vertices, delta, arc, contact)
        closed = points + (weights[..., None] * delta[corners]).sum(axis=2)
        np.testing.assert_allclose(closed[:, 0, 1], closed[:, 1, 1], atol=1.e-5)
        np.testing.assert_array_equal(delta[-1], 0)
        np.testing.assert_array_equal(delta[:, [0, 2]], 0)


if __name__ == "__main__":
    unittest.main()
