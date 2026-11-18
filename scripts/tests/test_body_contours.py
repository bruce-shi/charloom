"""Check that contour fitting is independent of the polyline's sampling density."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transfer.body import closest_contour


class BodyContourTest(unittest.TestCase):
    def test_sparse_edge_preserves_tangential_position(self):
        vertices = np.array([[0., 0.], [20., 0.], [20., 10.], [0., 10.]])
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        points = np.array([[3., -.5], [17., -.5], [20.5, 4.]])
        distance, target, trust = closest_contour(points, vertices, edges, np.ones(4))
        np.testing.assert_allclose(distance, .5)
        np.testing.assert_allclose(target, [[3., 0.], [17., 0.], [20., 4.]])
        np.testing.assert_allclose(trust, 1)

    def test_confidence_interpolates_on_the_selected_edge(self):
        vertices = np.array([[0., 0.], [20., 0.]])
        edges = np.array([[0, 1]])
        _, _, trust = closest_contour(
            np.array([[5., 2.], [15., 2.]]), vertices, edges, np.array([0., 1.])
        )
        np.testing.assert_allclose(trust, [.25, .75])


if __name__ == "__main__":
    unittest.main()
