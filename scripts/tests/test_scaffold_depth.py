"""Regress frontal landmark fits that accidentally flatten unmeasured depth."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transfer.scaffold import landmark_warp


class ScaffoldDepthTest(unittest.TestCase):
    def setUp(self):
        self.old = np.array(
            [
                [-0.3, 0.2, 0.12],
                [0.3, 0.2, 0.12],
                [-0.2, 0.8, -0.04],
                [0.2, 0.8, -0.04],
                [0, 1.2, 0.15],
                [0, 0.5, 0.03],
            ]
        )
        self.wanted = self.old.copy()
        self.wanted[:, 2] = 0.1

    def test_frontal_field_retains_anterior_posterior_distance(self):
        warp = landmark_warp(self.old, self.wanted, "frontal_depth_preserving")
        pairs = np.array(
            [[0.1, 0.6, -0.12], [0.1, 0.6, 0.12], [-0.1, 0.3, -0.08], [-0.1, 0.3, 0.08]]
        )
        result = warp(pairs)
        np.testing.assert_allclose(
            result[1::2] - result[::2], pairs[1::2] - pairs[::2], atol=1e-12
        )
        self.assertLess(
            np.max(np.linalg.norm(warp(self.old) - self.wanted, axis=1)), 0.001
        )

    def test_default_retains_existing_spatial_result(self):
        points = np.array([[0.1, 0.6, -0.12], [0.1, 0.6, 0.12]])
        prior = RBFInterpolator(
            self.old,
            self.wanted - self.old,
            kernel="thin_plate_spline",
            smoothing=0.00005,
        )
        np.testing.assert_allclose(
            landmark_warp(self.old, self.wanted)(points),
            points + prior(points),
            atol=1e-12,
        )

    def test_overlapping_frontal_landmarks_are_rejected(self):
        old = np.vstack([self.old, self.old[0] + [0, 0, 0.2]])
        with self.assertRaisesRegex(ValueError, "Overlapping frontal"):
            landmark_warp(old, old, "frontal_depth_preserving")


if __name__ == "__main__":
    unittest.main()
