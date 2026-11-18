"""Offline calibration and image-registration checks; no model/API calls."""
import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
measurement = importlib.import_module("mh-bodysuit")


class ImageRegistrationTests(unittest.TestCase):
    def test_calibrated_mask_keeps_legs_joined_by_background_coloured_shorts(self):
        image = np.full((60, 40, 3), 100, dtype=np.uint8)
        image[5:30, 10:30] = 190
        image[30:42, 10:30] = 103
        image[42:55, 11:18] = 175
        image[42:55, 22:29] = 175
        mask = measurement.calibrated_mask(image)
        self.assertTrue(mask[54, 14] and mask[54, 25])
        self.assertFalse(mask[54, 20] or mask[55].any())

    def test_parallel_closeup_keeps_calibrated_pixel_position(self):
        axes = dict(right=[1, 0, 0], up=[0, 1, 0])
        cameras = {
            "body": dict(axes, center=[0, 1, 0], width=100, height=100, meters_per_pixel=.01),
            "face": dict(axes, center=[0, 1.2, .2], width=200, height=200, meters_per_pixel=.001),
        }
        guide = {"landmarks": {"nose_tip": [dict(view="face", px=[120, 130])]}}
        np.testing.assert_allclose(measurement.parallel_mark(guide, cameras, "body", "nose_tip"), [52, 33])
        cameras["face"]["right"] = [0, 0, -1]
        self.assertIsNone(measurement.parallel_mark(guide, cameras, "body", "nose_tip"))

    def test_known_translation_uses_only_valid_search_positions(self):
        rng = np.random.default_rng(143)
        original = np.full((90, 120, 3), 100, np.uint8)
        original[28:48, 38:62] = rng.integers(0, 255, (20, 24, 3), dtype=np.uint8)
        edited = np.full_like(original, 100)
        edited[23:43, 45:69] = original[28:48, 38:62]
        before, after, error = measurement.match_patch(original, edited, 50, 38, 24, 20, 10)
        np.testing.assert_allclose(np.subtract(after, before), [7, -5])
        self.assertLess(error, .001)
        before, after, _ = measurement.match_patch(original, edited, 50, 38, 24, 20, 3)
        self.assertLessEqual(np.abs(np.subtract(after, before)).max(), 3)

    def test_large_head_anchor_uses_face_features_instead_of_neck_ratio(self):
        camera = dict(center=[0, 0, 0], right=[1, 0, 0], up=[0, 1, 0],
                      width=768, height=1024, meters_per_pixel=.001)
        points = {"eye_l": [450, 462], "eye_r": [308, 462], "nose_tip": [382, 510],
                  "mouth_l": [451, 569], "mouth_r": [309, 569], "neck": [382, 647]}
        guide = {"landmarks": {name: [dict(view="body", px=point)] for name, point in points.items()}}
        x, y, width, height, radius = measurement.registration_patches(guide, {"body": camera}, "body")["face"]
        self.assertLess(y - height / 2, 462)
        self.assertGreater(y + height / 2, 569)
        self.assertLess(y + height / 2, 647)

    def test_measurement_key_comes_from_environment(self):
        with patch.dict(measurement.os.environ, {"OPENAI_KEY": "test-only"}, clear=True):
            self.assertEqual(measurement.key(), "test-only")
        with patch.dict(measurement.os.environ, {}, clear=True), patch.object(Path, "read_text", side_effect=AssertionError("unexpected credential file read")):
            with self.assertRaises(SystemExit):
                measurement.key()


if __name__ == "__main__":
    unittest.main()
