import unittest

import numpy as np

from data_utils.crop_utils import square_crop_from_landmarks


class CropUtilsTests(unittest.TestCase):
    def test_square_crop_matches_legacy_geometry_when_in_bounds(self):
        landmarks = np.zeros((110, 2), dtype=np.float32)
        landmarks[1] = [10, 0]
        landmarks[31] = [30, 0]
        landmarks[52] = [0, 20]
        crop = square_crop_from_landmarks(landmarks, (100, 100, 3))
        self.assertEqual(crop, (10, 20, 30, 40))

    def test_square_crop_is_clamped_to_image_bounds(self):
        landmarks = np.zeros((110, 2), dtype=np.float32)
        landmarks[1] = [-10, 0]
        landmarks[31] = [30, 0]
        landmarks[52] = [0, 90]
        crop = square_crop_from_landmarks(landmarks, (100, 100, 3))
        self.assertEqual(crop, (0, 90, 10, 100))

    def test_invalid_width_raises(self):
        landmarks = np.zeros((110, 2), dtype=np.float32)
        landmarks[1] = [30, 0]
        landmarks[31] = [10, 0]
        landmarks[52] = [0, 20]
        with self.assertRaises(ValueError):
            square_crop_from_landmarks(landmarks, (100, 100, 3))


if __name__ == "__main__":
    unittest.main()
