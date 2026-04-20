import unittest

import numpy as np

from data_utils.crop_utils import expanded_square_crop_from_landmarks


class CropExpandRatioTests(unittest.TestCase):
    def test_ratio_one_matches_legacy_geometry(self):
        landmarks = np.zeros((110, 2), dtype=np.float32)
        landmarks[1] = [10, 0]
        landmarks[31] = [30, 0]
        landmarks[52] = [0, 20]
        crop = expanded_square_crop_from_landmarks(landmarks, (100, 100, 3), expand_ratio=1.0)
        self.assertEqual(crop, (10, 20, 30, 40))

    def test_larger_ratio_expands_crop_around_center(self):
        landmarks = np.zeros((110, 2), dtype=np.float32)
        landmarks[1] = [10, 0]
        landmarks[31] = [30, 0]
        landmarks[52] = [0, 20]
        crop = expanded_square_crop_from_landmarks(landmarks, (100, 100, 3), expand_ratio=1.5)
        self.assertEqual(crop, (5, 15, 35, 45))

    def test_crop_is_clamped_after_expansion(self):
        landmarks = np.zeros((110, 2), dtype=np.float32)
        landmarks[1] = [5, 0]
        landmarks[31] = [25, 0]
        landmarks[52] = [0, 5]
        crop = expanded_square_crop_from_landmarks(landmarks, (40, 40, 3), expand_ratio=2.0)
        self.assertEqual(crop, (0, 0, 35, 35))

    def test_invalid_ratio_raises(self):
        landmarks = np.zeros((110, 2), dtype=np.float32)
        landmarks[1] = [10, 0]
        landmarks[31] = [30, 0]
        landmarks[52] = [0, 20]
        with self.assertRaises(ValueError):
            expanded_square_crop_from_landmarks(landmarks, (100, 100, 3), expand_ratio=0.0)


if __name__ == "__main__":
    unittest.main()
