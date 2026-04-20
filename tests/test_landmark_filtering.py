import unittest

import numpy as np

from data_utils.landmark_filtering import LandmarkOutlierRejector


class LandmarkFilteringTests(unittest.TestCase):
    def test_first_frame_is_kept(self):
        rejector = LandmarkOutlierRejector(mean_distance_threshold=8.0)
        points = np.array([[0, 0], [10, 10]], dtype=np.int32)
        filtered, rejected = rejector.filter(points)
        np.testing.assert_array_equal(filtered, points)
        self.assertFalse(rejected)

    def test_small_motion_is_kept(self):
        rejector = LandmarkOutlierRejector(mean_distance_threshold=8.0)
        rejector.filter(np.array([[0, 0], [10, 10]], dtype=np.int32))
        filtered, rejected = rejector.filter(np.array([[1, 1], [11, 11]], dtype=np.int32))
        np.testing.assert_array_equal(filtered, np.array([[1, 1], [11, 11]], dtype=np.int32))
        self.assertFalse(rejected)

    def test_large_jump_reuses_previous_frame(self):
        rejector = LandmarkOutlierRejector(mean_distance_threshold=2.0)
        previous = np.array([[0, 0], [10, 10]], dtype=np.int32)
        rejector.filter(previous)
        filtered, rejected = rejector.filter(np.array([[20, 20], [30, 30]], dtype=np.int32))
        np.testing.assert_array_equal(filtered, previous)
        self.assertTrue(rejected)

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            LandmarkOutlierRejector(mean_distance_threshold=0.0)


if __name__ == "__main__":
    unittest.main()
