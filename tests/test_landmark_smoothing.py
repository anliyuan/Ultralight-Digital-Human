import unittest

import numpy as np

from data_utils.landmark_smoothing import LandmarkSmoother


class LandmarkSmoothingTests(unittest.TestCase):
    def test_first_frame_is_unchanged(self):
        smoother = LandmarkSmoother(alpha=0.8)
        points = np.array([[10, 20], [30, 40]], dtype=np.int32)
        smoothed = smoother.smooth(points)
        np.testing.assert_array_equal(smoothed, points)

    def test_subsequent_frames_are_smoothed(self):
        smoother = LandmarkSmoother(alpha=0.5)
        first = np.array([[0, 0], [10, 10]], dtype=np.int32)
        second = np.array([[10, 10], [20, 20]], dtype=np.int32)
        smoother.smooth(first)
        smoothed = smoother.smooth(second)
        expected = np.array([[5, 5], [15, 15]], dtype=np.int32)
        np.testing.assert_array_equal(smoothed, expected)

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            LandmarkSmoother(alpha=1.5)


if __name__ == "__main__":
    unittest.main()
