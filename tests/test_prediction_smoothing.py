import unittest

import numpy as np

from prediction_smoothing import PredictionSmoother


class PredictionSmoothingTests(unittest.TestCase):
    def test_zero_alpha_returns_input(self):
        smoother = PredictionSmoother(alpha=0.0)
        prediction = np.ones((2, 2, 3), dtype=np.float32)
        smoothed = smoother.smooth(prediction)
        np.testing.assert_array_equal(smoothed, prediction)

    def test_first_frame_is_unchanged_when_enabled(self):
        smoother = PredictionSmoother(alpha=0.8)
        prediction = np.ones((2, 2, 3), dtype=np.float32)
        smoothed = smoother.smooth(prediction)
        np.testing.assert_array_equal(smoothed, prediction)

    def test_subsequent_frames_are_smoothed(self):
        smoother = PredictionSmoother(alpha=0.5)
        first = np.zeros((1, 1, 1), dtype=np.float32)
        second = np.ones((1, 1, 1), dtype=np.float32) * 10
        smoother.smooth(first)
        smoothed = smoother.smooth(second)
        np.testing.assert_array_equal(smoothed, np.array([[[5.0]]], dtype=np.float32))

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            PredictionSmoother(alpha=1.5)


if __name__ == "__main__":
    unittest.main()
