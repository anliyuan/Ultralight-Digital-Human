import unittest

import numpy as np

from data_utils.landmark_failure import PreviousLandmarkFallback


class LandmarkFailureTests(unittest.TestCase):
    def test_error_policy_reraises(self):
        fallback = PreviousLandmarkFallback()
        with self.assertRaises(RuntimeError):
            fallback.resolve(RuntimeError("boom"), "error")

    def test_previous_policy_uses_last_recorded_landmarks(self):
        fallback = PreviousLandmarkFallback()
        previous = np.array([[1, 2], [3, 4]], dtype=np.int32)
        fallback.record(previous)
        resolved, reused = fallback.resolve(RuntimeError("boom"), "previous")
        np.testing.assert_array_equal(resolved, previous)
        self.assertTrue(reused)

    def test_previous_policy_without_previous_reraises(self):
        fallback = PreviousLandmarkFallback()
        with self.assertRaises(RuntimeError):
            fallback.resolve(RuntimeError("boom"), "previous")

    def test_unknown_policy_raises_value_error(self):
        fallback = PreviousLandmarkFallback()
        with self.assertRaises(ValueError):
            fallback.resolve(RuntimeError("boom"), "unknown")


if __name__ == "__main__":
    unittest.main()
