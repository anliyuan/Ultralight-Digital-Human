import random
import unittest

from reference_sampling import sample_reference_index


class ReferenceSamplingTests(unittest.TestCase):
    def test_random_mode_stays_in_bounds(self):
        rng = random.Random(7)
        value = sample_reference_index(5, 20, mode="random", rng=rng)
        self.assertGreaterEqual(value, 0)
        self.assertLess(value, 20)

    def test_nearby_mode_stays_within_window(self):
        rng = random.Random(7)
        value = sample_reference_index(50, 100, mode="nearby", window=10, rng=rng)
        self.assertGreaterEqual(value, 40)
        self.assertLessEqual(value, 60)

    def test_min_offset_excludes_nearby_self_matches(self):
        rng = random.Random(7)
        value = sample_reference_index(50, 100, mode="nearby", window=10, min_offset=5, rng=rng)
        self.assertGreaterEqual(value, 40)
        self.assertLessEqual(value, 60)
        self.assertGreaterEqual(abs(value - 50), 5)

    def test_min_offset_falls_back_to_current_when_no_candidates_exist(self):
        rng = random.Random(7)
        value = sample_reference_index(0, 1, mode="random", min_offset=1, rng=rng)
        self.assertEqual(value, 0)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            sample_reference_index(0, 10, mode="unknown")
        with self.assertRaises(ValueError):
            sample_reference_index(0, 10, min_offset=-1)


if __name__ == "__main__":
    unittest.main()
