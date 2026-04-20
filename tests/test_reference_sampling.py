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

    def test_nearby_mode_clamps_at_edges(self):
        rng = random.Random(7)
        value = sample_reference_index(3, 20, mode="nearby", window=10, rng=rng)
        self.assertGreaterEqual(value, 0)
        self.assertLessEqual(value, 13)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            sample_reference_index(0, 10, mode="unknown")


if __name__ == "__main__":
    unittest.main()
