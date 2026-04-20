import random
import unittest

from crop_jitter import jitter_crop_box


class CropJitterTests(unittest.TestCase):
    def test_zero_jitter_keeps_crop(self):
        crop = jitter_crop_box(10, 20, 30, 40, (100, 100, 3), jitter_ratio=0.0, rng=random.Random(7))
        self.assertEqual(crop, (10, 20, 30, 40))

    def test_positive_jitter_stays_in_bounds(self):
        crop = jitter_crop_box(10, 20, 30, 40, (50, 50, 3), jitter_ratio=0.5, rng=random.Random(7))
        xmin, ymin, xmax, ymax = crop
        self.assertGreaterEqual(xmin, 0)
        self.assertGreaterEqual(ymin, 0)
        self.assertLessEqual(xmax, 50)
        self.assertLessEqual(ymax, 50)
        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)

    def test_invalid_ratio_raises(self):
        with self.assertRaises(ValueError):
            jitter_crop_box(10, 20, 30, 40, (100, 100, 3), jitter_ratio=-0.1)


if __name__ == "__main__":
    unittest.main()
