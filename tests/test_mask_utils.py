import unittest

import numpy as np

from data_utils.mask_utils import (
    DEFAULT_MASK_BOTTOM_RATIO,
    DEFAULT_MASK_LEFT_RATIO,
    DEFAULT_MASK_RIGHT_RATIO,
    DEFAULT_MASK_TOP_RATIO,
    apply_blackout_mask,
)


class MaskUtilsTests(unittest.TestCase):
    def test_default_ratios_match_current_geometry(self):
        image = np.full((160, 160, 3), 255, dtype=np.uint8)
        masked = apply_blackout_mask(image)
        self.assertTrue((masked[5:146, 5:151] == 0).all())
        self.assertTrue((masked[:5, :] == 255).all())

    def test_custom_ratios_mask_expected_region(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        masked = apply_blackout_mask(image, left_ratio=0.1, top_ratio=0.2, right_ratio=0.5, bottom_ratio=0.6)
        self.assertTrue((masked[20:61, 10:51] == 0).all())
        self.assertTrue((masked[:20, :] == 255).all())

    def test_invalid_ratio_order_raises(self):
        image = np.full((10, 10, 3), 255, dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_blackout_mask(image, left_ratio=0.6, right_ratio=0.5)
        with self.assertRaises(ValueError):
            apply_blackout_mask(image, top_ratio=0.8, bottom_ratio=0.2)


if __name__ == "__main__":
    unittest.main()
