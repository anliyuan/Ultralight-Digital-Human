import unittest

import numpy as np

from compositing import blend_patch, composite_prediction, feather_alpha


class CompositingTests(unittest.TestCase):
    def test_feather_alpha_center_is_opaque_and_edges_fade(self):
        alpha = feather_alpha((20, 20, 3), feather=4)
        self.assertEqual(alpha.shape, (20, 20, 1))
        self.assertAlmostEqual(float(alpha[10, 10, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(alpha[0, 0, 0]), 0.0, places=6)

    def test_blend_patch_preserves_edges_with_feathering(self):
        base = np.zeros((20, 20, 3), dtype=np.uint8)
        generated = np.full((20, 20, 3), 255, dtype=np.uint8)
        blended = blend_patch(base, generated, feather=4)
        self.assertEqual(int(blended[0, 0, 0]), 0)
        self.assertEqual(int(blended[10, 10, 0]), 255)

    def test_hard_crop_composite_replaces_inner_patch(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        crop = np.zeros((16, 16, 3), dtype=np.uint8)
        pred = np.full((8, 8, 3), 255, dtype=np.uint8)
        out = composite_prediction(
            frame, crop, pred, (8, 8, 24, 24), mode="hard_crop", inner_margin=4
        )
        self.assertGreater(out[16, 16, 0], 0)

    def test_invalid_mode_raises(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        crop = np.zeros((16, 16, 3), dtype=np.uint8)
        pred = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            composite_prediction(frame, crop, pred, (8, 8, 24, 24), mode="bad")


if __name__ == "__main__":
    unittest.main()
