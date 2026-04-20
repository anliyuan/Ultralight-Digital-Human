import unittest

import torch

from mouth_focus_loss_utils import build_mouth_focus_mask, mouth_focus_l1_loss


class MouthFocusLossUtilsTests(unittest.TestCase):
    def test_build_mouth_focus_mask_targets_lower_center_region(self):
        mask = build_mouth_focus_mask(160, 160, device="cpu", dtype=torch.float32)
        self.assertEqual(tuple(mask.shape), (1, 1, 160, 160))
        self.assertEqual(mask[:, :, :40, :].sum().item(), 0.0)
        self.assertGreater(mask[:, :, 80:, 40:120].sum().item(), 0.0)

    def test_mouth_focus_l1_loss_is_zero_for_identical_inputs(self):
        image = torch.rand(2, 3, 160, 160)
        loss = mouth_focus_l1_loss(image, image.clone())
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_mouth_focus_l1_loss_ignores_outside_region(self):
        preds = torch.zeros(1, 3, 160, 160)
        labels = torch.zeros(1, 3, 160, 160)
        labels[:, :, :20, :20] = 1.0
        loss = mouth_focus_l1_loss(preds, labels)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_mouth_focus_l1_loss_penalizes_inside_region(self):
        preds = torch.zeros(1, 3, 160, 160)
        labels = torch.zeros(1, 3, 160, 160)
        labels[:, :, 80:120, 50:110] = 1.0
        loss = mouth_focus_l1_loss(preds, labels)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
