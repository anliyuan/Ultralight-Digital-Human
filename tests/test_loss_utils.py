import unittest

import torch

from loss_utils import grayscale_l1_loss, rgb_to_grayscale


class LossUtilsTests(unittest.TestCase):
    def test_rgb_to_grayscale_uses_weighted_channels(self):
        image = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
        gray = rgb_to_grayscale(image)
        expected = 0.299 * 1.0 + 0.587 * 2.0 + 0.114 * 3.0
        self.assertAlmostEqual(gray.item(), expected, places=6)

    def test_grayscale_l1_loss_is_zero_for_identical_inputs(self):
        image = torch.rand(2, 3, 4, 4)
        criterion = torch.nn.L1Loss()
        loss = grayscale_l1_loss(image, image.clone(), criterion)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_grayscale_l1_loss_is_positive_for_different_inputs(self):
        preds = torch.zeros(1, 3, 4, 4)
        labels = torch.ones(1, 3, 4, 4)
        criterion = torch.nn.L1Loss()
        loss = grayscale_l1_loss(preds, labels, criterion)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
