import unittest

import torch

from ssim_loss_utils import gaussian_kernel, ssim_index, ssim_loss


class SsimLossUtilsTests(unittest.TestCase):
    def test_gaussian_kernel_is_normalized(self):
        kernel = gaussian_kernel(kernel_size=5, sigma=1.0)
        self.assertAlmostEqual(kernel.sum().item(), 1.0, places=6)

    def test_ssim_index_is_one_for_identical_inputs(self):
        image = torch.rand(2, 3, 32, 32)
        value = ssim_index(image, image.clone())
        self.assertAlmostEqual(value.item(), 1.0, places=5)

    def test_ssim_loss_is_zero_for_identical_inputs(self):
        image = torch.rand(2, 3, 32, 32)
        loss = ssim_loss(image, image.clone())
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_ssim_loss_is_positive_for_different_inputs(self):
        preds = torch.zeros(1, 3, 32, 32)
        labels = torch.ones(1, 3, 32, 32)
        loss = ssim_loss(preds, labels)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
