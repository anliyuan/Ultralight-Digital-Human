import unittest

import torch

from gradient_loss_utils import gradient_l1_loss, image_gradients


class GradientLossUtilsTests(unittest.TestCase):
    def test_image_gradients_are_zero_for_constant_image(self):
        image = torch.ones(1, 3, 4, 4)
        grad_h, grad_v = image_gradients(image)
        self.assertTrue(torch.equal(grad_h, torch.zeros_like(grad_h)))
        self.assertTrue(torch.equal(grad_v, torch.zeros_like(grad_v)))

    def test_gradient_l1_loss_is_zero_for_identical_inputs(self):
        image = torch.rand(2, 3, 4, 4)
        criterion = torch.nn.L1Loss()
        loss = gradient_l1_loss(image, image.clone(), criterion)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_gradient_l1_loss_is_positive_for_different_inputs(self):
        preds = torch.zeros(1, 3, 4, 4)
        labels = torch.zeros(1, 3, 4, 4)
        labels[:, :, :, 2:] = 1.0
        criterion = torch.nn.L1Loss()
        loss = gradient_l1_loss(preds, labels, criterion)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
