import unittest

import torch

from train_loss_config import combine_training_losses


class TrainLossConfigTests(unittest.TestCase):
    def test_combines_pixel_and_perceptual_losses(self):
        total = combine_training_losses(
            torch.tensor(2.0), torch.tensor(3.0), perceptual_weight=0.5
        )
        self.assertAlmostEqual(total.item(), 3.5, places=6)

    def test_combines_syncnet_loss_when_provided(self):
        total = combine_training_losses(
            torch.tensor(2.0),
            torch.tensor(3.0),
            perceptual_weight=0.5,
            sync_loss=torch.tensor(4.0),
            syncnet_weight=2.0,
        )
        self.assertAlmostEqual(total.item(), 11.5, places=6)


if __name__ == "__main__":
    unittest.main()
