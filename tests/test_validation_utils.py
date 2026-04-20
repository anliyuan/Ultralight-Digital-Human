import tempfile
import unittest

import torch

from validation_utils import (
    maybe_save_best_checkpoint,
    should_run_validation,
    split_train_val_dataset,
)


class ValidationUtilsTests(unittest.TestCase):
    def test_split_train_val_dataset_is_deterministic(self):
        dataset = list(range(10))
        train_a, val_a = split_train_val_dataset(dataset, val_split=0.2, seed=7)
        train_b, val_b = split_train_val_dataset(dataset, val_split=0.2, seed=7)

        self.assertEqual(len(train_a), 8)
        self.assertEqual(len(val_a), 2)
        self.assertEqual(train_a.indices, train_b.indices)
        self.assertEqual(val_a.indices, val_b.indices)

    def test_should_run_validation_uses_eval_interval(self):
        self.assertFalse(should_run_validation(0, 5))
        self.assertFalse(should_run_validation(3, 5))
        self.assertTrue(should_run_validation(4, 5))

    def test_maybe_save_best_checkpoint_only_updates_on_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_a = {"weight": torch.tensor([1.0])}
            best_val_loss, best_path = maybe_save_best_checkpoint(state_a, tmpdir, 0.5, float("inf"))
            self.assertEqual(best_val_loss, 0.5)
            self.assertTrue(best_path.exists())
            self.assertEqual(torch.load(best_path)["weight"].item(), 1.0)

            state_b = {"weight": torch.tensor([2.0])}
            best_val_loss, best_path = maybe_save_best_checkpoint(state_b, tmpdir, 0.7, best_val_loss)
            self.assertEqual(best_val_loss, 0.5)
            self.assertEqual(torch.load(best_path)["weight"].item(), 1.0)

            state_c = {"weight": torch.tensor([3.0])}
            best_val_loss, best_path = maybe_save_best_checkpoint(state_c, tmpdir, 0.4, best_val_loss)
            self.assertEqual(best_val_loss, 0.4)
            self.assertEqual(torch.load(best_path)["weight"].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
