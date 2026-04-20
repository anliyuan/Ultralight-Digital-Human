import unittest

from training_utils import build_train_dataloader


class TrainingUtilsTests(unittest.TestCase):
    def test_build_train_dataloader_uses_requested_batch_size(self) -> None:
        dataset = list(range(5))
        loader = build_train_dataloader(dataset, batch_size=2, num_workers=0)

        self.assertEqual(loader.batch_size, 2)
        self.assertEqual(len(loader), 3)


if __name__ == "__main__":
    unittest.main()
