import unittest
from unittest import mock

from datasetsss import MyDataset


class DatasetRetryTests(unittest.TestCase):
    def test_getitem_retries_until_success(self):
        dataset = object.__new__(MyDataset)
        dataset.max_sample_retries = 3
        dataset._attempts = 0

        def fake_load(idx):
            dataset._attempts += 1
            if dataset._attempts < 3:
                raise ValueError("bad sample")
            return ("ok", idx)

        dataset._load_sample = fake_load
        dataset.__len__ = lambda: 5

        with mock.patch("datasetsss.random.randint", return_value=1):
            result = MyDataset.__getitem__(dataset, 0)

        self.assertEqual(result, ("ok", 1))
        self.assertEqual(dataset._attempts, 3)

    def test_getitem_raises_after_exhausting_retries(self):
        dataset = object.__new__(MyDataset)
        dataset.max_sample_retries = 2
        dataset._load_sample = mock.Mock(side_effect=ValueError("bad sample"))
        dataset.__len__ = lambda: 5

        with mock.patch("datasetsss.random.randint", return_value=1):
            with self.assertRaises(RuntimeError):
                MyDataset.__getitem__(dataset, 0)


if __name__ == "__main__":
    unittest.main()
