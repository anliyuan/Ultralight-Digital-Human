import unittest
from unittest import mock

import numpy as np

from audio_activity import audio_feature_activity, is_low_activity
from datasetsss import MyDataset


class AudioActivityTests(unittest.TestCase):
    def test_audio_feature_activity_uses_mean_absolute_value(self):
        frame = np.array([[1.0, -3.0], [2.0, -2.0]], dtype=np.float32)
        self.assertAlmostEqual(audio_feature_activity(frame), 2.0, places=6)

    def test_low_activity_thresholding(self):
        frame = np.zeros((2, 2), dtype=np.float32)
        self.assertTrue(is_low_activity(frame, 0.1))
        self.assertFalse(is_low_activity(frame + 1.0, 0.1))
        with self.assertRaises(ValueError):
            is_low_activity(frame, -0.1)

    def test_dataset_resamples_low_activity_indices(self):
        dataset = object.__new__(MyDataset)
        dataset.audio_activity_policy = "resample"
        dataset.audio_activity_threshold = 0.5
        dataset.max_activity_retries = 4
        dataset.audio_feats = np.array(
            [
                np.zeros((2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            ]
        )
        dataset._load_sample = lambda idx: ("sample", idx)
        dataset.__len__ = lambda: 2

        with mock.patch("datasetsss.random.randint", return_value=1):
            result = MyDataset.__getitem__(dataset, 0)

        self.assertEqual(result, ("sample", 1))


if __name__ == "__main__":
    unittest.main()
