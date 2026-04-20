import random
import unittest

from syncnet_pair_sampling import sample_syncnet_audio_index


class SyncnetPairSamplingTests(unittest.TestCase):
    def test_zero_negative_pair_prob_keeps_positive_pair(self):
        rng = random.Random(7)
        audio_idx, label = sample_syncnet_audio_index(5, 20, negative_pair_prob=0.0, rng=rng)
        self.assertEqual(audio_idx, 5)
        self.assertEqual(label, 1.0)

    def test_full_negative_pair_prob_creates_mismatch(self):
        rng = random.Random(7)
        audio_idx, label = sample_syncnet_audio_index(5, 20, negative_pair_prob=1.0, rng=rng)
        self.assertNotEqual(audio_idx, 5)
        self.assertEqual(label, 0.0)

    def test_single_frame_dataset_stays_positive(self):
        rng = random.Random(7)
        audio_idx, label = sample_syncnet_audio_index(0, 1, negative_pair_prob=1.0, rng=rng)
        self.assertEqual(audio_idx, 0)
        self.assertEqual(label, 1.0)

    def test_invalid_probability_raises(self):
        with self.assertRaises(ValueError):
            sample_syncnet_audio_index(0, 10, negative_pair_prob=1.5)


if __name__ == "__main__":
    unittest.main()
