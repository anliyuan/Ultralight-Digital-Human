import unittest

import numpy as np

from audio_windows import get_padded_audio_window


class AudioWindowTests(unittest.TestCase):
    def test_hubert_window_pads_short_sequences(self) -> None:
        features = np.ones((2, 2, 1024), dtype=np.float32)
        window = get_padded_audio_window(features, index=0, left_context=4, right_context=4)

        self.assertEqual(tuple(window.shape), (8, 2, 1024))
        self.assertTrue((window[:4] == 0).all())
        self.assertTrue((window[4:6] == 1).all())
        self.assertTrue((window[6:] == 0).all())

    def test_syncnet_window_pads_short_sequences(self) -> None:
        features = np.ones((2, 2, 1024), dtype=np.float32)
        window = get_padded_audio_window(features, index=0, left_context=8, right_context=8)

        self.assertEqual(tuple(window.shape), (16, 2, 1024))
        self.assertTrue((window[:8] == 0).all())
        self.assertTrue((window[8:10] == 1).all())
        self.assertTrue((window[10:] == 0).all())


if __name__ == "__main__":
    unittest.main()
