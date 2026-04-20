import unittest

from audio_index_shift import shifted_audio_index


class AudioIndexShiftTests(unittest.TestCase):
    def test_zero_shift_keeps_index(self):
        self.assertEqual(shifted_audio_index(5, 0, 10), 5)

    def test_positive_shift_moves_forward_and_clamps(self):
        self.assertEqual(shifted_audio_index(5, 2, 10), 7)
        self.assertEqual(shifted_audio_index(9, 3, 10), 9)

    def test_negative_shift_moves_backward_and_clamps(self):
        self.assertEqual(shifted_audio_index(5, -2, 10), 3)
        self.assertEqual(shifted_audio_index(0, -3, 10), 0)

    def test_invalid_length_raises(self):
        with self.assertRaises(ValueError):
            shifted_audio_index(0, 0, 0)


if __name__ == "__main__":
    unittest.main()
