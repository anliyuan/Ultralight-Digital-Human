import unittest

from syncnet_warmup import syncnet_weight_for_epoch


class SyncnetWarmupTests(unittest.TestCase):
    def test_zero_warmup_uses_base_weight(self):
        self.assertEqual(syncnet_weight_for_epoch(0, base_weight=10.0, warmup_epochs=0), 10.0)

    def test_weight_ramps_linearly_during_warmup(self):
        self.assertAlmostEqual(
            syncnet_weight_for_epoch(0, base_weight=10.0, warmup_epochs=5), 2.0, places=6
        )
        self.assertAlmostEqual(
            syncnet_weight_for_epoch(2, base_weight=10.0, warmup_epochs=5), 6.0, places=6
        )
        self.assertAlmostEqual(
            syncnet_weight_for_epoch(4, base_weight=10.0, warmup_epochs=5), 10.0, places=6
        )

    def test_weight_caps_at_base_weight_after_warmup(self):
        self.assertAlmostEqual(
            syncnet_weight_for_epoch(10, base_weight=10.0, warmup_epochs=5), 10.0, places=6
        )

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            syncnet_weight_for_epoch(0, base_weight=-1.0, warmup_epochs=0)
        with self.assertRaises(ValueError):
            syncnet_weight_for_epoch(0, base_weight=1.0, warmup_epochs=-1)


if __name__ == "__main__":
    unittest.main()
