import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from data_utils.audit_dataset import audit_dataset_dir


def _write_image(path: Path):
    image = np.full((256, 256, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_landmarks(path: Path, *, valid=True):
    landmarks = np.zeros((53, 2), dtype=np.int32)
    if valid:
        landmarks[1] = [50, 0]
        landmarks[31] = [150, 0]
        landmarks[52] = [0, 50]
    else:
        landmarks[1] = [240, 0]
        landmarks[31] = [300, 0]
        landmarks[52] = [0, 240]
    with path.open("w", encoding="utf-8") as f:
        for x, y in landmarks:
            f.write(f"{x} {y}\n")


class AuditDatasetTests(unittest.TestCase):
    def test_valid_dataset_has_no_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "full_body_img").mkdir()
            (root / "landmarks").mkdir()
            for idx in range(3):
                _write_image(root / "full_body_img" / f"{idx}.jpg")
                _write_landmarks(root / "landmarks" / f"{idx}.lms")
            np.save(root / "aud_hu.npy", np.zeros((3, 2, 1024), dtype=np.float32))

            result = audit_dataset_dir(root, mode="hubert")

            self.assertEqual(result["errors"], [])
            self.assertEqual(result["warnings"], [])

    def test_small_frame_audio_mismatch_is_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "full_body_img").mkdir()
            (root / "landmarks").mkdir()
            for idx in range(3):
                _write_image(root / "full_body_img" / f"{idx}.jpg")
                _write_landmarks(root / "landmarks" / f"{idx}.lms")
            np.save(root / "aud_hu.npy", np.zeros((2, 2, 1024), dtype=np.float32))

            result = audit_dataset_dir(root, mode="hubert")

            self.assertEqual(result["errors"], [])
            self.assertEqual(len(result["warnings"]), 1)

    def test_invalid_landmark_crop_is_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "full_body_img").mkdir()
            (root / "landmarks").mkdir()
            _write_image(root / "full_body_img" / "0.jpg")
            _write_landmarks(root / "landmarks" / "0.lms", valid=False)
            np.save(root / "aud_hu.npy", np.zeros((1, 2, 1024), dtype=np.float32))

            result = audit_dataset_dir(root, mode="hubert")

            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("outside image bounds", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
