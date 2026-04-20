import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_utils.landmark_io import (
    LEGACY_BACKEND,
    crop_box_from_landmarks,
    load_landmarks_file,
    save_landmarks_file,
)


class LandmarkIoTests(unittest.TestCase):
    def test_legacy_landmark_file_defaults_to_pfld_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.lms"
            path.write_text("1 2\n3 4\n", encoding="utf-8")
            backend, landmarks = load_landmarks_file(path)
            self.assertEqual(backend, LEGACY_BACKEND)
            self.assertEqual(tuple(landmarks.shape), (2, 2))

    def test_save_and_load_round_trip_preserves_backend_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.lms"
            points = np.array([[1, 2], [3, 4]], dtype=np.int32)
            save_landmarks_file(path, points, "mediapipe478")
            backend, landmarks = load_landmarks_file(path)
            self.assertEqual(backend, "mediapipe478")
            np.testing.assert_array_equal(landmarks, points)

    def test_crop_box_from_pfld_landmarks_uses_legacy_indices(self):
        landmarks = np.zeros((110, 2), dtype=np.float32)
        landmarks[1] = [10, 0]
        landmarks[31] = [30, 0]
        landmarks[52] = [0, 20]
        self.assertEqual(crop_box_from_landmarks(landmarks, "pfld110"), (10, 20, 30, 40))

    def test_crop_box_from_mediapipe_landmarks_uses_lower_face_quantiles(self):
        xs = np.linspace(20, 80, num=100)
        ys = np.linspace(20, 90, num=100)
        landmarks = np.stack([xs, ys], axis=1).astype(np.float32)
        xmin, ymin, xmax, ymax = crop_box_from_landmarks(landmarks, "mediapipe478")
        self.assertLess(xmin, xmax)
        self.assertLess(ymin, ymax)
        self.assertGreaterEqual(xmin, 20)
        self.assertLessEqual(xmax, 80)


if __name__ == "__main__":
    unittest.main()
