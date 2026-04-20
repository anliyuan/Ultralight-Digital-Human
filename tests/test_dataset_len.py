import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from datasetsss import MyDataset


def _write_landmarks(path: Path) -> None:
    # Keep the crop box inside a 256x256 test image.
    landmarks = np.zeros((53, 2), dtype=np.int32)
    landmarks[1] = [50, 0]
    landmarks[31] = [150, 0]
    landmarks[52] = [0, 50]
    with path.open("w", encoding="utf-8") as f:
        for x, y in landmarks:
            f.write(f"{x} {y}\n")


class DatasetLenTests(unittest.TestCase):
    def test_dataloader_handles_audio_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "full_body_img"
            lms_dir = root / "landmarks"
            image_dir.mkdir()
            lms_dir.mkdir()

            image = np.full((256, 256, 3), 127, dtype=np.uint8)
            for idx in range(3):
                cv2.imwrite(str(image_dir / f"{idx}.jpg"), image)
                _write_landmarks(lms_dir / f"{idx}.lms")

            audio_feats = np.zeros((2, 2, 1024), dtype=np.float32)
            np.save(root / "aud_hu.npy", audio_feats)

            dataset = MyDataset(str(root), "hubert")
            self.assertEqual(len(dataset), 2)

            loader = DataLoader(dataset, batch_size=1, shuffle=True)
            self.assertEqual(len(loader), 2)


if __name__ == "__main__":
    unittest.main()
