import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _numeric_stem(path: Path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def _read_landmarks(path: Path):
    points = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue
            if len(values) != 2:
                raise ValueError(f"expected 2 values per landmark row in {path}")
            points.append([float(values[0]), float(values[1])])
    return np.array(points, dtype=np.float32)


def _validate_crop(image_path: Path, landmarks_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return f"failed to read image {image_path}"

    landmarks = _read_landmarks(landmarks_path)
    if landmarks.shape[0] <= 52:
        return f"landmark file {landmarks_path} has too few points"

    xmin = int(landmarks[1][0])
    ymin = int(landmarks[52][1])
    xmax = int(landmarks[31][0])
    width = xmax - xmin
    ymax = ymin + width

    if width <= 0:
        return f"invalid crop width from landmarks in {landmarks_path}"

    height, img_width = image.shape[:2]
    if xmin < 0 or ymin < 0 or xmax > img_width or ymax > height:
        return f"crop derived from {landmarks_path} falls outside image bounds"

    return None


def audit_dataset_dir(dataset_dir, mode="hubert"):
    dataset_dir = Path(dataset_dir)
    frame_dir = dataset_dir / "full_body_img"
    landmark_dir = dataset_dir / "landmarks"
    audio_path = dataset_dir / ("aud_hu.npy" if mode == "hubert" else "aud_wenet.npy")

    result = {
        "dataset_dir": str(dataset_dir),
        "mode": mode,
        "frame_count": 0,
        "landmark_count": 0,
        "audio_feature_count": 0,
        "errors": [],
        "warnings": [],
    }

    if not frame_dir.exists():
        result["errors"].append(f"missing directory: {frame_dir}")
        return result
    if not landmark_dir.exists():
        result["errors"].append(f"missing directory: {landmark_dir}")
        return result
    if not audio_path.exists():
        result["errors"].append(f"missing audio features: {audio_path}")
        return result

    frames = sorted(frame_dir.glob("*.jpg"), key=_numeric_stem)
    landmarks = sorted(landmark_dir.glob("*.lms"), key=_numeric_stem)
    audio_features = np.load(audio_path)

    result["frame_count"] = len(frames)
    result["landmark_count"] = len(landmarks)
    result["audio_feature_count"] = int(audio_features.shape[0])

    if len(frames) != len(landmarks):
        result["errors"].append(
            f"frame/landmark count mismatch: {len(frames)} frames vs {len(landmarks)} landmark files"
        )

    diff = abs(len(frames) - int(audio_features.shape[0]))
    if diff > 0:
        message = f"frame/audio feature count mismatch: {len(frames)} frames vs {audio_features.shape[0]} audio features"
        if diff > 5:
            result["errors"].append(message)
        else:
            result["warnings"].append(message)

    for frame_path, landmarks_path in zip(frames, landmarks):
        crop_error = _validate_crop(frame_path, landmarks_path)
        if crop_error is not None:
            result["errors"].append(crop_error)
            break

    return result


def main():
    parser = argparse.ArgumentParser(description="Audit a processed training dataset")
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("--asr", type=str, default="hubert", choices=["hubert", "wenet"])
    args = parser.parse_args()

    result = audit_dataset_dir(args.dataset_dir, mode=args.asr)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
