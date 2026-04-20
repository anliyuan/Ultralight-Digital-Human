from pathlib import Path

import numpy as np


LEGACY_BACKEND = "pfld110"


def load_landmarks_file(path):
    path = Path(path)
    backend = LEGACY_BACKEND
    points = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# backend="):
                    backend = line.split("=", 1)[1].strip()
                continue
            values = line.split()
            if len(values) != 2:
                raise ValueError(f"invalid landmark row in {path}: {raw_line.rstrip()}")
            points.append([float(values[0]), float(values[1])])
    return backend, np.array(points, dtype=np.float32)


def save_landmarks_file(path, points, backend):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# backend={backend}\n")
        for x, y in points:
            f.write(f"{int(x)} {int(y)}\n")


def crop_box_from_landmarks(landmarks, backend):
    if backend in {"pfld", "pfld110"}:
        xmin = int(landmarks[1][0])
        ymin = int(landmarks[52][1])
        xmax = int(landmarks[31][0])
        width = xmax - xmin
        ymax = ymin + width
        return xmin, ymin, xmax, ymax

    if backend.startswith("mediapipe"):
        lower_face = landmarks[landmarks[:, 1] >= np.quantile(landmarks[:, 1], 0.45)]
        if lower_face.size == 0:
            raise ValueError("mediapipe landmarks do not contain a lower-face region")
        xmin = int(np.quantile(lower_face[:, 0], 0.05))
        xmax = int(np.quantile(lower_face[:, 0], 0.95))
        ymin = int(np.quantile(lower_face[:, 1], 0.10))
        width = xmax - xmin
        ymax = ymin + width
        return xmin, ymin, xmax, ymax

    raise ValueError(f"unsupported landmark backend: {backend}")
