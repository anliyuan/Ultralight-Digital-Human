import numpy as np


class LandmarkOutlierRejector:
    def __init__(self, mean_distance_threshold=8.0):
        if mean_distance_threshold <= 0:
            raise ValueError("mean_distance_threshold must be positive.")
        self.mean_distance_threshold = mean_distance_threshold
        self._previous = None

    def filter(self, points):
        points = np.asarray(points, dtype=np.float32)
        if self._previous is None:
            self._previous = points.copy()
            return np.rint(points).astype(np.int32), False

        distances = np.linalg.norm(points - self._previous, axis=1)
        if float(distances.mean()) > self.mean_distance_threshold:
            return np.rint(self._previous).astype(np.int32), True

        self._previous = points.copy()
        return np.rint(points).astype(np.int32), False
