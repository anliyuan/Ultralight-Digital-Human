import numpy as np


class LandmarkSmoother:
    def __init__(self, alpha=0.8):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = alpha
        self._previous = None

    def smooth(self, points):
        points = np.asarray(points, dtype=np.float32)
        if self._previous is None:
            self._previous = points.copy()
        else:
            self._previous = self.alpha * self._previous + (1.0 - self.alpha) * points
        return np.rint(self._previous).astype(np.int32)
