import numpy as np


class PredictionSmoother:
    def __init__(self, alpha=0.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = alpha
        self._previous = None

    def smooth(self, prediction):
        prediction = np.asarray(prediction, dtype=np.float32)
        if self.alpha == 0.0:
            return prediction
        if self._previous is None:
            self._previous = prediction.copy()
        else:
            self._previous = self.alpha * self._previous + (1.0 - self.alpha) * prediction
        return self._previous.copy()
