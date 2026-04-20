import numpy as np


class PreviousLandmarkFallback:
    def __init__(self):
        self._previous = None

    def record(self, landmarks):
        self._previous = np.asarray(landmarks, dtype=np.int32).copy()
        return self._previous

    def resolve(self, error, policy):
        if policy == "error":
            raise error
        if policy == "previous":
            if self._previous is None:
                raise error
            return self._previous.copy(), True
        raise ValueError(f"unsupported failure policy: {policy}")
