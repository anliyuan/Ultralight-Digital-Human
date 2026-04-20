import numpy as np


def audio_feature_activity(audio_feature_frame):
    feature = np.asarray(audio_feature_frame, dtype=np.float32)
    return float(np.mean(np.abs(feature)))


def is_low_activity(audio_feature_frame, threshold):
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    return audio_feature_activity(audio_feature_frame) < threshold
