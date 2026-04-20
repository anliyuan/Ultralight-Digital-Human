import torch


def get_padded_audio_window(features, index, left_context, right_context):
    feature_shape = tuple(features.shape[1:])
    start = max(0, index - left_context)
    end = min(features.shape[0], index + right_context)

    pad_left = max(0, left_context - index)
    pad_right = max(0, index + right_context - features.shape[0])

    window = torch.from_numpy(features[start:end])
    if pad_left:
        left_pad = torch.zeros((pad_left, *feature_shape), dtype=window.dtype)
        window = torch.cat([left_pad, window], dim=0)
    if pad_right:
        right_pad = torch.zeros((pad_right, *feature_shape), dtype=window.dtype)
        window = torch.cat([window, right_pad], dim=0)

    return window
