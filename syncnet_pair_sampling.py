import random


def sample_syncnet_audio_index(
    current_index,
    dataset_length,
    negative_pair_prob=0.0,
    negative_pair_mode="random",
    negative_pair_window=30,
    rng=None,
):
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive.")
    if not 0 <= current_index < dataset_length:
        raise ValueError("current_index must be within dataset bounds.")
    if not 0.0 <= negative_pair_prob <= 1.0:
        raise ValueError("negative_pair_prob must be in [0, 1].")
    if negative_pair_window < 0:
        raise ValueError("negative_pair_window must be non-negative.")
    if negative_pair_mode not in {"random", "nearby"}:
        raise ValueError(f"unsupported negative_pair_mode: {negative_pair_mode}")

    if rng is None:
        rng = random

    if dataset_length == 1 or rng.random() >= negative_pair_prob:
        return current_index, 1.0

    if negative_pair_mode == "random":
        candidates = [idx for idx in range(dataset_length) if idx != current_index]
    else:
        left = max(0, current_index - negative_pair_window)
        right = min(dataset_length - 1, current_index + negative_pair_window)
        candidates = [idx for idx in range(left, right + 1) if idx != current_index]

    if not candidates:
        return current_index, 1.0

    return candidates[rng.randint(0, len(candidates) - 1)], 0.0
