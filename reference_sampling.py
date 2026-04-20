import random


def sample_reference_index(
    current_index,
    dataset_length,
    mode="random",
    window=30,
    min_offset=0,
    rng=None,
):
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive.")
    if not 0 <= current_index < dataset_length:
        raise ValueError("current_index must be within dataset bounds.")
    if min_offset < 0:
        raise ValueError("min_offset must be non-negative.")

    if rng is None:
        rng = random

    def valid(candidate):
        return abs(candidate - current_index) >= min_offset

    if mode == "random":
        candidates = [idx for idx in range(dataset_length) if valid(idx)]
        if not candidates:
            return current_index
        return candidates[rng.randint(0, len(candidates) - 1)]

    if mode == "nearby":
        left = max(0, current_index - window)
        right = min(dataset_length - 1, current_index + window)
        candidates = [idx for idx in range(left, right + 1) if valid(idx)]
        if not candidates:
            return current_index
        return candidates[rng.randint(0, len(candidates) - 1)]

    raise ValueError(f"unsupported reference sampling mode: {mode}")
