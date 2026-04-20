import random


def sample_reference_index(
    current_index,
    dataset_length,
    mode="random",
    window=30,
    rng=None,
):
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive.")
    if not 0 <= current_index < dataset_length:
        raise ValueError("current_index must be within dataset bounds.")

    if rng is None:
        rng = random

    if mode == "random":
        return rng.randint(0, dataset_length - 1)

    if mode == "nearby":
        left = max(0, current_index - window)
        right = min(dataset_length - 1, current_index + window)
        return rng.randint(left, right)

    raise ValueError(f"unsupported reference sampling mode: {mode}")
