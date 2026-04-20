import random


def sample_syncnet_audio_index(
    current_index,
    dataset_length,
    negative_pair_prob=0.0,
    rng=None,
):
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive.")
    if not 0 <= current_index < dataset_length:
        raise ValueError("current_index must be within dataset bounds.")
    if not 0.0 <= negative_pair_prob <= 1.0:
        raise ValueError("negative_pair_prob must be in [0, 1].")

    if rng is None:
        rng = random

    if dataset_length == 1 or rng.random() >= negative_pair_prob:
        return current_index, 1.0

    mismatched = current_index
    while mismatched == current_index:
        mismatched = rng.randint(0, dataset_length - 1)
    return mismatched, 0.0
