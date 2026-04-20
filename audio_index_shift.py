def shifted_audio_index(index, shift, dataset_length):
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive.")
    shifted = index + shift
    if shifted < 0:
        return 0
    if shifted >= dataset_length:
        return dataset_length - 1
    return shifted
