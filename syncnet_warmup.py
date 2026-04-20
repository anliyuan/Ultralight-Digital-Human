def syncnet_weight_for_epoch(epoch_index, base_weight=10.0, warmup_epochs=0):
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if base_weight < 0:
        raise ValueError("base_weight must be non-negative.")
    if warmup_epochs == 0:
        return base_weight
    progress = min(epoch_index + 1, warmup_epochs) / warmup_epochs
    return base_weight * progress
