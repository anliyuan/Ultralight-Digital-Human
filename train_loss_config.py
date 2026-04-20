def combine_training_losses(
    loss_pixel,
    loss_perceptual,
    perceptual_weight,
    sync_loss=None,
    syncnet_weight=10.0,
):
    total = loss_pixel + perceptual_weight * loss_perceptual
    if sync_loss is not None:
        total = total + syncnet_weight * sync_loss
    return total
