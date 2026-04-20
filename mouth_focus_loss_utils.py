import torch


def build_mouth_focus_mask(
    height,
    width,
    device,
    dtype,
    x_range=(0.2, 0.8),
    y_range=(0.35, 0.95),
):
    mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    x_start = int(width * x_range[0])
    x_end = int(width * x_range[1])
    y_start = int(height * y_range[0])
    y_end = int(height * y_range[1])
    mask[:, :, y_start:y_end, x_start:x_end] = 1.0
    return mask


def mouth_focus_l1_loss(preds, labels, x_range=(0.2, 0.8), y_range=(0.35, 0.95)):
    mask = build_mouth_focus_mask(
        preds.shape[-2],
        preds.shape[-1],
        device=preds.device,
        dtype=preds.dtype,
        x_range=x_range,
        y_range=y_range,
    )
    masked_diff = (preds - labels).abs() * mask
    normalizer = mask.sum() * preds.shape[1]
    return masked_diff.sum() / normalizer
