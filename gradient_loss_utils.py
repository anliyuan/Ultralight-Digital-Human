import torch


def rgb_to_grayscale(image):
    weights = torch.tensor(
        [0.299, 0.587, 0.114], device=image.device, dtype=image.dtype
    ).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def image_gradients(image):
    gray = rgb_to_grayscale(image)
    gradient_h = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gradient_v = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    return gradient_h, gradient_v


def gradient_l1_loss(preds, labels, criterion):
    pred_h, pred_v = image_gradients(preds)
    label_h, label_v = image_gradients(labels)
    return criterion(pred_h, label_h) + criterion(pred_v, label_v)
