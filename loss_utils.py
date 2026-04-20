import torch


def rgb_to_grayscale(image):
    weights = torch.tensor(
        [0.299, 0.587, 0.114], device=image.device, dtype=image.dtype
    ).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def grayscale_l1_loss(preds, labels, criterion):
    gray_preds = rgb_to_grayscale(preds)
    gray_labels = rgb_to_grayscale(labels)
    return criterion(gray_preds, gray_labels)
