import torch
import torch.nn.functional as F


def gaussian_kernel(kernel_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def create_window(channels, kernel_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    kernel = gaussian_kernel(kernel_size, sigma, device=device, dtype=dtype)
    window = kernel.view(1, 1, kernel_size, kernel_size)
    return window.expand(channels, 1, kernel_size, kernel_size).contiguous()


def ssim_index(preds, labels, data_range=1.0, kernel_size=11, sigma=1.5):
    channels = preds.shape[1]
    window = create_window(
        channels,
        kernel_size=kernel_size,
        sigma=sigma,
        device=preds.device,
        dtype=preds.dtype,
    )

    padding = kernel_size // 2
    mu_x = F.conv2d(preds, window, padding=padding, groups=channels)
    mu_y = F.conv2d(labels, window, padding=padding, groups=channels)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(preds * preds, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(labels * labels, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(preds * labels, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator
    return ssim_map.mean()


def ssim_loss(preds, labels, data_range=1.0):
    return 1.0 - ssim_index(preds, labels, data_range=data_range)
