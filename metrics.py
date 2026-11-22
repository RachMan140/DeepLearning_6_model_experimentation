"""
metrics.py -- image-quality metrics implemented in plain PyTorch.

SSIM (Structural Similarity) compares two images by local luminance, contrast
and structure instead of raw pixel differences. Range is roughly [-1, 1];
1.0 means identical. Higher is better.

Variant used (state this in your report): 11x11 Gaussian window, sigma = 1.5,
K1 = 0.01, K2 = 0.03, data range 1.0, "valid" convolution (no padding).
"""
import torch
import torch.nn.functional as F


def _gaussian_window(size, sigma, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]   # shape [1, 1, size, size]


def ssim_batch(x, y, data_range=1.0, size=11, sigma=1.5):
    """Per-image SSIM for two batches of shape [B, 1, H, W]. Returns shape [B]."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    w = _gaussian_window(size, sigma, x.device, x.dtype)

    mu_x = F.conv2d(x, w)
    mu_y = F.conv2d(y, w)
    var_x = F.conv2d(x * x, w) - mu_x ** 2
    var_y = F.conv2d(y * y, w) - mu_y ** 2
    cov_xy = F.conv2d(x * y, w) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim_map.flatten(1).mean(dim=1)
