"""Legacy BYOL3D augmentation pipeline.

Kept verbatim from the original byol3d repo. Only used by byol3d-legacy.
Does NOT include SimulateLowResolution and uses original probability values.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


augment_config = dict(
    view1=dict(
        random_flip=True,
        random_rot90=True,
        intensity_jitter=dict(
            apply_prob=0.8,
            brightness_range=0.1,
            contrast_range=(0.75, 1.25),
        ),
        gamma_correction=dict(apply_prob=0.5, gamma_range=(0.7, 1.5)),
        gaussian_blur=dict(apply_prob=0.5, sigma_min=0.1, sigma_max=1.0, kernel_size=5),
        brightness_gradient=dict(apply_prob=0.2, strength_range=(-0.3, 0.3)),
        gaussian_noise=dict(apply_prob=0.3, var_range=(0.0, 0.03)),
    ),
    view2=dict(
        random_flip=True,
        random_rot90=True,
        intensity_jitter=dict(
            apply_prob=0.8,
            brightness_range=0.1,
            contrast_range=(0.75, 1.25),
        ),
        gamma_correction=dict(apply_prob=0.5, gamma_range=(0.7, 1.5)),
        gaussian_blur=dict(apply_prob=0.1, sigma_min=0.1, sigma_max=1.0, kernel_size=5),
        brightness_gradient=dict(apply_prob=0.2, strength_range=(-0.3, 0.3)),
        gaussian_noise=dict(apply_prob=0.3, var_range=(0.0, 0.03)),
    ),
)


def _random_flip_3d(x: torch.Tensor) -> torch.Tensor:
    for dim in (2, 3, 4):
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, [dim])
    return x


def _random_rot90_3d(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        axis_pairs = [(2, 3), (2, 4), (3, 4)]
        pair = axis_pairs[torch.randint(len(axis_pairs), (1,)).item()]
        k = torch.randint(1, 4, (1,)).item()
        x = torch.rot90(x, k, pair)
    return x


def _intensity_jitter(
    x: torch.Tensor,
    brightness_range: float = 0.1,
    contrast_range: Tuple[float, float] = (0.75, 1.25),
    apply_prob: float = 0.8,
) -> torch.Tensor:
    if torch.rand(1).item() > apply_prob:
        return x
    brightness = (2 * torch.rand(1).item() - 1) * brightness_range
    x = x + brightness
    contrast = torch.empty(1).uniform_(*contrast_range).item()
    mean = x.mean()
    x = (x - mean) * contrast + mean
    return x.clamp(0, 1)


def _gamma_correction(
    x: torch.Tensor,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
    apply_prob: float = 0.5,
) -> torch.Tensor:
    if torch.rand(1).item() > apply_prob:
        return x
    gamma = torch.empty(1).uniform_(*gamma_range).item()
    x = x.clamp(min=1e-8)
    return x.pow(gamma).clamp(0, 1)


def _gaussian_blur_3d(
    x: torch.Tensor,
    sigma_min: float = 0.1,
    sigma_max: float = 1.0,
    kernel_size: int = 5,
    apply_prob: float = 0.5,
) -> torch.Tensor:
    if torch.rand(1).item() > apply_prob:
        return x
    sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item()
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - half
    kernel_1d = torch.exp(-0.5 * (coords / max(sigma, 1e-8)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    B, C_in, D, H, W = x.shape
    pad = half
    k_d = kernel_1d.view(1, 1, -1, 1, 1)
    x = F.pad(x, (0, 0, 0, 0, pad, pad), mode="replicate")
    x = F.conv3d(x, k_d.expand(C_in, -1, -1, -1, -1), groups=C_in)
    k_h = kernel_1d.view(1, 1, 1, -1, 1)
    x = F.pad(x, (0, 0, pad, pad, 0, 0), mode="replicate")
    x = F.conv3d(x, k_h.expand(C_in, -1, -1, -1, -1), groups=C_in)
    k_w = kernel_1d.view(1, 1, 1, 1, -1)
    x = F.pad(x, (pad, pad, 0, 0, 0, 0), mode="replicate")
    x = F.conv3d(x, k_w.expand(C_in, -1, -1, -1, -1), groups=C_in)
    return x


def _brightness_gradient_3d(
    x: torch.Tensor,
    strength_range: Tuple[float, float] = (-0.3, 0.3),
    apply_prob: float = 0.2,
) -> torch.Tensor:
    if torch.rand(1).item() > apply_prob:
        return x
    strength = torch.empty(1).uniform_(*strength_range).item()
    axis = torch.randint(0, 3, (1,)).item()
    spatial_dim = x.shape[axis + 2]
    gradient = torch.linspace(
        -strength, strength, spatial_dim, device=x.device, dtype=x.dtype
    )
    shape = [1, 1, 1, 1, 1]
    shape[axis + 2] = spatial_dim
    gradient = gradient.view(*shape)
    return (x + gradient).clamp(0, 1)


def _gaussian_noise(
    x: torch.Tensor,
    var_range: Tuple[float, float] = (0.0, 0.03),
    apply_prob: float = 0.3,
) -> torch.Tensor:
    if torch.rand(1).item() > apply_prob:
        return x
    var = torch.empty(1).uniform_(*var_range).item()
    std = var**0.5
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(0, 1)


class LegacyAugment3D:
    """Legacy augmentation pipeline from original byol3d repo."""

    def __init__(
        self,
        random_flip: bool = True,
        random_rot90: bool = True,
        intensity_jitter: Optional[dict] = None,
        gamma_correction: Optional[dict] = None,
        gaussian_blur: Optional[dict] = None,
        brightness_gradient: Optional[dict] = None,
        gaussian_noise: Optional[dict] = None,
    ):
        self.random_flip = random_flip
        self.random_rot90 = random_rot90
        self.intensity_jitter_cfg = intensity_jitter or {}
        self.gamma_correction_cfg = gamma_correction or {}
        self.gaussian_blur_cfg = gaussian_blur or {}
        self.brightness_gradient_cfg = brightness_gradient or {}
        self.gaussian_noise_cfg = gaussian_noise or {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.random_flip:
            x = _random_flip_3d(x)
        if self.random_rot90:
            x = _random_rot90_3d(x)
        if self.intensity_jitter_cfg:
            x = _intensity_jitter(x, **self.intensity_jitter_cfg)
        if self.gamma_correction_cfg:
            x = _gamma_correction(x, **self.gamma_correction_cfg)
        if self.gaussian_blur_cfg:
            x = _gaussian_blur_3d(x, **self.gaussian_blur_cfg)
        if self.brightness_gradient_cfg:
            x = _brightness_gradient_3d(x, **self.brightness_gradient_cfg)
        if self.gaussian_noise_cfg:
            x = _gaussian_noise(x, **self.gaussian_noise_cfg)
        return x.clamp(0, 1)
