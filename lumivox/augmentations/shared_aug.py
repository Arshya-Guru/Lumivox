"""Unified augmentation pipeline for SimCLR and nnBYOL3D.

All augmentations operate on [B, 1, D, H, W] float tensors.
The only difference between methods is the blur probability for view2:
  - SimCLR: symmetric (p=0.5 both views)
  - nnBYOL3D: asymmetric (view1 p=0.5, view2 p=0.1)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------


def random_flip_3d(x: torch.Tensor) -> torch.Tensor:
    """Random flip along each spatial axis with p=0.5 independently."""
    for dim in (2, 3, 4):
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, [dim])
    return x


def random_rot90_3d(x: torch.Tensor) -> torch.Tensor:
    """Random 90-degree rotation around a random axis pair."""
    if torch.rand(1).item() < 0.5:
        axis_pairs = [(2, 3), (2, 4), (3, 4)]
        pair = axis_pairs[torch.randint(len(axis_pairs), (1,)).item()]
        k = torch.randint(1, 4, (1,)).item()
        x = torch.rot90(x, k, pair)
    return x


def gaussian_noise(
    x: torch.Tensor,
    var_range: Tuple[float, float] = (0.0, 0.03),
    apply_prob: float = 0.2,
) -> torch.Tensor:
    """Additive Gaussian noise."""
    if torch.rand(1).item() > apply_prob:
        return x
    var = torch.empty(1).uniform_(*var_range).item()
    std = var**0.5
    noise = torch.randn_like(x) * std
    return x + noise


def gaussian_blur_3d(
    x: torch.Tensor,
    sigma_range: Tuple[float, float] = (0.1, 1.0),
    kernel_size: int = 5,
    apply_prob: float = 0.5,
) -> torch.Tensor:
    """3D Gaussian blur with random sigma, separable implementation."""
    if torch.rand(1).item() > apply_prob:
        return x

    sigma = torch.empty(1).uniform_(*sigma_range).item()
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - half
    kernel_1d = torch.exp(-0.5 * (coords / max(sigma, 1e-8)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    B, C_in, D, H, W = x.shape
    pad = half

    # Along D
    k_d = kernel_1d.view(1, 1, -1, 1, 1)
    x = F.pad(x, (0, 0, 0, 0, pad, pad), mode="replicate")
    x = F.conv3d(x, k_d.expand(C_in, -1, -1, -1, -1), groups=C_in)

    # Along H
    k_h = kernel_1d.view(1, 1, 1, -1, 1)
    x = F.pad(x, (0, 0, pad, pad, 0, 0), mode="replicate")
    x = F.conv3d(x, k_h.expand(C_in, -1, -1, -1, -1), groups=C_in)

    # Along W
    k_w = kernel_1d.view(1, 1, 1, 1, -1)
    x = F.pad(x, (pad, pad, 0, 0, 0, 0), mode="replicate")
    x = F.conv3d(x, k_w.expand(C_in, -1, -1, -1, -1), groups=C_in)

    return x


def intensity_jitter(
    x: torch.Tensor,
    brightness_range: float = 0.1,
    contrast_range: Tuple[float, float] = (0.75, 1.25),
    apply_prob: float = 0.8,
) -> torch.Tensor:
    """Brightness and contrast jitter for single-channel volumes."""
    if torch.rand(1).item() > apply_prob:
        return x
    brightness = (2 * torch.rand(1).item() - 1) * brightness_range
    x = x + brightness
    contrast = torch.empty(1).uniform_(*contrast_range).item()
    mean = x.mean()
    x = (x - mean) * contrast + mean
    return x


def gamma_correction(
    x: torch.Tensor,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
    retain_stats: bool = True,
    apply_prob: float = 0.5,
) -> torch.Tensor:
    """Random gamma correction."""
    if torch.rand(1).item() > apply_prob:
        return x
    gamma = torch.empty(1).uniform_(*gamma_range).item()
    if retain_stats:
        mean_before = x.mean()
        std_before = x.std() + 1e-8
    x_min = x.min()
    x = (x - x_min).clamp(min=1e-8)
    x = x.pow(gamma)
    if retain_stats:
        mean_after = x.mean()
        std_after = x.std() + 1e-8
        x = (x - mean_after) / std_after * std_before + mean_before
    return x


def simulate_low_resolution(
    x: torch.Tensor,
    zoom_range: Tuple[float, float] = (0.5, 1.0),
    apply_prob: float = 0.1,
    p_per_channel: float = 0.5,
    order_downsample: int = 0,
    order_upsample: int = 3,
) -> torch.Tensor:
    """Simulate low resolution by downsampling and upsampling.

    PyTorch implementation of batchgenerators' SimulateLowResolutionTransform.
    """
    if torch.rand(1).item() > apply_prob:
        return x

    B, C, D, H, W = x.shape
    result = x.clone()

    down_mode = "nearest" if order_downsample == 0 else "trilinear"
    up_mode = "trilinear" if order_upsample >= 3 else "nearest"

    for c in range(C):
        if torch.rand(1).item() > p_per_channel:
            continue
        zoom = torch.empty(1).uniform_(*zoom_range).item()
        if zoom >= 1.0:
            continue
        target_shape = [max(1, int(round(s * zoom))) for s in (D, H, W)]
        down = F.interpolate(
            result[:, c : c + 1],
            size=target_shape,
            mode=down_mode,
        )
        up = F.interpolate(
            down,
            size=(D, H, W),
            mode=up_mode,
            align_corners=False if up_mode == "trilinear" else None,
        )
        result[:, c : c + 1] = up

    return result


def brightness_gradient_3d(
    x: torch.Tensor,
    strength_range: Tuple[float, float] = (-0.3, 0.3),
    apply_prob: float = 0.2,
) -> torch.Tensor:
    """Additive linear brightness gradient along a random axis (LSFM-specific)."""
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
    return x + gradient


# ---------------------------------------------------------------------------
# Composed augmentation pipeline
# ---------------------------------------------------------------------------


class SharedAugmentation:
    """Composed augmentation pipeline for one SSL view.

    Applies: spatial (flip, rot90) -> intensity (noise, blur, jitter,
    gamma, low-res, gradient).

    Input:  [B, 1, D, H, W] float tensor
    Output: [B, 1, D, H, W] float tensor
    """

    def __init__(
        self,
        random_flip: bool = True,
        random_rot90: bool = True,
        gaussian_noise_cfg: Optional[dict] = None,
        gaussian_blur_cfg: Optional[dict] = None,
        intensity_jitter_cfg: Optional[dict] = None,
        gamma_correction_cfg: Optional[dict] = None,
        simulate_low_resolution_cfg: Optional[dict] = None,
        brightness_gradient_cfg: Optional[dict] = None,
    ):
        self.random_flip = random_flip
        self.random_rot90 = random_rot90
        self.gaussian_noise_cfg = gaussian_noise_cfg or {}
        self.gaussian_blur_cfg = gaussian_blur_cfg or {}
        self.intensity_jitter_cfg = intensity_jitter_cfg or {}
        self.gamma_correction_cfg = gamma_correction_cfg or {}
        self.simulate_low_resolution_cfg = simulate_low_resolution_cfg or {}
        self.brightness_gradient_cfg = brightness_gradient_cfg or {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.random_flip:
            x = random_flip_3d(x)
        if self.random_rot90:
            x = random_rot90_3d(x)
        if self.gaussian_noise_cfg:
            x = gaussian_noise(x, **self.gaussian_noise_cfg)
        if self.gaussian_blur_cfg:
            x = gaussian_blur_3d(x, **self.gaussian_blur_cfg)
        if self.intensity_jitter_cfg:
            x = intensity_jitter(x, **self.intensity_jitter_cfg)
        if self.gamma_correction_cfg:
            x = gamma_correction(x, **self.gamma_correction_cfg)
        if self.simulate_low_resolution_cfg:
            x = simulate_low_resolution(x, **self.simulate_low_resolution_cfg)
        if self.brightness_gradient_cfg:
            x = brightness_gradient_3d(x, **self.brightness_gradient_cfg)
        return x

    def __repr__(self) -> str:
        parts = [f"SharedAugmentation(flip={self.random_flip}, rot90={self.random_rot90}"]
        for name in [
            "gaussian_noise_cfg",
            "gaussian_blur_cfg",
            "intensity_jitter_cfg",
            "gamma_correction_cfg",
            "simulate_low_resolution_cfg",
            "brightness_gradient_cfg",
        ]:
            cfg = getattr(self, name)
            if cfg:
                parts.append(f"  {name}={cfg}")
        return "\n".join(parts) + ")"


def get_augmentation_config(method: str = "simclr") -> Dict[str, dict]:
    """Return augmentation configs for view1 and view2.

    Args:
        method: 'simclr' (symmetric) or 'nnbyol3d' (asymmetric blur).
    """
    base = dict(
        random_flip=True,
        random_rot90=True,
        gaussian_noise_cfg=dict(var_range=(0.0, 0.03), apply_prob=0.2),
        intensity_jitter_cfg=dict(
            brightness_range=0.1, contrast_range=(0.75, 1.25), apply_prob=0.8
        ),
        gamma_correction_cfg=dict(gamma_range=(0.7, 1.5), retain_stats=True, apply_prob=0.5),
        simulate_low_resolution_cfg=dict(
            zoom_range=(0.5, 1.0), apply_prob=0.1, p_per_channel=0.5,
            order_downsample=0, order_upsample=3,
        ),
        brightness_gradient_cfg=dict(strength_range=(-0.3, 0.3), apply_prob=0.2),
    )

    if method == "simclr":
        view1 = {**base, "gaussian_blur_cfg": dict(sigma_range=(0.1, 1.0), apply_prob=0.5)}
        view2 = {**base, "gaussian_blur_cfg": dict(sigma_range=(0.1, 1.0), apply_prob=0.5)}
    elif method == "nnbyol3d":
        view1 = {**base, "gaussian_blur_cfg": dict(sigma_range=(0.1, 1.0), apply_prob=0.5)}
        view2 = {**base, "gaussian_blur_cfg": dict(sigma_range=(0.1, 1.0), apply_prob=0.1)}
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simclr' or 'nnbyol3d'.")

    return dict(view1=view1, view2=view2)
