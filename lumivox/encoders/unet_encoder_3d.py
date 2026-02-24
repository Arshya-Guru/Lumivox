"""Legacy UNetEncoder3D from the original BYOL3D implementation.

Kept as-is for backward compatibility and ablation against fair versions.
Only used by byol3d-legacy model.

Channels [32, 64, 128, 256, 320], 5 levels, ResConvBlock3D,
InstanceNorm3d, LeakyReLU.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two 3x3x3 convolutions with InstanceNorm + LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResConvBlock3D(nn.Module):
    """Residual variant: adds a skip from input to output."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class UNetEncoder3D(nn.Module):
    """Contracting path of a 3D U-Net (legacy BYOL3D encoder).

    Default channels [32, 64, 128, 256, 320] match the original byol3d config.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 5,
        channels: Optional[Sequence[int]] = None,
        use_residuals: bool = True,
    ):
        super().__init__()
        if channels is None:
            channels = []
            c = base_channels
            for i in range(num_levels):
                channels.append(min(c, 320))
                c *= 2
        self.channels = list(channels)
        self.num_levels = num_levels

        Block = ResConvBlock3D if use_residuals else ConvBlock3D

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(num_levels):
            c_in = in_channels if i == 0 else self.channels[i - 1]
            c_out = self.channels[i]
            self.encoders.append(Block(c_in, c_out))
            if i < num_levels - 1:
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.repr_dim = self.channels[-1]

    def forward(
        self, x: torch.Tensor, return_skips: bool = False
    ):
        skips = []
        out = x
        for i in range(self.num_levels):
            out = self.encoders[i](out)
            if i < self.num_levels - 1:
                skips.append(out)
                out = self.pools[i](out)

        if return_skips:
            return out, skips

        return self.gap(out).flatten(1)


class UNetDecoder3D(nn.Module):
    """Expanding path of a 3D U-Net for legacy fine-tuning."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        num_classes: int = 1,
        use_residuals: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        Block = ResConvBlock3D if use_residuals else ConvBlock3D

        dec_channels = list(reversed(encoder_channels[:-1]))
        bottleneck_ch = encoder_channels[-1]

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        in_ch = bottleneck_ch
        for out_ch in dec_channels:
            self.upconvs.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.dec_blocks.append(Block(out_ch * 2, out_ch))
            in_ch = out_ch

        self.final_conv = nn.Conv3d(dec_channels[-1], num_classes, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
    ) -> torch.Tensor:
        import torch.nn.functional as F

        skips = list(reversed(skips))
        x = bottleneck
        for i, (upconv, dec_block) in enumerate(zip(self.upconvs, self.dec_blocks)):
            x = upconv(x)
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        return self.final_conv(x)


class SegmentationUNet3D(nn.Module):
    """Full 3D U-Net for legacy segmentation fine-tuning."""

    def __init__(
        self,
        encoder: UNetEncoder3D,
        num_classes: int = 1,
        use_residuals: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder3D(
            encoder_channels=encoder.channels,
            num_classes=num_classes,
            use_residuals=use_residuals,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder(x, return_skips=True)
        return self.decoder(bottleneck, skips)

    @classmethod
    def from_byol_encoder(
        cls,
        byol_encoder: UNetEncoder3D,
        num_classes: int = 1,
        use_residuals: bool = True,
        freeze_encoder: bool = False,
    ) -> "SegmentationUNet3D":
        import copy

        encoder = copy.deepcopy(byol_encoder)
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
        return cls(encoder=encoder, num_classes=num_classes, use_residuals=use_residuals)
