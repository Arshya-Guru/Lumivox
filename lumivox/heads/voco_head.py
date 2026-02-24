"""Legacy VoCo 3-layer IN1d projection head.

Kept for reference/comparison only. Not used by any Lumivox model.
This was the head used in the original nnssl SimCLR via VoCoArchitecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VoCoProjectionHead(nn.Module):
    """3-layer MLP with InstanceNorm1d (legacy VoCo architecture)."""

    def __init__(
        self,
        total_channels: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
    ):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(total_channels, hidden_dim),
            nn.InstanceNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.InstanceNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
