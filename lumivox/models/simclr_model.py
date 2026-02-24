"""SimCLR model: encoder + projector + NT-Xent.

Uses the shared ResEncL encoder and ProjectionMLP.
Forward returns L2-normalized projection vectors for NT-Xent loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumivox.encoders.resenc_l import build_resenc_l
from lumivox.heads.projection import ProjectionMLP


class SimCLRModel(nn.Module):
    """SimCLR model with shared encoder + projector.

    encoder(x) -> skips -> skips[-1] -> GAP -> flatten -> ProjectionMLP -> z
    NT-Xent loss computed on L2-normalized z vectors.
    """

    def __init__(
        self,
        num_input_channels: int = 1,
        proj_hidden_dim: int = 4096,
        proj_output_dim: int = 256,
    ):
        super().__init__()
        self.encoder = build_resenc_l(num_input_channels)
        # ResEncL bottleneck: 320 channels after GAP
        encoder_output_dim = self.encoder.output_channels[-1]
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.projector = ProjectionMLP(
            input_dim=encoder_output_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning L2-normalized projection.

        Args:
            x: [B, C, D, H, W] input volume.

        Returns:
            z: [B, proj_output_dim] L2-normalized projection vector.
        """
        skips = self.encoder(x)  # list of tensors, one per stage
        bottleneck = skips[-1]  # [B, 320, D', H', W']
        h = self.gap(bottleneck).flatten(1)  # [B, 320]
        z = self.projector(h)  # [B, proj_output_dim]
        return F.normalize(z, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder representation (320-dim after GAP)."""
        skips = self.encoder(x)
        return self.gap(skips[-1]).flatten(1)
