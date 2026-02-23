"""Shared projection head for SimCLR and nnBYOL3D.

2-layer MLP: Linear -> BN1d -> ReLU -> Linear (no bias on final layer).
Uses BatchNorm1d because:
  - BYOL paper shows BN is critical for preventing collapse (Table 5)
  - Original SimCLR paper also uses BN in the projection head
  - InstanceNorm1d removes inter-sample information

Default dims: input=320 (ResEncL bottleneck after GAP),
hidden=4096, output=256.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionMLP(nn.Module):
    """2-layer MLP projection head shared by SimCLR and nnBYOL3D."""

    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 4096,
        output_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
