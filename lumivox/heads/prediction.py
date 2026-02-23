"""BYOL prediction head (nnBYOL3D only).

Same architecture as ProjectionMLP. Only attached to the online branch.
Paper: "The predictor uses the same architecture as the projector."
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictionMLP(nn.Module):
    """2-layer MLP predictor for nnBYOL3D online branch."""

    def __init__(
        self,
        input_dim: int = 256,
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
