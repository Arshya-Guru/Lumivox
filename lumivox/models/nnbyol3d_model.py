"""nnBYOL3D model: online(encoder+projector+predictor) + target(encoder+projector) + EMA.

Fair BYOL3D implementation using the same encoder backbone, projection head
dimensions, and augmentation base as SimCLR. The ONLY differences from SimCLR:
  (a) BYOL regression loss instead of NT-Xent
  (b) EMA target network
  (c) Predictor MLP on the online branch
  (d) Asymmetric view augmentation (different blur probs per view)
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn

from lumivox.encoders.resenc_l import build_resenc_l
from lumivox.heads.prediction import PredictionMLP
from lumivox.heads.projection import ProjectionMLP


class OnlineNetwork(nn.Module):
    """BYOL online network: encoder + projector + predictor."""

    def __init__(
        self,
        num_input_channels: int = 1,
        proj_hidden_dim: int = 4096,
        proj_output_dim: int = 256,
        pred_hidden_dim: int = 4096,
    ):
        super().__init__()
        self.encoder = build_resenc_l(num_input_channels)
        encoder_output_dim = self.encoder.output_channels[-1]
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.projector = ProjectionMLP(
            input_dim=encoder_output_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )
        self.predictor = PredictionMLP(
            input_dim=proj_output_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=proj_output_dim,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skips = self.encoder(x)
        h = self.gap(skips[-1]).flatten(1)
        z = self.projector(h)
        q = self.predictor(z)
        return {"projection": z, "prediction": q}


class TargetNetwork(nn.Module):
    """BYOL target network: encoder + projector (no predictor).

    All parameters are updated via EMA, not gradient.
    """

    def __init__(
        self,
        num_input_channels: int = 1,
        proj_hidden_dim: int = 4096,
        proj_output_dim: int = 256,
    ):
        super().__init__()
        self.encoder = build_resenc_l(num_input_channels)
        encoder_output_dim = self.encoder.output_channels[-1]
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.projector = ProjectionMLP(
            input_dim=encoder_output_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skips = self.encoder(x)
        h = self.gap(skips[-1]).flatten(1)
        z = self.projector(h)
        return {"projection": z}


class NnBYOL3DModel(nn.Module):
    """Full nnBYOL3D model with online and target networks.

    Provides create_pair() class method for convenient construction.
    """

    def __init__(self, online: OnlineNetwork, target: TargetNetwork):
        super().__init__()
        self.online = online
        self.target = target

    @classmethod
    def create_pair(
        cls,
        num_input_channels: int = 1,
        proj_hidden_dim: int = 4096,
        proj_output_dim: int = 256,
        pred_hidden_dim: int = 4096,
    ) -> "NnBYOL3DModel":
        """Create matched online + target pair with shared initial weights."""
        online = OnlineNetwork(
            num_input_channels=num_input_channels,
            proj_hidden_dim=proj_hidden_dim,
            proj_output_dim=proj_output_dim,
            pred_hidden_dim=pred_hidden_dim,
        )
        target = TargetNetwork(
            num_input_channels=num_input_channels,
            proj_hidden_dim=proj_hidden_dim,
            proj_output_dim=proj_output_dim,
        )

        # Initialize target from online weights
        target.encoder.load_state_dict(online.encoder.state_dict())
        target.projector.load_state_dict(online.projector.state_dict())

        # Freeze target parameters
        for p in target.parameters():
            p.requires_grad = False

        return cls(online=online, target=target)

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        """Forward both views through online and target.

        Returns:
            (online_out1, online_out2, target_out1, target_out2)
        """
        online_out1 = self.online(view1)
        online_out2 = self.online(view2)

        with torch.no_grad():
            target_out1 = self.target(view1)
            target_out2 = self.target(view2)

        return online_out1, online_out2, target_out1, target_out2
