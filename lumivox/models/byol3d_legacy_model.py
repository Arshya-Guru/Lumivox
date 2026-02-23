"""Legacy BYOL3D model wrapper.

Uses the original UNetEncoder3D from the old byol3d repo.
Kept as-is for backward compatibility and ablation against fair versions.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumivox.encoders.unet_encoder_3d import UNetEncoder3D


class LegacyProjectionMLP(nn.Module):
    """2-layer MLP: Linear -> BN -> ReLU -> Linear (original byol3d)."""

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyPredictionMLP(nn.Module):
    """Predictor MLP (same architecture as projector, original byol3d)."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 2048, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LegacyOnlineNetwork(nn.Module):
    """Legacy BYOL online network: UNetEncoder3D + projector + predictor."""

    def __init__(
        self,
        encoder: UNetEncoder3D,
        projector_hidden_dim: int = 2048,
        projector_output_dim: int = 256,
        predictor_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = LegacyProjectionMLP(
            input_dim=encoder.repr_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )
        self.predictor = LegacyPredictionMLP(
            input_dim=projector_output_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projector_output_dim,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.encoder(x, return_skips=False)
        z = self.projector(y)
        q = self.predictor(z)
        return {"projection": z, "prediction": q}


class LegacyTargetNetwork(nn.Module):
    """Legacy BYOL target network: UNetEncoder3D + projector (no predictor)."""

    def __init__(
        self,
        encoder: UNetEncoder3D,
        projector_hidden_dim: int = 2048,
        projector_output_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = LegacyProjectionMLP(
            input_dim=encoder.repr_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.encoder(x, return_skips=False)
        z = self.projector(y)
        return {"projection": z}


class BYOL3DLegacyModel(nn.Module):
    """Full legacy BYOL3D model with online and target networks."""

    def __init__(self, online: LegacyOnlineNetwork, target: LegacyTargetNetwork):
        super().__init__()
        self.online = online
        self.target = target

    @classmethod
    def create_pair(
        cls,
        in_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 5,
        encoder_channels: Optional[Sequence[int]] = None,
        use_residuals: bool = True,
        projector_hidden_dim: int = 2048,
        projector_output_dim: int = 256,
        predictor_hidden_dim: int = 2048,
    ) -> "BYOL3DLegacyModel":
        """Create matched online + target pair with shared initial weights."""
        encoder = UNetEncoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_levels=num_levels,
            channels=encoder_channels,
            use_residuals=use_residuals,
        )
        online = LegacyOnlineNetwork(
            encoder=encoder,
            projector_hidden_dim=projector_hidden_dim,
            projector_output_dim=projector_output_dim,
            predictor_hidden_dim=predictor_hidden_dim,
        )

        target_encoder = copy.deepcopy(encoder)
        target = LegacyTargetNetwork(
            encoder=target_encoder,
            projector_hidden_dim=projector_hidden_dim,
            projector_output_dim=projector_output_dim,
        )
        target.projector.load_state_dict(online.projector.state_dict())

        for p in target.parameters():
            p.requires_grad = False

        return cls(online=online, target=target)

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], ...]:
        online_out1 = self.online(view1)
        online_out2 = self.online(view2)

        with torch.no_grad():
            target_out1 = self.target(view1)
            target_out2 = self.target(view2)

        return online_out1, online_out2, target_out1, target_out2
