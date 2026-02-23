"""ResEncL encoder shared by SimCLR and nnBYOL3D.

Identical to nnssl's architecture_registry.get_res_enc_l(), but returns
only the encoder (not the full U-Net with decoder). Uses
dynamic_network_architectures' ResidualEncoder directly.

Architecture: 6 stages, features [32, 64, 128, 256, 320, 320]
Strides: [1,1,1] then [2,2,2] x 5
Blocks per stage: [1, 3, 4, 6, 6, 6]
Normalization: InstanceNorm3d (eps=1e-5, affine=True)
Activation: LeakyReLU(inplace=True)

When return_skips=True (always), forward returns a list of tensors:
  [B,32,96,96,96], [B,64,48,48,48], ..., [B,320,3,3,3]
for a 96^3 input.
"""

from __future__ import annotations

import torch.nn as nn
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder


def build_resenc_l(num_input_channels: int = 1) -> ResidualEncoder:
    """Build the ResEncL encoder, identical to nnssl's architecture_registry.py."""
    n_stages = 6
    encoder = ResidualEncoder(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * n_stages,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        return_skips=True,
        disable_default_stem=False,
        stem_channels=None,
    )
    return encoder
