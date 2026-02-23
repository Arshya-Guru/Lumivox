"""Export encoder weights for nnU-Net downstream fine-tuning.

Handles extracting encoder state_dicts from pretraining checkpoints
of all three model types.
"""

from __future__ import annotations

from typing import Dict

import torch


def extract_encoder_weights(checkpoint_path: str, model_type: str) -> Dict[str, torch.Tensor]:
    """Extract encoder state_dict from a pretraining checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint.
        model_type: 'simclr', 'nnbyol3d', or 'byol3d-legacy'.

    Returns:
        state_dict loadable into ResidualEncoder (for simclr/nnbyol3d)
        or UNetEncoder3D (for byol3d-legacy).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Try direct encoder_state_dict first
    if "encoder_state_dict" in ckpt:
        return ckpt["encoder_state_dict"]

    if model_type == "simclr":
        sd = ckpt["model_state_dict"]
        encoder_sd = {
            k.replace("encoder.", ""): v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }
        return encoder_sd

    elif model_type == "nnbyol3d":
        sd = ckpt["online_state_dict"]
        encoder_sd = {
            k.replace("encoder.", ""): v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }
        return encoder_sd

    elif model_type == "byol3d-legacy":
        sd = ckpt.get("online_state_dict", ckpt.get("model_state_dict", {}))
        encoder_sd = {
            k.replace("encoder.", ""): v
            for k, v in sd.items()
            if k.startswith("encoder.")
        }
        return encoder_sd

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_adaptation_plan(
    crop_size: int = 96,
    num_input_channels: int = 1,
) -> dict:
    """Build an nnssl-compatible adaptation plan dict.

    This can be saved in checkpoints for downstream nnU-Net loading.
    """
    return {
        "architecture_plans": {
            "arch_class_name": "ResEncL",
        },
        "pretrain_num_input_channels": num_input_channels,
        "recommended_downstream_patchsize": (crop_size, crop_size, crop_size),
        "key_to_encoder": "encoder.stages",
        "key_to_stem": "encoder.stem",
        "keys_to_in_proj": [
            "encoder.stem.convs.0.conv",
            "encoder.stem.convs.0.all_modules.0",
        ],
    }
