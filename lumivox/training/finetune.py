"""Unified fine-tuning pipeline for all three models.

All models produce a pretrained encoder. The fine-tuning pipeline is
identical for simclr and nnbyol3d (same decoder, same optimizer, same
augmentations). The byol3d-legacy model uses its own legacy decoder.

Usage:
    python -m lumivox.training.finetune --checkpoint ./checkpoints/simclr/pretrain_final.pt
    python -m lumivox.training.finetune --data-dir /path/to/labeled  # from scratch
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumivox.adaptation.checkpoint_export import extract_encoder_weights


# ---------------------------------------------------------------------------
# Loss functions for fine-tuning
# ---------------------------------------------------------------------------

def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    """Soft Dice loss for binary segmentation."""
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
) -> torch.Tensor:
    """Combined Dice + CrossEntropy loss (nnU-Net standard)."""
    probs = torch.sigmoid(logits)
    d_loss = dice_loss(probs, target)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target)
    return dice_weight * d_loss + ce_weight * ce_loss


# ---------------------------------------------------------------------------
# Build segmentation model
# ---------------------------------------------------------------------------

def build_segmentation_model(
    checkpoint_path: Optional[str] = None,
    num_classes: int = 1,
    deep_supervision: bool = True,
) -> nn.Module:
    """Build full U-Net for segmentation, optionally loading pretrained encoder.

    For simclr/nnbyol3d: uses ResidualEncoderUNet from dynamic_network_architectures.
    For byol3d-legacy: uses the legacy SegmentationUNet3D.
    For from-scratch: builds ResidualEncoderUNet with random init.
    """
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_type = ckpt.get("model_type", "simclr")
    else:
        model_type = "from_scratch"

    if model_type in ("simclr", "nnbyol3d", "from_scratch"):
        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

        n_stages = 6
        model = ResidualEncoderUNet(
            input_channels=1,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3]] * n_stages,
            strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=num_classes,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )

        if checkpoint_path is not None and model_type != "from_scratch":
            encoder_sd = extract_encoder_weights(checkpoint_path, model_type)
            model.encoder.load_state_dict(encoder_sd, strict=True)
            print(f"Loaded pretrained encoder weights ({model_type})")

        return model

    elif model_type == "byol3d-legacy":
        from lumivox.encoders.unet_encoder_3d import SegmentationUNet3D, UNetEncoder3D

        encoder_sd = extract_encoder_weights(checkpoint_path, model_type)
        cfg = ckpt.get("config", {})
        encoder = UNetEncoder3D(
            in_channels=cfg.get("num_input_channels", 1) if isinstance(cfg, dict) else 1,
            base_channels=cfg.get("base_channels", 32) if isinstance(cfg, dict) else 32,
            num_levels=cfg.get("num_levels", 5) if isinstance(cfg, dict) else 5,
        )
        encoder.load_state_dict(encoder_sd)
        print(f"Loaded pretrained legacy encoder weights")
        return SegmentationUNet3D(encoder=encoder, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    num_classes: int = 1,
    epochs: int = 200,
    batch_size: int = 2,
    lr: float = 1e-3,
    encoder_lr_factor: float = 0.1,
    weight_decay: float = 1e-2,
    crop_size: int = 96,
    precision: str = "bf16-mixed",
    deep_supervision: bool = True,
    save_dir: str = "./checkpoints/finetune",
    seed: int = 42,
) -> nn.Module:
    """Fine-tune segmentation model."""
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_segmentation_model(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        deep_supervision=deep_supervision,
    )
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_total:,} total params, {n_train:,} trainable")

    # Optimizer with differential LR
    if hasattr(model, "encoder"):
        encoder_params = list(model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in model.parameters() if id(p) not in encoder_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": lr * encoder_lr_factor},
                {"params": other_params, "lr": lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    # PolyLR scheduler
    max_steps = epochs * max(1, 50)  # placeholder steps per epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0, (1 - step / max(max_steps, 1)) ** 0.9)
    )

    # Deep supervision weights
    ds_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]

    # Mixed precision
    use_amp = precision != "32" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if "bf16" in precision else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    print(f"Fine-tuning for {epochs} epochs, lr={lr}, encoder_lr={lr*encoder_lr_factor}")
    print("NOTE: Replace placeholder data loop with real labeled DataLoader")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Placeholder training loop (replace with real data)
        for step in range(5):
            images = torch.rand(batch_size, 1, crop_size, crop_size, crop_size, device=device)
            masks = (torch.rand(batch_size, 1, crop_size, crop_size, crop_size, device=device) > 0.8).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(images)

                if deep_supervision and isinstance(outputs, (list, tuple)):
                    loss = torch.tensor(0.0, device=device)
                    for i, (out, w) in enumerate(zip(outputs, ds_weights)):
                        if i < len(outputs):
                            target_ds = F.interpolate(
                                masks, size=out.shape[2:], mode="nearest"
                            )
                            loss = loss + w * dice_ce_loss(out, target_ds)
                else:
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    loss = dice_ce_loss(outputs, masks)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4d}/{epochs} | loss={epoch_loss/5:.4f}")

    # Save final model
    final_path = save_path / "finetune_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "source_checkpoint": checkpoint_path,
        },
        final_path,
    )
    print(f"Fine-tuning complete. Saved: {final_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Lumivox Segmentation Fine-tuning")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument(
        "--precision", type=str, default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--save-dir", type=str, default="./checkpoints/finetune")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    finetune(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr_factor=args.encoder_lr_factor,
        weight_decay=args.weight_decay,
        crop_size=args.crop_size,
        precision=args.precision,
        save_dir=args.save_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
