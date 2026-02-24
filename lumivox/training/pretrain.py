"""Unified pretraining loop for all three models.

Handles SimCLR, nnBYOL3D, and byol3d-legacy with a single training loop.
Uses plain PyTorch for single-GPU training.

Usage:
    python -m lumivox.training.pretrain --method simclr --data-dir /path/to/data
    python -m lumivox.training.pretrain --method nnbyol3d --data-dir /path/to/data
    python -m lumivox.training.pretrain --method byol3d-legacy --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumivox.data.dataset_blosc2 import create_dataloader
from lumivox.losses.ntxent import NTXentLoss
from lumivox.losses.regression import regression_loss
from lumivox.models.byol3d_legacy_model import BYOL3DLegacyModel
from lumivox.models.nnbyol3d_model import NnBYOL3DModel
from lumivox.models.simclr_model import SimCLRModel
from lumivox.training.schedules import CosineWarmupScheduler
from lumivox.utils.ema import cosine_ema_schedule, update_target_ema


def build_model(method: str, cfg: Dict[str, Any]) -> nn.Module:
    """Build the appropriate model based on method name."""
    if method == "simclr":
        return SimCLRModel(
            num_input_channels=cfg.get("num_input_channels", 1),
            proj_hidden_dim=cfg.get("proj_hidden_dim", 4096),
            proj_output_dim=cfg.get("proj_output_dim", 256),
        )
    elif method == "nnbyol3d":
        return NnBYOL3DModel.create_pair(
            num_input_channels=cfg.get("num_input_channels", 1),
            proj_hidden_dim=cfg.get("proj_hidden_dim", 4096),
            proj_output_dim=cfg.get("proj_output_dim", 256),
            pred_hidden_dim=cfg.get("pred_hidden_dim", 4096),
        )
    elif method == "byol3d-legacy":
        return BYOL3DLegacyModel.create_pair(
            in_channels=cfg.get("num_input_channels", 1),
            base_channels=cfg.get("base_channels", 32),
            num_levels=cfg.get("num_levels", 5),
            projector_hidden_dim=cfg.get("proj_hidden_dim", 2048),
            projector_output_dim=cfg.get("proj_output_dim", 256),
            predictor_hidden_dim=cfg.get("pred_hidden_dim", 2048),
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def build_optimizer(
    model: nn.Module,
    method: str,
    optimizer_type: str = "default",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build optimizer with proper weight decay exclusion."""
    if optimizer_type == "legacy" and method == "simclr":
        # SimCLR legacy: SGD + Nesterov momentum 0.99
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.99,
            nesterov=True,
        )

    # Default: AdamW with weight decay exclusion
    if method == "simclr":
        params = model.parameters()
    else:
        # For BYOL models, only optimize online parameters
        params = model.online.parameters()

    decay_params = []
    no_decay_params = []
    for name, p in (
        model.named_parameters()
        if method == "simclr"
        else model.online.named_parameters()
    ):
        if not p.requires_grad:
            continue
        if "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def train(
    method: str = "simclr",
    data_dir: str | None = None,
    crop_size: int = 96,
    batch_size: int = 16,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_type: str = "default",
    warmup_epochs: int = 10,
    base_ema: float = 0.99,
    save_dir: str = "./checkpoints",
    save_every: int = 50,
    precision: str = "bf16-mixed",
    seed: int = 42,
    num_workers: int = 4,
) -> None:
    """Run pretraining."""
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Build model
    cfg = dict(
        num_input_channels=1,
        proj_hidden_dim=4096 if method != "byol3d-legacy" else 2048,
        proj_output_dim=256,
        pred_hidden_dim=4096 if method != "byol3d-legacy" else 2048,
        base_channels=32,
        num_levels=5,
    )
    model = build_model(method, cfg)
    model = model.to(device)

    # Build data loader
    use_synthetic = data_dir is None
    dataloader = create_dataloader(
        data_dir=data_dir,
        method=method,
        batch_size=batch_size,
        crop_size=crop_size,
        num_workers=num_workers,
        pin_memory=True,
        synthetic=use_synthetic,
        synthetic_num_samples=max(1000, batch_size * 50),
        synthetic_volume_size=max(32, crop_size + 8),
    )

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, method, optimizer_type, lr, weight_decay)
    steps_per_epoch = len(dataloader)
    max_steps = epochs * steps_per_epoch
    warmup_steps = min(warmup_epochs * steps_per_epoch, max_steps // 10)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

    # Build loss
    if method == "simclr":
        loss_fn = NTXentLoss(
            batch_size=batch_size,
            temperature=0.5,
            similarity_function="cosine",
            device=device,
        )

    # Mixed precision
    use_amp = precision != "32" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if "bf16" in precision else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    print(f"Starting {method} pretraining: {epochs} epochs, batch_size={batch_size}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}, precision: {precision}")

    global_step = 0
    for epoch in range(epochs):
        model.train() if method == "simclr" else model.online.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            view1 = batch["view1"].to(device, non_blocking=True)
            view2 = batch["view2"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                if method == "simclr":
                    z_i = model(view1)
                    z_j = model(view2)
                    loss, acc = loss_fn(z_i, z_j)
                else:
                    # BYOL (nnbyol3d or legacy)
                    o1, o2, t1, t2 = model(view1, view2)
                    loss_12 = regression_loss(
                        o1["prediction"], t2["projection"].detach()
                    )
                    loss_21 = regression_loss(
                        o2["prediction"], t1["projection"].detach()
                    )
                    loss = loss_12 + loss_21

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters() if method == "simclr" else model.online.parameters(),
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters() if method == "simclr" else model.online.parameters(),
                    1.0,
                )
                optimizer.step()

            scheduler.step()

            # EMA update for BYOL models
            if method in ("nnbyol3d", "byol3d-legacy"):
                tau = cosine_ema_schedule(global_step, base_ema, max_steps)
                update_target_ema(model.online, model.target, tau)

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:>4d}/{epochs} | loss={avg_loss:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt = {
                "model_type": method,
                "epoch": epoch + 1,
                "global_step": global_step,
                "config": cfg,
            }

            if method == "simclr":
                ckpt["model_state_dict"] = model.state_dict()
                ckpt["encoder_state_dict"] = model.encoder.state_dict()
            else:
                ckpt["online_state_dict"] = model.online.state_dict()
                ckpt["target_state_dict"] = model.target.state_dict()
                ckpt["encoder_state_dict"] = model.online.encoder.state_dict()

            ckpt["optimizer_state_dict"] = optimizer.state_dict()

            # Adaptation plan for fair models
            if method in ("simclr", "nnbyol3d"):
                ckpt["adaptation_plan"] = {
                    "architecture_plans": {"arch_class_name": "ResEncL"},
                    "pretrain_num_input_channels": 1,
                    "recommended_downstream_patchsize": (
                        crop_size,
                        crop_size,
                        crop_size,
                    ),
                    "key_to_encoder": "encoder.stages",
                    "key_to_stem": "encoder.stem",
                    "keys_to_in_proj": [
                        "encoder.stem.convs.0.conv",
                        "encoder.stem.convs.0.all_modules.0",
                    ],
                }

            ckpt_path = save_path / f"pretrain_epoch{epoch+1:04d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = save_path / "pretrain_final.pt"
    torch.save(ckpt, final_path)
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Lumivox SSL Pretraining")
    parser.add_argument(
        "--method",
        type=str,
        default="simclr",
        choices=["simclr", "nnbyol3d", "byol3d-legacy"],
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="default",
        choices=["default", "legacy"],
        help="'legacy' breaks fairness: SGD+Nesterov for SimCLR",
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--base-ema", type=float, default=0.99)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    train(
        method=args.method,
        data_dir=args.data_dir,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        warmup_epochs=args.warmup_epochs,
        base_ema=args.base_ema,
        save_dir=args.save_dir,
        save_every=args.save_every,
        precision=args.precision,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
