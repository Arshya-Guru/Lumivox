"""Lightning-based unified pretraining for all three models.

Handles multi-GPU DDP training via PyTorch Lightning.

Usage:
    python -m lumivox.training.pretrain_lightning --method simclr --data-dir /path
    python -m lumivox.training.pretrain_lightning --method nnbyol3d --devices 4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor
    from pytorch_lightning.strategies import DDPStrategy

from lumivox.data.dataset_blosc2 import create_dataloader
from lumivox.losses.ntxent import NTXentLoss
from lumivox.losses.regression import regression_loss
from lumivox.models.byol3d_legacy_model import BYOL3DLegacyModel
from lumivox.models.nnbyol3d_model import NnBYOL3DModel
from lumivox.models.simclr_model import SimCLRModel
from lumivox.training.schedules import CosineWarmupScheduler
from lumivox.utils.ema import cosine_ema_schedule, update_target_ema


class PretrainLightningModule(pl.LightningModule):
    """Unified Lightning module for all three SSL methods."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.method = cfg["method"]

        # Build model
        if self.method == "simclr":
            self.model = SimCLRModel(
                num_input_channels=cfg.get("num_input_channels", 1),
                proj_hidden_dim=cfg.get("proj_hidden_dim", 4096),
                proj_output_dim=cfg.get("proj_output_dim", 256),
            )
            self.loss_fn = NTXentLoss(
                batch_size=cfg["batch_size"],
                temperature=0.5,
                similarity_function="cosine",
            )
        elif self.method == "nnbyol3d":
            pair = NnBYOL3DModel.create_pair(
                num_input_channels=cfg.get("num_input_channels", 1),
                proj_hidden_dim=cfg.get("proj_hidden_dim", 4096),
                proj_output_dim=cfg.get("proj_output_dim", 256),
                pred_hidden_dim=cfg.get("pred_hidden_dim", 4096),
            )
            self.online = pair.online
            self.target = pair.target
        elif self.method == "byol3d-legacy":
            pair = BYOL3DLegacyModel.create_pair(
                in_channels=cfg.get("num_input_channels", 1),
                base_channels=cfg.get("base_channels", 32),
                num_levels=cfg.get("num_levels", 5),
                projector_hidden_dim=cfg.get("proj_hidden_dim", 2048),
                projector_output_dim=cfg.get("proj_output_dim", 256),
                predictor_hidden_dim=cfg.get("pred_hidden_dim", 2048),
            )
            self.online = pair.online
            self.target = pair.target

        # EMA config for BYOL models
        if self.method in ("nnbyol3d", "byol3d-legacy"):
            self.base_ema = cfg.get("base_ema", 0.99)
            self.max_steps = cfg.get("max_steps", 100000)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        view1 = batch["view1"]
        view2 = batch["view2"]

        if self.method == "simclr":
            z_i = self.model(view1)
            z_j = self.model(view2)
            # Update loss batch size if needed (DDP may change effective batch)
            self.loss_fn.device = self.device
            loss, acc = self.loss_fn(z_i, z_j)
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/acc", acc, sync_dist=True)
        else:
            self.online.train()
            o1 = self.online(view1)
            o2 = self.online(view2)
            with torch.no_grad():
                self.target.eval()
                t1 = self.target(view1)
                t2 = self.target(view2)

            loss_12 = regression_loss(o1["prediction"], t2["projection"].detach())
            loss_21 = regression_loss(o2["prediction"], t1["projection"].detach())
            loss = loss_12 + loss_21

            tau = cosine_ema_schedule(self.global_step, self.base_ema, self.max_steps)
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/tau", tau, prog_bar=True)

        self.log("train/lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update for BYOL models after each step."""
        if self.method in ("nnbyol3d", "byol3d-legacy"):
            tau = cosine_ema_schedule(self.global_step, self.base_ema, self.max_steps)
            update_target_ema(self.online, self.target, tau)

    def configure_optimizers(self):
        optimizer_type = self.cfg.get("optimizer_type", "default")

        if optimizer_type == "legacy" and self.method == "simclr":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.get("lr", 1e-2),
                weight_decay=self.cfg.get("weight_decay", 3e-5),
                momentum=0.99,
                nesterov=True,
            )
        else:
            if self.method == "simclr":
                params_iter = self.model.named_parameters()
            else:
                params_iter = self.online.named_parameters()

            decay_params = []
            no_decay_params = []
            for name, p in params_iter:
                if not p.requires_grad:
                    continue
                if "bias" in name or "norm" in name or "bn" in name:
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)

            optimizer = torch.optim.AdamW(
                [
                    {"params": decay_params, "weight_decay": self.cfg.get("weight_decay", 1e-4)},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                lr=self.cfg.get("lr", 1e-3),
                betas=(0.9, 0.999),
            )

        warmup_steps = self.cfg.get("warmup_steps", 1000)
        max_steps = self.cfg.get("max_steps", 100000)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps, max_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def train_dataloader(self):
        use_synthetic = self.cfg.get("data_dir") is None
        return create_dataloader(
            data_dir=self.cfg.get("data_dir"),
            method=self.method,
            batch_size=self.cfg["batch_size"],
            crop_size=self.cfg.get("crop_size", 96),
            num_workers=self.cfg.get("num_workers", 4),
            pin_memory=True,
            synthetic=use_synthetic,
            synthetic_num_samples=max(
                self.cfg.get("max_steps", 10000) * self.cfg["batch_size"], 100
            ),
            synthetic_volume_size=max(32, self.cfg.get("crop_size", 96) + 8),
        )


class CheckpointCallback(Callback):
    """Save checkpoints compatible with the Lumivox format."""

    def __init__(self, save_dir: str, save_every_epochs: int = 50):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_epochs = save_every_epochs

    def _save(self, trainer, pl_module, path: Path):
        method = pl_module.method
        ckpt = {
            "model_type": method,
            "epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
            "config": pl_module.cfg,
        }

        if method == "simclr":
            ckpt["model_state_dict"] = pl_module.model.state_dict()
            ckpt["encoder_state_dict"] = pl_module.model.encoder.state_dict()
        else:
            ckpt["online_state_dict"] = pl_module.online.state_dict()
            ckpt["target_state_dict"] = pl_module.target.state_dict()
            ckpt["encoder_state_dict"] = pl_module.online.encoder.state_dict()

        if trainer.optimizers:
            ckpt["optimizer_state_dict"] = trainer.optimizers[0].state_dict()

        if method in ("simclr", "nnbyol3d"):
            crop_size = pl_module.cfg.get("crop_size", 96)
            ckpt["adaptation_plan"] = {
                "architecture_plans": {"arch_class_name": "ResEncL"},
                "pretrain_num_input_channels": 1,
                "recommended_downstream_patchsize": (crop_size, crop_size, crop_size),
                "key_to_encoder": "encoder.stages",
                "key_to_stem": "encoder.stem",
                "keys_to_in_proj": [
                    "encoder.stem.convs.0.conv",
                    "encoder.stem.convs.0.all_modules.0",
                ],
            }

        torch.save(ckpt, path)
        if trainer.is_global_zero:
            print(f"  Saved checkpoint: {path}")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.save_every_epochs == 0 or epoch == trainer.max_epochs:
            path = self.save_dir / f"pretrain_epoch{epoch:04d}.pt"
            self._save(trainer, pl_module, path)

    def on_train_end(self, trainer, pl_module):
        path = self.save_dir / "pretrain_final.pt"
        self._save(trainer, pl_module, path)


def main():
    import warnings
    warnings.filterwarnings("ignore", message=".*AccumulateGrad.*stream.*")

    parser = argparse.ArgumentParser(description="Lumivox Lightning SSL Pretraining")
    parser.add_argument(
        "--method", type=str, default="simclr",
        choices=["simclr", "nnbyol3d", "byol3d-legacy"],
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", type=str, default="default",
        choices=["default", "legacy"],
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--base-ema", type=float, default=0.99)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--precision", type=str, default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Compute steps
    num_devices = args.devices or (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )

    # Estimate steps per epoch (will be recalculated if data is available)
    est_samples = 1000
    if args.data_dir:
        data_path = Path(args.data_dir)
        n = len(list(data_path.rglob("*.b2nd"))) + len(list(data_path.rglob("*.npy")))
        if n > 0:
            est_samples = n

    steps_per_epoch = max(est_samples // (args.batch_size * num_devices), 1)
    max_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_epochs * steps_per_epoch, max_steps // 10)

    cfg = dict(
        method=args.method,
        data_dir=args.data_dir,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_input_channels=1,
        proj_hidden_dim=4096 if args.method != "byol3d-legacy" else 2048,
        proj_output_dim=256,
        pred_hidden_dim=4096 if args.method != "byol3d-legacy" else 2048,
        base_channels=32,
        num_levels=5,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        base_ema=args.base_ema,
        num_workers=args.num_workers,
    )

    pl.seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    model = PretrainLightningModule(cfg)

    callbacks = [
        CheckpointCallback(save_dir=args.save_dir, save_every_epochs=args.save_every),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if num_devices > 1:
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=True)
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_devices,
        strategy=strategy,
        max_epochs=args.epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        enable_progress_bar=True,
        precision=args.precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
