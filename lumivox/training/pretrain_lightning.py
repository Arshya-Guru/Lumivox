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
from lumivox.data.dataset_omezarr import OMEZarrPatchDataset, _omezarr_worker_init_fn
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

        # Step offset for resume — shifts LR and EMA schedules
        self.step_offset = cfg.get("step_offset", 0)

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

            tau = cosine_ema_schedule(self.global_step + self.step_offset, self.base_ema, self.max_steps)
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)
            self.log("train/tau", tau, prog_bar=True)

        self.log("train/lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update for BYOL models after each step."""
        if self.method in ("nnbyol3d", "byol3d-legacy"):
            tau = cosine_ema_schedule(self.global_step + self.step_offset, self.base_ema, self.max_steps)
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

        # Fast-forward scheduler if resuming from weights
        if self.step_offset > 0:
            for _ in range(self.step_offset):
                scheduler.step()
            print(f"  LR scheduler fast-forwarded by {self.step_offset} steps")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def train_dataloader(self):
        manifest = self.cfg.get("manifest")
        if manifest is not None:
            from torch.utils.data import DataLoader
            dataset = OMEZarrPatchDataset(
                manifest_path=manifest,
                method=self.method,
                crop_size=self.cfg.get("crop_size", 96),
                localscratch_dir=self.cfg.get("localscratch_dir", "/localscratch/lumivox_patches"),
            )
            return DataLoader(
                dataset,
                batch_size=self.cfg["batch_size"],
                shuffle=True,
                num_workers=self.cfg.get("num_workers", 4),
                pin_memory=True,
                drop_last=True,
                persistent_workers=self.cfg.get("num_workers", 4) > 0,
                prefetch_factor=4 if self.cfg.get("num_workers", 4) > 0 else None,
                worker_init_fn=_omezarr_worker_init_fn,
            )

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

    def __init__(self, save_dir: str, save_every_epochs: int = 50, epoch_offset: int = 0):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_epochs = save_every_epochs
        self.epoch_offset = epoch_offset

    def _save(self, trainer, pl_module, path: Path):
        method = pl_module.method
        ckpt = {
            "model_type": method,
            "epoch": trainer.current_epoch + 1 + self.epoch_offset,
            "global_step": trainer.global_step + pl_module.step_offset,
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
        real_epoch = epoch + self.epoch_offset
        if epoch % self.save_every_epochs == 0 or epoch == trainer.max_epochs:
            path = self.save_dir / f"pretrain_epoch{real_epoch:04d}.pt"
            self._save(trainer, pl_module, path)

    def on_train_end(self, trainer, pl_module):
        path = self.save_dir / "pretrain_final.pt"
        self._save(trainer, pl_module, path)


class TrainingPlotCallback(Callback):
    """Generate training progress figures from logged metrics."""

    def __init__(self, save_dir: str, plot_every_epochs: int = 5, epoch_offset: int = 0):
        self.save_dir = Path(save_dir)
        self.plot_every = plot_every_epochs
        self.epoch_offset = epoch_offset
        self.epoch_losses = []
        self.epoch_lrs = []
        self.epoch_accs = []  # SimCLR only
        self.epoch_taus = []  # BYOL only
        self.step_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = trainer.callback_metrics.get("train/loss")
        if loss is not None:
            self.step_losses.append(loss.item() if hasattr(loss, "item") else float(loss))

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect epoch-level metrics
        loss = trainer.callback_metrics.get("train/loss")
        lr = trainer.callback_metrics.get("train/lr")
        acc = trainer.callback_metrics.get("train/acc")
        tau = trainer.callback_metrics.get("train/tau")

        if loss is not None:
            self.epoch_losses.append(loss.item() if hasattr(loss, "item") else float(loss))
        if lr is not None:
            self.epoch_lrs.append(lr.item() if hasattr(lr, "item") else float(lr))
        if acc is not None:
            self.epoch_accs.append(acc.item() if hasattr(acc, "item") else float(acc))
        if tau is not None:
            self.epoch_taus.append(tau.item() if hasattr(tau, "item") else float(tau))

        epoch = trainer.current_epoch + 1
        real_epoch = epoch + self.epoch_offset
        if trainer.is_global_zero and (epoch % self.plot_every == 0 or epoch == 1):
            self._make_plots(pl_module.method, real_epoch)

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._make_plots(pl_module.method, trainer.current_epoch + 1 + self.epoch_offset, final=True)

    def _make_plots(self, method, epoch, final=False):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        is_byol = method in ("nnbyol3d", "byol3d-legacy")
        n_panels = 4 if is_byol else (4 if self.epoch_accs else 3)

        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

        # Panel 1: Epoch loss
        ax = axes[0]
        if self.epoch_losses:
            ax.plot(range(1, len(self.epoch_losses) + 1), self.epoch_losses, "b-", lw=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss per Epoch")
            ax.grid(True, alpha=0.3)

        # Panel 2: Step loss (smoothed)
        ax = axes[1]
        if self.step_losses:
            raw = self.step_losses
            # EMA smoothing
            alpha = 0.01
            smoothed = [raw[0]]
            for v in raw[1:]:
                smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
            ax.plot(smoothed, "b-", lw=0.8, alpha=0.8, label="smoothed")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("Loss per Step (EMA smoothed)")
            ax.grid(True, alpha=0.3)

        # Panel 3: Learning rate
        ax = axes[2]
        if self.epoch_lrs:
            ax.plot(range(1, len(self.epoch_lrs) + 1), self.epoch_lrs, "g-", lw=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("LR")
            ax.set_title("Learning Rate Schedule")
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # Panel 4: Method-specific
        if is_byol and len(axes) > 3:
            ax = axes[3]
            if self.epoch_taus:
                ax.plot(range(1, len(self.epoch_taus) + 1), self.epoch_taus, "r-", lw=1.5)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("EMA tau")
                ax.set_title("EMA Schedule")
                ax.grid(True, alpha=0.3)
        elif self.epoch_accs and len(axes) > 3:
            ax = axes[3]
            ax.plot(range(1, len(self.epoch_accs) + 1), self.epoch_accs, "r-", lw=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("NT-Xent Top-1 Accuracy")
            ax.grid(True, alpha=0.3)

        tag = "final" if final else f"epoch{epoch:04d}"
        fig.suptitle(f"{method} pretraining — {tag}", fontsize=14)
        plt.tight_layout()

        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"training_progress_{tag}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved training plot: {path}")


def main():
    import warnings
    warnings.filterwarnings("ignore", message=".*AccumulateGrad.*stream.*")

    parser = argparse.ArgumentParser(description="Lumivox Lightning SSL Pretraining")
    parser.add_argument(
        "--method", type=str, default="simclr",
        choices=["simclr", "nnbyol3d", "byol3d-legacy"],
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to OME-Zarr patch manifest JSON")
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
    parser.add_argument("--accumulate-grad-batches", type=int, default=1,
                        help="Accumulate gradients over N batches (effective batch = batch_size * N)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to Lightning checkpoint to resume from (last.ckpt)")
    parser.add_argument("--resume-weights", type=str, default=None,
                        help="Path to Lumivox .pt checkpoint to load weights from (2-GPU compatible)")
    parser.add_argument("--epoch-offset", type=int, default=0,
                        help="Offset for checkpoint/plot epoch numbering (set to resume epoch)")
    args = parser.parse_args()

    # Compute steps
    num_devices = args.devices or (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )

    # Estimate steps per epoch (will be recalculated if data is available)
    est_samples = 1000
    if args.manifest:
        import json
        with open(args.manifest) as f:
            est_samples = json.load(f)["config"]["n_patches"]
    elif args.data_dir:
        data_path = Path(args.data_dir)
        n = len(list(data_path.rglob("*.b2nd"))) + len(list(data_path.rglob("*.npy")))
        if n > 0:
            est_samples = n

    accum = args.accumulate_grad_batches
    steps_per_epoch = max(est_samples // (args.batch_size * num_devices * accum), 1)
    max_steps = args.epochs * steps_per_epoch
    warmup_steps = min(args.warmup_epochs * steps_per_epoch, max_steps // 10)

    cfg = dict(
        method=args.method,
        data_dir=args.data_dir,
        manifest=args.manifest,
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

    # Clear stale CSV logger to avoid header conflicts between methods.
    # Each method gets its own log dir so parallel runs don't conflict.
    import shutil
    stale_logs = Path(args.save_dir) / "lightning_logs"
    if stale_logs.exists() and args.resume is None and args.resume_weights is None:
        shutil.rmtree(stale_logs)

    model = PretrainLightningModule(cfg)

    # Epoch offset for checkpoint/plot numbering (auto-set from checkpoint if resuming)
    epoch_offset = args.epoch_offset

    # Load weights from Lumivox .pt checkpoint (not Lightning .ckpt)
    if args.resume_weights is not None:
        ckpt = torch.load(args.resume_weights, map_location="cpu", weights_only=False)
        method = args.method
        if method == "simclr":
            model.model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.online.load_state_dict(ckpt["online_state_dict"])
            model.target.load_state_dict(ckpt["target_state_dict"])
        resume_epoch = ckpt.get("epoch", 0)
        resume_step = ckpt.get("global_step", 0)

        # Recompute max_steps to cover the FULL schedule (original + remaining).
        # The schedule (LR cosine + EMA tau) needs to span the total intended
        # training, not just the resume portion. step_offset fast-forwards to
        # the right position within that total schedule.
        total_max_steps = resume_step + max_steps  # original steps + new epochs
        total_warmup = min(args.warmup_epochs * steps_per_epoch, total_max_steps // 10)

        model.step_offset = resume_step
        model.cfg["step_offset"] = resume_step
        model.cfg["max_steps"] = total_max_steps
        model.cfg["warmup_steps"] = total_warmup

        # Also update max_steps for the Trainer
        max_steps = args.epochs * steps_per_epoch  # Trainer only runs the new epochs

        # Auto-set epoch offset if not explicitly provided
        if args.epoch_offset == 0:
            epoch_offset = resume_epoch

        print(f"Loaded weights from {args.resume_weights} (epoch {resume_epoch}, step {resume_step})")
        print(f"  Schedule total: {total_max_steps} steps (offset={resume_step} + new={max_steps})")
        print(f"  LR and EMA schedules continue from step {resume_step}/{total_max_steps}")
        print(f"  Epoch numbering offset: {epoch_offset} (next save = epoch {epoch_offset + args.save_every})")

    try:
        from lightning.pytorch.callbacks import ModelCheckpoint
    except ImportError:
        from pytorch_lightning.callbacks import ModelCheckpoint

    callbacks = [
        CheckpointCallback(save_dir=args.save_dir, save_every_epochs=args.save_every, epoch_offset=epoch_offset),
        TrainingPlotCallback(save_dir=args.save_dir, plot_every_epochs=args.save_every, epoch_offset=epoch_offset),
        # Lightning-native checkpoint for resume (saves full training state)
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="last",
            save_last=True,
            every_n_epochs=1,
            save_top_k=1,
        ),
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if num_devices > 1:
        # BYOL target network has frozen params that DDP needs to handle
        find_unused = args.method in ("nnbyol3d", "byol3d-legacy")
        strategy = DDPStrategy(
            find_unused_parameters=find_unused,
            static_graph=(not find_unused),
        )
    else:
        strategy = "auto"

    try:
        from lightning.pytorch.loggers import CSVLogger
    except ImportError:
        from pytorch_lightning.loggers import CSVLogger

    logger = CSVLogger(save_dir=args.save_dir, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_devices,
        strategy=strategy,
        max_epochs=args.epochs,
        max_steps=max_steps,
        accumulate_grad_batches=accum,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        precision=args.precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
