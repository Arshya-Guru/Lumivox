"""Lightning-based fine-tuning for segmentation.

Provides multi-GPU DDP support for fine-tuning.

Usage:
    python -m lumivox.training.finetune_lightning \
        --checkpoint ./checkpoints/simclr_abeta_50k/pretrain_epoch0045.pt \
        --manifest manifests/abeta_ft_first_pass.json \
        --devices 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.strategies import DDPStrategy

from lumivox.data.dataset_finetune import build_finetune_dataloaders
from lumivox.training.finetune import build_segmentation_model, dice_ce_loss


class FinetuneLightningModule(pl.LightningModule):
    """Lightning module for segmentation fine-tuning."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = build_segmentation_model(
            checkpoint_path=cfg.get("checkpoint_path"),
            num_classes=cfg.get("num_classes", 1),
            deep_supervision=cfg.get("deep_supervision", True),
            dropout=cfg.get("dropout", 0.0),
        )

        self.freeze_encoder = cfg.get("freeze_encoder", False)
        if self.freeze_encoder and hasattr(self.model, "encoder"):
            for p in self.model.encoder.parameters():
                p.requires_grad_(False)
            self.model.encoder.eval()
            n_frozen = sum(p.numel() for p in self.model.encoder.parameters())
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Encoder frozen: {n_frozen/1e6:.2f}M params  "
                  f"(decoder-only trainable: {n_train/1e6:.2f}M)")

        self.ds_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]

    def on_train_epoch_start(self):
        # Keep the encoder in eval mode every epoch when frozen so any future
        # stateful layers (BN, dropout) behave consistently. Currently the encoder
        # uses InstanceNorm + LeakyReLU only, so this is a no-op for behavior but
        # protects the invariant if the architecture changes.
        if self.freeze_encoder and hasattr(self.model, "encoder"):
            self.model.encoder.eval()

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        outputs = self.model(images)

        if self.cfg.get("deep_supervision", True) and isinstance(outputs, (list, tuple)):
            loss = torch.tensor(0.0, device=self.device)
            for i, (out, w) in enumerate(zip(outputs, self.ds_weights)):
                target_ds = F.interpolate(masks, size=out.shape[2:], mode="nearest")
                loss = loss + w * dice_ce_loss(out, target_ds)
        else:
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            loss = dice_ce_loss(outputs, masks)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"]

        outputs = self.model(images)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        loss = dice_ce_loss(outputs, masks)

        # Compute Dice score
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        intersection = (preds * masks).sum()
        union = preds.sum() + masks.sum()
        dice_score = (2 * intersection + 1) / (union + 1)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/dice", dice_score, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.cfg.get("lr", 1e-3)
        encoder_lr_factor = self.cfg.get("encoder_lr_factor", 0.1)
        weight_decay = self.cfg.get("weight_decay", 1e-2)

        if hasattr(self.model, "encoder"):
            encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            encoder_ids = {id(p) for p in self.model.encoder.parameters()}
            other_params = [
                p for p in self.model.parameters()
                if id(p) not in encoder_ids and p.requires_grad
            ]
            param_groups = [{"params": other_params, "lr": lr}]
            if encoder_params:
                param_groups.insert(
                    0, {"params": encoder_params, "lr": lr * encoder_lr_factor}
                )
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr, weight_decay=weight_decay,
            )

        max_steps = self.cfg.get("max_steps", 10000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0, (1 - step / max(max_steps, 1)) ** 0.9)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def main():
    parser = argparse.ArgumentParser(description="Lumivox Lightning Fine-tuning")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained Lumivox .pt checkpoint (encoder will be loaded)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="FT manifest JSON (e.g., manifests/abeta_ft_first_pass.json)")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout3d on decoder skip connections")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze pretrained encoder; only decoder + seg heads train")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of subjects held out for validation (ignored if --val-subjects)")
    parser.add_argument("--val-subjects", type=str, nargs="*", default=None,
                        help="Explicit list of subject_ids to use for validation")
    parser.add_argument("--train-repeats", type=int, default=8,
                        help="Repeat the train set N times per epoch (small N patches => longer epochs)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable training-time augmentation")
    parser.add_argument(
        "--precision", type=str, default="bf16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
    )
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/finetune")
    parser.add_argument("--seed", type=int, default=42,
                        help="Model seed (init + augmentation + shuffle). Vary this "
                             "across the 5 ensemble members.")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for the subject-level train/val split. Keep FIXED "
                             "across ensemble members so they share one val set.")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="lumivox-fine-tuning")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Lightning fast_dev_run for smoke testing")
    args = parser.parse_args()

    num_devices = args.devices or (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )

    pl.seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, val_subjects = build_finetune_dataloaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        val_subjects=args.val_subjects,
        seed=args.split_seed,
        train_repeats=args.train_repeats,
        augment_train=not args.no_augment,
    )
    print(f"Loaded FT manifest: {args.manifest}")
    print(f"  train patches: {len(train_loader.dataset)} "
          f"(unique: {len(train_loader.dataset.entries)}, repeats: {args.train_repeats})")
    print(f"  val patches:   {len(val_loader.dataset)}")
    print(f"  val subjects:  {val_subjects}")

    steps_per_epoch = max(1, len(train_loader) // max(1, num_devices))
    max_steps = args.epochs * steps_per_epoch

    cfg = dict(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        deep_supervision=True,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        lr=args.lr,
        encoder_lr_factor=args.encoder_lr_factor,
        weight_decay=args.weight_decay,
        max_steps=max_steps,
    )
    model = FinetuneLightningModule(cfg)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    strategy = (
        DDPStrategy(find_unused_parameters=False)
        if num_devices > 1
        else "auto"
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.save_dir,
            # auto_insert_metric_name=False because the default templating treats
            # "val/dice" literally — the "/" becomes a directory separator and the
            # ckpts get nested into bogus subdirs (finetune-epoch=0009-val/dice=0.7081.ckpt).
            filename="finetune-epoch{epoch:04d}-dice{val/dice:.4f}",
            auto_insert_metric_name=False,
            monitor="val/dice",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
    ]

    try:
        from lightning.pytorch.loggers import CSVLogger
    except ImportError:
        from pytorch_lightning.loggers import CSVLogger
    loggers = [CSVLogger(save_dir=args.save_dir, name="lightning_logs")]

    if args.wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError:
            from pytorch_lightning.loggers import WandbLogger
        run_name = args.wandb_run_name or Path(args.save_dir).name
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            save_dir=args.save_dir,
            config={**cfg, "manifest": args.manifest, "seed": args.seed,
                    "split_seed": args.split_seed, "freeze_encoder": args.freeze_encoder,
                    "dropout": args.dropout, "encoder_lr_factor": args.encoder_lr_factor,
                    "val_subjects": val_subjects},
        )
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_devices,
        strategy=strategy,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        precision=args.precision,
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
    )

    # Auto-resume: if a previous (interrupted) run left a last.ckpt in save-dir,
    # resume from it so re-launching a timed-out job continues instead of restarting.
    resume_ckpt = Path(args.save_dir) / "last.ckpt"
    ckpt_path = str(resume_ckpt) if resume_ckpt.exists() and not args.fast_dev_run else None
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
