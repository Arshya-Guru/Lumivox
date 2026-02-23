"""Lightning-based fine-tuning for segmentation.

Provides multi-GPU DDP support for fine-tuning.

Usage:
    python -m lumivox.training.finetune_lightning \
        --checkpoint ./checkpoints/simclr/pretrain_final.pt \
        --data-dir /path/to/labeled_data --devices 4
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
        )

        self.ds_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]

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
            encoder_params = list(self.model.encoder.parameters())
            encoder_ids = {id(p) for p in encoder_params}
            other_params = [p for p in self.model.parameters() if id(p) not in encoder_ids]

            optimizer = torch.optim.AdamW(
                [
                    {"params": encoder_params, "lr": lr * encoder_lr_factor},
                    {"params": other_params, "lr": lr},
                ],
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
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
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="./checkpoints/finetune")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_devices = args.devices or (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )

    cfg = dict(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        deep_supervision=True,
        lr=args.lr,
        encoder_lr_factor=args.encoder_lr_factor,
        weight_decay=args.weight_decay,
        max_steps=args.epochs * 50,
    )

    pl.seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

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
            filename="finetune-{epoch:04d}-{val/dice:.4f}",
            monitor="val/dice",
            mode="max",
            save_top_k=3,
        ),
    ]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_devices,
        strategy=strategy,
        max_epochs=args.epochs,
        callbacks=callbacks,
        precision=args.precision,
        gradient_clip_val=1.0,
    )

    # NOTE: User needs to provide train/val dataloaders via their data pipeline
    print("NOTE: Provide train/val DataLoaders via model.train_dataloader() / val_dataloader()")
    print("      or call trainer.fit(model, train_dataloaders=..., val_dataloaders=...)")


if __name__ == "__main__":
    main()
