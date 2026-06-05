"""Topological fine-fine-tuning.

A second, short fine-tuning phase that starts from a *converged* segmentation
checkpoint (produced by lumivox.training.finetune_lightning — e.g. a member of
the v4 sweep under checkpoints/ft_v4_sweep/) and continues training with a
topological loss (``topo_ce``: Dice+CE + persistent-homology term, with the GT
mask's topology as the per-sample constraint).

The pattern follows SynthTopo: train to convergence on Dice+CE, *then* run a
short topo phase (~20% of the epochs the base phase took to converge) to clean
up topology without wrecking the Dice the base phase earned.

Usage:
    python -m lumivox.training.finefinetune_lightning \
        --ft-checkpoint checkpoints/ft_v4_sweep/ftv4_simclr_d10_frozen_s0/finetune-epoch0060-dice0.6550.ckpt \
        --manifest manifests/abeta_ft_v4_A.json \
        --epochs 30 --lr 1e-4 --topo-weight 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

from lumivox.data.dataset_finetune import build_finetune_dataloaders
from lumivox.training.finetune import build_segmentation_model
from lumivox.losses.topo import TopoCELoss


def _load_ft_model(ft_checkpoint: str):
    """Rebuild the segmentation model from a finetune .ckpt and load its weights.

    Uses the saved hyper-parameters (num_classes / dropout / deep_supervision) so
    the architecture matches exactly, then loads the trained encoder+decoder
    weights. The pretrained encoder is NOT reloaded — we want the FT-trained
    weights, which are already in the checkpoint.
    """
    ckpt = torch.load(ft_checkpoint, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {}) or {}
    model = build_segmentation_model(
        checkpoint_path=None,
        num_classes=hp.get("num_classes", 1),
        deep_supervision=hp.get("deep_supervision", True),
        dropout=hp.get("dropout", 0.0),
    )
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded FT weights from {ft_checkpoint}")
    print(f"  hparams: num_classes={hp.get('num_classes', 1)} "
          f"dropout={hp.get('dropout', 0.0)} deep_supervision={hp.get('deep_supervision', True)}")
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return model, hp


class FineFinetuneLightningModule(pl.LightningModule):
    """Topological fine-fine-tuning module."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model, self.ft_hparams = _load_ft_model(cfg["ft_checkpoint"])

        self.freeze_encoder = cfg.get("freeze_encoder", False)
        if self.freeze_encoder and hasattr(self.model, "encoder"):
            for p in self.model.encoder.parameters():
                p.requires_grad_(False)
            self.model.encoder.eval()
            print("Encoder frozen for topo phase")

        # Topo loss uses the PRIMARY (full-res) output only — no deep supervision
        # (the topological term manages its own Dice+CE base internally).
        self.criterion = TopoCELoss(
            base_weight=cfg.get("base_weight", 1.0),
            topo_weight=cfg.get("topo_weight", 1e-4),
            construction=cfg.get("construction", "0"),
            thresh=cfg.get("thresh", None),
        )

    def on_train_epoch_start(self):
        if self.freeze_encoder and hasattr(self.model, "encoder"):
            self.model.encoder.eval()

    def _primary(self, images):
        outputs = self.model(images)
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs

    def training_step(self, batch, batch_idx):
        logits = self._primary(batch["image"])
        loss = self.criterion(logits, batch["mask"])
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/base", self.criterion.last_base, sync_dist=True)
        self.log("train/topo", self.criterion.last_topo, sync_dist=True)
        self.log("train/betti_err", self.criterion.last_betti_err, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self._primary(batch["image"])
        masks = batch["mask"]
        loss = self.criterion(logits, masks)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        inter = (preds * masks).sum()
        union = preds.sum() + masks.sum()
        dice = (2 * inter + 1) / (union + 1)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/dice", dice, prog_bar=True, sync_dist=True)
        self.log("val/topo", self.criterion.last_topo, prog_bar=True, sync_dist=True)
        self.log("val/betti_err", self.criterion.last_betti_err, sync_dist=True)
        return loss

    def configure_optimizers(self):
        lr = self.cfg.get("lr", 1e-4)
        encoder_lr_factor = self.cfg.get("encoder_lr_factor", 0.1)
        weight_decay = self.cfg.get("weight_decay", 1e-2)

        if hasattr(self.model, "encoder"):
            enc = [p for p in self.model.encoder.parameters() if p.requires_grad]
            enc_ids = {id(p) for p in self.model.encoder.parameters()}
            other = [p for p in self.model.parameters()
                     if id(p) not in enc_ids and p.requires_grad]
            groups = [{"params": other, "lr": lr}]
            if enc:
                groups.insert(0, {"params": enc, "lr": lr * encoder_lr_factor})
            optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr, weight_decay=weight_decay,
            )

        max_steps = self.cfg.get("max_steps", 1000)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0, (1 - step / max(max_steps, 1)) ** 0.9)
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


def main():
    p = argparse.ArgumentParser(description="Lumivox topological fine-fine-tuning")
    p.add_argument("--ft-checkpoint", type=str, required=True,
                   help="Converged finetune .ckpt to start from (e.g. a v4 sweep member)")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30,
                   help="Short schedule (~20%% of the epochs the base FT took to converge)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Low LR for the topo phase (the base phase already converged)")
    p.add_argument("--encoder-lr-factor", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--topo-weight", type=float, default=1e-4)
    p.add_argument("--base-weight", type=float, default=1.0)
    p.add_argument("--construction", type=str, default="0", choices=["0", "N"],
                   help="'0'=6-conn (cripser), 'N'=26-conn (tcripser)")
    p.add_argument("--thresh", type=float, default=None,
                   help="Restrict PH to the foreground bbox (prob>=thresh) to bound cost")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-subjects", type=str, nargs="*", default=None)
    p.add_argument("--train-repeats", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--precision", type=str, default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--save-dir", type=str, default="./checkpoints/finefinetune")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42,
                   help="Keep equal to the base FT split-seed so the val set matches")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="lumivox-fine-fine-tuning")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--fast-dev-run", action="store_true")
    args = p.parse_args()

    num_devices = args.devices or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
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
    steps_per_epoch = max(1, len(train_loader) // max(1, num_devices))
    max_steps = args.epochs * steps_per_epoch
    print(f"Topo fine-fine-tuning: {args.epochs} epochs, {max_steps} steps, "
          f"topo_weight={args.topo_weight}")
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  val_subjects={val_subjects}")

    cfg = dict(
        ft_checkpoint=args.ft_checkpoint,
        lr=args.lr,
        encoder_lr_factor=args.encoder_lr_factor,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder,
        topo_weight=args.topo_weight,
        base_weight=args.base_weight,
        construction=args.construction,
        thresh=args.thresh,
        max_steps=max_steps,
    )
    model = FineFinetuneLightningModule(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="finefinetune-epoch{epoch:04d}-dice{val/dice:.4f}",
            auto_insert_metric_name=False,
            monitor="val/dice",
            mode="max",
            save_top_k=2,
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
        loggers.append(WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name or Path(args.save_dir).name,
            save_dir=args.save_dir,
            config={**cfg, "manifest": args.manifest, "epochs": args.epochs,
                    "ft_checkpoint": args.ft_checkpoint},
        ))

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_devices,
        strategy="auto",
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        precision=args.precision,
        gradient_clip_val=1.0,
        fast_dev_run=args.fast_dev_run,
    )

    resume = Path(args.save_dir) / "last.ckpt"
    ckpt_path = str(resume) if resume.exists() and not args.fast_dev_run else None
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
