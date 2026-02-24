# Lumivox

Unified SSL Pretraining for 3D Lightsheet Fluorescence Microscopy.

A fair comparison framework for self-supervised learning methods on 3D LSFM data. Contains three model configurations sharing maximum infrastructure:

| Model | Encoder | Loss | EMA Target | Predictor | Augmentation |
|---|---|---|---|---|---|
| **SimCLR** | ResEncL | NT-Xent | No | No | Symmetric |
| **nnBYOL3D** | ResEncL | Regression | Yes | Yes | Asymmetric blur |
| **byol3d-legacy** | UNetEncoder3D | Regression | Yes | Yes | Legacy |

## Fairness Guarantee

SimCLR and nnBYOL3D share: encoder, projection head, crop logic, optimizer, LR schedule, augmentation base, and fine-tuning pipeline. The ONLY differences are the loss function, EMA target, predictor MLP, and asymmetric blur.

## Quick Start

```bash
# Smoke tests (no GPU required)
pixi run smoke-simclr
pixi run smoke-nnbyol3d
pixi run smoke-legacy

# Run tests
pixi run test

# Full pretraining (Lightning, multi-GPU)
pixi run pretrain -- --method simclr --data-dir /path/to/data --devices 4
pixi run pretrain -- --method nnbyol3d --data-dir /path/to/data --devices 4

# Fine-tuning
pixi run finetune -- --checkpoint ./checkpoints/simclr/pretrain_final.pt
```

## Repository Structure

```
lumivox/
├── encoders/        # ResEncL (shared) + UNetEncoder3D (legacy)
├── heads/           # ProjectionMLP, PredictionMLP
├── models/          # SimCLR, nnBYOL3D, BYOL3D legacy
├── losses/          # NT-Xent, regression
├── augmentations/   # Unified pipeline + legacy
├── data/            # Blosc2/npy dataset + DataLoader
├── training/        # Pretrain + finetune (plain + Lightning)
├── adaptation/      # Checkpoint export for nnU-Net
└── utils/           # EMA helpers
configs/             # Per-method configuration
tests/               # Unit tests
```
