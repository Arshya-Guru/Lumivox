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

# Full pretraining from blosc2/npy patches (Lightning, multi-GPU)
pixi run pretrain -- --method simclr --data-dir /path/to/data --devices 4
pixi run pretrain -- --method nnbyol3d --data-dir /path/to/data --devices 4

# Pretraining from OME-Zarr (SPIMquant datasets) -- see below
pixi run manifest -- \
    --dataset-roots /nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3 \
    --stain Abeta --n-patches 10000 --output manifests/abeta_cortex_hipp.json
python -m lumivox.training.pretrain --method simclr --manifest manifests/abeta_cortex_hipp.json

# Fine-tuning
pixi run finetune -- --checkpoint ./checkpoints/simclr/pretrain_final.pt

# Fine-tuning with decoder dropout (regularization against overfitting)
pixi run finetune -- --checkpoint ./checkpoints/simclr/pretrain_final.pt --dropout 0.2
```

## OME-Zarr Pretraining (SPIMquant Datasets)

Lumivox can pretrain directly on OME-Zarr lightsheet volumes with region-masked patch sampling, using [zarrnii](https://pypi.org/project/zarrnii/) for lazy loading. This is a two-step workflow.

### Step 1: Build a patch manifest

The manifest samples N patch centers from specified brain regions across one or more SPIMquant dataset roots. It uses the deformed atlas segmentation (e.g., roi22) warped to each subject's native space via zarrnii's `ZarrNiiAtlas.sample_region_patches()` to restrict sampling to anatomical regions of interest.

```bash
pixi run manifest -- \
    --dataset-roots /nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3 \
                    /nfs/trident3/lightsheet/prado/mouse_app_lecanemab_ki3_batch2 \
    --stain Abeta \
    --regions L_Isocortex R_Isocortex "L_Hippocampal formation" "R_Hippocampal formation" \
    --n-patches 10000 \
    --patch-size 256 \
    --crop-size 96 \
    --seed 42 \
    --output manifests/abeta_cortex_hipp.json
```

This produces a JSON manifest where each entry records a subject, the path to their OME-Zarr, and a pre-computed voxel coordinate for the resampled zarr. No image data is read at this stage -- only the low-resolution segmentation masks and zarr metadata.

Available stains: stain names are resolved from each zarr's omero metadata (channel ordering varies between datasets). Available segmentation levels: `roi22` (22 regions), `roi82`, `roi198`.

### How the manifest coordinate mapping works

The atlas segmentation and the OME-Zarr volumes live in different coordinate systems. The manifest builder handles this by chaining two transformations at build time:

1. **Physical centers from the atlas**: `ZarrNiiAtlas.sample_region_patches()` returns centers in the dseg NIfTI's physical space (mm), which is shared with the full-resolution OME-Zarr.

2. **Physical to full-res voxel**: The full-res zarr's affine (from zarrnii) correctly maps physical coordinates to voxel indices. This is used via `fullres_affine.invert() @ center`.

3. **Full-res voxel to resampled voxel**: The resampled 4 um zarr ships with a JSON sidecar (`*.ome.zarr.json`) containing the scale factors from full-res to resampled: `voxel_resampled = voxel_fullres * scale`.

The resulting voxel coordinates are stored as `center_vox` in the manifest. At training time, the dataset reads patches by direct array slicing -- no `crop_centered`, no affine interpretation at load time.

This two-step approach was necessary because the resampled zarr files have incomplete OME-Zarr metadata (no omero channel labels, zero-origin affine with orientation flags that don't match the actual voxel layout). The full-res zarr metadata is authoritative and the JSON sidecar provides the exact resampling relationship.

### Step 2: Pretrain with the manifest

```bash
# SimCLR
python -m lumivox.training.pretrain \
    --method simclr \
    --manifest manifests/abeta_cortex_hipp.json \
    --batch-size 28 \
    --num-workers 16 \
    --epochs 300

# nnBYOL3D
python -m lumivox.training.pretrain \
    --method nnbyol3d \
    --manifest manifests/abeta_cortex_hipp.json \
    --batch-size 24 \
    --num-workers 16 \
    --epochs 300
```

During training, each `__getitem__` reads a 256-voxel region from the resampled OME-Zarr at the manifest's pre-computed voxel coordinate, extracts an overlapping 96-voxel crop pair, z-score normalizes, and applies SSL augmentations. Data is never fully loaded into memory -- zarr chunks are fetched on demand and volume handles are cached per worker.

### GPU scaling reference (L40S 48GB, bf16-mixed)

Tested on NVIDIA L40S (48GB) with `--num-workers 16` and `--precision bf16-mixed`:

| Model | Batch size | VRAM usage | Notes |
|---|---|---|---|
| SimCLR | 28 | ~44 GB | OOM at 32 |
| nnBYOL3D | 24 | ~40 GB | Has online + target networks (2x encoder) |

For A100 80GB, batch sizes can roughly double. If GPU utilization drops below 80%, increase `--num-workers`. If IO-bound (many subjects across NFS), `--num-workers 24` helps keep the GPU fed during zarr chunk fetches.

### Stain channel resolution

Channel ordering varies between datasets (e.g., batch3 uses Abeta/CD31/YoPro while ki3 uses IBA1/Abeta/CD31). The manifest builder reads channel labels from the full-res zarr's omero metadata to resolve the correct channel index per subject. Resampled zarr files typically lack omero metadata, so the full-res zarr is always checked as a fallback.

### Expected data layout

Each dataset root should follow SPIMquant conventions:

```
dataset_root/
├── bids/
│   ├── sub-*/micr/*_SPIM.ome.zarr              # Full-resolution OME-Zarr (has omero metadata)
│   └── derivatives/resampled/
│       ├── sub-*/micr/*_res-4um_SPIM.ome.zarr   # 4 um isotropic (preferred for training)
│       └── sub-*/micr/*_res-4um_SPIM.ome.zarr.json  # Resampling sidecar (scale factors)
└── derivatives/spimquant*/                      # May have version suffix (e.g., spimquant_c270a40_atropos)
    └── sub-*/micr/
        ├── *_seg-roi22_from-ABAv3_*_dseg.nii.gz  # Deformed atlas segmentation
        └── *_seg-roi22_from-ABAv3_*_dseg.tsv      # Region label table
```

If the resampled 4 um zarr is not available for a subject, the manifest builder falls back to the full-res zarr with `downsample_near_isotropic=True`. However, the direct voxel slicing path (with pre-computed `center_vox`) is only used when both the resampled zarr and its JSON sidecar are present.

## Fine-tuning Regularization

The decoder (~17.5M params) is randomly initialized during fine-tuning while the encoder (~76.5M params) is pretrained. To prevent the decoder from overfitting on small labeled datasets, several regularization mechanisms are available:

| Mechanism | Flag | Default | Description |
|---|---|---|---|
| Differential LR | `--encoder-lr-factor` | 0.1 | Encoder trains at 10x lower LR than decoder |
| Weight decay | `--weight-decay` | 0.01 | AdamW L2 regularization |
| Deep supervision | built-in | enabled | Multi-scale loss from intermediate decoder stages |
| **Decoder dropout** | `--dropout` | 0.0 | `Dropout3d` on skip connections between encoder and decoder |

Decoder dropout (`--dropout`) applies `nn.Dropout3d` to encoder skip connection outputs before they enter the decoder, dropping entire feature channels. This forces the decoder to not rely on any single encoder feature and is automatically disabled during evaluation. Recommended values: 0.1--0.3.

## Repository Structure

```
lumivox/
├── encoders/        # ResEncL (shared) + UNetEncoder3D (legacy)
├── heads/           # ProjectionMLP, PredictionMLP
├── models/          # SimCLR, nnBYOL3D, BYOL3D legacy
├── losses/          # NT-Xent, regression
├── augmentations/   # Unified pipeline + legacy
├── data/            # Blosc2/npy dataset, OME-Zarr dataset, manifest builder
├── training/        # Pretrain + finetune (plain + Lightning)
├── adaptation/      # Checkpoint export for nnU-Net
└── utils/           # EMA helpers
configs/             # Per-method configuration
scripts/             # Pretraining launch scripts
tests/               # Unit tests
```
