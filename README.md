# Lumivox

Self-supervised pretraining and segmentation fine-tuning for 3D light-sheet
fluorescence microscopy (LSFM) — built to map Aβ plaques (and other markers)
across whole mouse brains.

Lumivox is also a **fair comparison framework** for SSL methods: three model
configurations share maximum infrastructure so the only differences are the ones
under study.

| Model | Encoder | Loss | EMA target | Predictor | Augmentation |
|---|---|---|---|---|---|
| **SimCLR** | ResEncL | NT-Xent | No | No | Symmetric |
| **nnBYOL3D** | ResEncL | Regression | Yes | Yes | Asymmetric blur |
| **byol3d-legacy** | UNetEncoder3D | Regression | Yes | Yes | Legacy |

SimCLR and nnBYOL3D share the encoder, projection head, crop logic, optimizer, LR
schedule, augmentation base, and fine-tuning pipeline. The *only* differences are
the loss, EMA target, predictor MLP, and asymmetric blur.

---

## The pipeline

Three stages take you from raw OME-Zarr volumes to a trained plaque segmenter:

```
  1. PRETRAIN              2. BUILD GROUND TRUTH          3. FINE-TUNE
  SSL on unlabeled    ->   curated labeled patches   ->   segmentation model
  OME-Zarr patches        (Otsu/atlas + human review)     (encoder + decoder)
  (SimCLR / nnBYOL3D)      manifests/abeta_ft_v4_A.json    deep-ensemble sweep
```

1. **Pretrain** an encoder on unlabeled patches sampled from SPIMquant OME-Zarr
   datasets. → [Pretraining](#1-pretraining-ome-zarr)
2. **Build ground truth**: generate candidate masks, then human-review them into
   an accepted patch set. → [Ground truth](#2-ground-truth)
3. **Fine-tune** the pretrained encoder into a segmentation model.
   → **[docs/finetuning.md](docs/finetuning.md)**

### Quick start

```bash
# Smoke tests (no GPU)
pixi run smoke-simclr && pixi run smoke-nnbyol3d && pixi run test

# 1. Pretrain from an OME-Zarr patch manifest
pixi run manifest -- --dataset-roots /path/to/spimquant_dataset \
    --stain Abeta --n-patches 10000 --output manifests/abeta.json
python -m lumivox.training.pretrain --method simclr --manifest manifests/abeta.json --devices 4

# 3. Fine-tune (see docs/finetuning.md for the full sweep)
pixi run python scripts/build_v4_ft_manifest.py          # build the accepted-patch manifest
scripts/launch_v4_sweep.sh                                # launch the ensemble sweep (SLURM)
```

---

## 1. Pretraining (OME-Zarr)

Lumivox pretrains directly on OME-Zarr volumes with region-masked patch sampling,
using [zarrnii](https://pypi.org/project/zarrnii/) for lazy loading. Two steps.

### Step 1 — build a patch manifest

```bash
pixi run manifest -- \
    --dataset-roots /nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch3 \
    --stain Abeta \
    --regions L_Isocortex R_Isocortex "L_Hippocampal formation" "R_Hippocampal formation" \
    --n-patches 10000 --patch-size 256 --crop-size 96 --seed 42 \
    --output manifests/abeta_cortex_hipp.json
```

The builder samples N patch centers from the named atlas regions across one or
more dataset roots, using the deformed atlas segmentation (`roi22` / `roi82` /
`roi198`) warped to each subject. No image data is read at this stage — only the
low-res segmentation and zarr metadata. Each entry stores a pre-computed
`center_vox` so training reads patches by direct array slicing.

> **Coordinate mapping:** atlas centers (physical mm) → full-res voxels (via the
> full-res zarr affine) → resampled-4µm voxels (via the `*.ome.zarr.json` sidecar
> scale factors). The full-res zarr metadata is authoritative; resampled zarrs
> have incomplete metadata. Stain channel index is resolved from the full-res
> zarr's omero labels (ordering varies between datasets).

### Step 2 — pretrain

```bash
python -m lumivox.training.pretrain --method simclr   --manifest manifests/abeta_cortex_hipp.json --batch-size 28 --epochs 300
python -m lumivox.training.pretrain --method nnbyol3d --manifest manifests/abeta_cortex_hipp.json --batch-size 24 --epochs 300
```

Each `__getitem__` reads a 256-voxel region from the resampled zarr, extracts an
overlapping 96-voxel crop pair, z-score normalizes, and applies SSL
augmentations. Zarr chunks are fetched on demand; volume handles are cached per
worker.

**GPU reference (L40S 48 GB, bf16-mixed, `--num-workers 16`):** SimCLR fits batch
28 (~44 GB, OOM at 32); nnBYOL3D fits batch 24 (~40 GB; it carries online+target
networks). On A100 80 GB roughly double. Raise `--num-workers` if GPU util drops
below 80%.

### Expected data layout (SPIMquant convention)

```
dataset_root/
├── bids/sub-*/micr/*_SPIM.ome.zarr                       # full-res (has omero metadata)
│   └── derivatives/resampled/sub-*/micr/
│       ├── *_res-4um_SPIM.ome.zarr                        # 4µm isotropic (preferred)
│       └── *_res-4um_SPIM.ome.zarr.json                   # resampling sidecar
└── derivatives/spimquant*/sub-*/micr/
    ├── *_seg-roi22_from-ABAv3_*_dseg.nii.gz               # deformed atlas labels
    └── *_seg-roi22_from-ABAv3_*_dseg.tsv                  # label table
```

---

## 2. Ground truth

Fine-tuning needs labeled patches. We have no hand-annotated plaques, so Lumivox
builds an "almost gold" ground truth automatically and then has a human review
it. The current accepted set lives in `manifests/abeta_ft_v4_A.json`; how it's
consumed is documented in **[docs/finetuning.md](docs/finetuning.md)**.

### How candidate masks are generated

Two approaches have been used:

- **Atlas / SPIMquant masks (v4, current):** atlas-sample patches across datasets
  and derive each mask from the SPIMquant level-0 Aβ `.ozx`, downsampled to 128³
  with block-max pooling (preserves sparse plaques). Built by
  `scripts/build_abeta_gt_v4.py`.
- **WT-normalized Otsu (earlier):** normalize each disease patch against a
  wild-type reference brain from the same batch (N4 bias correction → pooled WT
  μ/σ → z-score), then blur → multi-Otsu → connected components → size filter.
  Built by `scripts/build_finetune_normalized.py`.

WT reference subjects per batch are listed in the `WT_SUBJECTS` map in
`scripts/build_finetune_normalized.py`. Note: WT / `sub-C57BL6` animals have no
real Aβ, so their SPIMquant masks are noise and are excluded from the GT.

### Human review

Candidate masks are reviewed and corrected in 3D Slicer / nnInteractive (drop
edge artifacts and false positives, add missed plaques), then marked
**accept / reject / maybe / empty** per patch. The review markers and how the
accepted set becomes the training manifest are documented in
[docs/finetuning.md § 1](docs/finetuning.md#1-ground-truth--v4-accepted-patches-only).

Helper scripts (for the earlier WT-normalized pipeline): per-patch napari
launchers (`scripts/generate_view_scripts.py`), atlas-context QC figures
(`scripts/generate_roi_qc.py`), a read-only over-segmentation audit
(`scripts/audit_gold_patches.py`), and the init/review/manifest workflow
(`scripts/finalize_finetune_gold.py`).

---

## 3. Fine-tuning

A pretrained encoder + a freshly-initialised decoder, trained on the accepted
patch set into an Aβ-plaque segmenter. We train **deep ensembles** (5 seeds per
config) and regularise the decoder with **true `Dropout3d` on the decoder feature
maps** (the encoder is left untouched), then average ensemble members at inference
into a probability mask.

**See [docs/finetuning.md](docs/finetuning.md)** for the full approach: ground
truth, model, the 40-member dropout × frozen/unfrozen × encoder sweep, the SLURM
launcher (`scripts/launch_v4_sweep.sh`), W&B logging, and planned ensemble
inference.

Regularisers available in `finetune_lightning.py`:

| Mechanism | Flag | Default |
|---|---|---|
| Decoder dropout (`Dropout3d` on decoder stages) | `--dropout` | 0.0 |
| Differential LR (encoder vs decoder) | `--encoder-lr-factor` | 0.1 |
| Weight decay | `--weight-decay` | 0.01 |
| Deep supervision | built-in | enabled |

---

## Repository structure

```
lumivox/
├── encoders/      # ResEncL (shared) + UNetEncoder3D (legacy)
├── heads/         # ProjectionMLP, PredictionMLP
├── models/        # SimCLR, nnBYOL3D, BYOL3D legacy
├── losses/        # NT-Xent, regression
├── augmentations/ # Unified pipeline + legacy
├── data/          # OME-Zarr dataset, fine-tune dataset, manifest builder
├── training/      # pretrain_lightning, finetune_lightning, finetune
├── adaptation/    # Checkpoint export (encoder extraction)
└── utils/         # EMA helpers
scripts/
├── build_*_manifest*.py / build_abeta_gt_v4.py  # patch manifests + v4 ground truth
├── build_v4_ft_manifest.py                       # accepted-patch -> fine-tune manifest
├── build_finetune_normalized.py                  # WT-normalized GT pipeline
├── finalize_finetune_gold.py / audit_gold_patches.py  # review + audit (WT pipeline)
├── generate_view_scripts.py / generate_roi_qc.py # napari launchers + QC figures
├── launch_v4_sweep.sh / finetune_v4_job.sbatch   # fine-tune ensemble sweep (SLURM)
└── *.sh / *.sbatch                               # pretraining + inference launchers
docs/finetuning.md                                # fine-tuning approach (read this)
manifests/                                        # patch + fine-tune manifests
tests/                                            # unit tests
```

Ground-truth patches are also mirrored at `$GT/Abeta/...` (typically
`/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized`) so collaborators
can reach them outside the worktree.
