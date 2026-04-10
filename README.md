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

# Build fine-tune ground truth (WT-normalized "almost gold" segmentation)
pixi run python scripts/build_finetune_normalized.py
pixi run python scripts/finalize_finetune_gold.py init --use otsu2
pixi run python scripts/generate_view_scripts.py
# ...napari pass to drop boundary artifacts (see Fine-tune Ground Truth section)
pixi run python scripts/finalize_finetune_gold.py manifest

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

## Fine-tune Ground Truth Generation

Fine-tuning a 3D segmentation decoder needs labeled patches. We don't have hand-annotated plaques, so Lumivox builds an "almost gold" ground truth by **normalizing each disease patch against a wild-type (WT) reference brain from the same batch** and then running a deterministic blur → multi-Otsu → connected components → size filter pipeline. The output is clean enough that a quick napari pass to drop a handful of boundary artifacts yields gold-quality masks.

There are two pipelines that produce labeled patches today: the simpler `build_finetune_data.py` which lifts SPIMquant fieldfrac masks directly, and the WT-normalized `build_finetune_normalized.py` which is the recommended path for plaque (Abeta) segmentation. The rest of this section documents the WT-normalized pipeline.

### What it does

For each disease patch in `ft/Abeta/manifest.json` (excluding any patches that came from WT animals — those are tracked and listed in the output):

1. **N4 bias-field correction** is computed once per WT animal at a downsampled resolution and cached as a NIfTI in `ft_normalized/Abeta/wt_references/`. Subsequent runs skip recomputation. SimpleITK is used directly (zarrnii's antspyx-based plugin is not in the env).
2. A **256³ window** is pulled from the disease zarr at the patch's atlas-sampled physical center, and matching 256³ windows are pulled from each in-batch WT brain (mapped through each WT's own affine chain).
3. **Pooled mu/sigma** are computed from the bias-corrected WT windows (within-tissue voxels only). When the disease patch's physical coordinate doesn't land inside a given WT brain — different physical-space conventions across animals can put the WT box "out from under" some patches — the pipeline transparently **falls back to whole-brain WT statistics** so no patch is dropped. Fallback usage is recorded in each patch's `meta.json` and visible in the `finalize status` table.
4. The disease window is **z-score normalized** with the pooled stats and the central 128³ is cropped from both the raw and normalized volumes.
5. The normalized 128³ is segmented with **two candidate Otsu thresholds (k=2 and k=3)** so you can compare them in the QC figure and pick the better one per patch.
6. **Outputs** for each patch land alongside the existing `ft/` tree at `ft_normalized/Abeta/{dataset}/{subject}/`:

   ```
   sub-XXX_patchNN_crop128.nii.gz          # raw 128³ disease intensity (training input)
   sub-XXX_patchNN_normalized128.nii.gz    # WT z-score normalized 128³ (for QC)
   sub-XXX_patchNN_seg_otsu2.nii.gz        # candidate mask, k=2
   sub-XXX_patchNN_seg_otsu3.nii.gz        # candidate mask, k=3
   sub-XXX_patchNN_qc.png                  # 5×3 QC figure
   sub-XXX_patchNN_meta.json               # mu_wt, sigma_wt, WT refs used, fallback flag
   ```

### Vaccine batch (no in-batch WT)

The vaccine cohort has no healthy controls. For its patches the pipeline automatically pools all WT animals from every other batch and averages μ/pools σ across them, so vaccine patches normalize against a synthetic-cohort baseline with no special configuration needed.

### WT subject discovery

Healthy controls are identified per batch via a hardcoded `WT_SUBJECTS` map in `scripts/build_finetune_normalized.py`. As of the initial run these are:

| Batch | WT subject(s) | Genotype |
|---|---|---|
| `mouse_app_lecanemab_batch2` | sub-AS118M9 | ApoE3 NegCtrl |
| `mouse_app_lecanemab_batch3` | sub-AS161F3, sub-AS164F5, sub-AS168F1 | ApoE3 + PBS (3 refs averaged) |
| `mouse_app_lecanemab_ki3_batch1` | sub-AS7F3 | WT N control |
| `mouse_app_lecanemab_ki3_batch2` | sub-AS7F1 | WT N control |
| `mouse_app_lecanemab_ki3_batch3` | sub-C57BL6 | C57BL6 N control |
| `mouse_app_vaccine_batch` | *(pooled from all other batches)* | — |

### Build script: `scripts/build_finetune_normalized.py`

```bash
# Full run (recommended) — processes every disease patch in ft/Abeta/manifest.json
pixi run python scripts/build_finetune_normalized.py

# Smoke test on one subject
pixi run python scripts/build_finetune_normalized.py --subject sub-AS40F2

# Run/rerun a single batch (merges into existing manifest, doesn't clobber other patches)
pixi run python scripts/build_finetune_normalized.py --dataset mouse_app_lecanemab_batch2

# Limit to first N disease patches (for quick iteration)
pixi run python scripts/build_finetune_normalized.py --limit 3

# Rebuild ft_normalized/Abeta/manifest.json from on-disk per-patch meta.json files.
# Use this if a partial run accidentally dropped entries from the manifest.
pixi run python scripts/build_finetune_normalized.py --rebuild-manifest-from-disk
```

The first full run takes ~1 hour: each WT brain incurs a one-time ~6 minute SimpleITK N4 cost (cached), then each disease patch costs only a few seconds of zarr I/O + segmentation. Reruns skip the N4 caching entirely.

### Per-patch napari view scripts: `scripts/generate_view_scripts.py`

For visual inspection (and for the human-review pass below), Lumivox can drop a tiny bash launcher next to every patch:

```bash
pixi run python scripts/generate_view_scripts.py
# -> ft_normalized/Abeta/{dataset}/{subject}/view_<patch_id>.sh   (one per patch)
# -> ft_normalized/Abeta/view_index.txt                            (master index)
```

Each generated script `cd`s to its own directory and opens napari with **5 layers**: raw (gray), normalized z-score (magma, hidden by default), otsu k=2 labels (visible), otsu k=3 labels (hidden — toggle to compare), and the editable GOLD labels layer (only if `seg_gold.nii.gz` has been initialized — see below). When the GOLD layer is loaded, `Ctrl+S` is bound to save it back to disk in place, so no file dialog dance. Invoke any patch with:

```bash
bash ft_normalized/Abeta/mouse_app_lecanemab_batch2/sub-AS40F2/view_sub-AS40F2_patch00.sh
```

The top-level `view_index.txt` lists every patch with dataset, region, and component counts (k2, k3) so you can scan it and prioritise the high-count patches where the k=2 vs k=3 decision matters most.

### Finalising the gold masks: `scripts/finalize_finetune_gold.py`

After `build_finetune_normalized.py` produces the candidate masks, the human-review step turns them into gold via five subcommands:

```bash
# 1) Bulk-init: pick whichever otsu variant you think wins more often.
#    Copies *_seg_otsu2.nii.gz -> *_seg_gold.nii.gz for every patch.
pixi run python scripts/finalize_finetune_gold.py init --use otsu2

# 2) See where you stand
pixi run python scripts/finalize_finetune_gold.py status

# 3) Override variant for individual patches where the OTHER one wins
pixi run python scripts/finalize_finetune_gold.py init --use otsu3 \
        --patch sub-AS208F2_patch00 --force

# 4) Open one patch in napari with all layers loaded for hand-editing
pixi run python scripts/finalize_finetune_gold.py napari --patch sub-AS40F2_patch01
# (or just run the per-patch view_*.sh script directly)

# 5) Mark a patch reviewed once you've cleaned it up
pixi run python scripts/finalize_finetune_gold.py review --patch sub-AS40F2_patch01 \
        --notes "deleted 2 boundary blobs"

# 6) When all patches are reviewed: write the final fine-tune training manifest
pixi run python scripts/finalize_finetune_gold.py manifest
# -> ft_normalized/Abeta/manifest_gold.json   (every entry has a seg_gold_path)
```

Review state lives in `ft_normalized/Abeta/.review_state.json` so you can quit and resume the next day. The `status` table flags patches that used WT global fallback during normalization (look for the `*` column) — those are worth a second look since their stats came from the whole WT brain, not from the matched anatomical region.

### Coordinate system caveats

The atlas-sampled center coordinates are in each disease subject's *own* full-resolution physical space (mm). Mapping the same physical mm into a different animal's voxel grid only works approximately — orientation is consistent but the physical-space origins can be offset between animals. In practice this affects ~13% of patches (6/46 in the initial Abeta run), all in `mouse_app_lecanemab_ki3_batch2` where AS7F1 covers a different physical bounding box than the disease subjects. The whole-brain WT fallback handles these cases automatically; no manual intervention required.

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
scripts/
├── build_training_manifests.py    # Build patch manifests for SSL pretraining
├── build_finetune_data.py         # Simple fine-tune patch builder (uses SPIMquant fieldfrac)
├── build_finetune_normalized.py   # WT-normalized fine-tune ground truth pipeline
├── finalize_finetune_gold.py      # Init / review / napari / manifest_gold workflow
├── generate_view_scripts.py       # Per-patch napari view_*.sh launchers
├── cache_patches.py               # Batch extract patches to .npy on localscratch
├── generate_qc_images.py          # QC figures for any patch manifest
└── *.sh                           # SLURM/launch scripts for pretraining + resume
ft/                                 # Output of build_finetune_data.py (simple variant)
└── {Abeta,Iba1}/{dataset}/{subject}/...
ft_normalized/                      # Output of build_finetune_normalized.py + finalize
└── Abeta/
    ├── manifest.json               # All processed patches + which WT(s) used
    ├── manifest_gold.json          # Final reviewed manifest (after finalize manifest)
    ├── .review_state.json          # Review progress, regenerated from finalize review
    ├── view_index.txt              # Index of per-patch view_*.sh launchers
    ├── wt_references/              # Cached N4 bias fields per WT animal
    └── {dataset}/{subject}/
        ├── *_crop128.nii.gz        # raw disease 128³ (training input)
        ├── *_normalized128.nii.gz  # WT z-score normalized 128³
        ├── *_seg_otsu2.nii.gz      # k=2 Otsu candidate
        ├── *_seg_otsu3.nii.gz      # k=3 Otsu candidate
        ├── *_seg_gold.nii.gz       # finalised gold mask (post review)
        ├── *_qc.png                # 5×3 QC figure
        ├── *_meta.json             # per-patch normalization metadata
        └── view_*.sh               # napari launcher for this patch
tests/               # Unit tests
```
