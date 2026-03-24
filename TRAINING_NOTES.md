# Lumivox Training Notes

Reference document for pretraining decisions, debugging history, and operational knowledge.

## Data Pipeline

### Manifest System

Patches are pre-sampled from brain regions using zarrnii's `ZarrNiiAtlas.sample_region_patches()`. The manifest stores physical-space (mm) coordinates from the dseg atlas, which are converted to resampled zarr voxel coordinates at build time via:

1. Physical center -> full-res voxel (full-res zarr affine inverse)
2. Full-res voxel -> resampled voxel (JSON sidecar scale factors)

This two-step conversion is necessary because the resampled 4um zarr files have unreliable OME-Zarr metadata (zero-origin affine, orientation flags that don't match voxel layout). The full-res zarr metadata is authoritative. `crop_centered()` works on full-res but produces garbage on resampled — so we bypass it entirely with direct voxel slicing.

### Channel Resolution

Channel ordering varies between datasets (batch3: Abeta/CD31/YoPro, ki3: Iba1/Abeta/CD31). The manifest builder reads omero labels from the full-res zarr (resampled zarrs lack omero metadata) and stores the resolved channel index per patch.

### SPIMquant Preferences

Some datasets have multiple SPIMquant runs. `PREFERRED_SPIMQUANT` in `manifest.py` encodes which run to use per dataset. Datasets without per-subject TSV label files fall back to the bundled reference TSV (`lumivox/data/reference/seg-roi22_dseg.tsv`).

### Acquisition Filtering

Subjects with 45deg/90deg tilted acquisitions are filtered out by `_prefer_standard_acq()` — only standard `acq-imaris4x` zarrs are used.

## Localscratch Write-Through

The biggest training speedup. During epoch 1, each patch read from NFS zarr is written to `/localscratch/lumivox_patches/` as a uint16 `.npy` file. From epoch 2 onward, patches are loaded from local SSD via `np.load(mmap_mode="r")`.

**Impact measured:**

| Phase | GPU0 active | GPU1 active |
|---|---|---|
| Epoch 1 (NFS zarr) | 56-61% | 57-61% |
| Epoch 2+ (localscratch) | 97-98% | 98-99% |

**Size:** 50k patches x 256^3 x uint16 = ~1.6TB. h-nodes have 7TB localscratch.

**Important:** localscratch is per-node and per-job. If the job dies or you switch nodes, epoch 1 re-extracts everything. Resume with `--resume last.ckpt` picks up model state but not localscratch.

## Memory Management

### The Problem

32 DataLoader workers each caching zarr handles caused heap fragmentation and memory growth to 490GB+, triggering OOM kills. Python's allocator doesn't return freed arenas to the OS if any small object survives in them.

### The Fix

1. **Reduced workers to 16** (from 32) — halved baseline memory
2. **Periodic cache flush** every 5000 samples per worker — `_vol_cache.clear()` + `gc.collect()` + `malloc_trim(0)`. This forces glibc to return freed heap to OS
3. **Localscratch eliminates the problem entirely** — after epoch 1, zarr handles are freed and workers just do `np.load()`. Memory drops to ~50GB

### Flush Interval History

- 100: caused 30-90s GPU stalls (workers reopening zarr handles too often)
- 1000: better, but still some stalls
- 5000: ~0.6 flushes per epoch per worker, negligible impact
- After localscratch kicks in: flushes stop entirely (`_zarr_freed=True`)

## GPU Utilization Debugging

### DDP Sync Stalls

With DDP, both GPU processes synchronize gradients after every batch. If one process's workers deliver data slower (NFS latency variance), the other GPU sits idle. Measured 35-42% idle time on NFS.

Localscratch fixed this — uniform local SSD latency means both processes get data at the same speed. Idle time dropped to 1-2%.

### Monitoring

`scripts/monitor_training.sh` logs CSV every 5s with GPU util/mem/power, CPU load, RAM, and training metrics. Run on the compute node:

```bash
bash scripts/monitor_training.sh --interval 5 > simclr_monitor.csv &
```

Analyze with:
```bash
awk -F',' 'NR>1 && $2~/^[0-9]/ {if($2+0>=75)t++; n++} END {printf "GPU0 active: %.0f%%\n", t/n*100}' simclr_monitor.csv
```

## Model-Specific Notes

### SimCLR

- **Loss:** NT-Xent (contrastive). Starts ~2.5, should drop to ~1.0 within 5-10 epochs
- **Batch size sensitivity:** Less sensitive than BYOL — negatives in NT-Xent prevent collapse regardless of batch size
- **Current status:** batch_size=8 per GPU, 2 GPUs, effective batch=16. Working well
- **Note:** With DDP, each GPU only sees local 8 samples as negatives (no cross-GPU all_gather implemented). This is suboptimal but functional

### nnBYOL3D

- **Loss:** Regression (2 - 2*cosine_similarity). Very small values (1e-5) are normal IF representations are still diverse. Near-zero loss with collapsed representations = bad
- **Collapse risk:** BYOL has no negatives. Relies on predictor MLP + EMA target asymmetry to avoid collapse. This breaks down at small batch sizes
- **v1 collapsed:** batch=8, lr=1e-3, warmup=3, ema=0.99. Loss hit 1e-5 by epoch 3 and flatlined = complete collapse
- **v2 anti-collapse fix:**
  - `accumulate_grad_batches=4` → effective batch 32
  - `lr=5e-4` → gentler updates
  - `warmup_epochs=8` → more time for EMA target to stabilize
  - `base_ema=0.996` → slower target movement
- **How to detect collapse:** If loss drops to <1e-4 within the first 5 epochs and plateaus, it's collapsed. Good training: loss decreases gradually over many epochs, staying in the 0.01-1.0 range

### Warmup

Linear LR ramp from 0 to target LR over the first N epochs. Prevents the model from making huge jumps with random-weight gradients at the start. Especially important for BYOL where the online network can outrun the EMA target.

- SimCLR: 3 warmup epochs is fine (NT-Xent is more forgiving)
- nnBYOL3D: 8 warmup epochs (more fragile, needs gradual start)

## Hardware Reference

### Nodes

| Node type | GPUs | VRAM | CPUs | RAM |
|---|---|---|---|---|
| h-nodes | 2x L40S | 48GB each | 128 | 503GB |
| v-nodes | 2x A100 | 80GB each | 64 | 302GB |

### GPU Memory (L40S 48GB, bf16-mixed, crop=128)

| Model | Batch size | VRAM | Notes |
|---|---|---|---|
| SimCLR | 8 | ~34GB | Could go to 12-14 |
| SimCLR | 16 | OOM | |
| nnBYOL3D | 8 | ~33GB | 2x encoder (online+target) |

### Job Allocation

```bash
# Full h-node, 2 GPUs, 48h
srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --time=48:00:00 --pty bash
```

### Timing (with localscratch, 50k patches)

| Model | Epoch 1 (NFS write-through) | Epoch 2+ (localscratch) |
|---|---|---|
| SimCLR | ~5 hours | ~1 hour |
| nnBYOL3D | ~6.5 hours | ~1.5 hours |

## Training Scripts

All in `scripts/`:

| Script | Stain | Method | Status |
|---|---|---|---|
| `pretrain_abeta_simclr.sh` | Abeta | SimCLR | Running, loss decreasing |
| `pretrain_abeta_nnbyol3d.sh` | Abeta | nnBYOL3D v1 | Collapsed, do not use |
| `pretrain_abeta_nnbyol3d_v2.sh` | Abeta | nnBYOL3D v2 | Anti-collapse fix |
| `pretrain_iba1_simclr.sh` | Iba1 | SimCLR | Not yet run |
| `pretrain_iba1_nnbyol3d.sh` | Iba1 | nnBYOL3D | Needs v2 hyperparams |
| `resume_abeta_nnbyol3d.sh` | Abeta | nnBYOL3D | Resume from last.ckpt |

## Manifests

| Manifest | Stain | Patches | Subjects | Regions |
|---|---|---|---|---|
| `abeta_50k_alldata.json` | Abeta | 50k | 52 (all datasets) | 40% cortex, 30% hippo, 20% striatum, 10% cerebellum |
| `iba1_50k_ki3.json` | Iba1 | 50k | 28 (ki3 only) | Same distribution |
| `qc_3per_subject.json` | Abeta | 156 | 52 | QC validation |

## Subject Availability

| Dataset | Subjects | 4um zarr | dseg | Notes |
|---|---|---|---|---|
| lecanemab_batch2 | 8 | all | all | A-P orientation fix applied |
| lecanemab_batch3 | 10 | all | all | Fully validated |
| ki3_batch1 | 8 | all | all | sub-AS48F66 has 45deg variant (filtered out) |
| ki3_batch2 | 11 | all | all | sub-AS11F3 was missing dseg, now present |
| ki3_batch3 | 9 | all | all | Validated |
| vaccine_batch | 6/7 | 6/7 | 6/7 | sub-AS176F6 fullres only, sub-AS176F7 was missing dseg |

## Debugging Checklist

### Training won't start
- Delete `lightning_logs/` (stale CSV headers between methods)
- Check `--save-dir` is method-specific (parallel runs conflict otherwise)

### OOM (RAM)
- Reduce `--num-workers` (16 is safe for 256GB nodes)
- Check localscratch is working (`ls /localscratch/lumivox_patches/ | wc -l`)
- Flush interval too low causes memory growth between flushes

### OOM (GPU)
- Reduce `--batch-size`
- crop_size=128 uses ~2.4x more VRAM than crop_size=96

### BYOL collapse
- Loss drops to <1e-4 within first 5 epochs and flatlines
- Fix: increase effective batch (accumulate_grad_batches), lower LR, increase warmup, increase base_ema

### Wrong channel / weird patches
- Check omero metadata: `zarr.open(fullres_zarr).attrs["omero"]["channels"]`
- Some resampled zarrs lack omero — resolved via full-res zarr fallback
- Use QC images (`scripts/generate_qc_images.py`) to visually validate

### Localscratch not working
- Check from compute node: `ls /localscratch/lumivox_patches/ | wc -l`
- Should show 50000 after epoch 1
- `/localscratch/` is node-local, not visible from login node
