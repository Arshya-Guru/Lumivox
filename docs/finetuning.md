# Fine-tuning: Aβ plaque segmentation

This document describes how Lumivox fine-tunes a pretrained SSL encoder into a 3D
Aβ-plaque segmentation model, and the deep-ensemble + decoder-dropout sweep used
to make robust probability masks.

The design is informed by the **ACE pipeline** (Goubran lab, *Nature Methods*
2025), which segments teravoxel light-sheet volumes with a dropout-regularised
network and gets its robustness from **Monte-Carlo dropout + model ensembling**.
We adapt the same two ideas — dropout regularisation and ensembling — to a small
expert-curated label set, with a self-supervised encoder standing in for ACE's
transformer backbone.

---

## 1. Ground truth — v4, accepted patches only

Fine-tuning trains **only on the v4 ground-truth patch set**, and within it, only
on patches the annotator explicitly **accepted**.

- **Location:** `/nfs/trident3/lightsheet/khan/arshya_gt_patches/ft_normalized/Abeta/v4/`
- **Built by:** `scripts/build_abeta_gt_v4.py` (atlas-sampled OZX patches +
  precomputed v0.5 patches → `crop_4um`/`mask_4um` at 128³, plus full-res pairs).

Each patch directory carries a one-character **review marker** the annotator
dropped during manual QC in 3D Slicer / nnInteractive:

| Marker file | Meaning |
|---|---|
| `A` / `A.txt` | **accept** — mask is good (edits already baked into `mask_4um.nii.gz`) |
| `R` / `R.txt` | reject |
| `M` / `M.txt` | maybe (uncertain) |
| `empty` | mask should be all-zero (a true negative) |

An empty marker *file* still counts — the annotator just touched it without notes.
Manual edits (dropping edge connected-components / false positives, adding missed
plaques or vessels via nnInteractive) are already saved into `mask_4um.nii.gz`.

### Building the training manifest

```bash
pixi run python scripts/build_v4_ft_manifest.py --count-pos
# -> manifests/abeta_ft_v4_A.json
```

`build_v4_ft_manifest.py` joins the v4 manifest with the markers and emits **only
clean accepts** (an `A` with no competing `R`/`M`, and no ambiguous compound
name). Everything else — `M`, `R`, edge cases, unreviewed — is reported but
excluded, so every patch in the manifest is genuinely an accept. It also resolves
a gotcha: the v0.5 patch dirs were renamed with a leading `NNN_` index *after* the
v4 manifest was written, so their stored paths are stale and are re-matched by
basename.

Current set: **162 accepted patches across 31 subjects** (105 atlas-OZX + 57 v0.5;
4 are accepted-empty negatives). The manifest schema matches
`lumivox/data/dataset_finetune.py` (`raw_path` = `crop_4um`, `seg_gold_path` =
`mask_4um`, plus `subject_id`/`patch_id`).

---

## 2. Model

A pretrained encoder is loaded into a `ResidualEncoderUNet`; the decoder + seg
heads are randomly initialised and trained.

- **Encoders (128³):** `checkpoints/simclr_abeta_50k/pretrain_epoch0045.pt` and
  `checkpoints/nnbyol3d_abeta_50k_v2/pretrain_epoch0040.pt` (the non-`_96`
  variants — `_96` are 96³). Weights are extracted via
  `extract_encoder_weights()` and loaded `strict=True` into `model.encoder`.
- **Encoder ≈ 90 M params, decoder ≈ 12 M.**

### True decoder dropout

`DecoderDropoutWrapper` (`lumivox/training/finetune.py`) applies `nn.Dropout3d`
to the **output of every decoder stage** via forward hooks, leaving the
pretrained encoder completely untouched. It's standard dropout — stochastic in
training, off at eval — and preserves deep supervision.

> This replaced an earlier wrapper that only dropped the encoder *skip-connection
> inputs*. Dropping the decoder's own feature maps is the regulariser we actually
> want for a randomly-initialised decoder on a small label set.

### Frozen vs unfrozen

- **Frozen:** encoder is frozen + `eval()`; only the 12 M decoder trains.
- **Unfrozen:** the encoder trains too, but at **0.1× the decoder LR** (two AdamW
  param groups: encoder @ 1e-4, decoder @ 1e-3) so the pretrained features are
  only gently nudged.

---

## 3. The sweep — 40 members, 5-model deep ensembles

We don't train one model; we train **five per configuration** so inference can
average them into a probability mask (if 4 of 5 fire on a voxel, p≈0.8). The five
members share one **fixed** subject-level val split (`--split-seed 42`) and vary
only `--seed` (decoder init + augmentation + shuffle) — a deep ensemble, not
cross-validation.

The full matrix is **40 runs**:

| Axis | Values |
|---|---|
| Encoder | `simclr`, `nnbyol3d` |
| Decoder dropout | `0.1`, `0.2` |
| Encoder | `frozen`, `unfrozen` (@0.1× LR) |
| Seed | `0, 1, 2, 3, 4` |

→ 2 × 2 × 2 × 5 = 40. It's a "try-a-few-things" sweep over the regularisation
knobs (dropout strength, frozen vs fine-tuned encoder, which SSL backbone), each
giving a 5-model ensemble.

### Training settings (per member)

| | |
|---|---|
| Epochs | 150 |
| Batch size | 2 (InstanceNorm → batch-size-insensitive; ~258 steps/epoch at `train-repeats 4`) |
| Loss | Dice + BCE (0.5/0.5) with deep supervision (`1, .5, .25, .125, .0625`) |
| Optimizer | AdamW, weight-decay 1e-2, PolyLR `(1−t)^0.9` |
| Precision | bf16-mixed |
| Augmentation | per-sample z-score, flips, rot90, intensity jitter (train only) |
| GPU | 1× L40S, 48 h wall limit |

---

## 4. Running the sweep

Everything is driven by one idempotent launcher (**run it from a SLURM submit
node**):

```bash
scripts/launch_v4_sweep.sh --test     # one 2-epoch run that exercises W&B end-to-end
scripts/launch_v4_sweep.sh --status   # print done/running/pending; submit nothing
scripts/launch_v4_sweep.sh            # submit pending members, up to the cap
```

How it works:

- Each member runs on **1× L40S**; at most **`MAX_CONCURRENT` (default 2)** run at
  once so the sweep doesn't hog the cluster.
- One invocation submits only what still needs doing: a member with a `DONE`
  sentinel is skipped, one already in `squeue` is skipped, the rest are submitted
  up to the cap. **Re-run it** (or `watch -n 600 scripts/launch_v4_sweep.sh`, or
  cron) to keep topping up until all 40 are done.
- A member that is killed or hits the 48 h limit leaves `last.ckpt`, and the next
  submission **auto-resumes** from it.

Overridable via env vars (no file edits): `MAX_CONCURRENT`, `EPOCHS`,
`TRAIN_REPEATS`, `SPLIT_SEED`, `WANDB_PROJECT`.

### Weights & Biases

Every member logs to the **`lumivox-fine-tuning`** project (run name =
`ftv4_<enc>_<dropout>_<freeze>_s<seed>`), streaming `train/loss`, `val/loss`,
`val/dice`, and LR. Group by the name fields to compare frozen-vs-unfrozen,
d10-vs-d20, simclr-vs-nnbyol3d. Run `wandb login` once before launching; if the
compute nodes are air-gapped, set `WANDB_MODE=offline` and `wandb sync` later.

### Outputs

Per member, under `checkpoints/ft_v4_sweep/<run_name>/`:

- top-3 checkpoints by `val/dice` + `last.ckpt` (for resume),
- `lightning_logs/.../metrics.csv`,
- `DONE` on clean completion.

> Each `.ckpt` is a full Lightning checkpoint (~0.5 GB frozen, ~1.2 GB unfrozen);
> the whole sweep is ~140 GB. For the ensemble you only need the single best
> checkpoint per member, so `save_top_k` can be dropped to 1 to save disk.

---

## 5. Inference (planned)

Not built yet. The intended path: for a chosen configuration, run its **5 member
checkpoints** over a target volume and average their predictions into a
**probability mask**; threshold for a binary mask. The per-voxel disagreement
across members (and/or MC-dropout sampling within a member) gives an
**uncertainty map** — which is exactly the signal to drive active learning over
the remaining `M` / unreviewed patches.
