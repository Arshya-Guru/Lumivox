# Topological fine-fine-tuning

A second, short fine-tuning phase that starts from a **converged** segmentation
model (a member of the v4 fine-tuning sweep) and continues training with a
**topological loss** — to see whether enforcing the right topology improves both
Dice *and* the structural correctness of the Aβ-plaque masks.

This follows the fine-tuning in **[finetuning.md](finetuning.md)**: that phase
trains to convergence on Dice+CE; this phase is the topo cleanup on top.

---

## Why a separate phase (the SynthTopo pattern)

The design is lifted from **SynthTopo**
(`/nfs/khan/trainees/apooladi/topo/SynthTopo`), a SynthSeg-style pipeline:

1. Train to convergence on a conventional loss (Dice + CE).
2. **Then** run a short topological-loss phase — typically **~20% of the number
   of epochs the first phase took to converge**.

The split exists because persistent-homology (PH) losses are expensive and
unstable to train from scratch, but very effective as a *finishing* step once the
network already produces good-quality masks. Our frozen sweep members converge by
~epoch 50, so a topo phase of ~10–15 epochs is the natural target (the entry
defaults to 30; tune to ~20% of *your* convergence point).

---

## The loss: `topo_ce` (GT topology as the constraint)

Implemented in `lumivox/losses/topo.py` as `TopoCELoss` — a focused binary port
of SynthTopo's `topo_ce` (`TopologicalLoss`, `base_loss='dice+ce'`,
`topo_weight=1e-4`), adapted from its multi-class form to our single-foreground
case.

```
total = base_weight · DiceCE(logits, target)
      + topo_weight · TopoTerm(sigmoid(logits), target)
```

**The topological constraint comes from the ground truth, per sample.** For each
patch:

1. Compute the GT mask's Betti numbers `(b0, b1, b2)` = (connected components,
   loops, voids) via cubical PH. These are the *target topology*.
2. Compute the prediction's PH barcode on the complement `1 − p(foreground)` (so
   confident foreground is "born" early), and read each finite feature's
   **persistence** (death − birth) from the live probability tensor — so the term
   is differentiable.
3. Match the **top-`b_d`** most-persistent features in dimension `d` to the GT
   target and split the loss into two terms:
   - **A-term** — for the matched features, push persistence → 1 (keep the real
     components/loops crisp): `Σ (1 − persistence)`.
   - **Z-term** — for every *extra* feature beyond the target count, push
     persistence → 0 (kill spurious components, merged blobs, pinholes):
     `Σ persistence`.

For sparse plaques the dominant signal is `b0`: the prediction is pushed to have
the *same number of distinct plaques* as the GT — no spurious extra components and
no incorrectly-merged ones — which is exactly the failure mode the human review
notes flagged ("dropped edge CCs", "FPs").

PH backend: **CubicalRipser** (`cripser` = 6-connectivity, `tcripser` =
26-connectivity), the same engine SynthTopo uses. The differentiable-barcode
trick (structure computed on a detached copy, persistence values re-read from the
live tensor) is ported directly from `synthtopo/losses.py`.

> Deep supervision is intentionally **off** here: the topo term acts on the
> primary full-res output only, and carries its own Dice+CE base — matching
> SynthTopo, where topological losses are not wrapped in deep supervision.

---

## How it follows the sweep

`lumivox/training/finefinetune_lightning.py`:

- Loads a converged `*.ckpt` from `checkpoints/ft_v4_sweep/<member>/` via
  `--ft-checkpoint`, rebuilding the exact architecture from the checkpoint's saved
  hyper-parameters (`num_classes` / `dropout` / `deep_supervision`) and loading the
  FT-trained encoder+decoder weights (the pretrained encoder is *not* reloaded).
- Reuses the same dataset / dataloaders and the same **fixed val split**
  (`--split-seed 42`) so `val/dice` is directly comparable to the base FT run.
- Trains a short schedule at a **low LR** (default 1e-4 — the base phase already
  converged) with `TopoCELoss`, logging `train/{loss,base,topo,betti_err}` and
  `val/{loss,dice,topo,betti_err}`.

```bash
python -m lumivox.training.finefinetune_lightning \
    --ft-checkpoint checkpoints/ft_v4_sweep/ftv4_simclr_d10_frozen_s0/finetune-epoch0060-dice0.6550.ckpt \
    --manifest manifests/abeta_ft_v4_A.json \
    --epochs 30 --lr 1e-4 --topo-weight 1e-4 \
    --save-dir checkpoints/ftft_v4/ftv4_simclr_d10_frozen_s0 \
    --wandb
```

Key flags: `--topo-weight` (1e-4), `--construction` (`0`=6-conn / `N`=26-conn),
`--thresh` (restrict PH to the foreground bbox to bound cost), `--freeze-encoder`,
`--lr`. W&B project defaults to `lumivox-fine-fine-tuning`; checkpoints keep top-2
by `val/dice` + `last.ckpt`, with the same auto-resume as the base FT.

---

## Dependency & status

- **CubicalRipser is required for the topo term and is not yet in the env.** Until
  it's installed, `TopoCELoss` auto-detects its absence, warns once, and falls
  back to plain Dice+CE — so the module runs but does nothing topological. Add it
  with `pip install CubicalRipser` (provides both `cripser` and `tcripser`), then
  smoke-test with `--fast-dev-run`.
- **Backbone only.** No SLURM launcher yet (deliberately). When you're ready to
  run it across the sweep, the natural shape mirrors `launch_v4_sweep.sh`: one
  topo phase per chosen base checkpoint.
- **Untested path:** the cripser-backed topo computation has been ported faithfully
  from SynthTopo but not run end-to-end here (no cripser in the env). The Dice+CE
  fallback, the differentiability, and the module wiring are verified.

## What to evaluate

The whole point is the question: **does the topo phase improve Dice while reducing
topological error?** Compare, on the fixed val split, the base FT checkpoint vs.
its topo-fine-tuned version on:

- `val/dice` (should not regress — ideally improves),
- `val/betti_err` (should drop — fewer spurious/merged components),
- spurious-component count and plaque-count accuracy vs. GT.

This also dovetails with the ensemble: a topo-cleaned member should produce
crisper, less fragmented probability masks.
