#!/bin/bash
# Pipeline smoke test for Abeta FT.
#
# Three stages on 1 GPU:
#   1) fast-dev-run  -> 1 train batch + 1 val batch, validates ckpt load / forward / backward / val.
#   2) 3-epoch real run at batch=2, captures VRAM peak.
#   3) 3-epoch real run at batch=4, confirms VRAM headroom for the real launch.
#
# Each stage writes into a separate save-dir so they don't clobber each other.

set -uo pipefail

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

CKPT="./checkpoints/simclr_abeta_50k/pretrain_epoch0045.pt"
MANIFEST="manifests/abeta_ft_first_pass.json"
SMOKE_ROOT="./checkpoints/finetune_abeta_smoke"
mkdir -p "$SMOKE_ROOT"

run_stage() {
    local name="$1"; shift
    echo "===================================================="
    echo "STAGE: $name"
    echo "ARGS: $*"
    echo "Start: $(date)"
    echo "===================================================="
    pixi run python -m lumivox.training.finetune_lightning \
        --checkpoint "$CKPT" \
        --manifest "$MANIFEST" \
        --freeze-encoder \
        --num-workers 4 \
        --precision bf16-mixed \
        --devices 1 \
        --seed 42 \
        "$@"
    local rc=$?
    echo "Stage '$name' exit code: $rc"
    return $rc
}

# Stage 1: fast-dev-run
run_stage "fast-dev-run" \
    --fast-dev-run \
    --batch-size 2 \
    --train-repeats 1 \
    --save-dir "${SMOKE_ROOT}/fdr" || exit 1

# Stage 2: 3 epochs, batch=2
run_stage "real-bs2" \
    --epochs 3 \
    --batch-size 2 \
    --train-repeats 4 \
    --val-fraction 0.2 \
    --save-dir "${SMOKE_ROOT}/bs2" || exit 1

# Stage 3: 3 epochs, batch=4
run_stage "real-bs4" \
    --epochs 3 \
    --batch-size 4 \
    --train-repeats 4 \
    --val-fraction 0.2 \
    --save-dir "${SMOKE_ROOT}/bs4" || exit 1

echo "===================================================="
echo "ALL SMOKE STAGES PASSED"
echo "End: $(date)"
echo "===================================================="
