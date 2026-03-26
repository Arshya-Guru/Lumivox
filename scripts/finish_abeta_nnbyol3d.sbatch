#!/bin/bash
#SBATCH --job-name=finish_abeta_nn
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --tmp=4000000
#SBATCH --output=logs/finish_abeta_nnbyol3d_%j.out
#SBATCH --error=logs/finish_abeta_nnbyol3d_%j.err

# Finish Abeta nnBYOL3D v2 — epochs 41-50 with correct schedule continuation
#
# Loads pretrain_epoch0040.pt, runs 10 more epochs.
# LR + EMA tau schedules continue from step ~32099 within the original
# 50-epoch schedule (max_steps=39050). This produces the final 10 epochs
# that the original run couldn't complete due to the 48h SLURM limit.
#
# Submit: sbatch scripts/finish_abeta_nnbyol3d.sbatch

set -uo pipefail

cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

SAVE_DIR="./checkpoints/nnbyol3d_abeta_50k_v2"
MANIFEST="manifests/abeta_50k_alldata.json"
CKPT="${SAVE_DIR}/pretrain_epoch0040.pt"

echo "======================================"
echo "Finish Abeta nnBYOL3D v2 — Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Resuming from: $CKPT"
echo "Start: $(date)"
echo "======================================"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint $CKPT not found!"
    exit 1
fi

# Start monitor
bash scripts/monitor_training.sh --interval 10 > "logs/finish_abeta_nnbyol3d_${SLURM_JOB_ID}_monitor.csv" &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# Run 10 more epochs with schedule continuation.
# --epochs 10 means the Trainer runs 10 new epochs.
# --resume-weights loads the model and sets step_offset from the checkpoint's
# global_step. total_max_steps = step_offset + (10 * steps_per_epoch) so the
# LR cosine and EMA tau schedules continue from the right position within a
# total schedule that covers original + remaining training.
pixi run python -m lumivox.training.pretrain_lightning \
    --method nnbyol3d \
    --manifest ${MANIFEST} \
    --batch-size 8 \
    --crop-size 128 \
    --epochs 10 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --warmup-epochs 8 \
    --base-ema 0.996 \
    --accumulate-grad-batches 4 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ${SAVE_DIR} \
    --save-every 5 \
    --seed 42 \
    --resume-weights "$CKPT"

echo "======================================"
echo "End: $(date)"
echo "======================================"

if [ -f "${SAVE_DIR}/pretrain_final.pt" ]; then
    echo "Training COMPLETE. pretrain_final.pt saved."
    echo "Use ${SAVE_DIR}/pretrain_final.pt for fine-tuning."
else
    echo "WARNING: pretrain_final.pt not found. Check logs."
fi
