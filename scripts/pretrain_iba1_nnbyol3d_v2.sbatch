#!/bin/bash
#SBATCH --job-name=iba1_nnbyol3d
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/iba1_nnbyol3d_%j.out
#SBATCH --error=logs/iba1_nnbyol3d_%j.err

# Iba1 nnBYOL3D v2 — auto-resume, self-resubmitting
#
# Submit:  sbatch scripts/pretrain_iba1_nnbyol3d_v2.sbatch
# Monitor: tail -f logs/iba1_nnbyol3d_*.out
#
# This script:
#   1. Checks if a checkpoint exists → resumes with proper schedule continuation
#   2. If no checkpoint → starts fresh
#   3. Starts GPU/training monitor in background
#   4. On SLURM timeout (exit code 0 from signal), resubmits itself automatically

# Don't use set -e — we need to handle non-zero exit codes for resubmit logic
set -uo pipefail

cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

SAVE_DIR="./checkpoints/nnbyol3d_iba1_50k"
MANIFEST="manifests/iba1_50k_ki3.json"
EPOCHS=50
METHOD="nnbyol3d"

echo "======================================"
echo "Iba1 nnBYOL3D v2 — Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Start: $(date)"
echo "======================================"

# Start monitor in background
bash scripts/monitor_training.sh --interval 10 > "logs/iba1_nnbyol3d_${SLURM_JOB_ID}_monitor.csv" &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

# Check for existing checkpoint
CKPT=$(ls -t ${SAVE_DIR}/pretrain_epoch*.pt 2>/dev/null | head -1)

if [ -n "$CKPT" ]; then
    echo "Found checkpoint: $CKPT"
    echo "Resuming with schedule continuation..."

    pixi run python -m lumivox.training.pretrain_lightning \
        --method ${METHOD} \
        --manifest ${MANIFEST} \
        --batch-size 8 \
        --crop-size 128 \
        --epochs ${EPOCHS} \
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
    EXIT_CODE=$?
else
    echo "No checkpoint found. Starting fresh..."

    pixi run python -m lumivox.training.pretrain_lightning \
        --method ${METHOD} \
        --manifest ${MANIFEST} \
        --batch-size 8 \
        --crop-size 128 \
        --epochs ${EPOCHS} \
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
        --seed 42
    EXIT_CODE=$?
fi

echo "======================================"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "======================================"

# Check if training completed (pretrain_final.pt exists)
if [ -f "${SAVE_DIR}/pretrain_final.pt" ]; then
    echo "Training COMPLETE. pretrain_final.pt exists."
else
    echo "Training INCOMPLETE — likely timed out."
    echo "Latest checkpoint: $(ls -t ${SAVE_DIR}/pretrain_epoch*.pt 2>/dev/null | head -1)"

    # Auto-resubmit if not complete
    echo "Resubmitting job..."
    sbatch scripts/pretrain_iba1_nnbyol3d_v2.sbatch
    echo "Resubmitted."
fi
