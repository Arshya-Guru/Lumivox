#!/bin/bash
#SBATCH --job-name=ft_abeta_nnbyol3d
#SBATCH --partition=hx
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Sizing rationale: dataset is 28 patches at 128^3 (~8MB each), workers=8 per GPU × 2 GPUs
# DDP, smoke peak VRAM 9.3 GiB. Actual CPU usage ~10 cores, actual RAM <2 GiB. The original
# 48 CPU / 256G allocation was inherited from pretraining sbatches and is grossly oversized
# for this FT workload — leaves more headroom for queued jobs on shared L40S nodes.
#SBATCH --output=logs/ft_abeta_nnbyol3d_%j.out
#SBATCH --error=logs/ft_abeta_nnbyol3d_%j.err

# Abeta FT from nnBYOL3D pretrained encoder. Encoder FROZEN.
# Settings unchanged from scripts/finetune_abeta_nnbyol3d.sh — sbatch wrapper only.
# Submit:  sbatch scripts/finetune_abeta_nnbyol3d.sbatch
# Monitor: tail -f logs/ft_abeta_nnbyol3d_*.out

set -uo pipefail

cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

SAVE_DIR="./checkpoints/finetune_abeta_nnbyol3d_frozen"

echo "===================================================="
echo "Abeta FT (nnBYOL3D, frozen) — Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi -L 2>/dev/null || echo "  nvidia-smi unavailable"
echo "Start: $(date)"
echo "Save dir: $SAVE_DIR"
echo "===================================================="

pixi run python -m lumivox.training.finetune_lightning \
    --checkpoint ./checkpoints/nnbyol3d_abeta_50k_v2/pretrain_epoch0040.pt \
    --manifest manifests/abeta_ft_first_pass.json \
    --freeze-encoder \
    --epochs 200 \
    --batch-size 2 \
    --lr 1e-3 \
    --weight-decay 1e-2 \
    --dropout 0.0 \
    --val-fraction 0.2 \
    --train-repeats 8 \
    --num-workers 8 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir "$SAVE_DIR" \
    --seed 42
EXIT_CODE=$?

echo "===================================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "===================================================="
exit $EXIT_CODE
