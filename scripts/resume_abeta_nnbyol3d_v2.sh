#!/bin/bash
# Resume Abeta nnBYOL3D v2 — 2x L40S, load weights from .pt checkpoint
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --time=24:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

# Find latest .pt checkpoint
CKPT=$(ls -t checkpoints/nnbyol3d_abeta_50k_v2/pretrain_epoch*.pt 2>/dev/null | head -1)
echo "Resuming weights from: $CKPT"

pixi run python -m lumivox.training.pretrain_lightning \
    --method nnbyol3d \
    --manifest manifests/abeta_50k_alldata.json \
    --batch-size 8 \
    --crop-size 128 \
    --epochs 10 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --warmup-epochs 1 \
    --base-ema 0.996 \
    --accumulate-grad-batches 4 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ./checkpoints/nnbyol3d_abeta_50k_v2 \
    --save-every 5 \
    --seed 42 \
    --resume-weights "$CKPT"
