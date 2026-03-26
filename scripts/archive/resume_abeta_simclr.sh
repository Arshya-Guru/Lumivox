#!/bin/bash
# Resume Abeta SimCLR — 2x L40S, load weights from .pt checkpoint
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --time=24:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

# Find latest .pt checkpoint
CKPT=$(ls -t checkpoints/simclr_abeta_50k/pretrain_epoch*.pt 2>/dev/null | head -1)
echo "Resuming weights from: $CKPT"

pixi run python -m lumivox.training.pretrain_lightning \
    --method simclr \
    --manifest manifests/abeta_50k_alldata.json \
    --batch-size 8 \
    --crop-size 128 \
    --epochs 10 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-epochs 1 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ./checkpoints/simclr_abeta_50k \
    --save-every 5 \
    --seed 42 \
    --resume-weights "$CKPT"
