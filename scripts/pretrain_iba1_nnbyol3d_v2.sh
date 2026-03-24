#!/bin/bash
# Iba1 nnBYOL3D v2 — 2x L40S, anti-collapse hyperparams
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --time=48:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.pretrain_lightning \
    --method nnbyol3d \
    --manifest manifests/iba1_50k_ki3.json \
    --batch-size 8 \
    --crop-size 128 \
    --epochs 50 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --warmup-epochs 8 \
    --base-ema 0.996 \
    --accumulate-grad-batches 4 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ./checkpoints/nnbyol3d_iba1_50k \
    --save-every 5 \
    --seed 42
