#!/bin/bash
# Iba1 SimCLR — 96³ crops, batch=24/gpu, 2x L40S
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --tmp=4000000 --time=48:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.pretrain_lightning \
    --method simclr \
    --manifest manifests/iba1_50k_ki3.json \
    --batch-size 24 \
    --crop-size 96 \
    --epochs 50 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-epochs 3 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ./checkpoints/simclr_iba1_50k_96 \
    --save-every 5 \
    --seed 42
