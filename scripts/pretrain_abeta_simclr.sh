#!/bin/bash
# Abeta SimCLR — 2x L40S, full h-node
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --time=24:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.pretrain_lightning \
    --method simclr \
    --manifest manifests/abeta_50k_alldata.json \
    --batch-size 8 \
    --crop-size 128 \
    --epochs 300 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-epochs 10 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ./checkpoints/simclr_abeta_50k \
    --save-every 25 \
    --seed 42
