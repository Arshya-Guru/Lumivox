#!/bin/bash
# Abeta nnBYOL3D — 2x L40S, full h-node
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --time=48:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.pretrain_lightning \
    --method nnbyol3d \
    --manifest manifests/abeta_50k_alldata.json \
    --batch-size 8 \
    --crop-size 128 \
    --epochs 50 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-epochs 3 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ./checkpoints/nnbyol3d_abeta_50k \
    --save-every 5 \
    --seed 42
