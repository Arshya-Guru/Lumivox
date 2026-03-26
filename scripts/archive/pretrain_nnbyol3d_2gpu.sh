#!/bin/bash
# nnBYOL3D pretraining — 2x L40S via Lightning DDP
# srun --gres=gpu:l40s:2 --cpus-per-task=32 --mem=128G --time=4:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.pretrain_lightning \
    --method nnbyol3d \
    --manifest manifests/batch3_abeta_cortex_hipp_10k.json \
    --batch-size 24 \
    --crop-size 96 \
    --epochs 5 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-epochs 1 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ./checkpoints/nnbyol3d_batch3_2gpu \
    --save-every 5 \
    --seed 42
