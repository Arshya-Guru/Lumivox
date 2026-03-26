#!/bin/bash
# Abeta nnBYOL3D v2 — anti-collapse fix
# Changes from v1:
#   - accumulate_grad_batches=4 (effective batch=32)
#   - warmup 8 epochs (more time for EMA target to stabilize)
#   - lr=5e-4 (gentler for small effective batch)
#   - base_ema=0.996 (slower target, more stable)
#
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --time=48:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.pretrain_lightning \
    --method nnbyol3d \
    --manifest manifests/abeta_50k_alldata.json \
    --batch-size 8 \
    --crop-size 128 \
    --epochs 50 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --warmup-epochs 8 \
    --base-ema 0.996 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --accumulate-grad-batches 4 \
    --save-dir ./checkpoints/nnbyol3d_abeta_50k_v2 \
    --save-every 5 \
    --seed 42
