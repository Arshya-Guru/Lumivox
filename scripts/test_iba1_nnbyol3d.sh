#!/bin/bash
# TEST Iba1 nnBYOL3D — 1x L40S, 2 epochs, verify schedules work
# srun --gres=gpu:l40s:1 --cpus-per-task=24 --mem=128G --time=4:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.pretrain_lightning \
    --method nnbyol3d \
    --manifest manifests/iba1_50k_ki3.json \
    --batch-size 8 \
    --crop-size 128 \
    --epochs 2 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --warmup-epochs 8 \
    --base-ema 0.996 \
    --accumulate-grad-batches 4 \
    --num-workers 8 \
    --precision bf16-mixed \
    --devices 1 \
    --save-dir ./checkpoints/test_iba1_nnbyol3d \
    --save-every 1 \
    --seed 42
