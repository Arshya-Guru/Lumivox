#!/bin/bash
# SimCLR pretraining test — run from interactive GPU session
# srun --gres=gpu:l40s:1 --cpus-per-task=32 --mem=64G --time=4:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.pretrain \
    --method simclr \
    --manifest manifests/batch3_abeta_cortex_hipp_10k.json \
    --batch-size 28 \
    --crop-size 96 \
    --epochs 5 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-epochs 1 \
    --num-workers 16 \
    --precision bf16-mixed \
    --save-dir ./checkpoints/simclr_batch3_test \
    --save-every 5 \
    --seed 42
