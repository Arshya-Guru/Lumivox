#!/bin/bash
# Abeta segmentation FT from SimCLR pretrained encoder.
# Encoder FROZEN — only the 12M decoder + seg heads train.
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --time=24:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

pixi run python -m lumivox.training.finetune_lightning \
    --checkpoint ./checkpoints/simclr_abeta_50k/pretrain_epoch0045.pt \
    --manifest manifests/abeta_ft_first_pass.json \
    --freeze-encoder \
    --epochs 200 \
    --batch-size 2 \
    --lr 1e-3 \
    --weight-decay 1e-2 \
    --dropout 0.0 \
    --val-fraction 0.2 \
    --train-repeats 8 \
    --num-workers 8 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ./checkpoints/finetune_abeta_simclr_frozen \
    --seed 42
