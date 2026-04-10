#!/bin/bash
# Resume Abeta SimCLR 96³ — auto-detects latest checkpoint
# srun --gres=gpu:l40s:2 --cpus-per-task=48 --mem=256G --tmp=4000000 --time=48:00:00 --pty bash

cd /nfs/khan/trainees/apooladi/abeta/Lumivox

SAVE_DIR="./checkpoints/simclr_abeta_50k_96"
CKPT=$(ls -t ${SAVE_DIR}/pretrain_epoch*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "No checkpoint found in $SAVE_DIR!"; exit 1; fi

EPOCH=$(python3 -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False)['epoch'])")
REMAINING=$((50 - EPOCH))
echo "Resuming from: $CKPT (epoch $EPOCH, $REMAINING epochs remaining)"

pixi run python -m lumivox.training.pretrain_lightning \
    --method simclr \
    --manifest manifests/abeta_50k_alldata.json \
    --batch-size 24 \
    --crop-size 96 \
    --epochs ${REMAINING} \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-epochs 3 \
    --num-workers 16 \
    --precision bf16-mixed \
    --devices 2 \
    --save-dir ${SAVE_DIR} \
    --save-every 5 \
    --seed 42 \
    --resume-weights "$CKPT"
