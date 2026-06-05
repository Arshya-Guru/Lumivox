#!/bin/bash
#SBATCH --job-name=ft_v4
#SBATCH --partition=hx
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One member of the v4 fine-tuning sweep. Driven entirely by env vars exported
# by scripts/launch_v4_sweep.sh (do not submit this directly — use the launcher,
# which sets the job name, save dir, and all hyperparameters):
#   RUN_NAME  CKPT  MANIFEST  DROPOUT  FREEZE_FLAG  ENC_LR_FACTOR  SEED
#   SPLIT_SEED  SAVE_DIR  WANDB_PROJECT  EPOCHS  TRAIN_REPEATS
#
# On clean completion it writes $SAVE_DIR/DONE so the launcher won't resubmit.
# If the job is killed (e.g. 48h limit) it leaves last.ckpt and the training
# script auto-resumes from it on the next launch.

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs "$SAVE_DIR"

echo "===================================================="
echo "v4 FT sweep member: ${RUN_NAME}   Job ${SLURM_JOB_ID}"
echo "Node: $(hostname)"; nvidia-smi -L 2>/dev/null || echo "  (no nvidia-smi)"
echo "ckpt=${CKPT}"
echo "dropout=${DROPOUT}  freeze='${FREEZE_FLAG}'  enc_lr_factor=${ENC_LR_FACTOR}  seed=${SEED}"
echo "save_dir=${SAVE_DIR}"
echo "Start: $(date)"
echo "===================================================="

pixi run python -m lumivox.training.finetune_lightning \
    --checkpoint "${CKPT}" \
    --manifest "${MANIFEST}" \
    ${FREEZE_FLAG} \
    --epochs "${EPOCHS}" \
    --batch-size 2 \
    --lr 1e-3 \
    --encoder-lr-factor "${ENC_LR_FACTOR}" \
    --weight-decay 1e-2 \
    --dropout "${DROPOUT}" \
    --val-fraction 0.2 \
    --train-repeats "${TRAIN_REPEATS}" \
    --num-workers 8 \
    --precision bf16-mixed \
    --devices 1 \
    --save-dir "${SAVE_DIR}" \
    --seed "${SEED}" \
    --split-seed "${SPLIT_SEED}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${RUN_NAME}"
EXIT_CODE=$?

echo "===================================================="
echo "End: $(date)   exit=${EXIT_CODE}"
if [ "${EXIT_CODE}" -eq 0 ]; then
    touch "${SAVE_DIR}/DONE"
    echo "wrote ${SAVE_DIR}/DONE"
fi
echo "===================================================="
exit ${EXIT_CODE}
