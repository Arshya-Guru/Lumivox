#!/bin/bash
#SBATCH --job-name=eval_simclr
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_simclr_%A_%a.out
#SBATCH --error=logs/eval_simclr_%A_%a.err

# Ensemble-inference every eval input with the simclr models.
# One array task per {frz,unf} x {10,20} config (4 tasks, 1 L40S each); each task
# runs scripts/inference.py sequentially over every relevant input .nii.gz —
# the 1024^3 patches (../simclr_eval/sub-*/NNN.nii.gz) AND the val crops
# (../simclr_eval/val_patches/*/crop_4um.nii.gz). Outputs land in a per-model
# subdir next to each input (e.g. .../sub-AS40F2/sim_frz20/001_prob.nii.gz).
#
#   sbatch scripts/eval_simclr.sbatch

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs
EVAL=../simclr_eval

MODELS=(sim_frz10 sim_frz20 sim_unf10 sim_unf20)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "===================================================="
echo "Array task ${SLURM_ARRAY_TASK_ID} -> model ${MODEL}   Job ${SLURM_ARRAY_JOB_ID}"
echo "Node: $(hostname)"; nvidia-smi -L 2>/dev/null | head -1
echo "Start: $(date)"
echo "===================================================="

# Every input: 1024^3 patches + val crops (exclude *_ozxmask/_prob/_maskthr05 and
# the per-model output subdirs by matching only the bare names).
mapfile -t FILES < <(
  ls "$EVAL"/sub-*/[0-9][0-9][0-9].nii.gz 2>/dev/null
  ls "$EVAL"/val_patches/*/crop_4um.nii.gz 2>/dev/null
)
echo "Found ${#FILES[@]} input files for ${MODEL}"

ok=0; fail=0
for f in "${FILES[@]}"; do
  echo "---- ${MODEL} :: ${f} ----"
  if pixi run python scripts/inference.py "$f" --model "$MODEL"; then
    ok=$((ok+1))
  else
    echo "  FAILED: $f"; fail=$((fail+1))
  fi
done

echo "===================================================="
echo "End: $(date)   model=${MODEL}   ok=${ok}  failed=${fail}"
echo "===================================================="
