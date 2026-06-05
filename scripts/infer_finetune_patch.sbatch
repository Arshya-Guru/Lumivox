#!/bin/bash
#SBATCH --job-name=ft_infer
#SBATCH --partition=hx,vx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/ft_infer_%j.out
#SBATCH --error=logs/ft_infer_%j.err

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs qc

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi -L 2>/dev/null
echo "Args: $*"

# Run inferences on a few val patches with both best checkpoints
SIMCLR=checkpoints/finetune_abeta_simclr_frozen/finetune-ep0009-dice0.7081.ckpt
NNBYOL=checkpoints/finetune_abeta_nnbyol3d_frozen/finetune-ep0064-dice0.7350.ckpt

# Pick val patches: one with plaques, one with sparse plaques, one empty
for patch in sub-AS39Me1_patch00 sub-AS135M8_patch00 sub-AS164F3_patch01; do
    echo "===== $patch ====="
    for ckpt_name in simclr nnbyol3d; do
        if [ "$ckpt_name" = "simclr" ]; then ckpt=$SIMCLR; else ckpt=$NNBYOL; fi
        echo "--- $ckpt_name ---"
        pixi run python scripts/infer_finetune_patch.py \
            --checkpoint "$ckpt" \
            --manifest manifests/abeta_ft_first_pass.json \
            --patch-id "$patch" \
            --output "qc/infer_${ckpt_name}_${patch}.png"
    done
done

echo "Done at $(date)"
ls -la qc/infer_*.png
