#!/bin/bash
#SBATCH --job-name=ft_brain_smoke
#SBATCH --partition=hx,vx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ft_brain_smoke_%j.out
#SBATCH --error=logs/ft_brain_smoke_%j.err

# Whole-brain inference smoke test: 16 tiles only, end-to-end including
# pyramid build + ozx packing, on the AS36F4 brain with the nnBYOL3D ckpt.

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

INZARR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/bids/derivatives/resampled/sub-AS36F4/micr/sub-AS36F4_sample-brain_acq-imaris4x_res-4um_SPIM.ome.zarr
CKPT=checkpoints/finetune_abeta_nnbyol3d_frozen/finetune-ep0064-dice0.7350.ckpt
OUT_DIR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/derivatives/lumivox_abeta_test/sub-AS36F4
OUT_OZX=$OUT_DIR/SMOKE_DELETE_ME_sub-AS36F4_smoke16tiles.ozx

mkdir -p "$OUT_DIR"
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi -L 2>/dev/null
echo ""

pixi run python scripts/infer_finetune_brain.py \
    --input-zarr "$INZARR" \
    --channel 0 \
    --level 0 \
    --checkpoint "$CKPT" \
    --output-ozx "$OUT_OZX" \
    --batch-size 8 \
    --build-pyramid 0 \
    --limit-tiles 16

echo "Smoke output: $OUT_OZX"
ls -la "$OUT_OZX"
