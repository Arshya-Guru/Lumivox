#!/bin/bash
#SBATCH --job-name=infer_AS36F4_simclr
#SBATCH --partition=hx,vx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/infer_AS36F4_simclr_%j.out
#SBATCH --error=logs/infer_AS36F4_simclr_%j.err

# Whole-brain Abeta segmentation inference on sub-AS36F4 (4 um isotropic),
# using the best SimCLR-frozen FT checkpoint (ep 9, val/dice 0.7081).
# Outputs uint8 probability map (sigmoid * 255) as a multi-resolution OME-NGFF
# v0.5 zarr packed as a .ozx file (zip, store).

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

INZARR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/bids/derivatives/resampled/sub-AS36F4/micr/sub-AS36F4_sample-brain_acq-imaris4x_res-4um_SPIM.ome.zarr
CKPT=checkpoints/finetune_abeta_simclr_frozen/finetune-ep0009-dice0.7081.ckpt
OUT_DIR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/derivatives/lumivox_abeta_test/sub-AS36F4
OUT_OZX=$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTsimclrFrozenEp0009Dice7081Stride64_probmap.ozx

mkdir -p "$OUT_DIR"
echo "===================================================="
echo "Whole-brain inference (SimCLR-frozen) — Job $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi -L 2>/dev/null
echo "Input:    $INZARR"
echo "Ckpt:     $CKPT"
echo "Output:   $OUT_OZX"
echo "Start:    $(date)"
echo "===================================================="

pixi run python scripts/infer_finetune_brain.py \
    --input-zarr "$INZARR" \
    --channel 0 \
    --level 0 \
    --checkpoint "$CKPT" \
    --output-ozx "$OUT_OZX" \
    --stride 64 \
    --batch-size 32 \
    --build-pyramid 2
EXIT_CODE=$?

echo "===================================================="
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
ls -la "$OUT_OZX" 2>/dev/null
echo "===================================================="
exit $EXIT_CODE
