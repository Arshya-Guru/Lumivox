#!/bin/bash
#SBATCH --job-name=orient_variants
#SBATCH --partition=hx,vx
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/orient_variants_%j.out
#SBATCH --error=logs/orient_variants_%j.err

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

SRC=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/bids/derivatives/resampled/sub-AS36F4/micr/sub-AS36F4_sample-brain_acq-imaris4x_res-4um_SPIM.ome.zarr
REF=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/bids/derivatives/resampled/sub-AS36F4/micr/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-3_SPIM.nii
OUT_DIR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/derivatives/lumivox_abeta_test/sub-AS36F4

echo "Job $SLURM_JOB_ID on $(hostname)"
pixi run python scripts/orient_variants.py \
    --source-zarr "$SRC" \
    --reference-nii "$REF" \
    --out-dir "$OUT_DIR"
echo "exit: $?"
ls -la "$OUT_DIR"/VARIANT_*.nii.gz 2>/dev/null
