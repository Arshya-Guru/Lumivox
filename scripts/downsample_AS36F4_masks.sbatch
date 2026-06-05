#!/bin/bash
#SBATCH --job-name=mask_to_nii
#SBATCH --partition=hx,vx
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/mask_to_nii_%j.out
#SBATCH --error=logs/mask_to_nii_%j.err

# Resample each AS36F4 Abeta mask (OME-Zarr) onto the level-3 reference NIfTI
# grid (768x1280x248 at 13x13x8 um voxels) using manual IAR -> RPI conversion.
# Also produces a source-channel-via-same-conversion nifti for alignment QC.

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

# Use level-5 SPIM nifti from spimquant as the reference grid — it's the only
# one with anatomically correct orientation (RAI axcodes) and full brain z
# extent. Level-3 (in bids/derivatives/resampled) is A-P mirrored and has
# wrong z voxel size.
REF=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/derivatives/spimquant-v0.6.0rc2_84a605e_ozx/sub-AS36F4/micr/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-5_SPIM.nii.gz
SRC=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/bids/derivatives/resampled/sub-AS36F4/micr/sub-AS36F4_sample-brain_acq-imaris4x_res-4um_SPIM.ome.zarr
OUT_DIR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/derivatives/lumivox_abeta_test/sub-AS36F4

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Start: $(date)"
echo ""

# --- QC: source Abeta channel through the same conversion. If this overlays
#     the level-3 SPIM nifti in your viewer, the conversion is correct.
echo "===================================================="
echo "QC: source Abeta channel via manual RPI conversion"
echo "----------------------------------------------------"
pixi run python scripts/downsample_mask_to_nii.py \
    --mask-zarr "$SRC" \
    --reference-nii "$REF" \
    --output-nii "$OUT_DIR/QC_sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-5_desc-lumivoxResampledIntensity.nii.gz" \
    --reduction mean
echo "exit: $?"
echo ""

# --- The three masks
for tag in \
    "lumivoxFTnnbyol3dFrozenEp0064Dice7350Thr0p5" \
    "lumivoxFTnnbyol3dFrozenEp0064Dice7350Stride64Thr0p5" \
    "lumivoxFTsimclrFrozenEp0009Dice7081Stride64Thr0p5"; do
    MASK=$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-${tag}_mask.ome.zarr
    OUT=$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-5_desc-${tag}_mask.nii.gz
    echo "===================================================="
    echo "src:  $MASK"
    echo "dst:  $OUT"
    echo "----------------------------------------------------"
    pixi run python scripts/downsample_mask_to_nii.py \
        --mask-zarr "$MASK" \
        --reference-nii "$REF" \
        --output-nii "$OUT" \
        --reduction max
    echo "exit: $?"
done

echo ""
echo "===================================================="
echo "End: $(date)"
ls -la "$OUT_DIR"/*.nii.gz "$OUT_DIR"/QC_*.nii.gz 2>/dev/null
