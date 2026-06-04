#!/bin/bash
#SBATCH --job-name=binarize_AS36F4
#SBATCH --partition=hx,vx
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/binarize_AS36F4_%j.out
#SBATCH --error=logs/binarize_AS36F4_%j.err

# Binarize the metadata-fixed nnBYOL3D probmap (0.5 threshold) and write as
# multi-resolution OME-Zarr directory (not zipped). No GPU needed.

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

OUT_DIR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/derivatives/lumivox_abeta_test/sub-AS36F4
SRC=$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTnnbyol3dFrozenEp0064Dice7350_probmap_fixed.ozx
DST=$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTnnbyol3dFrozenEp0064Dice7350Thr0p5_mask.ome.zarr

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "src: $SRC"
echo "dst: $DST"
echo "Start: $(date)"

# Clean any prior attempt
rm -rf "$DST"

pixi run python scripts/binarize_ozx_to_omezarr.py \
    --input "$SRC" \
    --output "$DST" \
    --threshold 0.5 \
    --orientation IAR \
    --build-pyramid 2

EXIT=$?
echo "End: $(date)  exit=$EXIT"
echo "=== output ==="
ls -la "$DST/" 2>/dev/null
du -sh "$DST" 2>/dev/null
exit $EXIT
