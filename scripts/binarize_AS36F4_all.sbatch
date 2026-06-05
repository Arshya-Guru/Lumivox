#!/bin/bash
#SBATCH --job-name=binarize_AS36F4_all
#SBATCH --partition=hx,vx
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/binarize_AS36F4_all_%j.out
#SBATCH --error=logs/binarize_AS36F4_all_%j.err

# Binarize all 3 metadata-fixed probmaps (v1 nnBYOL3D, v2 nnBYOL3D, v2 SimCLR)
# at 0.5 threshold and write each as a multi-resolution OME-Zarr directory.

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

OUT_DIR=/nfs/trident3/lightsheet/prado/mouse_app_lecanemab_batch2/derivatives/lumivox_abeta_test/sub-AS36F4

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "Start: $(date)"

run_binarize() {
    local src="$1" dst="$2"
    echo "===================================================="
    echo "src: $src"
    echo "dst: $dst"
    echo "----------------------------------------------------"
    rm -rf "$dst"
    pixi run python scripts/binarize_ozx_to_omezarr.py \
        --input "$src" \
        --output "$dst" \
        --threshold 0.5 \
        --orientation IAR \
        --build-pyramid 2
    local rc=$?
    echo "exit: $rc"
    if [ -d "$dst" ]; then
        du -sh "$dst"
        ls -la "$dst"
    fi
    return $rc
}

# --- v1 nnBYOL3D ---
run_binarize \
    "$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTnnbyol3dFrozenEp0064Dice7350_probmap_fixed.ozx" \
    "$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTnnbyol3dFrozenEp0064Dice7350Thr0p5_mask.ome.zarr"

# --- v2 nnBYOL3D (stride 64 Gaussian) ---
run_binarize \
    "$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTnnbyol3dFrozenEp0064Dice7350Stride64_probmap_fixed.ozx" \
    "$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTnnbyol3dFrozenEp0064Dice7350Stride64Thr0p5_mask.ome.zarr"

# --- v2 SimCLR (stride 64 Gaussian) ---
run_binarize \
    "$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTsimclrFrozenEp0009Dice7081Stride64_probmap_fixed.ozx" \
    "$OUT_DIR/sub-AS36F4_sample-brain_acq-imaris4x_stain-Abeta_level-0_desc-lumivoxFTsimclrFrozenEp0009Dice7081Stride64Thr0p5_mask.ome.zarr"

echo "===================================================="
echo "End: $(date)"
echo "=== final .ome.zarr inventory ==="
for d in "$OUT_DIR"/*.ome.zarr; do
    [ -d "$d" ] || continue
    echo "$d"
    ls "$d/"
    echo "  size: $(du -sh "$d" | cut -f1)"
done
