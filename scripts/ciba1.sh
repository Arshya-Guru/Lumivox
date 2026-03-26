#!/bin/bash
#SBATCH --job-name=cache_patches
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/cache_patches_%j.out

cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs


echo ""
echo "=== Caching Iba1 patches ==="
echo "Start: $(date)"
pixi run python scripts/cache_patches.py \
    --manifest manifests/iba1_50k_ki3.json \
    --output-dir /nfs/scratch/apooladi/iba1

echo ""
echo "=== Done ==="
echo "End: $(date)"
