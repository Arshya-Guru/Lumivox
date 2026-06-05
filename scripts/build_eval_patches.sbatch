#!/bin/bash
#SBATCH --job-name=build_eval_patches
#SBATCH --partition=hx,vx
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_eval_patches_%j.out
#SBATCH --error=logs/build_eval_patches_%j.err

# Build the 1024^3 eval crops + OZX masks (and re-organize val patches).
# No GPU — this is zarr I/O + memory. 64G is generous headroom for the
# streamed full-res OZX slabs and the 1024^3 (2 GB uint16) crops.
#   sbatch scripts/build_eval_patches.sbatch

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs
echo "Job $SLURM_JOB_ID on $(hostname)  start $(date)"

pixi run python scripts/build_eval_patches.py

echo "exit=$?  end $(date)"
