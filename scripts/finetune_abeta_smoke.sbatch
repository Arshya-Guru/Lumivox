#!/bin/bash
#SBATCH --job-name=ft_abeta_smoke
#SBATCH --partition=hx,vx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ft_abeta_smoke_%j.out
#SBATCH --error=logs/ft_abeta_smoke_%j.err

set -uo pipefail

cd /nfs/khan/trainees/apooladi/abeta/Lumivox
mkdir -p logs

GPU_LOG="logs/ft_abeta_smoke_${SLURM_JOB_ID}_gpu.csv"

echo "===================================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU(s):"
nvidia-smi -L 2>/dev/null || echo "  nvidia-smi unavailable"
echo "Start: $(date)"
echo "===================================================="

# Background nvidia-smi sampler @ 5s
(
    echo "ts,gpu_util_pct,mem_used_mib,mem_total_mib"
    while true; do
        ts=$(date +%s)
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                   --format=csv,noheader,nounits 2>/dev/null \
            | awk -v t="$ts" -F', *' '{print t","$1","$2","$3}'
        sleep 5
    done
) > "$GPU_LOG" &
SAMPLER_PID=$!
trap "kill $SAMPLER_PID 2>/dev/null" EXIT

bash scripts/finetune_abeta_smoke.sh
RC=$?

echo "===================================================="
echo "Smoke script exit: $RC"
echo "End: $(date)"
echo "===================================================="

# Quick VRAM peak summary
if [ -s "$GPU_LOG" ]; then
    echo "VRAM peak (MiB used / MiB total):"
    awk -F',' 'NR>1 {if ($3 > peak) {peak=$3; tot=$4}} END {printf "  %s / %s MiB\n", peak, tot}' "$GPU_LOG"
fi

exit $RC
