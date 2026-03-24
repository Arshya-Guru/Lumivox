#!/usr/bin/env bash
# monitor_training.sh — Log GPU/CPU/memory stats + training progress over time
#
# Usage:
#   bash scripts/monitor_training.sh > simclr_monitor.csv &
#   bash scripts/monitor_training.sh --interval 10 > nnbyol3d_monitor.csv &
#
# Outputs CSV with columns:
#   timestamp, gpu0_util%, gpu0_mem_MB, gpu0_power_W,
#   gpu1_util%, gpu1_mem_MB, gpu1_power_W,
#   cpu_load_1m, cpu_load_5m, cpu_load_15m,
#   mem_used_GB, mem_available_GB, mem_buff_cache_GB,
#   train_epoch, train_step, train_loss, train_lr, epoch_progress%

INTERVAL=${1:-5}  # seconds between samples, default 5
if [[ "$1" == "--interval" ]]; then
    INTERVAL=$2
fi

# Find the lightning CSV log (search all checkpoint dirs)
find_training_log() {
    local latest=""
    # Check save_dir-specific locations first, then fallback
    for base in checkpoints/*/lightning_logs checkpoints/*/lightning_logs/lightning_logs lightning_logs; do
        for d in ${base}/version_*/; do
            if [[ -d "$d" && -f "${d}metrics.csv" ]]; then
                latest="${d}metrics.csv"
            fi
        done
    done
    echo "$latest"
}

# Print header
echo "timestamp,gpu0_util_pct,gpu0_mem_mb,gpu0_power_w,gpu1_util_pct,gpu1_mem_mb,gpu1_power_w,cpu_load_1m,cpu_load_5m,cpu_load_15m,mem_used_gb,mem_available_gb,mem_buff_cache_gb,train_epoch,train_step,train_loss,train_lr,epoch_progress_pct"

while true; do
    ts=$(date +%Y-%m-%d\ %H:%M:%S)

    # GPU stats
    gpu_data=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null)
    gpu0_util=""
    gpu0_mem=""
    gpu0_power=""
    gpu1_util=""
    gpu1_mem=""
    gpu1_power=""

    while IFS=', ' read -r idx util mem_used mem_total power; do
        if [[ "$idx" == "0" ]]; then
            gpu0_util=$util
            gpu0_mem=$mem_used
            gpu0_power=$power
        elif [[ "$idx" == "1" ]]; then
            gpu1_util=$util
            gpu1_mem=$mem_used
            gpu1_power=$power
        fi
    done <<< "$gpu_data"

    # CPU load
    read -r load1 load5 load15 _ < /proc/loadavg

    # Memory (parse free output)
    mem_line=$(free -g | awk '/^Mem:/ {print $3, $7, $6}')
    mem_used=$(echo "$mem_line" | awk '{print $1}')
    mem_avail=$(echo "$mem_line" | awk '{print $2}')
    mem_cache=$(echo "$mem_line" | awk '{print $3}')

    # Training progress from Lightning CSV log
    train_epoch=""
    train_step=""
    train_loss=""
    train_lr=""
    epoch_pct=""

    log_file=$(find_training_log)
    if [[ -n "$log_file" && -f "$log_file" ]]; then
        # Get last line with actual data
        last_line=$(tail -1 "$log_file" 2>/dev/null)
        if [[ -n "$last_line" && "$last_line" != *"epoch"* ]]; then
            # Lightning CSV format varies, try to extract key fields
            # Typically: epoch,step,train/loss,train/lr,...
            train_epoch=$(echo "$last_line" | awk -F',' '{print $1}')
            train_step=$(echo "$last_line" | awk -F',' '{print $2}')
            # Find loss and lr columns from header
            header=$(head -1 "$log_file" 2>/dev/null)
            loss_col=$(echo "$header" | tr ',' '\n' | grep -n "train/loss" | head -1 | cut -d: -f1)
            lr_col=$(echo "$header" | tr ',' '\n' | grep -n "train/lr" | head -1 | cut -d: -f1)
            if [[ -n "$loss_col" ]]; then
                train_loss=$(echo "$last_line" | cut -d',' -f"$loss_col")
            fi
            if [[ -n "$lr_col" ]]; then
                train_lr=$(echo "$last_line" | cut -d',' -f"$lr_col")
            fi
        fi
    fi

    # Also try to scrape from the progress bar output if available
    # Check for the Lightning progress bar in stderr by reading the most recent log line
    # This is a fallback — the CSV log is more reliable

    # Estimate epoch progress from step count (3125 steps per epoch for 50k/8/2)
    if [[ -n "$train_step" && "$train_step" =~ ^[0-9]+$ ]]; then
        steps_per_epoch=3125
        epoch_pct=$(awk "BEGIN {printf \"%.1f\", ($train_step % $steps_per_epoch) / $steps_per_epoch * 100}")
    fi

    # Output CSV row
    echo "${ts},${gpu0_util:-0},${gpu0_mem:-0},${gpu0_power:-0},${gpu1_util:-0},${gpu1_mem:-0},${gpu1_power:-0},${load1},${load5},${load15},${mem_used:-0},${mem_avail:-0},${mem_cache:-0},${train_epoch:-},${train_step:-},${train_loss:-},${train_lr:-},${epoch_pct:-}"

    sleep "$INTERVAL"
done
