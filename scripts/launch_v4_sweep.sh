#!/bin/bash
# Idempotent launcher for the v4 fine-tuning sweep.
#
# Sweep matrix (40 runs):
#   encoder   : simclr (128), nnbyol3d (128)
#   dropout   : 0.1, 0.2          (true per-stage decoder Dropout3d)
#   encoder   : frozen, unfrozen  (unfrozen trains at 0.1x the decoder LR)
#   seed      : 0..4              (5-member deep ensemble; shared val split)
#   => 2 x 2 x 2 x 5 = 40 members
#
# Each member runs on 1x L40S (this FT workload is small; no DDP needed).
# At most MAX_CONCURRENT members run at once so we don't hog the cluster.
#
# This launcher just submits whatever still needs doing and exits:
#   - a member with $SAVE_DIR/DONE is finished        -> skipped
#   - a member already in squeue                       -> skipped (running/queued)
#   - otherwise it's submitted, up to the concurrency cap
# Re-run it any time (or from cron / a watch loop) until everything is DONE.
# A killed/timed-out member auto-resumes from last.ckpt on its next submission.
#
# Usage:
#   scripts/launch_v4_sweep.sh            # submit pending up to the cap
#   scripts/launch_v4_sweep.sh --status   # just print state, submit nothing
#   scripts/launch_v4_sweep.sh --test     # submit ONE 2-epoch run that hits W&B
#   MAX_CONCURRENT=4 scripts/launch_v4_sweep.sh
#
# --test is a wandb/plumbing sanity check: it submits a single real (NOT
# fast-dev-run) 2-epoch job through SLURM so wandb.init() runs from the compute
# node and a run appears in the project. It uses a throwaway run/dir
# (ftv4_TEST / $SWEEP_ROOT/_test) that is wiped each time and never counts
# toward the 40-member sweep.

set -uo pipefail
cd /nfs/khan/trainees/apooladi/abeta/Lumivox

MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
MANIFEST="${MANIFEST:-manifests/abeta_ft_v4_A.json}"
WANDB_PROJECT="${WANDB_PROJECT:-lumivox-fine-tuning}"
SWEEP_ROOT="${SWEEP_ROOT:-checkpoints/ft_v4_sweep}"
EPOCHS="${EPOCHS:-150}"
TRAIN_REPEATS="${TRAIN_REPEATS:-4}"
SPLIT_SEED="${SPLIT_SEED:-42}"
JOB_PREFIX="ftv4"            # squeue job-name prefix used for cap counting

STATUS_ONLY=0
TEST_MODE=0
case "${1:-}" in
  --status) STATUS_ONLY=1 ;;
  --test)   TEST_MODE=1 ;;
  "")       ;;
  *) echo "unknown arg: $1  (use --status or --test)"; exit 1 ;;
esac

declare -A CKPT=(
  [simclr]="checkpoints/simclr_abeta_50k/pretrain_epoch0045.pt"
  [nnbyol3d]="checkpoints/nnbyol3d_abeta_50k_v2/pretrain_epoch0040.pt"
)
declare -A DROPVAL=( [d10]="0.1" [d20]="0.2" )

# --- preflight ---
if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: manifest not found: $MANIFEST  (run scripts/build_v4_ft_manifest.py)"; exit 1
fi
for enc in "${!CKPT[@]}"; do
  [ -f "${CKPT[$enc]}" ] || { echo "ERROR: missing checkpoint ${CKPT[$enc]}"; exit 1; }
done
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
  echo "ERROR: no W&B credentials found. Run 'pixi run wandb login' once (or export"
  echo "       WANDB_API_KEY), or set WANDB_MODE=offline to log locally and sync later."
  [ "$STATUS_ONLY" -eq 0 ] && exit 1
fi

# --- --test: single short real run to verify wandb plumbing end-to-end ---
if [ "$TEST_MODE" -eq 1 ]; then
  test_name="${JOB_PREFIX}_TEST"
  test_dir="${SWEEP_ROOT}/_test"
  rm -rf "$test_dir"          # fresh run each time (no stale DONE / last.ckpt)
  EXPORTS="ALL"
  EXPORTS+=",RUN_NAME=${test_name},CKPT=${CKPT[simclr]},MANIFEST=${MANIFEST}"
  EXPORTS+=",DROPOUT=0.1,FREEZE_FLAG=--freeze-encoder,ENC_LR_FACTOR=0.1"
  EXPORTS+=",SEED=0,SPLIT_SEED=${SPLIT_SEED},SAVE_DIR=${test_dir}"
  EXPORTS+=",WANDB_PROJECT=${WANDB_PROJECT},EPOCHS=2,TRAIN_REPEATS=1"
  echo "=== --test: submitting one 2-epoch wandb sanity run ==="
  if sbatch --job-name="$test_name" --export="$EXPORTS" scripts/finetune_v4_job.sbatch; then
    echo "  submitted ${test_name} (frozen simclr, 2 epochs, train-repeats=1)"
    echo "  watch:  squeue -u $USER -n ${test_name}"
    echo "  log:    logs/${test_name}_<jobid>.out"
    echo "  wandb:  project '${WANDB_PROJECT}', run '${test_name}'"
  else
    echo "  FAILED to submit ${test_name}"; exit 1
  fi
  exit 0
fi

# Names already in the queue (running or pending). Use bare '%j' (full name, NO
# padding) and strip any trailing whitespace — '%200j' left-pads to 200 chars,
# which breaks the exact-match skip below and lets duplicates slip through.
QUEUED="$(squeue -u "$USER" -h -o '%j' 2>/dev/null | sed 's/[[:space:]]*$//')"
n_active="$(printf '%s\n' "$QUEUED" | grep -c "^${JOB_PREFIX}_" || true)"

n_done=0; n_run=0; n_sub=0; n_pend=0; total=0
echo "=== v4 sweep launcher ===  cap=${MAX_CONCURRENT}  active=${n_active}  manifest=${MANIFEST}"

for enc in simclr nnbyol3d; do
  for dk in d10 d20; do
    for fk in frozen unfrozen; do
      for seed in 0 1 2 3 4; do
        total=$((total+1))
        run_name="${JOB_PREFIX}_${enc}_${dk}_${fk}_s${seed}"
        save_dir="${SWEEP_ROOT}/${run_name}"

        if [ -f "${save_dir}/DONE" ]; then
          n_done=$((n_done+1)); continue
        fi
        if printf '%s\n' "$QUEUED" | grep -qxF "$run_name"; then
          n_run=$((n_run+1)); continue
        fi
        if [ "$STATUS_ONLY" -eq 1 ]; then
          n_pend=$((n_pend+1)); continue
        fi
        if [ "$n_active" -ge "$MAX_CONCURRENT" ]; then
          n_pend=$((n_pend+1)); continue
        fi

        if [ "$fk" = "frozen" ]; then FREEZE_FLAG="--freeze-encoder"; else FREEZE_FLAG=""; fi

        EXPORTS="ALL"
        EXPORTS+=",RUN_NAME=${run_name},CKPT=${CKPT[$enc]},MANIFEST=${MANIFEST}"
        EXPORTS+=",DROPOUT=${DROPVAL[$dk]},FREEZE_FLAG=${FREEZE_FLAG},ENC_LR_FACTOR=0.1"
        EXPORTS+=",SEED=${seed},SPLIT_SEED=${SPLIT_SEED},SAVE_DIR=${save_dir}"
        EXPORTS+=",WANDB_PROJECT=${WANDB_PROJECT},EPOCHS=${EPOCHS},TRAIN_REPEATS=${TRAIN_REPEATS}"

        if sbatch --job-name="$run_name" --export="$EXPORTS" \
                  scripts/finetune_v4_job.sbatch >/dev/null; then
          echo "  submitted: $run_name"; n_sub=$((n_sub+1)); n_active=$((n_active+1))
        else
          echo "  FAILED to submit: $run_name"
        fi
      done
    done
  done
done

echo "-------------------------------------------------------"
echo "  total=${total}  done=${n_done}  running/queued=${n_run}  submitted_now=${n_sub}  still_pending=${n_pend}"
remaining=$((total - n_done))
if [ "$remaining" -eq 0 ]; then
  echo "  ALL ${total} MEMBERS DONE."
else
  echo "  ${remaining} not yet done. Re-run this script to top up to the cap."
fi
