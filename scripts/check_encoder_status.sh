#!/bin/bash
# Check the status of all encoder comparison jobs and identify what needs
# to be submitted fresh, resumed (timed out), or is already done.
#
# Job states:
#   NOT_STARTED — save dir doesn't exist, needs TRAIN
#   RUNNING     — currently in the SLURM queue
#   TIMED_OUT   — last.ckpt exists but training log shows time limit hit
#   DONE        — training completed (early stop or max_steps reached)
#
# Usage:
#   bash scripts/check_encoder_status.sh            # show all
#   bash scripts/check_encoder_status.sh BATCH 1   # show only batch 1
#   bash scripts/check_encoder_status.sh RESUME     # print resume commands for timed-out jobs
#   bash scripts/check_encoder_status.sh RESUME SUBMIT  # actually resubmit timed-out jobs

ACTION=${1:-show}
BATCH_FILTER=${2:-}
SLURM_OUT_DIR="/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs"
MODEL_ROOT="/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/encoder_comparison"
CONFIG_ROOT="configs/encoder_comparison"
SBATCH_SCRIPT="sbatch/train_encoder_comparison.sh"
EXCLUDE="--exclude=dw-2-4,cs-1-2"

LANGUAGES="autsl asl_citizen gsl multilingual"
ENCODERS="gru stgcn spoter"

# Get currently running job names
RUNNING_JOBS=$(squeue -u "$USER" -o "%j" --noheader 2>/dev/null)

not_started=0
running=0
timed_out=0
done_count=0

check_job() {
    local config=$1
    local jobname=$2
    local save_dir=$3
    local batch=$4

    # Filter by batch if requested (ignore SUBMIT — that's the submit flag, not a batch number)
    if [ -n "$BATCH_FILTER" ] && [ "$BATCH_FILTER" != "SUBMIT" ] && [ "$BATCH_FILTER" != "$batch" ]; then
        return
    fi

    local ckpt_dir="$save_dir/checkpoints"
    local last_ckpt="$ckpt_dir/last.ckpt"

    # Check if currently running
    if echo "$RUNNING_JOBS" | grep -qx "$jobname"; then
        running=$((running + 1))
        if [ "$ACTION" = "show" ]; then
            echo "RUNNING     $jobname"
        fi
        return
    fi

    # Not running — check save dir
    if [ ! -d "$save_dir" ]; then
        not_started=$((not_started + 1))
        if [ "$ACTION" = "show" ]; then
            echo "NOT_STARTED $jobname"
        fi
        return
    fi

    # Restore last.ckpt if a test job left a .bak behind (cancelled mid-move).
    # Check both the new location (save_dir/last.ckpt.bak) and the old
    # in-checkpoints-dir location for backward compatibility.
    [ ! -f "$last_ckpt" ] && [ -f "${save_dir}/last.ckpt.bak" ] && mv "${save_dir}/last.ckpt.bak" "$last_ckpt"
    [ ! -f "$last_ckpt" ] && [ -f "${last_ckpt}.bak" ] && mv "${last_ckpt}.bak" "$last_ckpt"

    # Save dir exists — check if timed out or done
    if [ -f "$last_ckpt" ]; then
        # Check slurm log for time limit
        local timed_out_flag=false
        local log=$(ls -t "$SLURM_OUT_DIR"/*_${jobname}.out 2>/dev/null | head -1)
        if [ -n "$log" ] && grep -q "DUE TO TIME LIMIT\|CANCELLED\|TIMEOUT" "$log" 2>/dev/null; then
            timed_out_flag=true
        elif [ -f "$ckpt_dir/last.ckpt" ] && ! ls "$ckpt_dir"/*.ckpt 2>/dev/null | grep -qv "last.ckpt"; then
            # Only last.ckpt exists, no top-k checkpoints yet — likely very early timeout
            timed_out_flag=true
        fi

        if [ "$timed_out_flag" = "true" ]; then
            timed_out=$((timed_out + 1))
            if [ "$ACTION" = "show" ]; then
                echo "TIMED_OUT   $jobname"
            elif [ "$ACTION" = "RESUME" ]; then
                if [ "${BATCH_FILTER}" = "SUBMIT" ] || [ "${2}" = "SUBMIT" ]; then
                    sbatch --job-name="$jobname" $EXCLUDE "$SBATCH_SCRIPT" "$config" RESUME
                else
                    echo "sbatch --job-name=$jobname $EXCLUDE $SBATCH_SCRIPT $config RESUME"
                fi
            fi
            return
        fi
    fi

    # Check for completion marker in logs
    local log=$(ls -t "$SLURM_OUT_DIR"/*_${jobname}.out 2>/dev/null | head -1)
    if [ -n "$log" ] && grep -q "TRAINING ENDED" "$log" 2>/dev/null; then
        done_count=$((done_count + 1))
        if [ "$ACTION" = "show" ]; then
            echo "DONE        $jobname"
        fi
    else
        # Save dir exists, not timed out, not done — ambiguous, treat as not started
        not_started=$((not_started + 1))
        if [ "$ACTION" = "show" ]; then
            echo "NOT_STARTED $jobname  (save dir exists but state unclear)"
        fi
    fi
}

# ── Batch 1: Full skeleton ───────────────────────────────────────────────────
for enc in $ENCODERS; do
    for lang in $LANGUAGES; do
        jobname="${enc}_${lang}_k543"
        save_dir="$MODEL_ROOT/$lang/$enc/k543"
        check_job "$CONFIG_ROOT/$lang/$enc/k543.yaml" "$jobname" "$save_dir" 1
    done
done

# ── Batch 2: Iterative ───────────────────────────────────────────────────────
for k in 10 24 48 100 270; do
    for enc in $ENCODERS; do
        for lang in $LANGUAGES; do
            config="$CONFIG_ROOT/$lang/$enc/iter_${k}.yaml"
            [ -f "$config" ] || continue
            jobname="${enc}_${lang}_iter${k}"
            save_dir="$MODEL_ROOT/$lang/$enc/iter_${k}"
            check_job "$config" "$jobname" "$save_dir" 2
        done
    done
done

# ── Batch 3: Top-K ──────────────────────────────────────────────────────────
for k in 10 24 48 100; do
    for enc in $ENCODERS; do
        for lang in $LANGUAGES; do
            jobname="${enc}_${lang}_k${k}"
            save_dir="$MODEL_ROOT/$lang/$enc/k${k}"
            check_job "$CONFIG_ROOT/$lang/$enc/k${k}.yaml" "$jobname" "$save_dir" 3
        done
    done
done

# ── Batch 4: Random baselines ────────────────────────────────────────────────
for k in 10 24 48 100 270; do
    for draw in 0 1 2; do
        for enc in $ENCODERS; do
            for lang in $LANGUAGES; do
                jobname="${enc}_${lang}_rand${k}_d${draw}"
                save_dir="$MODEL_ROOT/$lang/$enc/random_${k}_draw${draw}"
                check_job "$CONFIG_ROOT/$lang/$enc/random/random_${k}_draw${draw}.yaml" "$jobname" "$save_dir" 4
            done
        done
    done
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
echo "  NOT_STARTED: $not_started"
echo "  RUNNING:     $running"
echo "  TIMED_OUT:   $timed_out"
echo "  DONE:        $done_count"
echo "  Total:       $((not_started + running + timed_out + done_count))"

if [ "$ACTION" = "show" ] && [ "$timed_out" -gt 0 ]; then
    echo ""
    echo "To resubmit timed-out jobs (dry run):  bash scripts/check_encoder_status.sh RESUME"
    echo "To resubmit timed-out jobs (submit):   bash scripts/check_encoder_status.sh RESUME SUBMIT"
fi
