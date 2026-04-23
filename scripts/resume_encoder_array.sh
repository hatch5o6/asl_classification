#!/bin/bash
# Detect timed-out tasks in an array batch and resubmit only those indices.
#
# A task is considered timed-out if:
#   - Its model save dir exists
#   - last.ckpt is present (training started, checkpoint saved)
#   - The job is NOT currently in the SLURM queue
#
# A task is considered crashed (no checkpoint) if:
#   - Its model save dir exists
#   - No last.ckpt and no best checkpoint
#   - The job is NOT currently in the SLURM queue
#   -> These are resubmitted with TRAIN instead of RESUME
#
# Usage (legacy single-list batches):
#   bash scripts/resume_encoder_array.sh 3           # dry run batch 3 (old single list)
#   bash scripts/resume_encoder_array.sh 4 SUBMIT    # submit resumes for batch 4 (old single list)
#
# Usage (new split layout — cs = autsl+gsl @ 24h, matrix = asl_citizen+multilingual @ 3d):
#   bash scripts/resume_encoder_array.sh 3_cs
#   bash scripts/resume_encoder_array.sh 3_matrix
#   bash scripts/resume_encoder_array.sh 4_d0_cs
#   bash scripts/resume_encoder_array.sh 4_d0_matrix    (same for d1, d2)

BATCH=${1:?"Usage: bash scripts/resume_encoder_array.sh <3|4|3_cs|3_matrix|4_d{0,1,2}_{cs,matrix}> [SUBMIT]"}
MODE=${2:-DRY}

CONFIG_ROOT="configs/encoder_comparison"
MODEL_ROOT="/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/encoder_comparison"
EXCLUDE="--exclude=dw-2-4,cs-1-2"

# Default sbatch script; matrix variants override below.
SBATCH_ARRAY="sbatch/train_encoder_array.sh"

case "$BATCH" in
    3)           CONFIG_LIST="$CONFIG_ROOT/batch3_configs.txt"; THROTTLE=16 ;;
    4)           CONFIG_LIST="$CONFIG_ROOT/batch4_configs.txt"; THROTTLE=20 ;;
    3_cs)        CONFIG_LIST="$CONFIG_ROOT/batch3_cs_configs.txt"; THROTTLE=16 ;;
    3_matrix)    CONFIG_LIST="$CONFIG_ROOT/batch3_matrix_configs.txt"; THROTTLE=8
                 SBATCH_ARRAY="sbatch/train_encoder_array_matrix.sh" ;;
    4_d0_cs)     CONFIG_LIST="$CONFIG_ROOT/batch4_d0_cs_configs.txt"; THROTTLE=16 ;;
    4_d1_cs)     CONFIG_LIST="$CONFIG_ROOT/batch4_d1_cs_configs.txt"; THROTTLE=16 ;;
    4_d2_cs)     CONFIG_LIST="$CONFIG_ROOT/batch4_d2_cs_configs.txt"; THROTTLE=16 ;;
    4_d0_matrix) CONFIG_LIST="$CONFIG_ROOT/batch4_d0_matrix_configs.txt"; THROTTLE=8
                 SBATCH_ARRAY="sbatch/train_encoder_array_matrix.sh" ;;
    4_d1_matrix) CONFIG_LIST="$CONFIG_ROOT/batch4_d1_matrix_configs.txt"; THROTTLE=8
                 SBATCH_ARRAY="sbatch/train_encoder_array_matrix.sh" ;;
    4_d2_matrix) CONFIG_LIST="$CONFIG_ROOT/batch4_d2_matrix_configs.txt"; THROTTLE=8
                 SBATCH_ARRAY="sbatch/train_encoder_array_matrix.sh" ;;
    *) echo "Unknown batch: $BATCH. Use: 3, 4, 3_cs, 3_matrix, 4_d{0,1,2}_{cs,matrix}"; exit 1 ;;
esac

if [ ! -f "$CONFIG_LIST" ]; then
    echo "Error: Config list not found: $CONFIG_LIST"
    exit 1
fi

# Get all currently running/pending job names
RUNNING_JOBS=$(squeue -u "$USER" -o "%j" --noheader 2>/dev/null)

resume_indices=()
train_indices=()

idx=0
while IFS=$'\t' read -r config jobname; do
    [ -z "$config" ] && { idx=$((idx+1)); continue; }

    # Derive save dir from config path:
    # configs/encoder_comparison/{lang}/{enc}/{subset}.yaml
    # -> models/encoder_comparison/{lang}/{enc}/{subset}
    rel=${config#configs/encoder_comparison/}   # lang/enc/subset.yaml
    subset_yaml=$(basename "$rel")
    subdir="${subset_yaml%.yaml}"
    lang_enc=$(dirname "$rel")                  # lang/enc  (or lang/enc/random)
    save_dir="$MODEL_ROOT/$lang_enc/$subdir"

    ckpt_dir="$save_dir/checkpoints"
    last_ckpt="$ckpt_dir/last.ckpt"
    in_queue=$(echo "$RUNNING_JOBS" | grep -cx "$jobname" || true)

    # Skip jobs currently in the queue
    if [ "$in_queue" -gt 0 ]; then
        idx=$((idx+1))
        continue
    fi

    # Skip jobs with no checkpoint dir — not started yet
    if [ ! -d "$ckpt_dir" ]; then
        idx=$((idx+1))
        continue
    fi

    # Skip jobs that have already been tested (fully done)
    has_test=$(ls "$save_dir/predictions/"*.metrics.json 2>/dev/null | head -1)
    if [ -n "$has_test" ]; then
        idx=$((idx+1))
        continue
    fi

    # Skip jobs whose SLURM log shows clean completion ("TRAINING ENDED")
    # Search most recent log for this jobname (array jobs use jobname via scontrol)
    SLURM_OUT_DIR="/home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs"
    recent_log=$(ls -t "$SLURM_OUT_DIR"/*_${jobname}.out 2>/dev/null | head -1)
    if [ -n "$recent_log" ] && grep -q "TRAINING ENDED\|TRAINING COMPLETED" "$recent_log" 2>/dev/null; then
        # Training completed cleanly — just not tested yet, no resubmission needed
        idx=$((idx+1))
        continue
    fi

    # If last.ckpt.bak exists and last.ckpt does NOT, a test job is actively
    # running — skip to avoid interfering. Check both new location (save_dir/)
    # and old location (checkpoints/) for backward compatibility.
    if [ ! -f "$last_ckpt" ] && { [ -f "${save_dir}/last.ckpt.bak" ] || [ -f "${last_ckpt}.bak" ]; }; then
        # If the test has finished (no running enc_test job), restore; otherwise skip
        if squeue -u "$USER" -n enc_test --noheader 2>/dev/null | grep -q .; then
            echo "TESTING    [$idx]: $jobname (skipped — test in progress)"
            idx=$((idx+1))
            continue
        fi
        [ -f "${save_dir}/last.ckpt.bak" ] && mv "${save_dir}/last.ckpt.bak" "$last_ckpt"
        [ -f "${last_ckpt}.bak" ] && mv "${last_ckpt}.bak" "$last_ckpt"
    fi

    if [ -f "$last_ckpt" ]; then
        resume_indices+=("$idx")
        echo "TIMED_OUT [$idx]: $jobname"
    else
        train_indices+=("$idx")
        echo "CRASHED   [$idx]: $jobname"
    fi

    idx=$((idx+1))
done < "$CONFIG_LIST"

echo ""
echo "Timed-out (RESUME): ${#resume_indices[@]}"
echo "Crashed   (TRAIN):  ${#train_indices[@]}"

submit_array() {
    local indices_arr=("${!1}")
    local train_mode=$2
    if [ ${#indices_arr[@]} -eq 0 ]; then return; fi

    # Build comma-separated index list, e.g. "3,7,12"
    local idx_str=$(IFS=,; echo "${indices_arr[*]}")
    local array_spec="${idx_str}%${THROTTLE}"

    if [ "$MODE" = "SUBMIT" ]; then
        sbatch --array="$array_spec" $EXCLUDE "$SBATCH_ARRAY" "$CONFIG_LIST" "$train_mode"
        echo "Submitted $train_mode array: $array_spec"
    else
        echo "sbatch --array=$array_spec $EXCLUDE $SBATCH_ARRAY $CONFIG_LIST $train_mode"
    fi
}

echo ""
if [ ${#resume_indices[@]} -eq 0 ] && [ ${#train_indices[@]} -eq 0 ]; then
    echo "Nothing to resubmit."
    exit 0
fi

submit_array resume_indices[@] RESUME
submit_array train_indices[@] TRAIN

if [ "$MODE" != "SUBMIT" ]; then
    echo ""
    echo "(Dry run — pass SUBMIT to actually submit)"
fi
