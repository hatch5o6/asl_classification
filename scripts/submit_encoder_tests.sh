#!/bin/bash
# Submit test evaluation jobs for a completed training batch.
#
# Scans the batch's config list, finds jobs that are TRAINED (have a val_acc
# checkpoint but no predictions/metrics.json), writes a filtered test config
# list, and submits it as a job array on qos=matrix.
#
# Can be run repeatedly — already-tested jobs are automatically skipped.
#
# Usage:
#   bash scripts/submit_encoder_tests.sh 1          # dry run batch 1
#   bash scripts/submit_encoder_tests.sh 1 SUBMIT   # submit batch 1 tests
#   bash scripts/submit_encoder_tests.sh all SUBMIT # submit all batches

BATCH=${1:?"Usage: bash scripts/submit_encoder_tests.sh <1|2|3|4|all> [SUBMIT]"}
MODE=${2:-DRY}

SBATCH_SCRIPT="sbatch/test_encoder_array.sh"
CONFIG_ROOT="configs/encoder_comparison"
MODEL_ROOT="/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/encoder_comparison"
GPUS_PER_NODE=8
THROTTLE=2    # concurrent nodes — each node runs 8 tests in parallel

is_trained() {
    local config=$1
    local jobname=$2
    # Derive save dir from config path
    local rel=${config#configs/encoder_comparison/}
    local subset_yaml=$(basename "$rel")
    local subdir="${subset_yaml%.yaml}"
    local lang_enc=$(dirname "$rel")
    local save_dir="$MODEL_ROOT/$lang_enc/$subdir"
    local ckpt_dir="$save_dir/checkpoints"

    # Must have a best val_acc checkpoint
    local has_best=$(ls "$ckpt_dir"/*.ckpt 2>/dev/null | grep -v last | head -1)
    [ -z "$has_best" ] && return 1

    # Must NOT be currently running or pending in SLURM
    echo "$RUNNING_JOBS" | grep -qx "$jobname" && return 1

    # Must NOT already have test predictions
    local has_test=$(ls "$save_dir/predictions/"*.metrics.json 2>/dev/null | head -1)
    [ -n "$has_test" ] && return 1

    # Must NOT have a test currently running (last.ckpt.bak means a test job
    # has hidden last.ckpt and is actively running — submitting another test
    # would cause two processes to write to the same predictions/ dir)
    [ -f "${ckpt_dir}/last.ckpt.bak" ] && return 1
    [ -f "${save_dir}/last.ckpt.bak" ] && return 1

    return 0
}

submit_batch() {
    local batch_num=$1
    local source_list="$CONFIG_ROOT/batch${batch_num}_configs.txt"
    # Timestamp the test list so re-runs never overwrite a file used by a pending array
    local ts=$(date +%Y%m%d_%H%M%S)
    local test_list="$CONFIG_ROOT/batch${batch_num}_test_configs_${ts}.txt"

    if [ ! -f "$source_list" ]; then
        echo "Error: $source_list not found"
        return
    fi

    echo "=== BATCH $batch_num ==="

    # Cache currently queued jobs so is_trained can exclude them
    RUNNING_JOBS=$(squeue -u "$USER" -o "%j" --noheader 2>/dev/null)

    # Build filtered test config list — only fully trained and not in queue
    > "$test_list"
    while IFS=$'\t' read -r config jobname; do
        [ -z "$config" ] && continue
        if is_trained "$config" "$jobname"; then
            echo "$config	$jobname" >> "$test_list"
        fi
    done < "$source_list"

    local count=$(wc -l < "$test_list")
    if [ "$count" -eq 0 ]; then
        echo "  No trained jobs ready to test in batch $batch_num."
        rm -f "$test_list"
        echo ""
        return
    fi

    # Each array task fills one full node (8 GPUs). Round up for node count.
    local n_nodes=$(( (count + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    local last=$(( n_nodes - 1 ))
    echo "  Trained and ready to test: $count jobs → $n_nodes nodes (${GPUS_PER_NODE} tests/node)"

    if [ "$MODE" = "SUBMIT" ]; then
        sbatch --array="0-${last}%${THROTTLE}" "$SBATCH_SCRIPT" "$test_list"
    else
        echo "  sbatch --array=0-${last}%${THROTTLE} $SBATCH_SCRIPT $test_list"
        echo "  (first few entries in test list:)"
        head -3 "$test_list" | awk '{print "    " $2}'
        rm -f "$test_list"   # don't leave dry-run files on disk
    fi
    echo ""
}

case "$BATCH" in
    1)   submit_batch 1 ;;
    2)   submit_batch 2 ;;
    3)   submit_batch 3 ;;
    4)   submit_batch 4 ;;
    all) submit_batch 1; submit_batch 2; submit_batch 3; submit_batch 4 ;;
    *)   echo "Unknown batch: $BATCH. Use 1, 2, 3, 4, or all."; exit 1 ;;
esac

if [ "$MODE" != "SUBMIT" ]; then
    echo "(Dry run — pass SUBMIT to actually submit)"
fi