#!/bin/bash
# Submit encoder comparison jobs in priority order, in batches.
#
# Batches 1 & 2 use individual sbatch jobs (already partially run).
# Batches 3 & 4 use job arrays with throttling for cleaner submission.
#
# Batches:
#   BATCH 1 — Full skeleton     (12 jobs,  individual)
#   BATCH 2 — Iterative         (51 jobs,  individual)
#   BATCH 3 — Top-K             (48 jobs,  array, 16 concurrent)
#   BATCH 4 — Random baselines  (180 jobs, array, 20 concurrent)
#
# Usage:
#   bash scripts/submit_encoder_comparison.sh 1          # dry run
#   bash scripts/submit_encoder_comparison.sh 3 SUBMIT   # submit batch 3
#   bash scripts/submit_encoder_comparison.sh all SUBMIT # submit everything

BATCH=${1:?"Usage: bash scripts/submit_encoder_comparison.sh <1|2|3|4|all> [SUBMIT]"}
MODE=${2:-DRY}
SBATCH_SINGLE="sbatch/train_encoder_comparison.sh"
SBATCH_ARRAY="sbatch/train_encoder_array.sh"
CONFIG_ROOT="configs/encoder_comparison"
EXCLUDE="--exclude=dw-2-4,cs-1-2"

LANGUAGES="autsl asl_citizen gsl multilingual"
ENCODERS="gru stgcn spoter"

# Submit a single job
submit_single() {
    local config=$1
    local jobname=$2
    if [ "$MODE" = "SUBMIT" ]; then
        sbatch --job-name="$jobname" $EXCLUDE "$SBATCH_SINGLE" "$config"
    else
        echo "sbatch --job-name=$jobname $EXCLUDE $SBATCH_SINGLE $config"
    fi
}

# Submit an array job from a config list file
# Args: config_list, n_jobs, throttle, train_mode
submit_array() {
    local config_list=$1
    local n_jobs=$2
    local throttle=$3
    local train_mode=${4:-TRAIN}
    local last=$((n_jobs - 1))
    local array_spec="0-${last}%${throttle}"

    if [ "$MODE" = "SUBMIT" ]; then
        sbatch --array="$array_spec" $EXCLUDE "$SBATCH_ARRAY" "$config_list" "$train_mode"
    else
        echo "sbatch --array=$array_spec $EXCLUDE $SBATCH_ARRAY $config_list $train_mode"
    fi
}

count=0

run_batch1() {
    echo "=== BATCH 1: Full skeleton (K=543) — individual jobs ==="
    for enc in $ENCODERS; do
        for lang in $LANGUAGES; do
            submit_single "$CONFIG_ROOT/$lang/$enc/k543.yaml" "${enc}_${lang}_k543"
            count=$((count + 1))
        done
    done
    echo ""
}

run_batch2() {
    echo "=== BATCH 2: Iterative selection — individual jobs ==="
    for k in 10 24 48 100 270; do
        for enc in $ENCODERS; do
            for lang in $LANGUAGES; do
                config="$CONFIG_ROOT/$lang/$enc/iter_${k}.yaml"
                if [ -f "$config" ]; then
                    submit_single "$config" "${enc}_${lang}_iter${k}"
                    count=$((count + 1))
                fi
            done
        done
    done
    echo ""
}

run_batch3() {
    echo "=== BATCH 3: Top-K selection — array job (48 tasks, 16 concurrent) ==="
    submit_array "$CONFIG_ROOT/batch3_configs.txt" 48 16
    count=$((count + 48))
    echo ""
}

run_batch4() {
    echo "=== BATCH 4: Random baselines — array job (180 tasks, 20 concurrent) ==="
    submit_array "$CONFIG_ROOT/batch4_configs.txt" 180 20
    count=$((count + 180))
    echo ""
}

case "$BATCH" in
    1)   run_batch1 ;;
    2)   run_batch2 ;;
    3)   run_batch3 ;;
    4)   run_batch4 ;;
    all) run_batch1; run_batch2; run_batch3; run_batch4 ;;
    *)   echo "Unknown batch: $BATCH. Use 1, 2, 3, 4, or all."; exit 1 ;;
esac

echo "=== Total: $count jobs ==="
if [ "$MODE" != "SUBMIT" ]; then
    echo "(Dry run — pass SUBMIT to actually submit)"
fi
