#!/bin/bash
# Submit all Top-K Independent optimization + training jobs
# These can all run in parallel since they're independent
#
# Usage: bash sbatch/submit_all_topk.sh [optimize|train]
#   optimize: submit hyperparameter search jobs (default)
#   train:    submit full training jobs (run after optimization)

MODE=${1:-"optimize"}

if [ "$MODE" = "optimize" ]; then
    echo "Submitting hyperparameter optimization jobs for all Top-K models..."
    for K in 270 100 48 24 10; do
        echo "  Submitting topk_${K} optimization..."
        sbatch --job-name="opt_topk_${K}" sbatch/optimize_informed_selection.sh "topk_${K}"
    done
elif [ "$MODE" = "train" ]; then
    echo "Submitting full training jobs for all Top-K models..."
    for K in 270 100 48 24 10; do
        echo "  Submitting topk_${K} training..."
        sbatch --job-name="train_topk_${K}" sbatch/train_informed_selection.sh "topk_${K}"
    done
else
    echo "Usage: bash sbatch/submit_all_topk.sh [optimize|train]"
    exit 1
fi

echo "Done. Check job status with: squeue -u \$USER"
