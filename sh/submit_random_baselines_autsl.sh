#!/bin/bash
# Submit random selection baseline training jobs for AUTSL only.
# Usage: sh sh/submit_random_baselines_autsl.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Submitting AUTSL Random Selection Baseline Jobs ==="

for k in 270 100 48 24 10; do
    for draw in 0 1 2; do
        sbatch --job-name="autsl_rand_${k}_d${draw}" \
            sbatch/train_informed_selection.sh \
            "configs/random_selection/random_${k}_draw${draw}.yaml"
    done
done

echo "=== Submitted 15 AUTSL random baseline jobs (5 K-values x 3 draws) ==="
