#!/bin/bash
# Submit random selection baseline training jobs for all datasets.
# Uses the same sbatch/train_informed_selection.sh as learned selection runs.
# Usage: sh sh/submit_random_baselines.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Submitting Random Selection Baseline Jobs ==="

# AUTSL
echo "--- AUTSL ---"
for k in 270 100 48 24 10; do
    for draw in 0 1 2; do
        sbatch --job-name="autsl_rand_${k}_d${draw}" \
            sbatch/train_informed_selection.sh \
            "configs/random_selection/random_${k}_draw${draw}.yaml"
    done
done

# ASL Citizen
echo "--- ASL Citizen ---"
for k in 270 100 48 24 10; do
    for draw in 0 1 2; do
        sbatch --job-name="asl_rand_${k}_d${draw}" \
            sbatch/train_informed_selection.sh \
            "configs/asl_citizen/random_selection/random_${k}_draw${draw}.yaml"
    done
done

# GSL
echo "--- GSL ---"
for k in 270 100 48 24 10; do
    for draw in 0 1 2; do
        sbatch --job-name="gsl_rand_${k}_d${draw}" \
            sbatch/train_informed_selection.sh \
            "configs/gsl/random_selection/random_${k}_draw${draw}.yaml"
    done
done

# Multilingual
echo "--- Multilingual ---"
for k in 270 100 48 24 10; do
    for draw in 0 1 2; do
        sbatch --job-name="mul_rand_${k}_d${draw}" \
            sbatch/train_informed_selection.sh \
            "configs/multilingual/random_selection/random_${k}_draw${draw}.yaml"
    done
done

echo "=== Submitted 60 random baseline jobs (4 datasets x 5 K-values x 3 draws) ==="