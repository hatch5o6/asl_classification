#!/bin/bash
# Submit all generalization training experiments.
# Each job gets a unique name so clean_slurm_outputs.py won't delete any.
# TTA is test-only (no training needed), so it's excluded here.
# Usage: sh sh/submit_generalization_train.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Submitting Generalization Training Jobs ==="

# ASL Citizen experiments (based on topk_48)
echo "--- ASL Citizen ---"
sbatch --job-name=asl_gen_aug   sbatch/train_generalization.sh configs/asl_citizen/generalization/augmentation.yaml
sbatch --job-name=asl_gen_sbal  sbatch/train_generalization.sh configs/asl_citizen/generalization/signer_balanced.yaml
sbatch --job-name=asl_gen_reg   sbatch/train_generalization.sh configs/asl_citizen/generalization/regularization.yaml
sbatch --job-name=asl_gen_all   sbatch/train_generalization.sh configs/asl_citizen/generalization/all_combined.yaml

# Multilingual experiments (based on iterative_10)
echo "--- Multilingual ---"
sbatch --job-name=mul_gen_aug   sbatch/train_generalization.sh configs/multilingual/generalization/augmentation.yaml
sbatch --job-name=mul_gen_sbal  sbatch/train_generalization.sh configs/multilingual/generalization/signer_balanced.yaml
sbatch --job-name=mul_gen_reg   sbatch/train_generalization.sh configs/multilingual/generalization/regularization.yaml
sbatch --job-name=mul_gen_all   sbatch/train_generalization.sh configs/multilingual/generalization/all_combined.yaml

echo "=== All training jobs submitted ==="