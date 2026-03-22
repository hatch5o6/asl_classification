#!/bin/bash
# Submit all generalization test experiments (including TTA on existing checkpoints).
# Run this AFTER training jobs complete.
# Each job gets a unique name so clean_slurm_outputs.py won't delete any.
# Usage: sh sh/submit_generalization_test.sh

set -e
cd "$(dirname "$0")/.."

echo "=== Submitting Generalization Test Jobs ==="

# ASL Citizen experiments
echo "--- ASL Citizen ---"
sbatch --job-name=asl_gen_aug_test   sbatch/test_generalization.sh configs/asl_citizen/generalization/augmentation.yaml
sbatch --job-name=asl_gen_sbal_test  sbatch/test_generalization.sh configs/asl_citizen/generalization/signer_balanced.yaml
sbatch --job-name=asl_gen_reg_test   sbatch/test_generalization.sh configs/asl_citizen/generalization/regularization.yaml
sbatch --job-name=asl_gen_all_test   sbatch/test_generalization.sh configs/asl_citizen/generalization/all_combined.yaml
# TTA on existing topk_48 checkpoint (no retraining needed)
sbatch --job-name=asl_gen_tta_test   sbatch/test_generalization.sh configs/asl_citizen/generalization/tta.yaml

# Multilingual experiments
echo "--- Multilingual ---"
sbatch --job-name=mul_gen_aug_test   sbatch/test_generalization.sh configs/multilingual/generalization/augmentation.yaml
sbatch --job-name=mul_gen_sbal_test  sbatch/test_generalization.sh configs/multilingual/generalization/signer_balanced.yaml
sbatch --job-name=mul_gen_reg_test   sbatch/test_generalization.sh configs/multilingual/generalization/regularization.yaml
sbatch --job-name=mul_gen_all_test   sbatch/test_generalization.sh configs/multilingual/generalization/all_combined.yaml
# TTA on existing iterative_10 checkpoint (no retraining needed)
sbatch --job-name=mul_gen_tta_test   sbatch/test_generalization.sh configs/multilingual/generalization/tta.yaml

echo "=== All test jobs submitted ==="