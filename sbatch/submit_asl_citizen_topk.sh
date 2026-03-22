#!/bin/bash
# Submit all ASL Citizen Top-K training jobs.
# Run AFTER extracting joint indices from the trained asl_citizen/s model:
#
#   sbatch sbatch/extract_joint_indices.sh \
#       asl_citizen/s configs/asl_citizen/s.yaml \
#       "270 100 48 24 10" data/asl_citizen/informed_selection/topk top
#
# Then run this script:
#   bash sbatch/submit_asl_citizen_topk.sh

echo "Submitting ASL Citizen Top-K training jobs..."
for K in 270 100 48 24 10; do
    echo "  Submitting topk_${K}..."
    sbatch --job-name="asl_citizen_topk_${K}" sbatch/train_informed_selection.sh \
        configs/asl_citizen/informed_selection/topk_${K}.yaml
done

echo "Done. Check job status with: squeue -u \$USER"