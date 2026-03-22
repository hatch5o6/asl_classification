#!/bin/bash
# Full topk pipeline for the multilingual model.
# Run this after multilingual/s finishes training.
#
# Step 1: Extract joint probabilities + generate top-K index files
sbatch --job-name=extract_multilingual_joints sbatch/extract_joint_indices.sh \
    "multilingual/s" \
    configs/multilingual/s.yaml \
    "270 100 48 24 10" \
    data/multilingual/informed_selection/topk \
    top

# Step 2: Train all 5 topk models (run after Step 1 completes)
sbatch --job-name=multi_topk_270 sbatch/train_informed_selection.sh configs/multilingual/informed_selection/topk_270.yaml
sbatch --job-name=multi_topk_100 sbatch/train_informed_selection.sh configs/multilingual/informed_selection/topk_100.yaml
sbatch --job-name=multi_topk_48  sbatch/train_informed_selection.sh configs/multilingual/informed_selection/topk_48.yaml
sbatch --job-name=multi_topk_24  sbatch/train_informed_selection.sh configs/multilingual/informed_selection/topk_24.yaml
sbatch --job-name=multi_topk_10  sbatch/train_informed_selection.sh configs/multilingual/informed_selection/topk_10.yaml
