#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=viz_sweep
#SBATCH --qos=matrix

# Usage: sbatch visualize_pruning_sweep_single.sh <model_name> <config_path>
# Example: sbatch visualize_pruning_sweep_single.sh pruning_sweep_l0_0.01 configs/pruning_sweep/l0_0.01.yaml

MODEL_NAME=${1:?"Error: model_name required"}
CONFIG_PATH=${2:?"Error: config_path required"}

MODELS_DIR="/home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/models"
MODEL_DIR="$MODELS_DIR/$MODEL_NAME"
CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"
OUTPUT_DIR="$MODEL_DIR/figures"

echo "========================================"
echo "Joint Pruning Visualization"
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_PATH"
echo "========================================"

source ~/.bashrc
conda init
conda activate asl

nvidia-smi

# Find best checkpoint
BEST_CHECKPOINT=$(ls -1 "$CHECKPOINTS_DIR"/*.ckpt 2>/dev/null | while read f; do
    acc=$(basename "$f" | grep -oP 'val_acc=\K[0-9.]+')
    echo "$acc $f"
done | sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoints found in $CHECKPOINTS_DIR"
    exit 1
fi

echo "Best checkpoint: $BEST_CHECKPOINT"
echo "========================================"

python src/visualize_joint_pruning.py \
    --checkpoint "$BEST_CHECKPOINT" \
    --config "$CONFIG_PATH" \
    --output "$OUTPUT_DIR"

echo "Done! Figures saved to: $OUTPUT_DIR"
