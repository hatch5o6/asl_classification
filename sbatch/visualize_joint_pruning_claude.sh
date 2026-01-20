#!/bin/bash
#
# Visualize joint pruning results for any model
#
# Usage:
#   sbatch visualize_joint_pruning_claude.sh <model_name> [config_name]
#
# Examples:
#   sbatch visualize_joint_pruning_claude.sh s_tslformer_claude
#   sbatch visualize_joint_pruning_claude.sh s_tslformer_claude_v2
#   sbatch visualize_joint_pruning_claude.sh s_tslformer_claude_v2 s_tslformer_claude_v2
#
# If config_name is not provided, it defaults to model_name

#SBATCH --time=01:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=visualize_joints
#SBATCH --qos=matrix

# Parse arguments
MODEL_NAME=${1:-"s_tslformer_claude_v2"}
CONFIG_NAME=${2:-$MODEL_NAME}

# Set paths
MODELS_DIR="/home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/models"
MODEL_DIR="$MODELS_DIR/$MODEL_NAME"
CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"
OUTPUT_DIR="$MODEL_DIR/figures"
CONFIG_PATH="configs/${CONFIG_NAME}.yaml"

echo "========================================"
echo "Joint Pruning Visualization"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_PATH"
echo "Checkpoints dir: $CHECKPOINTS_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

source ~/.bashrc

conda init
conda activate asl

nvidia-smi

# Find the best checkpoint (highest val_acc)
BEST_CHECKPOINT=$(ls -1 "$CHECKPOINTS_DIR"/*.ckpt 2>/dev/null | while read f; do
    # Extract val_acc from filename (format: epoch=X-step=Y-val_loss=Z-val_acc=W.ckpt)
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

echo "========================================"
echo "Done! Figures saved to: $OUTPUT_DIR"
echo "========================================"
