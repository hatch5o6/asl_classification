#!/bin/bash
#
# Visualize joint pruning results for all gating experiments
#
# Usage:
#   sbatch visualize_all_gates.sh
#
# This script visualizes all 3 gating models created by submit_gating_sweeps.sh:
#   1. pruning_sweep_gating_only   - Gating without L0 (baseline)
#   2. pruning_sweep_gating_l0     - Gating + L0, no random init
#   3. pruning_sweep_gating_hybrid - Gating + L0 + random init

#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=visualize_gates
#SBATCH --qos=matrix

# Models to visualize (from submit_gating_sweeps.sh)
MODELS=(
    "pruning_sweep_gating_only:configs/pruning_sweep/gating_only.yaml"
    "pruning_sweep_gating_l0:configs/pruning_sweep/gating_l0.yaml"
    "pruning_sweep_gating_hybrid:configs/pruning_sweep/gating_hybrid.yaml"
)

MODELS_DIR="/home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/models"

echo "========================================"
echo "Visualizing All Gating Experiments"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

source ~/.bashrc

conda init
conda activate asl

nvidia-smi

# Function to find best checkpoint
find_best_checkpoint() {
    local checkpoints_dir="$1"
    ls -1 "$checkpoints_dir"/*.ckpt 2>/dev/null | while read f; do
        acc=$(basename "$f" | grep -oP 'val_acc=\K[0-9.]+')
        echo "$acc $f"
    done | sort -rn | head -1 | cut -d' ' -f2
}

# Visualize each model
for entry in "${MODELS[@]}"; do
    MODEL_NAME="${entry%%:*}"
    CONFIG_PATH="${entry#*:}"

    MODEL_DIR="$MODELS_DIR/$MODEL_NAME"
    CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"
    OUTPUT_DIR="$MODEL_DIR/figures"

    echo ""
    echo "========================================"
    echo "Model: $MODEL_NAME"
    echo "Config: $CONFIG_PATH"
    echo "========================================"

    # Check if model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        echo "WARNING: Model directory not found: $MODEL_DIR"
        echo "Skipping..."
        continue
    fi

    # Find best checkpoint
    BEST_CHECKPOINT=$(find_best_checkpoint "$CHECKPOINTS_DIR")

    if [ -z "$BEST_CHECKPOINT" ]; then
        echo "WARNING: No checkpoints found in $CHECKPOINTS_DIR"
        echo "Skipping..."
        continue
    fi

    echo "Best checkpoint: $BEST_CHECKPOINT"

    # Run visualization
    python src/visualize_joint_pruning.py \
        --checkpoint "$BEST_CHECKPOINT" \
        --config "$CONFIG_PATH" \
        --output "$OUTPUT_DIR"

    echo "Figures saved to: $OUTPUT_DIR"
done

echo ""
echo "========================================"
echo "Done! All visualizations complete."
echo "End time: $(date)"
echo "========================================"
