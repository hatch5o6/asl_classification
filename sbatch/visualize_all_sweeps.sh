#!/bin/bash

# Submit visualization jobs for all pruning sweep models
# Usage: bash sbatch/visualize_all_sweeps.sh

MODELS_DIR="/home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/models"

echo "Submitting visualization jobs for all pruning sweep models..."

for MODEL_DIR in "$MODELS_DIR"/pruning_sweep_*; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    # Map model name to config: pruning_sweep_l0_0.01 -> configs/pruning_sweep/l0_0.01.yaml
    CONFIG_NAME=$(echo "$MODEL_NAME" | sed 's/pruning_sweep_//')
    CONFIG_PATH="configs/pruning_sweep/${CONFIG_NAME}.yaml"

    if [ ! -f "$CONFIG_PATH" ]; then
        echo "WARNING: Config $CONFIG_PATH not found for $MODEL_NAME, skipping"
        continue
    fi

    CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"
    if [ ! -d "$CHECKPOINTS_DIR" ] || [ -z "$(ls "$CHECKPOINTS_DIR"/*.ckpt 2>/dev/null)" ]; then
        echo "WARNING: No checkpoints found for $MODEL_NAME, skipping"
        continue
    fi

    echo "  Submitting: $MODEL_NAME (config: $CONFIG_PATH)"
    sbatch --job-name="viz_${CONFIG_NAME}" sbatch/visualize_pruning_sweep_single.sh "$MODEL_NAME" "$CONFIG_PATH"
done

echo "Done. Check queue with: squeue -u \$USER"
