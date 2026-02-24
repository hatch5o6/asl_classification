#!/bin/bash
#
# Extract joint importance probabilities and generate top-K index files.
#
# Usage (standard - extracts from a 543-joint model):
#   sbatch sbatch/extract_joint_indices.sh <model_dir_name> <config_path>
#
# Usage (iterative cascade - sub-selects from a reduced model):
#   sbatch sbatch/extract_joint_indices.sh <model_dir_name> <config_path> \
#       <k_values> <output_dir> <prefix> <source_indices_file> [train_config]
#
# Args:
#   1. model_dir_name     - Model directory under $MODELS_DIR
#   2. config_path        - Config YAML used to train the model
#   3. k_values           - Space-separated k values (default: "270 100 48 24 10")
#   4. output_dir         - Where to write index JSON files (default: data/informed_selection/topk)
#   5. prefix             - Filename prefix for index files (default: top)
#   6. source_indices     - Source indices JSON for cascade sub-selection (default: none)
#   7. train_config       - If set, submits this config for training after extraction
#
# Examples:
#   # Standard (543-joint model → topk indices):
#   sbatch sbatch/extract_joint_indices.sh \
#       pruning_sweep_l0_fixed_v3_optimized_v2 configs/pruning_sweep/l0_fixed_v3.yaml
#
#   # Cascade (topk_270 → iter_100 indices + start training):
#   sbatch sbatch/extract_joint_indices.sh \
#       informed_selection/topk_270 configs/informed_selection/topk_270.yaml \
#       "100" data/informed_selection/iterative iter \
#       data/informed_selection/iterative/iter_270_indices.json \
#       configs/informed_selection/iterative_100.yaml

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
#SBATCH --job-name=extract_joints
#SBATCH --qos=matrix

MODEL_NAME=${1:?"Error: model dir name required"}
CONFIG_PATH=${2:?"Error: config path required"}
K_VALUES=${3:-"270 100 48 24 10"}
OUTPUT_DIR=${4:-"data/informed_selection/topk"}
PREFIX=${5:-"top"}
SOURCE_INDICES=${6:-""}
TRAIN_CONFIG=${7:-""}

MODELS_DIR="/home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/models"
MODEL_DIR="$MODELS_DIR/$MODEL_NAME"
CHECKPOINTS_DIR="$MODEL_DIR/checkpoints"
FIGURES_DIR="$MODEL_DIR/figures"

echo "=========================================="
echo "Extract Joint Indices Pipeline"
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_PATH"
echo "K values: $K_VALUES"
echo "Output dir: $OUTPUT_DIR"
echo "Prefix: $PREFIX"
echo "Source indices: ${SOURCE_INDICES:-none}"
echo "Chain training: ${TRAIN_CONFIG:-none}"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source ~/.bashrc
conda init
conda activate asl

nvidia-smi

# Step 1: Find best checkpoint by val_acc
BEST_CHECKPOINT=$(ls -1 "$CHECKPOINTS_DIR"/*.ckpt 2>/dev/null | while read f; do
    acc=$(basename "$f" | grep -oP 'val_acc=\K[0-9.]+')
    echo "$acc $f"
done | sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "ERROR: No checkpoints found in $CHECKPOINTS_DIR"
    exit 1
fi

echo "Best checkpoint: $BEST_CHECKPOINT"
echo "=========================================="

# Step 2: Extract joint probabilities and generate figures
echo "Step 1/2: Extracting joint probabilities..."
python src/visualize_joint_pruning.py \
    --checkpoint "$BEST_CHECKPOINT" \
    --config "$CONFIG_PATH" \
    --output "$FIGURES_DIR"

PROBS_CSV="$FIGURES_DIR/joint_probabilities.csv"
if [ ! -f "$PROBS_CSV" ]; then
    echo "ERROR: joint_probabilities.csv not found at $PROBS_CSV"
    exit 1
fi

# Step 3: Generate top-K index files
echo ""
echo "Step 2/2: Generating top-K index files (k=$K_VALUES)..."
SELECT_CMD="python src/select_top_k_joints.py \
    --probabilities $PROBS_CSV \
    --k $K_VALUES \
    --output-dir $OUTPUT_DIR \
    --prefix $PREFIX"

if [ -n "$SOURCE_INDICES" ]; then
    SELECT_CMD="$SELECT_CMD --source-indices $SOURCE_INDICES"
fi

eval $SELECT_CMD

echo "=========================================="
echo "Done!"
echo "  Probabilities: $PROBS_CSV"
echo "  Indices: $OUTPUT_DIR/${PREFIX}_*_indices.json"
echo "End time: $(date)"
echo "=========================================="

# Step 4 (optional): Chain training job
if [ -n "$TRAIN_CONFIG" ]; then
    echo ""
    # Derive job name from config filename (e.g. iterative_48.yaml -> iterative_48)
    TRAIN_JOB_NAME=$(basename "$TRAIN_CONFIG" .yaml)
    echo "Submitting chained training job: $TRAIN_CONFIG (job-name: $TRAIN_JOB_NAME)"
    TRAIN_JOB=$(sbatch --job-name="$TRAIN_JOB_NAME" sbatch/train_informed_selection.sh "$TRAIN_CONFIG" | grep -oP '\d+')
    echo "Submitted training job: $TRAIN_JOB"
fi
