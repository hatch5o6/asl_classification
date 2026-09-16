#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=163840M
#SBATCH --gpus=8
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%x_%A_%a.out
#SBATCH --job-name=infSel
#SBATCH --qos=matrix
# Log naming: %x (job name) MUST come first. clean_slurm_outputs.py -- which is
# shared with other users -- groups files by everything after the first '_' and
# keeps only the newest per group, so the old %A_%a_%x scheme made task N of every
# array share the key 'N_<jobname>.out' and a new submission deleted a RUNNING
# job's log. Leading with a non-numeric token makes its isdigit() guard skip ours.
#
# NOTE: skeleton configs should set `cache_val_skeletons: true`. Validation clips are
# preprocessed identically on every pass, and on ASL Citizen that is ~51% of all clip
# preprocessing (10,304-clip val split x4 passes/epoch at val_interval=0.25). The cache
# is ~60 MB, bit-identical, and self-disables if augmentation is configured.

# Usage: sbatch sbatch/train_informed_selection.sh <config_name>
# Example: sbatch sbatch/train_informed_selection.sh topk_270
#          sbatch sbatch/train_informed_selection.sh iterative_100

CONFIG_INPUT=${1:?"Error: config name or path required (e.g., topk_270, iterative_100, or configs/pruning_sweep/l0_fixed_v3.yaml)"}
MODE=${2:-TRAIN}

# Support both config names (in informed_selection/) and full paths
if [ -f "$CONFIG_INPUT" ]; then
    CONFIG_PATH="$CONFIG_INPUT"
elif [ -f "configs/informed_selection/${CONFIG_INPUT}.yaml" ]; then
    CONFIG_PATH="configs/informed_selection/${CONFIG_INPUT}.yaml"
else
    echo "Error: Config not found as '$CONFIG_INPUT' or 'configs/informed_selection/${CONFIG_INPUT}.yaml'"
    exit 1
fi

echo "=========================================="
echo "Informed Selection - Mode: $MODE"
echo "Config: $CONFIG_PATH"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source ~/.bashrc

conda init
conda activate asl

python src/utils/clean_slurm_outputs.py --user "$USER"

nvidia-smi

# TEST mode must run on 1 GPU (srun with DDP splits test data across GPUs)
if [ "$MODE" = "TEST" ]; then
    python src/train.py \
        -c "$CONFIG_PATH" \
        -m "$MODE"
else
    srun python src/train.py \
        -c "$CONFIG_PATH" \
        -m "$MODE"
fi

python src/utils/clean_slurm_outputs.py --user "$USER"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
