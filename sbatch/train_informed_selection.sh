#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=8
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=informed_sel
#SBATCH --qos=matrix

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
