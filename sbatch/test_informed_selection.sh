#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=informed_sel_test
#SBATCH --qos=matrix

# Usage: sbatch sbatch/test_informed_selection.sh <config_name_or_path>
# Example: sbatch sbatch/test_informed_selection.sh configs/pruning_sweep/l0_fixed_v3_baseline.yaml

CONFIG_INPUT=${1:?"Error: config name or path required (e.g., topk_270, or configs/pruning_sweep/l0_fixed_v3.yaml)"}

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
echo "Informed Selection - Mode: TEST"
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

srun python src/train.py \
    -c "$CONFIG_PATH" \
    -m TEST

python src/utils/clean_slurm_outputs.py --user "$USER"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="