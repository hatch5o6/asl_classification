#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=4
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=pruning_sweep
#SBATCH --qos=matrix

# Usage: sbatch sbatch/train_pruning_sweep.sh <config_name>
# Example: sbatch sbatch/train_pruning_sweep.sh l0_0.01
#
# This will train using configs/pruning_sweep/<config_name>.yaml

CONFIG_NAME=${1:-"l0_0.01"}
CONFIG_PATH="configs/pruning_sweep/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file $CONFIG_PATH not found"
    exit 1
fi

echo "Training with config: $CONFIG_PATH"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source ~/.bashrc

conda init
conda activate asl

python src/clean_slurm_outputs.py --user "$USER"

nvidia-smi

srun python src/train.py \
    -c "$CONFIG_PATH" \
    -m TRAIN

python src/clean_slurm_outputs.py --user "$USER"
