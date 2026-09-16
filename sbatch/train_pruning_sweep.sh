#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=81920M
#SBATCH --gpus=4
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%x_%A_%a.out
#SBATCH --job-name=prSweep
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

python src/utils/clean_slurm_outputs.py --user "$USER"

nvidia-smi

srun python src/train.py \
    -c "$CONFIG_PATH" \
    -m TRAIN

python src/utils/clean_slurm_outputs.py --user "$USER"
