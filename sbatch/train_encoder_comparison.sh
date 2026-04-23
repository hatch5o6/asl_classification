#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=8
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=enc_cmp
#SBATCH --qos=cs

# Usage:
#   sbatch sbatch/train_encoder_comparison.sh <config_path> [MODE]
#   MODE: TRAIN (default), RESUME, TEST
#
# Examples:
#   sbatch --job-name=bert_autsl_k48 sbatch/train_encoder_comparison.sh configs/encoder_comparison/autsl/bert/k48.yaml
#   sbatch --job-name=bert_autsl_k48 sbatch/train_encoder_comparison.sh configs/encoder_comparison/autsl/bert/k48.yaml RESUME

CONFIG_PATH=${1:?"Error: config path required (e.g., configs/encoder_comparison/autsl/bert/k48.yaml)"}
MODE=${2:-TRAIN}

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
fi

echo "=========================================="
echo "Encoder Comparison - Mode: $MODE"
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
