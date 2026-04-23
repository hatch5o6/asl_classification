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
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%A_%a_%x.out
#SBATCH --job-name=enc_array
#SBATCH --qos=cs

# Job array sbatch script for encoder comparison.
#
# Each array task reads one line from a config list file to get its config
# and job name. The array index (SLURM_ARRAY_TASK_ID) is 0-based.
#
# Usage:
#   sbatch --array=0-47%16 sbatch/train_encoder_array.sh <config_list> [MODE]
#
# Config list format (one entry per line, tab-separated):
#   configs/encoder_comparison/autsl/gru/k10.yaml   gru_autsl_k10
#   configs/encoder_comparison/autsl/gru/k24.yaml   gru_autsl_k24
#   ...
#
# Modes: TRAIN (default), RESUME, TEST

CONFIG_LIST=${1:?"Error: config list file required"}
MODE=${2:-TRAIN}

if [ ! -f "$CONFIG_LIST" ]; then
    echo "Error: Config list not found: $CONFIG_LIST"
    exit 1
fi

# Read config and job name for this task (1-based line number)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")
CONFIG=$(echo "$LINE" | awk '{print $1}')
JOBNAME=$(echo "$LINE" | awk '{print $2}')

if [ -z "$CONFIG" ]; then
    echo "Error: No config found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

# Update the SLURM job name to reflect this specific task
scontrol update jobid="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" jobname="$JOBNAME" 2>/dev/null || true

echo "=========================================="
echo "Encoder Array Job"
echo "Array ID: ${SLURM_ARRAY_JOB_ID}, Task: ${SLURM_ARRAY_TASK_ID}"
echo "Job name: $JOBNAME"
echo "Config:   $CONFIG"
echo "Mode:     $MODE"
echo "Node:     $SLURMD_NODENAME"
echo "Start:    $(date)"
echo "=========================================="

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source ~/.bashrc
conda init
conda activate asl

python src/utils/clean_slurm_outputs.py --user "$USER"

nvidia-smi

if [ "$MODE" = "TEST" ]; then
    python src/train.py -c "$CONFIG" -m "$MODE"
else
    srun python src/train.py -c "$CONFIG" -m "$MODE"
fi

python src/utils/clean_slurm_outputs.py --user "$USER"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
