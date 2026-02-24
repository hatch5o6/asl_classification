#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_gating_sweep.out
#SBATCH --mail-type=END,FAIL
#SBATCH --qos=dw87

# Activate conda environment
source ~/.bashrc
conda activate slr

# Navigate to project directory
cd /home/ccoulson/asl_classification

# Get sweep ID from command line argument
SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Error: No sweep ID provided"
    echo "Usage: sbatch scripts/run_gating_sweep.sh <sweep-id>"
    exit 1
fi

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Sweep ID: $SWEEP_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Run wandb agent
wandb agent $SWEEP_ID

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
