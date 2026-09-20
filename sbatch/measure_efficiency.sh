#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=16384M
#SBATCH --gpus=1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%x_%j.out
#SBATCH --job-name=effmeas
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --exclude=dw-2-4,cs-1-2
# dw-2-4 and cs-1-2 fail every GPU job at CUDA init; see docs/submission_instructions.md.
#
# Measures inference cost of the stage-2 skeleton model across K, on GPU and CPU.
# Small and short: one process, no data loading, no training. 30 min is generous.
#
# Usage:
#   sbatch sbatch/measure_efficiency.sh

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "=========================================="
echo "Efficiency measurement"
echo "Job:   $SLURM_JOB_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

source ~/.bashrc
conda init
conda activate asl

nvidia-smi

set -o pipefail
STATUS=0

srun python scripts/measure_efficiency.py --device cuda --out docs/EFFICIENCY_gpu.md || STATUS=$?
srun python scripts/measure_efficiency.py --device cpu  --out docs/EFFICIENCY_cpu.md || STATUS=$?

echo "=========================================="
echo "End:   $(date)"
echo "Exit:  $STATUS"
echo "=========================================="
exit $STATUS
