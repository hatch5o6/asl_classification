#!/bin/bash

#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/smoke_%A_%x.out
#SBATCH --job-name=smoke
#SBATCH --qos=cs

# Quick per-node smoke test: confirm conda env activates and lightning imports.
# Submit with --nodelist=<node> to target a specific node.

set -e

echo "=========================================="
echo "Smoke test on $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

source ~/.bashrc
conda activate asl

python -c "
import sys
print('python:', sys.executable)
import lightning as L
print('lightning:', L.__version__)
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('NODE_OK')
"

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
