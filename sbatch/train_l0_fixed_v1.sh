#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --qos=cs
#SBATCH --partition=cs
#SBATCH --output=/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_train_l0_fixed_v1.out

# FIXED L0 Pruning Implementation
# This experiment implements all 5 critical fixes from research:
# 1. High initialization (0.98)
# 2. Normalized L0 penalty
# 3. Aggressive temperature annealing (10.0 → 0.01 over 50k steps)
# 4. Higher L0 weight (20.0) safe with normalization
# 5. Disabled early stopping

echo "Training L0 FIXED v1 - Research-based implementation"
echo "Expected: Bimodal distribution with probs near 0.0 and 1.0"
echo ""

source ~/.bashrc && conda activate asl

cd /home/ccoulson/asl_classification

srun python src/train.py --config configs/pruning_sweep/l0_fixed_v1.yaml
