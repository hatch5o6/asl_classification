#!/bin/bash

# Submit all 3 fixed L0 experiments
# v1: l0_weight = 20.0 (moderate)
# v2: l0_weight = 50.0 (aggressive)
# v3: l0_weight = 10.0 (conservative)

echo "Submitting FIXED L0 pruning sweep (3 experiments)"
echo "=================================================="
echo ""
echo "All experiments use:"
echo "  - init_keep_probability: 0.98 (high start)"
echo "  - Normalized L0 penalty (proper gradient balance)"
echo "  - Aggressive temp annealing (10.0 → 0.01 over 50k steps)"
echo "  - Disabled early stopping"
echo ""
echo "Testing L0 weights: 10.0, 20.0, 50.0"
echo ""

# v3: Conservative (10.0)
echo "Submitting v3 (l0_weight=10.0)..."
sbatch sbatch/train_pruning_sweep.sh l0_fixed_v3

# v1: Moderate (20.0)
echo "Submitting v1 (l0_weight=20.0)..."
sbatch sbatch/train_pruning_sweep.sh l0_fixed_v1

# v2: Aggressive (50.0)
echo "Submitting v2 (l0_weight=50.0)..."
sbatch sbatch/train_pruning_sweep.sh l0_fixed_v2

echo ""
echo "All jobs submitted! Monitor with: squeue -u $USER"
