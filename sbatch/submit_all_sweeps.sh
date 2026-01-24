#!/bin/bash

# Submit all pruning sweep experiments
# Usage: bash sbatch/submit_all_sweeps.sh

echo "Submitting pruning sweep experiments..."

# L0 weight sweep (init=0.9)
sbatch --job-name=sweep_l0_0.005 sbatch/train_pruning_sweep.sh l0_0.005
sbatch --job-name=sweep_l0_0.01 sbatch/train_pruning_sweep.sh l0_0.01
sbatch --job-name=sweep_l0_0.05 sbatch/train_pruning_sweep.sh l0_0.05
sbatch --job-name=sweep_l0_0.1 sbatch/train_pruning_sweep.sh l0_0.1

# Neutral init experiment
sbatch --job-name=sweep_init0.5 sbatch/train_pruning_sweep.sh init_0.5_l0_0.01

echo "Submitted 5 experiments. Check queue with: squeue -u \$USER"
