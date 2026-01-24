#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=analyze_sweep
#SBATCH --qos=matrix

source ~/.bashrc
conda init
conda activate asl

MODELS_DIR="/home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/models"

python src/analyze_pruning_sweep.py --models_dir "$MODELS_DIR"
