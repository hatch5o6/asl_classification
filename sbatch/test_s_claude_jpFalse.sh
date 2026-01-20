#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=s_test_claude_jpFalse
#SBATCH --qos=matrix

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

source ~/.bashrc

conda init
conda activate asl

python src/clean_slurm_outputs.py --user "$USER"

nvidia-smi

srun python src/train.py \
    -c configs/s_tslformer_claude_jpFalse.yaml \
    -m TEST

python src/clean_slurm_outputs.py --user "$USER"