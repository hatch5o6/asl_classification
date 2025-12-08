#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=optimize_hyperparams_rgb+d
#SBATCH --qos=matrix

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

conda init
conda activate asl

python src/clean_slurm_outputs.py

nvidia-smi

python src/optimize_hyperparams.py \
    -n 25 \
    -c configs/rgb+d.yaml

python src/clean_slurm_outputs.py