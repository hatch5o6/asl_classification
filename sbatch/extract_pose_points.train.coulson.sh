#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --output /home/ccoulson/asl_classification/slurm_outputs/%j_%x.out
#SBATCH --job-name=extract_posepoints.train
#SBATCH --qos=dw87
#SBATCH --cpus-per-task=64

python src/pose_points.py \
    --input_path /home/ccoulson/groups/grp_asl_classification/nobackup/archive/AUTSL/train/train \
    --output_path /home/ccoulson/groups/grp_asl_classification/nobackup/archive/AUTSL/train/train_skel \
    --mode extract --num_workers 64 

    