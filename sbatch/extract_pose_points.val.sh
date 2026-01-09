#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/CS650R/asl/sbatch/slurm_outputs/%j_%x.out
#SBATCH --job-name=extract_posepoints.val
#SBATCH --qos=dw87


python src/pose_points.py \
    --input_path /home/$USER/groups/grp_asl_classification/nobackup/archive/AUTSL/val/val \
    --output_path /home/$USER/groups/grp_asl_classification/nobackup/archive/AUTSL/val/val_skel \
    --mode extract
    