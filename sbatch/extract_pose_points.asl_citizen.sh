#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=extract_posepoints.asl_citizen
#SBATCH --qos=matrix

# ASL Citizen has ~84k videos in a single flat directory.
# All skeletons output to skel/ alongside the videos.

python src/pose/pose_points.py \
    --input_path /home/$USER/groups/grp_asl_classification/nobackup/archive/ASL/videos \
    --output_path /home/$USER/groups/grp_asl_classification/nobackup/archive/ASL/skel \
    --mode extract --num_workers 64
