#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=extract_posepoints.gsl
#SBATCH --qos=matrix

# GSL has ~40k MP4 clips (packed from JPEG frame dirs by pack_frames_to_mp4.gsl.sh).
# All skeletons output flat to skel/ alongside videos/.

source ~/.bashrc
conda init
conda activate asl

python src/pose/pose_points.py \
    --input_path /home/$USER/groups/grp_asl_classification/nobackup/archive/GSL/videos \
    --output_path /home/$USER/groups/grp_asl_classification/nobackup/archive/GSL/skel \
    --mode extract --num_workers 64
