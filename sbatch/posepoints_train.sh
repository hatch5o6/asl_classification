#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=a100:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/CS650R/asl/sbatch/slurm_outputs/%j_%x.out
#SBATCH --job-name=posepoints_train
#SBATCH --qos=cs
#SBATCH --partition=cs

python src/create_pose_points.py \
    -d /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/train/train