#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/CS650R/asl/sbatch/slurm_outputs/%j_%x.out
#SBATCH --job-name=extract_posepoints.test
#SBATCH --qos=dw87

python src/pose_points.py \
    --input_path /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/test/test \
    --output_path /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/test/test_skel \
    --mode extract

