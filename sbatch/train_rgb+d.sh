#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=4
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/CS650R/asl/sbatch/slurm_outputs/%j_%x.out
#SBATCH --job-name=rgb+d_train
#SBATCH --qos=dw87


python src/clean_slurm_outputs.py

nvidia-smi

srun python src/train.py \
    -c configs/rgb+d.yaml \
    -m TRAIN

python src/clean_slurm_outputs.py