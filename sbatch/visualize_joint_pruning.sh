#!/bin/bash

#SBATCH --time=01:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=visualize_joints
#SBATCH --qos=matrix

source ~/.bashrc

conda init
conda activate asl

nvidia-smi

python src/visualize_joint_pruning.py \
    --checkpoint /home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/models/s_tslformer_claude/checkpoints/epoch=14-step=13200-val_loss=7.427081-val_acc=0.125000.ckpt \
    --config configs/s_tslformer_claude.yaml \
    --output /home/$USER/groups/grp_asl_classification/nobackup/archive/SLR/models/s_tslformer_claude/figures
