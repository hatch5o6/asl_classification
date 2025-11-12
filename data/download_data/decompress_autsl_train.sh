#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --mem=128000M
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/fsl_groups/grp_mtlab/nobackup/archive/CS650R_ASL/AUTSL/train/SLURM_%j_%x.out
#SBATCH --job-name=decompress_autsl_train
#SBATCH --qos cs
#SBATCH --partition cs

ml load p7zip
# for f in train_set_vfbha39.zip.*; do
#     echo "extracting $f"
#     7z x "$f" -p"MdG3z6Eh1t" -y
# done
7z x train_set_vfbha39.zip.001 -p"MdG3z6Eh1t" -y