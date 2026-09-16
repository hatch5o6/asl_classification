#!/bin/bash
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=dw
#SBATCH --qos=matrix
#SBATCH --job-name=detrate
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%x_%j.out
#SBATCH --exclude=dw-2-4,cs-1-2
# dw-2-4 fails every job at CUDA init (torch._C._cuda_init) within ~30 s and, because it frees
# up so fast, the scheduler keeps feeding it work: on 2026-09-14 it swallowed 25 of 54 jobs
# while every other node succeeded. docs/submission_instructions.md already listed both nodes
# as known-bad; putting the exclusion here stops it depending on remembering a CLI flag.
# CPU-only: reads landmark arrays and counts detections. Requests no GPU, so it does not
# occupy one on the node. See scripts/measure_detection_rates.py for what is measured.
source ~/.bashrc
conda activate asl
export PYTHONUNBUFFERED=1
python scripts/measure_detection_rates.py --per-split 2000 --out data/detection_rates.json
exit $?
