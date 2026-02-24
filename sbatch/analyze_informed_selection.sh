#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=32000M
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=analyze_sel
#SBATCH --qos=matrix

# Usage: sbatch sbatch/analyze_informed_selection.sh
# Generates paper-quality figures from all informed selection models

echo "=========================================="
echo "Informed Selection Analysis"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

source ~/.bashrc

conda init
conda activate asl

MODELS_DIR="/home/${USER}/groups/grp_asl_classification/nobackup/archive/SLR/models/informed_selection"
OUTPUT_DIR="${MODELS_DIR}/paper_figures"

python src/analyze_informed_selection.py \
    --models-dir "$MODELS_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "Figures saved to: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="
