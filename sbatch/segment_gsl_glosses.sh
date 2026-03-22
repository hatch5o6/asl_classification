#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --mem=64000M
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user %u@byu.edu
#SBATCH --output /home/%u/groups/grp_asl_classification/nobackup/archive/SLR/slurm_outputs/%j_%x.out
#SBATCH --job-name=segment_gsl
#SBATCH --qos=matrix

# Converts GSL scenario zips into isolated gloss MP4 clips, one zip at a time.
# Zips must already be downloaded (compute nodes have no internet access).
#
# Before running:
#   cd /home/$USER/groups/grp_asl_classification/nobackup/archive/GSL
#   sh /home/$USER/asl_classification/data/download_data/download_gsl_zips.sh
#
# This script:
#   - Loops over all 15 scenarios
#   - Skips any whose MP4s already exist in videos/
#   - Segments each zip -> isolated gloss MP4s (JPEGs only ever in /tmp)
#   - Deletes the zip after successful segmentation
#   - Re-runnable: cancelled jobs can be safely resubmitted

ARCHIVE=/home/$USER/groups/grp_asl_classification/nobackup/archive/GSL
MERGED_CSV=/home/$USER/asl_classification/data/gsl_merged_continuous.csv
OUTPUT_DIR=${ARCHIVE}/videos

SCENARIOS="health1 health2 health3 health4 health5 \
           police1 police2 police3 police4 police5 \
           kep1 kep2 kep3 kep4 kep5"

mkdir -p "$OUTPUT_DIR"

for SCENARIO in $SCENARIOS; do
    ZIP=${ARCHIVE}/${SCENARIO}.zip

    # Skip if this scenario's MP4s are already in videos/
    DONE=$(ls "$OUTPUT_DIR/${SCENARIO}_"*.mp4 2>/dev/null | wc -l)
    if [ "$DONE" -gt 0 ]; then
        echo "=== ${SCENARIO}: ${DONE} MP4s already exist, skipping ==="
        continue
    fi

    # Skip if zip not on disk (needs to be downloaded first on login node)
    if [ ! -f "$ZIP" ]; then
        echo "=== ${SCENARIO}.zip not found — download it on the login node first ==="
        continue
    fi

    # Skip if zip is 0 bytes (failed wget)
    ZIP_SIZE=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
    if [ "$ZIP_SIZE" -lt 1000000 ]; then
        echo "=== ${SCENARIO}.zip is too small (${ZIP_SIZE} bytes) — likely corrupt, skipping ==="
        continue
    fi

    echo "=== Segmenting ${SCENARIO}.zip -> gloss MP4s ==="
    python src/data/segment_gsl_glosses.py \
        --zip_path   "$ZIP" \
        --merged_csv "$MERGED_CSV" \
        --output_dir "$OUTPUT_DIR" \
        --fps 25 \
        --delete_zip

    echo "=== Done with ${SCENARIO} ==="
    echo "    Running MP4 count: $(ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)"
done

echo ""
echo "All scenarios done."
echo "Total MP4s in videos/: $(ls "$OUTPUT_DIR"/*.mp4 2>/dev/null | wc -l)"
