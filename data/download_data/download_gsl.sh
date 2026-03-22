#!/bin/bash
# Downloads and processes the GSL (Greek Sign Language) dataset.
#
# Key insight: GSL zips contain JPEG frame directories (~100k files per zip).
# Extracting all zips at once exceeds the Lustre inode quota. Instead, this
# script processes each zip directly (zip → gloss MP4s, no JPEG extraction
# to archive) to keep inode usage minimal.
#
# Usage: run from anywhere, set ARCHIVE below. Downloads to ARCHIVE/GSL/.
#
#   cd /home/$USER/asl_classification
#   sh data/download_data/download_gsl.sh
#
# Expected structure after completion:
#   archive/GSL/
#   ├── videos/          <- ~40k isolated gloss MP4s
#   ├── skel/            <- skeleton .npy files (after extract_pose_points.gsl.sh)
#   ├── annotations/
#   │   ├── train_greek_iso.csv
#   │   ├── dev_greek_iso.csv
#   │   ├── test_greek_iso.csv
#   │   └── iso_classes.csv
#   ├── supplementary.zip
#   └── (scenario zips deleted after processing)
#
# data/gsl_merged_continuous.csv is stored in the project repo (not archive).
#
# After this script:
#   sbatch sbatch/extract_pose_points.gsl.sh   <- MediaPipe on videos/
#   sh sh/make_gsl_data_sets.sh                <- generate pipeline CSVs

set -e

ARCHIVE=/home/$USER/groups/grp_asl_classification/nobackup/archive/GSL
BASE="https://zenodo.org/records/4756317/files"

mkdir -p "$ARCHIVE/videos" "$ARCHIVE/skel" "$ARCHIVE/annotations"

# merged_continuous.csv is kept in the project repo to avoid inode usage
MERGED_CSV=/home/$USER/asl_classification/data/gsl_merged_continuous.csv

# ---- annotation splits (small, download directly to project dir) ----
if [ ! -f "$MERGED_CSV" ]; then
    echo "Extracting merged_continuous.csv from supplementary.zip ..."
    if [ ! -f "$ARCHIVE/supplementary.zip" ]; then
        wget -c "${BASE}/supplementary.zip?download=1" -O "$ARCHIVE/supplementary.zip"
    fi
    unzip -p "$ARCHIVE/supplementary.zip" merged_continuous.csv > "$MERGED_CSV"
    echo "  Saved -> $MERGED_CSV ($(wc -l < $MERGED_CSV) lines)"
fi

echo "Downloading annotation split files from slrzoo..."
RAW="https://raw.githubusercontent.com/iliasprc/slrzoo/master/files/GSL_isolated"
wget -c "${RAW}/train_greek_iso.csv" -O "$ARCHIVE/annotations/train_greek_iso.csv"
wget -c "${RAW}/dev_greek_iso.csv"   -O "$ARCHIVE/annotations/dev_greek_iso.csv"
wget -c "${RAW}/test_greek_iso.csv"  -O "$ARCHIVE/annotations/test_greek_iso.csv"
wget -c "${RAW}/iso_classes.csv"     -O "$ARCHIVE/annotations/iso_classes.csv"

# ---- process each RGB scenario zip directly (no JPEG extraction to disk) ----
SCENARIOS="health1 health2 health3 health4 health5 police1 police2 police3 police4 police5 kep1 kep2 kep3 kep4 kep5"

for SCENARIO in $SCENARIOS; do
    ZIP="$ARCHIVE/${SCENARIO}.zip"

    # Count how many MP4s already exist for this scenario
    DONE=$(ls "$ARCHIVE/videos/${SCENARIO}_"*.mp4 2>/dev/null | wc -l)
    if [ "$DONE" -gt 0 ]; then
        echo "=== ${SCENARIO}: $DONE MP4s already exist, skipping download ==="
        continue
    fi

    echo "=== Downloading ${SCENARIO}.zip ==="
    wget -c "${BASE}/${SCENARIO}.zip?download=1" -O "$ZIP"

    echo "=== Segmenting ${SCENARIO}.zip -> gloss MP4s ==="
    python src/data/segment_gsl_glosses.py \
        --zip_path   "$ZIP" \
        --merged_csv "$MERGED_CSV" \
        --output_dir "$ARCHIVE/videos" \
        --fps 25 \
        --delete_zip

    echo "=== Done with ${SCENARIO} ==="
done

echo ""
echo "Total MP4s in videos/:"
ls "$ARCHIVE/videos"/*.mp4 2>/dev/null | wc -l
echo ""
echo "Next steps:"
echo "  sbatch sbatch/extract_pose_points.gsl.sh"
echo "  sh sh/make_gsl_data_sets.sh"
