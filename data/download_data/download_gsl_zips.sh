#!/bin/bash
# Downloads all 15 GSL RGB scenario zips to the archive directory.
# Run this from the LOGIN NODE (compute nodes have no internet access).
#
# Usage:
#   cd /home/$USER/groups/grp_asl_classification/nobackup/archive/GSL
#   sh /home/$USER/asl_classification/data/download_data/download_gsl_zips.sh
#
# After all zips are downloaded, submit the segmentation job:
#   cd /home/$USER/asl_classification
#   sbatch sbatch/segment_gsl_glosses.sh

set -e

ARCHIVE=/home/$USER/groups/grp_asl_classification/nobackup/archive/GSL
BASE="https://zenodo.org/records/4756317/files"

SCENARIOS="health1 health2 health3 health4 health5 \
           police1 police2 police3 police4 police5 \
           kep1 kep2 kep3 kep4 kep5"

for SCENARIO in $SCENARIOS; do
    ZIP=${ARCHIVE}/${SCENARIO}.zip
    DONE=$(ls "${ARCHIVE}/videos/${SCENARIO}_"*.mp4 2>/dev/null | wc -l)

    if [ "$DONE" -gt 0 ]; then
        echo "Skipping ${SCENARIO} (${DONE} MP4s already exist)"
        continue
    fi

    echo "Downloading ${SCENARIO}.zip ..."
    wget -c "${BASE}/${SCENARIO}.zip?download=1" -O "$ZIP"
    echo "  Done: $(du -sh $ZIP | cut -f1)"
done

echo ""
echo "All zips downloaded. Now submit:"
echo "  cd /home/\$USER/asl_classification && sbatch sbatch/segment_gsl_glosses.sh"
