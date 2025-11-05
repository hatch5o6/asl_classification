python src/make_data_csvs.py \
    -d /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/test/test \
    -l /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/test/ground_truth.csv \
    --out /home/hatch5o6/CS650R/asl/data/test.csv

python src/make_data_csvs.py \
    -d /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/train/train \
    -l /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/train/train_labels.csv \
    --out /home/hatch5o6/CS650R/asl/data/train.csv

python src/make_data_csvs.py \
    -d /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/val/val \
    -l /home/hatch5o6/groups/grp_asl_classification/nobackup/archive/AUTSL/val/ground_truth.csv \
    --out /home/hatch5o6/CS650R/asl/data/val.csv