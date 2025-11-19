
echo "MAKING TEST"
python src/make_data_csvs.py \
    -d ~/groups/grp_asl_classification/nobackup/archive/AUTSL/test/test \
    -l ~/groups/grp_asl_classification/nobackup/archive/AUTSL/test/ground_truth.csv \
    --out data/test.csv

echo "MAKING TRAIN"
python src/make_data_csvs.py \
    -d ~/groups/grp_asl_classification/nobackup/archive/AUTSL/train/train \
    -l ~/groups/grp_asl_classification/nobackup/archive/AUTSL/train/train_labels.csv \
    --out data/train.csv

echo "MAKING VAL"
python src/make_data_csvs.py \
    -d ~/groups/grp_asl_classification/nobackup/archive/AUTSL/val/val \
    -l ~/groups/grp_asl_classification/nobackup/archive/AUTSL/val/ground_truth.csv \
    --out data/val.csv
