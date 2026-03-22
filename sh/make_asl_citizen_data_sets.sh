
echo "Generating ASL Citizen pipeline CSVs..."
python src/data/make_asl_citizen_csvs.py \
    --data_dir ~/groups/grp_asl_classification/nobackup/archive/ASL \
    --out_dir data/asl_citizen
