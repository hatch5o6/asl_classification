# set this
DATA_DIR=/home/$USER/groups/grp_asl_classification/nobackup/archive/AUTSL
rm -r $DATA_DIR

# don't touch the rest
TRAIN_DIR=$DATA_DIR/train
VAL_DIR=$DATA_DIR/val
TEST_DIR=$DATA_DIR/test

mkdir $DATA_DIR
mkdir $DATA_DIR/class_ids
mkdir $TRAIN_DIR
mkdir $VAL_DIR
mkdir $TEST_DIR

cp download_autsl_train.sh $TRAIN_DIR
cp decompress_autsl_train.sh $TRAIN_DIR

cp download_autsl_val.sh $VAL_DIR
cp decompress_autsl_val.sh $VAL_DIR

cp download_autsl_test.sh $TEST_DIR
cp decompress_autsl_test.sh $TEST_DIR

cp download_class_ids.sh $DATA_DIR/class_ids

cp notes $DATA_DIR

# Then, run these at your leisure:
# sh $TRAIN_DIR/download_autsl_train.sh
# sh $TRAIN_DIR/decompress_autsl_train.sh
# then do the same for val and test :)
