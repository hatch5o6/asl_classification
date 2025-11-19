# Feature Ablation for Turkish Sign Language Classification

All files are written to be run from inside the root folder of the repo 'asl'. (If yours is called something different, you may need to change some file paths).
#TODO -- fix this soon.

# Requirements
```
pip install -r requirements.txt
```

For hyperparameter sweep, install:
```
pip install optuna
pip install optuna-integration[pytorch_lightning]
```


### Data
Data is saved in the shared folder ~/groups/grp_asl_classification/nobackup/archive/AUTSL/ on the super computer
You'll need to make some data manifest files:
```
sh sh/make_data_sets.sh
```

