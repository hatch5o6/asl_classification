import argparse
import optuna
from train import train   # assuming train() returns best val_acc
from train import read_config
import copy
import os
from optuna.integration import PyTorchLightningPruningCallback
import gc, torch

# We're going to optimize RGB + D, and Skel separately

def main(config, n_trials, RESUME):
    assert RESUME in [True, False]

    base_cfg =  read_config(config)
    # for optimization, will train on one gpu only.
    base_cfg["batch_size"] = base_cfg["effective_batch_size"]
    base_cfg["n_gpus"] = 1
    base_cfg["device"] = "cuda"
    print("HYPERPARAMETER OPTIMIZATION BASE CONFIG:")
    for k, v in base_cfg.items():
        print(f"\t-{k}=`{v}`")
    print("\n\n")

    print(f"TUNING {n_trials} TRIALS for modalities {base_cfg['modalities']}")
    def objective(trial):
        gc.collect()
        torch.cuda.empty_cache()

        cfg = copy.deepcopy(base_cfg)
        print(f"Running trial {trial.number}")

        assert cfg["modalities"] in [["rgb", "depth"], ["skeleton"]]
        if cfg["modalities"] == ["rgb", "depth"]:

            # cfg["batch_size"] = trial.suggest_categorical("batch_size", [16, 24, 32])
            # cfg["batch_size"] = trial.suggest_categorical("batch_size", [64, 96, 128]) # effective batch sizes
            cfg["fusion_dim"] = trial.suggest_categorical("fusion_dim", [256, 384, 512, 768])


            cfg["class_learning_rate"] = trial.suggest_float("class_learning_rate", 5e-5, 5e-4,log=True)
            cfg["pretrained_learning_rate"] = trial.suggest_float("pretrained_learning_rate", 2e-5, 1e-4,log=True)
            cfg["new_learning_rate"] = trial.suggest_float("new_learning_rate", 5e-5, 5e-4,log=True)
            
            # cfg["depth_image_size"] = trial.suggest_categorical("depth_image_size", [112, 168, 224]) # Keep it fixed at 224, which is the default
            cfg["depth_hidden_dim"] = trial.suggest_categorical("depth_hidden_dim", [384, 512, 768])
            cfg["depth_hidden_layers"] = trial.suggest_categorical("depth_hidden_layers", [4, 6, 8])
            cfg["depth_att_heads"] = trial.suggest_categorical("depth_att_heads", [4, 8])
            cfg["depth_intermediate_size"] = trial.suggest_categorical("depth_intermediate_size", [768, 1024, 1536, 2048, 3072])
            # cfg["depth_patch_size"] = trial.suggest_categorical("depth_patch_size", [16])
        
            print("SAMPLED THE FOLLOWING VALUES")
            for k in ["fusion_dim", "class_learning_rate", "pretrained_learning_rate", "new_learning_rate", "depth_hidden_dim", "depth_hidden_layers", "depth_att_heads", "depth_intermediate_size"]:
                print(f"{k}={cfg[k]}")
            print("\n")
        elif cfg["modalities"] == ["skeleton"]:
            # we will optimize [rgb + depth] first and use batch_size, fusion_dim, and class_learning rate from that study's results
            # cfg["batch_size"] = trial.suggest_categorical("batch_size", [])
            # cfg["fusion_dim"] = trial.suggest_categorical("fusion_dim", [])

            # cfg["class_learning_rate"] = trial.suggest_float("class_learning_rate", ,log=True)
            cfg["skel_learning_rate"] = trial.suggest_float("skel_learning_rate", 5e-5, 5e-4,log=True)
            
            cfg["bert_hidden_layers"] = trial.suggest_categorical("bert_hidden_layers", [4, 6, 8])
            cfg["bert_hidden_dim"] = trial.suggest_categorical("bert_hidden_dim", [256, 384, 512])
            cfg["bert_att_heads"] = trial.suggest_categorical("bert_att_heads", [4, 8])
            cfg["bert_intermediate_size"] = trial.suggest_categorical("bert_intermediate_size", [512, 768, 1024, 1536, 2048])
            print("SAMPLED THE FOLLOWING VALUES")
            for k in ["skel_learning_rate", "bert_hidden_layers", "bert_hidden_dim", "bert_att_heads", "bert_intermediate_size"]:
                print(f"{k}={cfg[k]}")
            print("\n")

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_acc")
        val_acc = train(cfg, trial, limit_train_batches=0.25, additional_callbacks=[pruning_callback])
        return val_acc

    assert config.endswith(".yaml")
    config_str = config.replace("/", "-")[:-5]
    study_name = f"slr_opt_config={config_str}"
    storage_db = f"sqlite:////home/hatch5o6/groups/grp_asl_classification/nobackup/archive/SLR/sqlite_dbs/{study_name}.db"
    if RESUME == False:
        if os.path.exists(storage_db):
            print("RESUME == False, deleting storage_db", storage_db)
            os.remove(storage_db)
        assert not os.path.exists(storage_db)
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=200,
            interval_steps=100
        ),
        study_name=study_name,
        storage=storage_db,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials, timeout=172800)
    print("##### BEST PARAMS #####")
    print(study.best_params)
    print("##### BEST VALUE #####")
    print(study.best_value)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_trials", type=int)
    parser.add_argument("-c", "--config")
    parser.add_argument("--RESUME", action="store_true", default=False, help="If passed, will resume the study if the db file already exists. If not passed, will delete the existing db and study and begin anew.")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}=`{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    print("###########################")
    print("# optimize_hyperparams.py #")
    print("###########################")
    args = get_args()
    main(config=args.config, n_trials=args.n_trials, RESUME=args.RESUME)


