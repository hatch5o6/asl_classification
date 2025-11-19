import argparse
import optuna
from train import train   # assuming train() returns best val_acc
from train import read_config
import copy

# We're going to optimize RGB + D, and Skel separately

def main(config, n_trials):
    base_cfg =  read_config(config)
    def objective(trial):
        cfg = copy.deepcopy(base_cfg)

        assert cfg["modalities"] in [["rgb", "depth"], ["skeleton"]]
        if cfg["modalities"] == ["rgb", "depth"]:

            cfg["batch_size"] = trial.suggest_categorical("batch_size", [16, 24, 32])
            cfg["fusion_dim"] = trial.suggest_categorical("fusion_dim", [256, 384, 512, 768])

            cfg["class_learning_rate"] = trial.suggest_float("class_learning_rate", 5e-5, 5e-4,log=True)
            cfg["pretrained_learning_rate"] = trial.suggest_float("pretrained_learning_rate", 2e-5, 1e-4,log=True)
            cfg["new_learning_rate"] = trial.suggest_float("new_learning_rate", 5e-5, 5e-4,log=True)
            
            cfg["depth_image_size"] = trial.suggest_categorical("depth_image_size", [112, 168, 224])
            cfg["depth_hidden_dim"] = trial.suggest_categorical("depth_hidden_dim", [384, 512, 768])
            cfg["depth_hidden_layers"] = trial.suggest_categorical("depth_hidden_layers", [4, 6, 8])
            cfg["depth_att_heads"] = trial.suggest_categorical("depth_att_heads", [4, 8])
            cfg["depth_intermediate_size"] = trial.suggest_categorical("depth_intermediate_size", [768, 1024, 1536, 2048, 3072])
            # cfg["depth_patch_size"] = trial.suggest_categorical("depth_patch_size", [16])
        
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

        val_acc = train(cfg, trial)
        return val_acc

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=200,
            interval_steps=100
        ),
        study_name="slr_opt",
        storage="sqlite:///slr_opt.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials, timeout=7200)
    print("##### BEST PARAMS #####")
    print(study.best_params)
    print("##### BEST VALUE #####")
    print(study.best_value)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_trials", type=int)
    parser.add_argument("-c", "--config")
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
    main(config=args.config, n_trials=args.n_trials)


