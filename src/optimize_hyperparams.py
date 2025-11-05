import argparse
import optuna
from train import train   # assuming train() returns best val_acc
from train import read_config
import copy


def main(config, n_trials):
    base_cfg =  read_config(config)
    def objective(trial):
        cfg = copy.deepcopy(base_cfg)

        # parameters
        cfg["pretrained_learning_rate"] = trial.suggest_float("pretrained_learning_rate", ,log=True)
        cfg["new_learning_rate"] = trial.suggest_float("new_learning_rate", ,log=True)
        cfg["batch_size"] = trial.suggest_categorical("batch_size", [])
        cfg["fusion_dim"] = trial.suggest_categorical("fusion_dim", [])
        cfg["depth_image_size"] = trial.suggest_categorical("depth_image_size", [])
        cfg["depth_hidden_dim"] = trial.suggest_categorical("depth_hidden_dim", [])
        cfg["depth_hidden_layers"] = trial.suggest_categorical("depth_hidden_layers", [])
        cfg["depth_att_heads"] = trial.suggest_categorical("depth_att_heads", [])
        cfg["depth_intermediate_size"] = trial.suggest_categorical("depth_intermediate_size", [])
        cfg["depth_patch_size"] = trial.suggest_categorical("depth_patch_size", [])
        cfg["bert_hidden_layers"] = trial.suggest_categorical("bert_hidden_layers", [])
        cfg["bert_hidden_dim"] = trial.suggest_categorical("bert_hidden_dim", [])
        cfg["bert_att_heads"] = trial.suggest_categorical("bert_att_heads", [])
        cfg["bert_intermediate_size"] = trial.suggest_categorical("bert_intermediate_size", [])


        val_acc = train(cfg)
        return val_acc

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    print("##### BEST PARAMS #####")
    print(study.best_params)
    print("##### BEST VALUE #####")
    print(study.best_value)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_trials")
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


