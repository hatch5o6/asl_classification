import json
import pandas as pd
import argparse
import os


def main(models_dir, models_to_test=None):
    data = {"Model":[], "Acc": [], "Macro-F1": []}
    # If no models specified, use all directories that have predictions
    if models_to_test is None:
        models_to_test = []
        for d in os.listdir(models_dir):
            d_path = os.path.join(models_dir, d)
            preds_path = os.path.join(d_path, "predictions")
            if os.path.isdir(d_path) and os.path.exists(preds_path):
                models_to_test.append(d)
        models_to_test = sorted(models_to_test)
        print(f"Found {len(models_to_test)} models with predictions: {models_to_test}")
    for d in models_to_test:
        d_path = os.path.join(models_dir, d)
        if not os.path.exists(d_path):
            print(f"`{d_path}` does not exist. Skipping.")
            continue
        preds_dir = os.path.join(d_path, "predictions")
        pred_files = [f for f in os.listdir(preds_dir) if f.endswith("metrics.json")]
        for pf in pred_files:
            pf_path = os.path.join(preds_dir, pf)
            metrics = read_json(pf_path)
            name = f"{d}::{pf}"
            acc = metrics["accuracy"]
            my_acc = metrics["my_accuracy"]
            assert round(acc, 6) == round(my_acc, 6)
            print(f"ASSERT ACC ~ MY ACC: {name}")
            print(f"\tacc {acc} ~ my_acc {my_acc}")
            print(f"\t(ROUNDED) acc {round(acc, 6)} ~ my_acc {round(my_acc, 6)}")
            print("\n")
            macro_f1 = metrics["macro avg"]["f1-score"]
            data["Model"].append(name)
            data["Acc"].append(acc)
            data["Macro-F1"].append(macro_f1)
    df = pd.DataFrame(data)
    latex_out_f = os.path.join(models_dir, "latex_table.txt")
    df.to_latex(buf=latex_out_f)

def read_json(f):
    with open(f) as inf:
        data = json.load(inf)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", "-m", default="/home/hatch5o6/groups/grp_asl_classification/nobackup/archive/SLR/models")
    parser.add_argument("--models", "-l", type=str, default=None,
                        help="Comma-separated list of model names to include (default: all models with predictions)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    models_list = args.models.split(",") if args.models else None
    main(args.models_dir, models_to_test=models_list)
