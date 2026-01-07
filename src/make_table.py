import json
import pandas as pd
import argparse
import os


def main(models_dir):
    data = {"Model":[], "Acc": [], "Macro-F1": []}
    models_to_test = ["rgb", "rgb+d", "s_OG", "rgb+s", "rgb+d+s", "s_jpFalse", "rgb+s_jpFalse", "rgb+d+s_jpFalse"]
    # for d in os.listdir(models_dir):
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
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.models_dir)
