"""
Quick summary of encoder comparison results.

Reads completed runs from the model directory and prints a table of
val_acc (from checkpoint name) and test_acc (from predictions/metrics.json).

Usage:
    python scripts/show_encoder_results.py               # all results
    python scripts/show_encoder_results.py --lang autsl  # one language
    python scripts/show_encoder_results.py --enc gru     # one encoder
    python scripts/show_encoder_results.py --csv         # output as CSV
"""

import argparse
import json
import os
from pathlib import Path

MODEL_ROOT = Path("/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/encoder_comparison")
BERT_ROOT  = Path("/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models")

LANGUAGES = ["autsl", "asl_citizen", "gsl", "multilingual"]
ENCODERS  = ["bert", "gru", "stgcn", "spoter"]


def resolve_save_dir(lang: str, enc: str, sel_type: str, subdir: str, k: int) -> Path:
    """BERT results live under an older directory layout. AUTSL was the
    original default and sits at the root of models/ rather than under
    models/autsl/ for iterative and topk runs."""
    if enc != "bert":
        return MODEL_ROOT / lang / enc / subdir

    if sel_type == "full":   # K=543 full skeleton
        if lang == "autsl":
            return BERT_ROOT / "autsl" / "generalization" / "regularization"
        return BERT_ROOT / lang / "s"

    if sel_type == "iterative":
        name = f"iterative_{k}"
        if lang == "autsl":
            return BERT_ROOT / "informed_selection" / name
        return BERT_ROOT / lang / "informed_selection" / name

    if sel_type == "topk":
        name = f"topk_{k}"
        if lang == "autsl":
            return BERT_ROOT / "informed_selection" / name
        return BERT_ROOT / lang / "informed_selection" / name

    # No BERT results for the random baselines
    return BERT_ROOT / "__bert_unavailable__"

# All selection types in display order
SELECTIONS = (
    [("full",      "k543",  543, "")] +
    [("iterative", f"iter_{k}", k, "") for k in [10, 24, 48, 100, 270]] +
    [("topk",      f"k{k}",  k, "")  for k in [10, 24, 48, 100, 270]] +
    [("random",    f"random_{k}_draw{d}", k, f"d{d}") for k in [10, 24, 48, 100, 270] for d in [0, 1, 2]]
)


def get_val_acc(save_dir: Path):
    """Read best val_acc from the top checkpoint filename."""
    ckpt_dir = save_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    for f in ckpt_dir.iterdir():
        if f.suffix == ".ckpt" and "val_acc=" in f.name and f.stem != "last":
            try:
                return float(f.stem.split("val_acc=")[-1])
            except ValueError:
                pass
    return None


def get_test_acc(save_dir: Path):
    """Read test accuracy from predictions/metrics.json."""
    pred_dir = save_dir / "predictions"
    if not pred_dir.exists():
        return None
    for f in pred_dir.glob("*.metrics.json"):
        try:
            with open(f) as fh:
                d = json.load(fh)
            return d.get("my_accuracy")
        except Exception:
            pass
    return None


def fmt(val):
    if val is None:
        return "—"
    return f"{val*100:.1f}%"


def collect_results(lang_filter=None, enc_filter=None):
    rows = []
    for lang in LANGUAGES:
        if lang_filter and lang != lang_filter:
            continue
        for enc in ENCODERS:
            if enc_filter and enc != enc_filter:
                continue
            for sel_type, subdir, k, draw in SELECTIONS:
                save_dir = resolve_save_dir(lang, enc, sel_type, subdir, k)

                val_acc  = get_val_acc(save_dir)
                test_acc = get_test_acc(save_dir)

                if test_acc is not None:
                    status = "DONE"
                elif val_acc is not None:
                    status = "TRAINED (no test)"
                elif (save_dir / "checkpoints").exists():
                    status = "RUNNING"
                else:
                    status = "not started"

                rows.append({
                    "lang": lang,
                    "enc": enc,
                    "type": sel_type,
                    "K": k,
                    "draw": draw,
                    "subdir": subdir,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "status": status,
                })
    return rows


def print_table(rows):
    # Group by lang + enc for clean display
    current_group = None
    header = f"{'Lang':<12} {'Enc':<7} {'Type':<10} {'K':>4} {'':>3}  {'Val':>7}  {'Test':>7}  {'Status'}"
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for r in rows:
        group = (r["lang"], r["enc"])
        if group != current_group:
            if current_group is not None:
                print()
            current_group = group

        print(f"{r['lang']:<12} {r['enc']:<7} {r['type']:<10} {r['K']:>4} {r['draw']:<3}  "
              f"{fmt(r['val_acc']):>7}  {fmt(r['test_acc']):>7}  {r['status']}")

    print(sep)


def print_csv(rows):
    print("lang,encoder,type,K,val_acc,test_acc,status")
    for r in rows:
        val  = f"{r['val_acc']*100:.2f}" if r["val_acc"]  is not None else ""
        test = f"{r['test_acc']*100:.2f}" if r["test_acc"] is not None else ""
        print(f"{r['lang']},{r['enc']},{r['type']},{r['K']},{r['draw']},{val},{test},{r['status']}")


def print_summary(rows):
    done        = sum(1 for r in rows if r["status"] == "DONE")
    trained     = sum(1 for r in rows if r["status"] == "TRAINED (no test)")
    running     = sum(1 for r in rows if r["status"] == "RUNNING")
    not_started = sum(1 for r in rows if r["status"] == "not started")

    print(f"\nDONE: {done}  |  TRAINED (no test): {trained}  |  "
          f"RUNNING: {running}  |  not started: {not_started}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=LANGUAGES, help="Filter by language")
    parser.add_argument("--enc",  choices=ENCODERS,  help="Filter by encoder")
    parser.add_argument("--csv",  action="store_true", help="Output as CSV")
    parser.add_argument("--all",  action="store_true", help="Show not-started rows too")
    args = parser.parse_args()

    rows = collect_results(lang_filter=args.lang, enc_filter=args.enc)

    if not args.all:
        rows = [r for r in rows if r["status"] != "not started"]

    if args.csv:
        print_csv(rows)
    else:
        print_table(rows)
        print_summary(rows)
