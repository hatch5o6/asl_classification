#!/usr/bin/env python3
"""Generate per-language test configs for multilingual models.

For each model config, creates three variants with:
  - test_csv pointing to the per-language test set
  - save pointing to a per-language subdirectory to avoid overwriting predictions
  - test_checkpoint set to the best checkpoint discovered from checkpoints/

Per-language save dirs: {original_save}/per_language/{lang}/
  Each gets a checkpoints/ subdir (required by train.py's assertion).

Usage:
    # Generate for all models (except iterative_10 which may still be training):
    python src/data/make_per_language_configs.py

    # Generate for iterative_10 once training finishes:
    python src/data/make_per_language_configs.py --iterative10
"""

import argparse
import os
import re

ARCHIVE = os.path.expanduser(
    "~/groups/grp_asl_classification/nobackup/archive/SLR/models"
)

# (config_path, original_save_in_config)
# best checkpoint is discovered dynamically from the checkpoints/ dir
MODELS = [
    (
        "configs/multilingual/s.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/s",
    ),
    (
        "configs/multilingual/informed_selection/topk_270.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/topk_270",
    ),
    (
        "configs/multilingual/informed_selection/topk_100.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/topk_100",
    ),
    (
        "configs/multilingual/informed_selection/topk_48.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/topk_48",
    ),
    (
        "configs/multilingual/informed_selection/topk_24.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/topk_24",
    ),
    (
        "configs/multilingual/informed_selection/topk_10.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/topk_10",
    ),
    (
        "configs/multilingual/informed_selection/iterative_100.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/iterative_100",
    ),
    (
        "configs/multilingual/informed_selection/iterative_48.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/iterative_48",
    ),
    (
        "configs/multilingual/informed_selection/iterative_24.yaml",
        "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/iterative_24",
    ),
]

ITERATIVE_10 = (
    "configs/multilingual/informed_selection/iterative_10.yaml",
    "/home/ccoulson/groups/grp_asl_classification/nobackup/archive/SLR/models/multilingual/informed_selection/iterative_10",
)

LANG_TEST_CSVS = {
    "autsl": "data/multilingual/test_autsl.csv",
    "asl":   "data/multilingual/test_asl.csv",
    "gsl":   "data/multilingual/test_gsl.csv",
}

ORIGINAL_TEST_CSV = "data/multilingual/test.csv"


def find_best_checkpoint(save_dir):
    """Return the path of the highest val_acc checkpoint, or None."""
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    best_ckpt, best_val = None, None
    for f in os.listdir(ckpt_dir):
        m = re.search(r"val_acc=(\d+\.\d+)", f)
        if m:
            v = float(m.group(1))
            if best_val is None or v > best_val:
                best_val = v
                best_ckpt = os.path.join(ckpt_dir, f)
    return best_ckpt


def generate_configs(models):
    for config_path, orig_save in models:
        best_ckpt = find_best_checkpoint(orig_save)
        if best_ckpt is None:
            print(f"WARNING: No checkpoint found in {orig_save}/checkpoints — skipping {config_path}")
            continue

        with open(config_path) as f:
            content = f.read()

        assert ORIGINAL_TEST_CSV in content, \
            f"Expected '{ORIGINAL_TEST_CSV}' in {config_path}"
        assert f"save: {orig_save}" in content, \
            f"Expected 'save: {orig_save}' in {config_path}"

        config_dir = os.path.dirname(config_path)
        stem = os.path.splitext(os.path.basename(config_path))[0]

        for lang, test_csv in LANG_TEST_CSVS.items():
            per_lang_save = f"{orig_save}/per_language/{lang}"

            new_content = content
            new_content = new_content.replace(
                f"test_csv: {ORIGINAL_TEST_CSV}",
                f"test_csv: {test_csv}",
            )
            new_content = new_content.replace(
                f"save: {orig_save}",
                f"save: {per_lang_save}",
            )
            new_content = new_content.replace(
                "test_checkpoint: null",
                f"test_checkpoint: {best_ckpt}",
            )

            out_path = os.path.join(config_dir, f"{stem}_{lang}.yaml")
            with open(out_path, "w") as f:
                f.write(new_content)

            # Create per-language save dir and checkpoints subdir
            ckpt_subdir = os.path.join(per_lang_save, "checkpoints")
            os.makedirs(ckpt_subdir, exist_ok=True)

            print(f"Created {out_path}")
            print(f"  checkpoint: {os.path.basename(best_ckpt)}")
            print(f"  save: {per_lang_save}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterative10", action="store_true",
                        help="Generate configs for iterative_10 (use after training finishes)")
    args = parser.parse_args()

    if args.iterative10:
        models = [ITERATIVE_10]
        print("Generating per-language configs for iterative_10...")
    else:
        models = MODELS
        print(f"Generating per-language configs for {len(models)} models...")

    generate_configs(models)
    n = len(models) * len(LANG_TEST_CSVS)
    print(f"\nDone — created/updated {n} per-language configs.")

    if args.iterative10:
        print("\nTo submit iterative_10 per-language tests:")
        print("  sbatch sbatch/test_informed_selection.sh configs/multilingual/informed_selection/iterative_10_autsl.yaml")
        print("  sbatch sbatch/test_informed_selection.sh configs/multilingual/informed_selection/iterative_10_asl.yaml")
        print("  sbatch sbatch/test_informed_selection.sh configs/multilingual/informed_selection/iterative_10_gsl.yaml")
        print("\nAnd the combined multilingual test (if not already submitted):")
        print("  sbatch sbatch/test_informed_selection.sh configs/multilingual/informed_selection/iterative_10.yaml")


if __name__ == "__main__":
    main()
