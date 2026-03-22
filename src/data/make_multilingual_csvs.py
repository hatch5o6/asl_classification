"""
Build combined multilingual train/val/test CSVs from AUTSL, ASL Citizen, and GSL.

Each class label is prefixed with its language code so the model can distinguish
between signs from different languages that share the same original integer ID.

Output class_ids.csv format (ClassId, TR, EN):
  ClassId=0    TR=autsl_abla    EN=autsl_sister
  ClassId=226  TR=asl_1DOLLAR   EN=asl_1DOLLAR
  ClassId=2957 TR=gsl_1         EN=gsl_1

Usage:
    python src/data/make_multilingual_csvs.py --out_dir data/multilingual
"""

import argparse
import csv
import os
import random


# Fixed language order and their source paths
LANGUAGES = [
    {
        "code": "autsl",
        "train_csv": "data/train.csv",
        "val_csv": "data/val.csv",
        "test_csv": "data/test.csv",
        "class_id_csv": "data/SignList_ClassId_TR_EN.csv",
    },
    {
        "code": "asl",
        "train_csv": "data/asl_citizen/train.csv",
        "val_csv": "data/asl_citizen/val.csv",
        "test_csv": "data/asl_citizen/test.csv",
        "class_id_csv": "data/asl_citizen/class_ids.csv",
    },
    {
        "code": "gsl",
        "train_csv": "data/gsl/train.csv",
        "val_csv": "data/gsl/val.csv",
        "test_csv": "data/gsl/test.csv",
        "class_id_csv": "data/gsl/class_ids.csv",
    },
]


def load_class_id_map(class_id_csv):
    """Load class_id_csv -> {original_int: (TR, EN)}."""
    mapping = {}
    with open(class_id_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["ClassId", "TR", "EN"], f"Unexpected header in {class_id_csv}"
    for row in rows[1:]:
        cid, tr, en = int(row[0]), row[1], row[2]
        mapping[cid] = (tr, en)
    return mapping


def load_csv_rows(csv_path):
    """Load pipeline CSV -> list of (rgb_path, depth_path, skel_path, label_int)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["rgb_path", "depth_path", "skel_path", "label"]
    return [(r[0], r[1], r[2], int(r[3])) for r in rows[1:]]


def build_global_class_map(languages):
    """
    Build global integer ID mapping across all languages.

    For each language, each original class gets a prefixed name:
        autsl_abla, asl_1DOLLAR, gsl_1, ...

    Returns:
        global_class_ids: {(lang_code, orig_int): global_int}
        class_id_rows: list of [global_int, prefixed_TR, prefixed_EN]
    """
    global_class_ids = {}
    class_id_rows = []
    next_id = 0

    for lang in languages:
        code = lang["code"]
        id_map = load_class_id_map(lang["class_id_csv"])
        for orig_int in sorted(id_map.keys()):
            tr, en = id_map[orig_int]
            global_class_ids[(code, orig_int)] = next_id
            class_id_rows.append([next_id, f"{code}_{tr}", f"{code}_{en}"])
            next_id += 1

    return global_class_ids, class_id_rows


def remap_and_combine(languages, split, global_class_ids):
    """Combine rows from all languages for a given split with remapped labels."""
    combined = []
    for lang in languages:
        code = lang["code"]
        csv_path = lang[f"{split}_csv"]
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        rows = load_csv_rows(csv_path)
        for rgb, depth, skel, orig_label in rows:
            global_label = global_class_ids[(code, orig_label)]
            combined.append((rgb, depth, skel, global_label))
    return combined


def write_csv(rows, out_path, shuffle=False, seed=42):
    """Write pipeline CSV rows to file."""
    if shuffle:
        rng = random.Random(seed)
        rows = list(rows)
        rng.shuffle(rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "depth_path", "skel_path", "label"])
        for row in rows:
            writer.writerow(row)


def write_class_ids(class_id_rows, out_path):
    """Write combined class_ids.csv."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ClassId", "TR", "EN"])
        for row in class_id_rows:
            writer.writerow(row)


def get_args():
    parser = argparse.ArgumentParser(
        description="Build multilingual pipeline CSVs from AUTSL, ASL Citizen, and GSL"
    )
    parser.add_argument(
        "--out_dir", default="data/multilingual",
        help="Output directory for multilingual CSVs (default: data/multilingual)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling train split (default: 42)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("MULTILINGUAL CSV BUILDER")
    print("=" * 60)

    # Build global class mapping
    print("\nBuilding global class mapping...")
    global_class_ids, class_id_rows = build_global_class_map(LANGUAGES)
    total_classes = len(class_id_rows)
    print(f"  Total classes: {total_classes}")
    for lang in LANGUAGES:
        code = lang["code"]
        count = sum(1 for (lc, _) in global_class_ids if lc == code)
        start = global_class_ids[(code, 0)]
        print(f"    {code}: {count} classes (global IDs {start}–{start + count - 1})")

    # Save class_ids.csv
    class_id_path = os.path.join(args.out_dir, "class_ids.csv")
    write_class_ids(class_id_rows, class_id_path)
    print(f"\nSaved class IDs -> {class_id_path}")

    # Build each split
    print("\nGenerating pipeline CSVs...")
    for split in ("train", "val", "test"):
        rows = remap_and_combine(LANGUAGES, split, global_class_ids)
        shuffle = (split == "train")
        out_path = os.path.join(args.out_dir, f"{split}.csv")
        write_csv(rows, out_path, shuffle=shuffle, seed=args.seed)
        lang_counts = {}
        for lang in LANGUAGES:
            code = lang["code"]
            csv_path = lang[f"{split}_csv"]
            if os.path.exists(csv_path):
                lang_counts[code] = len(load_csv_rows(csv_path))
        breakdown = ", ".join(f"{k}={v}" for k, v in lang_counts.items())
        print(f"  [{split}] written={len(rows)} ({breakdown}) -> {out_path}")

    print(f"\nDone. Output in {args.out_dir}/")
    print(f"  Total classes: {total_classes} "
          f"(autsl={sum(1 for (lc,_) in global_class_ids if lc=='autsl')}, "
          f"asl={sum(1 for (lc,_) in global_class_ids if lc=='asl')}, "
          f"gsl={sum(1 for (lc,_) in global_class_ids if lc=='gsl')})")