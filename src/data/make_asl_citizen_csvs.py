"""
Converts ASL Citizen split CSVs into the pipeline format:
    rgb_path, depth_path, skel_path, label

ASL Citizen CSV format (ASL_Citizen/splits/{train,val,test}.csv):
    Participant ID, Video file, Gloss, ASL-LEX Code
    (header row is present; columns accessed by index)

The gloss -> integer label mapping is built from the training set only,
sorted alphabetically, and also saved as a class ID CSV for reference.

Usage:
    python src/data/make_asl_citizen_csvs.py \
        --data_dir /home/$USER/groups/grp_asl_classification/nobackup/archive/ASL \
        --out_dir data/asl_citizen

    Outputs:
        data/asl_citizen/train.csv
        data/asl_citizen/val.csv
        data/asl_citizen/test.csv
        data/asl_citizen/class_ids.csv   <- gloss, class_id mapping
"""

import argparse
import csv
import os
from pathlib import Path


def read_asl_citizen_csv(csv_path):
    """
    Returns list of (user, filename, gloss) tuples.
    Skips the header row. Columns: [user, filename, gloss, ...]
    """
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            user, filename, gloss = row[0], row[1], row[2].strip()
            rows.append((user, filename, gloss))
    return rows


def build_gloss_dict(train_rows):
    """Sorts glosses alphabetically from the training set -> 0-indexed int IDs."""
    glosses = sorted(set(gloss for _, _, gloss in train_rows))
    return {gloss: i for i, gloss in enumerate(glosses)}


def make_csv(rows, gloss_dict, videos_dir, skel_dir, out_path, split_name):
    missing_video = 0
    missing_skel = 0
    skipped_gloss = 0
    written = 0

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "depth_path", "skel_path", "label"])

        for user, filename, gloss in rows:
            if gloss not in gloss_dict:
                # Gloss in val/test not seen in training - skip
                skipped_gloss += 1
                continue

            # Video path: ASL Citizen filenames may or may not include a subdir.
            # Try both: videos/filename and videos/basename(filename)
            video_path = os.path.join(videos_dir, filename)
            if not os.path.exists(video_path):
                video_path = os.path.join(videos_dir, os.path.basename(filename))

            if not os.path.exists(video_path):
                missing_video += 1
                continue

            # Skeleton path: {stem}_landmarks.npy
            stem = Path(os.path.basename(filename)).stem
            skel_path = os.path.join(skel_dir, f"{stem}_landmarks.npy")

            # Allow missing skeletons (they may not be extracted yet)
            skel_path_val = skel_path if os.path.exists(skel_path) else ""
            if not os.path.exists(skel_path):
                missing_skel += 1

            label = gloss_dict[gloss]
            writer.writerow([video_path, "", skel_path_val, label])
            written += 1

    print(f"  [{split_name}] written={written}, missing_video={missing_video}, "
          f"missing_skel={missing_skel}, skipped_gloss={skipped_gloss}")
    return written


def save_class_ids(gloss_dict, out_path):
    # Header must match ("ClassId", "TR", "EN") expected by lightning_asl.py and train.py.
    # ASL Citizen has no Turkish translations, so gloss is used for both TR and EN.
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ClassId", "TR", "EN"])
        for gloss, cid in sorted(gloss_dict.items(), key=lambda x: x[1]):
            writer.writerow([cid, gloss, gloss])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Root of ASL Citizen archive, e.g. .../archive/ASL. "
             "Must contain ASL_Citizen/splits/{train,val,test}.csv, videos/, skel/."
    )
    parser.add_argument(
        "--out_dir", default="data/asl_citizen",
        help="Output directory for pipeline CSVs (default: data/asl_citizen)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data_dir = args.data_dir
    splits_dir = os.path.join(data_dir, "ASL_Citizen", "splits")
    videos_dir = os.path.join(data_dir, "videos")
    skel_dir = os.path.join(data_dir, "skel")
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # Validate expected paths exist
    for p, label in [(splits_dir, "ASL_Citizen/splits/"), (videos_dir, "videos/"), (skel_dir, "skel/")]:
        if not os.path.exists(p):
            print(f"WARNING: {label} not found at {p}")

    for split in ("train", "val", "test"):
        csv_path = os.path.join(splits_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {split}.csv not found at {csv_path}, skipping.")

    # Read all splits
    train_rows = read_asl_citizen_csv(os.path.join(splits_dir, "train.csv"))
    val_rows   = read_asl_citizen_csv(os.path.join(splits_dir, "val.csv"))
    test_rows  = read_asl_citizen_csv(os.path.join(splits_dir, "test.csv"))

    print(f"Rows: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    # Build label mapping from training set
    gloss_dict = build_gloss_dict(train_rows)
    print(f"Unique glosses (from train): {len(gloss_dict)}")

    # Save class ID reference file
    class_ids_path = os.path.join(out_dir, "class_ids.csv")
    save_class_ids(gloss_dict, class_ids_path)
    print(f"Saved class IDs -> {class_ids_path}")

    # Generate pipeline CSVs
    print("Generating pipeline CSVs...")
    for split, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        out_path = os.path.join(out_dir, f"{split}.csv")
        make_csv(rows, gloss_dict, videos_dir, skel_dir, out_path, split)
        print(f"  Saved -> {out_path}")

    print("\nDone.")
    print("NOTE: If missing_skel > 0, run pose extraction first, then re-run this script.")
