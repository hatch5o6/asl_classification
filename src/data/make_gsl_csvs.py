"""
Converts GSL isolated annotation files into the pipeline format:
    rgb_path, depth_path, skel_path, label

GSL annotation format (pipe-delimited, no header):
    police1_signer1_rep1_glosses/glosses0000|ΓΕΙΑ
    police1_signer1_rep1_glosses/glosses0001|ΕΓΩ(1)
    ...

Video path convention (after pack_gsl_frames.py):
    videos/{parent}_{child}.mp4
where annotation path "parent/child" -> "{parent}_{child}.mp4" (replace '/' with '_').

Gloss -> integer label mapping is built from the training set only,
sorted alphabetically, and also saved as class_ids.csv.

Usage:
    python src/data/make_gsl_csvs.py \
        --data_dir /home/$USER/groups/grp_asl_classification/nobackup/archive/GSL \
        --out_dir data/gsl

    Outputs:
        data/gsl/train.csv
        data/gsl/val.csv
        data/gsl/test.csv
        data/gsl/class_ids.csv   <- ClassId, gloss mapping
"""

import argparse
import csv
import os
from pathlib import Path


def read_gsl_annotation(ann_path):
    """
    Reads a pipe-delimited GSL annotation file (no header).
    Returns list of (clip_path_str, gloss) tuples.
    clip_path_str is like "police1_signer1_rep1_glosses/glosses0000".
    """
    rows = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            clip_path = parts[0].strip()
            gloss = parts[1].strip()
            rows.append((clip_path, gloss))
    return rows


def build_gloss_dict(train_rows):
    """Alphabetically sorted glosses from training set -> 0-indexed int IDs."""
    glosses = sorted(set(gloss for _, gloss in train_rows))
    return {gloss: i for i, gloss in enumerate(glosses)}


def annotation_path_to_mp4_stem(clip_path_str):
    """
    "police1_signer1_rep1_glosses/glosses0000"
    -> "police1_signer1_rep1_glosses_glosses0000"
    """
    return clip_path_str.replace("/", "_").replace(os.sep, "_")


def make_csv(rows, gloss_dict, videos_dir, skel_dir, out_path, split_name):
    missing_video = 0
    missing_skel = 0
    skipped_gloss = 0
    written = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rgb_path", "depth_path", "skel_path", "label"])

        for clip_path_str, gloss in rows:
            if gloss not in gloss_dict:
                # Gloss not seen in training set
                skipped_gloss += 1
                continue

            stem = annotation_path_to_mp4_stem(clip_path_str)
            video_path = os.path.join(videos_dir, stem + ".mp4")

            if not os.path.exists(video_path):
                missing_video += 1
                continue

            skel_path = os.path.join(skel_dir, f"{stem}_landmarks.npy")
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
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ClassId", "TR", "EN"])
        for gloss, cid in sorted(gloss_dict.items(), key=lambda x: x[1]):
            writer.writerow([cid, gloss, gloss])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", required=True,
        help="Root of GSL archive, e.g. .../archive/GSL. "
             "Must contain videos/, skel/, annotations/."
    )
    parser.add_argument(
        "--out_dir", default="data/gsl",
        help="Output directory for pipeline CSVs (default: data/gsl)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    data_dir = args.data_dir
    ann_dir = os.path.join(data_dir, "annotations")
    videos_dir = os.path.join(data_dir, "videos")
    skel_dir = os.path.join(data_dir, "skel")
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    # Validate expected directories
    for p, label in [(ann_dir, "annotations/"), (videos_dir, "videos/"), (skel_dir, "skel/")]:
        if not os.path.exists(p):
            print(f"WARNING: {label} not found at {p}")

    train_path = os.path.join(ann_dir, "train_greek_iso.csv")
    val_path   = os.path.join(ann_dir, "dev_greek_iso.csv")
    test_path  = os.path.join(ann_dir, "test_greek_iso.csv")

    for p in (train_path, val_path, test_path):
        if not os.path.exists(p):
            print(f"WARNING: Annotation file not found: {p}")

    # Read all splits
    train_rows = read_gsl_annotation(train_path)
    val_rows   = read_gsl_annotation(val_path)
    test_rows  = read_gsl_annotation(test_path)

    print(f"Rows: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    # Build label mapping from training set only
    gloss_dict = build_gloss_dict(train_rows)
    print(f"Unique glosses (from train): {len(gloss_dict)}")

    # Save class ID reference
    class_ids_path = os.path.join(out_dir, "class_ids.csv")
    save_class_ids(gloss_dict, class_ids_path)
    print(f"Saved class IDs -> {class_ids_path}")

    # Generate pipeline CSVs
    print("Generating pipeline CSVs...")
    for split, rows, out_name in [
        ("train", train_rows, "train.csv"),
        ("val",   val_rows,   "val.csv"),
        ("test",  test_rows,  "test.csv"),
    ]:
        out_path = os.path.join(out_dir, out_name)
        make_csv(rows, gloss_dict, videos_dir, skel_dir, out_path, split)
        print(f"  Saved -> {out_path}")

    print("\nDone.")
    print("NOTE: If missing_video > 0, run pack_frames_to_mp4.gsl.sh first.")
    print("NOTE: If missing_skel > 0, run extract_pose_points.gsl.sh first, then re-run this script.")
