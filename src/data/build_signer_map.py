"""Build signer mapping files for signer-balanced sampling.

Reads original annotation CSVs (which contain Participant ID) and outputs
a JSON mapping {video_filename: signer_id}.

Usage:
    python src/data/build_signer_map.py --dataset asl_citizen
    python src/data/build_signer_map.py --dataset multilingual
"""
import argparse
import csv
import json
import os


ARCHIVE = os.path.expanduser(
    "~/groups/grp_asl_classification/nobackup/archive"
)


def build_asl_citizen_map():
    """Build signer map from ASL Citizen split CSVs."""
    signer_map = {}
    for split in ("train", "val", "test"):
        csv_path = os.path.join(ARCHIVE, "ASL", "ASL_Citizen", "splits", f"{split}.csv")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_file = row["Video file"]
                participant = row["Participant ID"]
                signer_map[video_file] = participant
    return signer_map


def build_autsl_map():
    """Build signer map for AUTSL from directory structure.

    AUTSL videos are named like signer{N}_{sample_id}.mp4
    """
    signer_map = {}
    for split_dir in ("train/train", "val/val", "test/test"):
        video_dir = os.path.join(ARCHIVE, "AUTSL", split_dir)
        if not os.path.isdir(video_dir):
            continue
        for f in os.listdir(video_dir):
            if f.endswith("_color.mp4"):
                # Format: signerN_sampleID_color.mp4
                parts = f.split("_")
                signer = parts[0]  # "signerN"
                signer_map[f] = signer
    return signer_map


def build_gsl_map():
    """Build signer map for GSL from video filenames.

    GSL videos are named like: {scenario}{N}_signer{M}_rep{R}_glosses_glosses{NNNN}.mp4
    """
    signer_map = {}
    video_dir = os.path.join(ARCHIVE, "GSL", "videos")
    if os.path.isdir(video_dir):
        for f in os.listdir(video_dir):
            if f.endswith(".mp4"):
                # Extract signer from filename
                parts = f.split("_")
                for p in parts:
                    if p.startswith("signer"):
                        signer_map[f] = p
                        break
                else:
                    signer_map[f] = "unknown"
    return signer_map


def build_multilingual_map():
    """Combine all dataset signer maps with dataset prefix to avoid collisions."""
    combined = {}

    asl_map = build_asl_citizen_map()
    for video, signer in asl_map.items():
        combined[video] = f"asl_{signer}"

    autsl_map = build_autsl_map()
    for video, signer in autsl_map.items():
        combined[video] = f"autsl_{signer}"

    gsl_map = build_gsl_map()
    for video, signer in gsl_map.items():
        combined[video] = f"gsl_{signer}"

    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["asl_citizen", "autsl", "gsl", "multilingual"])
    args = parser.parse_args()

    if args.dataset == "asl_citizen":
        signer_map = build_asl_citizen_map()
        out_path = "data/asl_citizen/signer_map.json"
    elif args.dataset == "autsl":
        signer_map = build_autsl_map()
        out_path = "data/signer_map_autsl.json"
    elif args.dataset == "gsl":
        signer_map = build_gsl_map()
        out_path = "data/gsl/signer_map.json"
    elif args.dataset == "multilingual":
        signer_map = build_multilingual_map()
        out_path = "data/multilingual/signer_map.json"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(signer_map, f)

    print(f"Wrote {len(signer_map)} entries to {out_path}")

    # Print summary
    from collections import Counter
    signer_counts = Counter(signer_map.values())
    print(f"  {len(signer_counts)} unique signers")
    counts = sorted(signer_counts.values())
    print(f"  Samples per signer: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")


if __name__ == "__main__":
    main()
