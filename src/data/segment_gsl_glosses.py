"""
Converts GSL sentence-level JPEG frame archives into isolated gloss MP4 clips.

The GSL zip files contain sentence-level clips (frame directories):
    {scenario}_{signer}_{rep}_sentences/sentences{NNNN}/frame_0001.jpg ...

This script reads each sentence from the zip WITHOUT extracting to the archive
(uses /tmp for temp frames to avoid Lustre inode quota issues), then writes
one MP4 per gloss to the output videos directory.

Gloss assignment: per-sentence frames are split equally among the glosses in
that sentence (standard approach; precise timing is not in the public release).

Output naming convention:
    {scenario}_{signer}_{rep}_glosses_glosses{NNNN:04d}.mp4
where NNNN is the cumulative gloss index across all sentences for that
signer_group, ordered by sentence index (sentences0000, sentences0001, ...).
This naming matches the slrzoo annotation files (train_greek_iso.csv, etc.).

Usage (process a single zip):
    python src/data/segment_gsl_glosses.py \
        --zip_path  .../archive/GSL/kep3.zip \
        --merged_csv .../archive/GSL/annotations/merged_continuous.csv \
        --output_dir .../archive/GSL/videos \
        --fps 25 \
        --num_workers 8

    # --delete_zip  flag removes the zip after processing (saves disk)
    # --force       flag re-processes clips that already have an MP4

Notes:
  - Run this on a login node or compute node; /tmp is local per-node
  - Chain: for each of kep3/kep4/kep5, run this script then optionally delete zip
  - Logs which sentence dirs were skipped (not present in zip or already done)
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_merged_continuous(csv_path: str) -> Dict[str, List[str]]:
    """
    Parse merged_continuous.csv.

    Format (pipe-delimited, no header):
        police1_signer1_rep1_sentences/sentences0000|sentence text|GLOSS1 GLOSS2 ...

    Returns:
        dict mapping sentence_key -> list of glosses
        e.g. "police1_signer1_rep1_sentences/sentences0000" -> ["ΓΕΙΑ", "ΕΓΩ(1)", ...]
    """
    sentence_glosses: Dict[str, List[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            sentence_key = parts[0].strip()
            gloss_str = parts[2].strip()
            glosses = [g.strip() for g in gloss_str.split() if g.strip()]
            if glosses:
                sentence_glosses[sentence_key] = glosses
    return sentence_glosses


def extract_signer_group(sentence_key: str) -> str:
    """
    "police1_signer1_rep1_sentences/sentences0000"
    -> "police1_signer1_rep1"
    """
    top_dir = sentence_key.split("/")[0]  # "police1_signer1_rep1_sentences"
    return top_dir.replace("_sentences", "")


def sentence_index(sentence_key: str) -> int:
    """
    "police1_signer1_rep1_sentences/sentences0042" -> 42
    """
    subdir = sentence_key.split("/")[1]  # "sentences0042"
    return int(subdir.replace("sentences", ""))


def compute_gloss_offsets(
    sentence_glosses: Dict[str, List[str]]
) -> Dict[str, int]:
    """
    For each sentence key, compute the cumulative gloss starting index within
    its signer_group (ordered by sentence index).

    Returns:
        dict mapping sentence_key -> global gloss start index within that signer group
    """
    by_group: Dict[str, List[Tuple[int, str, int]]] = defaultdict(list)
    for key, glosses in sentence_glosses.items():
        group = extract_signer_group(key)
        by_group[group].append((sentence_index(key), key, len(glosses)))

    offsets: Dict[str, int] = {}
    for group, entries in by_group.items():
        entries.sort(key=lambda x: x[0])
        cum = 0
        for _, key, n_glosses in entries:
            offsets[key] = cum
            cum += n_glosses
    return offsets


# ---------------------------------------------------------------------------
# Zip inspection
# ---------------------------------------------------------------------------

def list_sentence_dirs_in_zip(zf: zipfile.ZipFile) -> Dict[str, List[str]]:
    """
    Scan the zip for JPEG frame files and group them by sentence key.

    Returns:
        dict mapping sentence_key -> sorted list of zip member paths for that sentence
        e.g. "police1_signer1_rep1_sentences/sentences0000"
             -> ["police1_signer1_rep1_sentences/sentences0000/frame_0001.jpg", ...]
    """
    sentence_frames: Dict[str, List[str]] = defaultdict(list)
    for name in zf.namelist():
        lower = name.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            parts = name.split("/")
            if len(parts) >= 3:
                # parts[0] = "police1_signer1_rep1_sentences"
                # parts[1] = "sentences0000"
                sentence_key = f"{parts[0]}/{parts[1]}"
                sentence_frames[sentence_key].append(name)
    for key in sentence_frames:
        sentence_frames[key] = sorted(sentence_frames[key])
    return dict(sentence_frames)


# ---------------------------------------------------------------------------
# Core: extract one sentence from zip → write N gloss MP4s
# ---------------------------------------------------------------------------

def segment_sentence(
    zf: zipfile.ZipFile,
    sentence_key: str,
    frame_paths_in_zip: List[str],
    glosses: List[str],
    gloss_start_idx: int,
    output_dir: str,
    fps: float,
    force: bool,
    tmp_root: str,
) -> Tuple[int, int]:
    """
    Extract one sentence's frames from the zip to a temp dir, split equally
    among glosses, write one MP4 per gloss, clean up temp dir.

    Returns (n_written, n_skipped).
    """
    n_glosses = len(glosses)
    n_frames = len(frame_paths_in_zip)

    if n_frames == 0 or n_glosses == 0:
        return 0, 0

    signer_group = extract_signer_group(sentence_key)

    # Determine output paths for all N glosses
    output_paths = []
    for i in range(n_glosses):
        gloss_idx = gloss_start_idx + i
        mp4_name = f"{signer_group}_glosses_glosses{gloss_idx:04d}.mp4"
        output_paths.append(os.path.join(output_dir, mp4_name))

    # Check which gloss clips still need to be written
    todo_indices = [i for i in range(n_glosses) if not os.path.exists(output_paths[i]) or force]
    if not todo_indices:
        return 0, n_glosses  # all already done

    # Compute equal frame splits for each gloss
    split_sizes = [n_frames // n_glosses] * n_glosses
    for i in range(n_frames % n_glosses):
        split_sizes[i] += 1  # distribute remainder to first splits
    splits: List[List[str]] = []
    start = 0
    for sz in split_sizes:
        splits.append(frame_paths_in_zip[start : start + sz])
        start += sz

    # Extract frame files to a temp directory (local /tmp, not archive)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root, prefix="gsl_seg_")
    try:
        # Only extract frames needed by gloss clips we still need to write
        needed_frame_set = set()
        for i in todo_indices:
            needed_frame_set.update(splits[i])

        for zip_member in needed_frame_set:
            data = zf.read(zip_member)
            local_fname = os.path.basename(zip_member)
            local_path = os.path.join(tmp_dir, local_fname)
            with open(local_path, "wb") as f:
                f.write(data)

        written = 0
        for i in todo_indices:
            gloss_frames = [os.path.join(tmp_dir, os.path.basename(p)) for p in splits[i]]
            gloss_frames = [p for p in gloss_frames if os.path.exists(p)]
            if not gloss_frames:
                continue

            ok = _write_mp4_ffmpeg(gloss_frames, output_paths[i], fps)
            if ok:
                written += 1

        return written, n_glosses - len(todo_indices)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_mp4_ffmpeg(frame_paths: List[str], output_path: str, fps: float) -> bool:
    """Write a list of JPEG frame files to an MP4 using ffmpeg concat demuxer."""
    list_file = output_path + ".frames.txt"
    frame_duration = 1.0 / fps
    try:
        with open(list_file, "w") as f:
            for p in frame_paths:
                f.write(f"file '{p}'\n")
                f.write(f"duration {frame_duration:.8f}\n")
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c:v", "mpeg4",
                "-q:v", "3",
                "-loglevel", "error",
                output_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"  ✗ ffmpeg error for {os.path.basename(output_path)}: "
                  f"{result.stderr.decode().strip()[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  ✗ Exception for {os.path.basename(output_path)}: {e}")
        return False
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Segment GSL sentence zip into isolated gloss MP4 clips."
    )
    p.add_argument("--zip_path", required=True,
                   help="Path to one scenario zip (e.g. .../archive/GSL/kep3.zip)")
    p.add_argument("--merged_csv", required=True,
                   help="Path to merged_continuous.csv from supplementary.zip")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for gloss MP4 files (.../archive/GSL/videos)")
    p.add_argument("--fps", type=float, default=25.0,
                   help="Output video frame rate (default: 25)")
    p.add_argument("--tmp_dir", default="/tmp",
                   help="Root for temporary frame extraction (default: /tmp)")
    p.add_argument("--delete_zip", action="store_true",
                   help="Delete the zip file after processing")
    p.add_argument("--force", action="store_true",
                   help="Re-process clips that already have an output MP4")
    return p.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.zip_path):
        print(f"✗ Zip not found: {args.zip_path}")
        sys.exit(1)
    if not os.path.exists(args.merged_csv):
        print(f"✗ merged_continuous.csv not found: {args.merged_csv}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading merged_continuous.csv ...")
    sentence_glosses = parse_merged_continuous(args.merged_csv)
    print(f"  {len(sentence_glosses)} sentences loaded")

    print(f"Computing gloss offsets ...")
    gloss_offsets = compute_gloss_offsets(sentence_glosses)

    print(f"Scanning zip: {args.zip_path} ...")
    with zipfile.ZipFile(args.zip_path, "r") as zf:
        zip_sentences = list_sentence_dirs_in_zip(zf)
    print(f"  {len(zip_sentences)} sentence dirs found in zip")

    # Only process sentences that exist in both the zip and merged_continuous.csv
    to_process = {k: v for k, v in zip_sentences.items() if k in sentence_glosses}
    not_in_csv = len(zip_sentences) - len(to_process)
    if not_in_csv:
        print(f"  ({not_in_csv} sentence dirs not in merged_continuous.csv — skipped)")

    print(f"\nProcessing {len(to_process)} sentences from {os.path.basename(args.zip_path)} ...")
    total_written = 0
    total_skipped = 0

    with zipfile.ZipFile(args.zip_path, "r") as zf:
        for i, (sentence_key, frame_paths) in enumerate(sorted(to_process.items()), 1):
            glosses = sentence_glosses[sentence_key]
            gloss_start = gloss_offsets[sentence_key]
            written, skipped = segment_sentence(
                zf=zf,
                sentence_key=sentence_key,
                frame_paths_in_zip=frame_paths,
                glosses=glosses,
                gloss_start_idx=gloss_start,
                output_dir=args.output_dir,
                fps=args.fps,
                force=args.force,
                tmp_root=args.tmp_dir,
            )
            total_written += written
            total_skipped += skipped
            if i % 50 == 0 or i == len(to_process):
                print(f"  [{i}/{len(to_process)}] written={total_written} skipped={total_skipped}")

    print(f"\nDone.")
    print(f"  Total MP4s written: {total_written}")
    print(f"  Already existed (skipped): {total_skipped}")
    print(f"  Output: {args.output_dir}")

    if args.delete_zip:
        if total_written > 0 or total_skipped > 0:
            os.remove(args.zip_path)
            print(f"  Deleted zip: {args.zip_path}")
        else:
            print(f"  WARNING: 0 clips written/skipped — zip NOT deleted to prevent data loss.")
            print(f"  Check ffmpeg is available and the zip is valid before re-running.")


if __name__ == "__main__":
    main()
