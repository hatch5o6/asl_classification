"""Per-anatomical-group detection rates for the three corpora.

Answers a direct reviewer request: does the gate drop facial landmarks because the face is
poorly detected (so detection noise masquerades as linguistic irrelevance), or despite
reliable detection?

MediaPipe Holistic runs separate detectors for the face mesh, the body, and each hand, and
returns NaN for an entire component when its detector does not fire -- a component is never
partially present. So a frame-level detection rate per component is well defined. We report:

  raw         fraction of frames in which the component is present in the extractor output
  after fill  the same fraction after our gap filling (linear, up to 5 consecutive frames),
              which is what the encoder actually receives
  in-frame    for the body only: fraction of present landmarks whose normalized coordinates
              lie inside the image. BlazePose regresses all 33 body landmarks whether or not
              they are visible, so "present" does not imply "observed" for the body.

Usage:
    python scripts/measure_detection_rates.py [--per-split 2000] [--out data/detection_rates.json]
"""

import argparse
import csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, "src")
from data.asl_dataset import RGBDSkel_Dataset  # noqa: E402

GROUPS = {
    "face": (0, 468),
    "body": (468, 501),
    "left_hand": (501, 522),
    "right_hand": (522, 543),
}
DATASETS = {
    "AUTSL": "data",
    "ASL Citizen": "data/asl_citizen",
    "GSL": "data/gsl",
}


class _Filler:
    """Borrow the exact gap-filling routine the training pipeline uses."""
    interpolate_with_gaps = RGBDSkel_Dataset.interpolate_with_gaps


def group_presence(xy):
    """(T, J, 2) -> per-group boolean (T,) arrays: is the component present in each frame."""
    finite = np.isfinite(xy).all(axis=2)  # (T, J)
    return {g: finite[:, a:b].all(axis=1) for g, (a, b) in GROUPS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-split", type=int, default=2000,
                    help="max clips sampled per split per dataset (all if fewer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/detection_rates.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    filler = _Filler()
    results = {}

    for ds, base in DATASETS.items():
        acc = {g: {"raw": [0, 0], "filled": [0, 0]} for g in GROUPS}
        body_in = [0, 0]
        clips = 0
        for split in ("train", "val", "test"):
            rows = list(csv.DictReader(open(f"{base}/{split}.csv")))
            rng.shuffle(rows)
            for r in rows[: args.per_split]:
                try:
                    k = np.load(r["skel_path"])[:, :, :2].astype(np.float64)
                except Exception:
                    continue
                clips += 1
                raw = group_presence(k)
                filled_xy = filler.interpolate_with_gaps(k.copy(), max_gap=5, sentinel=np.nan)
                filled = group_presence(filled_xy)
                for g in GROUPS:
                    acc[g]["raw"][0] += int(raw[g].sum())
                    acc[g]["raw"][1] += raw[g].size
                    acc[g]["filled"][0] += int(filled[g].sum())
                    acc[g]["filled"][1] += filled[g].size
                a, b = GROUPS["body"]
                body = k[:, a:b, :]
                present = np.isfinite(body).all(axis=2)
                inside = present & (body[..., 0] >= 0) & (body[..., 0] <= 1) \
                    & (body[..., 1] >= 0) & (body[..., 1] <= 1)
                body_in[0] += int(inside.sum())
                body_in[1] += int(present.sum())

        results[ds] = {
            "clips": clips,
            **{g: {"raw": acc[g]["raw"][0] / acc[g]["raw"][1],
                   "filled": acc[g]["filled"][0] / acc[g]["filled"][1]} for g in GROUPS},
            "body_in_frame": body_in[0] / body_in[1],
        }
        r = results[ds]
        print(f"\n{ds}  ({clips} clips)")
        print(f"  {'group':12}{'raw':>9}{'after fill':>12}")
        for g in GROUPS:
            print(f"  {g:12}{100 * r[g]['raw']:>8.2f}%{100 * r[g]['filled']:>11.2f}%")
        print(f"  body landmarks inside the image: {100 * r['body_in_frame']:.2f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
