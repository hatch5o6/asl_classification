"""
Measure which MediaPipe pose landmarks were never actually filmed.

DIAGNOSTIC ONLY -- nothing in the training pipeline consumes this. It exists to
document a data property in the paper, and to corroborate the fact that a correctly
configured L0 gate demotes these landmarks on its own (on GSL the annealed sigmoid
gate ranked 7 of 8 never-filmed landmarks inside the top-48; the hard-concrete gate
at init 0.5 ranks none of them there). We deliberately do NOT filter on this: the
gate learning to ignore fabricated coordinates is a result, whereas deleting them by
hand would be an intervention with a hand-tuned threshold to defend.

MediaPipe's BlazePose regresses all 33 pose landmarks from the detected person
ROI whether or not they fall inside the frame, so landmarks for body parts that
were never filmed still receive plausible-looking coordinates. MediaPipe signals
this through the `visibility` channel, which our extraction stores as channel 3
of the (T, 543, 4) arrays written by pose_points.py but which the dataloader
discards when it slices down to (x, y).

This script recovers that signal and reports it.

The exclusion criterion is whether MediaPipe places a landmark *outside the image
bounds*, which is a direct statement that the body part was not filmed. We do not
threshold on the visibility score itself: it is a confidence estimate, and on
tightly-cropped corpora it runs low for joints that are plainly in frame (on
ASL Citizen the left wrist scores 0.082 despite being visible and its hand
landmarks being detected). Visibility is recorded alongside as corroborating
evidence, since the two criteria agree on the landmarks that matter.

Only the 33 pose landmarks are eligible for exclusion. MediaPipe emits them
unconditionally, whereas face and hand landmarks are simply absent when not
detected; face and hand landmarks are therefore always kept.

Usage:
    python src/pose/compute_visibility_mask.py \
        --train-csv data/gsl/train.csv \
        --output data/gsl/visibility_mask.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# MediaPipe Holistic structure (543 total), matching src/selection/select_top_k_joints.py
POSE_START, POSE_END = 468, 501
N_LANDMARKS = 543

POSE_NAMES = {
    468: 'nose', 469: 'left_eye_inner', 470: 'left_eye', 471: 'left_eye_outer',
    472: 'right_eye_inner', 473: 'right_eye', 474: 'right_eye_outer',
    475: 'left_ear', 476: 'right_ear', 477: 'mouth_left', 478: 'mouth_right',
    479: 'left_shoulder', 480: 'right_shoulder', 481: 'left_elbow',
    482: 'right_elbow', 483: 'left_wrist', 484: 'right_wrist',
    485: 'left_pinky', 486: 'right_pinky', 487: 'left_index',
    488: 'right_index', 489: 'left_thumb', 490: 'right_thumb',
    491: 'left_hip', 492: 'right_hip', 493: 'left_knee', 494: 'right_knee',
    495: 'left_ankle', 496: 'right_ankle', 497: 'left_heel', 498: 'right_heel',
    499: 'left_foot_index', 500: 'right_foot_index',
}


def measure_landmarks(train_csv, max_clips, seed):
    """Per-landmark out-of-frame rate and median visibility over training clips."""
    df = pd.read_csv(train_csv)
    paths = [p for p in df['skel_path'].tolist() if isinstance(p, str)]
    random.Random(seed).shuffle(paths)

    vis_per_clip = []
    oob_per_clip = []
    for path in paths:
        if len(vis_per_clip) >= max_clips:
            break
        if not Path(path).exists():
            continue
        arr = np.load(path)
        if arr.ndim != 3 or arr.shape[1] != N_LANDMARKS or arr.shape[2] < 4:
            continue
        if arr.shape[0] < 1:
            continue
        # Median over frames within the clip, ignoring frames MediaPipe dropped.
        vis_per_clip.append(np.nanmedian(arr[:, :, 3], axis=0))
        # MediaPipe coordinates are image-relative, so anything outside [0, 1]
        # was placed beyond the frame.
        xy = arr[:, :, :2]
        outside = (xy < 0.0) | (xy > 1.0)
        oob_per_clip.append(np.nanmean(outside.any(axis=2), axis=0))

    if not vis_per_clip:
        raise RuntimeError(f"No usable skeleton files found from {train_csv}")

    # Median across clips is robust to a handful of pathological videos.
    return (np.nanmedian(np.stack(vis_per_clip), axis=0),
            np.nanmean(np.stack(oob_per_clip), axis=0),
            len(vis_per_clip))


def main():
    parser = argparse.ArgumentParser(
        description="Report which pose landmarks were never filmed (diagnostic)"
    )
    parser.add_argument("--train-csv", type=str, required=True,
                        help="Training CSV with a skel_path column")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write the mask JSON")
    parser.add_argument("--oob-threshold", type=float, default=0.8,
                        help="Exclude pose landmarks placed outside the image in "
                             "more than this fraction of frames (default: 0.8). "
                             "The default separates never-filmed landmarks from "
                             "articulators that merely leave frame intermittently: "
                             "on ASL Citizen the wrists sit at 55-67%% because the "
                             "signers' hands exit the camera view, and must be kept.")
    parser.add_argument("--max-clips", type=int, default=300,
                        help="Number of training clips to sample (default: 300)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed (default: 0)")
    args = parser.parse_args()

    visibility, out_of_frame, n_clips = measure_landmarks(
        args.train_csv, args.max_clips, args.seed
    )

    # Only pose landmarks are eligible; MediaPipe emits them unconditionally.
    excluded = [
        idx for idx in range(POSE_START, POSE_END)
        if not np.isnan(out_of_frame[idx]) and out_of_frame[idx] > args.oob_threshold
    ]
    kept = [idx for idx in range(N_LANDMARKS) if idx not in set(excluded)]

    print(f"Sampled {n_clips} clips from {args.train_csv}")
    print(f"Criterion: placed outside the frame in > {args.oob_threshold:.0%} "
          f"of frames (pose landmarks only)\n")
    print(f"  {'landmark':20s} {'out-of-frame':>13s} {'visibility':>11s}  {'status':>8s}")
    for idx in range(POSE_START, POSE_END):
        status = "EXCLUDED" if idx in set(excluded) else "kept"
        print(f"  {POSE_NAMES[idx]:20s} {out_of_frame[idx]:12.1%} "
              f"{visibility[idx]:11.3f}  {status:>8s}")

    print(f"\n{len(excluded)} of 33 pose landmarks are never filmed "
          f"(placed outside the frame in >{args.oob_threshold:.0%} of frames).")

    payload = {
        'oob_threshold': args.oob_threshold,
        'n_clips_sampled': n_clips,
        'source_csv': args.train_csv,
        'never_filmed_indices': excluded,
        'never_filmed_names': [POSE_NAMES[i] for i in excluded],
        'observed_indices': kept,
        'pose_out_of_frame_rate': {
            POSE_NAMES[i]: (None if np.isnan(out_of_frame[i]) else round(float(out_of_frame[i]), 4))
            for i in range(POSE_START, POSE_END)
        },
        'pose_visibility': {
            POSE_NAMES[i]: (None if np.isnan(visibility[i]) else round(float(visibility[i]), 4))
            for i in range(POSE_START, POSE_END)
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
