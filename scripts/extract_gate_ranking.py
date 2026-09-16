"""Select the next cascade subset from a trained gate checkpoint.

Runs as its own process so that `scripts/run_cascade.py` never has to import torch or
numpy. A parent that imports them before spawning `src/train.py` leaves
`MKL_THREADING_LAYER=INTEL` in the environment, which collides with the libgomp that
torch loads in the child ("MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1").
Keeping the orchestrator import-free means every child sees the same environment it would
get from a plain sbatch invocation.

Usage:
    python scripts/extract_gate_ranking.py \
        --checkpoint MODEL/checkpoints/last.ckpt \
        --pool data/.../cascade/iter_270_indices.json \
        --keep 100 \
        --out data/.../cascade/iter_100_indices.json
"""

import argparse
import json

import numpy as np
import torch

# Mirrored from src/models/joint_pruning.py.
HC_GAMMA, HC_ZETA, HC_BETA = -0.1, 1.1, 2.0 / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pool", required=True, help="JSON list of 543-space indices the gate saw")
    ap.add_argument("--keep", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = json.load(open(args.pool))
    state = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    logits = state["joint_pruning.joint_logits"].float().numpy()

    if len(logits) != len(pool):
        raise SystemExit(
            f"gate has {len(logits)} logits but pool has {len(pool)} indices -- "
            f"num_pose_points and joint_indices_file disagree"
        )
    if args.keep > len(pool):
        raise SystemExit(f"cannot keep {args.keep} of {len(pool)}")

    # P(gate > 0): the quantity the expected-L0 penalty sums, and the importance score.
    shift = HC_BETA * np.log(-HC_GAMMA / HC_ZETA)
    probs = 1.0 / (1.0 + np.exp(-(logits - shift)))

    keep_local = np.argsort(probs)[::-1][: args.keep]
    keep = sorted(int(pool[i]) for i in keep_local)
    json.dump(keep, open(args.out, "w"))

    gates = np.clip(1.0 / (1.0 + np.exp(-logits)) * (HC_ZETA - HC_GAMMA) + HC_GAMMA, 0.0, 1.0)
    face = sum(1 for i in keep if i < 468)
    pose = sum(1 for i in keep if 468 <= i < 501)
    lh = sum(1 for i in keep if 501 <= i < 522)
    rh = sum(1 for i in keep if i >= 522)
    print(f"    kept {len(keep)} of {len(pool)}: face {face} / pose {pose} / LH {lh} / RH {rh}")
    print(f"    open-prob range [{probs.min():.4f}, {probs.max():.4f}], "
          f"IQR {np.percentile(probs, 75) - np.percentile(probs, 25):.4f}, "
          f"hard zeros {int((gates == 0).sum())}/{len(pool)}")
    print(f"    -> {args.out}")


if __name__ == "__main__":
    main()
