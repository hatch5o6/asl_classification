#!/usr/bin/env python3
"""
visualize_skeleton_selection.py

Multi-panel skeleton overlay showing the joint selection progression
(K = All → 270 → 100 → 48 → 24 → 10).

Each panel renders:
  - Selected joints: filled circles, colored by body region
  - Dropped joints: tiny faint gray circles
  - Skeleton connections: light gray edges (pose + hands; face oval when active)

Usage
-----
# Preview with hardcoded canonical positions (no data needed):
    python src/visualize_skeleton_selection.py \\
        --indices-dir data/informed_selection/topk \\
        --output-dir figures/skeleton_selection

# Paper-quality figure using a real skeleton sample:
    python src/visualize_skeleton_selection.py \\
        --indices-dir data/informed_selection/topk \\
        --ref-npy /path/to/any_landmarks.npy \\
        --output-dir figures/skeleton_selection

# Average positions over many real samples (most robust):
    python src/visualize_skeleton_selection.py \\
        --indices-dir data/informed_selection/topk \\
        --ref-csv data/val.csv \\
        --ref-n-samples 50 \\
        --output-dir figures/skeleton_selection

# Full suite: progression + topk-vs-iterative comparison:
    python src/visualize_skeleton_selection.py \\
        --indices-dir data/informed_selection/topk \\
        --iterative-dir data/informed_selection/iterative \\
        --ref-csv data/val.csv \\
        --output-dir figures/skeleton_selection
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Region definitions  (must match visualize_joint_pruning.py)
# ---------------------------------------------------------------------------

REGIONS: Dict[str, tuple] = {
    "face":       (0,   468),
    "pose":       (468, 501),
    "left_hand":  (501, 522),
    "right_hand": (522, 543),
}

REGION_COLORS: Dict[str, str] = {
    "face":       "#4C72B0",   # blue
    "pose":       "#DD8452",   # orange
    "left_hand":  "#55A868",   # green
    "right_hand": "#C44E52",   # red
}

REGION_LABELS: Dict[str, str] = {
    "face":       "Face",
    "pose":       "Pose",
    "left_hand":  "Left Hand",
    "right_hand": "Right Hand",
}


def joint_region(idx: int) -> str:
    for name, (lo, hi) in REGIONS.items():
        if lo <= idx < hi:
            return name
    raise ValueError(f"Joint {idx} out of range 0-542")


# ---------------------------------------------------------------------------
# MediaPipe skeleton connections (global 543-space indices)
# ---------------------------------------------------------------------------

_POSE_LOCAL = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (31, 27),
    (24, 26), (26, 28), (28, 30), (30, 32), (32, 28),
]
POSE_CONNECTIONS = [(468 + a, 468 + b) for a, b in _POSE_LOCAL]

_HAND_LOCAL = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),             # palm arc
]
LEFT_HAND_CONNECTIONS  = [(501 + a, 501 + b) for a, b in _HAND_LOCAL]
RIGHT_HAND_CONNECTIONS = [(522 + a, 522 + b) for a, b in _HAND_LOCAL]

# Simplified face oval contour for readability (indices 0-467 are global for face)
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]
FACE_OVAL_CONNECTIONS = [
    (_FACE_OVAL[i], _FACE_OVAL[i + 1]) for i in range(len(_FACE_OVAL) - 1)
]

BODY_CONNECTIONS = POSE_CONNECTIONS + LEFT_HAND_CONNECTIONS + RIGHT_HAND_CONNECTIONS


# ---------------------------------------------------------------------------
# Canonical joint positions (fallback when no reference skeleton is provided)
# ---------------------------------------------------------------------------

def _hand_unit_positions() -> np.ndarray:
    """(21, 2) positions in unit space, wrist at bottom-center, fingers pointing up."""
    return np.array([
        [0.50, 0.00],  # 0  wrist
        [0.30, 0.22],  # 1  thumb CMC
        [0.17, 0.38],  # 2  thumb MCP
        [0.08, 0.50],  # 3  thumb IP
        [0.02, 0.60],  # 4  thumb tip
        [0.38, 0.30],  # 5  index MCP
        [0.37, 0.54],  # 6  index PIP
        [0.37, 0.70],  # 7  index DIP
        [0.37, 0.85],  # 8  index tip
        [0.50, 0.32],  # 9  middle MCP
        [0.50, 0.56],  # 10 middle PIP
        [0.50, 0.73],  # 11 middle DIP
        [0.50, 0.88],  # 12 middle tip
        [0.62, 0.30],  # 13 ring MCP
        [0.63, 0.53],  # 14 ring PIP
        [0.63, 0.68],  # 15 ring DIP
        [0.63, 0.81],  # 16 ring tip
        [0.73, 0.22],  # 17 pinky MCP
        [0.75, 0.44],  # 18 pinky PIP
        [0.75, 0.58],  # 19 pinky DIP
        [0.75, 0.72],  # 20 pinky tip
    ], dtype=np.float32)


def make_canonical_positions() -> np.ndarray:
    """Return (543, 2) approximate canonical positions in [0,1] display space.

    These are hardcoded estimates for a frontal standing pose.
    For paper figures, use --ref-npy or --ref-csv instead.
    """
    xy = np.zeros((543, 2), dtype=np.float32)

    # Face (0-467): lay out as 26×18 grid over the head region
    COLS, ROWS = 26, 18
    x0, y0, fw, fh = 0.37, 0.03, 0.26, 0.29
    for i in range(468):
        col = i % COLS
        row = i // COLS
        xy[i] = [x0 + (col / (COLS - 1)) * fw, y0 + (row / (ROWS - 1)) * fh]

    # Pose (468-500): anatomical positions for a frontal standing pose
    pose_pts = [
        (0.500, 0.12),  # 0  nose
        (0.480, 0.10),  # 1  left-eye inner
        (0.470, 0.09),  # 2  left-eye
        (0.460, 0.10),  # 3  left-eye outer
        (0.520, 0.10),  # 4  right-eye inner
        (0.530, 0.09),  # 5  right-eye
        (0.540, 0.10),  # 6  right-eye outer
        (0.455, 0.13),  # 7  left-ear
        (0.545, 0.13),  # 8  right-ear
        (0.490, 0.19),  # 9  mouth-left
        (0.510, 0.19),  # 10 mouth-right
        (0.410, 0.30),  # 11 left-shoulder
        (0.590, 0.30),  # 12 right-shoulder
        (0.300, 0.43),  # 13 left-elbow
        (0.700, 0.43),  # 14 right-elbow
        (0.210, 0.55),  # 15 left-wrist
        (0.790, 0.55),  # 16 right-wrist
        (0.195, 0.54),  # 17 left-pinky
        (0.775, 0.54),  # 18 right-pinky
        (0.200, 0.55),  # 19 left-index
        (0.780, 0.55),  # 20 right-index
        (0.205, 0.56),  # 21 left-thumb
        (0.785, 0.56),  # 22 right-thumb
        (0.435, 0.57),  # 23 left-hip
        (0.565, 0.57),  # 24 right-hip
        (0.425, 0.70),  # 25 left-knee
        (0.575, 0.70),  # 26 right-knee
        (0.420, 0.85),  # 27 left-ankle
        (0.580, 0.85),  # 28 right-ankle
        (0.415, 0.90),  # 29 left-heel
        (0.585, 0.90),  # 30 right-heel
        (0.425, 0.93),  # 31 left-foot
        (0.575, 0.93),  # 32 right-foot
    ]
    for i, (x, y) in enumerate(pose_pts):
        xy[468 + i] = [x, y]

    # Left hand (501-521): offset from left-wrist
    hand = _hand_unit_positions()
    lw = np.array([0.210, 0.570], dtype=np.float32)
    lh = hand * np.array([0.11, 0.14]) + lw + np.array([-0.055, 0.0])
    xy[501:522] = lh

    # Right hand (522-542): mirror around right-wrist
    rw = np.array([0.790, 0.570], dtype=np.float32)
    rh = hand * np.array([0.11, 0.14]) + rw + np.array([-0.055, 0.0])
    rh[:, 0] = rw[0] + (rw[0] - rh[:, 0])  # mirror x-axis
    xy[522:543] = rh

    return xy


# ---------------------------------------------------------------------------
# Load reference positions from real skeleton data
# ---------------------------------------------------------------------------

def load_ref_from_npy(npy_path: str) -> np.ndarray:
    """Load (543, 2) xy positions from a raw landmarks .npy file.

    Expects shape (T, 543, 4) with [x, y, z, visibility] in [0, 1].
    Uses the middle frame to represent a typical pose.
    """
    data = np.load(npy_path)
    if data.ndim == 3:               # (T, 543, 4)
        mid = data.shape[0] // 2
        xy = data[mid, :543, :2].astype(np.float32)
    elif data.ndim == 2:             # (543, 4) — single frame
        xy = data[:543, :2].astype(np.float32)
    else:
        raise ValueError(f"Unexpected .npy shape: {data.shape}")
    return xy


def load_ref_from_csv(csv_path: str, n_samples: int = 30) -> np.ndarray:
    """Compute median joint positions over N skeleton files listed in a CSV.

    Looks for the first column whose header contains 'skel'.
    """
    paths: List[str] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        skel_col = next(
            (c for c in fieldnames if "skel" in c.lower()), None
        )
        if skel_col is None:
            raise ValueError(f"No column matching 'skel*' found in {csv_path}")
        for i, row in enumerate(reader):
            if i >= n_samples:
                break
            paths.append(row[skel_col])

    frames = []
    for p in paths:
        try:
            data = np.load(p)
            mid = data.shape[0] // 2
            frames.append(data[mid, :543, :2])
        except Exception as e:
            print(f"  [warn] Skipping {p}: {e}")

    if not frames:
        raise ValueError("No skeleton files could be loaded from CSV")

    print(f"  Computed median positions from {len(frames)} samples")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Load index files
# ---------------------------------------------------------------------------

def load_indices(indices_dir: str, prefix: str = "top") -> Dict[int, List[int]]:
    """Load all {prefix}_{K}_indices.json files from a directory.

    Returns {K: [joint_index, ...]} sorted by K.
    """
    result: Dict[int, List[int]] = {}
    for path in sorted(Path(indices_dir).glob(f"{prefix}_*_indices.json")):
        m = re.search(r"_(\d+)_indices", path.name)
        if m:
            result[int(m.group(1))] = json.loads(path.read_text())
    return result


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def draw_panel(
    ax: plt.Axes,
    xy: np.ndarray,       # (543, 2) positions
    selected: Set[int],
    title: str,
    *,
    dot_scale: float = 1.0,
    show_face_oval: bool = False,
) -> None:
    """Render one skeleton panel onto ax."""
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

    # Connections: always draw full body skeleton in light gray
    connections = list(BODY_CONNECTIONS)
    if show_face_oval:
        connections += FACE_OVAL_CONNECTIONS
    segs = [[xy[a], xy[b]] for a, b in connections if a < 543 and b < 543]
    ax.add_collection(LineCollection(segs, colors="#d0d0d0", linewidths=0.7, zorder=1))

    # Dropped joints: tiny faint gray dots so the reader sees what was removed
    dropped = [i for i in range(543) if i not in selected]
    if dropped:
        ax.scatter(
            xy[dropped, 0], xy[dropped, 1],
            s=3 * dot_scale, c="#cccccc", linewidths=0, zorder=2,
        )

    # Selected joints: filled circles colored by body region
    for region, (lo, hi) in REGIONS.items():
        sel_r = [i for i in selected if lo <= i < hi]
        if not sel_r:
            continue
        # Face dots are smaller because 31-195 joints cluster densely
        size = 8 * dot_scale if region == "face" else 28 * dot_scale
        ax.scatter(
            xy[sel_r, 0], xy[sel_r, 1],
            s=size, c=REGION_COLORS[region], linewidths=0, zorder=3,
        )

    # Joint count below panel title
    ax.text(
        0.5, -0.02, f"n = {len(selected)}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8, color="#666666",
    )

    # Fit view to the full skeleton with a small margin
    margin = 0.05
    ax.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
    # y=0 at top (MediaPipe normalized coordinate convention)
    ax.set_ylim(xy[:, 1].max() + margin, xy[:, 1].min() - margin)


def _legend_handles() -> List[mpatches.Patch]:
    handles = [
        mpatches.Patch(color=REGION_COLORS[r], label=REGION_LABELS[r])
        for r in REGION_COLORS
    ]
    handles.append(mpatches.Patch(color="#cccccc", label="Dropped"))
    return handles


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_progression_figure(
    xy: np.ndarray,
    indices_by_k: Dict[int, List[int]],
    k_values: List[int],
    *,
    output_path: str,
    ncols: int = 3,
    fig_width: float = 12.0,
    dpi: int = 150,
) -> None:
    """Build the main progression figure: All → K_max → … → K_min."""
    # First panel is always the full set
    panels: List[tuple] = [(543, set(range(543)))]
    for k in sorted(k_values, reverse=True):
        if k in indices_by_k:
            panels.append((k, set(indices_by_k[k])))
        else:
            print(f"  [warn] K={k} not found in indices — skipping")

    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig_height = (fig_width / ncols) * nrows * 1.35

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = np.array(axes).flatten()

    for i, (k, selected) in enumerate(panels):
        label = "All (543)" if k == 543 else f"Top-{k}"
        n_face = sum(1 for j in selected if j < 468)
        draw_panel(axes[i], xy, selected, label, show_face_oval=(n_face > 30))

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.legend(
        handles=_legend_handles(), loc="lower center",
        ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("Joint Selection Progression", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def build_comparison_figure(
    xy: np.ndarray,
    topk: Dict[int, List[int]],
    iterative: Dict[int, List[int]],
    k_values: List[int],
    *,
    output_path: str,
    fig_width: float = 12.0,
    dpi: int = 150,
) -> None:
    """Build a 2-row comparison: top row = Top-K, bottom row = Iterative."""
    valid_k = sorted([k for k in k_values if k in topk and k in iterative])
    if not valid_k:
        print("  [warn] No common K values in both index sets — skipping comparison figure")
        return

    n = len(valid_k)
    panel_w = fig_width / n
    fig, axes = plt.subplots(2, n, figsize=(fig_width, panel_w * 2 * 1.35))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, k in enumerate(valid_k):
        n_face_topk = sum(1 for j in topk[k] if j < 468)
        n_face_iter = sum(1 for j in iterative[k] if j < 468)
        draw_panel(axes[0, col], xy, set(topk[k]),
                   f"Top-{k}", show_face_oval=(n_face_topk > 30))
        draw_panel(axes[1, col], xy, set(iterative[k]),
                   f"Iterative-{k}", show_face_oval=(n_face_iter > 30))

    row_labels = ["Top-K\n(direct)", "Iterative\n(cascade)"]
    for row, label in enumerate(row_labels):
        axes[row, 0].text(
            -0.10, 0.5, label, transform=axes[row, 0].transAxes,
            va="center", ha="right", fontsize=9, fontweight="bold", rotation=90,
        )

    fig.legend(
        handles=_legend_handles(), loc="lower center",
        ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("Top-K vs Iterative Joint Selection", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--indices-dir", required=True,
        help="Directory containing top_{K}_indices.json files",
    )
    p.add_argument(
        "--iterative-dir", default=None,
        help="Directory containing iter_{K}_indices.json files (optional)",
    )
    p.add_argument(
        "--ref-npy", default=None,
        help="Path to a single .npy landmarks file (T, 543, 4) for canonical positions",
    )
    p.add_argument(
        "--ref-csv", default=None,
        help="Path to a CSV with a 'skel_path' column — medians N samples for stable positions",
    )
    p.add_argument(
        "--ref-n-samples", type=int, default=30,
        help="Number of CSV rows to average for reference positions (default: 30)",
    )
    p.add_argument(
        "--output-dir", default="figures/skeleton_selection",
        help="Output directory for saved figures (default: figures/skeleton_selection)",
    )
    p.add_argument(
        "--k-values", nargs="+", type=int, default=[270, 100, 48, 24, 10],
        help="K values to include in the progression (default: 270 100 48 24 10)",
    )
    p.add_argument(
        "--ncols", type=int, default=3,
        help="Columns in the progression grid (default: 3)",
    )
    p.add_argument("--fig-width", type=float, default=12.0,
                   help="Figure width in inches (default: 12)")
    p.add_argument("--dpi", type=int, default=150,
                   help="Output DPI (default: 150; use 300 for final paper figures)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Canonical joint positions ---
    if args.ref_npy:
        print(f"Loading reference positions from: {args.ref_npy}")
        xy = load_ref_from_npy(args.ref_npy)
    elif args.ref_csv:
        print(f"Computing median positions from {args.ref_n_samples} samples in: {args.ref_csv}")
        xy = load_ref_from_csv(args.ref_csv, n_samples=args.ref_n_samples)
    else:
        print("No reference skeleton provided — using hardcoded canonical positions")
        xy = make_canonical_positions()

    # --- Load index files ---
    print(f"\nLoading top-K indices from: {args.indices_dir}")
    topk = load_indices(args.indices_dir, prefix="top")
    print(f"  K values found: {sorted(topk.keys())}")

    iterative: Dict[int, List[int]] = {}
    if args.iterative_dir:
        print(f"Loading iterative indices from: {args.iterative_dir}")
        iterative = load_indices(args.iterative_dir, prefix="iter")
        print(f"  K values found: {sorted(iterative.keys())}")

    out_dir = args.output_dir
    print(f"\nOutput directory: {out_dir}")

    # --- Figure 1: progression ---
    print("\nBuilding progression figure...")
    build_progression_figure(
        xy, topk, args.k_values,
        output_path=f"{out_dir}/skeleton_progression.png",
        ncols=args.ncols,
        fig_width=args.fig_width,
        dpi=args.dpi,
    )

    # --- Figure 2: topk vs iterative (only if iterative dir was given) ---
    if iterative:
        print("\nBuilding top-K vs iterative comparison figure...")
        common_k = [k for k in args.k_values if k in iterative]
        build_comparison_figure(
            xy, topk, iterative, common_k,
            output_path=f"{out_dir}/skeleton_topk_vs_iterative.png",
            fig_width=args.fig_width,
            dpi=args.dpi,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
