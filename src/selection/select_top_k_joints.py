"""
Select top-K joints from learned joint probabilities.

Reads joint_probabilities.csv (output of visualize_joint_pruning.py) and generates
JSON files containing the indices of the top-K most important joints.

Supports two modes:
1. Direct selection from 543-joint model (default)
2. Sub-selection from a reduced model (via --source-indices), mapping back to 543-space

Usage:
    # Generate top-K indices from the best 543-joint model
    python src/select_top_k_joints.py \
        --probabilities models/pruning_sweep_gating_l0/figures/joint_probabilities.csv \
        --k 270 100 48 24 10 \
        --output-dir data/informed_selection/topk/

    # Sub-select from a reduced model (iterative approach)
    python src/select_top_k_joints.py \
        --probabilities models/iterative_270/figures/joint_probabilities.csv \
        --k 100 \
        --output-dir data/informed_selection/iterative/ \
        --source-indices data/informed_selection/iterative/iter_270_indices.json \
        --prefix iter
"""

import argparse
import json
import numpy as np
from pathlib import Path


# MediaPipe Holistic structure (543 total)
BODY_PARTS = {
    'face': {'start': 0, 'end': 468},
    'pose': {'start': 468, 'end': 501},
    'left_hand': {'start': 501, 'end': 522},
    'right_hand': {'start': 522, 'end': 543},
}


def get_body_part(idx):
    """Map a joint index (0-542) to its body part name."""
    for part, bounds in BODY_PARTS.items():
        if bounds['start'] <= idx < bounds['end']:
            return part
    return 'unknown'


def load_probabilities(csv_path):
    """Load joint probabilities from CSV (output of visualize_joint_pruning.py)."""
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    return data


def select_top_k(probabilities, k, source_indices=None):
    """
    Select top-K joints by probability.

    Args:
        probabilities: (N,) array of keep probabilities
        k: number of joints to select
        source_indices: if provided, list of N original-space indices
                       (for mapping back from reduced model)

    Returns:
        selected_indices: list of K indices in original 543-space
        selected_probs: list of K probabilities
    """
    assert k <= len(probabilities), (
        f"Cannot select {k} joints from {len(probabilities)} available"
    )

    # Sort by probability descending
    sorted_order = np.argsort(probabilities)[::-1]
    top_k_local = sorted_order[:k]

    # Map to original 543-space if needed
    if source_indices is not None:
        selected_indices = [source_indices[i] for i in top_k_local]
    else:
        selected_indices = top_k_local.tolist()

    selected_probs = probabilities[top_k_local].tolist()

    # Sort by index for consistent ordering
    paired = sorted(zip(selected_indices, selected_probs))
    selected_indices = [p[0] for p in paired]
    selected_probs = [p[1] for p in paired]

    return selected_indices, selected_probs


def compute_body_part_breakdown(indices):
    """Compute how many joints are from each body part."""
    breakdown = {part: 0 for part in BODY_PARTS}
    for idx in indices:
        part = get_body_part(idx)
        if part in breakdown:
            breakdown[part] += 1
    return breakdown


def main():
    parser = argparse.ArgumentParser(
        description="Select top-K joints from learned probabilities"
    )
    parser.add_argument(
        "--probabilities", type=str, required=True,
        help="Path to joint_probabilities.csv"
    )
    parser.add_argument(
        "--k", type=int, nargs='+', required=True,
        help="K values to generate (e.g., 270 100 48 24 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save output JSON files"
    )
    parser.add_argument(
        "--source-indices", type=str, default=None,
        help="Path to source indices JSON (for sub-selection from reduced model)"
    )
    parser.add_argument(
        "--prefix", type=str, default="top",
        help="Prefix for output files (default: 'top', use 'iter' for iterative)"
    )
    args = parser.parse_args()

    # Load probabilities
    probabilities = load_probabilities(args.probabilities)
    n_joints = len(probabilities)
    print(f"Loaded {n_joints} joint probabilities from {args.probabilities}")
    print(f"  Range: [{probabilities.min():.6f}, {probabilities.max():.6f}]")
    print(f"  Mean: {probabilities.mean():.6f}, Std: {probabilities.std():.6f}")

    # Load source indices if provided
    source_indices = None
    if args.source_indices:
        with open(args.source_indices) as f:
            source_indices = json.load(f)
        assert len(source_indices) == n_joints, (
            f"Source indices ({len(source_indices)}) must match "
            f"probabilities ({n_joints})"
        )
        print(f"Using source indices from {args.source_indices} "
              f"({len(source_indices)} indices)")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate top-K for each K value
    all_summaries = {}
    for k in args.k:
        if k > n_joints:
            print(f"\nWARNING: k={k} > n_joints={n_joints}, skipping")
            continue

        selected_indices, selected_probs = select_top_k(
            probabilities, k, source_indices
        )

        # Save indices JSON
        output_file = output_dir / f"{args.prefix}_{k}_indices.json"
        with open(output_file, 'w') as f:
            json.dump(selected_indices, f, indent=2)

        # Compute body part breakdown
        breakdown = compute_body_part_breakdown(selected_indices)

        # Compute probability stats for selected vs excluded
        prob_stats = {
            'selected_min': min(selected_probs),
            'selected_max': max(selected_probs),
            'selected_mean': sum(selected_probs) / len(selected_probs),
        }

        if k < n_joints:
            # Get excluded probabilities
            all_local_indices = set(range(n_joints))
            if source_indices:
                selected_local = [source_indices.index(idx) for idx in selected_indices]
            else:
                selected_local = selected_indices
            excluded_local = list(all_local_indices - set(selected_local))
            excluded_probs = probabilities[excluded_local]
            prob_stats['excluded_min'] = float(excluded_probs.min())
            prob_stats['excluded_max'] = float(excluded_probs.max())
            prob_stats['excluded_mean'] = float(excluded_probs.mean())

        summary = {
            'k': k,
            'body_part_breakdown': breakdown,
            'probability_stats': prob_stats,
            'indices': selected_indices,
        }
        all_summaries[k] = summary

        print(f"\nTop {k} joints:")
        print(f"  Saved to: {output_file}")
        print(f"  Body parts: {breakdown}")
        print(f"  Prob range: [{prob_stats['selected_min']:.6f}, "
              f"{prob_stats['selected_max']:.6f}]")

    # Save comprehensive summary
    summary_file = output_dir / f"{args.prefix}_selection_summary.json"
    # Convert keys to strings for JSON
    json_summaries = {str(k): v for k, v in all_summaries.items()}
    with open(summary_file, 'w') as f:
        json.dump(json_summaries, f, indent=2)
    print(f"\nSaved selection summary to {summary_file}")


if __name__ == "__main__":
    main()
