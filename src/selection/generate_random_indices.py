"""
Generate random joint index selections as baselines for comparison.

Creates N random draws of K joints from the full 543-joint MediaPipe space.
Output format matches the learned selection indices (sorted JSON arrays).

Usage:
    python src/selection/generate_random_indices.py \
        --k 270 100 48 24 10 \
        --n-draws 3 \
        --seed 42 \
        --output-dir data/random_selection/

    # Generate for a specific dataset subdirectory
    python src/selection/generate_random_indices.py \
        --k 270 100 48 24 10 \
        --n-draws 3 \
        --seed 42 \
        --output-dir data/asl_citizen/random_selection/
"""

import argparse
import json
import numpy as np
from pathlib import Path


TOTAL_JOINTS = 543

# MediaPipe Holistic structure
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


def compute_body_part_breakdown(indices):
    """Compute how many joints are from each body part."""
    breakdown = {part: 0 for part in BODY_PARTS}
    for idx in indices:
        part = get_body_part(idx)
        if part in breakdown:
            breakdown[part] += 1
    return breakdown


def generate_random_selection(k, rng):
    """Generate a random selection of k joints from 543, returned sorted."""
    selected = rng.choice(TOTAL_JOINTS, size=k, replace=False)
    return sorted(selected.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Generate random joint index selections as baselines"
    )
    parser.add_argument(
        "--k", type=int, nargs='+', required=True,
        help="K values to generate (e.g., 270 100 48 24 10)"
    )
    parser.add_argument(
        "--n-draws", type=int, default=3,
        help="Number of random draws per K value (default: 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (each draw uses seed + draw_index)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save output JSON files"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for k in args.k:
        assert k < TOTAL_JOINTS, f"k={k} must be less than {TOTAL_JOINTS}"
        k_summaries = []

        print(f"\nK={k}:")
        for draw in range(args.n_draws):
            seed = args.seed + draw
            rng = np.random.default_rng(seed)
            indices = generate_random_selection(k, rng)
            breakdown = compute_body_part_breakdown(indices)

            # Save index file: random_{k}_draw{draw}_indices.json
            filename = f"random_{k}_draw{draw}_indices.json"
            filepath = output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(indices, f, indent=2)

            print(f"  Draw {draw} (seed={seed}): {breakdown} -> {filepath}")

            k_summaries.append({
                'draw': draw,
                'seed': seed,
                'body_part_breakdown': breakdown,
                'indices': indices,
            })

        all_summaries[str(k)] = {
            'k': k,
            'n_draws': args.n_draws,
            'base_seed': args.seed,
            'draws': k_summaries,
        }

    # Save summary
    summary_file = output_dir / "random_selection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summary to {summary_file}")


if __name__ == "__main__":
    main()
