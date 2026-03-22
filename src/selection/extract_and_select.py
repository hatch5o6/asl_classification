"""
Extract joint probabilities from a trained model and select top-K for next iteration.

Used in Approach 2 (Iterative Cascade):
  After training a model with N points, this script:
  1. Loads the best checkpoint
  2. Extracts L0 joint probabilities (N values)
  3. Maps them back to original 543-space using the source indices
  4. Selects top K and writes a new indices JSON for the next stage

Usage:
    python src/selection/extract_and_select.py \
        --checkpoint models/iterative_270/checkpoints/best.ckpt \
        --config configs/informed_selection/iterative_270.yaml \
        --target-k 100 \
        --output data/informed_selection/iterative/iter_100_indices.json

    # Auto-detect best checkpoint:
    python src/selection/extract_and_select.py \
        --model-dir models/informed_selection/iterative_270 \
        --config configs/informed_selection/iterative_270.yaml \
        --target-k 100 \
        --output data/informed_selection/iterative/iter_100_indices.json
"""

import argparse
import json
import os
import yaml
import torch
import numpy as np
from pathlib import Path


def find_best_checkpoint(model_dir):
    """Find the checkpoint with highest val_acc in the checkpoints directory."""
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    assert os.path.exists(checkpoints_dir), f"No checkpoints dir: {checkpoints_dir}"

    best_ckpt = None
    best_val_acc = -1
    for f in os.listdir(checkpoints_dir):
        if not f.endswith(".ckpt"):
            continue
        try:
            val_acc = float(f.split(".ckpt")[0].split("-val_acc=")[1])
        except (IndexError, ValueError):
            continue
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(checkpoints_dir, f)

    assert best_ckpt is not None, f"No valid checkpoints in {checkpoints_dir}"
    print(f"Best checkpoint: {best_ckpt} (val_acc={best_val_acc:.6f})")
    return best_ckpt


def load_config(config_path):
    """Load and process config (subset of read_config from train.py)."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Replace user placeholder
    import getpass
    config_str = json.dumps(config)
    config_str = config_str.replace("hatch5o6", getpass.getuser())
    config = json.loads(config_str)

    # Set defaults needed for model loading
    config.setdefault("num_frames", "video_mae")
    config.setdefault("bert_dropout", 0.0)
    config.setdefault("batch_size",
                       config.get("effective_batch_size", 32) // config.get("n_gpus", 4))
    config.setdefault("warmup_steps", round(0.05 * config.get("max_steps", 200000)))
    if "gate_learning_rate" in config:
        config["gate_learning_rate"] = float(config["gate_learning_rate"])

    # Load source indices
    if "joint_indices_file" in config and config["joint_indices_file"] is not None:
        with open(config["joint_indices_file"]) as jf:
            config["selected_joint_indices"] = json.load(jf)

    return config


def get_body_part(idx):
    """Map a joint index (0-542) to its body part name."""
    if idx < 468:
        return 'face'
    elif idx < 501:
        return 'pose'
    elif idx < 522:
        return 'left_hand'
    elif idx < 543:
        return 'right_hand'
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description="Extract joint probabilities and select top-K for next iteration"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (or use --model-dir for auto-detection)"
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Model directory (auto-detects best checkpoint)"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML for the trained model"
    )
    parser.add_argument(
        "--target-k", type=int, required=True,
        help="Number of joints to select for next stage"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON file with selected indices"
    )
    args = parser.parse_args()

    assert args.checkpoint or args.model_dir, \
        "Must provide either --checkpoint or --model-dir"

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_best_checkpoint(args.model_dir)

    # Load config
    config = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Model num_pose_points: {config['num_pose_points']}")

    # Load model
    # Import here to avoid import issues when running standalone
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.lightning_asl import SignClassificationLightning

    model = SignClassificationLightning.load_from_checkpoint(
        checkpoint_path, config=config
    )
    model.eval()

    # Extract joint probabilities
    assert hasattr(model, 'joint_pruning'), "Model does not have joint pruning enabled"
    joint_probs = model.joint_pruning.get_selection_probs().detach().cpu().numpy()
    n_model_joints = len(joint_probs)
    print(f"\nExtracted {n_model_joints} joint probabilities")
    print(f"  Range: [{joint_probs.min():.6f}, {joint_probs.max():.6f}]")
    print(f"  Mean: {joint_probs.mean():.6f}, Std: {joint_probs.std():.6f}")

    assert args.target_k <= n_model_joints, \
        f"target-k ({args.target_k}) > model joints ({n_model_joints})"

    # Get source indices (original 543-space)
    source_indices = config.get("selected_joint_indices", None)
    if source_indices is None:
        # Model was trained on all 543 joints
        source_indices = list(range(543))
    assert len(source_indices) == n_model_joints, \
        f"Source indices ({len(source_indices)}) != model joints ({n_model_joints})"

    # Sort by probability descending, select top K
    sorted_order = np.argsort(joint_probs)[::-1]
    top_k_local = sorted_order[:args.target_k]

    # Map back to original 543-space
    selected_original = sorted([source_indices[i] for i in top_k_local])
    selected_probs = [float(joint_probs[i]) for i in top_k_local]

    # Body part breakdown
    breakdown = {}
    for idx in selected_original:
        part = get_body_part(idx)
        breakdown[part] = breakdown.get(part, 0) + 1

    print(f"\nSelected top {args.target_k} joints:")
    print(f"  Body parts: {breakdown}")
    print(f"  Prob range of selected: [{min(selected_probs):.6f}, {max(selected_probs):.6f}]")

    # Show what was excluded
    excluded_local = sorted_order[args.target_k:]
    excluded_probs = [float(joint_probs[i]) for i in excluded_local]
    if excluded_probs:
        print(f"  Prob range of excluded: [{min(excluded_probs):.6f}, {max(excluded_probs):.6f}]")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(selected_original, f, indent=2)
    print(f"\nSaved {args.target_k} indices to {output_path}")

    # Also save probabilities CSV for the model (for analysis later)
    probs_csv = output_path.parent / f"probabilities_from_{n_model_joints}_to_{args.target_k}.csv"
    np.savetxt(probs_csv, joint_probs, delimiter=',', header='probability', comments='')
    print(f"Saved probabilities to {probs_csv}")


if __name__ == "__main__":
    main()
