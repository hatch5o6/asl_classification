"""
Visualization utilities for joint pruning analysis.
Creates publication-quality figures for paper.

Usage:
    python visualize_joint_pruning.py --checkpoint path/to/checkpoint.ckpt --output figures/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from lightning_asl import SignClassificationLightning
import yaml


def plot_joint_importance_heatmap(joint_probs, save_path, joint_names=None):
    """
    Create heatmap showing importance of each joint.

    Args:
        joint_probs: (543,) tensor of probabilities
        save_path: Where to save the figure
        joint_names: Optional list of joint names
    """
    # Reshape into groups: face (468), pose (33), left_hand (21), right_hand (21)
    face_probs = joint_probs[:468]
    pose_probs = joint_probs[468:501]
    left_hand_probs = joint_probs[501:522]
    right_hand_probs = joint_probs[522:543]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Joint Importance by Body Part', fontsize=16, fontweight='bold')

    # Face landmarks
    ax = axes[0, 0]
    face_grid = face_probs.reshape(26, 18).detach().cpu().numpy()  # Approximate 2D layout
    sns.heatmap(face_grid, ax=ax, cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'Keep Probability'})
    ax.set_title(f'Face Landmarks (n=468)\nMean prob: {face_probs.mean():.3f}')

    # Pose landmarks
    ax = axes[0, 1]
    pose_display = pose_probs.detach().cpu().numpy().reshape(1, -1)
    sns.heatmap(pose_display, ax=ax, cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'Keep Probability'})
    ax.set_title(f'Pose Landmarks (n=33)\nMean prob: {pose_probs.mean():.3f}')
    ax.set_xlabel('Joint Index')

    # Left hand
    ax = axes[1, 0]
    left_hand_grid = left_hand_probs.detach().cpu().numpy().reshape(3, 7)
    sns.heatmap(left_hand_grid, ax=ax, cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'Keep Probability'})
    ax.set_title(f'Left Hand (n=21)\nMean prob: {left_hand_probs.mean():.3f}')

    # Right hand
    ax = axes[1, 1]
    right_hand_grid = right_hand_probs.detach().cpu().numpy().reshape(3, 7)
    sns.heatmap(right_hand_grid, ax=ax, cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'Keep Probability'})
    ax.set_title(f'Right Hand (n=21)\nMean prob: {right_hand_probs.mean():.3f}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {save_path}")


def plot_top_k_joints_bar(joint_probs, save_path, k=50):
    """
    Bar plot showing top-K most important joints.
    """
    top_k_values, top_k_indices = torch.topk(joint_probs, k=k)

    plt.figure(figsize=(16, 6))
    colors = []
    labels = []

    for idx in top_k_indices:
        if idx < 468:
            colors.append('lightblue')
            labels.append(f'Face-{idx}')
        elif idx < 501:
            colors.append('orange')
            labels.append(f'Pose-{idx-468}')
        elif idx < 522:
            colors.append('green')
            labels.append(f'LH-{idx-501}')
        else:
            colors.append('red')
            labels.append(f'RH-{idx-522}')

    plt.bar(range(k), top_k_values.detach().cpu().numpy(), color=colors)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Keep Probability', fontsize=12)
    plt.title(f'Top-{k} Most Important Joints', fontsize=14, fontweight='bold')
    plt.legend(['Threshold (0.5)', 'Face', 'Pose', 'Left Hand', 'Right Hand'])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar plot to {save_path}")


def plot_pruning_summary(joint_probs, save_path):
    """
    Summary statistics plot with distribution.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram of probabilities
    ax = axes[0]
    ax.hist(joint_probs.detach().cpu().numpy(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Keep Probability', fontsize=12)
    ax.set_ylabel('Number of Joints', fontsize=12)
    ax.set_title('Distribution of Joint Probabilities', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # CDF
    ax = axes[1]
    sorted_probs = torch.sort(joint_probs, descending=True)[0].detach().cpu().numpy()
    ax.plot(range(len(sorted_probs)), sorted_probs, linewidth=2, color='darkgreen')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Joint Rank (sorted)', fontsize=12)
    ax.set_ylabel('Keep Probability', fontsize=12)
    ax.set_title('Sorted Joint Importance', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Summary by body part
    ax = axes[2]
    face_active = (joint_probs[:468] > 0.5).sum().item()
    pose_active = (joint_probs[468:501] > 0.5).sum().item()
    lh_active = (joint_probs[501:522] > 0.5).sum().item()
    rh_active = (joint_probs[522:543] > 0.5).sum().item()

    parts = ['Face\n(468)', 'Pose\n(33)', 'Left Hand\n(21)', 'Right Hand\n(21)']
    active = [face_active, pose_active, lh_active, rh_active]
    total = [468, 33, 21, 21]

    x = range(len(parts))
    ax.bar(x, active, color=['lightblue', 'orange', 'green', 'red'], alpha=0.7, label='Active')
    ax.plot(x, total, 'ko-', linewidth=2, markersize=8, label='Total')
    ax.set_xticks(x)
    ax.set_xticklabels(parts)
    ax.set_ylabel('Number of Joints', fontsize=12)
    ax.set_title('Active Joints by Body Part', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {save_path}")


def analyze_checkpoint(checkpoint_path, output_dir, config_path):
    """
    Load checkpoint and create all visualizations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model = SignClassificationLightning.load_from_checkpoint(
        checkpoint_path,
        config=config
    )
    model.eval()

    # Get joint probabilities
    if hasattr(model, 'joint_pruning'):
        joint_probs = model.joint_pruning.get_selection_probs()

        print(f"\n{'='*60}")
        print("JOINT PRUNING ANALYSIS")
        print(f"{'='*60}")

        summary = model.joint_pruning.get_summary()
        print(f"\nActive joints: {summary['num_active']}/{summary['num_total']}")
        print(f"Pruning ratio: {summary['pruning_ratio']:.1%}")
        print(f"Avg probability: {summary['avg_prob']:.3f}")
        print(f"Min probability: {summary['min_prob']:.3f}")
        print(f"Max probability: {summary['max_prob']:.3f}")

        # Create visualizations
        print(f"\n{'='*60}")
        print("GENERATING FIGURES")
        print(f"{'='*60}\n")

        plot_joint_importance_heatmap(
            joint_probs,
            output_dir / "joint_importance_heatmap.png"
        )

        plot_top_k_joints_bar(
            joint_probs,
            output_dir / "top_50_joints.png",
            k=50
        )

        plot_pruning_summary(
            joint_probs,
            output_dir / "pruning_summary.png"
        )

        # Save probabilities to CSV for custom analysis
        np.savetxt(
            output_dir / "joint_probabilities.csv",
            joint_probs.detach().cpu().numpy(),
            delimiter=',',
            header='probability',
            comments=''
        )
        print(f"\nSaved raw probabilities to {output_dir / 'joint_probabilities.csv'}")

        print(f"\n{'='*60}")
        print("DONE! Check output directory for figures.")
        print(f"{'='*60}\n")

    else:
        print("ERROR: Model does not have joint pruning enabled!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize joint pruning results")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="figures/joint_pruning", help="Output directory")

    args = parser.parse_args()

    analyze_checkpoint(args.checkpoint, args.output, args.config)
