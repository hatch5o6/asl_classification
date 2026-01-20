"""
Visualization utilities for joint pruning analysis.
Creates publication-quality figures for paper.

Usage:
    python visualize_joint_pruning.py --checkpoint path/to/checkpoint.ckpt --output figures/
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import argparse
from lightning_asl import SignClassificationLightning
import yaml


# Joint region definitions for MediaPipe Holistic
JOINT_REGIONS = {
    'face': {'start': 0, 'end': 468, 'count': 468, 'color': 'lightblue'},
    'pose': {'start': 468, 'end': 501, 'count': 33, 'color': 'orange'},
    'left_hand': {'start': 501, 'end': 522, 'count': 21, 'color': 'green'},
    'right_hand': {'start': 522, 'end': 543, 'count': 21, 'color': 'red'},
}

# Hand landmark names (MediaPipe convention)
HAND_LANDMARKS = [
    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
    'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
    'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
    'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
]

# Key pose landmarks (subset for readability)
POSE_LANDMARKS = [
    'NOSE', 'L_EYE_IN', 'L_EYE', 'L_EYE_OUT', 'R_EYE_IN', 'R_EYE', 'R_EYE_OUT',
    'L_EAR', 'R_EAR', 'MOUTH_L', 'MOUTH_R', 'L_SHOULDER', 'R_SHOULDER',
    'L_ELBOW', 'R_ELBOW', 'L_WRIST', 'R_WRIST', 'L_PINKY', 'R_PINKY',
    'L_INDEX', 'R_INDEX', 'L_THUMB', 'R_THUMB', 'L_HIP', 'R_HIP',
    'L_KNEE', 'R_KNEE', 'L_ANKLE', 'R_ANKLE', 'L_HEEL', 'R_HEEL',
    'L_FOOT', 'R_FOOT'
]


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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Joint Importance by Body Part\n(Green = Keep, Red = Prune)',
                 fontsize=16, fontweight='bold')

    # Face landmarks (468 points arranged in 26x18 grid)
    ax = axes[0, 0]
    face_grid = face_probs.reshape(26, 18).detach().cpu().numpy()
    active_face = (face_probs > 0.5).sum().item()
    sns.heatmap(face_grid, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Keep Probability'},
                xticklabels=5, yticklabels=5)
    ax.set_title(f'Face Landmarks (n=468)\nActive: {active_face}/468 | Mean prob: {face_probs.mean():.3f}')
    ax.set_xlabel('Column Index (0-17)')
    ax.set_ylabel('Row Index (0-25)')

    # Pose landmarks (33 body keypoints)
    ax = axes[0, 1]
    pose_display = pose_probs.detach().cpu().numpy().reshape(1, -1)
    active_pose = (pose_probs > 0.5).sum().item()
    # Use abbreviated pose landmark names for x-axis
    pose_labels = [POSE_LANDMARKS[i][:6] if i < len(POSE_LANDMARKS) else str(i)
                   for i in range(33)]
    sns.heatmap(pose_display, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Keep Probability'},
                xticklabels=pose_labels, yticklabels=[''])
    ax.set_title(f'Pose Landmarks (n=33)\nActive: {active_pose}/33 | Mean prob: {pose_probs.mean():.3f}')
    ax.set_xlabel('Body Keypoint')
    ax.tick_params(axis='x', rotation=90, labelsize=7)

    # Left hand (21 landmarks: wrist + 4 fingers × 5 joints each)
    ax = axes[1, 0]
    left_hand_grid = left_hand_probs.detach().cpu().numpy().reshape(3, 7)
    active_lh = (left_hand_probs > 0.5).sum().item()
    # Create meaningful labels: columns are fingers, rows are joint types
    finger_labels = ['WRIST', 'THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY', '']
    joint_labels = ['Base', 'Middle', 'Tip']
    sns.heatmap(left_hand_grid, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Keep Probability'},
                xticklabels=finger_labels, yticklabels=joint_labels,
                annot=True, fmt='.2f', annot_kws={'size': 8})
    ax.set_title(f'Left Hand (n=21)\nActive: {active_lh}/21 | Mean prob: {left_hand_probs.mean():.3f}')
    ax.set_xlabel('Finger')
    ax.set_ylabel('Joint Position')

    # Right hand (21 landmarks)
    ax = axes[1, 1]
    right_hand_grid = right_hand_probs.detach().cpu().numpy().reshape(3, 7)
    active_rh = (right_hand_probs > 0.5).sum().item()
    sns.heatmap(right_hand_grid, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Keep Probability'},
                xticklabels=finger_labels, yticklabels=joint_labels,
                annot=True, fmt='.2f', annot_kws={'size': 8})
    ax.set_title(f'Right Hand (n=21)\nActive: {active_rh}/21 | Mean prob: {right_hand_probs.mean():.3f}')
    ax.set_xlabel('Finger')
    ax.set_ylabel('Joint Position')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {save_path}")


def plot_top_k_joints_bar(joint_probs, save_path, k=50):
    """
    Bar plot showing top-K most important joints with proper color legend.
    """
    top_k_values, top_k_indices = torch.topk(joint_probs, k=k)

    fig, ax = plt.subplots(figsize=(16, 7))
    colors = []
    labels = []

    # Categorize each joint by body region
    for idx in top_k_indices:
        idx_val = idx.item()
        if idx_val < 468:
            colors.append(JOINT_REGIONS['face']['color'])
            labels.append(f'Face-{idx_val}')
        elif idx_val < 501:
            colors.append(JOINT_REGIONS['pose']['color'])
            pose_idx = idx_val - 468
            pose_name = POSE_LANDMARKS[pose_idx] if pose_idx < len(POSE_LANDMARKS) else str(pose_idx)
            labels.append(f'Pose:{pose_name}')
        elif idx_val < 522:
            colors.append(JOINT_REGIONS['left_hand']['color'])
            hand_idx = idx_val - 501
            hand_name = HAND_LANDMARKS[hand_idx] if hand_idx < len(HAND_LANDMARKS) else str(hand_idx)
            labels.append(f'LH:{hand_name}')
        else:
            colors.append(JOINT_REGIONS['right_hand']['color'])
            hand_idx = idx_val - 522
            hand_name = HAND_LANDMARKS[hand_idx] if hand_idx < len(HAND_LANDMARKS) else str(hand_idx)
            labels.append(f'RH:{hand_name}')

    # Plot bars
    bars = ax.bar(range(k), top_k_values.detach().cpu().numpy(), color=colors, edgecolor='black', linewidth=0.5)

    # Add threshold line
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, zorder=5)
    ax.text(k + 0.5, 0.5, 'Threshold\n(0.5)', fontsize=9, va='center', ha='left')

    # Create proper legend with color patches
    legend_patches = [
        mpatches.Patch(color=JOINT_REGIONS['face']['color'], label=f"Face (468 joints)"),
        mpatches.Patch(color=JOINT_REGIONS['pose']['color'], label=f"Pose (33 joints)"),
        mpatches.Patch(color=JOINT_REGIONS['left_hand']['color'], label=f"Left Hand (21 joints)"),
        mpatches.Patch(color=JOINT_REGIONS['right_hand']['color'], label=f"Right Hand (21 joints)"),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=10, title='Body Region')

    # Set labels with joint names on x-axis (rotated for readability)
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=90, fontsize=7, ha='center')
    ax.set_xlabel('Joint (by importance rank)', fontsize=12)
    ax.set_ylabel('Keep Probability', fontsize=12)
    ax.set_ylim(0, 1.05)

    # Count joints above threshold by region
    above_threshold = top_k_values > 0.5
    face_count = sum(1 for i, idx in enumerate(top_k_indices) if idx < 468 and above_threshold[i])
    pose_count = sum(1 for i, idx in enumerate(top_k_indices) if 468 <= idx < 501 and above_threshold[i])
    lh_count = sum(1 for i, idx in enumerate(top_k_indices) if 501 <= idx < 522 and above_threshold[i])
    rh_count = sum(1 for i, idx in enumerate(top_k_indices) if idx >= 522 and above_threshold[i])

    ax.set_title(
        f'Top-{k} Most Important Joints (by keep probability)\n'
        f'Above threshold: Face={face_count}, Pose={pose_count}, LH={lh_count}, RH={rh_count}',
        fontsize=14, fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar plot to {save_path}")


def plot_pruning_summary(joint_probs, save_path):
    """
    Summary statistics plot with distribution, sorted importance curve, and body part breakdown.
    """
    probs_np = joint_probs.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Calculate overall statistics
    total_joints = len(probs_np)
    active_joints = (joint_probs > 0.5).sum().item()
    pruned_joints = total_joints - active_joints
    pruning_ratio = pruned_joints / total_joints

    fig.suptitle(
        f'Joint Pruning Summary: {active_joints}/{total_joints} joints active ({pruning_ratio:.1%} pruned)',
        fontsize=14, fontweight='bold', y=1.02
    )

    # Panel 1: Histogram of probabilities
    ax = axes[0]
    n, bins, patches = ax.hist(probs_np, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2)

    # Color bars below/above threshold
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        if left_edge < 0.5:
            patch.set_facecolor('salmon')
        else:
            patch.set_facecolor('lightgreen')

    # Add text annotations
    below_threshold = (probs_np < 0.5).sum()
    above_threshold = (probs_np >= 0.5).sum()
    ax.text(0.25, ax.get_ylim()[1] * 0.9, f'Pruned\n({below_threshold})',
            ha='center', fontsize=11, color='darkred', fontweight='bold')
    ax.text(0.75, ax.get_ylim()[1] * 0.9, f'Kept\n({above_threshold})',
            ha='center', fontsize=11, color='darkgreen', fontweight='bold')

    ax.set_xlabel('Keep Probability', fontsize=12)
    ax.set_ylabel('Number of Joints', fontsize=12)
    ax.set_title('Distribution of Joint Keep Probabilities\n(Salmon = pruned, Green = kept)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)

    # Panel 2: Sorted importance curve (inverse CDF)
    ax = axes[1]
    sorted_probs = torch.sort(joint_probs, descending=True)[0].detach().cpu().numpy()
    ax.plot(range(len(sorted_probs)), sorted_probs, linewidth=2, color='darkgreen')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2)
    ax.fill_between(range(len(sorted_probs)), sorted_probs, 0.5,
                    where=sorted_probs >= 0.5, alpha=0.3, color='green', label='Above threshold')
    ax.fill_between(range(len(sorted_probs)), sorted_probs, 0.5,
                    where=sorted_probs < 0.5, alpha=0.3, color='red', label='Below threshold')

    # Mark threshold crossing point
    threshold_idx = (sorted_probs >= 0.5).sum()
    ax.axvline(x=threshold_idx, color='orange', linestyle=':', linewidth=2)
    ax.text(threshold_idx + 5, 0.55, f'{threshold_idx} joints\nabove threshold',
            fontsize=10, color='orange')

    ax.set_xlabel('Joint Rank (1 = most important)', fontsize=12)
    ax.set_ylabel('Keep Probability', fontsize=12)
    ax.set_title('Sorted Joint Importance Curve\n(joints ranked by learned importance)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(sorted_probs))
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Panel 3: Summary by body part with percentages
    ax = axes[2]
    face_active = (joint_probs[:468] > 0.5).sum().item()
    pose_active = (joint_probs[468:501] > 0.5).sum().item()
    lh_active = (joint_probs[501:522] > 0.5).sum().item()
    rh_active = (joint_probs[522:543] > 0.5).sum().item()

    parts = ['Face', 'Pose', 'Left Hand', 'Right Hand']
    active = [face_active, pose_active, lh_active, rh_active]
    total = [468, 33, 21, 21]
    colors = [JOINT_REGIONS['face']['color'], JOINT_REGIONS['pose']['color'],
              JOINT_REGIONS['left_hand']['color'], JOINT_REGIONS['right_hand']['color']]

    x = np.arange(len(parts))
    width = 0.35

    bars1 = ax.bar(x - width/2, active, width, color=colors, alpha=0.9, label='Active Joints', edgecolor='black')
    bars2 = ax.bar(x + width/2, total, width, color=colors, alpha=0.3, label='Total Joints', edgecolor='black', hatch='//')

    # Add percentage labels on active bars
    for bar, act, tot in zip(bars1, active, total):
        pct = (act / tot) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add count labels
    for bar, act, tot in zip(bars1, active, total):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{act}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}\n({t} total)' for p, t in zip(parts, total)])
    ax.set_ylabel('Number of Joints', fontsize=12)
    ax.set_title('Active vs Total Joints by Body Region\n(percentage = retention rate)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
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
