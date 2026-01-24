"""
Analyze pruning sweep results.

Compares different L0 configurations:
- How many joints are kept vs pruned
- Accuracy vs sparsity tradeoff
- Which joints are consistently important across experiments

Usage:
    python src/analyze_pruning_sweep.py --models_dir /path/to/models
"""

import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


# Joint region definitions (from MediaPipe)
JOINT_REGIONS = {
    'face': list(range(0, 468)),
    'pose': list(range(468, 501)),
    'left_hand': list(range(501, 522)),
    'right_hand': list(range(522, 543)),
}

# Important pose landmarks
POSE_LANDMARKS = {
    468: 'nose', 469: 'left_eye_inner', 470: 'left_eye', 471: 'left_eye_outer',
    472: 'right_eye_inner', 473: 'right_eye', 474: 'right_eye_outer',
    475: 'left_ear', 476: 'right_ear', 477: 'mouth_left', 478: 'mouth_right',
    479: 'left_shoulder', 480: 'right_shoulder', 481: 'left_elbow', 482: 'right_elbow',
    483: 'left_wrist', 484: 'right_wrist', 485: 'left_pinky', 486: 'right_pinky',
    487: 'left_index', 488: 'right_index', 489: 'left_thumb', 490: 'right_thumb',
    491: 'left_hip', 492: 'right_hip', 493: 'left_knee', 494: 'right_knee',
    495: 'left_ankle', 496: 'right_ankle', 497: 'left_heel', 498: 'right_heel',
    499: 'left_foot_index', 500: 'right_foot_index',
}


def find_best_checkpoint(checkpoints_dir):
    """Find checkpoint with highest validation accuracy."""
    best_ckpt = None
    best_acc = -1
    for f in os.listdir(checkpoints_dir):
        if f.endswith('.ckpt'):
            try:
                acc = float(f.split('val_acc=')[1].split('.ckpt')[0])
                if acc > best_acc:
                    best_acc = acc
                    best_ckpt = os.path.join(checkpoints_dir, f)
            except:
                continue
    return best_ckpt, best_acc


def load_joint_probs(checkpoint_path):
    """Load joint probabilities from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Find joint pruning logits
    logits_key = None
    for key in state_dict.keys():
        if 'joint_pruning' in key and 'logits' in key:
            logits_key = key
            break

    if logits_key is None:
        return None

    logits = state_dict[logits_key]
    probs = torch.sigmoid(logits).numpy()
    return probs


def analyze_single_model(model_dir, threshold=0.5):
    """Analyze a single model's pruning results."""
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        return None

    best_ckpt, val_acc = find_best_checkpoint(checkpoints_dir)
    if best_ckpt is None:
        return None

    probs = load_joint_probs(best_ckpt)
    if probs is None:
        return None

    # Count joints above threshold (kept)
    num_kept = (probs >= threshold).sum()
    num_pruned = (probs < threshold).sum()

    # Analyze by region
    region_stats = {}
    for region, indices in JOINT_REGIONS.items():
        region_probs = probs[indices]
        region_stats[region] = {
            'kept': (region_probs >= threshold).sum(),
            'total': len(indices),
            'mean_prob': region_probs.mean(),
            'std_prob': region_probs.std(),
        }

    return {
        'model_name': os.path.basename(model_dir),
        'val_acc': val_acc,
        'probs': probs,
        'num_kept': num_kept,
        'num_pruned': num_pruned,
        'sparsity': num_pruned / len(probs),
        'region_stats': region_stats,
    }


def main(models_dir, output_dir=None):
    """Analyze all pruning sweep models."""

    # Find all pruning sweep models
    sweep_models = glob(os.path.join(models_dir, 'pruning_sweep_*'))

    if not sweep_models:
        print(f"No pruning sweep models found in {models_dir}")
        print("Looking for models matching 'pruning_sweep_*'")
        return

    print(f"Found {len(sweep_models)} sweep models")

    # Analyze each model
    results = []
    all_probs = {}

    for model_dir in sorted(sweep_models):
        print(f"\nAnalyzing: {os.path.basename(model_dir)}")
        result = analyze_single_model(model_dir)
        if result:
            results.append(result)
            all_probs[result['model_name']] = result['probs']
            print(f"  Val Acc: {result['val_acc']:.4f}")
            print(f"  Joints Kept: {result['num_kept']}/543 ({100*(1-result['sparsity']):.1f}%)")
            print(f"  Sparsity: {result['sparsity']:.1%}")

    if not results:
        print("No results to analyze")
        return

    # Create summary DataFrame
    summary_data = []
    for r in results:
        row = {
            'Model': r['model_name'],
            'Val Acc': r['val_acc'],
            'Joints Kept': r['num_kept'],
            'Sparsity': r['sparsity'],
        }
        for region, stats in r['region_stats'].items():
            row[f'{region}_kept'] = stats['kept']
            row[f'{region}_mean_prob'] = stats['mean_prob']
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

    # Save outputs
    if output_dir is None:
        output_dir = os.path.join(models_dir, 'pruning_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Save summary CSV
    df.to_csv(os.path.join(output_dir, 'sweep_summary.csv'), index=False)

    # Plot: Accuracy vs Sparsity
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        ax.scatter(r['sparsity'], r['val_acc'], s=100, label=r['model_name'])
        ax.annotate(r['model_name'].replace('pruning_sweep_', ''),
                   (r['sparsity'], r['val_acc']),
                   textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.set_xlabel('Sparsity (fraction pruned)')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Accuracy vs Sparsity Tradeoff')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_sparsity.png'), dpi=150)
    plt.close()

    # Plot: Region-wise pruning
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    regions = ['face', 'pose', 'left_hand', 'right_hand']

    for ax, region in zip(axes.flat, regions):
        model_names = [r['model_name'].replace('pruning_sweep_', '') for r in results]
        kept_counts = [r['region_stats'][region]['kept'] for r in results]
        total = results[0]['region_stats'][region]['total']

        bars = ax.bar(model_names, kept_counts)
        ax.axhline(y=total, color='r', linestyle='--', label=f'Total: {total}')
        ax.set_xlabel('Model')
        ax.set_ylabel('Joints Kept')
        ax.set_title(f'{region.replace("_", " ").title()} ({total} joints)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'region_pruning.png'), dpi=150)
    plt.close()

    # Find consistently important joints across all models
    if len(all_probs) > 1:
        prob_matrix = np.stack([all_probs[k] for k in sorted(all_probs.keys())])
        mean_probs = prob_matrix.mean(axis=0)
        std_probs = prob_matrix.std(axis=0)

        # Top 50 most consistently kept joints
        top_50_idx = np.argsort(mean_probs)[-50:][::-1]

        print("\n" + "="*80)
        print("TOP 50 MOST IMPORTANT JOINTS (by mean probability across experiments)")
        print("="*80)
        for i, idx in enumerate(top_50_idx):
            region = 'face' if idx < 468 else 'pose' if idx < 501 else 'left_hand' if idx < 522 else 'right_hand'
            name = POSE_LANDMARKS.get(idx, f'{region}_{idx}')
            print(f"{i+1:2d}. Joint {idx:3d} ({name:20s}): mean={mean_probs[idx]:.4f}, std={std_probs[idx]:.4f}")

        # Save joint importance
        importance_df = pd.DataFrame({
            'joint_idx': range(543),
            'mean_prob': mean_probs,
            'std_prob': std_probs,
        })
        importance_df.to_csv(os.path.join(output_dir, 'joint_importance.csv'), index=False)

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    import getpass
    default_models_dir = f'/home/{getpass.getuser()}/groups/grp_asl_classification/nobackup/archive/SLR/models'

    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', '-m', type=str,
                       default=default_models_dir,
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Directory to save analysis outputs')
    args = parser.parse_args()

    main(args.models_dir, args.output_dir)
