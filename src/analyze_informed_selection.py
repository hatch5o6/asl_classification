"""
Paper figure generation for informed pose point selection experiments.

Collects results from all trained models (topk and iterative) and generates
publication-quality figures comparing approaches.

Usage:
    python src/analyze_informed_selection.py \
        --models-dir /path/to/models/informed_selection/ \
        --output-dir /path/to/paper_figures/

Expected directory structure:
    models/informed_selection/
        topk_270/figures/model_metrics.json, joint_probabilities.csv
        topk_100/figures/...
        topk_48/figures/...
        topk_24/figures/...
        topk_10/figures/...
        iterative_270/figures/...
        iterative_100/figures/...
        iterative_48/figures/...
        iterative_24/figures/...
        iterative_10/figures/...
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict


K_VALUES = [270, 100, 48, 24, 10]
APPROACHES = ['topk', 'iterative']

BODY_PARTS = {
    'face': {'start': 0, 'end': 468, 'color': '#88CCEE'},
    'pose': {'start': 468, 'end': 501, 'color': '#EE8866'},
    'left_hand': {'start': 501, 'end': 522, 'color': '#44AA99'},
    'right_hand': {'start': 522, 'end': 543, 'color': '#CC6677'},
}


def load_model_results(models_dir):
    """Load model_metrics.json and training logs from all models."""
    results = {}
    for approach in APPROACHES:
        results[approach] = {}
        for k in K_VALUES:
            model_name = f"{approach}_{k}"
            model_dir = Path(models_dir) / model_name

            if not model_dir.exists():
                print(f"  WARNING: {model_dir} not found, skipping")
                continue

            metrics_file = model_dir / "figures" / "model_metrics.json"
            probs_file = model_dir / "figures" / "joint_probabilities.csv"
            indices_file = None

            # Find the indices file
            if approach == 'topk':
                indices_file = f"data/informed_selection/topk/top_{k}_indices.json"
            else:
                indices_file = f"data/informed_selection/iterative/iter_{k}_indices.json"

            result = {
                'k': k,
                'approach': approach,
                'model_dir': str(model_dir),
            }

            # Load metrics
            if metrics_file.exists():
                with open(metrics_file) as f:
                    result['metrics'] = json.load(f)
            else:
                print(f"  WARNING: {metrics_file} not found")
                result['metrics'] = None

            # Load probabilities
            if probs_file.exists():
                result['probabilities'] = np.genfromtxt(
                    probs_file, delimiter=',', skip_header=1
                )
            else:
                result['probabilities'] = None

            # Load joint indices
            if indices_file and os.path.exists(indices_file):
                with open(indices_file) as f:
                    result['indices'] = json.load(f)
            else:
                result['indices'] = None

            # Find best val_acc from checkpoints
            ckpt_dir = model_dir / "checkpoints"
            if ckpt_dir.exists():
                best_acc = -1
                for f in os.listdir(ckpt_dir):
                    if f.endswith(".ckpt"):
                        try:
                            acc = float(f.split(".ckpt")[0].split("-val_acc=")[1])
                            best_acc = max(best_acc, acc)
                        except (IndexError, ValueError):
                            pass
                result['best_val_acc'] = best_acc if best_acc > 0 else None
            else:
                result['best_val_acc'] = None

            results[approach][k] = result

    return results


def plot_accuracy_vs_k(results, output_dir, baseline_acc=None, tslformer_acc=None):
    """
    Figure 1: Accuracy vs Number of Pose Points.
    Main result comparing Top-K vs Iterative approaches.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for approach, color, marker in [('topk', '#0077BB', 'o'), ('iterative', '#EE7733', 's')]:
        k_vals = []
        acc_vals = []
        for k in K_VALUES:
            if k in results[approach] and results[approach][k]['best_val_acc'] is not None:
                k_vals.append(k)
                acc_vals.append(results[approach][k]['best_val_acc'] * 100)

        if k_vals:
            label = 'Top-K Independent' if approach == 'topk' else 'Iterative Cascade'
            ax.plot(k_vals, acc_vals, color=color, marker=marker, markersize=10,
                    linewidth=2.5, label=label, zorder=5)
            for kv, av in zip(k_vals, acc_vals):
                ax.annotate(f'{av:.1f}%', (kv, av), textcoords="offset points",
                           xytext=(0, 12), ha='center', fontsize=9, color=color)

    # Baseline markers
    if baseline_acc is not None:
        ax.axhline(y=baseline_acc * 100, color='gray', linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'Baseline 543pts ({baseline_acc*100:.1f}%)')
    if tslformer_acc is not None:
        ax.scatter([50], [tslformer_acc * 100], color='purple', marker='D', s=120,
                  zorder=6, label=f'TSLFormer 50pts ({tslformer_acc*100:.1f}%)')

    ax.set_xlabel('Number of Pose Points', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.set_title('Classification Accuracy vs. Number of Pose Points\n'
                 'Informed Selection from L0 Pruning', fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xticks(K_VALUES + ([50] if tslformer_acc else []) + ([543] if baseline_acc else []))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    save_path = output_dir / "fig1_accuracy_vs_k.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig1_accuracy_vs_k.pdf", bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_body_part_composition(results, output_dir):
    """
    Figure 2: Body Part Composition per K (stacked bar chart).
    Shows face/pose/left_hand/right_hand breakdown at each K level.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, approach, title in zip(
        axes, APPROACHES,
        ['Top-K Independent Selection', 'Iterative Cascade Selection']
    ):
        face_counts = []
        pose_counts = []
        lh_counts = []
        rh_counts = []
        valid_k = []

        for k in K_VALUES:
            if k not in results[approach]:
                continue
            indices = results[approach][k].get('indices')
            if indices is None:
                continue

            valid_k.append(k)
            face = sum(1 for i in indices if i < 468)
            pose = sum(1 for i in indices if 468 <= i < 501)
            lh = sum(1 for i in indices if 501 <= i < 522)
            rh = sum(1 for i in indices if 522 <= i < 543)
            face_counts.append(face)
            pose_counts.append(pose)
            lh_counts.append(lh)
            rh_counts.append(rh)

        if not valid_k:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        x = np.arange(len(valid_k))
        width = 0.6

        ax.bar(x, face_counts, width, label='Face',
               color=BODY_PARTS['face']['color'], edgecolor='black', linewidth=0.5)
        ax.bar(x, pose_counts, width, bottom=face_counts, label='Pose',
               color=BODY_PARTS['pose']['color'], edgecolor='black', linewidth=0.5)
        bottom2 = [f + p for f, p in zip(face_counts, pose_counts)]
        ax.bar(x, lh_counts, width, bottom=bottom2, label='Left Hand',
               color=BODY_PARTS['left_hand']['color'], edgecolor='black', linewidth=0.5)
        bottom3 = [b + l for b, l in zip(bottom2, lh_counts)]
        ax.bar(x, rh_counts, width, bottom=bottom3, label='Right Hand',
               color=BODY_PARTS['right_hand']['color'], edgecolor='black', linewidth=0.5)

        # Add total count labels
        for i, k in enumerate(valid_k):
            total = face_counts[i] + pose_counts[i] + lh_counts[i] + rh_counts[i]
            ax.text(i, total + 2, str(total), ha='center', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in valid_k])
        ax.set_xlabel('Number of Selected Points', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Number of Joints', fontsize=12)
    fig.suptitle('Body Part Composition at Each Selection Level',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = output_dir / "fig2_body_part_composition.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig2_body_part_composition.pdf", bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_importance_distributions(results, output_dir):
    """
    Figure 3: Joint Importance Distribution per Model (violin/box plots).
    Shows the distribution of learned L0 probabilities at each stage.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, approach, title in zip(
        axes, APPROACHES,
        ['Top-K Independent', 'Iterative Cascade']
    ):
        data = []
        labels = []
        for k in K_VALUES:
            if k not in results[approach]:
                continue
            probs = results[approach][k].get('probabilities')
            if probs is None:
                continue
            data.append(probs)
            labels.append(f'K={k}\n(n={len(probs)})')

        if not data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#4477AA')
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('#EE6677')
        parts['cmedians'].set_color('#228833')

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_xlabel('Model Configuration', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Keep Probability', fontsize=12)
    fig.suptitle('Distribution of Learned Joint Importance\n'
                 '(L0 Keep Probabilities per Model)',
                 fontsize=15, fontweight='bold', y=1.04)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#EE6677', linewidth=2, label='Mean'),
        Line2D([0], [0], color='#228833', linewidth=2, label='Median'),
        Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)'),
    ]
    axes[1].legend(handles=legend_elements, loc='lower left', fontsize=9)

    plt.tight_layout()
    save_path = output_dir / "fig3_importance_distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig3_importance_distributions.pdf", bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_overlap_heatmap(results, output_dir):
    """
    Figure 4: Joint Selection Overlap Heatmap (Jaccard similarity).
    Shows agreement between models at different K values and approaches.
    """
    # Collect all index sets
    model_names = []
    index_sets = []

    for approach in APPROACHES:
        for k in K_VALUES:
            if k not in results[approach]:
                continue
            indices = results[approach][k].get('indices')
            if indices is None:
                continue
            label = f"{'TK' if approach == 'topk' else 'IT'}-{k}"
            model_names.append(label)
            index_sets.append(set(indices))

    if len(index_sets) < 2:
        print("Not enough models for overlap heatmap, skipping")
        return

    n = len(model_names)
    jaccard_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = len(index_sets[i] & index_sets[j])
            union = len(index_sets[i] | index_sets[j])
            jaccard_matrix[i][j] = intersection / union if union > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(jaccard_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=model_names, yticklabels=model_names,
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Jaccard Similarity'})

    ax.set_title('Joint Selection Overlap Between Models\n'
                 '(Jaccard Similarity: TK=Top-K, IT=Iterative)',
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    save_path = output_dir / "fig4_overlap_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig4_overlap_heatmap.pdf", bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_cascade_flow(results, output_dir):
    """
    Figure 5: Iterative Cascade Joint Retention.
    Shows which body parts survive each pruning stage.
    """
    if 'iterative' not in results:
        return

    stages = []
    for k in K_VALUES:
        if k not in results['iterative']:
            continue
        indices = results['iterative'][k].get('indices')
        if indices is None:
            continue
        stages.append((k, indices))

    if len(stages) < 2:
        print("Not enough iterative stages for cascade flow, skipping")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    face_vals = []
    pose_vals = []
    lh_vals = []
    rh_vals = []
    k_labels = []

    for k, indices in stages:
        k_labels.append(str(k))
        face_vals.append(sum(1 for i in indices if i < 468))
        pose_vals.append(sum(1 for i in indices if 468 <= i < 501))
        lh_vals.append(sum(1 for i in indices if 501 <= i < 522))
        rh_vals.append(sum(1 for i in indices if 522 <= i < 543))

    x = range(len(k_labels))

    ax.fill_between(x, 0, face_vals, alpha=0.7,
                    color=BODY_PARTS['face']['color'], label='Face')
    bottom = face_vals
    ax.fill_between(x, bottom, [b + p for b, p in zip(bottom, pose_vals)], alpha=0.7,
                    color=BODY_PARTS['pose']['color'], label='Pose')
    bottom = [b + p for b, p in zip(bottom, pose_vals)]
    ax.fill_between(x, bottom, [b + l for b, l in zip(bottom, lh_vals)], alpha=0.7,
                    color=BODY_PARTS['left_hand']['color'], label='Left Hand')
    bottom = [b + l for b, l in zip(bottom, lh_vals)]
    ax.fill_between(x, bottom, [b + r for b, r in zip(bottom, rh_vals)], alpha=0.7,
                    color=BODY_PARTS['right_hand']['color'], label='Right Hand')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{kl} pts' for kl in k_labels], fontsize=12)
    ax.set_xlabel('Cascade Stage', fontsize=14)
    ax.set_ylabel('Number of Joints', fontsize=14)
    ax.set_title('Iterative Cascade: Joint Retention by Body Part\n'
                 '543 \u2192 270 \u2192 100 \u2192 48 \u2192 24 \u2192 10',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "fig5_cascade_flow.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig5_cascade_flow.pdf", bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_results_table(results, output_dir):
    """
    Generate a CSV results table with all model metrics.
    """
    rows = [['Approach', 'K', 'Val Accuracy', 'Active Joints', 'Pruning Ratio',
             'Face', 'Pose', 'Left Hand', 'Right Hand',
             'Prob Mean', 'Prob Std', 'Prob Min', 'Prob Max']]

    for approach in APPROACHES:
        for k in K_VALUES:
            if k not in results[approach]:
                continue
            r = results[approach][k]
            acc = r.get('best_val_acc', None)
            m = r.get('metrics', {}) or {}
            bp = m.get('source_body_part_breakdown', {})
            ps = m.get('probability_stats', {})

            rows.append([
                'Top-K' if approach == 'topk' else 'Iterative',
                k,
                f"{acc*100:.2f}" if acc else 'N/A',
                m.get('num_active_joints', 'N/A'),
                f"{m.get('pruning_ratio', 0):.3f}" if m.get('pruning_ratio') is not None else 'N/A',
                bp.get('face', 'N/A'),
                bp.get('pose', 'N/A'),
                bp.get('left_hand', 'N/A'),
                bp.get('right_hand', 'N/A'),
                f"{ps.get('mean', 0):.4f}" if ps.get('mean') is not None else 'N/A',
                f"{ps.get('std', 0):.4f}" if ps.get('std') is not None else 'N/A',
                f"{ps.get('min', 0):.4f}" if ps.get('min') is not None else 'N/A',
                f"{ps.get('max', 0):.4f}" if ps.get('max') is not None else 'N/A',
            ])

    csv_path = output_dir / "results_table.csv"
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Saved results table to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures for informed selection experiments"
    )
    parser.add_argument(
        "--models-dir", type=str, required=True,
        help="Root directory containing all informed selection model dirs"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save paper figures"
    )
    parser.add_argument(
        "--baseline-acc", type=float, default=None,
        help="Baseline accuracy (543 points, no selection) for reference line"
    )
    parser.add_argument(
        "--tslformer-acc", type=float, default=None,
        help="TSLFormer accuracy (50 fixed points) for reference marker"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("INFORMED SELECTION ANALYSIS")
    print("=" * 60)
    print(f"Models dir: {args.models_dir}")
    print(f"Output dir: {args.output_dir}")

    # Load all results
    print("\nLoading model results...")
    results = load_model_results(args.models_dir)

    # Summary
    print("\nAvailable results:")
    for approach in APPROACHES:
        for k in K_VALUES:
            if k in results[approach]:
                r = results[approach][k]
                acc = r.get('best_val_acc')
                acc_str = f"{acc*100:.2f}%" if acc else "N/A"
                has_metrics = r.get('metrics') is not None
                has_probs = r.get('probabilities') is not None
                has_indices = r.get('indices') is not None
                print(f"  {approach}_{k}: acc={acc_str}, "
                      f"metrics={'Y' if has_metrics else 'N'}, "
                      f"probs={'Y' if has_probs else 'N'}, "
                      f"indices={'Y' if has_indices else 'N'}")

    # Generate figures
    print("\nGenerating figures...")

    print("\n[1/6] Accuracy vs K...")
    plot_accuracy_vs_k(results, output_dir, args.baseline_acc, args.tslformer_acc)

    print("[2/6] Body part composition...")
    plot_body_part_composition(results, output_dir)

    print("[3/6] Importance distributions...")
    plot_importance_distributions(results, output_dir)

    print("[4/6] Overlap heatmap...")
    plot_overlap_heatmap(results, output_dir)

    print("[5/6] Cascade flow...")
    plot_cascade_flow(results, output_dir)

    print("[6/6] Results table...")
    generate_results_table(results, output_dir)

    print(f"\n{'='*60}")
    print(f"All figures saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
