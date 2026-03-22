"""
Investigates the gap between ASL Citizen validation and test set performance.

Key findings this script surfaces:
  - Splits are signer-independent (train/val/test signers are disjoint).
  - Val has 6 signers; test has 11 different signers.
  - The val→test accuracy gap reflects signer variability, not class distribution.

Produces three figures:
  1. class_frequency.png       — per-class sample counts, val vs test
  2. accuracy_by_freq_bin.png  — mean per-class test accuracy by test-freq bin
  3. participant_dist.png      — signer-level sample counts per split

Usage:
    python src/data/asl_split_analysis.py [--predictions PATH] [--out_dir OUT]

    --predictions  Path to a *.predictions.csv file from the models/
                   predictions directory. Defaults to the monolingual
                   asl_citizen/informed_selection/iterative_10 model.
    --out_dir      Where to write output figures (default: analysis/asl_split).
"""

import argparse
import csv
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ARCHIVE = os.path.expanduser(
    "~/groups/grp_asl_classification/nobackup/archive"
)
PIPELINE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "asl_citizen"
)
DEFAULT_PREDICTIONS = os.path.join(
    ARCHIVE, "SLR/models/asl_citizen/informed_selection/iterative_10",
    "predictions",
    "epoch=28-step=35766-val_loss=2.094845-val_acc=0.621894.ckpt.predictions.csv",
)
RAW_SPLITS_DIR = os.path.join(ARCHIVE, "ASL/ASL_Citizen/splits")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pipeline_labels(split: str) -> list[int]:
    path = os.path.join(PIPELINE_DIR, f"{split}.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [int(row["label"]) for row in reader]


def load_raw_rows(split: str) -> list[tuple[str, str]]:
    """Returns list of (participant_id, gloss) from raw ASL Citizen splits."""
    path = os.path.join(RAW_SPLITS_DIR, f"{split}.csv")
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                rows.append((row[0].strip(), row[2].strip()))
    return rows


def load_class_ids() -> dict[int, str]:
    path = os.path.join(PIPELINE_DIR, "class_ids.csv")
    mapping = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["ClassId"])] = row["EN"]
    return mapping


def load_predictions(pred_path: str) -> tuple[list[int], list[int]]:
    """Returns (labels, predictions) lists."""
    labels, preds = [], []
    with open(pred_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            preds.append(int(row["prediction"]))
    return labels, preds


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def frequency_table(labels: list[int], n_classes: int) -> np.ndarray:
    """Returns array of length n_classes with per-class counts."""
    counts = np.zeros(n_classes, dtype=int)
    for c in Counter(labels).items():
        counts[c[0]] = c[1]
    return counts


def per_class_accuracy(labels: list[int], preds: list[int], n_classes: int
                       ) -> np.ndarray:
    """Returns per-class accuracy array (NaN for classes with no samples)."""
    correct = np.zeros(n_classes, dtype=int)
    total   = np.zeros(n_classes, dtype=int)
    for l, p in zip(labels, preds):
        total[l] += 1
        correct[l] += (l == p)
    with np.errstate(invalid="ignore"):
        acc = np.where(total > 0, correct / total, np.nan)
    return acc


def bin_by_frequency(freq: np.ndarray, acc: np.ndarray, bins: list[int]
                     ) -> tuple[list[str], list[float], list[int]]:
    """
    Groups classes by their frequency bin, returns mean accuracy per bin.
    bins defines the RIGHT edges (inclusive); last bin is open-ended.
    """
    bin_labels, bin_acc, bin_n = [], [], []
    edges = [0] + list(bins)
    for i in range(len(edges) - 1):
        lo, hi = edges[i] + 1, edges[i + 1]
        mask = (freq >= lo) & (freq <= hi)
        accs = acc[mask & ~np.isnan(acc)]
        bin_labels.append(f"{lo}–{hi}")
        bin_acc.append(float(np.mean(accs)) if len(accs) else np.nan)
        bin_n.append(int(mask.sum()))
    # Last open-ended bin
    lo = edges[-1] + 1
    mask = freq >= lo
    accs = acc[mask & ~np.isnan(acc)]
    bin_labels.append(f"{lo}+")
    bin_acc.append(float(np.mean(accs)) if len(accs) else np.nan)
    bin_n.append(int(mask.sum()))
    return bin_labels, bin_acc, bin_n


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_frequency_distributions(val_freq: np.ndarray, test_freq: np.ndarray,
                                  out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ASL Citizen — Class Frequency Distributions", fontsize=13)

    n_classes = len(val_freq)
    present_val  = (val_freq  > 0).sum()
    present_test = (test_freq > 0).sum()

    # Left: histogram of per-class sample counts
    ax = axes[0]
    val_nonzero  = val_freq[val_freq > 0]
    test_nonzero = test_freq[test_freq > 0]
    max_freq = max(val_nonzero.max(), test_nonzero.max())
    bins = np.arange(0.5, max_freq + 1.5, 1)
    ax.hist(val_nonzero,  bins=bins, alpha=0.6, label=f"val  (n={len(val_nonzero)} classes)", color="steelblue")
    ax.hist(test_nonzero, bins=bins, alpha=0.6, label=f"test (n={len(test_nonzero)} classes)", color="tomato")
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("Number of classes")
    ax.set_title("Frequency histogram (classes present in split)")
    ax.legend()

    # Right: scatter — val freq vs test freq per class (present in both)
    ax = axes[1]
    both_mask = (val_freq > 0) & (test_freq > 0)
    val_only  = (val_freq > 0) & (test_freq == 0)
    test_only = (test_freq > 0) & (val_freq == 0)
    neither   = (val_freq == 0) & (test_freq == 0)

    ax.scatter(val_freq[both_mask], test_freq[both_mask],
               alpha=0.25, s=8, color="purple",
               label=f"in both ({both_mask.sum()})")
    ax.scatter(val_freq[val_only] + np.random.uniform(-0.15, 0.15, val_only.sum()),
               np.zeros(val_only.sum()) + np.random.uniform(-0.15, 0.15, val_only.sum()),
               alpha=0.4, s=8, color="steelblue", label=f"val only ({val_only.sum()})")
    ax.scatter(np.zeros(test_only.sum()) + np.random.uniform(-0.15, 0.15, test_only.sum()),
               test_freq[test_only] + np.random.uniform(-0.15, 0.15, test_only.sum()),
               alpha=0.4, s=8, color="tomato", label=f"test only ({test_only.sum()})")

    # Diagonal reference (perfectly proportional)
    scale = test_nonzero.mean() / val_nonzero.mean()
    lim = max(val_freq.max(), test_freq.max() / scale) + 1
    xs = np.linspace(0, val_freq.max() + 1, 100)
    ax.plot(xs, xs * scale, "k--", lw=0.8, alpha=0.5,
            label=f"proportional ({scale:.1f}×)")

    ax.set_xlabel("Val samples per class")
    ax.set_ylabel("Test samples per class")
    ax.set_title("Per-class frequency: val vs test")
    ax.legend(fontsize=8)

    ax.text(0.02, 0.97,
            f"Zero in both: {neither.sum()}\nTotal classes: {n_classes}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_accuracy_by_freq_bin(val_freq: np.ndarray, test_freq: np.ndarray,
                               test_acc: np.ndarray, out_path: str) -> None:
    """
    For each test-frequency bin, shows mean per-class accuracy and class count.
    Also shows how many of those classes had only 1–3 samples in val.
    """
    freq_bins = [1, 3, 5, 7, 9, 12, 15]
    bin_labels, bin_acc, bin_n = bin_by_frequency(test_freq, test_acc, freq_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("ASL Citizen — Per-Class Test Accuracy by Test-Set Frequency",
                 fontsize=13)

    x = np.arange(len(bin_labels))
    colors = ["tomato" if not np.isnan(a) else "lightgray" for a in bin_acc]

    bars = ax1.bar(x, [a if not np.isnan(a) else 0 for a in bin_acc],
                   color=colors, edgecolor="white", linewidth=0.5)
    for bar, a in zip(bars, bin_acc):
        if not np.isnan(a):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005, f"{a:.1%}",
                     ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Mean per-class accuracy (test)")
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.axhline(np.nanmean(test_acc[test_freq > 0]), color="gray", ls="--",
                lw=0.9, label=f"overall mean ({np.nanmean(test_acc[test_freq > 0]):.1%})")
    ax1.legend(fontsize=8)

    # Bottom: class count per bin + breakdown of val coverage
    edges = [0] + freq_bins
    bar_total = []
    bar_sparse = []  # classes where val_freq < 3
    for i in range(len(edges) - 1):
        lo, hi = edges[i] + 1, edges[i + 1]
        mask = (test_freq >= lo) & (test_freq <= hi)
        bar_total.append(mask.sum())
        bar_sparse.append(((val_freq < 3) & mask).sum())
    # last open bin
    lo = edges[-1] + 1
    mask = test_freq >= lo
    bar_total.append(mask.sum())
    bar_sparse.append(((val_freq < 3) & mask).sum())

    ax2.bar(x, bar_total, color="steelblue", alpha=0.7, label="all classes")
    ax2.bar(x, bar_sparse, color="gold", alpha=0.9,
            label="val freq < 3 (sparse in val)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax2.set_xlabel("Test samples per class (bin)")
    ax2.set_ylabel("Number of classes")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_participant_distribution(
        train_rows: list[tuple[str, str]],
        val_rows:   list[tuple[str, str]],
        test_rows:  list[tuple[str, str]],
        out_path: str) -> None:
    """Bar chart: samples per signer for each split. Highlights disjoint signer design."""
    tc = Counter(p for p, _ in train_rows)
    vc = Counter(p for p, _ in val_rows)
    xc = Counter(p for p, _ in test_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        "ASL Citizen — Samples per Signer (train/val/test signers are completely disjoint)",
        fontsize=12,
    )

    for ax, cnt, title, color in [
        (axes[0], tc, f"Train  ({len(tc)} signers, {sum(tc.values())} samples)", "steelblue"),
        (axes[1], vc, f"Val    ({len(vc)} signers, {sum(vc.values())} samples)", "seagreen"),
        (axes[2], xc, f"Test   ({len(xc)} signers, {sum(xc.values())} samples)", "tomato"),
    ]:
        participants = sorted(cnt.keys())
        values = [cnt[p] for p in participants]
        ax.bar(range(len(participants)), values, color=color,
               edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(participants)))
        ax.set_xticklabels(participants, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("Number of samples")
        ax.set_xlabel("Participant ID")
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.01, str(v), ha="center",
                    fontsize=6, rotation=90)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_summary(train_rows, val_rows, test_rows, val_freq, test_freq,
                  test_acc, class_ids):
    n = len(val_freq)
    train_p = set(p for p, _ in train_rows)
    val_p   = set(p for p, _ in val_rows)
    test_p  = set(p for p, _ in test_rows)

    print("\n=== Signer Design ===")
    print(f"  Train: {len(train_p)} signers, {len(train_rows)} samples")
    print(f"  Val:   {len(val_p)} signers, {len(val_rows)} samples")
    print(f"  Test:  {len(test_p)} signers, {len(test_rows)} samples")
    print(f"  Overlap train∩val: {len(train_p & val_p)}  "
          f"train∩test: {len(train_p & test_p)}  val∩test: {len(val_p & test_p)}")

    print("\n=== Class Frequency ===")
    print(f"  Total classes (from train): {n}")
    print(f"  Val : {(val_freq>0).sum()} present | "
          f"freq min={val_freq[val_freq>0].min()}, max={val_freq[val_freq>0].max()}, "
          f"mean={val_freq[val_freq>0].mean():.1f}")
    print(f"  Test: {(test_freq>0).sum()} present | "
          f"freq min={test_freq[test_freq>0].min()}, max={test_freq[test_freq>0].max()}, "
          f"mean={test_freq[test_freq>0].mean():.1f}")

    print("\n=== Test Accuracy by Frequency Bin ===")
    present = test_freq > 0
    print(f"  Overall: {np.nanmean(test_acc[present]):.1%} (mean per-class)")
    for lo, hi in [(7, 9), (10, 12), (13, 15), (16, 20)]:
        mask = (test_freq >= lo) & (test_freq <= hi)
        if mask.sum():
            print(f"  Test freq {lo:2d}–{hi:2d}: {np.nanmean(test_acc[mask]):.1%}  "
                  f"({mask.sum()} classes)")

    print("\n=== Lowest Per-Class Test Accuracy (≥10 test samples) ===")
    mask = test_freq >= 10
    idxs = np.where(mask)[0]
    idxs_sorted = idxs[np.argsort(test_acc[idxs])][:20]
    print(f"  {'Class':>5}  {'Gloss':<22}  {'Test acc':>8}  {'Val n':>5}  {'Test n':>6}")
    for idx in idxs_sorted:
        gloss = class_ids.get(int(idx), "?")
        print(f"  {idx:>5}  {gloss:<22}  {test_acc[idx]:>8.1%}  "
              f"{val_freq[idx]:>5}  {test_freq[idx]:>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", default=DEFAULT_PREDICTIONS,
        help="Path to *.predictions.csv from a model run on the ASL test set."
    )
    parser.add_argument("--out_dir", default="analysis/asl_split")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading splits...")
    val_labels  = load_pipeline_labels("val")
    test_labels = load_pipeline_labels("test")
    class_ids   = load_class_ids()
    n_classes   = len(class_ids)

    train_rows = load_raw_rows("train")
    val_rows   = load_raw_rows("val")
    test_rows  = load_raw_rows("test")

    val_freq  = frequency_table(val_labels,  n_classes)
    test_freq = frequency_table(test_labels, n_classes)

    print("Plotting frequency distributions...")
    plot_frequency_distributions(
        val_freq, test_freq,
        os.path.join(args.out_dir, "class_frequency.png")
    )

    print("Plotting participant distribution...")
    plot_participant_distribution(
        train_rows, val_rows, test_rows,
        os.path.join(args.out_dir, "participant_dist.png")
    )

    # Predictions-dependent plots
    if not os.path.exists(args.predictions):
        print(f"WARNING: predictions file not found: {args.predictions}")
        print("Skipping accuracy plots.")
        return

    print(f"Loading predictions from:\n  {args.predictions}")
    pred_labels, pred_preds = load_predictions(args.predictions)
    test_acc = per_class_accuracy(pred_labels, pred_preds, n_classes)

    print("Plotting accuracy by frequency bin...")
    plot_accuracy_by_freq_bin(
        val_freq, test_freq, test_acc,
        os.path.join(args.out_dir, "accuracy_by_freq_bin.png")
    )

    print_summary(train_rows, val_rows, test_rows, val_freq, test_freq,
                  test_acc, class_ids)
    print(f"\nAll outputs written to: {args.out_dir}/")


if __name__ == "__main__":
    main()