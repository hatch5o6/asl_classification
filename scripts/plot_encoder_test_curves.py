"""
Iterative K curves using TEST accuracy only (no fallback to val).

Generates just the 4-panel "iterative K vs accuracy" figure from
plot_encoder_results.py, but plots only points that have a test_acc.
Useful as the final paper figure once tests have caught up.

Usage:
    python scripts/plot_encoder_test_curves.py            # plots/encoder_test_curves.png
    python scripts/plot_encoder_test_curves.py --out plots/foo.png
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def collect():
    result = subprocess.run(
        [sys.executable, "scripts/show_encoder_results.py", "--csv", "--all"],
        capture_output=True, text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("lang"):
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        lang, enc, sel_type, k, draw, val_acc, test_acc, *status_parts = parts
        test = float(test_acc) if test_acc else None
        rows.append(dict(lang=lang, enc=enc, type=sel_type, K=int(k), test=test))
    return rows


LANG_LABELS = {
    "autsl": "AUTSL",
    "asl_citizen": "ASL Citizen",
    "gsl": "GSL",
    "multilingual": "Multilingual",
}
ENC_COLORS = {"bert": "#9C27B0", "gru": "#2196F3", "stgcn": "#FF9800", "spoter": "#4CAF50"}
ENC_LABELS = {"bert": "BERT", "gru": "GRU", "stgcn": "ST-GCN", "spoter": "SPOTER"}

LANGUAGES = ["autsl", "asl_citizen", "gsl", "multilingual"]
ENCODERS  = ["bert", "gru", "stgcn", "spoter"]

ITER_K_ORDER = [543, 270, 100, 48, 24, 10]
K_POS = {k: i for i, k in enumerate(ITER_K_ORDER)}


ENC_MARKERS = {"bert": "o", "gru": "s", "stgcn": "^", "spoter": "D"}

def plot_iterative_curves(rows, axes):
    """One subplot per language — test_acc vs K for iter + full skeleton."""
    iter_rows = [r for r in rows if r["type"] in ("iterative", "full")]

    for ax, lang in zip(axes, LANGUAGES):
        lang_rows = [r for r in iter_rows if r["lang"] == lang]

        for enc in ENCODERS:
            done = sorted(
                [r for r in lang_rows if r["enc"] == enc and r["test"] is not None],
                key=lambda r: r["K"],
            )
            if not done:
                continue
            xs = [K_POS[r["K"]] for r in done if r["K"] in K_POS]
            ys = [r["test"] for r in done if r["K"] in K_POS]
            
            # 2. Plot with markers and line styles
            ax.plot(xs, ys, color=ENC_COLORS[enc], marker=ENC_MARKERS[enc], 
                    markersize=6, label=ENC_LABELS[enc], linewidth=1.5)

        ax.set_title(LANG_LABELS[lang], fontsize=12, fontweight="bold")
        ax.set_xlabel("K (joints)", fontsize=11)
        ax.set_ylabel("Test Acc (%)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(ITER_K_ORDER)))
        ax.set_xticklabels(ITER_K_ORDER)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="plots/encoder_test_curves.png")
    args = parser.parse_args()

    rows = collect()
    if not rows:
        print("No results found.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Encoder Comparison — Iterative K vs Test Accuracy",
                 fontsize=13, fontweight="bold", y=1.02)

    plot_iterative_curves(rows, axes)

    handles = [mpatches.Patch(color=ENC_COLORS[e], label=ENC_LABELS[e]) for e in ENCODERS]
    axes[-1].legend(handles=handles, fontsize=12, loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()