"""
Visualization of encoder comparison results (current training state).

Usage:
    python scripts/plot_encoder_results.py              # saves plots/encoder_results.png
    python scripts/plot_encoder_results.py --show       # also opens interactive window
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Collect results via show_encoder_results.py ───────────────────────────────

def collect():
    result = subprocess.run(
        [sys.executable, "scripts/show_encoder_results.py", "--csv", "--all"],
        capture_output=True, text=True
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
        status = ",".join(status_parts)
        val = float(val_acc) if val_acc else None
        test = float(test_acc) if test_acc else None
        rows.append(dict(lang=lang, enc=enc, type=sel_type, K=int(k),
                         draw=draw, val=val, test=test, status=status.strip()))
    return rows


# ── Plot helpers ──────────────────────────────────────────────────────────────

LANG_LABELS = {
    "autsl": "AUTSL",
    "asl_citizen": "ASL Citizen",
    "gsl": "GSL",
    "multilingual": "Multilingual",
}
ENC_COLORS = {"bert": "#9C27B0", "gru": "#2196F3", "stgcn": "#FF9800", "spoter": "#4CAF50"}
ENC_LABELS = {"bert": "BERT", "gru": "GRU", "stgcn": "ST-GCN", "spoter": "SPOTER"}
MARKER_RUNNING = "o"
MARKER_DONE    = "o"
RUNNING_ALPHA  = 0.35

LANGUAGES = ["autsl", "asl_citizen", "gsl", "multilingual"]
ENCODERS  = ["bert", "gru", "stgcn", "spoter"]


def acc(row):
    """Return best available accuracy (test preferred, then val). None if no data."""
    return row["test"] if row["test"] is not None else row["val"]

def is_complete(row):
    return row["status"] in ("DONE", "TRAINED (no test)")


# ── Figure 1: Iterative K curves per language ─────────────────────────────────

ITER_K_ORDER = [543, 270, 100, 48, 24, 10]
K_POS = {k: i for i, k in enumerate(ITER_K_ORDER)}  # evenly-spaced positions


def plot_iterative_curves(rows, axes):
    """One subplot per language — val_acc vs K for iter + full skeleton.
    X-axis is evenly spaced by K index, not by raw K value."""
    iter_rows = [r for r in rows if r["type"] in ("iterative", "full")]

    for ax, lang in zip(axes, LANGUAGES):
        lang_rows = [r for r in iter_rows if r["lang"] == lang]

        for enc in ENCODERS:
            all_enc  = sorted([r for r in lang_rows if r["enc"] == enc], key=lambda r: r["K"])
            done_enc = [r for r in all_enc if is_complete(r) and acc(r) is not None]
            pend_enc = [r for r in all_enc if not is_complete(r)]

            # Solid line + filled points for completed runs
            if done_enc:
                xs = [K_POS[r["K"]] for r in done_enc]
                ys = [acc(r) for r in done_enc]
                ax.plot(xs, ys, color=ENC_COLORS[enc], linewidth=2)
                ax.scatter(xs, ys, color=ENC_COLORS[enc], s=60, zorder=5,
                           edgecolors="white", linewidths=0.8)

            # Faded dashed vertical lines at K values still running
            for r in pend_enc:
                if r["K"] in K_POS:
                    ax.axvline(K_POS[r["K"]], color=ENC_COLORS[enc], linewidth=1,
                               alpha=RUNNING_ALPHA, linestyle="--")

        ax.set_title(LANG_LABELS[lang], fontsize=11, fontweight="bold")
        ax.set_xlabel("K (joints)", fontsize=9)
        ax.set_ylabel("Val Acc (%)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(ITER_K_ORDER)))
        ax.set_xticklabels(ITER_K_ORDER)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)


# ── Figure 2: Full skeleton (K=543) bar chart ─────────────────────────────────

def plot_full_skeleton_bars(rows, ax):
    full_rows = {(r["lang"], r["enc"]): acc(r)
                 for r in rows if r["type"] == "full" and is_complete(r) and acc(r) is not None}

    x = np.arange(len(LANGUAGES))
    width = 0.25

    for i, enc in enumerate(ENCODERS):
        heights = [full_rows.get((lang, enc), 0) for lang in LANGUAGES]
        alphas  = [1.0 if full_rows.get((lang, enc)) is not None else 0.2
                   for lang in LANGUAGES]
        bars = ax.bar(x + (i - 1) * width, heights, width,
                      label=ENC_LABELS[enc], color=ENC_COLORS[enc])
        for bar, h, a in zip(bars, heights, alphas):
            bar.set_alpha(a)
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_LABELS[l] for l in LANGUAGES], fontsize=9)
    ax.set_ylabel("Val Acc (%)", fontsize=9)
    ax.set_title("Full Skeleton (K=543) — All Encoders", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=8)


# ── Figure 3: Progress heatmap ────────────────────────────────────────────────

def plot_progress_heatmap(rows, ax):
    """Colour = val_acc; grey = running; white = not started."""
    iter_types = ["full"] + [f"iter_{k}" for k in [10, 24, 48, 100, 270]]
    col_labels  = ["543"] + ["iter\n10", "iter\n24", "iter\n48", "iter\n100", "iter\n270"]

    row_labels = [f"{ENC_LABELS[e]}\n{LANG_LABELS[l]}"
                  for e in ENCODERS for l in LANGUAGES]

    lookup = {(r["lang"], r["enc"], r["type"] if r["type"] != "full" else "full",
               r["K"]): r
              for r in rows}

    def get_row(enc, lang):
        out = []
        for st, k in zip(iter_types, [543, 10, 24, 48, 100, 270]):
            sel = "full" if st == "full" else "iterative"
            r = lookup.get((lang, enc, sel, k))
            out.append(r)
        return out

    data   = np.full((len(row_labels), len(col_labels)), np.nan)
    colors = np.zeros((len(row_labels), len(col_labels), 4))

    for ri, (enc, lang) in enumerate((e, l) for e in ENCODERS for l in LANGUAGES):
        for ci, r in enumerate(get_row(enc, lang)):
            if r is None:
                colors[ri, ci] = [0.95, 0.95, 0.95, 1]  # not started — light grey
            elif acc(r) is not None:
                v = acc(r) / 100
                # Blue-to-green gradient
                colors[ri, ci] = plt.cm.YlGn(v)
                data[ri, ci] = acc(r)
            else:
                colors[ri, ci] = [0.85, 0.85, 0.85, 1]  # running — mid grey

    ax.imshow(colors, aspect="auto")

    # Annotate cells
    for ri in range(len(row_labels)):
        for ci in range(len(col_labels)):
            v = data[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f"{v:.1f}", ha="center", va="center",
                        fontsize=7.5, fontweight="bold",
                        color="black" if v > 30 else "white")
            else:
                r = None
                enc_i = ri // len(LANGUAGES)
                lang_i = ri % len(LANGUAGES)
                enc = ENCODERS[enc_i]
                lang = LANGUAGES[lang_i]
                sel = "full" if ci == 0 else "iterative"
                k = [543, 10, 24, 48, 100, 270][ci]
                r = lookup.get((lang, enc, sel, k))
                if r and r["status"] == "RUNNING":
                    ax.text(ci, ri, "▶", ha="center", va="center",
                            fontsize=9, color="#666666")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title("Training Progress — Val Acc (%) | grey=running, white=not started",
                 fontsize=10, fontweight="bold")

    # Dividers between encoders (one line per encoder boundary)
    for i in range(1, len(ENCODERS)):
        ax.axhline(i * len(LANGUAGES) - 0.5, color="black", linewidth=1.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Open interactive window")
    parser.add_argument("--out", default="plots/encoder_results.png")
    args = parser.parse_args()

    rows = collect()
    if not rows:
        print("No results found.")
        return

    Path(args.out).parent.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(20, 18))
    fig.patch.set_facecolor("#FAFAFA")

    # Layout: top row = 4 iterative curve plots, bottom left = bars, bottom right = heatmap
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    curve_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    bar_ax     = fig.add_subplot(gs[1, :2])
    heat_ax    = fig.add_subplot(gs[1:, 2:])

    plot_iterative_curves(rows, curve_axes)
    plot_full_skeleton_bars(rows, bar_ax)
    plot_progress_heatmap(rows, heat_ax)

    # Shared legend for curves
    handles = [mpatches.Patch(color=ENC_COLORS[e], label=ENC_LABELS[e]) for e in ENCODERS]
    handles += [plt.Line2D([0], [0], color="grey", alpha=RUNNING_ALPHA,
                           linewidth=2, label="still running (partial)")]
    curve_axes[3].legend(handles=handles, fontsize=8, loc="lower right")

    fig.suptitle("Encoder Comparison — Current Training State", fontsize=14, fontweight="bold", y=0.97)

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()