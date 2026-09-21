"""
Regenerate every figure in docs/emnlp_v2.tex from the CURRENT (hard-concrete) run set.

The figures shipped with the previous submission were built from a superseded run set and
drew three specific reviewer complaints: inconsistent y-axis ranges across panels, no error
bars, and an unlabelled dot-cloud with no reference for chance overlap. Each is addressed
below and noted where it is handled.

Outputs (PDF for the paper, PNG for quick viewing) into docs/figures/.

Usage:  python scripts/make_paper_figures.py [--outdir docs/figures]
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from analysis.visualize_skeleton_selection import make_canonical_positions, BODY_CONNECTIONS

# ── palette ───────────────────────────────────────────────────────────────────
# Validated with the dataviz skill's checker (light mode): lightness band, chroma
# floor, adjacent-pair CVD separation (worst dE 9.1 protan) and normal-vision floor
# all PASS. Contrast-vs-surface WARNs on the two lighter hues, whose required relief
# is a visible legend plus the numbers in the tables -- both present.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GROUP_COLOR = {"face": BLUE, "body": ORANGE, "left_hand": AQUA, "right_hand": YELLOW}
GROUP_LABEL = {"face": "Face", "body": "Body", "left_hand": "Left hand", "right_hand": "Right hand"}
GROUPS = ["face", "body", "left_hand", "right_hand"]          # fixed order, never cycled
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SEQ = ["#eef2f7", "#9cc0ea", "#4e8fd9", "#1b4f8f"]
# open-probability below which the hard-concrete gate clamps to exactly zero
P_CLOSED = 0.3102            # 0..3 languages, light->dark

LANGS = {"AUTSL": "configs/stage2_hc/autsl",
         "ASL Citizen": "configs/stage2_ext/asl_citizen",
         "GSL": "configs/stage2_hc/gsl"}
IDXDIR = {"AUTSL": "data/informed_selection_hc/topk",
          "ASL Citizen": "data/asl_citizen/informed_selection_hc/topk",
          "GSL": "data/gsl/informed_selection_hc/topk"}
PROBS = {"AUTSL": "data/informed_selection_hc/joint_probabilities.csv",
         "ASL Citizen": "data/asl_citizen/informed_selection_hc/joint_probabilities.csv",
         "GSL": "data/gsl/informed_selection_hc/joint_probabilities.csv"}
KS = [543, 270, 100, 48, 24, 10]


def rcparams():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "legend.frameon": False,
    })


def acc(cfg):
    if not os.path.exists(cfg):
        return None
    save = yaml.safe_load(open(cfg))["save"]
    fs = glob.glob(os.path.join(save, "predictions", "*.metrics.json"))
    if not fs:
        return None
    return json.load(open(sorted(fs)[-1]))["accuracy"] * 100


def series(d, pattern, ks):
    """mean, sd, n per K for a config-name pattern containing {K}."""
    out = {}
    for K in ks:
        v = [acc(c) for c in sorted(glob.glob(os.path.join(d, pattern.format(K=K))))]
        v = [x for x in v if x is not None]
        if v:
            out[K] = (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
    return out


def group_of(i):
    return "face" if i < 468 else "body" if i < 501 else "left_hand" if i < 522 else "right_hand"


def save_(fig, outdir, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


# ── Figure 2: Pareto frontier ────────────────────────────────────────────────
def fig_pareto(outdir):
    """Reviewer fixes: every panel uses the SAME y-span (14 points) so slopes are
    directly comparable across corpora, and every series carries a +/-1 sd band.
    The uniform-random baseline is deliberately NOT drawn here -- at 8-73% it forces
    a ~65-point range that flattens the curves this figure exists to show. It is in
    Table 2 and Appendix B instead."""
    SPAN = 14.0
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    for ax, (lang, d) in zip(axes, LANGS.items()):
        topk = series(d, "topk{K}_seed*.yaml", KS)
        casc = series(d, "casc{K}_seed*.yaml", [100, 48, 24, 10])
        # seeds 4003/4004 exist only under the "_2gpu" filename variant, so take every
        # tslformer config and keep those that actually have test metrics -- exactly the
        # five the paper reports (87.04 / 60.96 / 66.85).
        tsl = [acc(c) for c in sorted(glob.glob(
            f"configs/tslformer/{os.path.basename(d)}_seed4*.yaml"))]
        tsl = [t for t in tsl if t is not None]
        assert len(tsl) == 5, f"{lang}: expected 5 TSLFormer seeds, got {len(tsl)}"

        full = topk.get(543, (None,))[0]
        if full is not None:
            ax.axhline(full, color=INK, lw=0.8, ls=":", zorder=1)
        if tsl:
            ax.axhline(float(np.mean(tsl)), color=MUTED, lw=1.0, ls=(0, (5, 2)), zorder=4)

        for name, s_, colr in (("Top-$K$", topk, BLUE), ("Cascade", casc, ORANGE)):
            ks = sorted(s_)
            m = np.array([s_[k][0] for k in ks]); sd = np.array([s_[k][1] for k in ks])
            ax.fill_between(ks, m - sd, m + sd, color=colr, alpha=0.18, linewidth=0)
            ax.plot(ks, m, color=colr, lw=1.4, marker="o", ms=3.4, label=name,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=3)

        vals = [v[0] for v in topk.values()] + [v[0] for v in casc.values()]
        hi = max(vals) + 0.14 * SPAN
        lo = hi - SPAN
        ax.set_ylim(lo, hi)
        # direct labels for the two reference lines, placed inside the axes
        if full is not None:
            ax.annotate("full skeleton", (0.97, full), xycoords=("axes fraction", "data"),
                        fontsize=5.8, color=INK, ha="right", va="bottom", zorder=5)
        if tsl:
            ax.annotate("hand-crafted (50)", (0.03, float(np.mean(tsl))),
                        xycoords=("axes fraction", "data"),
                        fontsize=5.8, color=MUTED, ha="left", va="bottom", zorder=5)
        ax.set_xscale("log"); ax.set_xlim(680, 8.2)     # K=543 leftmost, pruning to the right
        ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.minorticks_off()
        ax.grid(axis="y", lw=0.5, alpha=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_title(f"{lang}   (span {SPAN:.0f} pts)", color=INK, pad=4)
        ax.set_xlabel("Landmarks retained ($K$)")
    axes[0].set_ylabel("Top-1 test accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    ref = [Line2D([], [], color=INK, ls=":", lw=0.8, label="Full skeleton"),
           Line2D([], [], color=MUTED, ls=(0, (5, 2)), lw=0.9, label="Hand-crafted (50)")]
    fig.legend(handles=h + ref, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12),
               handlelength=1.8, columnspacing=1.5)
    fig.tight_layout()
    save_(fig, outdir, "fig_pareto")


# ── Figure 3: cross-language consensus skeleton ──────────────────────────────
def fig_consensus(outdir, ks=(100, 48, 24, 10)):
    """Reviewer fixes: a chance reference is printed inside every panel (both the
    uniform and the anatomy-stratified expectation), and landmark indices are
    annotated where the consensus set is small enough to read (K <= 24)."""
    # stratified-null expectations reported in the paper
    NULL = {270: 102.8, 100: 52.9, 48: 32.4, 24: 3.0, 10: 0.3}
    UNIF = {270: 66.8, 100: 3.4, 48: 0.4, 24: 0.05, 10: 0.003}
    xy = make_canonical_positions()
    sel = {lang: {K: set(json.load(open(f"{IDXDIR[lang]}/top_{K}_indices.json"))) for K in ks}
           for lang in LANGS}

    fig, axes = plt.subplots(1, len(ks), figsize=(7.1, 2.75))
    for ax, K in zip(np.atleast_1d(axes), ks):
        count = np.array([sum(K in sel[l] and i in sel[l][K] for l in LANGS) for i in range(543)])
        for a, b in BODY_CONNECTIONS:
            ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=GRID, lw=0.6, zorder=1)
        for c in (0, 1, 2, 3):
            m = count == c
            if not m.any(): continue
            ax.scatter(xy[m, 0], xy[m, 1], s=0.9 if c == 0 else 7.5, c=SEQ[c],
                       linewidths=0.25 if c else 0,
                       edgecolors="white" if c else "none", zorder=2 + c)
        ax.set_title(f"$K={K}$", color=INK, pad=3)
        cons = list(np.where(count == 3)[0])
        note = (f"all three: {len(cons)}\nnull {NULL[K]:.1f} (strat.) / {UNIF[K]:g} (unif.)")
        if K <= 24 and cons:              # list indices where they are few enough to read
            wrapped, line = [], "indices: "
            for t in (str(c) for c in cons):
                if len(line) + len(t) + 2 > 30:
                    wrapped.append(line); line = "  "
                line += t + ", "
            wrapped.append(line.rstrip(", "))
            note += "\n" + "\n".join(wrapped)
        ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=5.6, color=MUTED)
        ax.set_xlim(0.08, 0.92); ax.set_ylim(1.0, -0.02); ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", ms=4, markerfacecolor=SEQ[c],
                      markeredgecolor="white", markeredgewidth=0.3,
                      label=f"{c} languages" if c != 1 else "1 language") for c in (0, 1, 2, 3)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09),
               handlelength=1.2, columnspacing=1.4)
    fig.tight_layout()
    save_(fig, outdir, "fig_consensus")


# ── Appendix: body-part composition ──────────────────────────────────────────
def fig_bodyparts(outdir, ks=(270, 100, 48, 24, 10)):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
    for ax, lang in zip(axes, LANGS):
        bottoms = np.zeros(len(ks))
        for g in GROUPS:
            vals = []
            for K in ks:
                idx = json.load(open(f"{IDXDIR[lang]}/top_{K}_indices.json"))
                vals.append(100.0 * sum(1 for i in idx if group_of(i) == g) / len(idx))
            vals = np.array(vals)
            ax.bar(range(len(ks)), vals, bottom=bottoms, color=GROUP_COLOR[g],
                   width=0.68, label=GROUP_LABEL[g], edgecolor="white", linewidth=0.8)
            for x, (v, b) in enumerate(zip(vals, bottoms)):      # selective direct labels
                if v >= 12:
                    ax.text(x, b + v / 2, f"{v:.0f}", ha="center", va="center",
                            fontsize=5.8, color="white")
            bottoms += vals
        ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("$K$"); ax.set_title(lang, color=INK, pad=4)
        ax.set_ylim(0, 100)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Share of selected subset (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12),
               handlelength=1.2, columnspacing=1.5)
    fig.tight_layout()
    save_(fig, outdir, "fig_bodyparts")


# ── Appendix: gate open-probability distribution ─────────────────────────────
def fig_gate_probs(outdir):
    """Sorted converged open probability per landmark, coloured by anatomical group.
    Shows both the ranking the method consumes and how many gates closed outright."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=True)
    for ax, lang in zip(axes, LANGS):
        p = np.loadtxt(PROBS[lang], skiprows=1)
        order = np.argsort(-p)
        ranks = np.arange(1, len(p) + 1)
        for g in GROUPS:
            m = np.array([group_of(i) == g for i in order])
            ax.scatter(ranks[m], p[order][m], s=1.4, c=GROUP_COLOR[g],
                       label=GROUP_LABEL[g], linewidths=0)
        # The CSV stores P(z>0) = sigma(alpha - beta*log(-gamma/zeta)), which never reaches
        # exactly 0. A gate is DETERMINISTICALLY closed when the stretched-and-clamped value
        # hits zero, i.e. sigma(alpha) <= -gamma/(zeta-gamma) = 1/12, which maps to
        # P(z>0) <= 0.3102. That threshold reproduces the 309/250/163 closed gates reported
        # in the method section exactly.
        nz = int((p <= P_CLOSED).sum())
        ax.axhspan(-0.03, P_CLOSED, color=GRID, alpha=0.55, linewidth=0, zorder=0)
        ax.axhline(P_CLOSED, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.annotate("gate closed", (560, P_CLOSED - 0.03), fontsize=5.8, color=MUTED,
                    ha="right", va="top")
        for K, lab in ((48, "48"), (10, "10")):
            ax.axvline(K, color=INK, lw=0.6, ls=":")
            ax.annotate(f"$K$={lab}", (K, 1.01), fontsize=5.6, color=INK,
                        ha="center", va="bottom")
        ax.set_xscale("log"); ax.set_xlim(0.85, 600); ax.set_ylim(-0.03, 1.06)
        ax.set_xlabel("Landmark rank")
        ax.set_title(f"{lang}  ({nz} of 543 closed)", color=INK, pad=4)
        ax.grid(axis="y", lw=0.5, alpha=0.7); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Open probability $p_j$")
    handles = [Line2D([], [], marker="o", ls="", ms=3.4, color=GROUP_COLOR[g],
                      label=GROUP_LABEL[g]) for g in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12),
               handlelength=1.2, columnspacing=1.5)
    fig.tight_layout()
    save_(fig, outdir, "fig_gate_probs")


# ── Appendix: stage-1 training dynamics ──────────────────────────────────────
STAGE1 = {"AUTSL": "configs/gate_stage1/autsl.yaml",
          "ASL Citizen": "configs/gate_stage1/asl_citizen.yaml",
          "GSL": "configs/gate_stage1_seeds/gsl_seed4000_rerun.yaml"}


def fig_dynamics(outdir):
    """Requested by review: how the gate probabilities move over stage-1 training.
    Per-landmark traces were not logged, but the p25/median/p75 envelope and the
    expected-L0 count were, which is what the claim actually rests on."""
    import pandas as pd
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.4), sharex=True)
    for col, (lang, cfg) in enumerate(STAGE1.items()):
        save = yaml.safe_load(open(cfg))["save"]
        d = pd.read_csv(os.path.join(save, "logs/lightning_logs/version_0/metrics.csv"))
        q = d.dropna(subset=["joint_prob_median"])
        ax = axes[0][col]
        ax.fill_between(q["step"], q["joint_prob_p25"], q["joint_prob_p75"],
                        color=BLUE, alpha=0.20, linewidth=0, label="p25--p75")
        ax.plot(q["step"], q["joint_prob_median"], color=BLUE, lw=1.1, label="median")
        ax.axhline(P_CLOSED, color=MUTED, lw=0.8, ls=(0, (4, 2)))
        ax.axvline(5000, color=INK, lw=0.6, ls=":")
        ax.annotate("$\\lambda$ at full strength", (5000, 1.02), fontsize=5.4, color=INK,
                    ha="left", va="bottom")
        ax.set_ylim(0, 1.06); ax.set_title(lang, color=INK, pad=4)
        if col == 0: ax.set_ylabel("Open prob. $p_j$")

        e = d.dropna(subset=["expected_l0_step"]) if "expected_l0_step" in d else None
        ax2 = axes[1][col]
        if e is not None and len(e):
            ax2.plot(e["step"], e["expected_l0_step"], color=ORANGE, lw=1.1)
        ax2.axhline(543, color=MUTED, lw=0.6, ls=":")
        ax2.set_xlabel("Training step")
        if col == 0: ax2.set_ylabel("Expected $\\|z\\|_0$")
        for a in (ax, ax2):
            a.grid(axis="y", lw=0.5, alpha=0.7); a.set_axisbelow(True)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    h.append(Line2D([], [], color=MUTED, ls=(0, (4, 2)), lw=0.8, label="gate-closure threshold"))
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout()
    save_(fig, outdir, "fig_dynamics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rcparams()
    print("regenerating paper figures from the current run set:")
    fig_pareto(a.outdir)
    fig_consensus(a.outdir)
    fig_bodyparts(a.outdir)
    fig_gate_probs(a.outdir)
    fig_dynamics(a.outdir)


if __name__ == "__main__":
    main()
