"""Generator for the crux REALISTIC (free-coefficient) leakage wc-ladder figure:
    figures/crux/freec_ladder_ranking.png

THE key crux-closure deliverable. The first-pass ranking is ORACLE (known-coefficient upper bound), and
this project has documented precedent that free-c FLIPS the activation ranking. This figure shows the
realistic free-c leakage (ctrl_margin_norm) across the wc-ladder {0.005, 0.03, 0.1, 0.3} and compares it
head-to-head with the oracle first-pass — answering: does "kinked leak most" survive the realistic attack,
and how does the ranking depend on weight_change?

Inputs (both rescored CSVs, same recompute_metrics columns):
  --freec  : rescore of results/exp_b_T1_r8_free_s42_a149_<act>_lr<LR>.pth (job 392821)
  --oracle : results/rescored_activations_857271_full_2026-08-28.csv (first-pass, free_coefficients=False)

Per activation the free-c CSV has 4 runs whose ACTUAL weight_change lands near the 4 rungs; each is matched
to its nearest rung. Leakage = ctrl_margin_norm (clip-robust; NOT raw, NOT eff_rank). Spearman over DISTINCT
activations per rung; the ranking's wc-DEPENDENCE and the free-c-vs-oracle sign are the headline.

Workflow:  python -m experiments.recompute_metrics --glob 'results/exp_b_T1_r8_free_s42_a149_*.pth' \
               --out results/rescored_freec_ladder_<date>.csv
           python -m experiments.plot_freec_ladder --freec results/rescored_freec_ladder_<date>.csv
"""
import argparse
import csv
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

OUT = "figures/crux/freec_ladder_ranking.png"
RUNGS = [0.005, 0.03, 0.1, 0.3]
SPECTRUM = ["relu", "leaky_relu", "hardswish", "elu", "celu", "selu", "tanh", "sigmoid",
            "gelu", "gelu_tanh", "mish", "silu", "softplus"]      # kinked -> smooth
SMOOTH_RANK = {a: i for i, a in enumerate(SPECTRUM)}              # 0=kinked ... 12=smooth
KINKED = {"relu", "leaky_relu", "hardswish", "selu"}


def _f(r, k):
    try:
        return float(r[k])
    except Exception:
        return np.nan


def _act_lr(fname, free):
    """Parse activation (and lr for free-c) from a result filename; None if not a clean base config."""
    base = os.path.basename(fname)
    if free:
        m = re.match(r"exp_b_T1_r8_free_s42_a149_([a-z_0-9.]+?)_lr[0-9.]+\.pth$", base)
    else:
        m = re.match(r"exp_b_T1_r8_s42_a149_([a-z_0-9.]+?)(?:_lr[0-9.]+)?\.pth$", base)
    if not m:
        return None
    tag = m.group(1)
    if any(s in tag for s in ("_npc", "_vw", "pbox")):
        return None
    return tag


def _by_activation_rung(csv_path, free):
    """activation -> rung -> (leakage ctrl_margin_norm, actual wc, feature_stability), nearest-rung matched."""
    out = {}
    for r in csv.DictReader(open(csv_path)):
        act = _act_lr(r["file"], free)
        if act is None or act not in SMOOTH_RANK:
            continue
        wc = _f(r, "weight_change")
        rung = min(RUNGS, key=lambda g: abs(wc - g))
        rec = (_f(r, "ctrl_margin_norm"), wc, _f(r, "feature_stability"))
        d = out.setdefault(act, {})
        # keep the run whose wc is closest to this rung if several map to it
        if rung not in d or abs(wc - rung) < abs(d[rung][1] - rung):
            d[rung] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freec", required=True)
    ap.add_argument("--oracle", default="results/rescored_activations_857271_full_2026-08-28.csv")
    args = ap.parse_args()

    freec = _by_activation_rung(args.freec, free=True)
    oracle = _by_activation_rung(args.oracle, free=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=200)
    spearmans = {}
    for ax, rung in zip(axes.flat, RUNGS):
        acts = [a for a in SPECTRUM if a in freec and rung in freec[a]]
        lk = [freec[a][rung][0] for a in acts]
        x = np.arange(len(acts))
        ax.bar(x, lk, 0.6, color=["#d62728" if a in KINKED else "#2ca02c" for a in acts], alpha=0.85,
               label="free-c (realistic)")
        # oracle overlay at the same rung
        oacts = [a for a in acts if a in oracle and rung in oracle[a]]
        ox = [acts.index(a) for a in oacts]
        ol = [oracle[a][rung][0] for a in oacts]
        ax.plot(ox, ol, "D", color="#1f77b4", ms=6, label="oracle (upper bound)")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(acts, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("leakage (ctrl_margin_norm)", fontsize=10)
        if len(acts) >= 3:
            rho = spearmanr([SMOOTH_RANK[a] for a in acts], lk).correlation
            spearmans[rung] = rho
            ax.set_title(f"wc rung ≈ {rung}   —   free-c Spearman(smoothness,leakage) = {rho:+.2f}",
                         fontsize=10, fontweight="bold")
        else:
            ax.set_title(f"wc rung ≈ {rung}   (n<3, no Spearman)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.25)

    # headline: does the sign flip vs oracle, and how does it move across the ladder?
    orho = {}
    for rung in RUNGS:
        acts = [a for a in SPECTRUM if a in oracle and rung in oracle[a]]
        if len(acts) >= 3:
            orho[rung] = spearmanr([SMOOTH_RANK[a] for a in acts],
                                   [oracle[a][rung][0] for a in acts]).correlation
    fr = "  ".join(f"{g}:{spearmans.get(g, float('nan')):+.2f}" for g in RUNGS)
    orc = "  ".join(f"{g}:{orho.get(g, float('nan')):+.2f}" for g in RUNGS)
    fig.suptitle("Crux (activation smoothness) — what we OBSERVE: the REALISTIC (free-c) leakage ranking "
                 "across the wc-ladder, vs the oracle upper bound",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.text(0.5, -0.005,
             "OBSERVED (MNIST): the realistic free-c attack does NOT flip the oracle ranking — kinked "
             "relu/leaky_relu/selu leak most at EVERY rung, stable across wc; the oracle (♦) tracks free-c closely.\n"
             f"free-c Spearman(smoothness,leakage) per rung: {fr}   |   oracle: {orc}   |  "
             "leakage=ctrl_margin_norm, Spearman over DISTINCT activations. "
             "NOT a 'smoother⇒less leakage' law (sign still flips within the smooth-only subset; flowers behaves "
             "differently — dataset-dependence OPEN). oracle@0.005 undersampled (first-pass used different LRs).",
             ha="center", va="top", fontsize=7.8, color="#333")
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")
    print(f"free-c Spearman per rung: {spearmans}")
    print(f"oracle Spearman per rung: {orho}")


if __name__ == "__main__":
    main()
