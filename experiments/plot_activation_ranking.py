"""Generator for the crux first-pass activation-ranking figure:
    figures/crux/activation_ranking_857271.png

Reads the full 152-config rescore of job 857271 (all surviving exp_b_T1_r8_s42_a149_* tensors) and
plots, smoothness-ordered, the two crux metrics that DIVERGE:
  - NTK survival  = feature_stability  (tracks smoothness: smoother -> more linear)
  - leakage       = ctrl_margin_norm   (clip-robust; recon ssim_norm - control ssim_norm)

Headline (corroborates notes/crux_activation_analysis.md on the FULL spectrum, 152 vs the earlier 27):
  the naive "smoother -> more leakage" law is REFUTED. Kinked relu/leaky_relu/selu leak the MOST
  (ctrl_margin_norm 0.49-0.59) while having the LOWEST NTK survival -> linearization fidelity does NOT
  set leakage magnitude. Spearman(smoothness, leakage) = -0.38 across the spectrum.

HONEST CAVEAT (load-bearing, shown on the figure): every config is T=1 and ntk_passed is False -> this is
a first-pass at the best-matched-wc (~0.10) point available, NOT the matched-wc leakage ranking. The clean
matched-weight_change ranking + the feature-stability-vs-T curve need NEW GPU runs with a corrected LR band.

Run:  python -m experiments.plot_activation_ranking
Data: results/rescored_activations_857271_full_2026-08-28.csv
"""
import csv
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = "results/rescored_activations_857271_full_2026-08-28.csv"
OUT = "figures/crux/activation_ranking_857271.png"
TARGET_WC = 0.10
# smoothness-ordered (kinked -> smooth) for the main spectrum panel
ORDER = ["relu", "leaky_relu", "hardswish", "elu", "celu", "selu", "tanh", "sigmoid",
         "gelu", "gelu_tanh", "mish", "silu", "softplus"]
BETAS = ["softplus_b50", "softplus_b10", "softplus_b5", "softplus_b2", "softplus_b1", "softplus_b0.5"]
KINKED = {"relu", "leaky_relu", "hardswish", "selu"}


def _f(r, k):
    try:
        return float(r[k])
    except Exception:
        return np.nan


def _load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        m = re.search(r"a149_(.+?)\.pth", r["file"])
        r["tag"] = m.group(1) if m else r["file"]
        r["is_variant"] = ("npc" in r["file"]) or ("vw" in r["file"])
        t = r["tag"]
        r["act"] = t.split("_lr")[0].split("_npc")[0] if t.startswith("softplus_b") else r["finetune_activation"]
    byact = {}
    for r in rows:
        if r["is_variant"]:
            continue
        byact.setdefault(r["act"], []).append(r)
    return byact


def _pick(rs):
    return min(rs, key=lambda r: abs(_f(r, "weight_change") - TARGET_WC))


def main():
    byact = _load()
    fig, (axS, axB) = plt.subplots(1, 2, figsize=(16, 6.2), dpi=200,
                                   gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.22})

    # ---- main spectrum ----
    acts = [a for a in ORDER if a in byact]
    fs = [_f(_pick(byact[a]), "feature_stability") for a in acts]
    lk = [_f(_pick(byact[a]), "ctrl_margin_norm") for a in acts]
    x = np.arange(len(acts))
    axS.axhline(0, color="k", lw=0.8)
    bars = axS.bar(x, lk, 0.6, color=["#d62728" if a in KINKED else "#2ca02c" for a in acts],
                   alpha=0.85, label="leakage = ctrl_margin_norm (clip-robust)")
    axS.set_ylabel("leakage  (ctrl_margin_norm)", fontsize=12)
    ax2 = axS.twinx()
    ax2.plot(x, fs, "D-", color="#1f77b4", lw=2, ms=7, label="NTK survival = feature_stability")
    ax2.set_ylabel("NTK survival  (feature_stability)", color="#1f77b4", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0.55, 1.02)
    axS.set_xticks(x)
    axS.set_xticklabels(acts, rotation=40, ha="right", fontsize=9)
    axS.set_title("Crux first-pass — leakage & NTK-survival DIVERGE across the smoothness spectrum",
                  fontsize=11, fontweight="bold")
    axS.text(0.015, 0.97,
             "← kinked            smoothness →            smooth\n"
             "REFUTED: kinked relu/leaky_relu/selu (red) leak the MOST yet have the\n"
             "LOWEST NTK survival — linearization fidelity does NOT set leakage.\n"
             "Spearman(smoothness, leakage) = −0.38  (feature_stability +0.29).",
             transform=axS.transAxes, va="top", fontsize=8.4,
             bbox=dict(boxstyle="round", fc="#fdecea", ec="#b00020", alpha=0.92))
    h1, l1 = axS.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axS.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.6, framealpha=0.95)
    # flag the worst unmatched-wc offender: sigmoid never reaches wc 0.10 (max ~0.01)
    if "sigmoid" in acts:
        i = acts.index("sigmoid")
        axS.annotate("sigmoid @ wc 0.01\n(never reaches 0.10)", (i, lk[i]),
                     textcoords="offset points", xytext=(0, 14), ha="center", fontsize=6.8,
                     color="#a06000", arrowprops=dict(arrowstyle="-", color="#a06000", lw=0.7))

    # ---- softplus-β knob (single family, only smoothness varies) ----
    bacts = [a for a in BETAS if a in byact]
    blk = [_f(_pick(byact[a]), "ctrl_margin_norm") for a in bacts]
    bfs = [_f(_pick(byact[a]), "feature_stability") for a in bacts]
    xb = np.arange(len(bacts))
    axB.bar(xb, blk, 0.6, color="#8888cc", alpha=0.85, label="leakage (ctrl_margin_norm)")
    axB.set_ylabel("leakage  (ctrl_margin_norm)", fontsize=11)
    axb2 = axB.twinx()
    axb2.plot(xb, bfs, "D-", color="#1f77b4", lw=2, ms=6, label="feature_stability")
    axb2.set_ylabel("feature_stability", color="#1f77b4", fontsize=11)
    axb2.tick_params(axis="y", labelcolor="#1f77b4")
    axb2.set_ylim(0.6, 1.02)
    axB.set_xticks(xb)
    axB.set_xticklabels([b.replace("softplus_", "") for b in bacts], rotation=30, ha="right", fontsize=9)
    axB.set_title("softplus-β knob (one family, only smoothness varies)", fontsize=11, fontweight="bold")
    axB.text(0.03, 0.97,
             "sharp ← β → smooth. Within-family: feature_stability rises\n"
             "with smoothness (+0.71); leakage +0.60 but NON-monotonic\n"
             "(b50 rises back toward relu-like) — not a clean law either.",
             transform=axB.transAxes, va="top", fontsize=8.0,
             bbox=dict(boxstyle="round", fc="#eef2fb", ec="#8899cc", alpha=0.92))

    fig.suptitle("Activation crux — FIRST-PASS on 152 surviving job-857271 configs (T=1, unmatched-wc): "
                 "smoothness→NTK-survival holds; smoothness→leakage REFUTED",
                 fontsize=11.5, fontweight="bold", y=1.0)
    fig.text(0.5, -0.03,
             f"data: {CSV} | leakage=ctrl_margin_norm (NOT raw ctrl_margin, NOT eff_rank) @ best-matched wc≈{TARGET_WC} | "
             "CAVEAT: all T=1, ntk_passed=False → first-pass, NOT the matched-wc ranking; "
             "matched-wc + feature-stability-vs-T need new GPU runs.",
             ha="center", fontsize=7.8, color="#555")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
