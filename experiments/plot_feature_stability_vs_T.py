"""Generator for the crux feature-stability-vs-T figure (NTK-survival curves):
    figures/crux/feature_stability_vs_T.png

Tests crux-mechanism part 1: smoother activations keep the first-order NTK/anchor linearization
accurate over MORE fine-tuning steps T. Metric feature_stability(T) = cos(∇f(θ0;x), ∇f(θ_T;x)); the
"NTK survival horizon" is the largest T at which it stays above NTK_FEATURE_COS_THRESHOLD (0.99).
Prediction: smoother -> survives to larger T.

Reads a rescored CSV of the fixed-lr (0.01=TRAIN_LR) T-sweep tensors exp_b_T{1,2,5,10,20,50}_r8_s42_a149_<act>.pth
(job 390026). Filters to the CLEAN activation-only T-series (drops _lr/_npc/_vw variants). Mode-independent
(feature_stability is a model diagnostic), so oracle-mode tensors are fine here.

Workflow:  python -m experiments.recompute_metrics --glob 'results/exp_b_T*_r8_s42_a149_*.pth' \
               --out results/rescored_Tsweep_<date>.csv
           python -m experiments.plot_feature_stability_vs_T --csv results/rescored_Tsweep_<date>.csv
"""
import argparse
import csv
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NTK_FS_THRESHOLD = 0.99
OUT = "figures/crux/feature_stability_vs_T.png"
# smoothness-ordered (smooth -> kinked) for the legend / colouring
SPECTRUM = ["softplus", "silu", "gelu", "mish", "gelu_tanh", "sigmoid", "tanh",
            "elu", "celu", "selu", "hardswish", "leaky_relu", "relu"]
KINKED = {"relu", "leaky_relu", "hardswish", "selu"}


def _f(r, k):
    try:
        return float(r[k])
    except Exception:
        return np.nan


def _clean_activation(fname):
    """Return the activation IFF the file is a clean activation-only T-series member, else None."""
    m = re.match(r"exp_b_T(\d+)_r8_s42_a149_([a-z_0-9.]+)\.pth$", os.path.basename(fname))
    if not m:
        return None, None
    tag = m.group(2)
    if any(s in tag for s in ("_lr", "_npc", "_vw", "free", "pbox")):
        return None, None
    return tag, int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="rescored CSV of the T-sweep tensors")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))

    # (activation) -> {T: (feature_stability, weight_change)}
    series = {}
    for r in rows:
        act, T = _clean_activation(r["file"])
        if act is None:
            continue
        series.setdefault(act, {})[T] = (_f(r, "feature_stability"), _f(r, "weight_change"))

    acts = [a for a in SPECTRUM if a in series]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(acts)))
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    ax.axhline(NTK_FS_THRESHOLD, color="#b00020", ls="--", lw=1.2)
    ax.text(1.02, NTK_FS_THRESHOLD + 0.002, f"NTK survival threshold ({NTK_FS_THRESHOLD})",
            color="#b00020", fontsize=8.5, va="bottom")

    horizon = {}
    for a, c in zip(acts, cmap):
        Ts = sorted(series[a])
        fs = [series[a][t][0] for t in Ts]
        style = "--" if a in KINKED else "-"
        ax.plot(Ts, fs, style, marker="o", color=c, lw=2, ms=5, label=a)
        above = [t for t in Ts if series[a][t][0] > NTK_FS_THRESHOLD]
        horizon[a] = max(above) if above else 0

    ax.set_xscale("log")
    ax.set_xlabel("fine-tuning steps  T  (fixed lr=0.01)", fontsize=12)
    ax.set_ylabel("feature_stability = cos(∇f(θ₀;x), ∇f(θ_T;x))", fontsize=12)
    ax.set_title("Crux mechanism (1/2): NTK survival — does smoothness keep the linearization "
                 "accurate for more steps T?", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.2, ncol=2, loc="lower left", framealpha=0.95, title="smooth → kinked (dashed)")

    # graded readout: feature_stability at the largest T (the strict 0.99 horizon is degenerate here —
    # only the smoothest reach it — so summarise by the ordering, which IS the signal)
    maxT = max(max(series[a]) for a in acts)
    fs_end = {a: series[a][max(series[a])][0] for a in acts}
    order = sorted(fs_end, key=lambda a: -fs_end[a])
    sig, soft = horizon.get("sigmoid", 0), horizon.get("softplus", 0)
    txt = ("What we OBSERVE: at every T the SMOOTHEST (sigmoid/softplus) sustain the highest linearization "
           "fidelity and the KINKED (relu/leaky_relu) the lowest — robust at the extremes. But NOT a clean "
           "monotone smoothness order across the middle: gelu/silu (C∞) start high yet decay fast; elu/celu "
           "(C¹) start low yet hold. Same two-cluster shape as the free-c ladder.\n"
           f"fs at T={maxT} (high→low):  " + "   ".join(f"{a} {fs_end[a]:.2f}" for a in order) + "\n"
           f"Strict NTK regime (fs>0.99) reached only briefly & only by the smoothest (sigmoid→T={sig}, "
           f"softplus→T={soft}) — a GRADED effect at the extremes, not a sharp horizon or a clean law.")
    ax.text(0.5, -0.19, txt, transform=ax.transAxes, ha="center", va="top", fontsize=7.8,
            color="#333", bbox=dict(boxstyle="round", fc="#eef6ee", ec="#8ab98a", alpha=0.9))
    fig.text(0.5, -0.30, "PAIR WITH freec_ladder_ranking.png (crux panel 2/2): kinked activations have the WORST "
             "linearization fidelity HERE yet LEAK THE MOST there → linearization fidelity does NOT drive "
             "leakage (the two dissociate).  |  "
             f"data: {args.csv}, fixed lr=0.01 r=8 N=2 seed 42 | feature_stability is coefficient-mode-independent "
             "(oracle tensors fine) | job 390026",
             ha="center", va="top", fontsize=7.4, color="#555")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")
    print("NTK survival horizon (T):", {a: horizon[a] for a in order})


if __name__ == "__main__":
    main()
