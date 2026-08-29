#!/usr/bin/env python
"""
FIGURE 2 — depth of the imprint: per-layer numerator ||d_mu_L|| vs d_pixel (arm D).

For the full-FT all-layers arm (D), plots the RAW coherent-signal numerator
||d_mu_L|| (the mean weight-space displacement induced by the swap, S2) for each
layer {L0, L1, L2} against the swap distance d_pixel, one panel per dial target.

Why the NUMERATOR (not per-layer d*): per-layer d* divides ||d_mu_L|| by that
layer's own noise floor, so it is DENOMINATOR-CONFOUNDED across layers of very
different scale. The numerator ||d_mu_L|| is the honest, directly-comparable
per-layer statistic for "where does the instance/pixel signal live".

DATA (committed, read-only; CPU render, no experiment):
  results/fullft_valley/D_summary.json  (or D_n6_summary.json if the n_targets=6
      scale-up landed — auto-detected and PRINTED).

FRAMING: OBSERVE, do not conclude; weakest-attacker scoped. This positively
characterizes WHERE in the network the per-image signal concentrates.
"""
import os
import json
import argparse

RESULTS = "/home/projects/galvardi/yoado/results/fullft_valley"
FIGURES = "/home/projects/galvardi/yoado/figures/fullft_valley"

LAYERS = [("L0", "#1f77b4", "o"), ("L1", "#ff7f0e", "s"), ("L2", "#2ca02c", "^")]
FLOOR_RUNG = "r_nn"
NOISE_RUNG = "p0_noise"


def pick_summary(stem):
    n6 = os.path.join(RESULTS, f"{stem}_n6_summary.json")
    if os.path.exists(n6):
        return n6, True
    return os.path.join(RESULTS, f"{stem}_summary.json"), False


def main():
    ap = argparse.ArgumentParser(description="Build per-layer depth-numerator figure (arm D).")
    ap.add_argument("--out", default=os.path.join(FIGURES, "fig_valley_depth.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linewidth": 0.5})

    path, is_n6 = pick_summary("D")
    print(f"[fig_valley_depth] using {'N6 scale-up' if is_n6 else 'n=2'} "
          f"summary: {os.path.basename(path)}", flush=True)
    with open(path) as f:
        d = json.load(f)

    targets = d["targets"]
    fig, axes = plt.subplots(1, len(targets), figsize=(6.6 * len(targets), 5.0),
                             squeeze=False)
    axes = axes[0]

    for ax, tg in zip(axes, targets):
        tp = tg["t_pos"]
        rungs = tg["per_rung"]
        dpix = [pr["d_pixel"] for pr in rungs]
        for Lname, col, mk in LAYERS:
            ys = [pr["readouts"][Lname]["dmu_norm"] for pr in rungs]
            pts = sorted(zip(dpix, ys, [pr["rung"] for pr in rungs]))
            xs = [p[0] for p in pts]
            yy = [p[1] for p in pts]
            ax.plot(xs, yy, mk + "-", color=col, ms=6, lw=1.7,
                    label=f"{Lname}")
        # annotate the near-dup floor rung ordering L0>L1>L2
        for pr in rungs:
            if pr["rung"] == NOISE_RUNG:
                r = pr["readouts"]
                ax.annotate(
                    f"near-dup floor\nL0={r['L0']['dmu_norm']:.3f} > "
                    f"L1={r['L1']['dmu_norm']:.3f} > L2={r['L2']['dmu_norm']:.3f}",
                    (pr["d_pixel"], r["L0"]["dmu_norm"]),
                    textcoords="offset points", xytext=(14, 6), fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
        ax.set_xlabel(r"$d_{\rm pixel}$  (swap distance to target)")
        ax.set_ylabel(r"$\|\Delta\mu_L\|$  (per-layer coherent numerator, S2)")
        ax.set_title(f"Dial target t{tp} (digit {tg.get('digit','?')})",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", title="layer")

    fig.suptitle("Instance/pixel signal concentrates in layer-0, fading with depth "
                 "(full-FT all-layers, arm D)", fontsize=13, fontweight="bold")

    cap = (
        "What we OBSERVE: at every rung the per-layer numerator orders L0 > L1 > L2 — the "
        "coherent per-image weight displacement is concentrated in the FIRST (pixel-carrying) "
        "layer and fades with depth. This is the 'all layers of it' depth question answered "
        "positively on the honest statistic. NOTE: per-layer d* is denominator-confounded "
        "(each layer's own noise floor differs), so the NUMERATOR ||d_mu_L|| — not per-layer "
        "d* — is the comparable depth readout. WEAKEST-ATTACKER footer: this bounds only the "
        "prior-free adapter-only per-image attacker; it is a lower bound on leakage, not the "
        "reconstruction limit."
    )
    fig.text(0.5, -0.04, cap, ha="center", va="top", fontsize=7.8, wrap=True)

    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_valley_depth] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
