#!/usr/bin/env python
"""
FIGURE F3 — WHO LEAKS: per-image sensitivity vs the g0 predictor.

Rebuilds the margin-at-scale figure from `results/margin_at_scale/summary.json`
+ `margin_at_scale.pth` (job 272504) with the fixes from
notes/meeting_figures_plan.md (F3):

  (a) RETITLE so "at scale" is NOT misread as dataset-size: this is a per-image
      WHO-leaks test (n=24 stratified targets), not an N-sweep.
  (b) Render the verdict HONESTLY. The verdict is INDETERMINATE
      (rho = +0.777, n=24, bootstrap 95% CI [+0.529, +0.907] half-width 0.189
      > the pre-registered 0.15; g0-tercile sign flip, ordered [LOW-g0, mid,
      HIGH-g0] = +0.881, +0.500, -0.119 - the predictor is STRONG at LOW g0 and
      FLATTENS/reverses at HIGH g0, i.e. rho(sens,g0) SATURATES as g0 grows; WHY
      is OPEN). We show rho WITH its bootstrap CI and the per-stratum
      (tercile) sign pattern. We DO NOT stamp a pass. "confirmed" -> "observed".

FRAMING (mandatory): OBSERVE, do not conclude. rho(sens, g0) measures how well a
base-model gradient quantity RANKS which images the weakest attacker detects;
it bounds only that weakest attacker (prior-free, adapter-only, per-image) and is
NOT the reconstruction limit.

Reads JSON + .pth only; renders a PNG (CPU, no GPU). Fires no experiment.
"""
import os
import json
import argparse

RESULTS = "/home/projects/galvardi/yoado/results/margin_at_scale"
FIGURES = "/home/projects/galvardi/yoado/figures/margin_at_scale"


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def _terciles_by_g0(per_target):
    """Split the targets into g0 low/mid/high terciles (by rank). Returns a list of
    tercile-index (0=low,1=mid,2=high) aligned to per_target order."""
    order = sorted(range(len(per_target)), key=lambda i: per_target[i]["g0"])
    n = len(order)
    tier = [0] * n
    for rank, i in enumerate(order):
        # 3 equal-ish groups
        tier[i] = min(2, rank * 3 // n)
    return tier


def build_figure(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = summary["headline"]
    per_target = summary["per_target"]
    cfg = summary["config"]
    n = h["n"]
    rho = h["rho_sens_g0"]
    ci = h["ci95"]                       # [lo, hi] bootstrap 95%
    ci_hw = h["ci_halfwidth"]
    perm_p = h["perm_p"]
    tercile_rhos = h["tercile_rhos"]     # [low, mid, high]
    sign_flip = h["sign_flip_across_terciles"]
    verdict = summary.get("verdict", "INDETERMINATE")

    g0 = [t["g0"] for t in per_target]
    sens = [t["sensitivity"] for t in per_target]
    tier = _terciles_by_g0(per_target)
    tier_colors = ["#4C72B0", "#DD8452", "#C44E52"]  # low, mid, high g0
    tier_names = ["g0 low tercile", "g0 mid tercile", "g0 high tercile"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))

    # ---- panel (a): scatter sens vs g0, colored by g0 tercile ---------------
    ax = axes[0]
    for tk in (0, 1, 2):
        xs = [g0[i] for i in range(n) if tier[i] == tk]
        ys = [sens[i] for i in range(n) if tier[i] == tk]
        ax.scatter(xs, ys, s=42, color=tier_colors[tk], label=tier_names[tk],
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlabel(r"$g_0$ = per-image base-model gradient norm "
                  r"$\|\nabla_{W_0}\,\mathrm{BCE}\|_F$  (the predictor)")
    ax.set_ylabel("adapter sensitivity to swapping this image out\n"
                  "(whitened per-image detectability)")
    ax.set_title("(a) Which images leak: per-image sensitivity vs the "
                 f"g0 predictor\n(per-image, n={n} stratified targets - "
                 "NOT a dataset-size sweep)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    # OBSERVED rho annotation (softened language; no pass stamp)
    ax.text(0.97, 0.05,
            f"OBSERVED Spearman rho(sens, g0) = {rho:+.3f}\n"
            f"perm-p = {perm_p:.1e}  (n={n})\n"
            f"bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray", alpha=0.9))

    # ---- panel (b): rho with CI + per-tercile sign pattern ------------------
    ax = axes[1]
    # overall rho as a point with the bootstrap CI as a horizontal error bar
    y_overall = 3
    lo_err = rho - ci[0]
    hi_err = ci[1] - rho
    ax.errorbar([rho], [y_overall], xerr=[[lo_err], [hi_err]], fmt="o",
                color="black", capsize=5, markersize=8, zorder=4,
                label="overall rho +/- boot 95% CI")
    # per-tercile rhos (the sign pattern that makes the verdict indeterminate)
    for tk, tr in enumerate(tercile_rhos):
        ax.scatter([tr], [tk], s=70, color=tier_colors[tk], zorder=4,
                   edgecolor="white", linewidth=0.6)
        ax.text(tr, tk + 0.18, f"{tr:+.3f}", ha="center", va="bottom",
                fontsize=8.5, color=tier_colors[tk])
    # reference lines: pre-registered PASS bar (+0.6) and KILL bar (+0.3), zero
    ax.axvline(0.0, color="gray", lw=0.8, ls=":")
    ax.axvline(0.6, color="green", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(0.3, color="red", lw=1.0, ls="--", alpha=0.7)
    ax.text(0.6, 3.7, "pre-reg PASS bar\n(rho>+0.6, CI hw<0.15)", fontsize=7,
            color="green", ha="center", va="bottom")
    ax.text(0.3, -0.75, "KILL bar\n(rho<+0.3)", fontsize=7, color="red",
            ha="center", va="top")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["g0 low tercile", "g0 mid tercile", "g0 high tercile",
                        "OVERALL"], fontsize=8.5)
    ax.set_xlim(-0.4, 1.05)
    ax.set_ylim(-1.2, 4.4)
    ax.set_xlabel("Spearman rho(sens, g0)")
    ax.set_title("(b) rho with bootstrap 95% CI + per-tercile sign pattern",
                 fontsize=10)
    ax.grid(True, axis="x", alpha=0.25)
    # honest verdict box — INDETERMINATE, NOT a pass
    flip_txt = "SIGN FLIP across terciles" if sign_flip else "no sign flip"
    ax.text(0.5, -1.05,
            f"VERDICT: INDETERMINATE (above KILL, below PASS)\n"
            f"CI half-width {ci_hw:.3f} > pre-registered 0.15  |  {flip_txt} "
            f"(LOW-g0 {tercile_rhos[0]:+.2f}, mid {tercile_rhos[1]:+.2f}, "
            f"HIGH-g0 {tercile_rhos[2]:+.2f}: strong at low g0, reverses at high g0)\n"
            "# yoado-6d: confirm CI + per-stratum",
            transform=ax.transData, ha="center", va="top", fontsize=7.8,
            bbox=dict(boxstyle="round", fc="#fff3e0", ec="#DD8452", alpha=0.95))

    fig.suptitle(
        "F3 - WHO leaks: per-image sensitivity vs the g0 predictor  "
        f"(per-image, n={n}; NOT dataset size)",
        fontsize=12.5, y=1.0,
    )
    fig.text(0.5, -0.02,
             "What we OBSERVE: g0 (a base-model gradient norm) ranks WHICH images "
             "the weakest attacker detects - the predictor is STRONG at LOW g0 "
             "(+0.88) and FLATTENS/reverses at HIGH g0 (-0.12), i.e. rho(sens,g0) "
             "SATURATES as g0 grows (tercile sign flip). The overall correlation is "
             "positive but the CI is wide: verdict INDETERMINATE, not a pass. This "
             "ranks weakest-attacker (prior-free, adapter-only, per-image) "
             "detectability; it is NOT the reconstruction limit. "
             "OPEN: WHY it saturates at high g0, and tighten the CI (n>24).",
             ha="center", va="top", fontsize=8, wrap=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[F3] saved {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build meeting figure F3 (who leaks / g0).")
    ap.add_argument("--summary", default=os.path.join(RESULTS, "summary.json"))
    ap.add_argument("--out", default=os.path.join(FIGURES, "f3_margin_who_leaks.png"))
    args = ap.parse_args()

    summary = load_summary(args.summary)
    h = summary["headline"]
    print(f"[F3] rho(sens,g0)={h['rho_sens_g0']:+.4f}  n={h['n']}  "
          f"perm_p={h['perm_p']:.2e}  CI={h['ci95']}  hw={h['ci_halfwidth']:.4f}  "
          f"terciles={h['tercile_rhos']}  verdict={summary.get('verdict')}", flush=True)
    build_figure(summary, args.out)


if __name__ == "__main__":
    main()
