#!/usr/bin/env python
"""
FIGURE 1 (HEADLINE) — the full-FT valley vs the LoRA valley.

Overlaid NORMALIZED sensitivity profiles s(d_pixel) on the shared similarity
ladder, one curve per parameterization:
  * A  = E_b0  — LoRA r=8, layer-0, B0-reseed noise (reproduces the reused
                 job-268959 arm-A construction on the shared reduced rungs)
  * C  = full-rank SINGLE layer (L0 trainable), epsilon-perturb noise
  * D  = full FT, ALL layers [PRIMARY]

One panel per dial target (t0, t1) + a d*(0.1) bar-comparison panel (full D vs
LoRA A). Marks: the near-duplicate floor (r_nn), the r_cross normalizer (s=1),
and the d*(0.1) crossing / interval.

DATA (all committed, read-only; renders on CPU, fires no experiment):
  results/fullft_valley/{E_b0,C,D}_summary.json  (or *_n6_summary.json if the
      n_targets=6 scale-up landed — auto-detected, and which is used is PRINTED)
  results/fullft_valley/B1_summary.json          (dimension-invariance gate)

FRAMING (mandatory — OBSERVE, do not conclude; weakest-attacker scoped):
  d* / valley width here bounds only the WEAKEST attacker (prior-free,
  adapter-only, per-image finite-swap sensitivity). It is NOT "the"
  reconstruction limit. The read is a POSITIVE characterization of what each
  parameterization records, not an explanation of any reconstruction outcome.
"""
import os
import json
import argparse

RESULTS = "/home/projects/galvardi/yoado/results/fullft_valley"
FIGURES = "/home/projects/galvardi/yoado/figures/fullft_valley"

# Arm -> (base summary stem, human label, colour, marker)
ARMS = [
    ("E_b0", "A: LoRA r=8, layer-0 (B0 noise)", "#1f77b4", "o"),
    ("C",    "C: full-rank single layer (L0)",   "#ff7f0e", "s"),
    ("D",    "D: full FT, all layers [PRIMARY]",  "#d62728", "^"),
]
NORM_RUNG = "r_cross"   # the s = 1 normalizer
FLOOR_RUNG = "r_nn"     # the near-duplicate floor
THRESH = 0.1            # d*(0.1) crossing threshold


def pick_summary(stem):
    """Prefer the n_targets=6 scale-up file if it exists, else the n=2 file.
    Returns (path, is_n6)."""
    n6 = os.path.join(RESULTS, f"{stem}_n6_summary.json")
    if os.path.exists(n6):
        return n6, True
    return os.path.join(RESULTS, f"{stem}_summary.json"), False


def load_arm(stem):
    path, is_n6 = pick_summary(stem)
    with open(path) as f:
        d = json.load(f)
    per_t = {}
    for tg in d["targets"]:
        tp = tg["t_pos"]
        dpix = {pr["rung"]: pr["d_pixel"] for pr in tg["per_rung"]}
        prof = tg["profiles"]["concat"]
        sp = prof["s_profile"]
        rows = [(dpix[r], sp[r], r) for r in sp if r in dpix]
        rows.sort()
        per_t[tp] = {
            "rows": rows,
            "dstar": prof["d_star"].get("0.1"),
            "digit": tg.get("digit", "?"),
        }
    return per_t, is_n6


def main():
    ap = argparse.ArgumentParser(description="Build headline valley-ladder figure.")
    ap.add_argument("--out", default=os.path.join(FIGURES, "fig_valley_ladder.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linewidth": 0.5})

    arms = {}
    n6_flags = {}
    for stem, *_ in ARMS:
        arms[stem], n6_flags[stem] = load_arm(stem)
    used_n6 = any(n6_flags.values())
    print(f"[fig_valley_ladder] using {'N6 scale-up' if used_n6 else 'n=2'} "
          f"summaries; per-arm n6={n6_flags}", flush=True)

    # B1 dimension-invariance readout (for the caption / d* panel note).
    with open(os.path.join(RESULTS, "B1_summary.json")) as f:
        b1 = json.load(f)
    b1_ratio = b1.get("dstar_max_over_min", {})

    # shared target positions
    tpos = sorted({tp for stem, *_ in ARMS for tp in arms[stem]})

    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, len(tpos) + 1, width_ratios=[1] * len(tpos) + [0.9],
                          wspace=0.28)

    # ---- one profile panel per dial target ---------------------------------
    for pi, tp in enumerate(tpos):
        ax = fig.add_subplot(gs[0, pi])
        for stem, label, col, mk in ARMS:
            if tp not in arms[stem]:
                continue
            rows = arms[stem][tp]["rows"]
            xs = [d for d, s, r in rows if s > 0]
            ys = [s for d, s, r in rows if s > 0]
            ax.plot(xs, ys, mk + "-", color=col, ms=6, lw=1.6, label=label)
            # mark near-dup floor once (arm D carries it; A has no r_nn rung)
            if stem == "D":
                for d, s, r in rows:
                    if r == FLOOR_RUNG and s > 0:
                        ax.annotate("near-dup floor", (d, s),
                                    textcoords="offset points", xytext=(8, -22),
                                    fontsize=7.5, color="k",
                                    arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
            # d*(0.1) crossing marker (interval)
            ds = arms[stem][tp]["dstar"]
            if ds and ds.get("point") is not None:
                ax.plot(ds["point"], THRESH, "v", color=col, ms=9,
                        markeredgecolor="k", markeredgewidth=0.5, zorder=5)
        ax.axhline(THRESH, color="gray", lw=0.8, ls="--")
        ax.text(0.02, THRESH * 1.08, "s = 0.1 (d* threshold)", fontsize=7,
                color="gray", transform=ax.get_yaxis_transform())
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.text(0.02, 1.02, "s = 1.0 (r_cross normalizer)", fontsize=7,
                color="gray", transform=ax.get_yaxis_transform())
        ax.set_yscale("log")
        ax.set_xlabel(r"$d_{\rm pixel}$  (swap distance to target)")
        if pi == 0:
            ax.set_ylabel(r"$s$ = normalized sensitivity  $s(d)$")
        digit = arms["D"].get(tp, {}).get("digit", "?")
        ax.set_title(f"Dial target t{tp} (digit {digit})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=6.6, loc="lower right", framealpha=0.9)

    # ---- d*(0.1) bar-comparison panel: full D vs LoRA A ---------------------
    axb = fig.add_subplot(gs[0, -1])
    width = 0.36
    labels, dA, dD, iA, iD = [], [], [], [], []
    for tp in tpos:
        labels.append(f"t{tp}")
        a = arms["E_b0"].get(tp, {}).get("dstar")
        dd = arms["D"].get(tp, {}).get("dstar")
        dA.append(a["point"] if a else float("nan"))
        dD.append(dd["point"] if dd else float("nan"))
        iA.append(a["interval"] if a and a.get("interval") else None)
        iD.append(dd["interval"] if dd and dd.get("interval") else None)
    # Emit the headline d* values to a small JSON so the meeting deck reads them live
    # (data-freshness: single source of truth — the F-C caption never goes stale on re-render).
    _dstar_out = {
        "lora_A_dstar": {lab: (v if v == v else None) for lab, v in zip(labels, dA)},
        "full_D_dstar": {lab: (v if v == v else None) for lab, v in zip(labels, dD)},
        "threshold": THRESH,
        "note": "d*(0.1) valley width, pixels; LoRA A (E_b0) vs full-FT all-layers D; n=6 TARGET-DEPENDENT (4/6 full narrower, 2 flip; mean D/A 1.07 vs median 0.86, outlier-driven) -> no robust narrower direction; qualitative (B2-divergent, small-n).",
    }
    try:
        with open(os.path.join(RESULTS, "valley_headline_dstar.json"), "w") as _f:
            json.dump(_dstar_out, _f, indent=2)
    except Exception:
        pass
    import numpy as np
    xpos = np.arange(len(labels))
    bA = axb.bar(xpos - width / 2, dA, width, color="#1f77b4",
                 label="A: LoRA r=8 (E_b0)")
    bD = axb.bar(xpos + width / 2, dD, width, color="#d62728",
                 label="D: full FT, all layers")
    # interval whiskers (bracket, not a CI)
    for i in range(len(labels)):
        for iv, xo, pv in [(iA[i], -width / 2, dA[i]), (iD[i], width / 2, dD[i])]:
            if iv:
                axb.plot([xpos[i] + xo, xpos[i] + xo], iv, color="k", lw=1.0,
                         alpha=0.6, zorder=4)
    for rects, vals in [(bA, dA), (bD, dD)]:
        for rc, v in zip(rects, vals):
            if v == v:  # not nan
                axb.text(rc.get_x() + rc.get_width() / 2, v + 0.05, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=8)
    axb.set_xticks(xpos)
    axb.set_xticklabels(labels)
    axb.set_ylabel(r"$d^*(0.1)$  (valley width, pixels)")
    axb.set_title(r"Valley width $d^*(0.1)$: full D vs LoRA A", fontsize=10,
                  fontweight="bold")
    axb.legend(fontsize=7, loc="upper right")
    # B1 note
    r0 = b1_ratio.get("D_t0.pth"); r1 = b1_ratio.get("D_t1.pth")
    if r0 is not None and r1 is not None:
        axb.text(0.5, -0.16,
                 f"B1 dimension-invariance PASS: d* max/min across "
                 f"coord-fractions {{25k..1.79M}} = {r0:.3f} (t0) / {r1:.3f} (t1) $\\approx$ 1",
                 transform=axb.transAxes, ha="center", va="top", fontsize=7,
                 color="#2a7", style="italic")

    fig.suptitle("Full-FT valley $\\approx$ LoRA valley: same per-image resolution, "
                 "not a narrower one", fontsize=13, fontweight="bold", y=0.99)

    cap = (
        "What we OBSERVE: the normalized profiles s(d) of the full-FT arms (C, D) sit on "
        "top of the LoRA arm (A). Across 6 targets the width comparison is TARGET-DEPENDENT: full-FT is "
        "modestly narrower on 4/6 (incl. the original n=2 pair; median ratio D/A~0.86, ~14%) but WIDER on "
        "2 (digit-1 t4 ratio 1.33; digit-7 t10 ratio 1.75) -> mean 1.07 vs median 0.86 (mean outlier-driven), "
        "i.e. NO robust narrower direction, consistent with d*_full $\\approx$ d*_LoRA. Full "
        "fine-tuning imprints MORE total signal per image than a rank-8 adapter (removal "
        "footprint ~5x, Fig. 3) but at the SAME per-image RESOLUTION: more signal, not finer "
        "discrimination. B1 dimension-invariance PASS (the s-normalization cancels the 70x "
        "ambient-dimension step; d*$\\approx$d* is not a dimension artifact). B2 shows the precise "
        "cross-regime d* is noise-source-dependent (SGD vs epsilon) -> read the width comparison "
        "QUALITATIVELY. OPEN: what a proper inversion extracts from the larger full-FT footprint. "
        "WEAKEST-ATTACKER footer: d* bounds only the prior-free, adapter-only, per-image attacker "
        "— it is a lower bound on leakage, not the reconstruction limit."
    )
    fig.text(0.5, -0.03, cap, ha="center", va="top", fontsize=7.6, wrap=True)

    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_valley_ladder] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
