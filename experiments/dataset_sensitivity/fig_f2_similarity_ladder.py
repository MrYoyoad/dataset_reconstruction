#!/usr/bin/env python
"""
FIGURE F2 — SIMILARITY LADDER (meeting figure).

Rebuilds the similarity-ladder figure from the saved ladder tensors
(`results/similarity_ladder/ladder_t*.pth`, job 268959) with the fixes from
notes/meeting_figures_plan.md (F2) + the yoado-6d stats specs:

  (a) ADD a companion s-vs-d LINE plot: x = pixel distance d_pixel (sorted
      ascending), y = normalized sensitivity s(d) = sens / sens(r_cross).
      One line per target; the near-duplicate floor and the rise are marked.
  (b) Overall title + a legend that SPELLS OUT what s and d are (a viewer of
      the old grid could not tell). d-axis = d_pixel (the plan's chosen axis).
  (c) The blur rung's pixel distance is RECOMPUTED from the saved T' image
      stack with the identical formula as every other rung (see BLUR_TODO
      below) — the old grid annotated d_encoder, which was nan on blur.
  (d) The two rows are labelled as TWO different digit-5 targets on
      INDEPENDENT s-scales (that is why we normalize before overlaying).

FRAMING (mandatory): OBSERVE, do not conclude. The sensitivity valley/rise
here bounds only the WEAKEST attacker (prior-free, recipe-blind, adapter-only,
per-image). It is a detection measurement, NOT the reconstruction limit.

This script only READS the .pth tensors and renders a PNG (CPU, no GPU, no
fine-tuning). It fires no experiment.
"""
import os
import math
import argparse

import torch

RESULTS = "/home/projects/galvardi/yoado/results/similarity_ladder"
FIGURES = "/home/projects/galvardi/yoado/figures/similarity_ladder"

# Rung used as the s-normalizer (the "as-different-as-a-valid-swap-gets" anchor).
NORM_RUNG = "r_cross"
# The near-duplicate floor rung (tiny additive noise) — annotated on the line plot.
FLOOR_RUNG = "p0_noise"
# The blur rung whose d we recompute from the image stack.
BLUR_RUNG = "p4_blur"


def recompute_d_pixel_from_stack(T_img, T_prime_stack):
    """L2 pixel distance ||T - T'||_2 in RAW [0,1] pixels for every rung, computed
    directly from the stored images — IDENTICAL formula to similarity_ladder.py:49
    (d_pixel = ||T - T'||_2 on raw [0,1] pixels). Recomputing from the stack for ALL
    rungs guarantees the blur rung lands on the SAME x-axis as near/median/far and
    removes any stale/undefined stored value.

    Returns a list[float], one per rung (same order as the stack).
    """
    d = []
    for c in range(T_prime_stack.shape[0]):
        d.append(torch.norm((T_prime_stack[c] - T_img).flatten()).item())
    return d


def load_targets(tag=""):
    """Load every ladder_t{idx}{tag}.pth present. Returns list of dicts."""
    targets = []
    idx = 0
    while True:
        path = os.path.join(RESULTS, f"ladder_t{idx}{tag}.pth")
        if not os.path.exists(path):
            break
        d = torch.load(path, weights_only=False)
        targets.append(d)
        idx += 1
    if not targets:
        raise FileNotFoundError(
            f"No ladder_t*{tag}.pth found under {RESULTS} — run similarity_ladder first."
        )
    return targets


def build_figure(targets, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_targets = len(targets)
    n_cols = max(t["T_prime_stack"].shape[0] for t in targets) + 1  # +1 for T itself

    # Recompute d_pixel from the saved stacks (fixes the blur d=nan on the old grid).
    for t in targets:
        t["d_pixel_recomp"] = recompute_d_pixel_from_stack(t["T_img"], t["T_prime_stack"])
        # ---- BLUR_TODO ------------------------------------------------------
        # yoado-6d spec (confirmed): the blur rung's distance is d_pixel, recomputed
        # with the IDENTICAL formula as every other rung —
        #   d_pixel = ||T - T'||_2 in RAW [0,1] pixels   (similarity_ladder.py:49),
        # with T' = _gauss_blur(T, sigma=1.0) which is the image stored in the ladder
        # .pth's T'-stack. So this value is ||T - blur(T, sigma=1.0)||_2 straight from
        # the stored images, landing on the SAME x-axis as near/median/far.
        # NOTE (caption, not a fix): the blur uses ZERO-PAD, so it slightly changes
        # ink/edge stats at the border — it is a REAL pixel perturbation, not an
        # artifact. Do NOT substitute d_encoder here.
        if BLUR_RUNG in t["rung_names"]:
            _bi = t["rung_names"].index(BLUR_RUNG)
            t["_blur_d_pixel"] = t["d_pixel_recomp"][_bi]  # used for reporting
        # ---------------------------------------------------------------------

    fig = plt.figure(figsize=(2.0 * n_cols, 2.1 * n_targets + 5.0))
    gs = GridSpec(
        n_targets + 1, n_cols, figure=fig,
        height_ratios=[1.0] * n_targets + [2.4], hspace=0.55, wspace=0.15,
    )

    # ---- (top) image grid: one row per target -------------------------------
    for r, t in enumerate(targets):
        digit = t.get("digit", "?")
        tpos = t.get("t_pos", r)
        # T itself
        ax0 = fig.add_subplot(gs[r, 0])
        ax0.imshow(t["T_img"].squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        ax0.set_title(f"Target {tpos} (digit {digit})\nT (original)", fontsize=8)
        ax0.axis("off")
        # left-edge row label with the independent-s-scale note (d)
        s_norm_ref = _norm_ref(t)
        ax0.text(-0.35, 0.5,
                 f"row {r}\nindep. s-scale\n(s(r_cross)={s_norm_ref:.2g})",
                 rotation=90, va="center", ha="center", fontsize=7,
                 transform=ax0.transAxes, color="dimgray")
        for c in range(t["T_prime_stack"].shape[0]):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(t["T_prime_stack"][c].squeeze(0).numpy(),
                      cmap="gray", vmin=0, vmax=1)
            s = t["sensitivity"][c]
            dpx = t["d_pixel_recomp"][c]
            name = t["rung_names"][c]
            s_str = f"s={s:.2g}" if math.isfinite(s) else "s=--"
            ax.set_title(f"{name}\nd={dpx:.2f}  {s_str}", fontsize=7)
            ax.axis("off")

    # ---- (bottom) companion s-vs-d line plot --------------------------------
    ax = fig.add_subplot(gs[n_targets, :])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for r, t in enumerate(targets):
        d = t["d_pixel_recomp"]
        s_ref = _norm_ref(t)
        s = [sv / s_ref if (s_ref and math.isfinite(sv)) else float("nan")
             for sv in t["sensitivity"]]
        # sort by ascending d_pixel
        order = sorted(range(len(d)), key=lambda i: d[i])
        d_sorted = [d[i] for i in order]
        s_sorted = [s[i] for i in order]
        names_sorted = [t["rung_names"][i] for i in order]
        col = colors[r % len(colors)]
        digit = t.get("digit", "?")
        tpos = t.get("t_pos", r)
        ax.plot(d_sorted, s_sorted, "-o", color=col, markersize=5,
                label=f"Target {tpos} (digit {digit})")
        # mark near-duplicate floor
        if FLOOR_RUNG in names_sorted:
            fi = names_sorted.index(FLOOR_RUNG)
            ax.annotate("near-dup floor", (d_sorted[fi], s_sorted[fi]),
                        textcoords="offset points", xytext=(6, 10), fontsize=7,
                        color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

    # horizontal reference lines: floor (0) and the normalizer (1.0 = r_cross)
    ax.axhline(0.0, color="gray", lw=0.7, ls=":")
    ax.axhline(1.0, color="gray", lw=0.7, ls="--")
    ax.text(0.005, 1.0, "s = 1.0  (r_cross normalizer)", fontsize=7,
            color="gray", va="bottom", ha="left", transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"$d$ = pixel distance to the target,  "
                  r"$\|T - T'\|_2$ in raw [0,1] pixels  (near $\to$ far)")
    ax.set_ylabel(r"$s$ = normalized sensitivity" "\n"
                  r"$s(d) = \mathrm{sens}(d)\,/\,\mathrm{sens}(r_{\mathrm{cross}})$")
    ax.set_title("Companion s-vs-d: adapter sensitivity rises as the swapped-in "
                 "image moves away from the target (near-duplicate = floor)",
                 fontsize=9)
    ax.annotate("the rise", xy=(0.62, 0.55), xycoords="axes fraction",
                fontsize=8, color="dimgray", rotation=25)
    ax.legend(fontsize=8, loc="upper left", title="one line per target "
              "(s normalized per target: scales are independent)",
              title_fontsize=7)
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "F2 - Similarity ladder: what we OBSERVE when the swapped-in image is made "
        "more similar to a training image\n"
        "d = pixel distance to the target; s = normalized adapter sensitivity to "
        "the swap. Two digit-5 targets, independent s-scales (normalized to overlay).",
        fontsize=11, y=0.995,
    )
    fig.text(0.5, 0.005,
             "What we OBSERVE: s vs d as the replacement approaches the target - a "
             "near-duplicate swap is near-invisible (floor), sensitivity rises with "
             "distance. This bounds only the WEAKEST attacker (prior-free, "
             "adapter-only, per-image); it is a detection measurement, NOT the "
             "reconstruction limit. OPEN: mechanism (raw distance vs a specific "
             "shared structure).",
             ha="center", va="bottom", fontsize=8, wrap=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[F2] saved {out_path}", flush=True)


def _norm_ref(t):
    """sens at the normalizer rung (r_cross); falls back to max sens if absent/zero."""
    names = t["rung_names"]
    sens = t["sensitivity"]
    if NORM_RUNG in names:
        v = sens[names.index(NORM_RUNG)]
        if v and math.isfinite(v):
            return v
    finite = [s for s in sens if math.isfinite(s) and s > 0]
    return max(finite) if finite else 1.0


def main():
    ap = argparse.ArgumentParser(description="Build meeting figure F2 (similarity ladder).")
    ap.add_argument("--tag", default="", help="ladder file tag (e.g. '_stage0'); '' = full run")
    ap.add_argument("--out", default=os.path.join(FIGURES, "f2_similarity_ladder.png"))
    args = ap.parse_args()

    targets = load_targets(tag=args.tag)
    # Report the recomputed blur distance(s) for the run log.
    for t in targets:
        if "_blur_d_pixel" in t:
            print(f"[F2] recomputed blur d_pixel (target {t.get('t_pos')}, "
                  f"digit {t.get('digit')}): {t['_blur_d_pixel']:.4f}", flush=True)
    build_figure(targets, args.out)


if __name__ == "__main__":
    main()
