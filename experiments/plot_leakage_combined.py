"""Generator for the CANONICAL combined leakage figure:
    figures/combined/leakage_identifiability_plus_reconstruction.png

ONE honest two-panel story, fully reproducible from committed .pth data (nothing hardcoded).
Merged 2026-08-28 (best-of-both, consolidating two parallel honest rebuilds):
  LEFT  — IDENTIFIABILITY (verified): q_eff|col(J) vs attacker budget ε, binary(BCE) vs 10-class(CE),
          read from the col(J)-restricted whitened count `colspace.q_eff` in the roundB canonical cell
          (N=20, T=1000, S=1280, r_J=160). Both HIGH; 10-class leaks FEWER at every ε (NOT amplified).
  RIGHT — RECONSTRUCTION (honest): decoded (adapter-only attack) raw-SSIM MINUS the mean-image baseline
          for all 12 gb_e2e cells (every bar negative -> 0/40 arms clear the trivial floor), with the
          TRUE-ΔW oracle gap overlaid (markers) so "even the oracle fails on small nets" is visible.
          Geometry leaks; pixels do NOT (yet) — an OPEN limitation, not a result.

History: this figure once carried an overclaim ("ssim_norm 0.61 / recognizable / leakage REAL in pixels")
that compared a mean/std-MATCHED score to a RAW baseline (apples-to-oranges). Retracted after the sibling
audit (yoado-a2/aa): on the honest raw-vs-raw test 0/40 decoded arms beat baseline. This generator plots
only the honest raw comparison. See notes/leakage_story_consolidated.md sec.3.

Right-panel gap logic + the anchor self-check are harvested from the parallel honest rebuild
(plot_combined_honest.py, yoado-6d) so the merged canonical figure keeps the stronger all-12-cells panel.

Run:  python -m experiments.plot_leakage_combined
Data: results/jacobian_j1_roundB_mnist_nc{2,10}_T1000_S1280.pth   (identifiability; job 484948 lineage)
      results/gb_e2e_{mnist,fashion}_N{2,4,10}_{gelu,softplus}.pth  (reconstruction; GB Phase-2 e2e)
"""
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = "figures/combined/leakage_identifiability_plus_reconstruction.png"
IDENT = {2: "results/jacobian_j1_roundB_mnist_nc2_T1000_S1280.pth",
         10: "results/jacobian_j1_roundB_mnist_nc10_T1000_S1280.pth"}
CELL = (1280, 0.01)                 # (S, shrink) key of the canonical converged cell
EPS = [0.1, 0.3, 1.0, 3.0, 10.0]
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9e9e9e"


def main():
    torch.set_default_dtype(torch.float64)

    # ---- LEFT: identifiability (roundB canonical N=20 cell), read from data ----
    b = torch.load(IDENT[2], map_location="cpu", weights_only=False)
    m = torch.load(IDENT[10], map_location="cpu", weights_only=False)
    bcs, mcs = b["results"][CELL]["colspace"], m["results"][CELL]["colspace"]
    rJ = int(bcs["r_J"])
    bq = np.array([bcs["q_eff"][e] for e in EPS])
    mq = np.array([mcs["q_eff"][e] for e in EPS])
    # anchor self-check (hard abort on data drift) — harvested from plot_combined_honest.py
    assert bcs["q_eff"][1.0] == 119 and mcs["q_eff"][1.0] == 80 and rJ == 160, \
        (f"ANCHOR FAIL: bin@ε1={bcs['q_eff'][1.0]} (exp 119), 10cls@ε1={mcs['q_eff'][1.0]} (exp 80), "
         f"r_J={rJ} (exp 160)")
    print(f"[anchor] OK — roundB N=20: bin 119 / 10cls 80 @ε=1, r_J=160, "
          f"iso {bcs['iso_ratio']:.3f}/{mcs['iso_ratio']:.3f}")

    # ---- RIGHT: reconstruction gap-to-baseline for all gb_e2e cells + oracle overlay ----
    labels, dec_gap, ora_gap = [], [], []
    for f in sorted(glob.glob("results/gb_e2e_mnist_N*.pth") + glob.glob("results/gb_e2e_fashion_N*.pth")):
        d = torch.load(f, map_location="cpu", weights_only=False)["results"]
        dec_ss, _norm, base = d["DECODED all-layers"]      # (raw ssim, ssim_norm, raw mean-image baseline)
        true_ss = d["TRUE ΔW (ceiling)"][0]
        labels.append(os.path.basename(f).replace("gb_e2e_", "").replace(".pth", ""))
        dec_gap.append(dec_ss - base)
        ora_gap.append(true_ss - base)
    dec_gap, ora_gap = np.array(dec_gap), np.array(ora_gap)
    n_beat = int((dec_gap > 0).sum())

    fig = plt.figure(figsize=(16, 6.4), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)

    # LEFT panel
    axL = fig.add_subplot(gs[0, 0])
    axL.fill_between(EPS, bq, mq, color=ORANGE, alpha=0.15, label="reversal gap (binary − 10-class)")
    axL.semilogx(EPS, bq, "o-", color=BLUE, lw=2.4, ms=8, label="binary (BCE)")
    axL.semilogx(EPS, mq, "s-", color=ORANGE, lw=2.4, ms=8, label="10-class (CE)")
    axL.axhline(rJ, color="grey", ls="--", lw=1, alpha=0.7)
    axL.text(EPS[0], rJ + 2, f"full rank r_J = {rJ}", fontsize=9, color="grey", va="bottom")
    for x, y in zip(EPS, bq):
        axL.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=8, color=BLUE, fontweight="bold")
    for x, y in zip(EPS, mq):
        axL.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, -15), ha="center",
                     fontsize=8, color="#d2691e", fontweight="bold")
    axL.set_xscale("log")
    axL.set_xticks(EPS)
    axL.set_xticklabels([("%g" % e) for e in EPS])
    axL.set_xlabel("attacker SNR budget  ε", fontsize=12)
    axL.set_ylabel(f"q_eff|col(J) = recoverable directions (of {rJ})", fontsize=12)
    axL.set_ylim(0, rJ + 14)
    axL.set_title("A. Identifiability — recoverable directions vs attacker budget ε",
                  fontsize=10.5, fontweight="bold")
    axL.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # RIGHT panel — 12-cell decoded gap + oracle overlay
    axR = fig.add_subplot(gs[0, 1])
    x = np.arange(len(labels))
    axR.bar(x, dec_gap, 0.6, color=[GREEN if g > 0 else RED for g in dec_gap], alpha=0.85,
            label="decoded − baseline (adapter-only attack)")
    axR.plot(x, ora_gap, "D", color=BLUE, ms=6, label="TRUE ΔW oracle − baseline")
    axR.axhline(0, color="k", lw=1)
    for xi, g in zip(x, dec_gap):
        axR.text(xi, g - 0.012, f"{g:+.2f}", ha="center", va="top", fontsize=7,
                 fontweight="bold", color=RED)
    axR.set_ylabel("raw SSIM − mean-image baseline", fontsize=12)
    axR.set_title("B. Pixel reconstruction — decoded vs mean-image baseline",
                  fontsize=10.5, fontweight="bold")
    axR.set_xticks(x)
    short = [l.replace("fashion", "F").replace("mnist", "M").replace("gelu", "ge")
             .replace("softplus", "sp").replace("_", " ") for l in labels]
    axR.set_xticklabels(short, rotation=40, ha="right", fontsize=7.5)
    axR.legend(loc="upper right", fontsize=8.6, framealpha=0.95)

    fig.suptitle("LoRA leakage — identifiability (left) and pixel reconstruction (right)",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.text(0.5, -0.13,
             "What we OBSERVE. LEFT: both bases recover a large share of private directions; 10-class recovers "
             "FEWER at every ε (not amplified). RIGHT: every decoded bar sits below the trivial mean-image "
             "baseline (0/40 arms clear it); the oracle (♦) clears it only on easy cells. This 0/40 is the "
             "adapter-only DECODER/recipe failing, NOT an information limit — identifiability (left) shows the "
             "info is present ⇒ decoder/recipe-limited, not information-limited. OPEN: whether priors / "
             "known-recipe inversion / a stronger decoder cross it.\n"
             "identifiability: results/jacobian_j1_roundB_*.pth (job 484948) | reconstruction: "
             "results/gb_e2e_*.pth + metrics.py baseline gate | audit yoado-a2/aa.",
             ha="center", fontsize=7.6, color="#444")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")
    print(f"  identifiability (r_J={rJ}): binary {list(bq)} vs 10-class {list(mq)} at ε={EPS}")
    print(f"  reconstruction: {n_beat}/{len(labels)} decoded arms beat baseline "
          f"(oracle beats on {int((ora_gap > 0).sum())}/{len(labels)})")


if __name__ == "__main__":
    main()
