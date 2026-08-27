"""Honest reconstruction figure: does the adapter-only decoder beat the trivial
mean-image baseline? (Corrects the overclaimed panel in the combined figure.)

Reads results/gb_e2e_*.pth (each arm = (ssim, ssim_norm, ssim_mean_baseline)).
The like-for-like comparison is decoded RAW ssim vs the RAW mean-image baseline.
Result: 0/40 decoded arms beat baseline (audit yoado-a2 + yoado-aa, 2026-08-26).
bsub-only.
"""
import glob
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN, RED = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
plt.rcParams.update({"axes.labelsize": 13, "axes.titlesize": 13, "xtick.labelsize": 10,
                     "ytick.labelsize": 11, "legend.fontsize": 10, "figure.titlesize": 15})
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

OUT = "figures/combined/reconstruction_vs_baseline.png"
os.makedirs("figures/combined", exist_ok=True)

rows = []   # (label, decoded_raw, oracle_raw, baseline)
for f in sorted(glob.glob("results/gb_e2e_mnist_N*.pth") + glob.glob("results/gb_e2e_fashion_N*.pth")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    r = d["results"]
    dec = r["DECODED all-layers"]      # (ssim, ssim_norm, baseline)
    tru = r["TRUE ΔW (ceiling)"]
    name = f.split("/")[-1].replace("gb_e2e_", "").replace(".pth", "")
    rows.append((name, dec[0], tru[0], dec[2]))

labels = [r[0] for r in rows]
dec = np.array([r[1] for r in rows])
ora = np.array([r[2] for r in rows])
base = np.array([r[3] for r in rows])
n = len(rows)
x = np.arange(n)

n_dec_beat = int((dec > base).sum())
n_ora_beat = int((ora > base).sum())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1.15, 1])
fig.suptitle("Adapter-only reconstruction does NOT beat the trivial mean-image baseline "
             f"({n_dec_beat}/{n} cells; 0/40 counting both decoded arms)", fontweight="bold")

# -- Panel 1: raw ssim, decoded vs oracle vs baseline --
w = 0.38
ax1.bar(x - w / 2, dec, w, color=ORANGE, label="DECODED (adapter-only) — the attack")
ax1.bar(x + w / 2, ora, w, color=BLUE, alpha=0.55, label="TRUE ΔW (oracle upper bound)")
ax1.plot(x, base, "D", color=RED, ms=9, label="mean-image baseline (RAW ssim floor)", zorder=5)
for xi, b in zip(x, base):
    ax1.hlines(b, xi - 0.46, xi + 0.46, color=RED, lw=1.6, zorder=4)
ax1.set_ylabel("RAW SSIM (like-for-like with baseline)")
ax1.set_title("A. Decoded raw ssim is below the mean-image floor in every cell; "
              "even the oracle fails on the small-net fashion cells")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=40, ha="right")
ax1.legend(loc="upper right")
ax1.text(0.012, 0.94, "a bar BELOW its red floor = carries NO instance-specific info\n"
         "(metrics.py: a result ≤ mean-image baseline learned nothing)",
         transform=ax1.transAxes, fontsize=9.5, va="top",
         bbox=dict(boxstyle="round", facecolor="#fdf6c3", edgecolor="#d9c86a"))

# -- Panel 2: gap to baseline (decoded - baseline), diverging --
gap = dec - base
colors = [GREEN if g > 0 else RED for g in gap]
ax2.bar(x, gap, 0.6, color=colors, alpha=0.85)
ax2.axhline(0, color="k", lw=1)
for xi, g in zip(x, gap):
    ax2.text(xi, g - 0.012 if g < 0 else g + 0.006, f"{g:+.2f}", ha="center",
             va="top" if g < 0 else "bottom", fontsize=8.5, fontweight="bold", color=RED if g < 0 else GREEN)
ax2.set_ylabel("decoded raw ssim − baseline")
ax2.set_title("B. Gap to baseline — every cell negative (decoder never clears the floor)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=40, ha="right")
ax2.text(0.012, 0.08, "The circulated 'ssim_norm ≈ 0.57–0.61' was the mean/std-MATCHED score "
         "(inflates raw by ~0.1–0.3),\nnever compared to a baseline. The honest raw-vs-raw comparison: "
         "0 of 40 decoded arms clear the floor.\nGeometry/identifiability is the result; adapter-only "
         "pixel reconstruction is an OPEN LIMITATION.",
         transform=ax2.transAxes, fontsize=9.5, va="bottom",
         bbox=dict(boxstyle="round", facecolor="#fdf6c3", edgecolor="#d9c86a"))

fig.text(0.5, 0.005, "data: results/gb_e2e_*.pth | metric gate: experiments/metrics.py (ssim_mean_baseline) "
         "| audit yoado-a2 + yoado-aa 2026-08-26", ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print(f"[saved] {OUT}")
print(f"decoded beats baseline: {n_dec_beat}/{n} (all-layers arm) | oracle beats: {n_ora_beat}/{n}")
for name, dv, ov, bv in rows:
    print(f"  {name:26s} decoded_raw={dv:.3f}  oracle_raw={ov:.3f}  baseline={bv:.3f}  "
          f"{'BEATS' if dv > bv else 'below'}")
