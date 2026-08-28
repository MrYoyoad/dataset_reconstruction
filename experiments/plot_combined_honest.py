"""Honest combined leakage figure — supersedes the overclaimed
leakage_identifiability_plus_reconstruction.png.

LEFT (identifiability, VERIFIED): q_eff|col(J) vs attacker budget ε, binary vs
10-class, N=20 canonical cell (roundB T=1000/S=1280). The reversal: multi-class
leaks FEWER directions at every ε.
RIGHT (reconstruction, HONEST): decoded adapter-only raw-SSIM minus the mean-image
baseline, per gb_e2e cell — every bar negative (0/40 arms clear the trivial floor).

Reads both halves from data (no hardcoding); anchor self-check on the roundB cell.
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
YELLOW = dict(boxstyle="round", facecolor="#fdf6c3", edgecolor="#d9c86a", alpha=0.95)
plt.rcParams.update({"axes.labelsize": 13, "axes.titlesize": 13, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10, "figure.titlesize": 15})
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid")

OUT = "figures/combined/leakage_honest_combined.png"
os.makedirs("figures/combined", exist_ok=True)
EPS = [0.1, 0.3, 1.0, 3.0, 10.0]
CELL = (1280, 0.01)

# ---- LEFT: identifiability (roundB canonical N=20 cell) ----
b = torch.load("results/jacobian_j1_roundB_mnist_nc2_T1000_S1280.pth", map_location="cpu", weights_only=False)
m = torch.load("results/jacobian_j1_roundB_mnist_nc10_T1000_S1280.pth", map_location="cpu", weights_only=False)
bcs, mcs = b["results"][CELL]["colspace"], m["results"][CELL]["colspace"]
rJ = bcs["r_J"]
bq = [bcs["q_eff"][e] for e in EPS]
mq = [mcs["q_eff"][e] for e in EPS]
# anchor self-check (hard abort)
assert bcs["q_eff"][1.0] == 119 and mcs["q_eff"][1.0] == 80 and rJ == 160, \
    f"ANCHOR FAIL: bin@1={bcs['q_eff'][1.0]} (exp 119), 10cls@1={mcs['q_eff'][1.0]} (exp 80), r_J={rJ} (exp 160)"
print(f"[anchor] OK — roundB N=20: bin 119 / 10cls 80 @ε=1, r_J=160, iso {bcs['iso_ratio']:.3f}/{mcs['iso_ratio']:.3f}")

# ---- RIGHT: reconstruction gap-to-baseline (gb_e2e) ----
rows = []
for f in sorted(glob.glob("results/gb_e2e_mnist_N*.pth") + glob.glob("results/gb_e2e_fashion_N*.pth")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    dec = d["results"]["DECODED all-layers"]           # (ssim, ssim_norm, baseline)
    rows.append((f.split("/")[-1].replace("gb_e2e_", "").replace(".pth", ""), dec[0], dec[2]))
labels = [r[0] for r in rows]
gap = np.array([r[1] - r[2] for r in rows])
n_beat = int((gap > 0).sum())

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6.2))
fig.suptitle("LoRA leakage: the geometry leaks (multi-class leaks FEWER directions) — "
             "but adapter-only pixel reconstruction doesn't beat the trivial baseline", fontweight="bold")

# LEFT panel
axL.fill_between(EPS, bq, mq, color=ORANGE, alpha=0.12, label="reversal gap (binary − 10-class)")
axL.semilogx(EPS, bq, "o-", color=BLUE, lw=2.4, ms=9, label="binary (BCE)")
axL.semilogx(EPS, mq, "s-", color=ORANGE, lw=2.4, ms=9, label="10-class (CE)")
axL.axhline(rJ, color="grey", ls="--", lw=1, alpha=0.7)
axL.text(EPS[0], rJ - 6, f"r_J = {rJ} (all directions)", fontsize=9, color="grey", va="top")
axL.set_xlabel("attacker SNR budget ε")
axL.set_ylabel(f"q_eff|col(J) = recoverable directions (of {rJ})")
axL.set_title("A. IDENTIFIABILITY (verified) — 10-class leaks FEWER at every ε")
axL.set_xticks(EPS)
axL.set_xticklabels([str(e) for e in EPS])
axL.legend(loc="upper left")
axL.text(0.03, 0.06, f"multi-class NOT amplified — it leaks fewer directions\n"
         f"(iso {mcs['iso_ratio']:.2f} > {bcs['iso_ratio']:.2f}: CE couples more noise into col(J)).\n"
         "N=20, T=1000, S=1280 (roundB canonical cell).",
         transform=axL.transAxes, fontsize=9.5, bbox=YELLOW, va="bottom")

# RIGHT panel
x = np.arange(len(rows))
axR.bar(x, gap, 0.62, color=[GREEN if g > 0 else RED for g in gap], alpha=0.85)
axR.axhline(0, color="k", lw=1)
for xi, g in zip(x, gap):
    axR.text(xi, g - 0.012, f"{g:+.2f}", ha="center", va="top", fontsize=8, fontweight="bold", color=RED)
axR.set_ylabel("decoded raw SSIM − mean-image baseline")
axR.set_title(f"B. RECONSTRUCTION (honest) — {n_beat}/{len(rows)} cells beat baseline (0/40 both arms)")
axR.set_xticks(x)
axR.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
axR.text(0.03, 0.06, "every bar negative: the adapter-only decoder carries NO\n"
         "instance-specific pixel info beyond the mean image.\n"
         "Geometry leaks; pixels do not (yet) — an OPEN limitation.",
         transform=axR.transAxes, fontsize=9.5, bbox=YELLOW, va="bottom")

fig.text(0.5, 0.005, "identifiability: results/jacobian_j1_roundB_*.pth (job 484948 lineage) | "
         "reconstruction: results/gb_e2e_*.pth + metrics.py baseline gate | audit yoado-a2/aa",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
fig.savefig(OUT, dpi=250, bbox_inches="tight")
print(f"[saved] {OUT}")
print(f"reconstruction: decoded beats baseline {n_beat}/{len(rows)} (all-layers arm)")
