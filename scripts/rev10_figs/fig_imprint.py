"""The mechanism: the release is a sum of per-example imprints, each set by that example's own accumulated error.
Left: ||C_i|| against the accumulated softmax residual, all 320 measurements, coloured by encoder — one line.
Right: ||C_i|| against the base model's margin, showing the exponential fall-off and where the release goes empty.
CPU-only from the committed jsonl (job 631392). style_guide/plots.md: DPI 200, matplotlib defaults, T5 fonts."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np

rows = [json.loads(l) for l in open("results/exact_inversion/step58_imprint_631392.jsonl") if l.strip()]
COL = {"random": "#1f77b4", "weak": "#2ca02c", "mid": "#ff7f0e", "strong": "#d62728"}
ACC = {"random": "8.7%", "weak": "78.5%", "mid": "95.1%", "strong": "97.9%"}
FLOOR = 1e-43   # below the smallest measured value, so nothing is clipped
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=200)

for e in ("random", "weak", "mid", "strong"):
    rs = [r for r in rows if r["encoder"] == e]
    x = np.array([max(r["res_traj_sum"], FLOOR) for r in rs])
    y = np.array([max(r["imprint_norm"], FLOOR) for r in rs])
    ax1.scatter(x, y, s=26, color=COL[e], alpha=0.75, edgecolor="none", label=f"{e} ({ACC[e]})")
    m = np.array([r["margin_W0"] for r in rs])
    ax2.scatter(m, y, s=26, color=COL[e], alpha=0.75, edgecolor="none", label=f"{e} ({ACC[e]})")

lo = min(max(r["res_traj_sum"], FLOOR) for r in rows); hi = max(r["res_traj_sum"] for r in rows)
gx = np.array([lo, hi])
ratio = np.median([r["imprint_norm"] / r["res_traj_sum"] for r in rows if r["res_traj_sum"] > 0])
ax1.plot(gx, ratio * gx, color="k", lw=1.2, ls="--", zorder=1,
         label=f"proportional (median ratio {ratio:.2g})")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel(r"accumulated softmax error of example $i$,   $\sum_t \|D_t[:,i]\|$", fontsize=12.5)
ax1.set_ylabel(r"imprint $\|C_i\|$ in the released adapter", fontsize=12.5)
ax1.set_title("Each example is recorded at the scale of its own error\n"
              "320 measurements, 40 batches, 4 encoders — Kendall 971/1120 pairs", fontsize=12.5, fontweight="bold")
ax1.grid(alpha=0.3); ax1.legend(fontsize=9.5, loc="lower right"); ax1.tick_params(labelsize=10.5)

ax2.set_yscale("log")
ax2.axhspan(FLOOR, 1e-23, color="gray", alpha=0.15)
ax2.text(2, 1e-33, "imprint below $10^{-23}$:\nthe example is not recorded", fontsize=10, color="#555")
ax2.set_xlabel(r"margin of example $i$ under the base model at $W_0$", fontsize=12.5)
ax2.set_ylabel(r"imprint $\|C_i\|$", fontsize=12.5)
ax2.set_title("What leaks is what the model had to learn\n"
              "an example fitted with margin $M$ enters at scale $e^{-M}$", fontsize=12.5, fontweight="bold")
ax2.grid(alpha=0.3); ax2.legend(fontsize=9.5, loc="upper right"); ax2.tick_params(labelsize=10.5)

out = os.path.join("figures/rev10", "fig_imprint.png")
fig.tight_layout(); fig.savefig(out)
print("saved", out, "|", len(rows), "measurements")
