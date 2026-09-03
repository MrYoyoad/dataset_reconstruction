"""sigma_min(D-rho) at the truth vs k: random encoder against a TRAINED backbone, same line k<18.
The line does not move; the conditioning to reach it worsens ~1000x by k=17. CPU-only, reads committed jsonl.
style_guide/plots.md: DPI 200, matplotlib defaults, T5 fonts."""
import json, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
def dedupe(pat, key):
    seen = {}
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if l.strip():
                r = json.loads(l); seen[tuple(r.get(x) for x in key)] = r
    return seen
sm = lambda r: (r.get("jac_sigma_min_truth") or r.get("jac_sigma_min"))
tr  = dedupe("results/exact_inversion/step41_trained_*.jsonl", ("k","seed"))
rer = dedupe("results/exact_inversion/step43_*rerun*.jsonl", ("k","seed"))
mn  = dedupe("results/exact_inversion/step31_mnist_*.jsonl", ("r","k","seed"))
rand = {k: sm(r) for (rr,k,s),r in mn.items() if rr == 16}
trn  = {k: sm(r) for (k,s),r in tr.items()}
floor_std = {k: (r.get("residual",9) < 1e-25) for (k,s),r in tr.items()}          # reached floor at 80 iters
floor_big = {k: (r.get("residual",9) < 1e-25) for (k,s),r in rer.items()}         # reached floor at 300 iters
LINE = 18
blue, orange, green, red = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"
fig, ax = plt.subplots(figsize=(10,6.6), dpi=200)
kr = sorted(rand); kt = sorted(trn)
ax.semilogy(kr, [rand[k] for k in kr], "o-", color=blue,   lw=2, ms=8, label="random frozen encoder (MNIST cells, r=16)")
ax.semilogy(kt, [trn[k]  for k in kt], "s-", color=orange, lw=2, ms=8, label="TRAINED backbone, unseen test digits (r=16)")
for k in kt:                                                                      # solvability markers
    if floor_std.get(k):      ax.plot(k, trn[k], "*", color=green, ms=17, zorder=5)
    elif floor_big.get(k):    ax.plot(k, trn[k], "*", color="#8c564b", ms=17, zorder=5)
ax.plot([], [], "*", color=green,     ms=13, ls="none", label="reached the residual floor, 80 iterations")
ax.plot([], [], "*", color="#8c564b", ms=13, ls="none", label="reached the floor only with 300 iterations")
ax.axvline(LINE, color=red, ls="--", lw=2)
ax.text(LINE-0.25, 3e-4, "capacity line  k = m + r − N = 18", rotation=90, va="top", ha="right", color=red, fontsize=11, fontweight="bold")
ax.axhspan(1e-21, 1e-16, color=red, alpha=0.07)
ax.text(6.3, 2e-19, "collapsed (rank-deficient at the truth)", color=red, fontsize=10.5)
for k in (6,10,14,17):                                                            # the ladder
    if k in rand and k in trn:
        ax.annotate("", xy=(k, trn[k]), xytext=(k, rand[k]), arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
        ax.text(k+0.22, np.sqrt(rand[k]*trn[k]), f"{rand[k]/trn[k]:.0f}×", fontsize=10.5, va="center")
ax.set_xlabel("k  (dimension of the attacker's search chart, per image)", fontsize=13)
ax.set_ylabel("σ_min of Dρ at the truth   (log)", fontsize=13)
ax.set_title("A trained encoder does not move the capacity line — it makes the line harder to reach\n"
             "same law, same collapse at k = 18; conditioning worse by 7× at k = 6 and 1100× at k = 17",
             fontsize=13, fontweight="bold")
ax.set_xlim(4, 27); ax.set_ylim(1e-21, 3e-3); ax.grid(alpha=0.3); ax.tick_params(labelsize=11)
ax.legend(fontsize=10, loc="lower left", framealpha=0.95)
out = os.path.join("figures/rev10", "fig_trained_conditioning.png")
fig.tight_layout(); fig.savefig(out); print("saved", out, "| random", len(kr), "trained", len(kt))
