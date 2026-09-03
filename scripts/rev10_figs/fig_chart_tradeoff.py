"""Expressiveness vs invertibility at a fixed budget k=16, on the trained backbone.
BOTH axes are computed at the TRUTH (chart representation error is a property of the chart; sigma_min is the
Jacobian at the true coordinates), so neither depends on where the solver stopped. CPU-only, committed jsonl."""
import json, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = {}
for f in glob.glob("results/exact_inversion/step4[456]_*.jsonl"):
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            if r.get("cell") == "a": rows.setdefault(r["chart"], r)
LABEL = {"pca":"global PCA","global":"global PCA (repeat job)","local":"label-local PCA",
         "vae":"VAE decoder (GELU)","vae_relu":"VAE decoder (ReLU)"}
COL   = {"pca":"#1f77b4","global":"#1f77b4","local":"#2ca02c","vae":"#ff7f0e","vae_relu":"#d62728"}
fig, ax = plt.subplots(figsize=(10,6.4), dpi=200)
for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["chart_repr_err"]):
    x, y = r["chart_repr_err"], r["jac_sigma_min_truth"]
    ax.scatter(x, y, s=190, color=COL[c], edgecolor="k", lw=0.8, zorder=5,
               marker=("o" if c != "global" else "D"))
    ax.annotate(f"{LABEL[c]}\nattack returned {r['err_vs_REAL_max']:.2f}", (x, y),
                textcoords="offset points", xytext=(12, -4 if c!="vae_relu" else 10), fontsize=10)
xs = [r["chart_repr_err"] for c,r in rows.items() if c != "global"]
ys = [r["jac_sigma_min_truth"] for c,r in rows.items() if c != "global"]
o  = sorted(zip(xs,ys))
ax.plot([p[0] for p in o], [p[1] for p in o], color="gray", lw=1.2, ls="--", zorder=2)
ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xlabel("← richer chart      chart's own representation error of the private digits", fontsize=12.5)
ax.set_ylabel("σ_min of Dρ at the truth   (log)\n← harder to invert", fontsize=12.5)
ax.set_title("Expressiveness versus invertibility, at one fixed budget (k = 16, below the line 18)\n"
             "every chart that draws better is harder to invert — both axes measured at the truth, not by the solver",
             fontsize=12.5, fontweight="bold")
ax.grid(alpha=0.3); ax.tick_params(labelsize=11)
ax.text(0.02, 0.04, "trained backbone (78.5%), N = 8 unseen test digits, r = 16, one seed\n"
                    "'attack returned' = error vs the real digit, SOLVER-LIMITED (200 iterations, none at the floor)",
        transform=ax.transAxes, fontsize=9.5, color="#444")
out = os.path.join("figures/rev10", "fig_chart_tradeoff.png")
fig.tight_layout(); fig.savefig(out); print("saved", out, "|", len(rows), "charts")
