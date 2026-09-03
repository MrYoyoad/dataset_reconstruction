#!/usr/bin/env python3
"""Figure for the certificate ladder: (left) per-image basin vs distance below the certificate line, one point
per on-chart cell, coloured by rank; (right) at r = 64, per-image basin vs k beside the chart's instance
identification curve (Step 22), so budget, reachability and fidelity sit on one axis.

  python experiments/exact_inversion/plot_certificate_ladder.py   (CPU; reads results/exact_inversion/step69_*, step76_*, step67_*, step72_*)
"""
import glob, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cells = []
for f in glob.glob("results/exact_inversion/step69_cert_rank_*.jsonl") + glob.glob("results/exact_inversion/step76_r64k32_*.jsonl") + glob.glob("results/exact_inversion/step67_cert_below_*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if d.get("part") != "B" or d.get("setting") != "on" or d.get("set") != "confident": continue
        cells.append(dict(r=d["r"], k=d["k"], Np=d["n_prime"], line=d["cert_line"], dist=d["cert_line"] - d["k"],
                          agg=d["frac_starts_on_a_recorded_image"], per=d["frac_starts_on_a_recorded_image"] / d["n_prime"], starts=d.get("starts_run", d["random_starts"])))
seen = {}
for c in cells: seen[(c["r"], c["k"])] = c            # last write wins (dedicated 728592 row for r=64 k=32 is equivalent)
cells = sorted(seen.values(), key=lambda c: (c["r"], c["k"]))
fid = {}
for f in glob.glob("results/exact_inversion/step72_fidelity_*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if "instance_identification" in d: fid[d["k"]] = (d["proj_acc"]["strong"], d["instance_identification"]["raw_pool"]["self_top1"])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
colors = {16: "tab:red", 32: "tab:orange", 64: "tab:blue"}
for c in cells:
    ax[0].scatter(c["dist"], 100 * c["per"], color=colors[c["r"]], s=40, zorder=3)
    ax[0].annotate(f"k={c['k']}", (c["dist"], 100 * c["per"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
for r, col in colors.items(): ax[0].scatter([], [], color=col, label=f"r = {r}")
ax[0].set_xlabel("distance below the certificate line, r − N′ − k"); ax[0].set_ylabel("basin per recorded image (% of random starts)")
ax[0].set_title("Reachability follows distance below the line", fontsize=9); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
r64 = [c for c in cells if c["r"] == 64]
ax[1].plot([c["k"] for c in r64], [100 * c["per"] for c in r64], "o-", color="tab:blue", label="per-image basin, r = 64 (%)")
ks = sorted(fid); ax2 = ax[1].twinx()
ax2.plot(ks, [100 * fid[k][1] for k in ks], "s--", color="tab:green", ms=4, label="instance identification (%)")
ax2.plot(ks, [100 * fid[k][0] for k in ks], "^:", color="tab:gray", ms=4, label="class survival (%)")
ax[1].set_xlabel("chart dimension k"); ax[1].set_ylabel("basin per recorded image (%)"); ax2.set_ylabel("held-out fidelity (%)")
ax[1].axvline(56, color="k", lw=.8, ls=":"); ax[1].annotate("line r − N′ = 56", (56, 5), fontsize=6, rotation=90, xytext=(-8, 0), textcoords="offset points")
ax[1].set_title("r = 64: reachability and fidelity vs k", fontsize=9); ax[1].grid(alpha=.3)
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
plt.tight_layout(); out = "figures/exact_inversion/certificate_ladder.png"; plt.savefig(out, dpi=130); print(out, [(c["r"], c["k"], round(100 * c["per"], 1)) for c in cells])
