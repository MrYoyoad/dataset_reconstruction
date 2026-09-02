#!/usr/bin/env python3
"""Aggregate results/exact_inversion/*.jsonl into markdown tables + figures.

  python experiments/exact_inversion/analyze_exact_inversion.py
Figures -> figures/exact_inversion/{basin_curve.png, phase_diagram_exact.png}; tables printed to stdout
(paste into RESULTS.md).  CPU-only (numpy/matplotlib)."""
import json, glob, os, sys
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "exact_inversion"); FIG = os.path.join(ROOT, "figures", "exact_inversion")
os.makedirs(FIG, exist_ok=True)

def load(name):
    """name may be a basename or a glob stem; all matching *.jsonl are concatenated."""
    out = []
    stem = name[:-6] if name.endswith(".jsonl") else name
    for p in sorted(glob.glob(os.path.join(RES, stem + "*.jsonl"))):
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out

def fmt(x):
    return f"{x:.1e}" if isinstance(x, float) and (abs(x) < 1e-2 or abs(x) >= 1e3) and x != 0 else (f"{x:.3g}" if isinstance(x, float) else str(x))

def table(rows, cols, title):
    if not rows: print(f"\n(no rows for {title})"); return
    print(f"\n### {title}\n")
    print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
    for r in rows: print("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")

# ---- step 1: validation table ----
s1 = load("step1_validation.jsonl")
table(s1, ["k", "N", "r_minus_N", "T", "lr", "seed", "rankC", "eps_inv", "fwd_check", "fwd_check_A", "deformation", "init_noise",
           "start_err_median", "final_err_median", "final_err_max", "residual", "verdict", "sec_per_iter", "seconds"], "Step 1 — validation (GPU)")
table(load("step1_timing_cpu.jsonl"), ["k", "N", "T", "device", "sec_per_iter", "final_err_median", "residual"], "Step 1 — CPU timing")

# ---- step 2: basin ----
near = load("step2_basin_near.jsonl"); init = load("step2_basin_init.jsonl")
if near:
    by = defaultdict(list)
    for r in near: by[r["init_noise"]].append(r)
    rows = []
    for nz in sorted(by):
        rs = by[nz]
        rows.append(dict(init_noise=nz, n_seeds=len(rs), start_err_median=float(np.median([r["start_err_median"] for r in rs])),
                         frac_recovered=float(np.mean([r["frac_recovered"] for r in rs])),
                         seeds_all_recovered=sum(r["frac_recovered"] == 1.0 for r in rs),
                         residual_median=float(np.median([r["residual"] for r in rs])),
                         n_alias=sum("alias" in r["verdict"] for r in rs), n_optfail=sum("optim" in r["verdict"] for r in rs),
                         restarts_used_mean=float(np.mean([r.get("restarts_used", 1) for r in rs]))))
    table(rows, list(rows[0].keys()), "Step 2 — basin study, init=near (k=12, N=8, T=1500, lr=.03), 5 seeds x up to 8 restarts")
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    xs = [r["start_err_median"] for r in rows]; ys = [r["frac_recovered"] for r in rows]
    ax.plot(xs, ys, "o-", color="#1f4e79")
    for r in rows: ax.annotate(f"noise {r['init_noise']}", (r["start_err_median"], r["frac_recovered"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("median start error (relative image error)"); ax.set_ylabel("fraction of images recovered (err < 1e-2)")
    ax.set_ylim(-0.05, 1.05); ax.set_title("Exact inversion: basin of attraction (k=12, N=8, T=1500, lr=.03)", fontsize=9)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "basin_curve.png")); print("saved basin_curve.png")
if init:
    by = defaultdict(list)
    for r in init: by[r["init"]].append(r)
    rows = []
    for nm in by:
        rs = by[nm]
        rows.append(dict(init=nm, n_seeds=len(rs), start_err_median=float(np.median([r["start_err_median"] for r in rs])),
                         frac_recovered=float(np.mean([r["frac_recovered"] for r in rs])),
                         frac_recovered_any=float(np.mean([r["frac_recovered_any"] for r in rs])),
                         residual_median=float(np.median([r["residual"] for r in rs])),
                         n_alias=sum("alias" in r["verdict"] for r in rs), n_optfail=sum("optim" in r["verdict"] for r in rs),
                         span_angle_mean_deg=float(np.mean([r["span_angle_mean_deg"] for r in rs])),
                         span_out_frac=float(np.mean([r.get("span_out_frac", np.nan) for r in rs])),
                         cert_anchor_resid=float(np.mean([r.get("cert_anchor_resid", np.nan) for r in rs]))))
    table(rows, list(rows[0].keys()), "Step 2 — attacker-available initialisers (same cell), 5 seeds x up to 8 restarts")

# ---- step 3: adam ----
s3 = load("step3_adam.jsonl")
if s3:
    by = defaultdict(list)
    for r in s3: by[r["init_noise"]].append(r)
    rows = [dict(init_noise=nz, n_seeds=len(rs), eps_inv=float(np.median([r["eps_inv"] for r in rs])), fwd_check=float(max(r["fwd_check"] for r in rs)),
                 deformation=float(np.median([r["deformation"] for r in rs])), start_err_median=float(np.median([r["start_err_median"] for r in rs])),
                 frac_recovered=float(np.mean([r["frac_recovered"] for r in rs])), final_err_median=float(np.median([r["final_err_median"] for r in rs])),
                 residual_median=float(np.median([r["residual"] for r in rs])),
                 n_alias=sum("alias" in r["verdict"] for r in rs), n_optfail=sum("optim" in r["verdict"] for r in rs)) for nz, rs in sorted(by.items())]
    table(rows, list(rows[0].keys()), "Step 3 — Adam release (k=12, N=8, T=800, lr=.003), unknowns = latents + full A0")

# ---- step 4: phase diagram (merged with the step-5 rescue pass, which re-runs the failed cells
#      with 4 restarts that re-seed the whole unknown vector) ----
s4 = load("step4_sweep.jsonl"); s5 = load("step5_rescue")
rescued = {(d["k"], d["N"]) for d in s5 if d["frac_recovered"] == 1.0}
by_cell = {(d["k"], d["N"]): d for d in s4}
for d in s5:
    if d["frac_recovered"] >= by_cell.get((d["k"], d["N"]), d)["frac_recovered"]:
        by_cell[(d["k"], d["N"])] = d
s4 = list(by_cell.values()) if s4 else []
if s4:
    Ns = sorted({r["N"] for r in s4}); ks = sorted({r["k"] for r in s4}); r0 = s4[0]["r"]
    M = np.full((len(ks), len(Ns)), np.nan); V = np.full_like(M, np.nan)
    for r in s4: M[ks.index(r["k"]), Ns.index(r["N"])] = r["frac_recovered"]; V[ks.index(r["k"]), Ns.index(r["N"])] = r["residual"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    im = ax.imshow(M, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(Ns))); ax.set_xticklabels(Ns); ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
    ax.set_xlabel("N  (private examples; certificate has r − N rows)"); ax.set_ylabel("k  (manifold dimension)")
    Nl = np.linspace(min(Ns), max(Ns), 100); kl = r0 - Nl
    ax.plot((Nl - min(Ns)) / 2, (kl - min(ks)) / 2, color="#333", lw=1.5, ls="--"); ax.text(4.55, 2.75, "k = r − N", color="#333", fontsize=9, rotation=-40)
    for a in range(len(ks)):
        for b in range(len(Ns)):
            if not np.isnan(M[a, b]):
                mark = "*" if (ks[a], Ns[b]) in rescued else ""
                ax.text(b, a, f"{M[a, b]:.2f}{mark}", ha="center", va="center", fontsize=7.5, color="white" if M[a, b] > 0.55 else "#222")
    ax.set_title(f"Exact inversion (backprop through the recipe), start 10% off   (r = {r0}, SGD T=400, FP64)\n"
                 f"* = needed 4 restarts; the certificate alone recovers NOTHING above the dashed line", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.045); cb.set_label("fraction of images recovered (err < 1e-2)")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "phase_diagram_exact.png")); print("saved phase_diagram_exact.png")
    print("\n### Step 4 — frac_recovered over (k rows, N cols)\n"); print("| k \\ N | " + " | ".join(map(str, Ns)) + " |"); print("|---|" + "---|" * len(Ns))
    for a, k in enumerate(ks): print(f"| {k} | " + " | ".join("" if np.isnan(x) else f"{x:.2f}" for x in M[a]) + " |")
    print("\nresidual (median over cells): %.1e; cells with residual>1e-16: %d / %d" % (np.nanmedian(V), int(np.nansum(V > 1e-16)), int(np.sum(~np.isnan(V)))))
    print(f"\ncells rescued by restarts (failed at restarts=1, recovered at restarts=4): {len(rescued)}")
    table([r for r in s4 if r["frac_recovered"] < 1], ["k", "N", "r_minus_N", "frac_recovered", "final_err_median", "residual", "verdict"], "Step 4 — cells not fully recovered")

# ---- the comparison figure: what the certificate alone can do vs what the recipe-simulating inversion does ----
# Left panel is TRANSCRIBED from results_rev9.pdf Figure 1 (the bundle's own finite-difference run, provisional,
# produced outside this repo).  Right panel is this repo's measurement.  Same axes, same r = 16.
CERT_ONLY = {  # [k][N] with N = 2,4,...,14
    14: [0.83, 0, 0, 0, 0, 0, 0], 12: [1.00, 0.67, 0, 0, 0, 0, 0], 10: [1.00, 1.00, 0.94, 0, 0, 0, 0],
    8: [1.00, 1.00, 0.94, 0.88, 0, 0, 0], 6: [1.00, 1.00, 1.00, 1.00, 0.67, 0, 0],
    4: [1.00, 0.92, 0.94, 1.00, 0.97, 0.67, 0], 2: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.81]}

if s4:
    Ns_c = [2, 4, 6, 8, 10, 12, 14]; ks_c = [2, 4, 6, 8, 10, 12, 14]
    Mc = np.array([CERT_ONLY[k] for k in ks_c], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), dpi=150, sharey=True)
    for ax, (Mx, title, sub) in zip(axes, [
            (Mc, "Certificate $C$ alone", "$CH=0$ has $r-N$ rows; blind above the line\n(transcribed from the bundle's finite-difference run)"),
            (M, "Exact inversion: simulate the recipe", "unknowns = latents and $X=A_0U$; $*$ = needed 4 restarts\n(this repo, backprop through the unrolled training)")]):
        im = ax.imshow(Mx, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(Ns_c))); ax.set_xticklabels(Ns_c); ax.set_yticks(range(len(ks_c))); ax.set_yticklabels(ks_c)
        ax.set_xlabel("N   (private examples)")
        Nl = np.linspace(2, 14, 100); ax.plot((Nl - 2) / 2, (16 - Nl - 2) / 2, color="#111", lw=1.6, ls="--")
        ax.set_title(title, fontsize=11, pad=26); ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, color="#444")
        for a in range(len(ks_c)):
            for b in range(len(Ns_c)):
                v = Mx[a, b]
                if np.isnan(v): continue
                mark = "*" if (Mx is M and (ks_c[a], Ns_c[b]) in rescued) else ""
                ax.text(b, a, f"{v:.2f}{mark}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.55 else "#222")
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("k   (manifold dimension)")
    axes[0].text(4.35, 2.35, "k = r − N", fontsize=8.5, rotation=-38, color="#111")
    cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02); cb.set_label("fraction of runs / images recovered")
    fig.suptitle("The $r-N$ budget bounds one channel, not the leakage   (r = 16, plain SGD, FP64)", fontsize=11.5, y=1.10)
    fig.savefig(os.path.join(FIG, "phase_diagram_comparison.png"), bbox_inches="tight"); print("saved phase_diagram_comparison.png")
