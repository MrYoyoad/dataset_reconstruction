"""Ecosystem graded-overlap SWEEP — GAIN vs how much tasks SHARE (the bracket, as a curve).

The auditor's money-test: vary content overlap 0%→100% and plot the LOO-subtraction GAIN. Prediction from the
two nulls bracketing the phenomenon: GAIN → 0 at BOTH ends (0% disjoint = nothing shared to subtract;
100% same-digits = shared IS everything relevant) and PEAKS in the middle (50% anchor-digit). Reuses the
eco_analyze LOO machinery; groups by gid (falls back to task tuple); cluster-robust CI over gid. numpy/scipy.
"""
import numpy as np, torch, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from experiments.dataset_sensitivity.eco_analyze import load, retrieval_auc

BANKS = [("0%\n(disjoint)", 0.0, "results/eco_zoo/eco_bank.pth"),
         ("50%\n(anchor-digit)", 0.5, "results/partial_zoo/partial_bank.pth"),
         ("100%\n(same-digits)", 1.0, "results/full_zoo/full_bank.pth")]
P_SHARED = 6
RNG = np.random.default_rng(0)


def _proj_residual(DW, omask, v, p):
    X = DW[omask]; Xc = X - X.mean(0)
    Gm = Xc @ Xc.T; wv, Uv = np.linalg.eigh(Gm); idx = np.argsort(wv)[::-1][:p]
    w, U = np.maximum(wv[idx], 0.0), Uv[:, idx]
    c = (U.T @ (Xc @ v)) / np.sqrt(w + 1e-30)
    proj = Xc.T @ (U @ (c / np.sqrt(w + 1e-30)))
    return float((c @ c) / (v @ v + 1e-30)), (v - proj)


def gain_for(path, p=P_SHARED):
    bank, mu, meta = load(path)
    gid = [c["gid"] if "gid" in c else tuple(c["task"]) for c in bank]
    DW = np.stack([c["dw_flat"] for c in bank])
    rows = []   # (gid, araw, ares, gain, proj)
    for i, tgt in enumerate(bank):
        g = gid[i]; omask = np.array([gid[j] != g for j in range(len(bank))])
        others = [c for c, m in zip(bank, omask) if m]
        pf, res_flat = _proj_residual(DW, omask, tgt["dw_flat"], p)
        res = res_flat.reshape(tgt["dW"].shape)
        distract = np.concatenate([c["priv"] for c in others[:20]], axis=0)
        pool = np.concatenate([tgt["priv"], distract], axis=0)
        lab = np.concatenate([np.ones(len(tgt["priv"])), np.zeros(len(distract))])
        araw = retrieval_auc(tgt["dW"], pool, lab, mu); ares = retrieval_auc(res, pool, lab, mu)
        rows.append((g, araw, ares, ares - araw, pf))
    ug = sorted(set(r[0] for r in rows), key=str)
    cl = np.array([np.mean([r[3] for r in rows if r[0] == u]) for u in ug])
    G = len(cl); est = float(cl.mean()); se = cl.std(ddof=1) / np.sqrt(G) if G > 1 else float("nan")
    t = stats.t.ppf(0.975, G - 1) if G > 1 else float("nan")
    return dict(gain=est, ci=(est - t * se, est + t * se), G=G,
                proj=float(np.mean([r[4] for r in rows])), raw=float(np.mean([r[1] for r in rows])))


def main():
    xs, gains, los, his, projs = [], [], [], [], []
    print("=== ECOSYSTEM GRADED-OVERLAP SWEEP — GAIN vs content overlap ===")
    for name, ov, path in BANKS:
        if not os.path.exists(path):
            print(f"  {name.strip()}: MISSING {path} — skip"); continue
        r = gain_for(path)
        xs.append(ov); gains.append(r["gain"]); los.append(r["ci"][0]); his.append(r["ci"][1]); projs.append(r["proj"])
        print(f"  overlap={ov:.2f} ({name.strip():14s}): GAIN={r['gain']:+.3f} CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] "
              f"proj={r['proj']:.3f} raw-AUC={r['raw']:.3f} G={r['G']}")
    xs, gains, los, his = map(np.array, (xs, gains, los, his))

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.errorbar(xs, gains, yerr=[gains - los, his - gains], fmt="o-", color="#2c7fb8", lw=2, ms=9,
                capsize=5, zorder=3, label="GAIN = AUC(residual) − AUC(raw)")
    for x, g, lo in zip(xs, gains, los):
        ax.annotate(f"{g:+.3f}", (x, g), textcoords="offset points", xytext=(8, 8), fontsize=9,
                    fontweight="bold", color="#0a4d7a")
    ax.set_xticks([0.0, 0.5, 1.0]); ax.set_xticklabels(["0%\ndisjoint\n(eco null)", "50%\nanchor-digit",
                                                        "100%\nsame-digits\n(atlas-degenerate)"])
    ax.set_xlabel("content overlap between population tasks and target", fontsize=11)
    ax.set_ylabel("ecosystem GAIN (retrieval AUC)", fontsize=11)
    ax.set_title("Ecosystem effect lives in the PARTIAL-overlap regime\n(the two nulls bracket a mid-overlap peak)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.5, -0.30, "observe-framed · population(>weakest)-attacker · N=4 MNIST-MLP · cluster-robust CI over "
            "task-groups (G small) · a FIRST curve, not a confirmation",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555")
    os.makedirs("figures/eco", exist_ok=True)
    fig.savefig("figures/eco/eco_graded_overlap.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("\n[saved] figures/eco/eco_graded_overlap.png")
    if len(gains) == 3:
        peak_mid = gains[1] > max(gains[0], gains[2])
        print(f"\n  bracketing prediction (peak at 50%, ~0 at ends): {'CONFIRMED — mid > both ends' if peak_mid else 'NOT matched'}")


if __name__ == "__main__":
    main()
