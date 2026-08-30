"""A — RESOLUTION LIMIT: matching accuracy vs swap-level k (the resolution curve).

For each k_pc (images swapped/class), leave-one-INIT-out kNN matching of set_id on the ΔW subspace — FULL
distance AND the Grassmann-ONLY norm-control. Accuracy-above-chance (chance=1/M) + per-k permutation null.
The finest cut k_pc=1 = "sets differ by ONE image per class" — its accuracy is the resolution of the
fingerprint. DETECTION not reconstruction; observe-framed, weakest-attacker.
"""
import numpy as np, os
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from experiments.dataset_sensitivity.atlas_analyze import _load, dw_distance, _knn_dist
from experiments.dataset_sensitivity.instance_recovery import grass_only_distance

BANK = "results/a_resolution_zoo/a_bank.pth"
RNG = np.random.default_rng(0)


def _loio(D, sid, init, uinit, k=3):
    per = []
    for ho in uinit:
        te = np.where(init == ho)[0]; tr = np.where(init != ho)[0]
        per.append(float((_knn_dist(D[np.ix_(te, tr)], sid[tr], k) == sid[te]).mean()))
    return np.array(per)


def main():
    bank, meta = _load(BANK)
    KPC = meta["kpc"]; M = meta["M"]; chance = 1.0 / M
    print(f"[A] M={M} sets chance={chance:.2f} | k_pc levels={KPC} | Npc-swap = 'differ by k_pc images/class'")
    curves = {"full": [], "grass": []}; cis = {"full": [], "grass": []}
    for k in KPC:
        sub = [c for c in bank if c["k_pc"] == k]
        sid = np.array([c["set_id"] for c in sub]); init = np.array([c["init"] for c in sub])
        uinit = sorted(set(init))
        for name, distfn in [("full", dw_distance), ("grass", grass_only_distance)]:
            D = distfn(sub)
            acc = _loio(D, sid, init, uinit); est = acc.mean()
            se = acc.std(ddof=1) / np.sqrt(len(acc)); t = stats.t.ppf(0.975, len(acc) - 1)
            ci = (est - t * se, est + t * se)
            null = np.array([_loio(D, RNG.permutation(sid), init, uinit).mean() for _ in range(300)])
            pval = float((null >= est).mean())
            curves[name].append(est); cis[name].append(ci)
            if name == "full":
                print(f"  k_pc={k} (differ by {k}/class): acc={est:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] "
                      f"above-chance={est-chance:+.3f} p={pval:.3f}{'*' if pval<0.05 else ''}  "
                      f"[grass-only={curves['grass'][-1]:.3f}]")

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    ax.axhline(chance, color="#d95f0e", ls=":", lw=1.4, label=f"chance = {chance:.2f}")
    for name, col, lbl in [("full", "#2c7fb8", "ΔW full distance"), ("grass", "#7bccc4", "Grassmann-only (norm-control)")]:
        m = np.array(curves[name]); lo = np.array([c[0] for c in cis[name]]); hi = np.array([c[1] for c in cis[name]])
        ax.errorbar(KPC, m, yerr=[m - lo, hi - m], fmt="o-", color=col, lw=2, ms=8, capsize=4, label=lbl)
    ax.set_xticks(KPC); ax.set_xlabel("swap level k  (candidate sets differ by k images PER CLASS →)", fontsize=11)
    ax.set_ylabel("set-matching accuracy", fontsize=11); ax.set_ylim(0, 1.05)
    ax.invert_xaxis()  # hardest (most similar, k=1) on the RIGHT→left = easier
    ax.set_title("Resolution limit of the instance fingerprint\n"
                 "how similar can two training sets be and still be told apart from ΔW?",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.text(0.5, -0.32, "DETECTION not reconstruction · observe-framed · weakest-attacker · N=8 MNIST-MLP · "
            "k=1 = 'differ by one image/class' (the finest cut) · LOIO, permutation null",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555")
    os.makedirs("figures/harder_id", exist_ok=True)
    fig.savefig("figures/harder_id/a_resolution.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("\n[saved] figures/harder_id/a_resolution.png")
    k1 = curves["full"][KPC.index(1)] if 1 in KPC else float("nan")
    print(f"  resolution headline: at k_pc=1 (differ by 1 image/class) acc={k1:.3f} vs chance {chance:.2f} "
          f"→ {'distinguishes near-identical sets' if k1 > chance + 0.2 else 'breaks down near identity'}")


if __name__ == "__main__":
    main()
