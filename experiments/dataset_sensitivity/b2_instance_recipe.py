"""B2 — INSTANCE-level recipe-invariance: match an adapter to its EXACT image-sample using references of a
DIFFERENT activation (different frozen base). same-activation (other init) baseline vs cross-activation test.
Full ΔW distance + Grassmann-only norm-control; kNN; chance=1/#samples; cluster-robust over samples;
permutation null. Cross-activation POSITIVE = instance fingerprint is recipe-invariant (strong); a NULL is
AMBIGUOUS (base-geometry gap too wide = test-power), NOT 'recipe-specific'. DETECTION not reconstruction.
"""
import numpy as np, os
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from experiments.dataset_sensitivity.atlas_analyze import _load, dw_distance, _knn_dist
from experiments.dataset_sensitivity.instance_recovery import grass_only_distance

BANK = "results/b2_instance_recipe_zoo/b2_bank.pth"
RNG = np.random.default_rng(0)
CONDS = [("same-activation", "same"), ("cross-activation\n(diff base)", "cross")]


def main():
    bank, meta = _load(BANK)
    act = np.array([c["activation"] for c in bank]); samp = np.array([c["composition"] for c in bank])
    init = np.array([c["init_seed"] for c in bank]); n = len(bank)
    usamp = sorted(set(samp)); chance = 1.0 / len(usamp)
    Dfull = dw_distance(bank); Dgrass = grass_only_distance(bank)
    print(f"[B2] {n} adapters | acts={sorted(set(act))} samples={usamp} chance={chance:.3f}")

    def tmask(i, cond):
        return (act == act[i]) & (init != init[i]) if cond == "same" else (act != act[i])

    K_REF, R = 1, 40   # EQUALIZE references to K_REF per label (min across conds) → fair same-vs-cross ordering

    def correct(cond, D, labels):
        """Reference-count-EQUALIZED matching: subsample K_REF references per label, average over R draws, so
        the same-vs-cross comparison is not a kNN-richness artifact (auditor). Same procedure used for the null."""
        cor = np.full(n, np.nan)
        for i in range(n):
            pool = np.where(tmask(i, cond))[0]
            if len(pool) == 0:
                continue
            accs = []
            for r in range(R):
                rng = np.random.default_rng(1000 * i + r)
                sel = []
                for lab in np.unique(labels[pool]):
                    cand = pool[labels[pool] == lab]
                    sel.extend(rng.choice(cand, min(K_REF, len(cand)), replace=False))
                sel = np.array(sel)
                accs.append(float(_knn_dist(D[i:i + 1, sel], labels[sel], min(3, len(sel)))[0] == labels[i]))
            cor[i] = float(np.mean(accs))
        return cor

    res = {}
    print("\n=== instance recipe-invariance (match to exact image-sample) ===")
    for name, cond in CONDS:
        row = {}
        for dn, D in [("full", Dfull), ("grass", Dgrass)]:
            cor = correct(cond, D, samp)
            cl = np.array([np.nanmean(cor[samp == s]) for s in usamp])
            est = float(np.nanmean(cl)); se = np.nanstd(cl, ddof=1) / np.sqrt(len(cl))
            t = stats.t.ppf(0.975, len(cl) - 1); ci = (est - t * se, est + t * se)
            null = np.array([np.nanmean(correct(cond, D, RNG.permutation(samp))) for _ in range(150)])
            row[dn] = (est, ci, float((null >= est).mean()))
        res[cond] = row
        (ef, cf, pf) = row["full"]; (eg, _, _) = row["grass"]
        print(f"  {name.splitlines()[0]:18s}: acc={ef:.3f} CI[{cf[0]:.3f},{cf[1]:.3f}] above-chance={ef-chance:+.3f} "
              f"p={pf:.3f}{'*' if pf<0.05 else ''}  [grass-only={eg:.3f}]")

    fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=140)
    ax.axhline(chance, color="#d95f0e", ls=":", lw=1.4, label=f"chance = {chance:.2f}")
    xs = range(len(CONDS))
    for dn, col, lbl in [("full", "#2c7fb8", "ΔW full"), ("grass", "#7bccc4", "Grassmann-only")]:
        m = np.array([res[c[1]][dn][0] for c in CONDS])
        lo = np.array([res[c[1]][dn][1][0] for c in CONDS]); hi = np.array([res[c[1]][dn][1][1] for c in CONDS])
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt="o-", color=col, lw=2, ms=9, capsize=5, label=lbl)
    ax.set_xticks(list(xs)); ax.set_xticklabels([c[0] for c in CONDS]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("instance (image-sample) matching accuracy", fontsize=11)
    ax.set_title("Does INSTANCE identity survive a recipe change?\nmatch to exact image-sample, references of a different activation",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.text(0.5, -0.26, "DETECTION not reconstruction · weakest-attacker · {0,1} N=4 MNIST-MLP · references "
            "EQUALIZED to 1/sample (fair same-vs-cross) · both conds ≫ chance = recipe-invariant instance id",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#555")
    os.makedirs("figures/harder_id", exist_ok=True)
    fig.savefig("figures/harder_id/b2_instance_recipe.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("\n[saved] figures/harder_id/b2_instance_recipe.png")


if __name__ == "__main__":
    main()
