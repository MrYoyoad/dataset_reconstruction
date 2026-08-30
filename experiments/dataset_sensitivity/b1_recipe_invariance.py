"""B1 — RECIPE-INVARIANT content matching (graded recipe-distance curve). Removes crutch iii (known recipe).

Off the atlas factorial zoo {activation × lr × composition × init} already on disk. Match a target adapter to
its COMPOSITION (digit content) using REFERENCE adapters at increasing RECIPE-DISTANCE:
  same-recipe (same act+lr, other init = baseline) → cross-lr (same act, diff lr, SAME base) →
  cross-activation (diff act, DIFFERENT frozen base = the strong cut).
Headline = the GRADED curve (accuracy vs recipe-distance), NOT a single cross-activation number — a
cross-activation null is AMBIGUOUS (test-power vs recipe-specific). kNN on the gauge-clean ΔW subspace;
chance=1/#comps; cluster-robust over compositions; per-condition permutation null. DETECTION, not reconstruction.
"""
import numpy as np, os
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from experiments.dataset_sensitivity.atlas_analyze import _load, dw_distance, _knn_dist

BANK = "results/atlas_zoo/zoo_bank.pth"
RNG = np.random.default_rng(0)
CONDS = [("same-recipe", "same"), ("cross-lr\n(same base)", "cross_lr"), ("cross-activation\n(diff base)", "cross_act")]


def main():
    bank, meta = _load(BANK)
    act = np.array([c["activation"] for c in bank]); lr = np.array([c["lr"] for c in bank])
    comp = np.array([c["composition"] for c in bank]); init = np.array([c["init_seed"] for c in bank])
    n = len(bank); ucomp = sorted(set(comp)); chance = 1.0 / len(ucomp)
    Ddw = dw_distance(bank)
    print(f"[B1] {n} adapters | acts={sorted(set(act))} lrs={sorted(set(lr))} comps={ucomp} chance={chance:.3f}")

    def train_mask(i, cond):
        if cond == "same":     return (act == act[i]) & (lr == lr[i]) & (init != init[i])
        if cond == "cross_lr": return (act == act[i]) & (lr != lr[i])
        if cond == "cross_act":return (act != act[i])

    def per_target_correct(cond, labels):
        cor = np.full(n, np.nan)
        for i in range(n):
            tr = np.where(train_mask(i, cond))[0]
            if len(tr) == 0 or len(set(labels[tr])) < 1:
                continue
            pred = _knn_dist(Ddw[i:i + 1, tr], labels[tr], 3)[0]
            cor[i] = float(pred == labels[i])
        return cor

    means, los, his, ps = [], [], [], []
    print("\n=== graded recipe-distance curve (content matching accuracy) ===")
    for name, cond in CONDS:
        cor = per_target_correct(cond, comp)
        # cluster-robust over composition
        cl = np.array([np.nanmean(cor[comp == c]) for c in ucomp])
        est = float(np.nanmean(cl)); se = np.nanstd(cl, ddof=1) / np.sqrt(len(cl))
        t = stats.t.ppf(0.975, len(cl) - 1); ci = (est - t * se, est + t * se)
        null = np.array([np.nanmean(per_target_correct(cond, RNG.permutation(comp))) for _ in range(300)])
        pval = float((null >= est).mean())
        means.append(est); los.append(ci[0]); his.append(ci[1]); ps.append(pval)
        above = est - chance
        print(f"  {name.splitlines()[0]:18s}: acc={est:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}] "
              f"above-chance={above:+.3f} p={pval:.3f}{'*' if pval < 0.05 else ''}")

    means, los, his = map(np.array, (means, los, his))
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    ax.axhline(chance, color="#d95f0e", ls=":", lw=1.4, label=f"chance = {chance:.2f}")
    ax.errorbar(range(len(CONDS)), means, yerr=[means - los, his - means], fmt="o-", color="#2c7fb8",
                lw=2, ms=10, capsize=5, zorder=3, label="content matching accuracy")
    for i, (m, p) in enumerate(zip(means, ps)):
        ax.annotate(f"{m:.2f}{'*' if p < 0.05 else ''}", (i, m), textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold", color="#0a4d7a")
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([c[0] for c in CONDS])
    ax.set_ylim(0, 1.05); ax.set_xlabel("recipe-distance of the reference adapters →", fontsize=11)
    ax.set_ylabel("cross-recipe content-matching accuracy", fontsize=11)
    ax.set_title("Recipe-invariance of the content fingerprint (graded)\n"
                 "does ΔW match an adapter to its data when the REFERENCE used a different recipe?",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.text(0.5, -0.32, "DETECTION not reconstruction · observe-framed · weakest-attacker · atlas zoo "
            "MNIST-MLP N=4 · cluster-robust over compositions · cross-act null is ambiguous (test-power)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555")
    os.makedirs("figures/harder_id", exist_ok=True)
    fig.savefig("figures/harder_id/b1_recipe_invariance.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("\n[saved] figures/harder_id/b1_recipe_invariance.png")
    print("  [DETECTION not reconstruction · a cross-activation drop = 'not invariant to a full base-geometry "
          "change', NOT 'recipe-specific fingerprint']")


if __name__ == "__main__":
    main()
