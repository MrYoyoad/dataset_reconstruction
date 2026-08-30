"""Instance-level recovery — CORRECT cross-fit (the Facet-C fold-isolation artifact fix).

The instance zoo has ONE activation + ONE lr, so atlas_analyze's Facet-C folds by composition-cell → each
image-sample is wholly held out → structurally unpredictable (+0.000 artifact). The right design: the
NUISANCE is init_seed, so LEAVE-ONE-INIT-OUT — hold out all adapters of one init as test, train on the other
inits (every image-sample present in train), kNN-predict the test adapter's image-sample from ΔW-subspace
distance. Answers the OPEN question: can ΔW recover WHICH image-instance (instance-level), beyond chance?
Cluster-robust over the 8 init folds; permutation null; chance = 1/n_samples. numpy/scipy only.
"""
import numpy as np
from scipy import stats
from experiments.dataset_sensitivity.atlas_analyze import _load, dw_distance, _knn_dist

BANK = "results/instance_zoo/instance_bank.pth"
RNG = np.random.default_rng(0)


def main():
    bank, meta = _load(BANK)
    comp = np.array([c["composition"] for c in bank])       # image-sample id (the instance)
    init = np.array([c["init_seed"] for c in bank])         # recipe nuisance
    Ddw = dw_distance(bank)
    n_samp = len(set(comp)); chance = 1.0 / n_samp
    uinit = sorted(set(init))
    print(f"[instance-recovery] {len(bank)} adapters | {n_samp} image-samples × {len(uinit)} inits | chance={chance:.3f}")

    # leave-one-init-out: every image-sample stays in TRAIN (via other inits) → predictable
    per_fold = []
    for ho in uinit:
        te = np.where(init == ho)[0]; tr = np.where(init != ho)[0]
        pred = _knn_dist(Ddw[np.ix_(te, tr)], comp[tr], 3)
        per_fold.append(float((pred == comp[te]).mean()))
    acc = np.array(per_fold); est = acc.mean()
    se = acc.std(ddof=1) / np.sqrt(len(acc)); t = stats.t.ppf(0.975, len(acc) - 1)
    ci = (est - t * se, est + t * se)

    # permutation null: shuffle composition labels, redo LOIO
    def _run(lab):
        p = []
        for ho in uinit:
            te = np.where(init == ho)[0]; tr = np.where(init != ho)[0]
            p.append((_knn_dist(Ddw[np.ix_(te, tr)], lab[tr], 3) == lab[te]).mean())
        return np.mean(p)
    null = np.array([_run(RNG.permutation(comp)) for _ in range(1000)])
    pval = float((null >= est).mean())

    print(f"\n=== INSTANCE-LEVEL recovery (leave-one-init-out kNN on ΔW subspace) ===")
    print(f"  recovery acc = {est:.3f}  CI95 [{ci[0]:.3f}, {ci[1]:.3f}]  (8 init-folds)")
    print(f"  chance = {chance:.3f}   permutation-null mean = {null.mean():.3f}  p = {pval:.3f}{'*' if pval < 0.05 else ''}")
    lift = est - chance
    verdict = ("INSTANCE-LEVEL recovery ABOVE chance (CI excludes chance, p<0.05)"
               if ci[0] > chance and pval < 0.05 else
               "INDETERMINATE — CI includes chance or null not beaten")
    print(f"  lift over chance = {lift:+.3f}  → {verdict}")
    print(f"\n  [observe-framed | weakest-attacker | same-digits {{0,1}}, composition = image-sample]")
    print(f"  [context: content-level atlas ARI≈near-perfect; here ΔW~instance ARI=+0.443 vs ΔW~init=+0.051]")


if __name__ == "__main__":
    main()
