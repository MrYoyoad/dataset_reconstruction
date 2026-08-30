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
from experiments.dataset_sensitivity.atlas_analyze import _load, dw_distance, _knn_dist, _svd_feats, _grass

BANK = "results/instance_zoo/instance_bank.pth"
RNG = np.random.default_rng(0)


def grass_only_distance(bank):
    """NORM-CONTROL: pure subspace-DIRECTION distance — Grassmann on U and V only, NO spectral (singular-value)
    term. The full atlas distance is already overall-norm-invariant (Grassmann is scale-free; the spectral term
    is the COSINE of singular-value vectors, also scale-free), so this strips the last magnitude-carrying piece:
    if recovery survives here it is the ΔW DIRECTION (content), not any spectral-magnitude fingerprint."""
    feats = [_svd_feats(c["dW"]) for c in bank]
    n = len(feats); D = np.zeros((n, n))
    for i in range(n):
        Ui, _si, Vi = feats[i]
        for j in range(i + 1, n):
            Uj, _sj, Vj = feats[j]
            D[i, j] = D[j, i] = _grass(Ui, Uj) + _grass(Vi, Vj)   # direction only
    return D


def _loio(D, comp, init, uinit, k=3):
    """Leave-one-init-out kNN recovery accuracy per init fold."""
    per_fold = []
    for ho in uinit:
        te = np.where(init == ho)[0]; tr = np.where(init != ho)[0]
        pred = _knn_dist(D[np.ix_(te, tr)], comp[tr], k)
        per_fold.append(float((pred == comp[te]).mean()))
    return np.array(per_fold)


def _report(name, D, comp, init, uinit, chance):
    acc = _loio(D, comp, init, uinit); est = acc.mean()
    se = acc.std(ddof=1) / np.sqrt(len(acc)); t = stats.t.ppf(0.975, len(acc) - 1)
    ci = (est - t * se, est + t * se)
    null = np.array([_loio(D, RNG.permutation(comp), init, uinit).mean() for _ in range(1000)])
    pval = float((null >= est).mean())
    ok = ci[0] > chance and pval < 0.05
    print(f"\n=== {name} ===")
    print(f"  recovery acc = {est:.3f}  CI95 [{ci[0]:.3f}, {ci[1]:.3f}]  ({len(uinit)} init-folds)")
    print(f"  chance = {chance:.3f}   perm-null mean = {null.mean():.3f}  p = {pval:.3f}{'*' if pval < 0.05 else ''}"
          f"  → {'ABOVE chance' if ok else 'INDETERMINATE'}")
    return est


def main():
    bank, meta = _load(BANK)
    comp = np.array([c["composition"] for c in bank])       # image-sample id (the instance)
    init = np.array([c["init_seed"] for c in bank])         # recipe nuisance
    n_samp = len(set(comp)); chance = 1.0 / n_samp
    uinit = sorted(set(init))
    print(f"[instance-recovery] {len(bank)} adapters | {n_samp} image-samples × {len(uinit)} inits | chance={chance:.3f}")

    full = _report("FULL ΔW subspace distance (Grassmann + spectral-cos)", dw_distance(bank), comp, init, uinit, chance)
    grass = _report("NORM-CONTROL: Grassmann-ONLY (pure direction, no spectral magnitudes)",
                    grass_only_distance(bank), comp, init, uinit, chance)
    print(f"\n=== NORM-CONTROL VERDICT ===")
    if grass >= 0.99 * full and grass > chance + 0.3:
        print(f"  full={full:.3f}  grass-only={grass:.3f}  → SURVIVES: instance identity is in the ΔW DIRECTION "
              f"(content), NOT a spectral-magnitude fingerprint. Claimable as a direction result.")
    else:
        print(f"  full={full:.3f}  grass-only={grass:.3f}  → DROPS: the full-distance recovery leaned on spectral "
              f"magnitudes; the direction-only signal is weaker. Scope accordingly.")
    print(f"\n  [observe-framed | weakest-attacker | EXISTENCE result in the EASIEST instance setting: closed-set")
    print(f"   1-of-{n_samp}, DISJOINT/max-separable samples, seed-generalization only. NOT a difficulty measure.]")
    print(f"  [context: ΔW~instance ARI=+0.443 vs ΔW~init=+0.051 — structure driven by images, not recipe.]")


if __name__ == "__main__":
    main()
