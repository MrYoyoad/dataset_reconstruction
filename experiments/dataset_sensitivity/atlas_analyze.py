"""Adapter-space atlas — ANALYSIS of the factorial zoo (plan §1/§4/§5/§12).

Consumes results/atlas_zoo/zoo_bank.pth (per-cell raw B,A over {activation}×{composition}×{lr}×{init}).
Runs the user's TWO clustering methods and the variance-decomposition:

  (1) ΔW = B·A method (gauge-clean, PRIMARY) — cluster the actual adapter value; decompose the partition
      across {init, lr, activation, composition}: adjusted-Rand vs each factor with a label-permutation null.
      Headline = "how much is the ΔW-clustering DIFFERENT from init/lr" (the residual = composition signal).
  (2) raw (B,A) method (SECOND, contrast) — UNcanonicalized (the init frame IS the signal). Test:
      raw (B,A) should associate with init/activation MORE than ΔW does (init/gauge lives in the factors,
      scrubbed by the product). Reported as association-vs-null, NOT bare partition divergence.

  Facet-C composition recovery (§12): cross-fitted held-out ACCURACY DIFFERENCE — a classifier predicting
  composition from (nuisance+ΔW features) minus one from (nuisance features only), scored on held-out cells.
  CI = cluster-robust t_{G-1}·sd_cluster/√G over composition-cells (NOT z·sd/√n_adapters — pseudo-replication).

Observe-framed, weakest-attacker scoped. Because the zoo is PER-SEED (not seed-means), it CAN attribute the
clustering to a factor — this is the honest read the pre-look could not give.

Run (bsub):  python -u -m experiments.dataset_sensitivity.atlas_analyze
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANK = "results/atlas_zoo/zoo_bank.pth"
OUT = "figures/atlas/atlas.png"
P = 8
RNG = np.random.default_rng(0)


def _load():
    d = torch.load(BANK, map_location="cpu", weights_only=False)
    bank = [c for c in d["bank"] if c.get("converged", True)]
    for c in bank:
        A = c["A"].to(torch.float64).numpy()      # (r, in)
        B = c["B"].to(torch.float64).numpy()      # (out, r)
        c["dW"] = B @ A                            # gauge-clean product
        c["BA_flat"] = np.concatenate([B.ravel(), A.ravel()])  # raw factors (uncanonicalized)
    return bank, d["meta"]


def _svd_feats(dw):
    U, s, Vt = np.linalg.svd(dw, full_matrices=False)
    p = min(P, len(s))
    return U[:, :p], s[:p], Vt[:p].T


def _grass(A, B):
    p = min(A.shape[1], B.shape[1]); A, B = A[:, :p], B[:, :p]
    m = A.T @ B
    return float(np.sqrt(max(p - (m * m).sum(), 0.0)))


def dw_distance(bank):
    feats = [_svd_feats(c["dW"]) for c in bank]
    n = len(feats); D = np.zeros((n, n))
    for i in range(n):
        Ui, si, Vi = feats[i]
        for j in range(i + 1, n):
            Uj, sj, Vj = feats[j]
            p = min(len(si), len(sj))
            cos = float((si[:p] @ sj[:p]) / (np.linalg.norm(si[:p]) * np.linalg.norm(sj[:p]) + 1e-12))
            D[i, j] = D[j, i] = _grass(Ui, Uj) + _grass(Vi, Vj) + (1 - cos)
    return D


def ba_distance(bank):
    # raw (B,A) two-sided Euclidean on the UNcanonicalized flattened factors (init frame kept on purpose)
    X = np.stack([c["BA_flat"] for c in bank]); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    G = X @ X.T
    return np.sqrt(np.maximum(2 - 2 * G, 0))


def assoc(D, labels, n_perm=2000):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    lab = np.array([str(x) for x in labels]); k = len(set(lab))
    if k < 2 or k >= len(lab):
        return None
    cl = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D)
    obs = adjusted_rand_score(lab, cl)
    null = np.array([adjusted_rand_score(RNG.permutation(lab), cl) for _ in range(n_perm)])
    return dict(ARI=obs, p=float((null >= obs).mean()), k=k)


def facet_c(bank):
    """Composition recovery beyond nuisance — cross-fitted held-out accuracy difference + cluster-robust CI."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GroupKFold
    from scipy import stats
    # nuisance features: init_seed, lr, activation (one-hot); ΔW features: top-P singular values (gauge-inv)
    acts = sorted(set(c["activation"] for c in bank))
    def feat(c, with_dw):
        nu = [c["init_seed"], c["lr"]] + [1.0 * (c["activation"] == a) for a in acts]
        if with_dw:
            s = _svd_feats(c["dW"])[1]
            nu = nu + list(s) + [float(np.linalg.norm(s))]
        return nu
    y = np.array([c["composition"] for c in bank])
    groups = np.array([f"{c['activation']}|{c['lr']}|{c['composition']}" for c in bank])  # cluster = comp-cell
    Xn = np.array([feat(c, False) for c in bank]); Xf = np.array([feat(c, True) for c in bank])
    G = len(set(groups))
    gkf = GroupKFold(n_splits=min(5, G))
    s_i = np.zeros(len(bank))
    for tr, te in gkf.split(Xf, y, groups):
        kn = min(3, len(tr))
        pf = KNeighborsClassifier(kn).fit(Xf[tr], y[tr]).predict(Xf[te])
        pn = KNeighborsClassifier(kn).fit(Xn[tr], y[tr]).predict(Xn[te])
        s_i[te] = (pf == y[te]).astype(float) - (pn == y[te]).astype(float)
    # cluster-robust CI over composition-cells (G clusters), t_{G-1}
    cl_means = np.array([s_i[groups == g].mean() for g in sorted(set(groups))])
    Gn = len(cl_means); est = cl_means.mean()
    se = cl_means.std(ddof=1) / np.sqrt(Gn) if Gn > 1 else float("nan")
    tcrit = stats.t.ppf(0.975, Gn - 1) if Gn > 1 else float("nan")
    return dict(acc_diff=float(est), ci=(float(est - tcrit * se), float(est + tcrit * se)), G=Gn)


def main():
    bank, meta = _load()
    n = len(bank)
    print(f"[atlas] {n} converged adapters | factors: acts={meta['acts']} comps={meta['comps']} "
          f"lrs={meta['lrs']} inits={len(meta['inits'])}")
    Ddw, Dba = dw_distance(bank), ba_distance(bank)

    factors = ["composition", "activation", "lr", "init_seed"]
    print("\n=== (1) ΔW-clustering decomposition — adjusted-Rand vs factor (permutation p) ===")
    dw_assoc = {}
    for fkey in factors:
        r = assoc(Ddw, [c[fkey] for c in bank]); dw_assoc[fkey] = r
        if r:
            print(f"  ΔW ~ {fkey:12s} ARI={r['ARI']:+.3f} p={r['p']:.3f}{'*' if r['p']<0.05 else ''}")
    print("\n=== (2) raw (B,A) init-contrast — should associate with init/activation MORE than ΔW ===")
    for fkey in ["init_seed", "activation"]:
        rba = assoc(Dba, [c[fkey] for c in bank]); rdw = dw_assoc.get(fkey)
        if rba and rdw:
            print(f"  (B,A) ~ {fkey:10s} ARI={rba['ARI']:+.3f} p={rba['p']:.3f}  |  ΔW ARI={rdw['ARI']:+.3f} "
                  f"→ {'(B,A) higher: init frame in factors, scrubbed by product' if rba['ARI']>rdw['ARI'] else 'no init-contrast'}")
    print("\n=== Facet-C: composition recovery beyond nuisance (cross-fitted acc-diff, cluster-robust CI) ===")
    fc = facet_c(bank)
    real = fc["ci"][0] > 0
    print(f"  acc(nuisance+ΔW) − acc(nuisance) = {fc['acc_diff']:+.3f}  CI95 [{fc['ci'][0]:+.3f},{fc['ci'][1]:+.3f}] "
          f"(G={fc['G']} cells) → {'RECOVERS composition beyond nuisance' if real else 'INDETERMINATE (CI spans 0)'}")

    # figures: MDS coloured by factor for both methods
    from sklearn.manifold import MDS
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=140)
    for row, (name, D, cols) in enumerate([("ΔW (gauge-clean)", Ddw, ["composition", "activation", "init_seed"]),
                                           ("raw (B,A)", Dba, ["init_seed", "activation", "composition"])]):
        emb = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto").fit_transform(D)
        for col, fkey in enumerate(cols):
            ax = axes[row, col]; vals = [str(c[fkey]) for c in bank]; uniq = sorted(set(vals))
            cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 2)))
            for u, cc in zip(uniq, cmap):
                idx = [i for i, v in enumerate(vals) if v == u]
                ax.scatter(emb[idx, 0], emb[idx, 1], color=cc, s=28, label=u, alpha=0.8, edgecolor="k", linewidth=0.2)
            ax.set_title(f"{name} — colour = {fkey}", fontsize=9, fontweight="bold")
            ax.legend(fontsize=6, framealpha=0.9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Adapter-space atlas — ΔW (gauge-clean) vs raw (B,A); per-seed factorial zoo (CAN attribute)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, -0.01, f"n={n} adapters, {fc['G']} composition-cells | OBSERVE-framed, weakest-attacker | "
             "headline = ΔW-clustering residual beyond init/lr/activation; (B,A) init-contrast; Facet-C acc-diff CI",
             ha="center", va="top", fontsize=8, color="#444")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
