"""Adapter-space atlas — ANALYSIS of the factorial zoo (plan §1/§4/§5/§12). numpy+scipy only (no sklearn).

Consumes results/atlas_zoo/zoo_bank.pth (per-cell raw B,A over {activation}×{composition}×{lr}×{init}).
Two clustering methods + the variance-decomposition:
  (1) ΔW=B·A (gauge-clean, PRIMARY) — cluster the actual adapter value; decompose the partition across
      {init,lr,activation,composition}: adjusted-Rand vs each factor with a label-permutation null.
      Headline = "how much is the ΔW-clustering DIFFERENT from init/lr" (residual = composition signal).
  (2) raw (B,A) (SECOND, contrast) — UNcanonicalized (init frame IS the signal): does it associate with
      init/activation MORE than ΔW does? (association-vs-null, not bare divergence.)
  Facet-C composition recovery (§12): cross-fitted held-out ACCURACY DIFFERENCE (kNN on nuisance+ΔW feats
  minus nuisance-only), CI = cluster-robust t_{G-1}·sd_cluster/√G over composition-cells.
Observe-framed, weakest-attacker. Per-seed zoo → CAN attribute (unlike the seed-mean pre-look).
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy import stats

BANK = "results/atlas_zoo/zoo_bank.pth"
OUT = "figures/atlas/atlas.png"
P = 8
RNG = np.random.default_rng(0)


def _load():
    d = torch.load(BANK, map_location="cpu", weights_only=False)
    bank = [c for c in d["bank"] if c.get("converged", True)]
    for c in bank:
        A = c["A"].to(torch.float64).numpy(); B = c["B"].to(torch.float64).numpy()
        c["dW"] = B @ A
        c["BA_flat"] = np.concatenate([B.ravel(), A.ravel()])
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
    X = np.stack([c["BA_flat"] for c in bank]); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return np.sqrt(np.maximum(2 - 2 * (X @ X.T), 0))


def _clabels(D, k):
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, k, criterion="maxclust")


def _ari(a, b):
    a = np.asarray([str(x) for x in a]); b = np.asarray(b)
    ca, cb = np.unique(a), np.unique(b)
    cont = np.array([[np.sum((a == x) & (b == y)) for y in cb] for x in ca], dtype=float)
    comb = lambda v: (v * (v - 1) / 2).sum()
    si = comb(cont.sum(1)); sj = comb(cont.sum(0)); s = comb(cont.ravel())
    n = len(a); tot = n * (n - 1) / 2
    exp = si * sj / tot; mx = 0.5 * (si + sj)
    return (s - exp) / (mx - exp) if mx != exp else 0.0


def assoc(D, labels, n_perm=2000):
    lab = np.array([str(x) for x in labels]); k = len(set(lab))
    if k < 2 or k >= len(lab):
        return None
    cl = _clabels(D, k)
    obs = _ari(lab, cl)
    null = np.array([_ari(RNG.permutation(lab), cl) for _ in range(n_perm)])
    return dict(ARI=float(obs), p=float((null >= obs).mean()), k=k)


def _knn_predict(Xtr, ytr, Xte, k):
    k = max(1, min(k, len(Xtr)))
    preds = []
    for x in Xte:
        d = ((Xtr - x) ** 2).sum(1)
        nn = ytr[np.argsort(d)[:k]]
        vals, cnts = np.unique(nn, return_counts=True)
        preds.append(vals[np.argmax(cnts)])
    return np.array(preds)


def facet_c(bank):
    acts = sorted(set(c["activation"] for c in bank))
    def feat(c, with_dw):
        nu = [c["init_seed"], c["lr"]] + [1.0 * (c["activation"] == a) for a in acts]
        if with_dw:
            s = _svd_feats(c["dW"])[1]; nu = nu + list(s) + [float(np.linalg.norm(s))]
        return nu
    y = np.array([c["composition"] for c in bank])
    groups = np.array([f"{c['activation']}|{c['lr']}|{c['composition']}" for c in bank])
    Xn = np.array([feat(c, False) for c in bank], float)
    Xf = np.array([feat(c, True) for c in bank], float)
    # standardize columns (kNN scale)
    for X in (Xn, Xf):
        sd = X.std(0); sd[sd == 0] = 1; X[:] = (X - X.mean(0)) / sd
    ug = sorted(set(groups)); G = len(ug)
    fold_of = {g: i % min(5, G) for i, g in enumerate(ug)}
    folds = np.array([fold_of[g] for g in groups])
    s_i = np.zeros(len(bank))
    for f in range(min(5, G)):
        te = folds == f; tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        pf = _knn_predict(Xf[tr], y[tr], Xf[te], 3)
        pn = _knn_predict(Xn[tr], y[tr], Xn[te], 3)
        s_i[te] = (pf == y[te]).astype(float) - (pn == y[te]).astype(float)
    cl_means = np.array([s_i[groups == g].mean() for g in ug])
    Gn = len(cl_means); est = float(cl_means.mean())
    se = cl_means.std(ddof=1) / np.sqrt(Gn) if Gn > 1 else float("nan")
    t = stats.t.ppf(0.975, Gn - 1) if Gn > 1 else float("nan")
    return dict(acc_diff=est, ci=(est - t * se, est + t * se), G=Gn)


def _mds(D):
    n = D.shape[0]; J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 1e-12))


def main():
    bank, meta = _load()
    n = len(bank)
    print(f"[atlas] {n} converged adapters | acts={meta['acts']} comps={meta['comps']} lrs={meta['lrs']} inits={len(meta['inits'])}")
    Ddw, Dba = dw_distance(bank), ba_distance(bank)

    print("\n=== (1) ΔW-clustering decomposition — adjusted-Rand vs factor (permutation p) ===")
    dw_assoc = {}
    for fkey in ["composition", "activation", "lr", "init_seed"]:
        r = assoc(Ddw, [c[fkey] for c in bank]); dw_assoc[fkey] = r
        if r:
            print(f"  ΔW ~ {fkey:12s} ARI={r['ARI']:+.3f} p={r['p']:.3f}{'*' if r['p']<0.05 else ' '} (k={r['k']})")
    print("\n=== (2) raw (B,A) init-contrast — should associate with init/activation MORE than ΔW ===")
    for fkey in ["init_seed", "activation"]:
        rba = assoc(Dba, [c[fkey] for c in bank]); rdw = dw_assoc.get(fkey)
        if rba and rdw:
            print(f"  (B,A) ~ {fkey:10s} ARI={rba['ARI']:+.3f} p={rba['p']:.3f}  |  ΔW ARI={rdw['ARI']:+.3f}  "
                  f"{'→ (B,A) HIGHER (init frame in factors, scrubbed by product)' if rba['ARI']>rdw['ARI']+0.02 else '→ no clear init-contrast'}")
    print("\n=== Facet-C: composition recovery beyond nuisance (cross-fitted acc-diff, cluster-robust CI) ===")
    fc = facet_c(bank)
    real = fc["ci"][0] > 0
    print(f"  acc(nuisance+ΔW) − acc(nuisance) = {fc['acc_diff']:+.3f}  CI95 [{fc['ci'][0]:+.3f},{fc['ci'][1]:+.3f}] "
          f"(G={fc['G']} cells) → {'RECOVERS composition beyond nuisance' if real else 'INDETERMINATE (CI spans 0)'}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=140)
    for row, (name, D, cols) in enumerate([("ΔW (gauge-clean)", Ddw, ["composition", "activation", "init_seed"]),
                                           ("raw (B,A)", Dba, ["init_seed", "activation", "composition"])]):
        emb = _mds(D)
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
