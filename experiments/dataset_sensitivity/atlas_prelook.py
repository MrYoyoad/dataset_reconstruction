"""Adapter-space atlas — CPU PRE-LOOK (plan step 1, notes/adapter_atlas_plan.md).

EXPLORATORY ONLY. Clusters the EXISTING arm ΔW (arm_b/c/d/e) under gauge-invariant intrinsic distances and
asks whether the partition tracks any composition label ABOVE a label-permutation null. Two hard caveats,
stated on every output:
  (1) SEED-BLIND: the arm ΔW are seed-MEANS, so this literally cannot see the seed/init factor — any
      composition signal is an UPPER BOUND, not evidence.
  (2) EFFECTIVE-n capped + confounded: a few dozen adapters, and rank/dataset/N/composition all co-vary, so
      "clusters by X" here is exploratory. The honest attribution waits for the per-seed factorial zoo.

Gauge note: ΔW = BA is GL(r)-invariant (the raw B,A are not) — so clustering on ΔW is BOTH "the actual
adapter value" and gauge-clean. Featurize ΔW by its SVD (spectrum + subspaces); never flatten raw entries.

Distances (>=2, per plan §6): (A) SUBSPACE = chordal Grassmann on the top-p col AND row subspaces of ΔW;
(B) SPECTRAL = cosine distance on the top-p singular-value profile. Two genuinely different gauge-invariant
views. (Two-sided Bures-Wasserstein is the zoo headline; deferred.)

Run:  python -m experiments.dataset_sensitivity.atlas_prelook
"""
import glob
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

P = 8  # top-p singular directions (r=8 adapters; r=32 truncated to 8 — noted)
OUT = "figures/atlas/prelook.png"
RNG = np.random.default_rng(0)


def _load_dw(d):
    """Pull a single (out,in) ΔW from an arm result dict; average a context axis; None if absent."""
    for k in ("dW_seed_mean", "dW_ref", "dW_base_mean", "dW_distinct_mean"):
        if isinstance(d, dict) and k in d and hasattr(d[k], "shape"):
            w = d[k].detach().to(torch.float64).cpu().numpy()
            if w.ndim == 3:            # arm_d multi-context (C, out, in) -> mean over contexts
                w = w.mean(axis=0)
            return w if w.ndim == 2 else None
    return None


def load_population():
    """Return list of dicts: {dw, arm, dataset, N, rank, m, k, digits_sig}."""
    pop = []
    for arm in ("arm_b_dilution", "arm_c_imbalance", "arm_d_context", "arm_e_duplication"):
        for f in sorted(glob.glob(f"results/{arm}/*.pth")):
            try:
                d = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            dw = _load_dw(d)
            if dw is None:
                continue
            base = os.path.basename(f)
            digits = d.get("digits") if isinstance(d, dict) else None
            pop.append(dict(
                dw=dw, arm=arm.split("_")[1], file=base,
                dataset=("fashion" if "fashion" in base else "mnist"),
                N=int(d.get("N", -1)) if isinstance(d, dict) else -1,
                rank=int(d.get("rank", -1)) if isinstance(d, dict) else -1,
                m=int(d.get("m", -1)) if isinstance(d, dict) and "m" in d else -1,
                k=int(d.get("k", -1)) if isinstance(d, dict) and "k" in d else -1,
                digits_sig=tuple(sorted(set(digits))) if digits else None,
            ))
    return pop


def featurize(dw):
    """Top-p SVD of ΔW: (U_p [out,p], s_p [p], V_p [in,p]) — all gauge-invariant objects of the product."""
    U, s, Vt = np.linalg.svd(dw, full_matrices=False)
    p = min(P, len(s))
    return U[:, :p], s[:p], Vt[:p, :].T


def _grassmann_chordal(A, B):
    """Chordal Grassmann distance between two orthonormal bases (columns): sqrt(p - ||AᵀB||_F²)."""
    p = min(A.shape[1], B.shape[1])
    A, B = A[:, :p], B[:, :p]
    m = A.T @ B
    return float(np.sqrt(max(p - (m * m).sum(), 0.0)))


def dist_matrices(feats):
    n = len(feats)
    Dsub = np.zeros((n, n))
    Dspec = np.zeros((n, n))
    for i in range(n):
        Ui, si, Vi = feats[i]
        for j in range(i + 1, n):
            Uj, sj, Vj = feats[j]
            dsub = _grassmann_chordal(Ui, Uj) + _grassmann_chordal(Vi, Vj)   # col + row subspaces
            p = min(len(si), len(sj))
            a, b = si[:p], sj[:p]
            cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            dspec = 1.0 - cos
            Dsub[i, j] = Dsub[j, i] = dsub
            Dspec[i, j] = Dspec[j, i] = dspec
    return {"subspace (Grassmann col+row)": Dsub, "spectral (singular profile)": Dspec}


def assoc_vs_null(D, labels, n_perm=2000):
    """Adjusted-Rand between an agglomerative clustering of D and a label, vs a label-permutation null."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    lab = np.array([str(x) for x in labels])
    uniq = sorted(set(lab))
    kk = len(uniq)
    if kk < 2 or kk >= len(lab):
        return None
    cl = AgglomerativeClustering(n_clusters=kk, metric="precomputed", linkage="average").fit_predict(D)
    obs = adjusted_rand_score(lab, cl)
    null = np.array([adjusted_rand_score(RNG.permutation(lab), cl) for _ in range(n_perm)])
    p = float((null >= obs).mean())
    return dict(ARI=obs, p=p, k=kk, null_mean=float(null.mean()))


def main():
    pop = load_population()
    n = len(pop)
    feats = [featurize(p["dw"]) for p in pop]
    Ds = dist_matrices(feats)

    # effective-n honesty: distinct composition cells (ignoring seed, which is already averaged out)
    cells = set((p["arm"], p["dataset"], p["N"], p["rank"], p["m"], p["k"]) for p in pop)
    print(f"[atlas pre-look] n_adapters={n}  distinct composition-cells (seed-blind) = {len(cells)}")

    label_keys = ["dataset", "arm", "N", "rank"]
    print("\nassociation (adjusted-Rand vs label-permutation null):")
    lines = []
    for metric, D in Ds.items():
        for key in label_keys:
            r = assoc_vs_null(D, [p[key] for p in pop])
            if r:
                sig = "*" if r["p"] < 0.05 else " "
                line = f"  [{metric[:22]:22s}] {key:8s} ARI={r['ARI']:+.3f} p={r['p']:.3f}{sig} (k={r['k']})"
                print(line); lines.append((metric, key, r))

    # embed each distance (MDS) coloured by dataset/arm/N/rank
    from sklearn.manifold import MDS
    fig, axes = plt.subplots(len(Ds), 4, figsize=(19, 5 * len(Ds)), dpi=150)
    if len(Ds) == 1:
        axes = axes[None, :]
    for row, (metric, D) in enumerate(Ds.items()):
        emb = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto").fit_transform(D)
        for col, key in enumerate(label_keys):
            ax = axes[row, col]
            vals = [str(p[key]) for p in pop]
            uniq = sorted(set(vals))
            cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 2)))
            for u, c in zip(uniq, cmap):
                idx = [i for i, v in enumerate(vals) if v == u]
                ax.scatter(emb[idx, 0], emb[idx, 1], color=c, s=40, label=u, alpha=0.8, edgecolor="k", linewidth=0.3)
            ax.set_title(f"{metric.split(' ')[0]} MDS — colour = {key}", fontsize=9, fontweight="bold")
            ax.legend(fontsize=6, loc="best", framealpha=0.9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Adapter-space atlas — CPU PRE-LOOK (EXPLORATORY; SEED-BLIND; effective-n capped) — cluster the "
                 "actual ΔW (gauge-clean), does it track composition above a permutation null?",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.text(0.5, -0.01,
             f"n={n} arm adapters, {len(cells)} distinct composition-cells | ΔW=BA (GL(r)-invariant), top-{P} SVD | "
             "⚠ SEED-BLIND (arm ΔW are seed-means → cannot see seed/init) + rank/dataset/N/composition CONFOUNDED "
             "→ any 'clusters by X' here is an UPPER BOUND, exploratory; honest attribution needs the per-seed factorial zoo. "
             "Two-sided Bures-Wasserstein + the (B,A) contrast are the zoo, not this pre-look.",
             ha="center", va="top", fontsize=7.6, color="#444")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
