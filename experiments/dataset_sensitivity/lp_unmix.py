"""Open-world unmixing via LP VERTEX SEARCH (auditor yoado-d4) — the right method, CPU, no training.

The private images are identified by the BOX + SPARSITY, not independence: raw image x = Vc + m must satisfy
0 ≤ Vc+m ≤ 1, so the feasible set P={c: box} is a polytope; an MNIST digit has ~600 exact-zero pixels → it sits
on ~600 active "pixel=0" constraints = a VERTEX of P and the SPARSEST feasible points (mixtures have union of
supports = fewer zeros). Recover by LP: min ⟨w,c⟩ s.t. box for many random directions w (each lands on a vertex)
+ the min-total-intensity direction (sparsest); collect distinct vertices, rank by #zeros, take the N sparsest
mutually-non-collinear. No sign/scale ambiguity (the box fixes scale — ICA cannot). CPU pre-test: form V,m from
PLANTED images directly (exact span, no adapter — ΔW gives the same V to 1e-14). Note: this repo centers by the
DATASET mean (fixed/public), so span is N-dim (not N−1); the LP argument is unaffected. Eval: Hungarian ssim +
margin + mean-baseline; N-sweep; ICA on the same plot as the generic-prior baseline. SCOPE: A₀=0 first-layer,
N≤r, open-world, this-attacker.
"""
import argparse, os, numpy as np, torch
from scipy.optimize import linprog, linear_sum_assignment
from experiments.data_utils import _load_dataset, _get_binary_label

RESULTS = "/home/projects/galvardi/yoado/results/lp_unmix"
NS = [2, 3, 4, 6, 8]


def ssim(a, b):
    a, b = a.ravel(), b.ravel(); mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var(); cov = ((a - mu_a) * (b - mu_b)).mean(); c1, c2 = 1e-4, 9e-4
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def lp_vertex(V, m, w):
    A_ub = np.vstack([V, -V]); b_ub = np.concatenate([1 - m, m])
    r = linprog(w, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * V.shape[1], method="highs")
    return r.x if r.success else None


def recover(V, m, N, n_dir=120, seed=0):
    d = V.shape[1]; rng = np.random.default_rng(seed)
    dirs = [rng.standard_normal(d) for _ in range(n_dir)] + [V.sum(0), -V.sum(0)]   # +min-intensity (sparsest)
    verts = [c for c in (lp_vertex(V, m, w) for w in dirs) if c is not None]
    imgs = [np.clip(V @ c + m, 0, 1) for c in verts]
    zeros = [int((img < 0.05).sum()) for img in imgs]                               # sparsity = #near-zero pixels
    order = np.argsort(zeros)[::-1]
    picked = []
    for i in order:
        u = imgs[i] / (np.linalg.norm(imgs[i]) + 1e-12)
        if all(abs(u @ (imgs[j] / (np.linalg.norm(imgs[j]) + 1e-12))) < 0.97 for j in picked):
            picked.append(i)
        if len(picked) == N:
            break
    return [imgs[i] for i in picked]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true")
    ap.add_argument("--trials", type=int, default=8); args = ap.parse_args()
    ds = _load_dataset("mnist", train=True)
    tgt = ds.targets if torch.is_tensor(ds.targets) else torch.tensor(ds.targets)
    dat = ds.data.reshape(len(ds.data), -1).double() / 255.0
    idx01 = {d: (tgt == d).nonzero(as_tuple=True)[0] for d in (0, 1)}
    m = torch.cat([dat[idx01[d][:1500]] for d in (0, 1)]).mean(0).numpy()           # dataset {0,1} mean (public)
    print(f"[lp-unmix] CPU vertex search on the exact span (planted images) | N-sweep {NS} | m=public {{0,1}} mean")

    summary = {}
    for N in NS:
        g = torch.Generator().manual_seed(7 + N)
        recs, bases, exact_pix = [], [], 0
        for t in range(args.trials):
            per = {0: N - N // 2, 1: N // 2}
            sel = torch.cat([idx01[d][torch.randperm(len(idx01[d]), generator=g)[:per[d]]] for d in (0, 1)])
            X = dat[sel].numpy()                                                     # (N,784) raw images
            X0 = X - m
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            keep = int((S > 1e-6 * S[0]).sum()); V = Vt[:keep].T                     # span basis (784, keep)
            R = recover(V, m, N, seed=t)
            if len(R) < N:
                R += [m.copy()] * (N - len(R))
            C = np.array([[ssim(r, X[j]) for j in range(N)] for r in R])
            ri, ci = linear_sum_assignment(-C)
            for a, b in zip(ri, ci):
                recs.append(ssim(R[a], X[b])); bases.append(ssim(m, X[b]))
                exact_pix += int(ssim(R[a], X[b]) > 0.98)
        recs, bases = np.array(recs), np.array(bases); margin = recs - bases
        summary[N] = dict(ssim=float(recs.mean()), base=float(bases.mean()), margin=float(margin.mean()),
                          frac_recog=float((margin > 0).mean()), frac_exact=exact_pix / len(recs))
        tag = "N≤r=8" if N <= 8 else "N>r"
        print(f"  N={N} ({tag}): LP recon SSIM={recs.mean():.3f} vs baseline {bases.mean():.3f} "
              f"(margin {margin.mean():+.3f}) | frac>base={float((margin>0).mean()):.2f} | frac EXACT(>0.98)={exact_pix/len(recs):.2f}")

    print(f"\n  [vs ICA: N=2 0.55/N=4 0.41/N=8 0.37 — LP should be ≫ if the box+sparsity identifies the vertices]")
    print(f"  [SCOPE: A₀=0 first-layer, N≤r, open-world, this-attacker; ceilings closed-world=1.0, mean-image=baseline]")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(summary=summary, NS=NS), os.path.join(RESULTS, "lp_unmix.pth"))
        print(f"[saved] {RESULTS}/lp_unmix.pth")


if __name__ == "__main__":
    main()
