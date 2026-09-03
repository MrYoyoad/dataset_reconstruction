#!/usr/bin/env python3
"""Is the capacity boundary a property of the CHART or only of its dimension?

The point (user, 2026-09-03): k is not a property of the images.  It is the dimension of the chart chosen
to search in, so showing the boundary moves with r rules out ONE alternative (a chart-intrinsic threshold
independent of r) and says nothing about the fact that a different parameterisation is a different problem.

The counting argument only ever sees k: supply is capped by the release, demand is Nk + rN, and nothing
about how the chart is built enters.  That is a sharp, falsifiable prediction, so test it directly: hold
the DIGITS, N, m and r fixed, vary only the chart at matched dimension k.

  pca      psi(w) = mean + V_k w                     linear, data-fit, captures ~63% of variance at k=18
  warped   psi(w) = mean + V_k (w + c*tanh(M w))     the SAME manifold under a nonlinear reparametrisation
                                                     (M fixed random, the map is a diffeomorphism for small c)
  exact    psi(w) = mean + Q_k w  with the first N columns of Q_k spanning the private digits themselves
                                                     -- a chart that CONTAINS the true images exactly, so
                                                     recovery returns the digit, not a projection of it

Two predictions, which come apart:
  (i)  the identifiability boundary sits at the SAME k = m+r-N for all three charts (the count is blind to
       the chart), and
  (ii) how much of the TRUE digit is recovered differs enormously between them -- `exact` should return the
       real digits at machine precision at k as small as N, while `pca` returns only its own projection.
If (i) fails, the law is weaker than stated and depends on the parameterisation.
"""
import argparse, json, math, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import (
    train_release, simulate_sgd_reduced, invert_lm, qr_canon, git_hash, RECOVER_TOL)
from experiments.exact_inversion.mnist_capacity import load_mnist

torch.set_default_dtype(torch.float64)


class Chart:
    """A k-dimensional chart on image space, plus the frozen public encoder."""
    def __init__(self, kind, k, n, seed, dev, root, n_fit, N, warp_c=0.3):
        X, y = load_mnist(root, n_fit)
        Xt = torch.tensor(X, device=dev)
        self.mean = Xt.mean(0)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(Xt.shape[0], generator=g)[:N].to(dev)
        self.X_real = Xt[idx].T.contiguous()                      # (784, N) the private digits
        self.y = torch.tensor(y, device=dev)[idx]
        U, S, Vh = torch.linalg.svd(Xt - self.mean, full_matrices=False)
        self.kind, self.k, self.P, self.n = kind, k, 784, n
        if kind in ("pca", "warped"):
            self.B = Vh[:k].T.contiguous()                        # (784, k) principal directions
        elif kind == "exact":
            # a k-dim orthonormal basis whose span CONTAINS the N private digits exactly
            core = self.X_real - self.mean[:, None]               # (784, N)
            filler = (Vh[:max(0, k - N)].T if k > N else torch.zeros(784, 0, device=dev))
            M = torch.cat([core, filler], dim=1)
            self.B, _ = torch.linalg.qr(M)                        # (784, k)
        else:
            raise ValueError(kind)
        self.warp_c = warp_c if kind == "warped" else 0.0
        self.M = (torch.randn(k, k, generator=g) / math.sqrt(k)).to(dev)
        self.E1 = (torch.randn(128, 784, generator=g) / math.sqrt(784)).to(dev)
        self.E2 = (torch.randn(n, 128, generator=g) / math.sqrt(128)).to(dev)

    def psi(self, W):
        Z = W + self.warp_c * torch.tanh(self.M @ W) if self.warp_c else W
        return self.mean[:, None] + self.B @ Z

    def phi(self, Ximg):
        return torch.tanh(self.E2 @ torch.tanh(self.E1 @ Ximg))

    def features_from_latents(self, W):
        return self.phi(self.psi(W))

    def coords_of(self, Ximgs):
        """Latents whose psi gives these images (exact for pca/exact; Newton for the warp)."""
        Z = self.B.T @ (Ximgs - self.mean[:, None])
        if not self.warp_c: return Z
        W = Z.clone()
        for _ in range(200):                                       # invert w + c tanh(Mw) = Z
            F = W + self.warp_c * torch.tanh(self.M @ W) - Z
            if float(torch.linalg.norm(F)) < 1e-14: break
            W = W - 0.5 * F
        return W


def run(a, kind, k, log=print):
    dev = torch.device(a.device)
    ch = Chart(kind, k, a.n, a.seed, dev, a.data_root, a.n_fit, a.N)
    g = torch.Generator().manual_seed(a.seed + 7)
    N, m, n, r = a.N, a.m, a.n, a.r

    W_true = ch.coords_of(ch.X_real)
    X_img = ch.psi(W_true)                       # what the chart can represent
    chart_err = float((torch.linalg.norm(X_img - ch.X_real, dim=0) /
                       torch.linalg.norm(ch.X_real, dim=0)).median())
    y = ch.y; H = ch.phi(X_img)
    W0 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(dev)
    A0 = (a.sigma0 * torch.randn(r, n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, m, a.T, a.lr, "sgd")

    import torch.func as tf
    U_true, _ = qr_canon(H); nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T); nW = k * N

    def res_vec(v):
        Wc = v[:nW].reshape(k, N); aux = v[nW:].reshape(r, N)
        Bs, Xis, Uc = simulate_sgd_reduced(ch.features_from_latents(Wc), aux, W0, y, m, a.T, a.lr, 0.0)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])

    v0 = torch.cat([W_true.reshape(-1), (A0 @ U_true).reshape(-1)]).detach()
    sv = torch.linalg.svdvals(tf.jacfwd(res_vec)(v0).detach())
    smin, smax = float(sv[-1]), float(sv[0])

    W_init = W_true + a.init_noise * torch.randn(k, N, generator=g).to(dev) * W_true.std()
    with torch.no_grad():
        Uc, _ = qr_canon(ch.features_from_latents(W_init)); Xinit = A_T @ Uc
    args = argparse.Namespace(m=m, T=a.T, lr=a.lr, wd=0.0, release="sgd", seed=a.seed,
                              restarts=a.restarts, restart_noise=0.1, lm_iters=a.lm_iters, lm_lambda=1e-2,
                              lm_scale="identity", stage_x=0, jac="fwd", solver="lm", outer=30, lbfgs_iter=20)
    t0 = time.time()
    W_hat, aux, resid, sec, nrs, diag = invert_lm(ch, A_T, B_T, W0, y, args, W_init, Xinit, log)
    X_hat = ch.psi(W_hat)
    err_chart = (torch.linalg.norm(X_hat - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
    err_real = (torch.linalg.norm(X_hat - ch.X_real, dim=0) / torch.linalg.norm(ch.X_real, dim=0))

    line = m + r - N
    out = dict(chart=kind, k=k, r=r, N=N, m=m, n=n, T=a.T, capacity_line=line, below_line=bool(k < line),
               chart_repr_err=chart_err, jac_sigma_min_truth=smin, jac_sigma_max_truth=smax,
               jac_full_rank_truth=bool(smin > 1e-12 * smax),
               err_vs_chart_max=float(err_chart.max()), err_vs_REAL_max=float(err_real.max()),
               err_vs_REAL_median=float(err_real.median()), residual=resid,
               seed=a.seed, seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(),
               cmd=" ".join(sys.argv), **diag)
    log(json.dumps(out))
    if a.out:
        with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        torch.save(dict(x_real=ch.X_real.cpu(), x_chart=X_img.cpu(), x_hat=X_hat.cpu(),
                        labels=y.cpu(), meta=out),
                   os.path.join(a.save_dir, f"chart_{kind}_k{k}_r{r}_N{N}_s{a.seed}.pth"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", nargs="*", default=["pca", "warped", "exact"])
    ap.add_argument("--ks", type=int, nargs="*", default=[8, 14, 17, 18, 22])
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--m", type=int, default=10); ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=2); ap.add_argument("--lm-iters", type=int, default=80)
    ap.add_argument("--n-fit", type=int, default=10000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(a.n)
    log = (lambda s: None) if a.quiet else (lambda s: print(s, flush=True))
    print(f"# chart dependence  m={a.m} r={a.r} N={a.N} -> line k < {a.m + a.r - a.N}   git={git_hash()}",
          flush=True)
    for kind in a.charts:
        for k in a.ks:
            if kind == "exact" and k < a.N: continue
            print(f"##### chart={kind} k={k}", flush=True)
            run(a, kind, k, log)


if __name__ == "__main__":
    main()
