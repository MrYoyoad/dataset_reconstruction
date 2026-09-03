#!/usr/bin/env python3
"""Does the capacity law k < m + r - N hold on REAL images?

The law was derived and measured on a synthetic tanh manifold.  This is the first test on real data.

Translation to MNIST, with the reason for each choice:
  psi  the image manifold.  Real MNIST restricted to its own k-dimensional PCA subspace, so the "manifold
       coordinates" are genuine principal components of digits and k is a real intrinsic dimension rather
       than a knob.  psi(w) = mean + V_k w, differentiable and exact.
  phi  the frozen PUBLIC encoder, a fixed random tanh network 784 -> n, the same family as the synthetic
       testbed so the two are comparable.  It stands in for a public feature extractor.
  head m = 10 classes, the real MNIST labels.  LoRA rank r, B_0 = 0, plain SGD -- the recipe the law is
       stated for.

Why MNIST is the sharp test: with m = 10 and N = 8 the line sits at k < m + r - N = 18 for r = 16, and
CLAUDE.md records MNIST's intrinsic dimension as roughly 10-20. So the predicted boundary falls INSIDE the
range of real digit dimensionality, and the sweep straddles it with real images on both sides.

Prediction: recovery for k < m + r - N, collapse (sigma_min at the truth falling to the FP64 floor, and
release-consistent reconstructions that are not the truth) for k >= m + r - N; and the line must MOVE with
r, so the r-sweep is the control against a fixed-k explanation.

Per the repo's output rules this saves image tensors and a visual grid, not only numbers.
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch

from experiments.exact_inversion.lora_exact_inversion import (
    train_release, simulate_sgd_reduced, invert_lm, qr_canon, git_hash, RECOVER_TOL)

torch.set_default_dtype(torch.float64)


def load_mnist(root, n_fit=10000):
    """Read the raw idx files directly -- no torchvision transform pipeline, no download."""
    d = os.path.join(root, "MNIST", "raw")
    with open(os.path.join(d, "train-images-idx3-ubyte"), "rb") as f:
        f.read(16); img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
    with open(os.path.join(d, "train-labels-idx1-ubyte"), "rb") as f:
        f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
    return img[:n_fit].astype(np.float64) / 255.0, lab[:n_fit].astype(np.int64)


class MnistPCAWorld:
    """psi = PCA decoder (real digit subspace); phi = frozen public tanh encoder."""
    def __init__(self, k, n, seed, device, root, n_fit=10000):
        X, y = load_mnist(root, n_fit)
        self.mean = torch.tensor(X.mean(0), device=device)
        Xc = torch.tensor(X, device=device) - self.mean
        # economy SVD -> principal directions of real MNIST
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        self.V = Vh[:k].T.contiguous()                       # (784, k)
        self.sv = S[:k].clone()
        self.k, self.P, self.n = k, 784, n
        g = torch.Generator().manual_seed(seed)
        self.E1 = (torch.randn(128, 784, generator=g) / math.sqrt(784)).to(device)
        self.E2 = (torch.randn(n, 128, generator=g) / math.sqrt(128)).to(device)
        self.X_all = torch.tensor(X, device=device); self.y_all = torch.tensor(y, device=device)
        self.explained = float((S[:k] ** 2).sum() / (S ** 2).sum())

    def psi(self, W):                    # (k, N) -> (784, N)
        return self.mean[:, None] + self.V @ W

    def phi(self, Ximg):
        return torch.tanh(self.E2 @ torch.tanh(self.E1 @ Ximg))

    def features_from_latents(self, W):
        return self.phi(self.psi(W))

    def project(self, Ximgs):            # real digits -> exact manifold coordinates
        return self.V.T @ (Ximgs - self.mean[:, None])


def run(a, k, r, log=print):
    dev = torch.device(a.device)
    world = MnistPCAWorld(k, a.n, a.seed, dev, a.data_root, a.n_fit)
    g = torch.Generator().manual_seed(a.seed + 7)
    N, m, n = a.N, a.m, a.n

    # ---- private data: N REAL MNIST digits, restricted to the k-dim principal subspace ----
    idx = torch.randperm(world.X_all.shape[0], generator=g)[:N].to(dev)
    X_real = world.X_all[idx].T.contiguous()                 # (784, N) genuine digits
    W_true = world.project(X_real)                           # exact manifold coordinates
    X_img = world.psi(W_true)                                # the digits as the manifold represents them
    proj_err = float((torch.linalg.norm(X_img - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)).median())
    y = world.y_all[idx].to(dev)                             # the REAL labels
    H = world.phi(X_img)

    W0 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(dev)
    A0 = (a.sigma0 * torch.randn(r, n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, m, a.T, a.lr, "sgd")

    # ---- identifiability at the truth ----
    import torch.func as tf
    U_true, _ = qr_canon(H); nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T); nW = k * N

    def res_vec(v):
        Wc = v[:nW].reshape(k, N); aux = v[nW:].reshape(r, N)
        Hc = world.features_from_latents(Wc)
        Bs, Xis, Uc = simulate_sgd_reduced(Hc, aux, W0, y, m, a.T, a.lr, 0.0)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])

    v0 = torch.cat([W_true.reshape(-1), (A0 @ U_true).reshape(-1)]).detach()
    J = tf.jacfwd(res_vec)(v0).detach(); sv = torch.linalg.svdvals(J)
    smin, smax = float(sv[-1]), float(sv[0])
    res_truth = float(torch.linalg.norm(res_vec(v0)))

    # ---- the attack: start 10% off in manifold coordinates ----
    W_init = W_true + a.init_noise * torch.randn(k, N, generator=g).to(dev) * W_true.std()
    with torch.no_grad():
        Uc, _ = qr_canon(world.features_from_latents(W_init)); Xinit = A_T @ Uc
    args = argparse.Namespace(m=m, T=a.T, lr=a.lr, wd=0.0, release="sgd", seed=a.seed,
                              restarts=a.restarts, restart_noise=0.1, lm_iters=a.lm_iters,
                              lm_lambda=1e-2, lm_scale="identity", stage_x=0, jac="fwd", solver="lm",
                              outer=30, lbfgs_iter=20)
    t0 = time.time()
    W_hat, aux, resid, sec, nrs, diag = invert_lm(world, A_T, B_T, W0, y, args, W_init, Xinit, log)
    X_hat = world.psi(W_hat)
    err = (torch.linalg.norm(X_hat - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))

    line = m + r - N
    out = dict(dataset="mnist", k=k, r=r, N=N, m=m, n=n, T=a.T, lr=a.lr, seed=a.seed,
               capacity_line=line, below_line=bool(k < line), pca_explained_var=world.explained,
               pca_proj_err_median=proj_err, labels=[int(x) for x in y.cpu()],
               jac_sigma_min_truth=smin, jac_sigma_max_truth=smax, res_at_truth=res_truth,
               jac_full_rank_truth=bool(smin > 1e-12 * smax),
               start_err_median=float((torch.linalg.norm(world.psi(W_init) - X_img, dim=0) /
                                       torch.linalg.norm(X_img, dim=0)).median()),
               final_err_max=float(err.max()), final_err_median=float(err.median()),
               frac_recovered=float((err < RECOVER_TOL).double().mean()), residual=resid,
               seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv),
               **diag)
    log(json.dumps(out))
    if a.out:
        with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        torch.save(dict(x_real=X_real.cpu(), x_true=X_img.cpu(), x_init=world.psi(W_init).cpu(),
                        x_hat=X_hat.cpu(), labels=y.cpu(), meta=out),
                   os.path.join(a.save_dir, f"mnist_k{k}_r{r}_N{N}_T{a.T}_s{a.seed}.pth"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="*", default=[6, 10, 14, 17, 18, 22, 26])
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
    print(f"# MNIST capacity test  m={a.m} r={a.r} N={a.N}  =>  line k < m+r-N = {a.m + a.r - a.N}\n"
          f"# git={git_hash()} host={socket.gethostname()}", flush=True)
    for k in a.ks:
        line = a.m + a.r - a.N
        print(f"##### k={k}  ({'BELOW' if k < line else 'AT/ABOVE'} the line {line})", flush=True)
        run(a, k, a.r, log)


if __name__ == "__main__":
    main()
