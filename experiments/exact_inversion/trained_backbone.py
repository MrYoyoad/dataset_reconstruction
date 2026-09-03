#!/usr/bin/env python3
"""The capacity line with a TRAINED backbone and UNSEEN data.

Every earlier cell used a RANDOM public encoder and a RANDOM W0.  The user asked the obvious question --
are these reconstructions from a LoRA adapter over a *trained* model, on data it has never seen? -- and the
honest answer was: LoRA yes, unseen yes, trained no.  This closes that gap.

  backbone   the repo's TRAINED 10-class MNIST MLP (784-1000-1000-10, GELU), frozen.
             phi := the penultimate activations, so n = 1000 and the features are learned, not random.
             W0  := the trained output layer, so the frozen head is a real classifier head, m = 10.
  adapter    W_eff = W0 + B A on that head; A random r x n, B_0 = 0, plain SGD, T = 400, eta = 0.01.
  private    N = 8 real digits from the TEST split -- never seen by the backbone -- with their real labels,
             on a PCA chart built from the TRAIN split, so the chart is public and the digits are unseen.

The theorems say the line must still sit at k < m + r - N = 18: training the head changes only W0 and
training the encoder changes only H, and the count sees neither.  If it moves, the law is weaker than
stated.

A GATE runs first: the reassembled backbone must actually classify MNIST.  If the architecture has been
put together wrongly the accuracy will say so before any inversion number is produced.

  python -m experiments.exact_inversion.trained_backbone --ks 6 10 14 16 17 18 20 22
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import (
    train_release, simulate_sgd_reduced, invert_lm, qr_canon, git_hash, RECOVER_TOL)

torch.set_default_dtype(torch.float64)


def read_idx(root, split):
    d = os.path.join(root, "MNIST", "raw")
    ip = "train-images-idx3-ubyte" if split == "train" else "t10k-images-idx3-ubyte"
    lp = "train-labels-idx1-ubyte" if split == "train" else "t10k-labels-idx1-ubyte"
    with open(os.path.join(d, ip), "rb") as f:
        f.read(16); img = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
    with open(os.path.join(d, lp), "rb") as f:
        f.read(8); lab = np.frombuffer(f.read(), dtype=np.uint8)
    return img.astype(np.float64) / 255.0, lab.astype(np.int64)


class TrainedBackbone:
    """Frozen trained MLP: phi = penultimate activations (n=1000); W0 = the trained output layer."""
    def __init__(self, path, dev, act="gelu"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        # the checkpoint wraps the tensors under "state_dict" (alongside batch/epoch); an earlier version
        # indexed the top level and died with a KeyError.
        sd = blob["state_dict"] if "state_dict" in blob else blob
        self.W1 = sd["layers.0.weight"].to(dev).double()
        self.b1 = sd["layers.0.bias"].to(dev).double()
        self.W2 = sd["layers.1.weight"].to(dev).double()
        self.W0 = sd["layers.2.weight"].to(dev).double()      # the frozen head -> m x n
        self.act = torch.nn.functional.gelu if act == "gelu" else torch.relu
        self.m, self.n = self.W0.shape

    def phi(self, X):                                          # (784, B) -> (n, B)
        return self.act(self.W2 @ self.act(self.W1 @ X + self.b1[:, None]))

    def logits(self, X):
        return self.W0 @ self.phi(X)


class PCAChart:
    def __init__(self, Xtrain, k, dev):
        self.mean = Xtrain.mean(0)
        U, S, Vh = torch.linalg.svd(Xtrain - self.mean, full_matrices=False)
        self.V = Vh[:k].T.contiguous(); self.k = k
        self.explained = float((S[:k] ** 2).sum() / (S ** 2).sum())

    def psi(self, W): return self.mean[:, None] + self.V @ W
    def coords_of(self, X): return self.V.T @ (X - self.mean[:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--act", default="gelu")
    ap.add_argument("--ks", type=int, nargs="*", default=[6, 10, 14, 16, 17, 18, 20, 22])
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=2); ap.add_argument("--lm-iters", type=int, default=80)
    ap.add_argument("--n-fit", type=int, default=50000,
                    help="train images for the PCA chart; 50000 = the split the backbone was trained on")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    log = (lambda s: None) if a.quiet else (lambda s: print(s, flush=True))

    bb = TrainedBackbone(a.model, dev, a.act)
    m, n = bb.m, bb.n
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(n)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev)
    yte_t = torch.tensor(yte, device=dev)

    # ---- GATE: does the reassembled frozen backbone actually classify MNIST? ----
    with torch.no_grad():
        pred = bb.logits(Xte_t[:2000].T).argmax(0)
        acc = float((pred == yte_t[:2000]).double().mean())
    print(f"# TRAINED backbone {a.model}  m={m} n={n} act={a.act}", flush=True)
    # Gate threshold: chance is 10%. The repo's checkpoints are trained for the RECONSTRUCTION setting --
    # small balanced subsets, tiny init -- so they are genuinely learned but WEAK classifiers, not
    # full-MNIST models. This one is 78.45%, verified against the repo's own NeuralNetwork class to 3.7e-14
    # (job 607801), so a 90% threshold was miscalibrated rather than the code being wrong. What the gate is
    # for is catching a mis-assembled architecture, which would sit near chance.
    print(f"# GATE  test accuracy of the frozen backbone = {acc*100:.2f}%   "
          f"{'PASS (genuinely learned features)' if acc > 0.50 else 'FAIL -- near chance, architecture wrong'}",
          flush=True)
    if acc <= 0.50: return
    print(f"# line k < m + r - N = {m + a.r - a.N};  private data are TEST-split digits, unseen by the backbone",
          flush=True)

    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
    for k in a.ks:
        chart = PCAChart(Xtr_t, k, dev)
        W_true = chart.coords_of(X_real); X_img = chart.psi(W_true)
        chart_err = float((torch.linalg.norm(X_img - X_real, dim=0) /
                           torch.linalg.norm(X_real, dim=0)).median())
        H = bb.phi(X_img)
        A0 = (a.sigma0 * torch.randn(a.r, n, generator=g)).to(dev)
        A_T, B_T = train_release(H, A0, bb.W0, y, m, a.T, a.lr, "sgd")

        class W:            # adapter object for the shared solver
            psi = staticmethod(chart.psi)
            features_from_latents = staticmethod(lambda Wc: bb.phi(chart.psi(Wc)))
        U_true, _ = qr_canon(H); nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T); nW = k * a.N

        def res_vec(v):
            Wc = v[:nW].reshape(k, a.N); aux = v[nW:].reshape(a.r, a.N)
            Bs, Xis, Uc = simulate_sgd_reduced(W.features_from_latents(Wc), aux, bb.W0, y, m, a.T, a.lr, 0.0)
            return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])

        v0 = torch.cat([W_true.reshape(-1), (A0 @ U_true).reshape(-1)]).detach()
        sv = torch.linalg.svdvals(tf.jacfwd(res_vec)(v0).detach())
        smin, smax = float(sv[-1]), float(sv[0]); res_truth = float(torch.linalg.norm(res_vec(v0)))

        W_init = W_true + a.init_noise * torch.randn(k, a.N, generator=g).to(dev) * W_true.std()
        with torch.no_grad():
            Uc, _ = qr_canon(W.features_from_latents(W_init)); Xinit = A_T @ Uc
        args = argparse.Namespace(m=m, T=a.T, lr=a.lr, wd=0.0, release="sgd", seed=a.seed,
                                  restarts=a.restarts, restart_noise=0.1, lm_iters=a.lm_iters,
                                  lm_lambda=1e-2, lm_scale="identity", stage_x=0, jac="fwd", solver="lm",
                                  outer=30, lbfgs_iter=20)
        t0 = time.time()
        W_hat, aux, resid, sec, nrs, diag = invert_lm(W, A_T, B_T, bb.W0, y, args, W_init, Xinit, log)
        X_hat = chart.psi(W_hat)
        err_chart = (torch.linalg.norm(X_hat - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
        err_real = (torch.linalg.norm(X_hat - X_real, dim=0) / torch.linalg.norm(X_real, dim=0))
        line = m + a.r - a.N
        out = dict(setting="trained_backbone_unseen_test_digits", model=a.model, act=a.act,
                   backbone_test_acc=acc, k=k, r=a.r, N=a.N, m=m, n=n, T=a.T, lr=a.lr, seed=a.seed,
                   capacity_line=line, below_line=bool(k < line), chart_repr_err=chart_err,
                   pca_explained_var=chart.explained, labels=[int(x) for x in y.cpu()],
                   jac_sigma_min_truth=smin, jac_sigma_max_truth=smax, res_at_truth=res_truth,
                   jac_full_rank_truth=bool(smin > 1e-12 * smax),
                   err_vs_chart_max=float(err_chart.max()), err_vs_REAL_max=float(err_real.max()),
                   err_vs_REAL_median=float(err_real.median()),
                   frac_recovered=float((err_chart < RECOVER_TOL).double().mean()),
                   residual=resid, seconds=time.time() - t0, git=git_hash(),
                   host=socket.gethostname(), cmd=" ".join(sys.argv), **diag)
        print(json.dumps(out), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            torch.save(dict(x_real=X_real.cpu(), x_chart=X_img.cpu(), x_hat=X_hat.cpu(),
                            labels=y.cpu(), meta=out),
                       os.path.join(a.save_dir, f"trained_k{k}_r{a.r}_N{a.N}_s{a.seed}.pth"))


if __name__ == "__main__":
    main()
