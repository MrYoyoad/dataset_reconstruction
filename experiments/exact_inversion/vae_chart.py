#!/usr/bin/env python3
"""A LEARNED generator as the chart: does a prior at the same k turn 17 numbers per digit into a picture?

The counting says a prior must parameterise the image class in fewer than m + r - N degrees of freedom per
image.  The PCA chart meets that budget and still returns something half the image away, because a linear
chart of that dimension cannot draw a digit.  This asks whether a chart with the SAME budget but real
drawing power closes the gap -- the attacker-realizable version of the oracle-chart control in Step 11.

  chart      a small VAE decoder, latent k, GELU hidden + sigmoid output (both analytic, so the analytic
             step of the identifiability theorem still holds; no ReLU).  Trained on the TRAIN split only,
             so it is public.  Saved as an artefact next to the results.
  backbone   the same frozen trained 784-1000-1000-10 GELU MNIST MLP as the head-only cell.
  private    N digits from the TEST split, real labels: unseen by the backbone and by the chart.

  cell (a) ON-CHART   the private digits are first encode-decoded, so the truth lies on the chart exactly
                      (as the PCA cells did).  Tests drawing power under exact identifiability: err-vs-chart
                      should reach the floor, and err-vs-REAL is then the chart's own reconstruction error.
  cell (b) OFF-CHART  the adapter is fine-tuned on the RAW digits and the attacker searches the chart anyway.
                      The residual CANNOT reach the floor.  The honest comparison is err-vs-REAL against the
                      chart's own encode-decode of the same digit -- the best that chart could ever do.
                      This is the realistic attacker setting.

  python -m experiments.exact_inversion.vae_chart --k 16 --charts vae pca --cells a b
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.nn as nn, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import (
    train_release, simulate_sgd_reduced, invert_lm, qr_canon, git_hash, RECOVER_TOL)
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx

torch.set_default_dtype(torch.float64)


class VAE(nn.Module):
    """act='gelu' keeps psi ANALYTIC, which is what the generic-sufficiency argument assumes.
       act='relu' deliberately breaks that: the decoder is then piecewise-linear, so the analytic-subvariety
       dichotomy behind a.e. full rank does NOT apply. The capacity COUNT is unaffected (it is a dimension
       bound and needs no analyticity), so the two arms separate the count from the sufficiency argument."""
    def __init__(self, k, h=512, act="gelu"):
        super().__init__()
        A = (lambda: nn.GELU()) if act == "gelu" else (lambda: nn.ReLU())
        self.act_name = act
        self.enc = nn.Sequential(nn.Linear(784, h), A(), nn.Linear(h, h), A())
        self.mu = nn.Linear(h, k); self.lv = nn.Linear(h, k)
        self.dec = nn.Sequential(nn.Linear(k, h), A(), nn.Linear(h, h), A(),
                                 nn.Linear(h, 784), nn.Sigmoid())

    def encode(self, x):                      # x: (B, 784)
        e = self.enc(x); return self.mu(e), self.lv(e)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv


def train_vae(Xtr, k, dev, epochs, bs, lr, seed, act="gelu", log=print):
    torch.manual_seed(seed)
    v = VAE(k, act=act).to(dev).float()
    opt = torch.optim.Adam(v.parameters(), lr=lr)
    X = Xtr.float()
    for ep in range(epochs):
        perm = torch.randperm(X.shape[0], device=dev)
        tot = 0.0
        for i in range(0, X.shape[0], bs):
            xb = X[perm[i:i + bs]]
            xh, mu, lv = v(xb)
            rec = ((xh - xb) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            loss = rec + 1.0 * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * xb.shape[0]
        log(f"    vae epoch {ep+1}/{epochs}  loss {tot/X.shape[0]:.4f}")
    return v.double().eval()


class VAEChart:
    """psi(w) = decoder(w).  coords_of uses the encoder mean, then a short refinement so the
       on-chart truth is as close to the real digit as the decoder allows."""
    def __init__(self, vae, k):
        self.v = vae; self.k = k
    def psi(self, W):                          # (k, N) -> (784, N)
        return self.v.dec(W.T).T
    def coords_of(self, X, iters=300, lr=0.05):
        with torch.no_grad():
            mu, _ = self.v.encode(X.T)
        w = mu.clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(iters):
            opt.zero_grad(); loss = ((self.v.dec(w).T - X) ** 2).sum(); loss.backward(); opt.step()
        return w.detach().T


def invert_cell(chart, bb, X_real, y, a, cell, dev, g, log):
    """cell 'a': release trained on the ON-CHART images. cell 'b': on the RAW digits."""
    m, n = bb.m, bb.n
    W_true = chart.coords_of(X_real)
    X_on = chart.psi(W_true)                                     # the best the chart can do
    chart_repr = float((torch.linalg.norm(X_on - X_real, dim=0) /
                        torch.linalg.norm(X_real, dim=0)).median())
    X_train_on = X_on if cell == "a" else X_real                 # what the ADAPTER is fine-tuned on
    H = bb.phi(X_train_on)
    A0 = (a.sigma0 * torch.randn(a.r, n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, bb.W0, y, m, a.T, a.lr, "sgd")
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T); k = a.k; N = a.N; nW = k * N

    def res_vec(v):
        Wc = v[:nW].reshape(k, N); aux = v[nW:].reshape(a.r, N)
        Bs, Xis, Uc = simulate_sgd_reduced(bb.phi(chart.psi(Wc)), aux, bb.W0, y, m, a.T, a.lr, 0.0)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])

    smin = smax = res_truth = float("nan")
    if cell == "a":                                              # a truth exists in the chart
        U_true, _ = qr_canon(H); v0 = torch.cat([W_true.reshape(-1), (A0 @ U_true).reshape(-1)]).detach()
        sv = torch.linalg.svdvals(tf.jacfwd(res_vec)(v0).detach())
        smin, smax = float(sv[-1]), float(sv[0]); res_truth = float(torch.linalg.norm(res_vec(v0)))

    W_init = W_true + a.init_noise * torch.randn(k, N, generator=g).to(dev) * W_true.std()
    with torch.no_grad():
        Uc, _ = qr_canon(bb.phi(chart.psi(W_init))); Xinit = A_T @ Uc

    class Adapter:
        psi = staticmethod(chart.psi)
        features_from_latents = staticmethod(lambda Wc: bb.phi(chart.psi(Wc)))
    args = argparse.Namespace(m=m, T=a.T, lr=a.lr, wd=0.0, release="sgd", seed=a.seed,
                              restarts=a.restarts, restart_noise=0.1, lm_iters=a.lm_iters, lm_lambda=1e-2,
                              lm_scale="identity", stage_x=0, jac="fwd", solver="lm", outer=30, lbfgs_iter=20)
    t0 = time.time()
    W_hat, aux, resid, sec, nrs, diag = invert_lm(Adapter, A_T, B_T, bb.W0, y, args, W_init, Xinit, log)
    X_hat = chart.psi(W_hat)
    e_chart = (torch.linalg.norm(X_hat - X_on, dim=0) / torch.linalg.norm(X_on, dim=0))
    e_real = (torch.linalg.norm(X_hat - X_real, dim=0) / torch.linalg.norm(X_real, dim=0))
    return dict(cell=cell, chart_repr_err=chart_repr, jac_sigma_min_truth=smin, jac_sigma_max_truth=smax,
                res_at_truth=res_truth, residual=resid,
                err_vs_chart_max=float(e_chart.max()), err_vs_REAL_max=float(e_real.max()),
                err_vs_REAL_median=float(e_real.median()),
                best_possible_vs_REAL=chart_repr, seconds=time.time() - t0, **diag), X_hat, X_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=3); ap.add_argument("--lm-iters", type=int, default=200)
    ap.add_argument("--charts", nargs="*", default=["vae", "pca"])
    ap.add_argument("--cells", nargs="*", default=["a", "b"])
    ap.add_argument("--vae-epochs", type=int, default=12); ap.add_argument("--vae-bs", type=int, default=256)
    ap.add_argument("--vae-lr", type=float, default=1e-3)
    ap.add_argument("--vae-act", choices=["gelu", "relu"], default="gelu",
                    help="gelu keeps the chart analytic; relu breaks it (see VAE docstring)")
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    bb = TrainedBackbone(a.model, dev, "gelu")
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(bb.n)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev)
    yte_t = torch.tensor(yte, device=dev)
    with torch.no_grad():
        acc = float((bb.logits(Xte_t[:2000].T).argmax(0) == yte_t[:2000]).double().mean())
    line = bb.m + a.r - a.N
    print(f"# learned-chart cell  backbone acc {acc*100:.2f}%  k={a.k}  line k < m+r-N = {line}  "
          f"({'below' if a.k < line else 'AT/ABOVE'})  vae_act={a.vae_act}"
          f"{'  [NON-ANALYTIC chart: generic-sufficiency argument does not apply; the count still does]' if a.vae_act=='relu' else ''}"
          f"  git={git_hash()}", flush=True)
    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]

    charts = {}
    if "pca" in a.charts: charts["pca"] = PCAChart(Xtr_t, a.k, dev)
    if "vae" in a.charts:
        print("# training the VAE chart on the TRAIN split only (public)", flush=True)
        vae = train_vae(Xtr_t, a.k, dev, a.vae_epochs, a.vae_bs, a.vae_lr, a.seed, a.vae_act)
        charts["vae"] = VAEChart(vae, a.k)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            torch.save(vae.state_dict(),
                       os.path.join(a.save_dir, f"vae_{a.vae_act}_decoder_k{a.k}_s{a.seed}.pth"))

    for cname in a.charts:
        ch = charts[cname]
        for cell in a.cells:
            print(f"##### chart={cname} cell={cell}", flush=True)
            row, X_hat, X_on = invert_cell(ch, bb, X_real, y, a, cell, dev, g, lambda s: None)
            row.update(chart=(cname if cname != "vae" else f"vae_{a.vae_act}"),
                       chart_analytic=bool(cname != "vae" or a.vae_act == "gelu"), k=a.k, N=a.N, r=a.r, m=bb.m, n=bb.n, T=a.T, lr=a.lr, seed=a.seed,
                       capacity_line=line, below_line=bool(a.k < line), backbone_test_acc=acc,
                       git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
            if a.save_dir:
                os.makedirs(a.save_dir, exist_ok=True)
                torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat=X_hat.cpu(), meta=row),
                           os.path.join(a.save_dir, f"{cname}_{cell}_k{a.k}_s{a.seed}.pth"))


if __name__ == "__main__":
    main()
