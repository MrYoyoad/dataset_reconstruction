#!/usr/bin/env python3
"""Spend the k coordinates on WITHIN-CLASS variation: global vs class-local charts at the same budget.

The capacity count caps the chart's DIMENSION, not its quality.  So at a fixed k the attacker should not
waste coordinates telling a 3 from an 8 -- the label is already known (it is part of the conditioning the
attacker holds everywhere in this study, and R3b showed a wrong label assignment cannot reach the residual
floor, so it is recoverable rather than assumed).  Spend them on how THIS class varies instead.

Three charts, all k = 16, same trained backbone, same 8 unseen test digits:
  global   psi(w)      = mu + V_k w                     principal directions of ALL digits   [control]
  local    psi_y(w)    = mu_y + V_{y,k} w               fitted on the TRAIN split of class y only
  cvae     psi(w, y)   = decoder([w, onehot(y)])        class-conditional VAE, GELU + sigmoid (analytic)

Two readings:
  (i)  err-vs-REAL at fixed k across the three -- prediction: global >> local > cvae;
  (ii) sigma_min at the truth per chart -- a class-local chart should be better CONDITIONED, and
       conditioning is what the solver is fighting on a trained backbone (2.06e-6 there against 2.04e-4
       with a random encoder at the same k).

Labels: GIVEN, as everywhere else here; R3b (job 484255) is the evidence they are recoverable from the
release rather than needing to be assumed.

  python -m experiments.exact_inversion.conditional_charts --k 16 --charts global local cvae --cells a b
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.nn as nn

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.vae_chart import invert_cell

torch.set_default_dtype(torch.float64)


class LocalPCAChart:
    """One PCA chart per class; column i uses the chart of its own label. psi stays linear per column."""
    def __init__(self, Xtr, ytr, k, labels, dev):
        self.k = k; self.labels = labels
        self.mean = {}; self.V = {}
        for c in sorted(set(int(v) for v in labels.cpu())):
            Xc = Xtr[ytr == c]
            mu = Xc.mean(0); self.mean[c] = mu
            _, S, Vh = torch.linalg.svd(Xc - mu, full_matrices=False)
            self.V[c] = Vh[:k].T.contiguous()
        self.M = torch.stack([self.mean[int(c)] for c in labels], dim=1)          # (784, N)
        self.Vs = torch.stack([self.V[int(c)] for c in labels], dim=0)            # (N, 784, k)

    def psi(self, W):                       # (k, N) -> (784, N)
        return self.M + torch.einsum("npk,kn->pn", self.Vs, W)

    def coords_of(self, X):
        return torch.einsum("npk,pn->kn", self.Vs, X - self.M)


class CondVAE(nn.Module):
    def __init__(self, k, n_cls=10, h=512):
        super().__init__()
        self.k = k; self.n_cls = n_cls
        self.enc = nn.Sequential(nn.Linear(784 + n_cls, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.mu = nn.Linear(h, k); self.lv = nn.Linear(h, k)
        self.dec = nn.Sequential(nn.Linear(k + n_cls, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, 784), nn.Sigmoid())

    def encode(self, x, oh): 
        e = self.enc(torch.cat([x, oh], 1)); return self.mu(e), self.lv(e)

    def decode(self, z, oh): return self.dec(torch.cat([z, oh], 1))


class CondVAEChart:
    def __init__(self, cvae, k, labels, dev):
        self.v = cvae; self.k = k
        self.oh = torch.eye(cvae.n_cls, device=dev)[labels]

    def psi(self, W): return self.v.decode(W.T, self.oh).T

    def coords_of(self, X, iters=400, lr=0.05):
        with torch.no_grad():
            mu, _ = self.v.encode(X.T, self.oh)
        w = mu.clone().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(iters):
            opt.zero_grad(); ((self.v.decode(w, self.oh).T - X) ** 2).sum().backward(); opt.step()
        return w.detach().T


def train_cvae(Xtr, ytr, k, dev, epochs, bs, lr, seed, log=print):
    torch.manual_seed(seed)
    v = CondVAE(k).to(dev).float()
    opt = torch.optim.Adam(v.parameters(), lr=lr)
    X = Xtr.float(); OH = torch.eye(10, device=dev)[ytr].float()
    for ep in range(epochs):
        perm = torch.randperm(X.shape[0], device=dev); tot = 0.0
        for i in range(0, X.shape[0], bs):
            j = perm[i:i + bs]; xb, ob = X[j], OH[j]
            mu, lv = v.encode(xb, ob)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            xh = v.decode(z, ob)
            loss = ((xh - xb) ** 2).sum(1).mean() + (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * xb.shape[0]
        log(f"    cvae epoch {ep+1}/{epochs}  loss {tot/X.shape[0]:.4f}")
    return v.double().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=3); ap.add_argument("--lm-iters", type=int, default=200)
    ap.add_argument("--charts", nargs="*", default=["global", "local", "cvae"])
    ap.add_argument("--cells", nargs="*", default=["a", "b"])
    ap.add_argument("--vae-epochs", type=int, default=12); ap.add_argument("--vae-bs", type=int, default=256)
    ap.add_argument("--vae-lr", type=float, default=1e-3)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    bb = TrainedBackbone(a.model, dev, "gelu")
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(bb.n)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); ytr_t = torch.tensor(ytr[:a.n_fit], device=dev)
    Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    with torch.no_grad():
        acc = float((bb.logits(Xte_t[:2000].T).argmax(0) == yte_t[:2000]).double().mean())
    line = bb.m + a.r - a.N
    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
    print(f"# conditional charts  backbone acc {acc*100:.2f}%  k={a.k}  line k<{line}  labels GIVEN "
          f"(R3b shows them recoverable)  digit labels {[int(v) for v in y.cpu()]}  git={git_hash()}",
          flush=True)

    charts = {}
    if "global" in a.charts: charts["global"] = PCAChart(Xtr_t, a.k, dev)
    if "local" in a.charts:  charts["local"] = LocalPCAChart(Xtr_t, ytr_t, a.k, y, dev)
    if "cvae" in a.charts:
        print("# training the class-conditional VAE on the TRAIN split only (public)", flush=True)
        cv = train_cvae(Xtr_t, ytr_t, a.k, dev, a.vae_epochs, a.vae_bs, a.vae_lr, a.seed)
        charts["cvae"] = CondVAEChart(cv, a.k, y, dev)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            torch.save(cv.state_dict(), os.path.join(a.save_dir, f"cvae_k{a.k}_s{a.seed}.pth"))

    for cname in a.charts:
        for cell in a.cells:
            print(f"##### chart={cname} cell={cell}", flush=True)
            row, X_hat, X_on = invert_cell(charts[cname], bb, X_real, y, a, cell, dev, g, lambda s: None)
            row.update(chart=cname, chart_kind=("global linear" if cname == "global" else
                                                "class-local linear" if cname == "local" else
                                                "class-conditional VAE"),
                       k=a.k, N=a.N, r=a.r, m=bb.m, n=bb.n, T=a.T, lr=a.lr, seed=a.seed,
                       capacity_line=line, below_line=bool(a.k < line), backbone_test_acc=acc,
                       labels=[int(v) for v in y.cpu()], labels_given=True,
                       git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
            if a.save_dir:
                os.makedirs(a.save_dir, exist_ok=True)
                torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat=X_hat.cpu(), meta=row),
                           os.path.join(a.save_dir, f"cond_{cname}_{cell}_k{a.k}_s{a.seed}.pth"))


if __name__ == "__main__":
    main()
