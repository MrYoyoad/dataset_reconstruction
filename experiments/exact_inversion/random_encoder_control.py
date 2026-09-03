#!/usr/bin/env python3
"""Control B: the same global-vs-local PCA comparison against a RANDOM encoder of the SAME architecture.

On the trained backbone the class-local chart is 135x WORSE conditioned than the global one at k=16 (Step 14),
against the prediction that it would be better.  Both charts have an orthonormal dpsi/dw, so the gap is not
decoder geometry; it is what the encoder does to the directions each chart spans.  Hypothesis: a trained
classifier compresses within-class variation -- exactly where a local chart spends its coordinates.  If so the
gap should largely vanish for an encoder that has never seen a label.  Decisive in both directions:
  gap -> ~1     the effect is a property of TRAINED features ("trained encoders punish the charts that draw best")
  gap survives  the hypothesis is wrong and the span alone explains it.

The random backbone keeps TrainedBackbone's shapes (784 -> 1000 -> 1000 GELU, head 10 x 1000) and matches each
layer's Frobenius norm to the trained checkpoint, so only the weights' training is removed -- also the zero
point of the encoder-quality ladder (random / 78% / 96% / >=97%) at a fixed architecture, which the Step-10
random encoder (n=96 tanh) was not.  Same digits, labels, rank, horizon and recipe as vae_chart.py.

  python -m experiments.exact_inversion.random_encoder_control --ks 16 --charts global local --lm-iters 3000
"""
import argparse, json, math, os, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash, train_release
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.conditional_charts import LocalPCAChart
from experiments.exact_inversion.vae_chart import invert_cell

torch.set_default_dtype(torch.float64)


class RandomBackbone:
    """TrainedBackbone's forward with random Gaussian weights, each layer scaled to the trained layer's norm."""
    def __init__(self, trained, dev, seed, act="gelu"):
        gg = torch.Generator().manual_seed(seed)
        def like(W):
            R = torch.randn(W.shape, generator=gg).to(dev)
            return R * (torch.linalg.norm(W) / torch.linalg.norm(R))
        self.W1, self.b1, self.W2, self.W0 = like(trained.W1), like(trained.b1), like(trained.W2), like(trained.W0)
        self.act = torch.nn.functional.gelu if act == "gelu" else torch.relu
        self.m, self.n = self.W0.shape
        self.norms = dict(W1=float(torch.linalg.norm(self.W1)), b1=float(torch.linalg.norm(self.b1)),
                          W2=float(torch.linalg.norm(self.W2)), W0=float(torch.linalg.norm(self.W0)))
    phi = TrainedBackbone.phi
    logits = TrainedBackbone.logits


def release_ranks(bb, X_train, y, a, gen_state):
    """Per-cell witness for (A4): rank B_T = rank P_T needs N independent accumulated residual trajectories,
       which repeated labels do not guarantee.  Re-draws A0 from the generator state invert_cell consumed and
       re-runs the release (T steps, cheap) to read rank B_T, the graded ratio sigma_N/sigma_1, and rank X."""
    g2 = torch.Generator().set_state(gen_state)
    A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g2)).to(X_train.device)
    A_T, B_T = train_release(bb.phi(X_train), A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
    sB = torch.linalg.svdvals(B_T); sX = torch.linalg.svdvals(A0 @ torch.linalg.qr(bb.phi(X_train))[0])
    return dict(rank_B_T=int((sB > 1e-12 * sB[0]).sum()), B_T_sigma_ratio=float(sB[a.N - 1] / sB[0]),
                B_T_sigma_Np1_over_1=float(sB[a.N] / sB[0]) if sB.numel() > a.N else 0.0,
                rank_X=int((sX > 1e-12 * sX[0]).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--encoder", choices=["random", "trained"], default="random")
    ap.add_argument("--encoder-seed", type=int, default=101)
    ap.add_argument("--ks", nargs="*", type=int, default=[16]); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=4); ap.add_argument("--lm-iters", type=int, default=3000)
    ap.add_argument("--labels", choices=["repeated", "distinct"], default="repeated",
                    help="repeated = vae_chart.py's draw ([0,3,0,3,5,0,1,9] at seed 1: three 0s, two 3s, so the "
                         "local chart gives the three 0-columns an identical map); distinct = first N digits of "
                         "the same permutation with pairwise distinct labels")
    ap.add_argument("--charts", nargs="*", default=["global", "local"])
    ap.add_argument("--cells", nargs="*", default=["a"])
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    trained = TrainedBackbone(a.model, dev, "gelu")
    bb = RandomBackbone(trained, dev, a.encoder_seed) if a.encoder == "random" else trained
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(bb.n)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); ytr_t = torch.tensor(ytr[:a.n_fit], device=dev)
    Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    with torch.no_grad():
        acc = float((bb.logits(Xte_t[:2000].T).argmax(0) == yte_t[:2000]).double().mean())
    line = bb.m + a.r - a.N
    print(f"# encoder={a.encoder}  test acc {acc*100:.2f}%  line k < m+r-N = {line}  ks={a.ks}  charts={a.charts}"
          f"  norms={getattr(bb, 'norms', 'trained')}  git={git_hash()}", flush=True)
    g = torch.Generator().manual_seed(a.seed + 7)               # same private digits as vae_chart.py
    perm = torch.randperm(Xte_t.shape[0], generator=g)
    if a.labels == "repeated":
        idx = perm[:a.N].to(dev)
    else:
        idx, seen = [], set()
        for i in perm.tolist():
            if int(yte[i]) not in seen: idx.append(i); seen.add(int(yte[i]))
            if len(idx) == a.N: break
        idx = torch.tensor(idx, device=dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
    print(f"# labels={a.labels}: {y.tolist()}", flush=True)

    for k in a.ks:
        a.k = k
        charts = {}
        if "global" in a.charts: charts["global"] = PCAChart(Xtr_t, k, dev)
        if "local" in a.charts:  charts["local"] = LocalPCAChart(Xtr_t, ytr_t, k, y, dev)
        for cname in a.charts:
            for cell in a.cells:
                print(f"##### encoder={a.encoder} k={k} chart={cname} cell={cell}", flush=True)
                gs = g.get_state()
                row, X_hat, X_on = invert_cell(charts[cname], bb, X_real, y, a, cell, dev, g, lambda s: None)
                row.update(release_ranks(bb, X_on if cell == "a" else X_real, y, a, gs))
                row.update(encoder=a.encoder, encoder_seed=a.encoder_seed, labels=a.labels, y=y.tolist(),
                           chart=cname, chart_analytic=True,
                           k=k, N=a.N, r=a.r, m=bb.m, n=bb.n, T=a.T, lr=a.lr, seed=a.seed,
                           capacity_line=line, below_line=bool(k < line), backbone_test_acc=acc,
                           git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                print(json.dumps(row), flush=True)
                if a.out:
                    with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
                if a.save_dir:
                    os.makedirs(a.save_dir, exist_ok=True)
                    torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat=X_hat.cpu(), meta=row),
                               os.path.join(a.save_dir, f"{a.encoder}_{cname}_{cell}_k{k}_s{a.seed}.pth"))


if __name__ == "__main__":
    main()
