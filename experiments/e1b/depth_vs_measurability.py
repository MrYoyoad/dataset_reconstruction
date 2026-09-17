#!/usr/bin/env python3
"""At what DEPTH does the identifiability question stop being measurable? Isolating depth as the cause.

Two results sit either side of a gap in the evidence:

  * a 3-layer trained network: the law holds 7/7, ranks are sharp, gaps at the cut are effectively infinite
    (job 355535);
  * a 15-layer trained network: no well-defined rank at all -- gap at the cut 1.35 and 1.05, the count sliding
    by ~110 across ten decades of threshold (job 355750).

They differ in depth, but also in width, rank, dataset and training budget, so depth is not isolated. This sweeps
DEPTH ALONE with everything else fixed, and reports the diagnostic that decides measurability rather than the
nullity, which is the quantity that stops existing.

The reported quantity is `gap_at_cut` and the rank ladder. The question is not "does the law hold" -- that is only
askable where a rank exists -- but "at what depth does a rank stop existing", which is the precondition.

Everything is small enough that the Jacobian is exact and the whole sweep is minutes, because the question is
about depth and not about scale.

  python -m experiments.e1b.depth_vs_measurability
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from experiments.cifar.cifar_newclass import load_cifar10, load_cifar100_class
from experiments.utils.identifiability import identifiability, assert_chart_contains, block_normalised_residual, rank_ladder
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)


def log(s): print(s, flush=True)


class DeepNet(nn.Module):
    """Depth is the ONLY thing that varies: width, head width, dataset and budget are fixed across the sweep."""
    def __init__(s, depth, width=256, n=32, m=10):
        super().__init__()
        assert depth >= 2
        dims = [3072] + [width] * (depth - 2) + [n]
        s.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        s.head = nn.Linear(n, m, bias=False)

    def phi(s, x):
        h = x.reshape(len(x), -1)
        for lay in s.layers: h = F.gelu(lay(h))
        return h

    def forward(s, x): return s.head(s.phi(x))


def train(root, dev, depth, epochs, seed, width, n):
    Xtr_np, ytr_np, _, _ = load_cifar10(root)
    Xtr = torch.tensor(Xtr_np, dtype=torch.float64, device=dev); ytr = torch.tensor(ytr_np, device=dev)
    torch.manual_seed(seed)
    net = DeepNet(depth, width, n).to(dev).double()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        for i in torch.randperm(len(Xtr), device=dev).split(256):
            opt.zero_grad(); F.cross_entropy(net(Xtr[i]), ytr[i]).backward(); opt.step()
    with torch.no_grad():
        acc = float((net(Xtr[:5000]).argmax(1) == ytr[:5000]).double().mean())
    for p in net.parameters(): p.requires_grad_(False)
    return net, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--width", type=int, default=256); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--m", type=int, default=11); ap.add_argument("--r", type=int, default=12)
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--T", type=int, default=20); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs-per-depth", type=int, default=0,
                    help="if >0, train for epochs + epochs_per_depth*(depth-2) so deeper nets are not simply "
                         "less trained -- depth and trainedness are otherwise confounded")
    ap.add_argument("--cls", default="motorcycle"); ap.add_argument("--data-root", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/e1b/depth_vs_measurability.jsonl")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cap = a.m + a.r - a.N - 1
    pred = a.N * max(0, a.k - cap)
    log(f"# width={a.width} n={a.n} m={a.m} r={a.r} N={a.N} k={a.k} T={a.T}  cap={cap}, law predicts nullity {pred}")
    log(f"# DEPTH is the only variable. The reported quantity is GAP AT CUT: it decides whether a rank EXISTS.\n")

    pool, cname = load_cifar100_class(a.data_root, a.cls)
    Pub = torch.tensor(pool["train"], dtype=torch.float64, device=dev)
    Pri = torch.tensor(pool["test"], dtype=torch.float64, device=dev)
    g = np.random.RandomState(a.seed)
    X = Pri[g.permutation(len(Pri))[: a.N]].T
    mean = Pub.mean(0); _, _, Vh = torch.linalg.svd(Pub - mean, full_matrices=False)
    V = Vh[: a.k].T.contiguous()
    psi = lambda W: mean[:, None] + V @ W
    Won = V.T @ (X - mean[:, None]); X_on = psi(Won)
    assert_chart_contains(psi, Won, X_on)

    log(f"{'depth':>6} {'ep':>4} {'train acc':>10} {'rank':>7} {'nullity':>8} {'pred':>6} {'GAP@cut':>11} "
        f"{'cond':>11} {'spread':>7}  {'verdict':>12}")
    rows = []
    for depth in a.depths:
        ep = a.epochs + a.epochs_per_depth * (depth - 2)
        net, acc = train(a.data_root, dev, depth, ep, a.seed, a.width, a.n)
        W0 = torch.cat([net.head.weight.double(), torch.zeros(1, a.n, dtype=torch.float64, device=dev)], 0)
        y = torch.full((a.N,), a.m - 1, device=dev)
        Y = torch.eye(a.m, device=dev, dtype=torch.float64)[y].T
        A0 = (1.0 / math.sqrt(a.n) * torch.randn(a.r, a.n, dtype=torch.float64,
                                                 generator=torch.Generator().manual_seed(a.seed + 7))).to(dev)
        phi = lambda Xp: net.phi(Xp.T).T

        def roll(Hc, A0c):
            A, B = A0c, torch.zeros(a.m, a.r, dtype=torch.float64, device=dev)
            for _ in range(a.T):
                z = W0 @ Hc + B @ (A @ Hc); D = (torch.softmax(z, 0) - Y) / a.N
                B, A = B - a.lr * (D @ (A @ Hc).T), A - a.lr * (B.T @ D @ Hc.T)
            return B, A

        B_T, A_T = roll(phi(X_on), A0)
        res = block_normalised_residual(lambda Wc, A0c: roll(phi(psi(Wc)), A0c), (B_T, A_T))
        r_ = identifiability(res, (Won, A0), mode="fwd")
        lad, spread = rank_ladder(r_)
        # THE GAP is the criterion: a cut sitting in a gap of many orders IS a rank cut. The ladder spread is
        # secondary information -- it can be moderate even with a large gap, because directions far above the cut
        # also thin out with the threshold. An earlier version required spread <= 4 and mislabelled every row.
        measurable = r_.gap_at_cut > 1e3
        verdict = "rank EXISTS" if measurable else "NO RANK"
        log(f"{depth:>6} {ep:>4} {acc*100:>9.1f}% {r_.rank:>7} {r_.nullity:>8} {pred:>6} {r_.gap_at_cut:>11.3e} "
            f"{r_.condition_number:>11.3e} {spread:>7} {verdict:>13}")
        rows.append(dict(depth=depth, epochs=ep, train_acc=acc, nullity=r_.nullity, predicted=pred,
                         gap_at_cut=r_.gap_at_cut, condition_number=r_.condition_number,
                         rank_ladder=lad, ladder_spread=spread, rank_exists=bool(measurable),
                         hit=bool(r_.nullity == pred and measurable), n_unknowns=r_.n_unknowns,
                         n_equations=r_.n_equations, rank=r_.rank))
    ok = [x["depth"] for x in rows if x["rank_exists"]]
    bad = [x["depth"] for x in rows if not x["rank_exists"]]
    log(f"\n# rank well defined at depths {ok or 'none'};  NOT well defined at {bad or 'none'}")
    accs = {x['depth']: x['train_acc'] for x in rows}
    if ok and bad:
        log(f"# CONFOUND CHECK -- train accuracy at the boundary: depth {max(ok)} = {accs[max(ok)]*100:.1f}%, "
            f"depth {min(bad)} = {accs[min(bad)]*100:.1f}%. If accuracy falls with depth, depth and TRAINEDNESS "
            f"are confounded and this boundary is not attributable to depth alone.")
        log(f"# The identifiability question stops being measurable between depth {max(ok)} and {min(bad)} "
            f"at this width and budget -- depth ISOLATED, everything else held fixed.")
    elif not bad:
        log("# A rank exists at every depth swept: depth alone does not destroy measurability here, so the "
            "15-layer collapse must be attributed to something else (width, budget, or the release itself).")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(part="depth_vs_measurability", precision="fp64", width=a.width, n=a.n, m=a.m,
                                 r=a.r, N=a.N, k=a.k, T=a.T, cap=cap, predicted=pred, cls=cname, cells=rows,
                                 git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))) + "\n")


if __name__ == "__main__":
    main()
