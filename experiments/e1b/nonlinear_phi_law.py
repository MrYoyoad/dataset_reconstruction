#!/usr/bin/env python3
"""Does the identifiability law survive a NONLINEAR phi? The one cell that settles the extrapolation.

The law `nullity = N * max(0, k - (m + r - N - 1))` is derived and confirmed (12/12, job 350967) only where the
map from chart coordinates to the adapted layer's input is AFFINE. Every use of it on a real release -- including
the two-walls result's identifiability cap -- is therefore an extrapolation, because a trained network sits in
that path. This measures it directly.

WHY NOT ON THE CIFAR RELEASE ITSELF. There the Jacobian is about 17088 x 17152, which is 2.3 GB in fp64 before
the SVD workspace, and computing it means one forward pass per unknown through an unrolled training loop. The
SCALE adds nothing to the question -- what is being tested is whether a nonlinearity in phi breaks the rank
argument -- so the cell is built small enough to measure EXACTLY, with a genuinely trained nonlinear backbone.

  m = 11, r = 12, N = 8, penultimate width n = 32  ->  cap = m + r - N - 1 = 14
  predicted:  nullity 0 for k <= 14,  then exactly 8 per unit of k above it

THE CONTROL IS THE POINT. The same shapes are run twice: once with `phi` the trained network (nonlinear) and once
with `phi` the identity on the same feature dimension (affine path). If the affine arm reproduces the law and the
trained arm does not, the nonlinearity is the cause and nothing else is. If BOTH reproduce it, the extrapolation
this project has been carrying is justified and can be reported as measured rather than assumed.

  python -m experiments.e1b.nonlinear_phi_law
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from experiments.cifar.cifar_newclass import load_cifar10, load_cifar100_class
from experiments.utils.identifiability import identifiability, assert_chart_contains, block_normalised_residual
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)


def log(s): print(s, flush=True)


class SmallNet(nn.Module):
    """A real trained backbone, deliberately narrow so the Jacobian is exact and cheap. phi is genuinely nonlinear."""
    def __init__(s, n=32, m=10):
        super().__init__()
        s.l1 = nn.Linear(3072, 256); s.l2 = nn.Linear(256, n)
        s.head = nn.Linear(n, m, bias=False)

    def phi(s, x): return F.gelu(s.l2(F.gelu(s.l1(x.reshape(len(x), -1)))))
    def forward(s, x): return s.head(s.phi(x))


def train_backbone(root, dev, n, epochs, seed):
    Xtr_np, ytr_np, _, _ = load_cifar10(root)        # returns (Xtr, ytr, Xte, yte), not a pair of splits
    Xtr = torch.tensor(Xtr_np, dtype=torch.float64, device=dev); ytr = torch.tensor(ytr_np, device=dev)
    torch.manual_seed(seed)
    net = SmallNet(n=n).to(dev).double()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in perm.split(256):
            opt.zero_grad(); F.cross_entropy(net(Xtr[i]), ytr[i]).backward(); opt.step()
        with torch.no_grad():
            acc = (net(Xtr[:5000]).argmax(1) == ytr[:5000]).double().mean()
        log(f"#   epoch {ep+1}/{epochs}: train acc {float(acc)*100:.1f}%")
    for p in net.parameters(): p.requires_grad_(False)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--m", type=int, default=11)
    ap.add_argument("--r", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--T", type=int, default=20, help="nullity does not depend on T")
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 12, 14, 15, 16, 20, 24])
    ap.add_argument("--cls", default="motorcycle"); ap.add_argument("--data-root", default="data")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/e1b/nonlinear_phi_law.jsonl")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cap = a.m + a.r - a.N - 1
    log(f"# n={a.n} m={a.m} r={a.r} N={a.N} T={a.T}  ->  cap = m+r-N-1 = {cap}; predicted nullity = N*max(0,k-cap)")

    net = train_backbone(a.data_root, dev, a.n, a.epochs, a.seed)
    pool, cname = load_cifar100_class(a.data_root, a.cls)
    Pub = torch.tensor(pool["train"], dtype=torch.float64, device=dev)
    Pri = torch.tensor(pool["test"], dtype=torch.float64, device=dev)
    g = np.random.RandomState(a.seed)
    X = Pri[g.permutation(len(Pri))[: a.N]].T                       # (3072, N)
    mean = Pub.mean(0)
    _, S_, Vh_ = torch.linalg.svd(Pub - mean, full_matrices=False)

    W0 = torch.cat([net.head.weight.double(), torch.zeros(1, a.n, dtype=torch.float64, device=dev)], 0)
    y = torch.full((a.N,), a.m - 1, device=dev)
    Y = torch.eye(a.m, device=dev, dtype=torch.float64)[y].T
    A0 = (1.0 / math.sqrt(a.n) * torch.randn(a.r, a.n, dtype=torch.float64,
                                             generator=torch.Generator().manual_seed(a.seed + 7))).to(dev)

    def make_release(H):
        A, B = A0.clone(), torch.zeros(a.m, a.r, dtype=torch.float64, device=dev)
        for _ in range(a.T):
            z = W0 @ H + B @ (A @ H); D = (torch.softmax(z, 0) - Y) / a.N
            B, A = B - a.lr * (D @ (A @ H).T), A - a.lr * (B.T @ D @ H.T)
        return A, B

    def sim_factory(A_T, B_T):
        def sim(Wc, A0c):
            Hc = phi_of(psi(Wc))
            A, B = A0c, torch.zeros(a.m, a.r, dtype=torch.float64, device=dev)
            for _ in range(a.T):
                z = W0 @ Hc + B @ (A @ Hc); D = (torch.softmax(z, 0) - Y) / a.N
                B, A = B - a.lr * (D @ (A @ Hc).T), A - a.lr * (B.T @ D @ Hc.T)
            return (B, A)
        return sim

    rows = []
    for arm in ("trained_phi_NONLINEAR", "identity_phi_AFFINE_control"):
        log(f"\n## {arm}")
        if arm.startswith("trained"):
            phi_of = lambda Xp: net.phi(Xp.T).T                      # (3072,B) -> (n,B), genuinely nonlinear
            src = X
        else:
            # affine control at the SAME shapes: features are a fixed linear image of the chart, phi = identity
            Wlin = (torch.randn(a.n, 3072, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(a.seed + 21)).to(dev) / math.sqrt(3072))
            phi_of = lambda Xp: Wlin @ Xp
            src = X
        log(f"   {'k':>4} {'unknowns':>9} {'rank':>6} {'nullity':>8} {'predicted':>10} {'hit':>5}  gap at cut")
        for k in a.ks:
            V = Vh_[:k].T.contiguous()
            psi = lambda W: mean[:, None] + V @ W
            coords = lambda Xp: V.T @ (Xp - mean[:, None])
            Won = coords(src)
            X_on = psi(Won)                                          # privates ON the chart, so the chart contains the truth
            assert_chart_contains(psi, Won, X_on)
            H = phi_of(X_on)
            A_T, B_T = make_release(H)
            res = block_normalised_residual(sim_factory(A_T, B_T), (B_T, A_T))
            r = identifiability(res, (Won, A0), mode="fwd")
            pred = a.N * max(0, k - cap)
            hit = (r.nullity == pred)
            log(f"   {k:>4} {r.n_unknowns:>9} {r.rank:>6} {r.nullity:>8} {pred:>10} {'HIT' if hit else 'MISS':>5}  "
                f"{r.gap_at_cut:.1e}")
            rows.append(dict(arm=arm, k=k, cap=cap, predicted=pred, hit=bool(hit), **r.as_dict()))

    for arm in ("trained_phi_NONLINEAR", "identity_phi_AFFINE_control"):
        sub = [x for x in rows if x["arm"] == arm]
        log(f"\n# {arm}: {sum(x['hit'] for x in sub)}/{len(sub)} cells match the affine law")
    nl = [x for x in rows if x["arm"].startswith("trained")]
    af = [x for x in rows if x["arm"].startswith("identity")]
    if all(x["hit"] for x in af) and all(x["hit"] for x in nl):
        log("# VERDICT: the law SURVIVES a nonlinear phi. The extrapolation this project carries is measured, not assumed.")
    elif all(x["hit"] for x in af):
        log("# VERDICT: the law holds on the AFFINE control and FAILS with a trained phi -- the nonlinearity is the cause, "
            "and every use of the cap on a real release must be withdrawn or remeasured.")
    else:
        log("# VERDICT: the AFFINE CONTROL ITSELF failed -- this harness does not reproduce the law, so nothing here "
            "speaks to nonlinearity. Fix the harness before reading the trained arm.")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(part="nonlinear_phi_law", precision="fp64", n=a.n, m=a.m, r=a.r, N=a.N, T=a.T,
                                 cap=cap, cls=cname, cells=rows, git=git_hash(), host=socket.gethostname(),
                                 cmd=" ".join(sys.argv))) + "\n")


if __name__ == "__main__":
    main()
