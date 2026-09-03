#!/usr/bin/env python3
"""Does the released B_T satisfy the simplex constraint 1^T B_T = 0?

This is the mechanism behind the STRICT form of the capacity law: softmax cross-entropy gives 1^T D_t = 0,
so 1^T grad_B = 0, so with B_0 = 0 and any scalar-linear update 1^T B_t = 0 for every t.  B_T therefore
lies in 1-perp tensor R^r and the rank-N locus there has dimension N((m-1)+r-N) -- a deficit of exactly N
against the plain cap, which is what turns k <= m+r-N into the strict k < m+r-N.

Adam's entrywise normalisation does NOT preserve it, so under Adam the constraint should be absent and the
cap should revert to the plain N(m+r-N).  Both halves are checked here and written as result rows; the
number was previously an uncommitted inline check, which an audit rightly refused to dagger.

  python -m experiments.exact_inversion.simplex_check --out results/exact_inversion/step35_simplex.jsonl
"""
import argparse, json, math, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import World, train_release, git_hash

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["sgd:96:12:8:16:20:400:0.01", "sgd:96:12:8:16:20:1500:0.03",
                                                   "sgd:32:6:4:16:20:200:0.01", "adam:32:6:4:16:20:200:0.003",
                                                   "adam:96:12:8:16:20:800:0.003"],
                    help="release:n:k:N:r:m:T:lr")
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    print(f"# simplex check  ||1^T B_T|| / ||B_T||   git={git_hash()} host={socket.gethostname()}", flush=True)
    print(f"{'release':>7} {'n':>4} {'k':>3} {'N':>3} {'r':>3} {'m':>3} {'T':>5} {'seed':>4} "
          f"{'||1^T B||/||B||':>16} {'holds?':>7}")
    for spec in a.cells:
        rel, n, k, N, r, m, T, lr = spec.split(":")
        n, k, N, r, m, T, lr = int(n), int(k), int(N), int(r), int(m), int(T), float(lr)
        for seed in a.seeds:
            world = World(k, 64, n, seed, dev)
            g = torch.Generator().manual_seed(seed + 7)
            W = torch.randn(k, N, generator=g).to(dev)
            H = world.phi(world.psi(W))
            y = (torch.arange(N) % m).to(dev)
            W0 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(dev)
            A0 = (torch.randn(r, n, generator=g) / math.sqrt(n)).to(dev)
            A_T, B_T = train_release(H, A0, W0, y, m, T, lr, rel)
            ratio = float(torch.linalg.norm(B_T.sum(dim=0)) / torch.linalg.norm(B_T))
            holds = bool(ratio < 1e-12)
            out = dict(release=rel, n=n, k=k, N=N, r=r, m=m, T=T, lr=lr, seed=seed,
                       colsum_ratio=ratio, simplex_holds=holds,
                       cap_plain=N * (m + r - N), cap_simplex=N * ((m - 1) + r - N),
                       git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            print(f"{rel:>7} {n:>4} {k:>3} {N:>3} {r:>3} {m:>3} {T:>5} {seed:>4} "
                  f"{ratio:>16.3e} {str(holds):>7}", flush=True)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
