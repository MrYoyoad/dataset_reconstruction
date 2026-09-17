#!/usr/bin/env python3
"""Is the capacity line the IDENTIFIABILITY boundary? Sweep the chart dimension k and measure the nullity.

Two lanes independently derived a closed form for the solution-family dimension, reconciling to

    nullity  =  N * max(0, k_i - (m + r - N - 1))

where k_i is the number of unknowns PER IMAGE (d for free features, k for a chart of dimension k). It comes from
B_T not being m*r free numbers: it lies on the rank-N, zero-column-sum variety of dimension N*(m-1+r-N) = 280
here, so the attainable rank is r*d + 280 = 1816 rather than 2016 -- which is exactly the rank measured at
job 350940, and the 200 "missing" equations are exactly that difference.

Note what the threshold is: m + r - N - 1, i.e. the CAPACITY LINE k < m + r - N. So the formula says the capacity
line is not merely a counting heuristic -- it is the boundary at which the private data stops being identifiable,
with the family growing by exactly N dimensions per unit of k above it.

THE FALSIFIER, registered before the run (d=64, N=8, r=24, m=20 -> threshold 35):

    k = 34, 35  ->  nullity 0        k = 36  ->  nullity EXACTLY 8       k = 37  ->  16
    k = 40      ->  40               k = 48  ->  104                     k = 56  ->  168
    k = 64      ->  232  <- self-check: at k = d the chart is the whole space, and 232 is the measured free-H value

A one-unit step from 0 to 8 between k=35 and k=36 is discontinuous and cannot be produced by a smooth artefact.
If the sweep comes back smooth, or the step sits elsewhere, the formula is wrong.

The chart at each k CONTAINS the truth by construction (an orthonormal basis of col(H - b) padded with random
orthogonal directions), so k is the only thing moving. These are ORACLE charts -- the point is to isolate the
effect of chart DIMENSION on identifiability, not to model an attacker.

  python -m experiments.e1b.chart_capacity_sweep
"""
import argparse, json, os, socket, sys, time
import torch
import torch.func as tfn

from experiments.e1b.e1b_tiny import build_release, replay
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=24)
    ap.add_argument("--m", type=int, default=20); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--k-release", type=int, default=12)
    ap.add_argument("--T", type=int, default=20, help="dimensions do not depend on T")
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 12, 20, 30, 34, 35, 36, 37, 40, 48, 56, 64])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/e1b/chart_capacity.jsonl")
    a = ap.parse_args(); dev = torch.device(a.device)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    R = build_release(a.k_release, a.N, a.r, a.m, a.P, a.T, a.lr, a.seed, dev)
    H, A0, W0, y, A_T, B_T, b = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"], R["b"]
    d, N = H.shape
    nB, nA = torch.linalg.norm(B_T), torch.linalg.norm(A_T)
    thresh = a.m + a.r - N - 1
    print(f"# d={d} N={N} r={a.r} m={a.m} T={a.T}  threshold m+r-N-1 = {thresh}   fp64")
    print(f"# predicted nullity = N * max(0, k - {thresh});  charts CONTAIN the truth by construction (oracle)\n")

    def res(Hc, A0c):
        A_s, B_s = replay(Hc, A0c, W0, y, a.m, a.T, a.lr)
        return torch.cat([((B_s - B_T) / nB).reshape(-1), ((A_s - A_T) / nA).reshape(-1)])

    Hc0 = H - b[:, None]
    Q, _ = torch.linalg.qr(Hc0)                                  # orthonormal basis of col(H - b), dim <= N
    base = int(torch.linalg.matrix_rank(Hc0, rtol=1e-10))
    Q = Q[:, :base]
    g = torch.Generator().manual_seed(a.seed + 7)
    Rr = torch.randn(d, d, generator=g).to(dev)
    Rr = Rr - Q @ (Q.T @ Rr)
    Qp, _ = torch.linalg.qr(Rr)                                  # directions orthogonal to the truth's span

    rows = []
    for k in a.ks:
        if k < base or k > d:
            print(f"  k={k}: skipped (must satisfy {base} <= k <= {d})"); continue
        L = torch.cat([Q, Qp[:, :k - base]], dim=1)              # (d, k), contains col(H - b)
        W_true = L.T @ Hc0                                       # exact, since L is orthonormal and contains it
        assert float(torch.linalg.norm(L @ W_true + b[:, None] - H) / torch.linalg.norm(H)) < 1e-12, "chart must contain the truth"

        def f(W, A0c):
            return res(L @ W + b[:, None], A0c)

        t0 = time.time()
        J = tfn.jacrev(f, argnums=(0, 1))(W_true, A0)
        Jm = torch.cat([J[0].reshape(-1, W_true.numel()), J[1].reshape(-1, A0.numel())], dim=1)
        sv = torch.linalg.svdvals(Jm)
        floor = 1e-10 * float(sv[0])
        rk = int((sv > floor).sum())
        n_in = Jm.shape[1]; nul = n_in - rk
        pred = N * max(0, k - thresh)
        hit = (nul == pred)
        gap = (float(sv[rk - 1]) / float(sv[rk])) if 0 < rk < len(sv) else float("inf")
        print(f"  k={k:3d}:  J {Jm.shape[0]}x{n_in}  rank {rk}  NULLITY {nul:4d}   predicted {pred:4d}   "
              f"{'HIT' if hit else '*** MISS ***'}   gap at cut {gap:.1e}   [{time.time()-t0:.0f}s]")
        rows.append(dict(k=k, unknowns=n_in, rank=rk, nullity=nul, predicted=pred, hit=bool(hit),
                         gap_at_cut=gap, n_equations=Jm.shape[0]))

    hits = sum(r["hit"] for r in rows)
    print(f"\n# {hits}/{len(rows)} cells match the closed form.")
    step = [r for r in rows if r["k"] in (35, 36)]
    if len(step) == 2:
        lo, hi = sorted(step, key=lambda r: r["k"])
        print(f"# THE ONE-UNIT FALSIFIER: k=35 nullity {lo['nullity']} (predicted 0) -> k=36 nullity {hi['nullity']} "
              f"(predicted 8).  {'STEP CONFIRMED' if lo['nullity'] == 0 and hi['nullity'] == 8 else 'NOT AS PREDICTED'}")
    out = dict(part="chart_capacity_sweep", d=d, N=N, r=a.r, m=a.m, T=a.T, seed=a.seed, threshold=thresh,
               formula="nullity = N * max(0, k - (m + r - N - 1))", chart="ORACLE: contains col(H-b) by construction",
               precision="fp64", cells=rows, hits=hits, n_cells=len(rows),
               git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
    with open(a.out, "a") as fh: fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
