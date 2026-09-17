#!/usr/bin/env python3
"""Is the requirement on a start ALIGNMENT, or NORM? They were confounded, and this separates them.

The from-near-truth sweep found LM converging at cosine 0.598 and failing by 0.555, against random starts at
cosine ~0. But every point in that sweep lay on a ray H + eta*||H||*u, where

    cosine = 1 / sqrt(1 + eta^2)        norm ratio = sqrt(1 + eta^2)

are LOCKED to the single parameter eta -- the analytic values reproduce the measurements to four digits (0.5972
against 0.5979 measured; 1.674 against 1.675). So the converging point had cosine 0.598 AND norm 1.675, the
failing point cosine 0.555 AND norm 1.804, and the random starts cosine ~0 AND norm 0.897. The two families differ
in BOTH quantities, in OPPOSITE directions, and "the requirement is a cosine of about 0.6" is confounded with
"the requirement is a norm above the truth's".

This decouples them by construction. For a target cosine c and target norm ratio rho,

    start  =  rho * ||H|| * ( c * H/||H||  +  sqrt(1 - c^2) * u_perp ),     u_perp unit, orthogonal to H

has cosine exactly c and norm exactly rho*||H||, for ANY pair. Sweeping the plane locates the convergence
boundary: a VERTICAL boundary means direction is the whole story and the cosine threshold stands as stated; a
TILTED one means the norm is doing part of the work and 0.6 is a ray-specific number.

  python -m experiments.e1b.alignment_vs_norm
"""
import argparse, json, os, socket, sys, time
import torch

from experiments.e1b.e1b_tiny import build_release, replay
from experiments.e1b.e1b_lm import lm_solve
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=24)
    ap.add_argument("--m", type=int, default=20); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cosines", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.6, 0.7])
    ap.add_argument("--norms", type=float, nargs="+", default=[0.9, 1.0, 1.35, 1.7])
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/e1b/alignment_vs_norm.jsonl")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(os.path.dirname(a.out), exist_ok=True)

    R = build_release(a.k, a.N, a.r, a.m, a.P, a.T, a.lr, a.seed, dev)
    H, A0, W0, y, A_T, B_T = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"]
    d, N = H.shape
    nB, nA = torch.linalg.norm(B_T), torch.linalg.norm(A_T)
    nH = torch.linalg.norm(H)
    Hhat = (H / nH).reshape(-1)

    def f(x):
        A_s, B_s = replay(x.reshape(d, N), A0, W0, y, a.m, a.T, a.lr)
        return torch.cat([((B_s - B_T) / nB).reshape(-1), ((A_s - A_T) / nA).reshape(-1)])

    with torch.no_grad():
        print(f"# seed KNOWN arm, unknowns {d*N}, residual at the truth {float(torch.linalg.norm(f(H.reshape(-1)))):.3e}")
    print(f"# cosine and norm ratio set INDEPENDENTLY by construction (exact, not fitted)\n")
    print(f"{'cos':>6} " + "".join(f"{'rho=' + format(r, '.4g'):>14}" for r in a.norms))

    g = torch.Generator().manual_seed(a.seed + 909)
    rows = []
    for c in a.cosines:
        cells = []
        line = f"{c:>6.2f} "
        for rho in a.norms:
            best_err, best_obj = float("inf"), float("inf")
            for rep in range(a.reps):
                u = torch.randn(d * N, generator=g).to(dev)
                u = u - (u @ Hhat) * Hhat
                u = u / torch.linalg.norm(u)
                x0 = rho * nH * (c * Hhat + (1.0 - c * c) ** 0.5 * u)
                # verify the construction rather than trusting it
                cc = float(torch.nn.functional.cosine_similarity(x0, Hhat, dim=0))
                rr = float(torch.linalg.norm(x0) / nH)
                assert abs(cc - c) < 1e-9 and abs(rr - rho) < 1e-9, f"construction off: cos {cc} rho {rr}"
                xh, _ = lm_solve(f, x0, iters=a.iters)
                with torch.no_grad():
                    Hh = xh.reshape(d, N)
                    Dm = torch.cdist(Hh.T, H.T) / H.norm(dim=0)[None, :]
                    asg = Dm.argmin(1)
                    err = max(float(Dm[i, asg[i]]) for i in range(N))
                    obj = float(torch.linalg.norm(f(xh)))
                if err < best_err: best_err, best_obj = err, obj
            rec = best_err < 1e-2
            line += f"{('OK ' if rec else '.  ') + format(best_err, '.1e'):>14}"
            cells.append(dict(cosine=c, norm_ratio=rho, best_max_per_image_err=best_err, objective=best_obj,
                              recovered=bool(rec)))
        print(line, flush=True)
        rows += cells

    print("\n# BOUNDARY READING:")
    for c in a.cosines:
        got = [x for x in rows if x["cosine"] == c and x["recovered"]]
        print(f"   cos={c:.2f}: converges at norm ratios {[x['norm_ratio'] for x in got] or 'none'}")
    by_norm = {}
    for rho in a.norms:
        got = [x["cosine"] for x in rows if x["norm_ratio"] == rho and x["recovered"]]
        by_norm[rho] = min(got) if got else None
        print(f"   rho={rho:.4g}: lowest converging cosine {by_norm[rho] if got else 'none converged'}")
    thresholds = [v for v in by_norm.values() if v is not None]
    verdict = ("NORM-INDEPENDENT: the lowest converging cosine is the same at every norm ratio, so the boundary is "
               "VERTICAL and direction is the whole requirement."
               if len(set(thresholds)) == 1 and len(thresholds) == len(a.norms) else
               "NOT norm-independent: the cosine threshold MOVES with the norm ratio, so the boundary is tilted and "
               "a single cosine number is ray-specific. Report the plane, not a threshold.")
    print(f"\n# {verdict}")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(part="alignment_vs_norm", precision="fp64", d=d, N=N, r=a.r, m=a.m, T=a.T,
                                 seed=a.seed, arm="seed_known", cells=rows,
                                 lowest_converging_cosine_by_norm={str(k): v for k, v in by_norm.items()},
                                 verdict=verdict, git=git_hash(), host=socket.gethostname(),
                                 cmd=" ".join(sys.argv))) + "\n")


if __name__ == "__main__":
    main()
