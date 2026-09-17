#!/usr/bin/env python3
"""Nullity 0 and still no landings: is that CONDITIONING or BASIN? Two columns that separate them.

The seed-known arm has nullity 0 -- the truth is locally unique -- and neither Adam (0 of 120 starts) nor
Levenberg-Marquardt is finding it from random starts. Identifiability and recoverability have come apart, and
there are two different culprits with two different remedies:

  CONDITIONING -- the truth is unique but the Jacobian is so ill-conditioned that no solver can descend to it.
                  Remedy: preconditioning, or a parametrisation that improves the conditioning (a chart).
  BASIN        -- the Jacobian is fine but the attraction region around the truth is tiny, so random starts miss
                  it. Remedy: a better supply of starts (an initialiser), not a better chart.

Two measurements, neither of which uses random starts, because random-start landings conflate the solver with the
geometry:

  1. CONDITION NUMBER of the residual Jacobian at the truth, s[0]/s[rank-1] (over the NON-null directions, so a
     nullity does not make it trivially infinite). Free from the SVD already being computed.
  2. FROM-NEAR-TRUTH convergence: start LM at the truth perturbed by relative noise eta and see whether it comes
     back. Converging for small eta and failing for large eta LOCATES the basin radius; failing even at tiny eta
     says conditioning. This answers necessary-versus-sufficient directly and converges or fails fast.

  python -m experiments.e1b.basin_and_conditioning
"""
import argparse, json, os, socket, sys, time
import torch
import torch.func as tfn

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
    ap.add_argument("--etas", type=float, nargs="+", default=[1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    ap.add_argument("--iters", type=int, default=60); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/e1b/basin_conditioning.jsonl")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(os.path.dirname(a.out), exist_ok=True)

    R = build_release(a.k, a.N, a.r, a.m, a.P, a.T, a.lr, a.seed, dev)
    H, A0, W0, y, A_T, B_T, b, L = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"], R["b"], R["L"]
    d, N = H.shape
    nB, nA = torch.linalg.norm(B_T), torch.linalg.norm(A_T)
    nH = torch.linalg.norm(H)

    def res(Hc, A0c):
        A_s, B_s = replay(Hc, A0c, W0, y, a.m, a.T, a.lr)
        return torch.cat([((B_s - B_T) / nB).reshape(-1), ((A_s - A_T) / nA).reshape(-1)])

    Hc0 = H - b[:, None]
    W_true = torch.linalg.lstsq(L, Hc0).solution

    # (name, flat->(H, A0), truth-as-flat, per-image unknowns)
    setups = {
        "free_H_seed_known":  (lambda x: (x.reshape(d, N), A0), H.reshape(-1).clone(), d),
        "chart_seed_known":   (lambda x: (L @ x.reshape(a.k, N) + b[:, None], A0), W_true.reshape(-1).clone(), a.k),
    }

    rows = []
    for name, (unpack, x_true, ki) in setups.items():
        f = lambda x: res(*unpack(x))
        J = tfn.jacfwd(f)(x_true).reshape(-1, x_true.numel())
        sv = torch.linalg.svdvals(J)
        rk = int((sv > 1e-10 * float(sv[0])).sum())
        cond = float(sv[0] / sv[rk - 1])
        print(f"\n## {name}: unknowns {x_true.numel()} (k_i = {ki}), rank {rk}, nullity {x_true.numel() - rk}")
        print(f"   CONDITION NUMBER over the non-null directions: {cond:.3e}   "
              f"(s0 {float(sv[0]):.3e} -> s_rk {float(sv[rk-1]):.3e})")
        with torch.no_grad():
            r0 = float(torch.linalg.norm(f(x_true)))
        print(f"   residual at the truth: {r0:.3e}")
        print(f"   {'eta':>8}  {'final objective':>16}  {'max per-image err':>18}  {'recovered?':>11}")
        g = torch.Generator().manual_seed(a.seed + 101)
        cells = []
        for eta in a.etas:
            best = None
            for rep in range(a.reps):
                pert = torch.randn(x_true.shape, generator=g).to(dev)
                x0 = x_true + eta * torch.linalg.norm(x_true) * pert / torch.linalg.norm(pert)
                t0 = time.time()
                xh, cost = lm_solve(f, x0, iters=a.iters)
                with torch.no_grad():
                    Hh, _ = unpack(xh)
                    Dm = torch.cdist(Hh.T, H.T) / H.norm(dim=0)[None, :]
                    asg = Dm.argmin(1)
                    err = max(float(Dm[i, asg[i]]) for i in range(N))
                    obj = float(torch.linalg.norm(f(xh)))
                if best is None or err < best[1]: best = (obj, err, time.time() - t0)
            obj, err, sec = best
            rec = err < 1e-2
            print(f"   {eta:>8.4g}  {obj:>16.3e}  {err:>18.3e}  {'YES' if rec else 'no':>11}")
            cells.append(dict(eta=eta, objective=obj, max_per_image_err=err, recovered=bool(rec), seconds=sec))
        rec_etas = [c["eta"] for c in cells if c["recovered"]]
        radius = max(rec_etas) if rec_etas else None
        fail_small = (not cells[0]["recovered"])
        largest_tested = max(c["eta"] for c in cells)
        edge_found = bool(rec_etas) and radius < largest_tested
        if fail_small:
            verdict = ("CONDITIONING: fails even at the smallest perturbation, so the truth is unique but "
                       "unreachable")
        elif edge_found:
            verdict = (f"BASIN LOCATED: converges out to relative perturbation {radius:.2f} and fails beyond, so "
                       f"the truth is reachable and the obstruction is the SUPPLY OF STARTS")
        else:
            verdict = (f"BASIN NOT YET BOUNDED: converges at EVERY perturbation tested, up to {largest_tested:.2f}. "
                       f"The edge lies beyond the range tested -- this does NOT establish a radius, only a lower "
                       f"bound on one. The obstruction is not conditioning (condition number {cond:.1e}).")
        print(f"   -> {verdict}")
        rows.append(dict(setup=name, unknowns=int(x_true.numel()), k_per_image=ki, rank=rk,
                         nullity=int(x_true.numel() - rk), condition_number=cond, residual_at_truth=r0,
                         basin_radius_lower_bound=radius, basin_edge_located=bool(edge_found),
                         largest_eta_tested=largest_tested, verdict=verdict, cells=cells))
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(part="basin_and_conditioning", precision="fp64", d=d, N=N, r=a.r, m=a.m, T=a.T,
                                 k=a.k, seed=a.seed, setups=rows, git=git_hash(),
                                 host=socket.gethostname(), cmd=" ".join(sys.argv))) + "\n")
    # How far from the truth are the RANDOM starts the real arms use? Without this the basin radius has no
    # yardstick -- "converges out to eta = 1" only means something against the distance a random start sits at.
    gs = torch.Generator().manual_seed(a.seed + 31)
    scale = float(H.norm(dim=0).median())
    G0 = torch.randn(60, d, N, generator=gs).to(dev)
    G0 = G0 / G0.norm(dim=1, keepdim=True) * scale
    dist = torch.stack([torch.linalg.norm(G0[i] - H) / nH for i in range(60)])
    print(f"\n# YARDSTICK: the random starts the real arms use sit at relative distance "
          f"{float(dist.median()):.3f} from the truth (range {float(dist.min()):.3f}-{float(dist.max()):.3f}).")
    print("# A basin radius BELOW that distance is the quantitative statement of what an initialiser must supply.")
    print("# Comparing the two setups isolates the CHART's second job: if the chart's condition number is orders "
          "better, or its basin radius orders larger, the chart supplies reachability as well as identifiability.")


if __name__ == "__main__":
    main()
