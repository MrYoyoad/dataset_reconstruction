#!/usr/bin/env python3
"""Dump the FULL singular spectrum of the simulator Jacobian at the ground truth.

The jsonl carries only sigma_min/sigma_max; the resolution-ellipsoid reading of DF needs the whole
spectrum.  Three readings of the same object (this is the unification worth stating once):
    rank(DF)      -> local identifiability
    spectrum(DF)  -> the local resolution ellipsoid
    whitened DF   -> detectability

Imports the main testbed rather than copying it, so the recipe cannot drift and so a running multi-cell
job that re-imports the main module per cell is not disturbed.

  python -m experiments.exact_inversion.jacobian_spectrum --release sgd --n 96 --k 12 --N 8 --T 400 --lr 0.01
"""
import argparse, json, math, socket, sys
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import (
    World, train_release, simulate_sgd_reduced, simulate_adam_full, qr_canon, git_hash)

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save", default=None)
    a = ap.parse_args()
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(a.n)
    dev = torch.device(a.device)
    world = World(a.k, a.P, a.n, a.seed, dev)
    g = torch.Generator().manual_seed(a.seed + 7)
    N, r, m, n, k = a.N, a.r, a.m, a.n, a.k

    W_true = torch.randn(k, N, generator=g).to(dev)
    H = world.phi(world.psi(W_true))
    y = (torch.arange(N) % m).to(dev)
    W0 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(dev)
    A0 = (a.sigma0 * torch.randn(r, n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, m, a.T, a.lr, a.release, a.wd)
    U_true, _ = qr_canon(H)
    nB, nA = torch.linalg.norm(B_T), torch.linalg.norm(A_T)
    aux_true = (A0 @ U_true) if a.release == "sgd" else A0
    nW = k * N

    def res(v):
        Wc = v[:nW].reshape(k, N); aux = v[nW:].reshape(aux_true.shape)
        Hc = world.features_from_latents(Wc)
        if a.release == "sgd":
            Bs, Xis, Uc = simulate_sgd_reduced(Hc, aux, W0, y, m, a.T, a.lr, a.wd)
            return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])
        As_, Bs = simulate_adam_full(Hc, aux, W0, y, m, a.T, a.lr, a.wd)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((As_ - A_T) / nA).reshape(-1)])

    v = torch.cat([W_true.reshape(-1), aux_true.reshape(-1)]).detach()
    J = tf.jacfwd(res)(v).detach()
    sv = torch.linalg.svdvals(J)
    # split the supply: how much rank does each released block carry on its own?
    nB_rows = m * r
    svB = torch.linalg.svdvals(J[:nB_rows]); svA = torch.linalg.svdvals(J[nB_rows:])
    tol = 1e-12 * float(sv[0])
    out = dict(release=a.release, n=n, k=k, N=N, r=r, m=m, T=a.T, lr=a.lr, seed=a.seed,
               rows=int(J.shape[0]), cols=int(J.shape[1]),
               unknowns_data=N * k, unknowns_nuisance=(r * N if a.release == "sgd" else r * n),
               eqs_B=m * r, eqs_A=(r * N if a.release == "sgd" else r * n),
               rank=int((sv > tol).sum()), rank_B_block=int((svB > 1e-12 * float(svB[0])).sum()),
               rank_A_block=int((svA > 1e-12 * float(svA[0])).sum()),
               manifold_bound_N_m_r_N=N * (m + r - N), res_at_truth=float(torch.linalg.norm(res(v))),
               sigma=[float(x) for x in sv], sigma_B=[float(x) for x in svB], sigma_A=[float(x) for x in svA],
               git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
    print(f"{a.release} n={n} k={k} N={N}: J {out['rows']}x{out['cols']}  rank={out['rank']}  "
          f"rank(B block)={out['rank_B_block']} (manifold bound N(m+r-N)={out['manifold_bound_N_m_r_N']})  "
          f"rank(A block)={out['rank_A_block']}  res(truth)={out['res_at_truth']:.1e}  "
          f"sigma: max {sv[0]:.3e} min {sv[-1]:.3e}", flush=True)
    if a.out:
        with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")
    if a.save: torch.save(dict(sigma=sv.cpu(), sigma_B=svB.cpu(), sigma_A=svA.cpu(), meta=out), a.save)


if __name__ == "__main__":
    main()
