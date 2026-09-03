#!/usr/bin/env python3
"""How far does the release-consistent set actually extend past the capacity line?

Motivation (independent audit, 2026-09-03).  Past k = m+r-N the Jacobian at the truth is rank-deficient,
so the truth is not locally isolated.  But the measured image errors there are 1.7e-3 to 1.2e-2 -- SUB-
PERCENT -- and in 8 of 11 past-line cells every image is inside the study's own 1e-2 recovery tolerance.
So "the release is reproduced by the WRONG image" is not supported: what was measured is "no longer
pinned to machine precision", an 11-order jump from 1e-14 to 1e-3, not a failure to recover.

Worse, every past-line cell was started adjacent to the truth, so the solver only drifts as far along the
flat direction as LM happens to take it.  The measured 1e-3 is a LOWER bound on the fibre, not its
diameter.  To earn the word "alias" one has to TRAVEL the null direction and see how far the
release-consistent set actually reaches.

Method: compute J at the truth, take the right-singular vectors spanning the numerically null space,
then do continuation -- step along the null direction, re-minimise the residual back to the floor
(retraction), and record how far the image has moved while the release is still reproduced exactly.
Reports image error as a function of arc length, with the residual at every step so that a point only
counts if it genuinely reproduces the release.

  python -m experiments.exact_inversion.null_traverse --k 32 --N 8 --T 400 --lr 0.01
"""
import argparse, json, math, socket, sys
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import (
    World, train_release, simulate_sgd_reduced, qr_canon, git_hash)

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=32); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=24, help="continuation steps along the null direction")
    ap.add_argument("--step-size", type=float, default=0.25, help="arc length per continuation step")
    ap.add_argument("--retract-iters", type=int, default=25, help="LM iterations to return to the fibre")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(a.n)
    dev = torch.device(a.device)
    world = World(a.k, a.P, a.n, a.seed, dev)
    g = torch.Generator().manual_seed(a.seed + 7)
    N, r, m, k = a.N, a.r, a.m, a.k

    W_true = torch.randn(k, N, generator=g).to(dev)
    X_img = world.psi(W_true); H = world.phi(X_img)
    y = (torch.arange(N) % m).to(dev)
    W0 = (torch.randn(m, a.n, generator=g) / math.sqrt(a.n)).to(dev)
    A0 = (a.sigma0 * torch.randn(r, a.n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, m, a.T, a.lr, "sgd", a.wd)
    U_true, _ = qr_canon(H)
    nB, nA = torch.linalg.norm(B_T), torch.linalg.norm(A_T)
    nW = k * N

    def res_vec(v):
        Wc = v[:nW].reshape(k, N); aux = v[nW:].reshape(r, N)
        Hc = world.features_from_latents(Wc)
        Bs, Xis, Uc = simulate_sgd_reduced(Hc, aux, W0, y, m, a.T, a.lr, a.wd)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])

    v0 = torch.cat([W_true.reshape(-1), (A0 @ U_true).reshape(-1)]).detach()
    J = tf.jacfwd(res_vec)(v0).detach()
    Umat, S, Vh = torch.linalg.svd(J, full_matrices=False)
    tol = 1e-12 * float(S[0])
    n_null = int((S <= tol).sum())
    print(f"# null traverse  k={k} N={N} line k*={m+r-N}  J {tuple(J.shape)}  sigma_max={S[0]:.3e} "
          f"sigma_min={S[-1]:.3e}  numerically-null dims={n_null}  git={git_hash()}", flush=True)
    if n_null == 0:
        print("# WARNING: no numerically null direction at the truth -- this cell is BELOW the line; "
              "traversing anyway will simply be obstructed.", flush=True)
    d = Vh[-1].clone()                                     # the least-constrained direction
    d = d / torch.linalg.norm(d)

    def retract(v):
        """A few LM steps to drive the residual back to the floor without moving further than needed."""
        lam = 1e-8
        for _ in range(a.retract_iters):
            F = res_vec(v); f = float(F @ F)
            if f < 1e-28: break
            Jl = tf.jacfwd(res_vec)(v).detach()
            JtJ = Jl.T @ Jl; JtF = Jl.T @ F
            for _ in range(8):
                step = torch.linalg.solve(JtJ + lam * torch.eye(JtJ.shape[0], device=v.device), JtF)
                vn = v - step; Fn = res_vec(vn)
                if float(Fn @ Fn) < f: v = vn; lam = max(lam / 3, 1e-16); break
                lam *= 6
        return v

    v = v0.clone(); rows = []
    for i in range(a.steps + 1):
        if i > 0:
            v = retract(v + a.step_size * d)
            # re-aim along the current least-constrained direction (the fibre curves)
            Jl = tf.jacfwd(res_vec)(v).detach()
            dn = torch.linalg.svd(Jl, full_matrices=False)[2][-1]
            d = dn if float(dn @ d) >= 0 else -dn
        with torch.no_grad():
            F = res_vec(v); resid = float(F @ F)
            Wc = v[:nW].reshape(k, N)
            err = (torch.linalg.norm(world.psi(Wc) - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
            arc = float(torch.linalg.norm(v - v0))
        row = dict(step=i, arc_len=arc, residual=resid, on_fibre=bool(resid < 1e-24),
                   img_err_max=float(err.max()), img_err_median=float(err.median()),
                   k=k, N=N, r=r, m=m, T=a.T, lr=a.lr, seed=a.seed, line=m + r - N,
                   n_null=n_null, sigma_min=float(S[-1]), git=git_hash(), host=socket.gethostname(),
                   cmd=" ".join(sys.argv))
        rows.append(row)
        print(f"  step {i:>3} arc={arc:8.3f}  residual={resid:.2e}  on_fibre={row['on_fibre']}  "
              f"img_err max={row['img_err_max']:.3e} med={row['img_err_median']:.3e}", flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    on = [x for x in rows if x["on_fibre"]]
    if on:
        best = max(on, key=lambda x: x["img_err_max"])
        print(f"# FIBRE EXTENT: staying on the release-consistent set (residual < 1e-24), the image moved "
              f"to a MAX relative error of {best['img_err_max']:.3e} at arc length {best['arc_len']:.3f} "
              f"(step {best['step']} of {a.steps})", flush=True)


if __name__ == "__main__":
    main()
