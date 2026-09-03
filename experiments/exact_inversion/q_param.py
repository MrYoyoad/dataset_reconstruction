#!/usr/bin/env python3
"""The Q-parametrisation: solve for (w, S) instead of (w, X), and eliminate the seed in closed form.

Suggested by the user. The coefficient trajectory sees the seed only through
    Q_t = (A_0 H)^T (A_0 H) = R_H^T S R_H ,      S := X^T X   (X = A_0 U)
(verified separately as a gauge invariance: replacing A_0 by O A_0 for orthogonal O leaves P_t, M_t and
Omega untouched, job 487882). So (w, S) already determine the whole trajectory, hence Omega and P_T, and
then the released A-block gives X back in CLOSED FORM:

    A_T U_c(w) = X Omega    =>    X = A_T U_c(w) Omega(w, S)^{-1}

The unknown seed block therefore need not be searched at all. Unknowns drop from Nk + rN to
Nk + N(N+1)/2 -- 224 -> 132 at (N,k,r) = (8,12,16), a 1.7x smaller search space -- with the SAME capacity,
since the count is unchanged (the rN of the A-block paid exactly for X either way).

Why it might matter: the measured binding constraint on this attack is not identifiability but the
INITIALISER (1 of 20 release-only starts converged). A smaller search space is the most direct lever on
that, and it is free of any new assumption.

Residual (S is symmetric, carried as its lower triangle):
    r1 = B_sim(w, S) - B_T                      the released B-block
    r2 = S - X^T X   with X = A_T U_c Omega^-1  self-consistency of the eliminated seed

Gate: at the truth the residual must vanish to the FP64 floor. If it does not, the elimination is wrong
and nothing downstream means anything.

  python -m experiments.exact_inversion.q_param --k 12 --N 8 --T 400 --lr 0.01
"""
import argparse, json, math, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import (
    World, train_release, simulate_sgd_reduced, qr_canon, git_hash, RECOVER_TOL)

torch.set_default_dtype(torch.float64)


def tri_to_sym(v, N):
    """Lower-triangle vector -> symmetric N x N."""
    S = torch.zeros(N, N, dtype=v.dtype, device=v.device)
    idx = torch.tril_indices(N, N, device=v.device)
    S[idx[0], idx[1]] = v
    return S + torch.tril(S, -1).T


def sym_to_tri(S):
    idx = torch.tril_indices(S.shape[0], S.shape[0], device=S.device)
    return S[idx[0], idx[1]]


def coeffs_from_Q(R_H, S, W0, Hc, y, m, T, lr):
    """Run the recipe in COEFFICIENT space: the trajectory depends on the seed only through Q = R_H^T S R_H.
       Returns (P_T, M_T) with eta absorbed, i.e. B_T = P_T (A_0 H)^T and A_t = A_0 (I + H M_t H^T)."""
    N = R_H.shape[0]
    Y = torch.eye(m, device=Hc.device)[y].T
    G = Hc.T @ Hc
    Q = R_H.T @ S @ R_H                       # = (A_0 H)^T (A_0 H)
    P = torch.zeros(m, N, dtype=S.dtype, device=S.device)
    M = torch.zeros(N, N, dtype=S.dtype, device=S.device)
    WH = W0 @ Hc
    for _ in range(T):
        Z = WH + P @ Q @ (torch.eye(N, device=S.device) + M @ G)
        Zs = Z - Z.max(dim=0, keepdim=True).values
        Pr = torch.exp(Zs); Pr = Pr / Pr.sum(dim=0, keepdim=True)
        D = (Pr - Y) / N
        P_new = P - lr * D @ (torch.eye(N, device=S.device) + G @ M.T)
        M = M - lr * P.T @ D
        P = P_new
    return P, M


def make_residual(world, A_T, B_T, W0, y, a):
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T)
    k, N, m = a.k, a.N, a.m
    nW = k * N

    def res(v):
        Wc = v[:nW].reshape(k, N)
        S = tri_to_sym(v[nW:], N)
        Hc = world.features_from_latents(Wc)
        U_c, _ = qr_canon(Hc)
        R_H = U_c.T @ Hc
        P_T, M_T = coeffs_from_Q(R_H, S, W0, Hc, y, m, a.T, a.lr)
        Om = torch.eye(N, device=S.device) + R_H @ M_T @ R_H.T
        X = torch.linalg.solve(Om.T, (A_T @ U_c).T).T          # X = A_T U_c Omega^{-1}
        B_sim = P_T @ (X @ R_H).T                               # B_T = P_T (A_0 H)^T = P_T (X R_H)^T
        r1 = (B_sim - B_T) / nB
        r2 = (S - X.T @ X) / max(float(torch.linalg.norm(S)), 1e-300)
        return torch.cat([r1.reshape(-1), sym_to_tri(r2)])
    return res, nW


def lm(res, v, iters, lam=1e-2, log=print):
    F = res(v); f = float(F @ F); stall = 0; diag = {}
    for it in range(iters):
        J = tf.jacfwd(res)(v).detach()
        JtJ = J.T @ J; JtF = J.T @ F; acc = False
        for _ in range(12):
            step = torch.linalg.solve(JtJ + lam * torch.eye(JtJ.shape[0], device=v.device), JtF)
            vn = v - step; Fn = res(vn)
            if float(Fn @ Fn) < f:
                v, F, f = vn, Fn, float(Fn @ Fn); lam = max(lam / 3, 1e-15); acc = True; break
            lam *= 5
        stall = 0 if acc else stall + 1
        log(f"    lm {it:3d}  residual {f:.3e}  lambda {lam:.1e}")
        if f < 1e-30 or stall >= 2 or lam > 1e12:
            sv = torch.linalg.svdvals(J)
            diag = dict(lm_iters_used=it + 1, jac_sigma_min=float(sv[-1]),
                        jac_cond=float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf"))
            break
    return f, v, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--gen-hidden", type=int, default=32)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, nargs="*", default=[0.10])
    ap.add_argument("--lm-iters", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(a.n)
    log = (lambda s: None) if a.quiet else (lambda s: print(s, flush=True))
    dev = torch.device(a.device)
    world = World(a.k, a.P, a.n, a.seed, dev, a.gen_hidden)
    g = torch.Generator().manual_seed(a.seed + 7)
    N, k, r, m = a.N, a.k, a.r, a.m

    W_true = torch.randn(k, N, generator=g).to(dev)
    X_img = world.psi(W_true); H = world.phi(X_img)
    y = (torch.arange(N) % m).to(dev)
    W0 = (torch.randn(m, a.n, generator=g) / math.sqrt(a.n)).to(dev)
    A0 = (a.sigma0 * torch.randn(r, a.n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, m, a.T, a.lr, "sgd")
    U_true, _ = qr_canon(H); X_true = A0 @ U_true; S_true = X_true.T @ X_true

    res, nW = make_residual(world, A_T, B_T, W0, y, a)
    v_true = torch.cat([W_true.reshape(-1), sym_to_tri(S_true)]).detach()
    gate = float(torch.linalg.norm(res(v_true)))
    n_unk = nW + N * (N + 1) // 2
    print(f"# Q-param: unknowns Nk + N(N+1)/2 = {nW} + {N*(N+1)//2} = {n_unk}  "
          f"(vs Nk + rN = {nW} + {r*N} = {nW + r*N}, a {(nW + r*N)/n_unk:.2f}x reduction)", flush=True)
    print(f"# GATE  ||residual at the truth|| = {gate:.3e}   "
          f"{'PASS' if gate < 1e-10 else 'FAIL -- the elimination is wrong, stop here'}", flush=True)
    if gate >= 1e-10:
        return

    for nz in a.init_noise:
        W_init = W_true + nz * torch.randn(k, N, generator=g).to(dev)
        with torch.no_grad():
            Hc = world.features_from_latents(W_init); Uc, _ = qr_canon(Hc)
            X0 = A_T @ Uc                                    # same crude seed estimate as the (w,X) solver
            S0 = X0.T @ X0
        v = torch.cat([W_init.reshape(-1), sym_to_tri(S0)]).detach()
        t0 = time.time(); f, v, diag = lm(res, v, a.lm_iters, log=log)
        Wh = v[:nW].reshape(k, N)
        err = (torch.linalg.norm(world.psi(Wh) - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
        start = (torch.linalg.norm(world.psi(W_init) - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
        out = dict(param="Q", k=k, N=N, r=r, m=m, n=a.n, T=a.T, lr=a.lr, seed=a.seed,
                   unknowns=n_unk, unknowns_wX=nW + r * N, reduction=(nW + r * N) / n_unk,
                   gate_residual_at_truth=gate, init_noise=nz,
                   start_err_median=float(start.median()), start_err_max=float(start.max()),
                   final_err_max=float(err.max()), final_err_median=float(err.median()),
                   residual=f, recovered=bool(float(err.max()) < RECOVER_TOL),
                   seconds=time.time() - t0, git=git_hash(), host=socket.gethostname(),
                   cmd=" ".join(sys.argv), **diag)
        print(json.dumps(out), flush=True)
        if a.out:
            with open(a.out, "a") as fh: fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
