#!/usr/bin/env python3
"""How big is the solution family? Measure the fibre dimension instead of counting equations.

The approver derived, from equation counting, that the unreduced seed-free arm has a 32-dimensional fibre and that
the reduced arm makes the truth ISOLATED. Checking that raised two problems it could not have known about.

PROBLEM 1 -- the objective this project's E1B scripts actually use is WEAKER than the count assumes. It matches
B_T in full (m*r = 480 equations) but matches A only through the product `A_s @ Hc` against `A_T @ H`
(r*N = 192 equations), not `A_s` against the released `A_T` (r*d = 1536). So the count is 672 equations, not 2016.

PROBLEM 2 -- and this one is worse: the target `A_T @ H` is built from the TRUE H. It is not attacker-available,
and H is the very unknown being solved for.

Matching `A_s` to `A_T` directly is both stronger and attacker-available, so this script measures the local fibre
dimension under BOTH objectives and BOTH parametrisations, as the nullity of the residual Jacobian at the truth.
T is reduced (dimensions do not depend on T) to keep the Jacobian affordable.

  python -m experiments.e1b.fibre_dimension_check
"""
import argparse
import torch
import torch.func as tfn

from experiments.e1b.e1b_tiny import build_release, replay
from experiments.exact_inversion.certificate import certificate

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=40, help="dimensions do not depend on T; kept small for the Jacobian")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    k, N, r, m, P, lr, seed = 12, 8, 24, 20, 64, 0.05, 1
    R = build_release(k, N, r, m, P, a.T, lr, seed, dev)
    H, A0, W0, y, A_T, B_T = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"]
    d = H.shape[0]
    C, _, _ = certificate(A_T, B_T, tol=1e-12)
    U_b, S_b, Vh_b = torch.linalg.svd(B_T, full_matrices=False)
    qb = int((S_b > 1e-10 * float(S_b[0])).sum())
    Pb = Vh_b[:qb].T.contiguous()
    PiA_T = A_T - Pb @ (Pb.T @ A_T)

    print(f"# d={d} N={N} r={r} m={m} T={a.T} (T affects no dimension)   qb = rank B_T = {qb}")
    print(f"# unknowns: H = d*N = {d*N};  A_0 = r*d = {r*d};  reduced seed = qb*d + 1 = {qb*d+1}")
    print(f"# equations: B_T = m*r = {m*r};  full A_T = r*d = {r*d};  product A@H only = r*N = {r*N}\n")

    def sim(Hc, A0c):
        return replay(Hc, A0c, W0, y, m, a.T, lr)

    # --- the two objectives -------------------------------------------------------------------
    def res_product(Hc, A0c):                      # what the E1B scripts use today
        A_s, B_s = sim(Hc, A0c)
        return torch.cat([(B_s - B_T).reshape(-1), (A_s @ Hc - A_T @ H).reshape(-1)])

    def res_full(Hc, A0c):                         # attacker-available and stronger
        A_s, B_s = sim(Hc, A0c)
        return torch.cat([(B_s - B_T).reshape(-1), (A_s - A_T).reshape(-1)])

    def nullity(fn, args, names):
        J = tfn.jacrev(fn, argnums=tuple(range(len(args))))(*args)
        Jm = torch.cat([j.reshape(j.shape[0] if j.dim() > 1 else 1, -1) if False else
                        j.reshape(-1, arg.numel()) for j, arg in zip(J, args)], dim=1)
        s = torch.linalg.svdvals(Jm)
        tol = 1e-10 * float(s[0])
        rk = int((s > tol).sum())
        n_in = Jm.shape[1]
        return Jm.shape, rk, n_in - rk, float(s[rk - 1]), float(s[rk]) if rk < len(s) else 0.0

    for label, fn in (("product A@H  (current)", res_product), ("full A_T     (proposed)", res_full)):
        shp, rk, nul, slast, snext = nullity(fn, (H, A0), ("H", "A0"))
        print(f"UNREDUCED  {label}:  J {shp[0]}x{shp[1]}   rank {rk}   NULLITY {nul}"
              f"   gap {slast:.2e} -> {snext:.2e}")

    # --- reduced parametrisation: A_0 = PiA_T/c + Pb M ----------------------------------------
    M_true = Pb.T @ A0
    c_true = torch.ones((), device=dev, dtype=H.dtype)

    def make_red(res):
        def f(Hc, M, c):
            return res(Hc, PiA_T / c + Pb @ M)
        return f

    for label, fn in (("product A@H  (current)", res_product), ("full A_T     (proposed)", res_full)):
        shp, rk, nul, slast, snext = nullity(make_red(fn), (H, M_true, c_true), ("H", "M", "c"))
        print(f"REDUCED    {label}:  J {shp[0]}x{shp[1]}   rank {rk}   NULLITY {nul}"
              f"   gap {slast:.2e} -> {snext:.2e}")

    print("\n# NULLITY is the local dimension of the family of (H, seed) reproducing the release.")
    print("# 0 means the truth is locally ISOLATED and recovery is well posed; >0 means a solution family")
    print("# exists and no solver can pick the truth out of it without further information.")


if __name__ == "__main__":
    main()
