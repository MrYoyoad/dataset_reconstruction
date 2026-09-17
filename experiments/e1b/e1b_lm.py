#!/usr/bin/env python3
"""E1B v2 -- the SAME question with the solver 331384 actually used, and an attacker-available residual.

WHY THIS EXISTS. E1B v1 (jobs 670990 / 670993 / 675031) returned ZERO landings from 120 starts in all three
arms, at an objective ~3.5e-02 against 8.75e-16 at the truth: fourteen orders short, so the solver never reached a
zero at all. But job 331384 recovers all eight on the SAME release from 19 of 60 starts. v1 differed from it in
TWO ways at once, so its failure was not attributable:

  1. SOLVER.  331384 uses Levenberg-Marquardt (`invert_lm`); v1 used Adam, first order, 400 iterations.
  2. PARAMETRISATION.  331384 solves for CHART LATENTS (k=12 per image, 96 unknowns); v1 solves for FREE H in
     R^{64x8} (512 unknowns) -- which is the point of E1B, but it removes the chart at the same time.

This run fixes the solver to LM so that only the parametrisation differs, which is the comparison E1B is for:
chart-constrained replay (331384, 19/60) against free-feature replay (here). A failure here with LM means the
CHART is doing the work, not the optimiser.

AND IT FIXES THE RESIDUAL, which v1 got wrong in a way worth recording. v1 matched `A_s @ Hc` against `A_T @ H` --
r*N = 192 equations, and the target is built from the TRUE H, so it is not attacker-available and H is the very
unknown being solved for. This matches `A_s` against the released `A_T` directly: r*d = 1536 equations, using only
released quantities. Stronger AND legitimate. Equation counts, d=64 N=8 r=24 m=20:

  residual   B_T (m*r = 480) + A_T (r*d = 1536)            = 2016 equations
  unknowns   seed_known H = 512  |  seed_free H + A_0 = 2048  |  reduced H + M + c = 1025

So seed_known is overdetermined 2016 vs 512; seed_free is underdetermined by 32; the reduced arm is
overdetermined, BUT note that 1024 of the A_T equations are consumed by pinning Pi A_0, so its effective count is
992 against 1025 -- a deficit of 33, of which 1 is the deliberate c slack. The reduction is therefore NOT an
identifiability fix, contrary to what a naive intersection argument suggests: every point of the solution family
already satisfies Pi A_0 = Pi A_T, so the family lies INSIDE the reduced slice rather than transverse to it.
Measured directly by experiments/e1b/fibre_dimension_check.py.

  python -m experiments.e1b.e1b_lm --arm seed_known
"""
import argparse, json, os, socket, sys, time
import numpy as np
import torch
import torch.func as tfn

from experiments.e1b.e1b_tiny import build_release, replay, log
from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.certificate import certificate

torch.set_default_dtype(torch.float64)

SCOPE = ("phi = identity on this release, so Z_feature and Z_image coincide. A landing certifies that the DYNAMICS "
         "are invertible from a free H; it certifies nothing about free feature replay landing on representations "
         "no image produces, which this cell cannot exhibit by construction.")


def lm_solve(fun, x0, iters=60, lam0=1e-3, lam_min=1e-14, lam_max=1e12):
    """Levenberg-Marquardt on a least-squares residual -- the solver 331384 uses, on E1B's parametrisation."""
    x = x0.clone()
    r = fun(x); cost = float(r @ r); lam = lam0
    n = x.numel()
    for _ in range(iters):
        J = tfn.jacfwd(fun)(x).reshape(-1, n)
        g = J.T @ r
        JTJ = J.T @ J
        diag = torch.diagonal(JTJ).clamp(min=1e-300)
        step_taken = False
        for _ in range(12):                       # damping search
            try:
                delta = torch.linalg.solve(JTJ + lam * torch.diag(diag), -g)
            except Exception:
                lam = min(lam * 10, lam_max); continue
            xn = x + delta
            rn = fun(xn); cn = float(rn @ rn)
            if cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam / 10, lam_min); step_taken = True; break
            lam = min(lam * 10, lam_max)
        if not step_taken or cost < 1e-28:
            break
    return x, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["seed_known", "seed_free", "reduced"], required=True)
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=24); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--P", type=int, default=64); ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--starts", type=int, default=20); ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results/e1b")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.out_dir, exist_ok=True)

    R = build_release(a.k, a.N, a.r, a.m, a.P, a.T, a.lr, a.seed, dev)
    H, A0, W0, y, A_T, B_T = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"]
    d, N = H.shape
    C, _, _ = certificate(A_T, B_T, tol=1e-12)
    rank_H = int(torch.linalg.matrix_rank(H, rtol=1e-10))
    rank_C = int(torch.linalg.matrix_rank(C, rtol=1e-10))
    assert rank_H == N, f"A1: rank H = {rank_H} != N = {N}"
    U_b, S_b, Vh_b = torch.linalg.svd(B_T, full_matrices=False)
    qb = int((S_b > 1e-10 * float(S_b[0])).sum())
    Pb = Vh_b[:qb].T.contiguous()
    PiA_T = A_T - Pb @ (Pb.T @ A_T)
    nB, nA = torch.linalg.norm(B_T), torch.linalg.norm(A_T)

    log(f"# E1B-LM arm={a.arm}  d={d} N={N} r={a.r} m={a.m} T={a.T} seed={a.seed}  fp64  solver=Levenberg-Marquardt")
    log(f"# q = r - rank C = {a.r - rank_C}   N = {N}   rank H = {rank_H}   qb = rank B_T = {qb}")
    log(f"# residual matches FULL A_T (r*d={a.r*d}) and B_T (m*r={a.m*a.r}) = {a.r*d + a.m*a.r} equations, "
        f"all attacker-available")

    def unpack(x):
        Hc = x[:d * N].reshape(d, N)
        if a.arm == "seed_known":
            return Hc, A0
        if a.arm == "seed_free":
            return Hc, x[d * N:].reshape(a.r, d)
        M = x[d * N:d * N + qb * d].reshape(qb, d); c = x[-1]
        return Hc, PiA_T / c + Pb @ M

    def fun(x):
        Hc, A0c = unpack(x)
        A_s, B_s = replay(Hc, A0c, W0, y, a.m, a.T, a.lr)
        return torch.cat([((B_s - B_T) / nB).reshape(-1), ((A_s - A_T) / nA).reshape(-1)])

    def pack(Hc, A0c=None, M=None, c=None):
        parts = [Hc.reshape(-1)]
        if a.arm == "seed_free": parts.append(A0c.reshape(-1))
        if a.arm == "reduced": parts += [M.reshape(-1), c.reshape(1)]
        return torch.cat(parts)

    with torch.no_grad():
        x_true = pack(H, A0, Pb.T @ A0, torch.ones((), device=dev, dtype=H.dtype))
        r_truth = float(torch.linalg.norm(fun(x_true)))
    log(f"# residual at the truth (this parametrisation): {r_truth:.3e}   unknowns = {x_true.numel()}")

    gs = torch.Generator().manual_seed(a.seed + 31)
    scale = float(H.norm(dim=0).median())
    G0 = torch.randn(a.starts, d, N, generator=gs).to(dev); G0 = G0 / G0.norm(dim=1, keepdim=True) * scale
    _, _, Vh = torch.linalg.svd(C, full_matrices=True)
    ker = Vh[rank_C:].T.contiguous()
    K0 = torch.stack([ker @ (ker.T @ G0[i]) for i in range(a.starts)])
    K0 = K0 / K0.norm(dim=1, keepdim=True).clamp(min=1e-30) * scale

    rows = []; t_all = time.time()
    for fam, S0 in (("gaussian", G0), ("kerC", K0)):
        for s in range(a.starts):
            t0 = time.time()
            x0 = pack(S0[s], A_T, Pb.T @ A_T, torch.ones((), device=dev, dtype=H.dtype))
            xh, cost = lm_solve(fun, x0, iters=a.iters)
            with torch.no_grad():
                Hc, A0c = unpack(xh)
                obj = float(torch.linalg.norm(fun(xh)))
                Dm = torch.cdist(Hc.T, H.T) / H.norm(dim=0)[None, :]
                assign = Dm.argmin(1); per_err = [float(Dm[i, assign[i]]) for i in range(N)]
                covered = len(set(int(x) for x in assign))
                seed_err = float(torch.linalg.norm(A0c - A0) / torch.linalg.norm(A0))
            rows.append(dict(family=fam, start=s, objective=obj, per_image_err=per_err,
                             truths_covered=covered, seed_err=seed_err,
                             landed=bool(max(per_err) < 1e-2 and covered == N), seconds=time.time() - t0))
            if (s + 1) % 5 == 0:
                sub = [r for r in rows if r["family"] == fam]
                log(f"   {fam} {s+1}/{a.starts}: median objective {np.median([r['objective'] for r in sub]):.3e}, "
                    f"min {min(r['objective'] for r in sub):.3e}, landings {sum(r['landed'] for r in sub)}, "
                    f"{np.mean([r['seconds'] for r in sub]):.0f}s/start")
    land = [r for r in rows if r["landed"]]
    row = dict(part="E1B-LM", arm=a.arm, solver="levenberg-marquardt", release="affine_two_routes (331384 generator)",
               precision="fp64", residual="full A_T + B_T, attacker-available", n_equations=a.r * d + a.m * a.r,
               n_unknowns=int(x_true.numel()), d=d, N=N, r=a.r, m=a.m, T=a.T, lr=a.lr, seed=a.seed, qb=qb,
               rank_H=rank_H, rank_C=rank_C, residual_at_truth=r_truth, starts_per_family=a.starts, lm_iters=a.iters,
               landings={f: sum(1 for r in rows if r["family"] == f and r["landed"]) for f in ("gaussian", "kerC")},
               objective_median={f: float(np.median([r["objective"] for r in rows if r["family"] == f])) for f in ("gaussian", "kerC")},
               objective_min=float(min(r["objective"] for r in rows)),
               best_max_per_image_err=float(min(max(r["per_image_err"]) for r in rows)),
               seed_err_of_landings=[r["seed_err"] for r in land] or None,
               truths_covered_max=max(r["truths_covered"] for r in rows),
               compare_note="331384 recovers all eight from 19/60 on THIS release with LM but solving for k=12 CHART "
                            "LATENTS. This run holds the solver fixed at LM and frees H in R^d, so the only "
                            "difference is the parametrisation: a failure here isolates the CHART as what does the work.",
               scope=SCOPE, seconds=time.time() - t_all, runs=rows, git=git_hash(),
               host=socket.gethostname(), cmd=" ".join(sys.argv))
    with open(os.path.join(a.out_dir, "rows.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")
    log(f"\n=== E1B-LM {a.arm}: landings {row['landings']} of {a.starts}/family, objective min "
        f"{row['objective_min']:.3e} vs {r_truth:.3e} at the truth, best max-per-image error "
        f"{row['best_max_per_image_err']:.3e}")
    print(json.dumps({k: v for k, v in row.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
