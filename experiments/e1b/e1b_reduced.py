#!/usr/bin/env python3
"""E1B arm 3 / E3 -- the REDUCED seed-free arm: the release already pins most of the seed, so stop searching all of it.

The unreduced seed-free arm (job 670993) searches all of A_0: r*d unknowns. Quotient sensing says the component of
the seed orthogonal to row(B_T) is fixed by the release up to ONE scalar,

    Pi A_T = c_T Pi A_0,    Pi := P_{row(B_T)^perp},    and Pi A_T is exactly the certificate C,

because every SGD update to A has its columns inside row(B_T) and Pi annihilates them. So write

    A_0 = C / c  +  P_b M,   P_b an orthonormal basis of row(B_T) (r x qb),  M free (qb x d),  c a free scalar,

which contains the truth whenever the identity holds, and costs qb*d + 1 unknowns instead of r*d.

THE GATE COMES FIRST. With the known seed we check ||Pi A_T - c_hat Pi A_0|| / ||Pi A_T|| <= 1e-10 on this fp64
release. If it fails, the reduced arm is searching the WRONG subspace and its failure would mean nothing, so the
script stops at the gate and says so rather than reporting a null.

MATCHED TO THE UNREDUCED ARM ON PURPOSE: same release, same start generator (seed+31), same 60 Gaussian + 60 ker-C
starts, same optimiser and step count. Initialised at M = P_b^T A_T and c = 1, so the initial seed estimate is
exactly A_T -- the same starting point the unreduced arm uses. The only difference between the arms is the size of
the space searched, which is what E3 is asking about.

SCOPE, and it is written into the row rather than left to the reader: phi = identity on this release, so feature
space and image space coincide. A landing here says the DYNAMICS invert from H. It says nothing about whether free
feature replay lands on representations no image produces -- this cell cannot exhibit that failure by construction.

  python -m experiments.e1b.e1b_reduced
"""
import argparse, json, os, socket, sys, time
import numpy as np
import torch

from experiments.e1b.e1b_tiny import build_release, replay, log
from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.certificate import certificate

torch.set_default_dtype(torch.float64)

SCOPE = ("phi = identity on this release, so Z_feature and Z_image coincide. A landing certifies that the DYNAMICS "
         "are invertible from H; it certifies NOTHING about free feature replay landing on representations no image "
         "produces, which this cell cannot exhibit by construction. The real-backbone cell is the other half.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=24); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--P", type=int, default=64); ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--starts", type=int, default=60); ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--gate-tol", type=float, default=1e-10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results/e1b")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.out_dir, exist_ok=True)

    R = build_release(a.k, a.N, a.r, a.m, a.P, a.T, a.lr, a.seed, dev)
    H, A0, W0, y, A_T, B_T = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"]
    d, N = H.shape
    C, _, _ = certificate(A_T, B_T, tol=1e-12)
    rank_H = int(torch.linalg.matrix_rank(H, rtol=1e-10))
    log(f"# E1B reduced (E3)  d={d} N={N} r={a.r} m={a.m} T={a.T} seed={a.seed}  fp64")
    log(f"# A1: rank H = {rank_H}, N = {N} -> metric in force: {'per-image' if rank_H == N else 'SUBSPACE'}")
    assert rank_H == N, f"A1: rank H = {rank_H} != N = {N}; per-image metrics do not exist here"

    # ---- basis of row(B_T) and its complement -------------------------------------------------
    U_b, S_b, Vh_b = torch.linalg.svd(B_T, full_matrices=False)
    qb = int((S_b > 1e-10 * S_b[0]).sum())
    Pb = Vh_b[:qb].T.contiguous()                       # (r, qb), orthonormal, spans row(B_T)
    Pi = lambda X: X - Pb @ (Pb.T @ X)                  # projection off row(B_T)
    PiA_T, PiA_0 = Pi(A_T), Pi(A0)

    # ---- THE GATE: Pi A_T = c_T Pi A_0 --------------------------------------------------------
    c_T = float((PiA_T * PiA_0).sum() / (PiA_0 * PiA_0).sum())
    gate = float(torch.linalg.norm(PiA_T - c_T * PiA_0) / torch.linalg.norm(PiA_T))
    cert_rel = float(torch.linalg.norm(C - PiA_T) / torch.linalg.norm(PiA_T))
    log(f"# rank B_T = qb = {qb} (of r = {a.r}); unknowns {qb * d + 1} reduced vs {a.r * d} unreduced "
        f"({a.r * d / (qb * d + 1):.1f}x)")
    log(f"# E3 GATE: c_T = {c_T:.12f}   ||Pi A_T - c_T Pi A_0|| / ||Pi A_T|| = {gate:.3e}   (tol {a.gate_tol:.0e})")
    log(f"#          [definitional, NOT evidence: C is DEFINED as Pi A_T; agreement {cert_rel:.3e} only checks the code path]")
    log(f"#          The measured content is the line above: Pi A_T = c_T Pi A_0, a fact about the TRAINING MAP, which is")
    log(f"#          what licenses the reduced parametrisation. The definitional identity must never be quoted as a result.")
    passed = gate <= a.gate_tol
    if not passed:
        log("# GATE FAILED -> the reduced arm would search the wrong subspace; NOT running it. This is the finding.")
        row = dict(part="E1B-reduced", arm="seed_free_reduced", gate_passed=False, gate_residual=gate, c_T=c_T,
                   qb=qb, note="reduced arm not run: quotient-sensing identity does not hold numerically here",
                   scope=SCOPE, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
        with open(os.path.join(a.out_dir, "rows.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True); return

    nB = torch.linalg.norm(B_T); nAH = torch.linalg.norm(A_T @ H)

    def seed_of(M, c):
        return PiA_T / c + Pb @ M

    def resid(Hc, A0c):
        A_s, B_s = replay(Hc, A0c, W0, y, a.m, a.T, a.lr)
        return torch.cat([((B_s - B_T) / nB).reshape(-1), ((A_s @ Hc - A_T @ H) / nAH).reshape(-1)])

    with torch.no_grad():
        M_true = Pb.T @ A0
        r_truth_full = float(torch.linalg.norm(resid(H, A0)))
        r_truth_red = float(torch.linalg.norm(resid(H, seed_of(M_true, c_T))))
        seed_gap = float(torch.linalg.norm(seed_of(M_true, c_T) - A0) / torch.linalg.norm(A0))
    log(f"# residual at the truth: {r_truth_full:.3e} (true A_0) vs {r_truth_red:.3e} (truth EXPRESSED in the reduced "
        f"family; seed reconstruction error {seed_gap:.3e}) -- the family contains the truth")

    gs = torch.Generator().manual_seed(a.seed + 31)         # SAME starts as the unreduced arms
    scale = float(H.norm(dim=0).median())
    G0 = torch.randn(a.starts, d, N, generator=gs).to(dev); G0 = G0 / G0.norm(dim=1, keepdim=True) * scale
    rank_C = int(torch.linalg.matrix_rank(C, rtol=1e-10))
    _, _, Vh = torch.linalg.svd(C, full_matrices=True)
    ker = Vh[rank_C:].T.contiguous()
    K0 = torch.stack([ker @ (ker.T @ G0[i]) for i in range(a.starts)])
    K0 = K0 / K0.norm(dim=1, keepdim=True).clamp(min=1e-30) * scale

    rows = []; t_all = time.time()
    for fam, S0 in (("gaussian", G0), ("kerC", K0)):
        for s in range(a.starts):
            Hc = S0[s].clone().requires_grad_(True)
            M = (Pb.T @ A_T).clone().requires_grad_(True)      # c=1, M=Pb^T A_T  =>  seed starts exactly at A_T
            c = torch.ones((), device=dev, dtype=H.dtype).requires_grad_(True)
            opt = torch.optim.Adam([Hc, M, c], 5e-2); t0 = time.time()
            for _ in range(a.iters):
                f = resid(Hc, seed_of(M, c)); loss = f @ f
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                A0h = seed_of(M, c); f = resid(Hc, A0h); obj = float(torch.linalg.norm(f))
                Dm = torch.cdist(Hc.T, H.T) / H.norm(dim=0)[None, :]
                assign = Dm.argmin(1); per_err = [float(Dm[i, assign[i]]) for i in range(N)]
                cos = [float(torch.nn.functional.cosine_similarity(Hc[:, i], H[:, assign[i]], dim=0)) for i in range(N)]
                covered = len(set(int(x) for x in assign))
                rows.append(dict(family=fam, start=s, objective=obj, per_image_err=per_err, cosine=cos,
                                 truths_covered=covered, landed=bool(max(per_err) < 1e-2 and covered == N),
                                 c_hat=float(c), c_err=abs(float(c) - c_T) / abs(c_T),
                                 seed_err=float(torch.linalg.norm(A0h - A0) / torch.linalg.norm(A0)),
                                 seconds=time.time() - t0))
            if (s + 1) % 20 == 0:
                sub = [r for r in rows if r["family"] == fam]
                log(f"   {fam} {s+1}/{a.starts}: median objective {np.median([r['objective'] for r in sub]):.2e}, "
                    f"landings {sum(r['landed'] for r in sub)}")

    land = [r for r in rows if r["landed"]]
    row = dict(part="E1B-reduced", arm="seed_free_reduced", release="affine_two_routes (331384 generator)",
               precision="fp64", gate_passed=True, gate_residual=gate, c_T=c_T,
               cert_is_PiA_T_definitional_check=cert_rel,
               cert_note="C is DEFINED as Pi A_T; this field is a code-path check, NOT evidence. The measurement is gate_residual.",
               qb=qb, unknowns_reduced=qb * d + 1, unknowns_unreduced=a.r * d,
               d=d, N=N, r=a.r, m=a.m, T=a.T, lr=a.lr, seed=a.seed, rank_H=rank_H, metric_in_force="per-image",
               residual_at_truth=r_truth_full, residual_at_truth_in_family=r_truth_red, seed_family_gap=seed_gap,
               starts_per_family=a.starts,
               landings={f: sum(1 for r in rows if r["family"] == f and r["landed"]) for f in ("gaussian", "kerC")},
               objective_median={f: float(np.median([r["objective"] for r in rows if r["family"] == f])) for f in ("gaussian", "kerC")},
               objective_min=float(min(r["objective"] for r in rows)),
               best_max_per_image_err=float(min(max(r["per_image_err"]) for r in rows)),
               c_err_of_landings=[r["c_err"] for r in land] or None,
               seed_err_of_landings=[r["seed_err"] for r in land] or None,
               truths_covered_max=max(r["truths_covered"] for r in rows),
               compare_note="matched to the UNREDUCED seed-free arm (job 670993): same release, same start generator, "
                            "same optimiser and iteration count, same initial seed estimate A_T. The only difference "
                            "is the dimension of the seed search space.",
               manifold_test="VACUOUS ON THIS RELEASE (phi = identity); not a measurement.",
               scope=SCOPE, seconds=time.time() - t_all, runs=rows, git=git_hash(),
               host=socket.gethostname(), cmd=" ".join(sys.argv))
    with open(os.path.join(a.out_dir, "rows.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")
    torch.save(dict(H=H.cpu(), A_T=A_T.cpu(), B_T=B_T.cpu(), C=C.cpu(), A0=A0.cpu(), row=row),
               os.path.join(a.out_dir, "e1b_reduced.pth"))
    log(f"\n=== E1B reduced: landings {row['landings']} of {a.starts} per family, best max-per-image error "
        f"{row['best_max_per_image_err']:.3e}, objective min {row['objective_min']:.2e} vs {r_truth_full:.2e} at the truth")
    print(json.dumps({k: v for k, v in row.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
