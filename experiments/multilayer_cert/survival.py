#!/usr/bin/env python3
"""M1 + M2 + M3: does the certificate survive with depth, and does each layer add information?

For every adapted layer this measures, per configuration:

  drift        delta_l          = max_t ||H_{l,t} - H_l^0|| / ||H_l^0||             (the brief's eps)
               delta_l_perp     = max_t ||P_{col(H_l^0)^perp}(H_{l,t}-H_l^0)|| / ||H_l^0||   (theory/T3.2: the
                                  quantity that actually controls the error -- in-span drift is FREE)
               drift_rank       = numerical rank of [Delta_{l,0} ... Delta_{l,T-1}]  (theory/T4-C1: what inflates N')
  release      N_prime, rank_B_T, gap = sigma_N/sigma_{N+1}, beta_l = s||B A||_op
  certificate  rho_full   = ||C_full H_l^0||/(||C_full|| ||H_l^0||)   -> theory/T2 says MACHINE ZERO whenever
                            rank B_T == N', at ANY drift.  A nonzero value here falsifies Prop. A.
               rho_trunc  = same for the rank-(r-N) truncated certificate -> theory/T3 says O(delta_perp), slope 1
               rank_C_full, expected r - N_prime
  information  q_l (rank of C_l J_{Phi_l} J_G at the chart) and the stacked rank over layers 1..l  (M3)

Sweeps drift through lr and T (M2) rather than through a proxy, and records the actual adapter norms.

    python -u -m experiments.multilayer_cert.survival --sweep --save --device cuda
"""
import argparse, json, math, os, socket, subprocess, sys, time
import torch

from experiments.multilayer_cert.common import (MultiLoRANet, certificate, numrank, rel_annihilation, span_of,
                                                provenance)

torch.set_default_dtype(torch.float64)


def log(s): print(s, flush=True)


def _safe_beta(net, Al, Bl):
    """s||B A||_op, robust to a finite-but-overflowing product (svdvals raises on non-finite input)."""
    try:
        M = Bl @ Al
        if not torch.isfinite(M).all():
            return float("inf")
        return float(net.s * torch.linalg.matrix_norm(M, 2))
    except Exception:
        return float("inf")


def measure(net, X, y, T, lr, V, mean, zstar, k, tol=1e-10, drift_max=1e3, beta_max=1e6):
    A, B, reps = net.train(X, y, T=T, lr=lr)
    Hb = net.base_reps(X)
    # H_{l,T}: the representations AFTER the final update. The harness never computed these -- reps holds only
    # t = 0..T-1 (pre-update), while the certificate is built from the post-update A,B. Record and gate on them.
    reps_final = net.forward_reps(X, A, B)
    all_reps = reps + [reps_final]
    H0_norm = [float(Hb[l].norm()) for l in range(net.L)]
    # UNDERFLOW test specifically (not mere smallness): a base norm at 0 makes any relative drift inf/nan and is
    # the leading candidate for the one inf-drift row that no finite ||H0|| can explain.
    H0_underflow = [bool(H0_norm[l] == 0.0 or H0_norm[l] < 1e-290) for l in range(net.L)]
    finite = (all(torch.isfinite(h).all() for rt in all_reps for h in rt)
              and all(torch.isfinite(t).all() for t in A + B))
    beta = [_safe_beta(net, A[l], B[l]) for l in range(net.L)]

    def rel_drift_max_l(l):
        if H0_norm[l] == 0.0:
            return float("inf")
        return max(float((rt[l] - Hb[l]).norm() / H0_norm[l]) for rt in all_reps)
    rel = [rel_drift_max_l(l) if finite else float("inf") for l in range(net.L)]
    max_rel = max(rel) if rel else float("inf")
    max_beta = max(beta) if beta else float("inf")
    # GATE on the RELATIVE drift the row reports and on the adapter norm beta -- NOT on entry magnitude (fix,
    # 2026-09-17, GM/c4). The old abs gate max|entry|>1e100 has two blind spots: (a) entries can stay below 1e100
    # while ||H0|| is tiny and relative drift reaches ~1e103 (six of eight exploded rows passed it legitimately);
    # (b) a layer-0 explosion lives entirely in beta (delta=0 because its input is the frozen data) where no
    # representation can see it. Thresholds scoped to the physical scale: the intended band tops at ~9 (923%
    # drift) / beta ~573; the exploded population starts at ~1.2e20 / beta ~1.27e21, an 18-order gap, so 1e3 and
    # 1e6 separate cleanly and exclude no legitimate large-drift cell.
    diverged = (not finite) or (not math.isfinite(max_rel)) or (max_rel > drift_max) or (max_beta > beta_max)
    if diverged:
        return [dict(layer=l, diverged=True, H0_norm=H0_norm[l], H0_underflow=H0_underflow[l],
                     rel_drift_max=(rel[l] if math.isfinite(rel[l]) else None),
                     beta=(beta[l] if math.isfinite(beta[l]) else None)) for l in range(net.L)], A, B
    G = lambda z: mean + V @ z.reshape(k, -1)
    rows, blocks = [], []
    for l in range(net.L):
        H0 = Hb[l]
        P0 = torch.linalg.qr(H0)[0][:, :H0.shape[1]]                       # basis of col(H_l^0)
        D = [rt[l] - H0 for rt in reps]                                    # t < T: the theory training-span drift
        delta = max(float(d.norm() / H0.norm()) for d in D)
        dperp = max(float((d - P0 @ (P0.T @ d)).norm() / H0.norm()) for d in D)
        delta_final = float((reps_final[l] - H0).norm() / H0.norm())       # the post-final-update drift, recorded
        drank = span_of(D)[0] if delta > 0 else 0
        Nprime, _ = span_of([rt[l] for rt in reps])                        # training span is over t < T only
        Cf, q, sB = certificate(A[l], B[l], tol=tol)
        Ct, _, _ = certificate(A[l], B[l], keep=net.N, tol=tol)
        gap = float(sB[net.N - 1] / sB[net.N]) if len(sB) > net.N and float(sB[net.N]) > 0 else float("inf")
        rkC, _ = numrank(Cf, ref=float(A[l].norm()))
        Fl = lambda z, l=l, C=Ct: (C @ net.base_reps(G(z))[l][:, :1]).reshape(-1)
        J = torch.autograd.functional.jacobian(Fl, zstar.reshape(-1))
        blocks.append(J)
        ql, _ = numrank(J, rtol=1e-9)
        stacked, _ = numrank(torch.cat(blocks, 0), rtol=1e-9)
        rows.append(dict(layer=l, diverged=False, delta=delta, delta_perp=dperp, delta_final=delta_final,
                         drift_rank=drank, N_prime=Nprime, rank_B=q, B3_holds=bool(q == Nprime), gap=gap,
                         beta=beta[l], H0_norm=H0_norm[l], H0_underflow=H0_underflow[l], rank_C_full=rkC,
                         expect_rank_C=max(0, min(net.r - Nprime, H0.shape[0] - Nprime)),
                         rho_full=rel_annihilation(Cf, H0), rho_trunc=rel_annihilation(Ct, H0),
                         q_l=ql, rank_stacked=stacked))
    return rows, A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--din", type=int, default=40)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--sweep", action="store_true", help="M2: sweep lr x T")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/multilayer_cert/survival.jsonl")
    a = ap.parse_args()

    dims = [a.din] + [a.width] * (a.L - 1) + [a.classes]
    grid = [(lr, T) for lr in ([0.01, 0.03, 0.1, 0.3, 1.0, 3.0] if a.sweep else [a.lr])
            for T in ([2, 4, 8] if a.sweep else [a.T])]
    PROV = provenance(__file__)                          # attested provenance: commit (+ -dirty) AND a script hash
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    log(f"# multilayer survival | dims={dims} r={a.r} N={a.N} k={a.k} | {len(grid)} cells x {a.seeds} seeds "
        f"| device={a.device} | host={socket.gethostname()}")
    t0, out_rows, tensors = time.time(), [], {}
    for seed in range(a.seeds):
        net = MultiLoRANet(dims, a.r, a.N, seed=seed, dev=a.device)
        gg = torch.Generator().manual_seed(seed + 3)
        V = torch.linalg.qr(torch.randn(dims[0], a.k, generator=gg))[0].to(a.device)
        mean = torch.randn(dims[0], 1, generator=gg).to(a.device)
        Z = torch.randn(a.k, a.N, generator=gg).to(a.device)
        X = mean + V @ Z
        y = torch.arange(a.N) % dims[-1]
        for (lr, T) in grid:
            rows, A, B = measure(net, X, y, T, lr, V, mean, Z[:, :1], a.k)
            with open(a.out, "a") as f:                      # append per cell, not at the end
                for rw in rows:
                    rec = dict(rw, seed=seed, lr=lr, T=T, dims=dims, r=a.r, N=a.N, k=a.k,
                               git=PROV["git"], script_sha=PROV["script_sha"],
                               host=socket.gethostname(), cmd=" ".join(sys.argv))
                    out_rows.append(rec)
                    f.write(json.dumps(rec) + "\n")
            if rows[0].get("diverged"):
                log(f"  seed={seed} lr={lr:<5} T={T}: DIVERGED (trajectory non-finite) -- recorded, skipped")
                continue
            worst_full = max(rw["rho_full"] for rw in rows if rw["B3_holds"]) if any(rw["B3_holds"] for rw in rows) else float("nan")
            log(f"  seed={seed} lr={lr:<5} T={T}: "
                + " | ".join(f"l{rw['layer']} d~{rw['delta']:.1e}/{rw['delta_perp']:.1e} N'={rw['N_prime']}"
                             f" rkC={rw['rank_C_full']} full={rw['rho_full']:.1e} trunc={rw['rho_trunc']:.1e}"
                             f" q={rw['q_l']}" for rw in rows)
                + f"  || worst rho_full (B3 cells) = {worst_full:.1e}")
            if a.save:
                tensors[f"s{seed}_lr{lr}_T{T}"] = dict(A=[t.cpu() for t in A], B=[t.cpu() for t in B],
                                                       X=X.cpu(), V=V.cpu(), mean=mean.cpu(), Z=Z.cpu())
    if a.save:
        pth = a.out.replace(".jsonl", ".pth")
        torch.save(tensors, pth)
        log(f"# tensors -> {pth}")
    # headline falsification test: Prop. A must hold at EVERY drift wherever B3 holds
    bad = [rw for rw in out_rows if not rw.get("diverged") and rw["B3_holds"] and rw["rho_full"] > 1e-11]
    log(f"\n# {len(out_rows)} rows -> {a.out} in {time.time()-t0:.1f}s")
    log(f"# PROP-A FALSIFIED in {len(bad)} rows" if bad else
        "# Prop. A holds in every row where rank B_T == N' (rho_full < 1e-11 at ALL drifts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
