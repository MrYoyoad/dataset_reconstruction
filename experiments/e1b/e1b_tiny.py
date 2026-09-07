#!/usr/bin/env python3
"""E1B-tiny -- free feature replay: forget the images, make H the variable, ask whether the dynamics invert.

Release: the affine two-routes cell (job 331384), which §12 names and which is fp64 with a known seed.
d = 64, N = 8, r = 24, m = 20, T = 400, lr = 0.05, seed 1. Rebuilt from its generator.

THREE FIXES FROM THE PLAN AUDIT, applied rather than worked around:

  A1  q = N is ASSERTED before the solve. With q < N the per-image metrics do not exist -- q columns are a spanning
      set, not the training representations, and there is no correspondence to difference. rank C = 16 against
      r = 24 gives q = 8 = N here, and the assertion is on rank H itself. If it fails the metric in force is a
      subspace distance and the row says so.

  A2  The SCALE PRE-CHECK runs FIRST and needs no solve. Rescaling h -> alpha h with A0 -> A0/alpha leaves the
      adapter path A0 h fixed but sends the frozen base path to alpha W0 h, so the logits, the softmax residuals and
      hence B_T all move: the symmetry is BROKEN whenever W0 != 0. Registered prediction: the replay residual along
      (alpha h*, A0/alpha) is NOT flat and the seed-free arm returns ||h_hat||/||h|| ~ 1. If it IS flat the
      derivation is wrong and that is reported before any solve is trusted.

  A5  The manifold test is reported for what it is HERE. phi = identity on this release, so range(Phi0) = R^d and
      min_x ||Phi0(x) - h_hat|| is identically zero for every candidate BY CONSTRUCTION. Reported as vacuous with
      the reason, never as a zero. The real-backbone cell is a separate addition.

Arms: seed known (A0 given) and seed free (A0 unknown, initialised at the released A_T -- attacker-available).
Starts: 60 Gaussian in feature space + 60 projected onto ker C, matched across arms.

  python -m experiments.e1b.e1b_tiny --arm seed_known
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch

from experiments.exact_inversion.lora_exact_inversion import train_release, git_hash
from experiments.exact_inversion.certificate import certificate

torch.set_default_dtype(torch.float64)


def log(s): print(s, flush=True)


def build_release(k, N, r, m, P, T, lr, seed, dev):
    """The affine two-routes world, rebuilt exactly: psi affine, phi identity, so the adapted layer IS the input."""
    g = torch.Generator().manual_seed(seed)
    L = (torch.randn(P, k, generator=g) / math.sqrt(k)).to(dev)
    b = (0.7 * torch.randn(P, generator=g)).to(dev)
    W_true = torch.randn(k, N, generator=g).to(dev)
    H = L @ W_true + b[:, None]
    A0 = ((1.0 / math.sqrt(P)) * torch.randn(r, P, generator=g)).to(dev)
    W0 = (torch.randn(m, P, generator=g) / math.sqrt(P)).to(dev)
    y = torch.arange(N, device=dev) % m
    A_T, B_T = train_release(H, A0, W0, y, m, T=T, lr=lr, release="sgd")
    return dict(H=H, A0=A0, W0=W0, y=y, A_T=A_T, B_T=B_T, L=L, b=b)


def replay(H, A0, W0, y, m, T, lr):
    N = H.shape[1]; r = A0.shape[0]
    Y = torch.eye(m, device=H.device, dtype=H.dtype)[y].T
    A, B = A0, torch.zeros(m, r, device=H.device, dtype=H.dtype)
    for _ in range(T):
        z = W0 @ H + B @ (A @ H); D = (torch.softmax(z, 0) - Y) / N
        B, A = B - lr * (D @ (A @ H).T), A - lr * (B.T @ D @ H.T)
    return A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["seed_known", "seed_free"], required=True)
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=24); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--P", type=int, default=64); ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--starts", type=int, default=60); ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results/e1b")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.out_dir, exist_ok=True)
    R = build_release(a.k, a.N, a.r, a.m, a.P, a.T, a.lr, a.seed, dev)
    H, A0, W0, y, A_T, B_T = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"]
    d, N = H.shape
    C, Np, _ = certificate(A_T, B_T, tol=1e-12)
    rank_C = int(torch.linalg.matrix_rank(C, rtol=1e-10)); q = a.r - rank_C
    rank_H = int(torch.linalg.matrix_rank(H, rtol=1e-10))
    log(f"# E1B-tiny arm={a.arm}  d={d} N={N} r={a.r} m={a.m} T={a.T} lr={a.lr} seed={a.seed}  fp64")
    log(f"# q = r - rank C = {a.r} - {rank_C} = {q}   N = {N}   rank H = {rank_H}")
    metric = "per-image" if (rank_H == N and q == N) else "subspace (principal angles)"
    log(f"# A1 metric in force: {metric}")
    assert rank_H == N, f"A1: rank H = {rank_H} != N = {N}; per-image metrics do not exist on this cell"

    nB = torch.linalg.norm(B_T); nAH = torch.linalg.norm(A_T @ H)
    pre = []
    for alpha in (0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 2.0):
        A_a, B_a = replay(alpha * H, A0 / alpha, W0, y, a.m, a.T, a.lr)
        pre.append((alpha, float(torch.linalg.norm(B_a - B_T) / nB),
                    float(torch.linalg.norm(A_a @ (alpha * H) - A_T @ H) / nAH)))
    off = [p[1] for p in pre if abs(p[0] - 1.0) > 1e-12]
    flat = max(off) < 1e-8
    log("# A2 scale pre-check (no solve): replay residual along (alpha h*, A0/alpha)")
    for al, rb, ra in pre: log(f"#    alpha={al:<5} ||B_sim-B_T||/||B_T|| = {rb:.3e}   adapter-path rel. change {ra:.1e}")
    log(f"# A2 verdict: {'FLAT -> derivation WRONG, scale not identifiable here' if flat else 'NOT flat -> the base path breaks the symmetry; scale is identifiable'}")

    def resid(Hc, A0c):
        A_s, B_s = replay(Hc, A0c, W0, y, a.m, a.T, a.lr)
        return torch.cat([((B_s - B_T) / nB).reshape(-1), ((A_s @ Hc - A_T @ H) / nAH).reshape(-1)])
    with torch.no_grad(): r_truth = float(torch.linalg.norm(resid(H, A0)))
    log(f"# residual at the truth: {r_truth:.3e}")

    gs = torch.Generator().manual_seed(a.seed + 31)
    scale = float(H.norm(dim=0).median())
    G0 = torch.randn(a.starts, d, N, generator=gs).to(dev); G0 = G0 / G0.norm(dim=1, keepdim=True) * scale
    _, _, Vh = torch.linalg.svd(C, full_matrices=True)
    ker = Vh[rank_C:].T.contiguous()
    K0 = torch.stack([ker @ (ker.T @ G0[i]) for i in range(a.starts)])
    K0 = K0 / K0.norm(dim=1, keepdim=True).clamp(min=1e-30) * scale

    rows = []; t_all = time.time(); mem0 = 0
    for fam, S0 in (("gaussian", G0), ("kerC", K0)):
        for s in range(a.starts):
            Hc = S0[s].clone().requires_grad_(True); params = [Hc]
            A0c = A_T.clone().requires_grad_(True) if a.arm == "seed_free" else A0
            if a.arm == "seed_free": params.append(A0c)
            opt = torch.optim.Adam(params, 5e-2); t0 = time.time()
            for it in range(a.iters):
                f = resid(Hc, A0c); loss = f @ f
                opt.zero_grad(); loss.backward(); opt.step()
            if dev.type == "cuda": mem0 = max(mem0, torch.cuda.max_memory_allocated() / 2**30)
            with torch.no_grad():
                f = resid(Hc, A0c); obj = float(torch.linalg.norm(f))
                Dm = torch.cdist(Hc.T, H.T) / H.norm(dim=0)[None, :]
                assign = Dm.argmin(1)
                per_err = [float(Dm[i, assign[i]]) for i in range(N)]
                cos = [float(torch.nn.functional.cosine_similarity(Hc[:, i], H[:, assign[i]], dim=0)) for i in range(N)]
                sr = [float(Hc[:, i].norm() / H[:, assign[i]].norm()) for i in range(N)]
                covered = len(set(int(x) for x in assign))
                cert = float((torch.linalg.norm(C @ Hc, dim=0) / torch.linalg.norm(A_T @ Hc, dim=0)).max())
            rows.append(dict(family=fam, start=s, objective=obj, per_image_err=per_err, cosine=cos, scale_ratio=sr,
                             truths_covered=covered, landed=bool(max(per_err) < 1e-2 and covered == N),
                             cert_residual=cert, seconds=time.time() - t0))
            if (s + 1) % 20 == 0:
                sub = [r for r in rows if r["family"] == fam]
                log(f"   {fam} {s+1}/{a.starts}: median objective {np.median([r['objective'] for r in sub]):.2e}, landings {sum(r['landed'] for r in sub)}")
    land = [r for r in rows if r["landed"]]
    row = dict(part="E1B-tiny", arm=a.arm, release="affine_two_routes (331384 generator)", precision="fp64",
               d=d, N=N, q=q, rank_H=rank_H, r=a.r, m=a.m, T=a.T, lr=a.lr, seed=a.seed, metric_in_force=metric,
               residual_at_truth=r_truth, scale_precheck=[dict(alpha=al, residual=rb, adapter_path_rel=ra) for al, rb, ra in pre],
               scale_symmetry_broken=(not flat), starts_per_family=a.starts,
               landings={f: sum(1 for r in rows if r["family"] == f and r["landed"]) for f in ("gaussian", "kerC")},
               objective_median={f: float(np.median([r["objective"] for r in rows if r["family"] == f])) for f in ("gaussian", "kerC")},
               objective_min=float(min(r["objective"] for r in rows)),
               best_max_per_image_err=float(min(max(r["per_image_err"]) for r in rows)),
               scale_ratio_of_landings=[float(np.median(r["scale_ratio"])) for r in land] or None,
               cosine_of_landings=[float(np.median(r["cosine"])) for r in land] or None,
               truths_covered_max=max(r["truths_covered"] for r in rows),
               manifold_test="VACUOUS ON THIS RELEASE: phi = identity, so range(Phi0) = R^d and min_x ||Phi0(x)-h|| = 0 for "
                             "every candidate by construction. Not a measurement; the real-backbone cell is a separate addition.",
               chart_note=f"a PCA chart of a d={d} space is the whole space at k >= {d}; k=128 is not measurable here",
               gb_peak=mem0, seconds=time.time() - t_all, runs=rows, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
    with open(os.path.join(a.out_dir, "rows.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")
    torch.save(dict(H=H.cpu(), A_T=A_T.cpu(), B_T=B_T.cpu(), C=C.cpu(), A0=A0.cpu(), row=row),
               os.path.join(a.out_dir, f"e1b_tiny_{a.arm}.pth"))
    log(f"\n=== E1B-tiny {a.arm}: landings {row['landings']}, best max-per-image error {row['best_max_per_image_err']:.3e}, "
        f"objective min {row['objective_min']:.2e} vs {r_truth:.2e} at the truth, truths covered max {row['truths_covered_max']}/{N}")
    print(json.dumps({k: v for k, v in row.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
