#!/usr/bin/env python3
"""Near-duplicate degeneracy: which CHANNEL does it break?

Motivation (independent derivation check, 2026-09-03).  The certificate claim `C H = 0`, `rank C = r - N`
carries an unstated hypothesis: `rank P_T = N`, i.e. the N examples must produce N linearly independent
accumulated residual trajectories.  That fails for duplicated examples, already-fit examples, and
degenerate labels.  A reduced-algebra check predicted something sharper than "duplicates break it":

  * a well-separated pair is clean, and an EXACT duplicate is ALSO clean (the private span is genuinely
    (N-1)-dimensional and the certificate is correct on it);
  * the damage lives in the NEAR-duplicate band, where the span is truly N-dimensional but its Nth
    direction sits below the numerical rank tolerance, so the attacker's certificate silently MISSES a
    real private direction;
  * so `|CH|` is non-monotone in the separation eps (off, then on at the rank cutoff, peaking there,
    then decaying with eps, then off again at exact duplication) and is a FRAGILE detector, while
    `sigma_N(B_T)/sigma_1(B_T)` decays smoothly with eps and is the usable one.

The defense-relevant question this script answers: does near-duplication break the SIMULATION channel too,
or only the algebraic one?  `invert()` matches `B_T` and `A_T U` directly and never takes a numerical
rank, so the prediction is that it still recovers across the whole band.  If so, a defender who
perturbs-and-copies a record disables the closed-form attack while the data still leaks.

This module IMPORTS the main testbed rather than copying it, so the recipe cannot drift between the two.

  python -m experiments.exact_inversion.duplicate_degeneracy --eps 3e-1 1e-2 1e-4 1e-6 0 --device cuda
"""
import argparse, json, math, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import (
    World, train_release, certificate, invert, qr_canon, git_hash, RECOVER_TOL)

torch.set_default_dtype(torch.float64)


def run_eps(args, eps, log=print):
    """eps = latent-space separation between example 1 and example 0 (same label).  eps=None -> clean control."""
    dev = torch.device(args.device)
    world = World(args.k, args.P, args.n, args.seed, dev)
    g = torch.Generator().manual_seed(args.seed + 7)
    N, r, m, n, k = args.N, args.r, args.m, args.n, args.k

    W_true = torch.randn(k, N, generator=g).to(dev)
    y = (torch.arange(N) % m).to(dev)
    if eps is not None:
        d = torch.randn(k, generator=g).to(dev); d = d / torch.linalg.norm(d)
        W_true[:, 1] = W_true[:, 0] + eps * d          # near-duplicate of example 0 ...
        y[1] = y[0]                                     # ... with the SAME label
    X_img = world.psi(W_true); H = world.phi(X_img)
    W0 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(dev)
    A0 = (args.sigma0 * torch.randn(r, n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, m, args.T, args.lr, "sgd", args.wd)

    # ---- algebraic channel diagnostics ----
    sv_B = torch.linalg.svdvals(B_T)
    sig_ratio = float(sv_B[N - 1] / sv_B[0])            # sigma_N(B_T)/sigma_1(B_T): the graded quantity
    rankB = int((sv_B > args.tol * sv_B[0]).sum())
    C, rankB_c, rankC, _ = certificate(A_T, B_T, tol=args.tol)
    cert_err = float(torch.linalg.norm(C @ H) / (torch.linalg.norm(A_T, 2) * torch.linalg.norm(H)))
    cert_norm = float(torch.linalg.norm(C) / torch.linalg.norm(A_T))
    pair_sep = float(torch.linalg.norm(H[:, 1] - H[:, 0]) / torch.linalg.norm(H[:, 0]))

    # ---- simulation channel: does invert() still recover? ----
    W_init = W_true + args.init_noise * torch.randn(k, N, generator=g).to(dev)
    with torch.no_grad():
        U_init, _ = qr_canon(world.features_from_latents(W_init))
        Xinit = A_T @ U_init
    t0 = time.time()
    W_hat, aux, res, sec, n_rs, diag = invert(world, A_T, B_T, W0, y, args, W_init, Xinit, log)
    X_hat = world.psi(W_hat)
    err = (torch.linalg.norm(X_hat - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))

    # ---- which images fail, and HOW.  A failure confined to the near-duplicate pair {0,1}, with the two
    #      reconstructions being swaps or blends of each other, is an information limit specific to
    #      near-duplicates -- NOT certificate contamination and NOT a search failure.  Only a NONZERO
    #      residual indicates the latter.  (Three-way split suggested by the derivation check.)
    with torch.no_grad():
        x0, x1 = X_img[:, 0], X_img[:, 1]
        d01 = x0 - x1; nd = float(d01 @ d01)
        pair = {}
        for i in (0, 1):
            xh = X_hat[:, i]
            # least-squares alpha in  xh ~ alpha*x0 + (1-alpha)*x1 : 1 -> is image 0, 0 -> image 1, ~.5 -> blend
            alpha = float(((xh - x1) @ d01) / nd) if nd > 0 else float("nan")
            fit = xh - (alpha * x0 + (1 - alpha) * x1)
            pair[f"alpha_{i}"] = alpha
            pair[f"offline_resid_{i}"] = float(torch.linalg.norm(fit) / torch.linalg.norm(X_img[:, i]))
            pair[f"err_to_other_{i}"] = float(torch.linalg.norm(xh - X_img[:, 1 - i]) / torch.linalg.norm(X_img[:, 1 - i]))
        pair["err_pair_max"] = float(err[:2].max())
        pair["err_others_max"] = float(err[2:].max()) if N > 2 else float("nan")
        pair["failure_confined_to_pair"] = bool(float(err[:2].max()) >= RECOVER_TOL and
                                                (N <= 2 or float(err[2:].max()) < RECOVER_TOL))

    out = dict(eps=("clean" if eps is None else eps), pair_feature_sep=pair_sep,
               sigma_N_over_sigma_1=sig_ratio, rankB=rankB, rankC=rankC, rank_expected=r - N,
               cert_err=cert_err, cert_norm=cert_norm,
               final_err_max=float(err.max()), final_err_median=float(err.median()),
               frac_recovered=float((err < RECOVER_TOL).double().mean()), residual=res,
               per_image_err=[float(e) for e in err], **pair,
               recovered=bool(float(err.max()) < RECOVER_TOL),
               k=k, N=N, r=r, m=m, n=n, T=args.T, lr=args.lr, seed=args.seed, tol=args.tol,
               git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv), seconds=time.time() - t0, **diag)
    log(json.dumps(out))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(out) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, nargs="*", default=[3e-1, 1e-2, 1e-4, 1e-6, 0.0])
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=1); ap.add_argument("--restart-noise", type=float, default=0.1)
    ap.add_argument("--lm-iters", type=int, default=80); ap.add_argument("--lm-lambda", type=float, default=1e-2)
    ap.add_argument("--lm-scale", default="identity"); ap.add_argument("--stage-x", type=int, default=0)
    ap.add_argument("--jac", default="fwd"); ap.add_argument("--solver", default="lm")
    ap.add_argument("--outer", type=int, default=30); ap.add_argument("--lbfgs-iter", type=int, default=20)
    ap.add_argument("--release", default="sgd")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.sigma0 is None: args.sigma0 = 1.0 / math.sqrt(args.n)
    log = (lambda s: None) if args.quiet else (lambda s: print(s, flush=True))
    print(f"# near-duplicate degeneracy  git={git_hash()} host={socket.gethostname()}\n# cmd: {' '.join(sys.argv)}", flush=True)
    print(f"# clean control first, then eps sweep {args.eps}", flush=True)
    run_eps(args, None, log)                       # clean control: all examples distinct
    for e in args.eps:
        print(f"##### eps={e}", flush=True)
        run_eps(args, e, log)


if __name__ == "__main__":
    main()
