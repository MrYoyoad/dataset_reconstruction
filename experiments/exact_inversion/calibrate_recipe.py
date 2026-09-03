#!/usr/bin/env python3
"""Can the attacker MEASURE the recipe instead of assuming it?

Two ideas, both from the user (2026-09-03).

(1) The image error is NOT observable to an attacker — they do not have the private images. The RESIDUAL
    is: it is computed from the released factors and the candidate alone. So the residual is the attacker's
    Cauchy criterion — it certifies convergence without reference to the limit — and any recipe-selection
    rule must be built on it, never on reconstruction quality.

(2) Continuation calibration. The attacker holds the released adapter and can keep training it on data of
    their OWN choosing. Under a scalar-linear update one further step gives, exactly,
        Delta B = -eta * gB,     gB = D (A_T H')^T
    and the attacker knows A_T, B_T, their own H' and their own labels, hence knows gB. So eta falls out of
    a one-dimensional least squares. This converts "the recipe is known" from an assumption into a
    measurement, using no private data at all.
    It also DISCRIMINATES the optimizer family: under SGD the observed step is exactly parallel to gB
    (cosine 1); under Adam the coordinatewise normalisation destroys that, so the cosine drops.
    What it cannot recover is T, the number of steps already taken before the release.

Imports the main testbed so the recipe cannot drift.
  python -m experiments.exact_inversion.calibrate_recipe --release sgd --T 400 --lr 0.01
"""
import argparse, json, math, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import (
    World, train_release, softmax_cols, git_hash)

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
    ap.add_argument("--Nprobe", type=int, default=6, help="how many examples of the ATTACKER'S OWN data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(a.n)
    dev = torch.device(a.device)
    world = World(a.k, a.P, a.n, a.seed, dev)
    g = torch.Generator().manual_seed(a.seed + 7)
    N, r, m, n, k = a.N, a.r, a.m, a.n, a.k

    # ---- the private release (the attacker sees only A_T, B_T) ----
    W_true = torch.randn(k, N, generator=g).to(dev)
    H = world.phi(world.psi(W_true))
    y = (torch.arange(N) % m).to(dev)
    W0 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(dev)
    A0 = (a.sigma0 * torch.randn(r, n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, m, a.T, a.lr, a.release, a.wd)

    # ---- the attacker's OWN probe data: unrelated latents, labels of their choosing ----
    gp = torch.Generator().manual_seed(a.seed + 991)
    Wp = torch.randn(k, a.Nprobe, generator=gp).to(dev)
    Hp = world.phi(world.psi(Wp))
    yp = (torch.arange(a.Nprobe) % m).to(dev)
    Yp = torch.eye(m, device=dev)[yp].T

    # what the attacker can compute from the release + their own data, before taking any step
    with torch.no_grad():
        AH = A_T @ Hp
        D = (softmax_cols(W0 @ Hp + B_T @ AH) - Yp) / a.Nprobe
        gB = D @ AH.T                                      # the SGD B-gradient on the probe batch

    # ---- take ONE further step with the TRUE recipe (the attacker runs it; only eta is unknown) ----
    with torch.no_grad():
        if a.release == "sgd":
            B1 = (1 - a.lr * a.wd) * B_T - a.lr * gB
        else:
            b1, b2, eps = 0.9, 0.999, 1e-8
            mB = (1 - b1) * gB; vB = (1 - b2) * gB * gB
            B1 = (1 - a.lr * a.wd) * B_T - a.lr * (mB / (1 - b1)) / (torch.sqrt(vB / (1 - b2)) + eps)
        dB = B1 - B_T

    # ---- the attacker's estimate: one-dimensional least squares of dB on -gB ----
    with torch.no_grad():
        num = float((dB * (-gB)).sum()); den = float((gB * gB).sum())
        eta_hat = num / den
        cos = num / (float(torch.linalg.norm(dB)) * float(torch.linalg.norm(gB)) + 1e-300)
        resid_par = float(torch.linalg.norm(dB + eta_hat * gB) / torch.linalg.norm(dB))

    out = dict(true_release=a.release, eta_true=a.lr, eta_hat=eta_hat,
               eta_rel_err=abs(eta_hat - a.lr) / a.lr,
               cos_dB_minus_gB=cos, parallel_residual=resid_par,
               sgd_consistent=bool(resid_par < 1e-10),
               T=a.T, N=N, Nprobe=a.Nprobe, k=k, r=r, m=m, n=n, seed=a.seed,
               git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
    print(f"release={a.release}  true eta={a.lr}  ESTIMATED eta={eta_hat:.10f}  rel.err={out['eta_rel_err']:.2e}",
          flush=True)
    print(f"  cos(dB, -gB) = {cos:.12f}   ||dB + eta_hat gB||/||dB|| = {resid_par:.2e}   "
          f"-> SGD-consistent: {out['sgd_consistent']}", flush=True)
    print("  (the attacker used ONLY the released factors and their own probe data; no private data)", flush=True)
    if a.out:
        with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
