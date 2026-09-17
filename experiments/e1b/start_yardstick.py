#!/usr/bin/env python3
"""How far from the truth do the random starts actually sit? A basin radius means nothing without this.

The from-near-truth sweep brackets the basin edge along isotropic directions from the truth. That number is only
interpretable against the distance at which the random starts the real arms use actually sit -- if they sit inside
the basin and still fail, the basin is not a ball and distance alone does not characterise it.

Reports the distance for the EXACT starts the arms use (same generator, same normalisation), and for the two
reference distributions, so the comparison is not resting on a theoretical sqrt(2).

  python -m experiments.e1b.start_yardstick
"""
import argparse, json, os, socket, sys
import torch
from experiments.e1b.e1b_tiny import build_release
from experiments.exact_inversion.certificate import certificate
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=60); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/e1b/start_yardstick.jsonl")
    a = ap.parse_args()
    dev = torch.device("cpu"); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    R = build_release(12, 8, 24, 20, 64, 400, 0.05, a.seed, dev)
    H, A_T, B_T = R["H"], R["A_T"], R["B_T"]
    d, N = H.shape
    nH = torch.linalg.norm(H)
    C, _, _ = certificate(A_T, B_T, tol=1e-12)
    rank_C = int(torch.linalg.matrix_rank(C, rtol=1e-10))

    gs = torch.Generator().manual_seed(a.seed + 31)          # EXACTLY the arms' generator and order
    scale = float(H.norm(dim=0).median())
    G0 = torch.randn(a.starts, d, N, generator=gs)
    G0 = G0 / G0.norm(dim=1, keepdim=True) * scale
    _, _, Vh = torch.linalg.svd(C, full_matrices=True)
    ker = Vh[rank_C:].T.contiguous()
    K0 = torch.stack([ker @ (ker.T @ G0[i]) for i in range(a.starts)])
    K0 = K0 / K0.norm(dim=1, keepdim=True).clamp(min=1e-30) * scale

    rows = {}
    print(f"# ||H|| = {float(nH):.4f}   median column norm (the starts' per-column scale) = {scale:.4f}")
    print(f"# column norms of H: min {float(H.norm(dim=0).min()):.3f}, max {float(H.norm(dim=0).max()):.3f} "
          f"-- the starts set EVERY column to the median, so they differ from H in structure as well as position\n")
    for name, S in (("gaussian (the arms' starts)", G0), ("kerC (the arms' starts)", K0)):
        dist = torch.stack([torch.linalg.norm(S[i] - H) / nH for i in range(a.starts)])
        print(f"{name:30s} relative distance to the truth: median {float(dist.median()):.4f}   "
              f"range {float(dist.min()):.4f}-{float(dist.max()):.4f}")
        rows[name] = dict(median=float(dist.median()), min=float(dist.min()), max=float(dist.max()))
    # an isotropic perturbation of the same relative size, for reference -- this is what the eta sweep uses
    g2 = torch.Generator().manual_seed(a.seed + 555)
    for eta in (1.0, 1.25, 1.4142, 1.5):
        P = torch.randn(a.starts, d, N, generator=g2)
        P = P / P.reshape(a.starts, -1).norm(dim=1)[:, None, None]
        S = H[None] + eta * nH * P
        dist = torch.stack([torch.linalg.norm(S[i] - H) / nH for i in range(a.starts)])
        print(f"isotropic eta={eta:<7.4g}          relative distance: {float(dist.median()):.4f} (by construction)")
        rows[f"isotropic_eta_{eta}"] = dict(median=float(dist.median()))
    # DISTANCE IS NOT THE ONLY DIFFERENCE. A point at H + eta*||H||*u sits at the same DISTANCE as a random start
    # of similar norm, but it is not the same kind of point: it keeps a large component ALONG the truth, and it has
    # a larger norm. Both are measured here, because a from-near-truth test that quietly retains alignment would be
    # systematically easier than a random start at the same distance -- and that would invalidate the comparison.
    print("\n# ALIGNMENT, which distance alone hides:")
    print(f"   {'start family':34s} {'rel. distance':>13} {'cosine to truth':>16} {'norm / ||H||':>13}")
    for name, S in (("gaussian (the arms' starts)", G0), ("kerC (the arms' starts)", K0)):
        d_ = torch.stack([torch.linalg.norm(S[i] - H) / nH for i in range(a.starts)])
        c_ = torch.stack([torch.nn.functional.cosine_similarity(S[i].reshape(-1), H.reshape(-1), dim=0)
                          for i in range(a.starts)])
        n_ = torch.stack([torch.linalg.norm(S[i]) / nH for i in range(a.starts)])
        print(f"   {name:34s} {float(d_.median()):>13.4f} {float(c_.median()):>16.4f} {float(n_.median()):>13.4f}")
        rows[name].update(cosine_median=float(c_.median()), norm_ratio_median=float(n_.median()))
    for eta in (1.0, 1.25, 1.343, 1.4142, 1.5):
        P = torch.randn(a.starts, d, N, generator=torch.Generator().manual_seed(a.seed + 777))
        P = P / P.reshape(a.starts, -1).norm(dim=1)[:, None, None]
        S = H[None] + eta * nH * P
        d_ = torch.stack([torch.linalg.norm(S[i] - H) / nH for i in range(a.starts)])
        c_ = torch.stack([torch.nn.functional.cosine_similarity(S[i].reshape(-1), H.reshape(-1), dim=0)
                          for i in range(a.starts)])
        n_ = torch.stack([torch.linalg.norm(S[i]) / nH for i in range(a.starts)])
        print(f"   {'isotropic eta=' + format(eta, '.4g'):34s} {float(d_.median()):>13.4f} "
              f"{float(c_.median()):>16.4f} {float(n_.median()):>13.4f}")
        rows[f"isotropic_eta_{eta}"] = dict(median=float(d_.median()), cosine_median=float(c_.median()),
                                            norm_ratio_median=float(n_.median()))
    print("\n# READ THIS BEFORE COMPARING THE TWO. A from-near-truth start at the SAME distance as a random start is")
    print("# NOT an equivalent point: it retains a large cosine with the truth and a larger norm. So convergence at")
    print("# eta = 1.343 does NOT imply a random start at 1.343 should converge, and the from-near-truth sweep")
    print("# measures the basin ALONG RAYS FROM THE TRUTH rather than a radius in any isotropic sense.")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(part="start_yardstick", d=d, N=N, seed=a.seed, starts=a.starts,
                                 norm_H=float(nH), median_col_norm=scale,
                                 col_norm_min=float(H.norm(dim=0).min()), col_norm_max=float(H.norm(dim=0).max()),
                                 distances=rows, git=git_hash(), host=socket.gethostname(),
                                 cmd=" ".join(sys.argv))) + "\n")


if __name__ == "__main__":
    main()
