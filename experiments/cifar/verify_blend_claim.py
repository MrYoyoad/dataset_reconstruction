#!/usr/bin/env python3
"""Verify, at the tensors, the three numbers in RESULT.md section 1: the attractor's residual, the ideal blend's
residual, and the blend fraction. Reports every quantity under BOTH normalisations, because the replica's search
used ||Cx||/(||C|| ||x||) while the chart study uses ||Cx||/||A_T x|| and the two are not interchangeable.

  python -m experiments.cifar.verify_blend_claim
"""
import glob, json, os
import torch
torch.set_default_dtype(torch.float64)


def report(tag, C, A_T, H, X_found, R_saved=None):
    C, A_T, H, X_found = C.double(), A_T.double(), H.double(), X_found.double()   # the replica saved some tensors in fp32
    nC = torch.linalg.norm(C)
    o_cnorm = lambda X: (torch.linalg.norm(C @ X, dim=0) / (nC * torch.linalg.norm(X, dim=0)))
    o_at = lambda X: (torch.linalg.norm(C @ X, dim=0) / torch.linalg.norm(A_T @ X, dim=0))
    N = H.shape[1]
    print(f"\n===== {tag}:  C {tuple(C.shape)}  H {tuple(H.shape)}  found {tuple(X_found.shape)}")
    print(f"  truths           : ||Cx||/(||C||||x||) {o_cnorm(H).max():.3e}   ||Cx||/||A_T x|| {o_at(H).max():.3e}")
    # the IDEAL blend: least-squares fit of the mean found image in span(H), then evaluated exactly
    xbar = X_found.mean(1, keepdim=True)
    coef = torch.linalg.lstsq(H, xbar).solution                       # (N,1)
    ideal = H @ coef
    r = xbar - ideal
    frac_energy = float(1 - (torch.linalg.norm(r) / torch.linalg.norm(xbar)) ** 2)
    frac_norm = float(1 - torch.linalg.norm(r) / torch.linalg.norm(xbar))
    print(f"  IDEAL blend H@c  : ||Cx||/(||C||||x||) {float(o_cnorm(ideal)):.3e}   ||Cx||/||A_T x|| {float(o_at(ideal)):.3e}")
    print(f"  ATTRACTOR (mean) : ||Cx||/(||C||||x||) {float(o_cnorm(xbar)):.3e}   ||Cx||/||A_T x|| {float(o_at(xbar)):.3e}")
    print(f"  coefficients {[round(float(v), 3) for v in coef[:, 0]]}  sum {float(coef.sum()):.3f}  negatives {int((coef < 0).sum())}")
    print(f"  blend fraction: energy 1-(||r||/||x||)^2 = {frac_energy:.4f} ;  norm 1-||r||/||x|| = {frac_norm:.4f}")
    if R_saved is not None:
        Rs = R_saved
        print(f"  saved per-start R: min {float(Rs.min()):.3e}  median {float(Rs.median()):.3e}  max {float(Rs.max()):.3e}")
        lo = Rs < 3 * float(Rs.min())
        print(f"  starts within 3x of the best: {int(lo.sum())} / {len(Rs)}  (their median R {float(Rs[lo].median()):.3e})")


def main():
    f = "experiments/cifar/k32_onchart/release_and_search.pt"
    if os.path.exists(f):
        d = torch.load(f, map_location="cpu", weights_only=False)
        report("REPLICA on-chart k=32 (jobs 257893)", d["C"], d["A_T"], d["H_train"].T if d["H_train"].shape[0] != d["C"].shape[1] else d["H_train"],
               d["X_found"].T, d.get("R"))
    for g in sorted(glob.glob("experiments/cifar/charts/L1_*/release_and_search.pt")):
        d = torch.load(g, map_location="cpu", weights_only=False)
        Xtr = d["X_train"]; Xtr = Xtr if Xtr.shape[0] == d["C"].shape[1] else Xtr.T
        Xf = d["X_found"]; Xf = Xf if Xf.shape[0] == d["C"].shape[1] else Xf.T
        report(f"CHART STUDY {g.split('/')[-2]} (result: landed {d['result']['landed']}/{d['result']['starts']}, "
               f"images {d['result']['images_found']}/{d['result']['N']})", d["C"], d["A_T"], Xtr, Xf, d.get("R"))


if __name__ == "__main__":
    main()
