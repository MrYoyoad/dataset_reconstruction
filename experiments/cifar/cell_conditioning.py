#!/usr/bin/env python3
"""Why does ONE cell fail the linearised route when three matched cells recover? Measure, over all four.

The four matched cells (PCA k=16, T=1, N=8, r=64, head adapter, on-chart privates) split 3-1: CIFAR keyboard,
CIFAR keyboard+apple and MNIST letters a+t all recover 8 of 8 by the linearised route with variable projection;
MNIST letter a recovers 0 of 8. Two explanations were proposed and one is already refuted:

  REFUTED  batch class composition (one added class versus two): CIFAR keyboard is a single added class with all
           eight labels equal, and it recovers 8 of 8.
  LIVE     conditioning, predicted from the certificate residual at the truths, which is two orders worse in the
           failing cell (5.2e-13) than in the other three (3.5e-15, 4.7e-15, 6.7e-15).

The prediction under the live explanation: the design the eliminated coefficients invert, A_T H, is measurably worse
conditioned in the failing cell, and the same near-collinearity is what inflates the certificate residual. This
script rebuilds each release deterministically (same seeds, same recipe, so the releases are the sweep's own) and
reports the conditioning of everything the two routes actually invert, for ALL FOUR cells rather than the two that
fit a story.

  python -m experiments.cifar.cell_conditioning
"""
import json, math, socket, sys
import numpy as np
import torch

from experiments.cifar.ntk_vs_certificate import build_cifar, build_mnist

torch.set_default_dtype(torch.float64)

CELLS = [
    dict(tag="cifar keyboard",       dataset="cifar", newclass="cifar100:keyboard", newclass2=None,               letter="a", letter2=None, recovers=True),
    dict(tag="cifar keyboard+apple", dataset="cifar", newclass="cifar100:keyboard", newclass2="cifar100:apple",   letter="a", letter2=None, recovers=True),
    dict(tag="mnist letter a",       dataset="mnist", newclass="cifar100:keyboard", newclass2=None,               letter="a", letter2=None, recovers=False),
    dict(tag="mnist letters a+t",    dataset="mnist", newclass="cifar100:keyboard", newclass2=None,               letter="a", letter2="t",  recovers=True),
]


def cond(M):
    sv = torch.linalg.svdvals(M)
    return float(sv[0] / sv[-1].clamp(min=1e-300)), float(sv[-1]), float(sv[0])


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, r, k, T, lr, seed = 8, 64, 16, 1, 0.01, 1
    print(f"# cell_conditioning  N={N} r={r} k={k} T={T} lr={lr} seed={seed}  host={socket.gethostname()}", flush=True)
    rows = []
    for cell in CELLS:
        a = type("A", (), dict(newclass=cell["newclass"], newclass2=cell["newclass2"], letter=cell["letter"], letter2=cell["letter2"],
                               seed=seed, ckpt=None, model_mnist="models/exact_inversion/mnist_mlp_strong.pth",
                               data_root="data", mnist_root="dataset_reconstruction/data"))()
        phi, W0, Pub_l, Pri_l, cname, accs, shape = (build_cifar if cell["dataset"] == "cifar" else build_mnist)(a, dev)
        m, n = W0.shape
        g = torch.Generator().manual_seed(seed + 7); ncls = len(Pri_l); per = N // ncls
        X_raw = torch.cat([Pri_l[c][torch.randperm(Pri_l[c].shape[0], generator=g)[:per]].T for c in range(ncls)], 1).contiguous()
        y = torch.cat([torch.full((per,), m - ncls + c, device=dev) for c in range(ncls)])
        Pub = torch.cat(Pub_l, 0)
        mean = Pub.mean(0); _, _, Vh = torch.linalg.svd(Pub - mean, full_matrices=False); V = Vh[:k].T.contiguous()
        X_on = mean[:, None] + V @ (V.T @ (X_raw - mean[:, None]))
        H = phi(X_on)
        A0 = (1.0 / math.sqrt(n) * torch.randn(r, n, generator=torch.Generator().manual_seed(seed + 7), dtype=torch.float64)).to(dev)
        A, B = A0.clone(), torch.zeros(m, r, dtype=torch.float64, device=dev)
        Y = torch.eye(m, device=dev, dtype=torch.float64)[y].T
        for _ in range(T):
            z = W0 @ H + B @ (A @ H); Dm = (torch.softmax(z, 0) - Y) / N
            B, A = B - lr * (Dm @ (A @ H).T), A - lr * (B.T @ Dm @ H.T)
        A_T, B_T = A, B
        sB = torch.linalg.svdvals(B_T); Np = int((sB > 1e-12 * sB[0]).sum())
        _, _, VhB = torch.linalg.svd(B_T, full_matrices=False); Q = VhB[:Np].T
        C = A_T - Q @ (Q.T @ A_T)
        with torch.no_grad():
            cert = torch.linalg.norm(C @ H, dim=0) / torch.linalg.norm(A_T @ H, dim=0)
            F = A_T @ H                                              # THE DESIGN variable projection inverts
            cF, sF_min, sF_max = cond(F)
            cG, _, _ = cond(F.T @ F)                                 # the Gram itself
            cH, _, _ = cond(H)
            Hn = H / torch.linalg.norm(H, dim=0, keepdim=True); cos = (Hn.T @ Hn)
            off = cos[~torch.eye(N, dtype=torch.bool, device=dev)]
            Fn = F / torch.linalg.norm(F, dim=0, keepdim=True); cosF = (Fn.T @ Fn)
            offF = cosF[~torch.eye(N, dtype=torch.bool, device=dev)]
        row = dict(cell=cell["tag"], recovers_by_linearised_route=cell["recovers"], dataset=cell["dataset"], class_name=cname,
                   n_added_classes=ncls, labels=y.tolist(), m=m, n=n,
                   cert_residual_at_truths_max=float(cert.max()), cert_residual_at_truths_median=float(cert.median()),
                   cond_design_A_T_H=cF, sigma_min_design=sF_min, sigma_max_design=sF_max, cond_gram=cG,
                   cond_features_H=cH, feature_cosine_offdiag_max=float(off.max()), feature_cosine_offdiag_mean=float(off.mean()),
                   design_cosine_offdiag_max=float(offF.max()), design_cosine_offdiag_mean=float(offF.mean()),
                   B_T_norm=float(torch.linalg.norm(B_T)), B_T_sigma_ratio_N_over_1=float(sB[N - 1] / sB[0]),
                   B_T_gap_N_over_Nplus1=float(sB[N - 1] / sB[N]), rank_B_T=Np)
        rows.append(row); print(json.dumps(row), flush=True)
    print("\n| cell | recovers | cert residual at truths | cond(A_T H) | cond(Gram) | σ_min(design) | max off-diag cosine of the design | B_T σ_N/σ_1 |")
    print("|---|---|---|---|---|---|---|---|")
    for r_ in rows:
        print(f"| {r_['cell']} | {'yes' if r_['recovers_by_linearised_route'] else '**NO**'} | {r_['cert_residual_at_truths_max']:.1e} | "
              f"{r_['cond_design_A_T_H']:.2e} | {r_['cond_gram']:.2e} | {r_['sigma_min_design']:.2e} | "
              f"{r_['design_cosine_offdiag_max']:.4f} | {r_['B_T_sigma_ratio_N_over_1']:.1e} |")
    print("\n# The prediction was that the failing cell is the worst conditioned. Read the table, not this sentence.")


if __name__ == "__main__":
    main()
