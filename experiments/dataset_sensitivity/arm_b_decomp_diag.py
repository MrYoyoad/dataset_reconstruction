#!/usr/bin/env python
"""
d2(N) DECOMPOSITION diagnostic (yoado-34 check 2, the load-bearing one).
Per N, CROSS-FIT split A/B of the K seeds, one swap target:
  paired diff v_j = dW(D, seed_j) - dW(D_swap, seed_j)   (init cancels per pair)
  signal subspace U = top-p right singular vectors of {v_j : j in A}   (split A)
  denominator lam_i = variance of split-A RESEEDS along u_i             (noise in signal dirs)
  numerator num_i   = (mean{v_j : j in B} . u_i)^2                      (split B, disjoint)
  d2 = sum_i num_i / lam_i
Shows whether d2(N) grows because the NUMERATOR grows (literal anti-dilution) or the
DENOMINATOR (lam) shrinks faster than the signal (detectability via whitening).
"""
import os, json, math, argparse
import torch
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import build_set, train_adapter, draw_B0
from experiments.data_utils import get_control_images_in_distribution
torch.set_default_dtype(torch.float64)


def decomp_for_N(N, K, lr, T, rank, device, p=3):
    n_per_class = N // 2
    act = make_activation("gelu")
    x_ft, y_ft, digits = build_set(n_per_class, seed=42, device=device)
    _, frozen, b0, B0_all, ds_mean = _honest_target(x_ft, y_ft, T, rank, "gelu", lr, device, "mnist", num_classes=2)
    x0 = x_ft - ds_mean
    out_f = frozen[0].shape[0]
    controls, _, _ = get_control_images_in_distribution(digits, seed=123, dataset="mnist")
    controls = controls.to(torch.float64).to(device)
    x_sw = x_ft.clone(); x_sw[0] = controls[0]; x0_sw = x_sw - ds_mean   # swap target i=0
    reseeds, vs = [], []
    for j in range(K):
        _, _, _, r = train_adapter(frozen, b0, draw_B0(1000 + j, out_f, rank, device), x0, y_ft, lr, T, act, rank)
        _, _, _, s = train_adapter(frozen, b0, draw_B0(1000 + j, out_f, rank, device), x0_sw, y_ft, lr, T, act, rank)
        if torch.isfinite(r).all() and torch.isfinite(s).all():
            reseeds.append(r.reshape(-1).cpu()); vs.append((r - s).reshape(-1).cpu())
    R = torch.stack(reseeds); V = torch.stack(vs)          # [Kf, D]
    Kf = V.shape[0]; half = Kf // 2
    iA = list(range(half)); iB = list(range(half, Kf))
    _, _, VhA = torch.linalg.svd(V[iA], full_matrices=False)   # signal subspace from split A
    U = VhA[:p]                                                # [p, D]
    lam = (R[iA] @ U.t()).var(0, unbiased=True)                # noise variance in signal dirs (split A)
    dmuB = V[iB].mean(0)                                       # split-B signal (disjoint)
    num = (dmuB @ U.t()) ** 2                                  # [p]
    d2i = num / (lam + 1e-30)
    return dict(N=N, K=Kf, dmuB_norm=float(dmuB.norm()),
                numerator=[float(x) for x in num], lam=[float(x) for x in lam],
                d2_i=[float(x) for x in d2i], d2=float(d2i.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_list", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage0", action="store_true")
    a = ap.parse_args()
    if a.stage0:
        print(json.dumps(decomp_for_N(8, K=12, lr=a.lr, T=a.T, rank=a.rank, device=a.device), indent=2))
        print("STAGE-0 OK"); return
    print("=== d2(N) DECOMPOSITION (cross-fit): does NUMERATOR grow or DENOMINATOR shrink? ===", flush=True)
    rows = []
    for N in a.N_list:
        r = decomp_for_N(N, a.K, a.lr, a.T, a.rank, a.device)
        rows.append(r)
        print("N=%3d: d2=%.3f | numer=[%s] | lam=[%s] | ||dmuB||=%.3e"
              % (N, r["d2"], ", ".join("%.3e" % x for x in r["numerator"]),
                 ", ".join("%.3e" % x for x in r["lam"]), r["dmuB_norm"]), flush=True)
    os.makedirs("/home/projects/galvardi/yoado/results/arm_b_dilution", exist_ok=True)
    json.dump(rows, open("/home/projects/galvardi/yoado/results/arm_b_dilution/decomp_diag.json", "w"), indent=2)


if __name__ == "__main__":
    main()
