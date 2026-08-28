#!/usr/bin/env python
"""
Null-only d2(N) diagnostic - yoado-34's decisive test for the surprising d2(N) growth.
Replaces the SWAP with a RESEED (no composition change):
    reseed_list[j] = dW(D, seed_A_j)
    v_list[j]      = dW(D, seed_B_j) - dW(D, seed_A_j)   (pure init-noise diff, NO swap)
Same estimator/structure as the real measurement, NO real signal => sensitivity MUST be flat ~0 across N.
Growing with N => the observed swap d2(N) growth is ESTIMATOR DRIFT. Controls training-depth: same D/T per N.
"""
import os, json, math, argparse
import torch
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import build_set, train_adapter, draw_B0
from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity
torch.set_default_dtype(torch.float64)


def null_for_N(N, K, lr, T, rank, device, n_folds=5, n_perm=500):
    n_per_class = N // 2
    act = make_activation("gelu")
    x_ft, y_ft, digits = build_set(n_per_class, seed=42, device=device)
    _, frozen, b0, B0_all, ds_mean = _honest_target(x_ft, y_ft, T, rank, "gelu", lr, device, "mnist", num_classes=2)
    x0 = x_ft - ds_mean
    out_f = frozen[0].shape[0]
    reseed_list, v_list, mbces = [], [], []
    for j in range(K):
        _, _, m_a, dW_A = train_adapter(frozen, b0, draw_B0(1000 + j, out_f, rank, device), x0, y_ft, lr, T, act, rank)
        _, _, m_b, dW_B = train_adapter(frozen, b0, draw_B0(5000 + j, out_f, rank, device), x0, y_ft, lr, T, act, rank)
        if torch.isfinite(dW_A).all() and torch.isfinite(dW_B).all():
            reseed_list.append(dW_A)
            v_list.append(dW_B - dW_A)
            mbces.append(max(m_a, m_b))
    ws = whitened_sensitivity([v.cpu() for v in v_list], [r.cpu() for r in reseed_list],
                              n_folds=n_folds, p_max=3, n_perm=n_perm, seed=0)
    return dict(N=N, K=len(v_list), max_bce=(max(mbces) if mbces else float("nan")),
                null_sensitivity=ws["sensitivity"], null_d2_obs=ws["d2_obs"],
                null_d2_null_mean=ws["d2_null_mean"], null_pvalue=ws["pvalue"],
                null_qeff=ws["qeff_count"], sigma_eff_rank=ws["sigma_eff_rank"])


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
        r = null_for_N(8, K=12, lr=a.lr, T=a.T, rank=a.rank, device=a.device)
        print(json.dumps(r, indent=2))
        assert math.isfinite(r["null_sensitivity"]), "null_sensitivity NaN"
        print("STAGE-0 OK")
        return
    print("=== NULL-ONLY d2(N) - MUST be flat ~0 across N (else swap d2(N) growth is DRIFT) ===", flush=True)
    rows = []
    for N in a.N_list:
        r = null_for_N(N, a.K, a.lr, a.T, a.rank, a.device)
        rows.append(r)
        print("N=%3d: null_sens=%.3f  d2_obs=%.3f  d2_null_mean=%.3f  p=%.3f  qeff=%.1f  effrankSig=%.1f  max_bce=%.1e"
              % (N, r["null_sensitivity"], r["null_d2_obs"], r["null_d2_null_mean"], r["null_pvalue"],
                 r["null_qeff"], r["sigma_eff_rank"], r["max_bce"]), flush=True)
    os.makedirs("/home/projects/galvardi/yoado/results/arm_b_dilution", exist_ok=True)
    with open("/home/projects/galvardi/yoado/results/arm_b_dilution/null_diag.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nVERDICT: flat ~0 -> swap d2(N) REAL; growing with N -> estimator DRIFT.", flush=True)


if __name__ == "__main__":
    main()
