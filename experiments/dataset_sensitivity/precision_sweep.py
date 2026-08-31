"""Precision robustness (auditor yoado-d4 #3): noise-free is a MODELLING assumption — real released adapters are
float16/bfloat16 and often quantized/merged. Does the exact-vertex LP recovery survive? Quantize the released
factors (A,B) to float32 / bfloat16 / int8 at N=4, reform ΔW, take the row space, rerun the LP. Report SSIM vs
precision. Free on CPU after one tiny GPU train. SCOPE: first-layer, A₀=0, SGD, N≤r, no gallery, this-attacker.
"""
import argparse, os, numpy as np, torch
from scipy.optimize import linear_sum_assignment
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import draw_B0, build_set
from experiments.dataset_sensitivity.robustness_fixes import train_sgd, rowV
from experiments.dataset_sensitivity.lp_unmix import ssim, recover
from experiments.dataset_sensitivity.robustness_checks import load01

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/precision_sweep"


def quant(M, mode):
    if mode == "float64": return M
    if mode == "float32": return M.to(torch.float32).to(torch.float64)
    if mode == "bfloat16": return M.to(torch.bfloat16).to(torch.float64)
    if mode == "int8":
        s = M.abs().max() / 127.0
        return torch.round(M / (s + 1e-12)) * s
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    dat, idx, m = load01(); mnp = m.numpy(); act = make_activation("gelu")
    xr, yr, _ = build_set(2, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, _dsm = _honest_target(xr, yr, 200, 8, "gelu", 0.5, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]; B0_atk = draw_B0(900, out_f, 8, dev)
    print("[precision] N=4 | quantize released (A,B) → LP | float16/bf16/int8 realism")

    modes = ["float64", "float32", "bfloat16", "int8"]; res = {md: [] for md in modes}
    for t in range(8):
        sel = torch.cat([idx[d][torch.randperm(len(idx[d]), generator=torch.Generator().manual_seed(50 + t + 7 * d))[:2]] for d in (0, 1)])
        X = dat[sel]; y = torch.tensor([0., 0., 1., 1.], device=dev)
        # recover A,B (train_sgd returns dW,A; re-derive B via a second train to get both — simpler: patch train to expose B)
        dW, A_T, _ = train_sgd(frozen, b0, B0_atk, (X - m).to(dev), y, 0.5, 300, act, 8)
        # dW = B@A ; to quantize both factors we need B: B = dW @ pinv(A)
        A_np = A_T.to("cpu"); B_T = (dW.to("cpu") @ torch.linalg.pinv(A_np))
        for md in modes:
            dWq = quant(B_T, md) @ quant(A_np, md)
            V, keep = rowV(dWq, 8)
            R = recover(V.numpy(), mnp, 4, seed=t)
            if len(R) < 4: R += [mnp] * (4 - len(R))
            C = np.array([[ssim(r, X[j].numpy()) for j in range(4)] for r in R]); ri, ci = linear_sum_assignment(-C)
            res[md].append(float(np.mean([ssim(R[a], X[b].numpy()) for a, b in zip(ri, ci)])))
    print()
    for md in modes:
        print(f"  {md:9s}: LP-SSIM={np.mean(res[md]):.3f} ± {np.std(res[md]):.3f}")
    print(f"\n  [SCOPE: first-layer, A₀=0, SGD, N=4≤r, no gallery, this-attacker; quantize the RELEASED (A,B)]")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save({md: float(np.mean(res[md])) for md in modes}, os.path.join(RESULTS, "precision.pth"))
        print(f"[saved] {RESULTS}/precision.pth")


if __name__ == "__main__":
    main()
