"""Robustness fixes (auditor yoado-d4): (a) gentle-wd recheck, (b) CORRECT adapter-derived N>r cliff, (c)
which-pairs at N>r. Completes the scope line before the milestone writeup.

(a) WD: theory says A_{t+1}=(1−ηλ)A_t−η Bᵀ∂L/∂W stays a linear combo of gradient rows → row(A_T)⊆span{xᵢ} for
    ANY λ (SGD/mom). The wd=0.05 "0.74 break" was the row space being numerically undefined (A_T at rounding
    scale). Recheck wd∈{0,1e-4,1e-3,1e-2}, report ‖A_T‖_F, top singular values, residual on the top-N right
    singular vectors. Prediction: residual ~1e-14 with healthy ‖A_T‖ at gentle wd.
(b) CLIFF: train the RANK-8 adapter at N∈{8,9,12,16}, take its ACTUAL row space (≤8-dim), LP from there.
    Expected: exact through N=8, then a drop set by how much of span{xᵢ} the r-dim projection keeps (q_eff/r_J).
(c) WHICH-PAIRS at N=12: report failing images' max support-Jaccard with another (nested-support prediction).
SCOPE: first-layer LoRA (input=pixels), A₀=0, SGD-family (Adam/AdamW break it), N≤r, no gallery, reads only A.
"""
import argparse, os, numpy as np, torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import draw_B0, build_set, forward_logits
from experiments.dataset_sensitivity.lp_unmix import ssim, recover
from experiments.dataset_sensitivity.robustness_checks import load01

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/robustness_fixes"


def train_sgd(frozen, b0, B0, x0, y, lr, T, act, rank, wd=0.0, momentum=0.0):
    in_f, out_f = frozen[0].shape[1], frozen[0].shape[0]
    A0 = torch.zeros(rank, in_f, dtype=torch.float64, device=x0.device, requires_grad=True)
    B0_ = B0.clone().detach().requires_grad_(True)
    opt = torch.optim.SGD([A0, B0_], lr=lr, momentum=momentum, weight_decay=wd)
    A, B = {0: A0}, {0: B0_}
    for _ in range(T):
        out = forward_logits(x0, frozen, b0, A, B, act).view(-1)
        F.binary_cross_entropy_with_logits(out, y).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        bce = F.binary_cross_entropy_with_logits(forward_logits(x0, frozen, b0, A, B, act).view(-1), y, reduction="none").max().item()
    return (B[0] @ A[0]).detach(), A0.detach(), bce


def rowV(dW, k):
    sv = torch.linalg.svd(dW.detach().to("cpu", torch.float64), full_matrices=False)
    keep = min(k, int((sv.S > 1e-6 * sv.S[0]).sum()))
    return sv.Vh[:keep].transpose(-1, -2).contiguous(), keep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    dat, idx, m = load01(); mnp = m.numpy(); act = make_activation("gelu")
    xr, yr, _ = build_set(2, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, _dsm = _honest_target(xr, yr, 200, 8, "gelu", 0.5, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]; B0_atk = draw_B0(900, out_f, 8, dev)

    # ===== (a) GENTLE WD =====
    print("=== (a) WEIGHT DECAY (SGD, N=4): does gentle wd keep the span exact? ===")
    g = torch.Generator().manual_seed(42)
    sel = torch.cat([idx[d][torch.randperm(len(idx[d]), generator=g)[:2]] for d in (0, 1)])
    X = dat[sel]; x0 = (X - m).to(dev); y4 = torch.tensor([0., 0., 1., 1.], device=dev)
    wd_res = {}
    for wd in [0.0, 1e-4, 1e-3, 1e-2, 5e-2]:
        dW, A_T, bce = train_sgd(frozen, b0, B0_atk, x0, y4, 0.5, 300, act, 8, wd=wd)
        V, keep = rowV(dW, 4); Vt = V.to(dev)
        x0v = (X.to(dev) - m.to(dev)); res = (x0v - x0v @ Vt @ Vt.T).norm(dim=1) / (x0v.norm(dim=1) + 1e-12)
        aF = A_T.norm().item(); sv = torch.linalg.svdvals(A_T.to("cpu")).tolist()[:3]
        wd_res[wd] = dict(resid=float(res.max()), Afro=aF, bce=bce)
        print(f"  wd={wd:.0e}: member-resid(max)={res.max():.2e}  ‖A_T‖_F={aF:.3e}  top-svals={[f'{s:.2e}' for s in sv]}  "
              f"bce={bce:.3f}  → {'EXACT' if res.max()<1e-6 else 'undefined/broken'}")

    # ===== (b) CORRECT N>r CLIFF (adapter-derived row space) =====
    print("\n=== (b) N>r CLIFF (rank-8 ADAPTER row space, LP) ===")
    cliff = {}
    for N in [8, 9, 12, 16]:
        gg = torch.Generator().manual_seed(100 + N); ss, keeps = [], []
        for t in range(6):
            per = {0: N - N // 2, 1: N // 2}
            s = torch.cat([idx[d][torch.randperm(len(idx[d]), generator=gg)[:per[d]]] for d in (0, 1)])
            Xn = dat[s]; y = torch.tensor([0.]*per[0] + [1.]*per[1], device=dev)
            dW, _, _ = train_sgd(frozen, b0, B0_atk, (Xn - m).to(dev), y, 0.5, 300, act, 8)
            V, keep = rowV(dW, 8); keeps.append(keep)
            R = recover(V.numpy(), mnp, N, seed=t)
            if len(R) < N: R += [mnp] * (N - len(R))
            C = np.array([[ssim(r, Xn[j].numpy()) for j in range(N)] for r in R]); ri, ci = linear_sum_assignment(-C)
            ss += [ssim(R[a], Xn[b].numpy()) for a, b in zip(ri, ci)]
        cliff[N] = (float(np.mean(ss)), float(np.mean(keeps)))
        print(f"  N={N:2d} ({'≤r' if N <= 8 else '>r'}): LP-SSIM={np.mean(ss):.3f}  adapter-row-rank≈{np.mean(keeps):.1f}")

    # ===== (c) WHICH-PAIRS at N=12 =====
    print("\n=== (c) WHICH-PAIRS at N=12 (adapter-derived): failing vs passing support overlap ===")
    gg = torch.Generator().manual_seed(1212); lows, highs = [], []
    for t in range(6):
        s = torch.cat([idx[d][torch.randperm(len(idx[d]), generator=gg)[:6]] for d in (0, 1)])
        Xn = dat[s]; y = torch.tensor([0.]*6 + [1.]*6, device=dev)
        dW, _, _ = train_sgd(frozen, b0, B0_atk, (Xn - m).to(dev), y, 0.5, 300, act, 8)
        V, keep = rowV(dW, 8); R = recover(V.numpy(), mnp, 12, seed=t)
        if len(R) < 12: R += [mnp] * (12 - len(R))
        C = np.array([[ssim(r, Xn[j].numpy()) for j in range(12)] for r in R]); ri, ci = linear_sum_assignment(-C)
        supp = [(Xn[j].numpy() > 0.3) for j in range(12)]
        for a, b in zip(ri, ci):
            sv_ss = ssim(R[a], Xn[b].numpy())
            ovl = max([(supp[b] & supp[k]).sum() / ((supp[b] | supp[k]).sum() + 1e-9) for k in range(12) if k != b])
            (lows if sv_ss < 0.85 else highs).append(ovl)
    print(f"  FAILING (SSIM<0.85): n={len(lows)} mean max-support-Jaccard={np.mean(lows) if lows else float('nan'):.3f}")
    print(f"  PASSING (SSIM≥0.85): n={len(highs)} mean max-support-Jaccard={np.mean(highs) if highs else float('nan'):.3f}")
    if lows and highs:
        print(f"  → failures {'ARE higher-overlap (nested-support mechanism)' if np.mean(lows) > np.mean(highs) else 'NOT clearly higher-overlap'}")

    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(wd=wd_res, cliff=cliff, fail_ovl=lows, pass_ovl=highs), os.path.join(RESULTS, "fixes.pth"))
        print(f"[saved] {RESULTS}/fixes.pth")


if __name__ == "__main__":
    main()
