"""Pre-writeup robustness checks for the row-span / LP-recovery milestone (auditor yoado-d4). All cheap.

#2 OPTIMIZER (decisive): row(A_T)⊆span{xᵢ} holds because each SGD/momentum A-update is a LINEAR map of the
   gradient rows. ADAM's m/√v is elementwise → NOT linear → predicted to break the exact span even for A₀=0.
   Adam is the fine-tuning default, so this is headline-level. Measure member residual onto row(ΔW) + LP recon
   SSIM at N=4 for {SGD, SGD+momentum, SGD+wd, Adam, AdamW}. wd should NOT break it (only shrinks A).
#3 N>r CLIFF: LP SSIM at N∈{8,9,12,16} — past r=8 the row space is an r-dim PROJECTION of the span, LP loses
   exactness. Show the cliff (ties to the rank/q_eff story).
#1 WHICH-PAIRS: at N=8 report per-image SSIM + digit + support-overlap (Jaccard of pixel supports) for the
   failing vs passing images — test the auditor's "nested-support pairs fail identifiably" prediction.
SCOPE: first-layer (input=pixels), A₀=0, N≤r, no gallery, this-attacker; reads only the released A factor.
"""
import argparse, os, numpy as np, torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import draw_B0, build_set, forward_logits
from experiments.dataset_sensitivity.lp_unmix import ssim, recover
from experiments.data_utils import _load_dataset, _get_binary_label

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/robustness_checks"
OPTS = {"SGD": ("sgd", 0.5, 300), "SGD+mom": ("sgdm", 0.5, 300), "SGD+wd": ("sgdwd", 0.5, 300),
        "Adam": ("adam", 0.02, 1500), "AdamW": ("adamw", 0.02, 1500)}


def train_opt(frozen, b0, B0, x0, y, act, rank, kind, lr, T):
    in_f, out_f = frozen[0].shape[1], frozen[0].shape[0]
    A0 = torch.zeros(rank, in_f, dtype=torch.float64, device=x0.device, requires_grad=True)   # A₀=0
    B0_ = B0.clone().detach().requires_grad_(True)
    if kind == "sgd":    opt = torch.optim.SGD([A0, B0_], lr=lr)
    elif kind == "sgdm": opt = torch.optim.SGD([A0, B0_], lr=lr, momentum=0.9)
    elif kind == "sgdwd":opt = torch.optim.SGD([A0, B0_], lr=lr, weight_decay=0.05)
    elif kind == "adam": opt = torch.optim.Adam([A0, B0_], lr=lr)
    else:                opt = torch.optim.AdamW([A0, B0_], lr=lr, weight_decay=0.05)
    A, B = {0: A0}, {0: B0_}
    for _ in range(T):
        out = forward_logits(x0, frozen, b0, A, B, act).view(-1)
        loss = F.binary_cross_entropy_with_logits(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        bce = F.binary_cross_entropy_with_logits(forward_logits(x0, frozen, b0, A, B, act).view(-1), y, reduction="none").max().item()
    return (B[0] @ A[0]).detach(), bce


def row_space(dW, tol=1e-6):
    sv = torch.linalg.svd(dW.detach().to("cpu", torch.float64), full_matrices=False)
    return sv.Vh[:(sv.S > tol * sv.S[0]).sum().item()].transpose(-1, -2).contiguous()


def load01():
    ds = _load_dataset("mnist", train=True)
    tgt = ds.targets if torch.is_tensor(ds.targets) else torch.tensor(ds.targets)
    dat = ds.data.reshape(len(ds.data), -1).double() / 255.0
    idx = {d: (tgt == d).nonzero(as_tuple=True)[0] for d in (0, 1)}
    m = torch.cat([dat[idx[d][:1500]] for d in (0, 1)]).mean(0)
    return dat, idx, m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    dat, idx, m = load01(); mnp = m.numpy(); act = make_activation("gelu")
    xr, yr, _ = build_set(2, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, _dsm = _honest_target(xr, yr, 200, 8, "gelu", 0.5, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]; B0_atk = draw_B0(900, out_f, 8, dev)
    y4 = torch.tensor([0., 0., 1., 1.], device=dev)

    # ===== #2 OPTIMIZER =====
    print("=== #2 OPTIMIZER (N=4): does the exact span survive? ===")
    g = torch.Generator().manual_seed(42); opt_res = {}
    sel = torch.cat([idx[d][torch.randperm(len(idx[d]), generator=g)[:2]] for d in (0, 1)])
    X = dat[sel]; x0 = (X - m).to(dev)
    for name, (kind, lr, T) in OPTS.items():
        dW, bce = train_opt(frozen, b0, B0_atk, x0, y4, act, 8, kind, lr, T)
        V = row_space(dW); Vt = V.to(dev)
        mres = ((X.to(dev) - m.to(dev)) - (X.to(dev) - m.to(dev)) @ Vt @ Vt.T).norm(dim=1) / ((X.to(dev) - m.to(dev)).norm(dim=1) + 1e-12)
        R = recover(V.numpy(), mnp, 4, seed=1)
        if len(R) < 4: R += [mnp] * (4 - len(R))
        C = np.array([[ssim(r, X[j].numpy()) for j in range(4)] for r in R]); ri, ci = linear_sum_assignment(-C)
        sval = float(np.mean([ssim(R[a], X[b].numpy()) for a, b in zip(ri, ci)]))
        opt_res[name] = dict(member_resid=float(mres.max()), lp_ssim=sval, bce=bce)
        print(f"  {name:8s}: member-resid(max)={mres.max():.2e}  LP-SSIM={sval:.3f}  (train max-bce={bce:.3f})  "
              f"→ {'span EXACT' if mres.max()<1e-6 else 'span BROKEN'}")

    # ===== #3 N>r CLIFF (planted spans, CPU) =====
    print("\n=== #3 N>r CLIFF (LP recon SSIM past r=8) ===")
    cliff = {}
    for N in [8, 9, 12, 16]:
        gg = torch.Generator().manual_seed(100 + N); ss = []
        for t in range(6):
            per = {0: N - N // 2, 1: N // 2}
            s = torch.cat([idx[d][torch.randperm(len(idx[d]), generator=gg)[:per[d]]] for d in (0, 1)])
            Xn = dat[s].numpy(); sv = np.linalg.svd(Xn - mnp, full_matrices=False)
            keep = int((sv[1] > 1e-6 * sv[1][0]).sum()); V = sv[2][:keep].T
            R = recover(V, mnp, N, seed=t)
            if len(R) < N: R += [mnp] * (N - len(R))
            C = np.array([[ssim(r, Xn[j]) for j in range(N)] for r in R]); ri, ci = linear_sum_assignment(-C)
            ss += [ssim(R[a], Xn[b]) for a, b in zip(ri, ci)]
        cliff[N] = float(np.mean(ss))
        print(f"  N={N:2d} ({'≤r' if N <= 8 else '>r'}): LP-SSIM={np.mean(ss):.3f}  keep(row-rank)≈{keep}")

    # ===== #1 WHICH-PAIRS (N=8 failing vs passing support overlap) =====
    print("\n=== #1 WHICH-PAIRS (N=8: do failures have nested/overlapping supports?) ===")
    gg = torch.Generator().manual_seed(208); lows, highs = [], []
    for t in range(8):
        s = torch.cat([idx[d][torch.randperm(len(idx[d]), generator=gg)[:4]] for d in (0, 1)])
        Xn = dat[s].numpy(); sv = np.linalg.svd(Xn - mnp, full_matrices=False)
        keep = int((sv[1] > 1e-6 * sv[1][0]).sum()); V = sv[2][:keep].T
        R = recover(V, mnp, 8, seed=t)
        if len(R) < 8: R += [mnp] * (8 - len(R))
        C = np.array([[ssim(r, Xn[j]) for j in range(8)] for r in R]); ri, ci = linear_sum_assignment(-C)
        supp = [(Xn[j] > 0.3) for j in range(8)]
        for a, b in zip(ri, ci):
            sv_ss = ssim(R[a], Xn[b])
            ovl = max([(supp[b] & supp[k]).sum() / ((supp[b] | supp[k]).sum() + 1e-9) for k in range(8) if k != b])
            (lows if sv_ss < 0.9 else highs).append(ovl)
    print(f"  FAILING (SSIM<0.9): n={len(lows)} mean max-support-Jaccard-with-another={np.mean(lows) if lows else float('nan'):.3f}")
    print(f"  PASSING (SSIM≥0.9): n={len(highs)} mean max-support-Jaccard-with-another={np.mean(highs) if highs else float('nan'):.3f}")
    print(f"  → {'failures ARE higher-overlap (nested-support) — mechanism, not noise' if (lows and highs and np.mean(lows)>np.mean(highs)) else 'no clear overlap difference'}")

    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(optimizer=opt_res, cliff=cliff, fail_ovl=lows, pass_ovl=highs), os.path.join(RESULTS, "robustness.pth"))
        print(f"[saved] {RESULTS}/robustness.pth")


if __name__ == "__main__":
    main()
