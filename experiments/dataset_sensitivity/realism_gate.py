"""Realism gate for the row-span theorem (auditor yoado-d4) — is 10/10 real or an A₀=0 artifact?

Part 1 — SCALE: the membership selector at |G|=10k (the residual gap 0.74 vs 1e-14 predicts it holds, but a
near-collinear gallery image is the only failure mode — check it at realistic gallery size).
Part 2 — STANDARD INIT (the realism gate): this repo uses A₀=0/B₀ random; HF PEFT default is A₀ random Kaiming/
B₀=0. With B₀=0 the data enters B first and A_T = A₀ + drift, so row(ΔW) MIXES the random A₀ rows with
span{xᵢ} → the membership test is NO LONGER exact; member residual moves from 1e-14 to O(‖A₀‖/‖drift‖),
seed-dependent, and shrinks only as drift dominates (large T·lr). Measure member vs non-member residual vs T
under standard init: that one curve says whether the theorem is an A₀=0 convention artifact or a LoRA property.
bsub GPU. SCOPE: first-layer (or any layer whose input you can name); N≤r; closed-world; this-attacker.
"""
import argparse, os, math, torch, numpy as np
import torch.nn.functional as F
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set, forward_logits
from experiments.data_utils import _load_dataset, _get_binary_label

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/realism_gate"
N, RANK, T, LR, ACT = 4, 8, 200, 0.5, "gelu"


def big_gallery(ds, gpc, dev):
    tgt = ds.targets if torch.is_tensor(ds.targets) else torch.tensor(ds.targets)
    g = torch.Generator().manual_seed(7); imgs, labs, digs = [], [], []
    for d in (0, 1):
        idx = (tgt == d).nonzero(as_tuple=True)[0]
        pick = idx[torch.randperm(len(idx), generator=g)[:gpc]]
        for i in pick:
            imgs.append(ds.data[int(i)].to(torch.float64).view(-1) / 255.0)
            labs.append(float(_get_binary_label(int(d)))); digs.append(d)
    return torch.stack(imgs).to(dev), torch.tensor(labs, dtype=torch.float64, device=dev), torch.tensor(digs)


def row_space(dW, tol=1e-6):
    svd = torch.linalg.svd(dW.detach().to("cpu", torch.float64), full_matrices=False)
    return svd.Vh[:(svd.S > tol * svd.S[0]).sum().item()].transpose(-1, -2).contiguous()


def resid(gx0_cpu, gnorm, V):
    proj = gx0_cpu @ V @ V.transpose(-1, -2)
    return (gx0_cpu - proj).norm(dim=1) / (gnorm + 1e-12)


def draw_A_kaiming(seed, rank, in_f, dev):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    bound = 1.0 / math.sqrt(in_f)
    return ((torch.rand(rank, in_f, generator=g, dtype=torch.float64) * 2 - 1) * bound).to(dev)


def train_stdinit(frozen, b0, x0, y, lr, T, act, rank, seed):
    """HF-PEFT-style init: A₀ random Kaiming, B₀ = 0."""
    in_f, out_f = frozen[0].shape[1], frozen[0].shape[0]
    A0 = draw_A_kaiming(seed, rank, in_f, x0.device).requires_grad_(True)
    B0 = torch.zeros(out_f, rank, dtype=torch.float64, device=x0.device, requires_grad=True)
    opt = torch.optim.SGD([A0, B0], lr=lr); A = {0: A0}; B = {0: B0}
    for _ in range(T):
        out = forward_logits(x0, frozen, b0, A, B, act).view(-1)
        loss = F.binary_cross_entropy_with_logits(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    return (B[0] @ A[0]).detach()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(2, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]; dsm = ds_mean.reshape(-1)

    # ---- Part 1: |G|=10k selector (A₀=0 repo init) ----
    gpc = 5000; gimg, glab, gdig = big_gallery(ds, gpc, dev); G = gimg.shape[0]
    gx0 = (gimg - dsm).to("cpu"); gnorm = gx0.norm(dim=1)
    c0 = (gdig == 0).nonzero().view(-1); c1 = (gdig == 1).nonzero().view(-1)
    rng = torch.Generator().manual_seed(3); B0_atk = draw_B0(900, out_f, RANK, dev)
    exact, mres, nres = 0, [], []
    for t in range(10):
        s = torch.cat([c0[torch.randperm(len(c0), generator=rng)[:N//2]], c1[torch.randperm(len(c1), generator=rng)[:N//2]]])
        _, _, _, dWt = train_adapter(frozen, b0, B0_atk, (gimg[s]-dsm), glab[s], LR, T, act, RANK)
        r = resid(gx0, gnorm, row_space(dWt)); cand = set(torch.argsort(r)[:N].tolist())
        exact += int(cand == set(s.tolist())); mres += r[s].tolist(); nres.append(r.mean().item())
    print(f"[Part1 SCALE |G|={G}] EXACT={exact}/10  member-resid={np.mean(mres):.2e} (max {np.max(mres):.2e})  "
          f"non-member-resid≈{np.mean(nres):.3f}  → {'holds at 10k' if exact>=9 else 'FAILS at scale'}")

    # ---- Part 2: STANDARD INIT (A₀ random, B₀=0) residual vs T ----
    print(f"\n[Part2 STANDARD-INIT realism gate] A₀ random Kaiming, B₀=0 | member vs non-member residual vs T")
    sg = big_gallery(ds, 100, dev); sgi, sgl, sgd = sg; sgx0 = (sgi - dsm).to("cpu"); sgn = sgx0.norm(dim=1)
    sc0 = (sgd == 0).nonzero().view(-1); sc1 = (sgd == 1).nonzero().view(-1)
    rng2 = torch.Generator().manual_seed(5)
    curve = {}
    for Tt in [50, 200, 1000, 5000]:
        mr, nr = [], []
        for t in range(6):
            s = torch.cat([sc0[torch.randperm(len(sc0), generator=rng2)[:N//2]], sc1[torch.randperm(len(sc1), generator=rng2)[:N//2]]])
            dWt = train_stdinit(frozen, b0, (sgi[s]-dsm), sgl[s], LR, Tt, act, RANK, seed=1000+t)
            r = resid(sgx0, sgn, row_space(dWt))
            mr += r[s].tolist(); nr.append(r.mean().item())
        curve[Tt] = (float(np.mean(mr)), float(np.mean(nr)))
        print(f"  T={Tt:5d}: member-resid={np.mean(mr):.4f}  non-member-resid={np.mean(nr):.4f}  "
              f"gap={np.mean(nr)-np.mean(mr):+.4f}")
    verdict = ("member resid → ~0 as T grows → theorem holds for standard init once drift dominates A₀"
               if curve[5000][0] < 0.05 else
               "member resid STAYS well above 0 → the exact test is an A₀=0 CONVENTION artifact; standard init needs init-projection/statistical recovery")
    print(f"  → {verdict}")

    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(scale_exact=exact, scale_member=float(np.mean(mres)), G=G, stdinit_curve=curve),
                   os.path.join(RESULTS, "realism.pth"))
        print(f"[saved] {RESULTS}/realism.pth")


if __name__ == "__main__":
    main()
