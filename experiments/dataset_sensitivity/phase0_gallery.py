"""Gallery-recovery PHASE 0 — the three pre-registered go/no-go pre-tests (plan §GO-NO-GO).

#1 FRAME: rank gallery images by LoRA-frame atom (one-image adapter, subspace-aligned to ΔW*) vs full-gradient
   atom (cosine of g(x)xᵀ to ΔW*). PASS = LoRA median true-member rank ≤ 10%·|G| AND ≤ ½ the full-grad rank.
#2 RESOLUTION: D(S*, one-swap) / D(S*, S*-reseeded) ≥ 3× at N=4 (subspace distance over K seeds); + whitened d²
   + the seed-cloud spread (noise floor). Needs a tiny GPU run (atlas has no single-swap adapter).
#3 COHERENCE: mutual coherence μ of the LoRA atom dictionary; greedy subspace-MP plant-and-recover; PASS =
   exact-rate ≥ 0.80 at low coherence (oracle-init, best case). Report exact-rate & right-digit-wrong-exemplar.
Observe-framed, this-attacker; DETECTION/RECOVERY not reconstruction. bsub GPU. Results → auditor yoado-d4.
"""
import argparse, os, torch, itertools
import torch.nn.functional as F
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set, forward_logits, subspace_cos
from experiments.data_utils import _load_dataset, _get_binary_label

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/phase0_gallery"
GPC = 50                    # gallery per class → |G|=100
N, T, RANK, LR, ACT = 4, 200, 8, 0.5, "gelu"
K_SEEDS = 15               # seeds for the resolution seed-cloud
N_TARGETS = 10             # S* targets for #1 and plant-recover
KSUB = 4                   # subspace dim for alignment


def gallery(ds, dev):
    tgt = ds.targets if torch.is_tensor(ds.targets) else torch.tensor(ds.targets)
    g = torch.Generator().manual_seed(7); imgs, labs, digs = [], [], []
    for d in (0, 1):
        idx = (tgt == d).nonzero(as_tuple=True)[0]
        pick = idx[torch.randperm(len(idx), generator=g)[:GPC]]
        for i in pick:
            imgs.append(ds.data[int(i)].to(torch.float64).unsqueeze(0) / 255.0)
            labs.append(float(_get_binary_label(int(d)))); digs.append(int(d))
    return torch.stack(imgs).to(dev), torch.tensor(labs, dtype=torch.float64, device=dev), torch.tensor(digs)


def full_grad_atom(frozen, b0, x0_i, y_i, act):
    W1 = frozen[0].clone().detach().requires_grad_(True)
    fz = {**frozen, 0: W1} if isinstance(frozen, dict) else [W1] + list(frozen[1:])
    A = {0: torch.zeros(RANK, frozen[0].shape[1], dtype=torch.float64, device=x0_i.device)}
    B = {0: torch.zeros(frozen[0].shape[0], RANK, dtype=torch.float64, device=x0_i.device)}
    out = forward_logits(x0_i.unsqueeze(0), fz, b0, A, B, act).view(-1)
    F.binary_cross_entropy_with_logits(out, y_i.view(1)).backward()
    return W1.grad.detach()


def topU(dW, k=KSUB):
    # CPU SVD: this GPU's float64 cusolver SVD fails to converge and falls back to a slow path, ~140 times.
    return torch.linalg.svd(dW.detach().to("cpu", torch.float64), full_matrices=False).U[:, :k].contiguous()


def align(Ua, Ub):
    """Mean principal cosine between two ORTHONORMAL subspaces (precomputed) — cheap k×k SVD, no 1000×784 SVD."""
    return torch.linalg.svdvals(Ua.transpose(-1, -2) @ Ub).mean().item()


def mp_select(dW_target, atom_U, N):
    """Greedy subspace matching pursuit: pick atoms whose col-space best covers the residual target subspace."""
    U = torch.linalg.svd(dW_target.detach().to("cpu", torch.float64), full_matrices=False).U[:, :2 * KSUB].clone()
    sel = []
    for _ in range(N):
        best, bs = -1, -1.0
        for i, Ua in enumerate(atom_U):
            if i in sel:
                continue
            sc = (U.transpose(-1, -2) @ Ua).norm().item()
            if sc > bs:
                bs, best = sc, i
        sel.append(best)
        Ua = atom_U[best]
        U = U - Ua @ (Ua.transpose(-1, -2) @ U)          # deflate: remove atom direction
        q, _ = torch.linalg.qr(U); U = q[:, :U.shape[1]]
    return sel


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(N // 2, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    gimg, glab, gdig = gallery(ds, dev); G = gimg.shape[0]
    gx0 = gimg - ds_mean
    B0_atk = draw_B0(900, out_f, RANK, dev)                # attacker's own fixed init
    print(f"[phase0] |G|={G} N={N} | building LoRA atoms + full-grad atoms")

    atom_dW, atom_U, atom_full = [], [], []
    for i in range(G):
        _, _, _, dWi = train_adapter(frozen, b0, B0_atk, gx0[i:i+1], glab[i:i+1], LR, T, act, RANK)
        atom_dW.append(dWi); atom_U.append(topU(dWi))
        atom_full.append(full_grad_atom(frozen, b0, gx0[i], glab[i], act))
    print("  atoms built")

    # ---- targets: N_TARGETS random S* (N/2 per class) ----
    rng = torch.Generator().manual_seed(11); c0 = (gdig == 0).nonzero().view(-1); c1 = (gdig == 1).nonzero().view(-1)
    targets = []
    for t in range(N_TARGETS):
        s = torch.cat([c0[torch.randperm(len(c0), generator=rng)[:N//2]], c1[torch.randperm(len(c1), generator=rng)[:N//2]]])
        _, _, _, dWt = train_adapter(frozen, b0, B0_atk, gx0[s], glab[s], LR, T, act, RANK)
        targets.append((s.tolist(), dWt))

    # ===== #1 FRAME =====
    lora_ranks, full_ranks = [], []
    for s, dWt in targets:
        U_t = topU(dWt)
        la = torch.tensor([align(atom_U[i], U_t) for i in range(G)])
        fa = torch.tensor([abs((atom_full[i] * dWt).sum().item()) / (atom_full[i].norm() * dWt.norm() + 1e-12).item() for i in range(G)])
        lord = torch.argsort(la, descending=True); ford = torch.argsort(fa, descending=True)
        for mi in s:
            lora_ranks.append((lord == mi).nonzero().item()); full_ranks.append((ford == mi).nonzero().item())
    import numpy as np
    lmed, fmed = float(np.median(lora_ranks)), float(np.median(full_ranks))
    p1 = (lmed <= 0.10 * G) and (lmed <= 0.5 * fmed)
    print(f"\n#1 FRAME: LoRA median member rank={lmed:.1f}/{G} ({100*lmed/G:.0f}%) | full-grad median={fmed:.1f} "
          f"| PASS={p1} (need ≤{0.10*G:.0f} AND ≤½·{fmed:.0f}={0.5*fmed:.0f})")

    # ===== #2 RESOLUTION (tiny GPU run) =====
    s_star = targets[0][0]; swap = s_star.copy()
    pool0 = [int(i) for i in c0.tolist() if i not in s_star]; swap[0] = pool0[0]   # one-image swap (same class)
    seeds = [1200 + j for j in range(K_SEEDS)]
    U_star = [topU(train_adapter(frozen, b0, draw_B0(sd, out_f, RANK, dev), gx0[torch.tensor(s_star)], glab[torch.tensor(s_star)], LR, T, act, RANK)[3]) for sd in seeds]
    U_swp = [topU(train_adapter(frozen, b0, draw_B0(sd, out_f, RANK, dev), gx0[torch.tensor(swap)], glab[torch.tensor(swap)], LR, T, act, RANK)[3]) for sd in seeds]
    d_reseed = [1 - align(U_star[a], U_star[b]) for a, b in itertools.combinations(range(K_SEEDS), 2)]
    d_swap = [1 - align(U_star[a], U_swp[b]) for a in range(K_SEEDS) for b in range(K_SEEDS)]
    reseed_med, swap_med = float(np.median(d_reseed)), float(np.median(d_swap))
    ratio = swap_med / (reseed_med + 1e-12)
    d_reseed_a, d_swap_a = np.array(d_reseed), np.array(d_swap)
    cohen = (d_swap_a.mean() - d_reseed_a.mean()) / (np.sqrt((d_swap_a.var() + d_reseed_a.var()) / 2) + 1e-12)
    p2 = ratio >= 3.0
    print(f"\n#2 RESOLUTION: D(reseed) med={reseed_med:.4f} (noise floor) | D(one-swap) med={swap_med:.4f} "
          f"| ratio={ratio:.2f}× | Cohen d={cohen:.2f} | PASS={p2} (need ≥3×; else whitened-d² significant)")

    # ===== #3 COHERENCE + PLANT-RECOVER =====
    coh = 0.0
    for a in range(G):
        for b in range(a + 1, G):
            coh = max(coh, align(atom_U[a], atom_U[b]))
    exact, rdwe = 0, 0
    for s, dWt in targets:
        picked = mp_select(dWt, atom_U, N); ps = set(picked); ss = set(s)
        exact += int(ps == ss)
        # right-digit-wrong-exemplar: correct class multiset but wrong indices
        rdwe += int(ps != ss and sorted(gdig[torch.tensor(picked)].tolist()) == sorted(gdig[torch.tensor(s)].tolist()))
    er = exact / N_TARGETS
    p3 = er >= 0.80
    print(f"\n#3 COHERENCE: dictionary μ(max pairwise subspace-cos)={coh:.3f} (OMP wants <{1/(2*N-1):.3f}) | "
          f"plant-recover EXACT={exact}/{N_TARGETS}={er:.2f} | right-digit-wrong-exemplar={rdwe}/{N_TARGETS} | PASS={p3} (need ≥0.80)")

    verdict = "GO — all 3 clear" if (p1 and p2 and p3) else "NO-GO / reshape (see failing gate)"
    print(f"\n==== PHASE-0 VERDICT: {verdict} ==== [#1 {p1} · #2 {p2} · #3 {p3}]")

    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(lmed=lmed, fmed=fmed, p1=p1, reseed_med=reseed_med, swap_med=swap_med, ratio=ratio,
                        cohen=cohen, d_reseed=d_reseed, d_swap=d_swap, mu=coh, exact_rate=er, rdwe=rdwe,
                        p2=p2, p3=p3, G=G, N=N),
                   os.path.join(RESULTS, "phase0.pth"))
        print(f"[saved] {RESULTS}/phase0.pth")


if __name__ == "__main__":
    main()
