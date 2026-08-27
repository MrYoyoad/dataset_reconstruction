#!/usr/bin/env python
"""
Arm B — per-image adapter SENSITIVITY vs dataset size N (the 1/N dilution law).

First spine of the dataset-composition-sensitivity program
(notes/dataset_sensitivity_program_plan.md). Circularity-free opener: no typicality
proxy needed; de-risks the M2 seed-noise-floor machinery.

Measures, per N:
  swap_abs(N)    = mean_i ‖ΔW(D) − ΔW(D with image i swapped for a held-out same-class one)‖_F
  reseed(N)      = mean_j ‖ΔW_reseed_j(D) − ΔW(D)‖_F        (init B0 varies)   [σ_reseed]
  repeat(N)      = mean_j ‖ΔW_repeat_j(D) − ΔW(D)‖_F        (SAME seed, re-run) [σ_repeat, GPU-nondet]
  rho(N)         = swap_abs(N) / reseed(N)                  (the sensitivity SNR)
  ratio(N)       = repeat(N) / reseed(N)                    (if non-trivial -> rho UNINTERPRETABLE)
Everything on ΔW = B·A (gauge-invariant), NEVER raw B,A. A principal-angle subspace term is
reported ALONGSIDE Frobenius (rule 5). swap in paired-seed (oracle) AND independent-seed (realistic).
σ_repeat DROPS non-finite draws + reports the dropped count (D3). raw swap_abs(N) and reseed(N)
reported un-normalized so the normalization's contribution to any 1/N exponent is visible (S2).

bsub-only. mnist / gelu / binary (nc=2, memorizes at every rank). float64.
"""
import os, json, math, argparse
import torch
import torch.nn.functional as F

import experiments.jacobian_spectrum as J
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.data_utils import get_finetuning_data, get_control_images_in_distribution
from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/arm_b_dilution"


# ---- self-contained forward (mirrors _partial_lora_forward, avoids signature drift) ----
def forward_logits(x, frozen, b0, A, B, act, target_layers=(0,), scaling=1.0):
    n_layers = len(frozen)
    h = x.view(x.shape[0], -1)
    for l in range(n_layers):
        w = frozen[l] + scaling * (B[l] @ A[l]) if l in target_layers else frozen[l]
        bias = b0 if l == 0 else None
        h = F.linear(h, w, bias)
        if l < n_layers - 1:
            h = act(h)
    return h


def draw_B0(seed, out_features, rank, device):
    # kaiming_uniform_(a=sqrt(5)) on [out, rank] == U(-1/sqrt(rank), 1/sqrt(rank)); matches _draw_B0.
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    bound = 1.0 / math.sqrt(rank)
    B = (torch.rand(out_features, rank, generator=g, dtype=torch.float64) * 2 - 1) * bound
    return B.to(device)


def train_adapter(frozen, b0, B0_0, x0, y, lr, T, act, rank, target_layers=(0,), scaling=1.0):
    """Standard SGD LoRA fine-tune (base frozen, A0=0, B0 given). Returns (A,B dicts, max_bce, dW0)."""
    in_f = frozen[target_layers[0]].shape[1]
    A0 = torch.zeros(rank, in_f, dtype=torch.float64, device=x0.device, requires_grad=True)
    B0 = B0_0.clone().detach().to(x0.device).requires_grad_(True)
    opt = torch.optim.SGD([A0, B0], lr=lr)
    A = {target_layers[0]: A0}
    B = {target_layers[0]: B0}
    for _ in range(T):
        out = forward_logits(x0, frozen, b0, A, B, act, target_layers, scaling)
        loss = F.binary_cross_entropy_with_logits(out.view(-1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = forward_logits(x0, frozen, b0, A, B, act, target_layers, scaling).view(-1)
        per_bce = F.binary_cross_entropy_with_logits(out, y, reduction="none")
        max_bce = per_bce.max().item()
        dW0 = (scaling * (B[target_layers[0]] @ A[target_layers[0]])).detach()
    return A, B, max_bce, dW0


def fro(a, b):
    return (a - b).norm().item()


def subspace_cos(dW_a, dW_b, k):
    # mean cosine of principal angles between the top-k left singular subspaces of ΔW (1 = aligned).
    Ua = torch.linalg.svd(dW_a, full_matrices=False).U[:, :k]
    Ub = torch.linalg.svd(dW_b, full_matrices=False).U[:, :k]
    s = torch.linalg.svdvals(Ua.transpose(-1, -2) @ Ub)
    return s.mean().item()


def build_set(n_per_class, seed, device, dataset="mnist"):
    x_ft, y_ft, digits, indices = get_finetuning_data(n_per_class, seed=seed, device=device, dataset=dataset)
    return x_ft.to(torch.float64), y_ft.to(torch.float64), list(digits)


def run_for_N(N, K, n_targets, lr, T, rank, device, ref_seed=0, subk=None, tag="", dataset="mnist"):
    n_per_class = N // 2
    act = make_activation("gelu")
    # honest θ0 + ds_mean from the reference set (frozen thereafter)
    x_ft, y_ft, digits = build_set(n_per_class, seed=42, device=device, dataset=dataset)
    _, frozen, b0, B0_all, ds_mean = _honest_target(x_ft, y_ft, T, rank, "gelu", lr, device, dataset, num_classes=2)
    x0 = (x_ft - ds_mean)
    out_f = frozen[0].shape[0]
    subk = subk or rank

    mean = lambda a: (sum(a) / len(a)) if a else float("nan")
    seeds = [1000 + j for j in range(K)]

    # reference (seed 0) — known-init bracket + subspace + norm
    B0_ref = draw_B0(ref_seed, out_f, rank, device)
    _, _, mbce_r, dW_ref = train_adapter(frozen, b0, B0_ref, x0, y_ft, lr, T, act, rank)
    ref_norm = dW_ref.norm().item()

    # reseeds: ΔW(D, seed_j) — the init ensemble (reused for the paired-diff AND the unknown-init noise)
    dW_refs = {}
    for s in seeds:
        _, _, _, dW = train_adapter(frozen, b0, draw_B0(s, out_f, rank, device), x0, y_ft, lr, T, act, rank)
        if torch.isfinite(dW).all():
            dW_refs[s] = dW
    reseed_dropped = K - len(dW_refs)
    stk = torch.stack(list(dW_refs.values()))                       # [Kf, out, in]
    reseed_mean = stk.mean(0)
    reseed_noise = ((stk - reseed_mean).flatten(1).norm(dim=1) ** 2).mean().sqrt().item()  # RMS spread = unknown-init noise
    reseed_floor_vs_ref = mean([fro(dW, dW_ref) for dW in dW_refs.values()])

    # σ_repeat: same D, SAME init seed, re-run K× (GPU-nondet floor). DROP non-finite (D3).
    repeat = []
    for _ in range(K):
        _, _, _, dW = train_adapter(frozen, b0, B0_ref, x0, y_ft, lr, T, act, rank)
        d = fro(dW, dW_ref)
        if math.isfinite(d):
            repeat.append(d)
    repeat_dropped = K - len(repeat)
    repeat_floor = mean(repeat)

    # swap targets (replace image i by a held-out same-class control; ds_mean FROZEN)
    controls, _, _ = get_control_images_in_distribution(digits, seed=123, dataset=dataset)
    controls = controls.to(torch.float64).to(device)
    tgt = list(range(min(n_targets, N)))

    known_sig, subs, coh, incoh, coh_vecs = [], [], [], [], []
    w_sens, w_pval, w_qeff, w_skew, w_kurt, w_erank = [], [], [], [], [], []
    n_folds = 5
    for i in tgt:
        x_sw = x_ft.clone(); x_sw[i] = controls[i]
        x0_sw = (x_sw - ds_mean)                                     # frozen ds_mean
        # KNOWN-init bracket: paired at the ref seed (init held fixed) -> signal vs GPU-nondet floor
        _, _, _, dW_sw0 = train_adapter(frozen, b0, B0_ref, x0_sw, y_ft, lr, T, act, rank)
        known_sig.append(fro(dW_sw0, dW_ref))
        subs.append(subspace_cos(dW_sw0, dW_ref, subk))
        # UNKNOWN-init bracket: PAIRED per-seed difference v_j = ΔW(D,seed_j) - ΔW(D_swap,seed_j) (init CANCELS per pair)
        vs, vs_reseed = [], []
        for s, dW_ref_s in dW_refs.items():
            _, _, _, dW_sw_s = train_adapter(frozen, b0, draw_B0(s, out_f, rank, device), x0_sw, y_ft, lr, T, act, rank)
            v = dW_ref_s - dW_sw_s
            if torch.isfinite(v).all():
                vs.append(v); vs_reseed.append(dW_ref_s)            # keep v_j aligned with its reseed (same seed j)
        if vs:
            vst = torch.stack(vs)
            v_coh = vst.mean(0)                                      # [out,in] init-averaged per-image shift
            coh.append(v_coh.norm().item())                         # ‖mean_j v_j‖ = E_seeds signal
            coh_vecs.append(v_coh)
            incoh.append(vst.flatten(1).norm(dim=1).mean().item())  # mean_j ‖v_j‖ (per-pair)
        # WHITENED metric (permutation-null + K-fold cross-fit) — the honest, bias-corrected detection statistic
        if len(vs) >= 2 * n_folds:
            # metric is small post-hoc matrix work → run it on CPU (matches its CPU self-test; avoids the
            # device mismatch and the GPU-thread oversubscription the module flagged).
            ws = whitened_sensitivity([v.cpu() for v in vs], [r.cpu() for r in vs_reseed],
                                      n_folds=n_folds, p_max=3, n_perm=500, seed=i)
            w_sens.append(ws["sensitivity"]); w_pval.append(ws["pvalue"]); w_qeff.append(ws["qeff_count"])
            w_skew.append(ws["gaussianity_skew"]); w_kurt.append(ws["gaussianity_kurt"])
            w_erank.append(ws["sigma_eff_rank"])

    # --- DIRECTIONAL / WHITENED analysis (total energy misses DIRECTION): how much of the swap lies
    #     OUTSIDE the seed-noise subspace (perfectly recoverable there), + per-direction whitened SNR = discrete q_eff.
    noise_c = (stk - reseed_mean).reshape(len(dW_refs), -1)          # [Kf, D]
    _, Sn, Vhn = torch.linalg.svd(noise_c, full_matrices=False)     # Vhn: [Kf, D] noise directions
    r_noise = int((Sn > 1e-9 * Sn[0]).sum().item())
    Vn = Vhn[:r_noise]                                              # [r_noise, D] the seed-noise subspace
    noise_std = Sn[:r_noise] / math.sqrt(max(len(dW_refs) - 1, 1))  # per-direction sample std
    frac_out, whit_max, qeff_dir = [], [], []
    for v_coh in coh_vecs:
        vf = v_coh.reshape(-1)
        vn2 = (vf @ vf).item()
        proj = Vn @ vf                                              # swap components along the noise directions
        frac_out.append(1.0 - (proj @ proj).item() / (vn2 + 1e-30))  # energy OUTSIDE noise subspace = recoverable
        dsnr = proj.abs() / (noise_std + 1e-30)                     # per-direction whitened SNR (within the noise subspace)
        whit_max.append(dsnr.max().item())
        qeff_dir.append(int((dsnr > 1.0).sum().item()))            # # noise-dirs the swap clears

    unknown_sig = mean(coh)
    res = dict(
        N=N, n_per_class=n_per_class, rank=rank, lr=lr, T=T, ref_seed=ref_seed, n_targets=len(tgt),
        ref_max_bce=mbce_r, ref_dW_norm=ref_norm, memorized=bool(mbce_r < 1e-3),
        reseed_dropped=reseed_dropped, repeat_dropped=repeat_dropped, swap_subspace_cos=mean(subs),
        # KNOWN-init bracket (init held fixed): signal vs the GPU-nondet floor -> attacker upper bound
        known_init_signal=mean(known_sig), repeat_floor=repeat_floor,
        known_init_SNR=(mean(known_sig) / repeat_floor) if (repeat and repeat_floor > 0) else float("inf"),
        # UNKNOWN-init bracket (privacy-relevant): init-CANCELLED paired-diff signal vs the reseed spread
        unknown_init_signal_coherent=unknown_sig,                   # ‖E_seeds[ΔW(D) - ΔW(D')]‖
        unknown_init_signal_incoherent=mean(incoh),                 # E_seeds‖ΔW(D) - ΔW(D')‖ (per-pair)
        reseed_noise=reseed_noise,                                  # unknown-init noise (RMS spread around the seed-mean)
        unknown_init_SNR=(unknown_sig / reseed_noise) if reseed_noise else float("nan"),
        reseed_floor_vs_ref=reseed_floor_vs_ref,
        # DIRECTIONAL (the honest recoverability — direction, not total energy): the discrete q_eff.
        noise_rank=r_noise,
        swap_frac_outside_noise=mean(frac_out),   # frac of swap energy in noise-FREE directions = recoverable
        swap_whitened_snr_max=mean(whit_max),     # max per-direction whitened SNR WITHIN the noise subspace
        swap_qeff_directional=mean(qeff_dir),     # # noise-directions the swap clears (q_eff-style count)
        # WHITENED (permutation-null + K-fold cross-fit): the AUDIT-BLESSED, bias-corrected detection statistic
        whitened_sensitivity=mean(w_sens), whitened_pvalue=mean(w_pval), whitened_qeff=mean(w_qeff),
        gaussianity_skew=mean(w_skew), gaussianity_exkurt=mean(w_kurt), sigma_eff_rank=mean(w_erank),
    )
    # save ΔW=BA for Phase-2 clustering (ref adapter + seed-mean adapter + metadata)
    os.makedirs(RESULTS, exist_ok=True)
    ds_tag = "" if dataset == "mnist" else f"_{dataset}"
    torch.save(dict(dW_ref=dW_ref.cpu(), dW_seed_mean=reseed_mean.cpu(), N=N, digits=digits,
                    rank=rank, lr=lr, T=T, dataset=dataset, metrics=res),
               os.path.join(RESULTS, f"armb_N{N}{ds_tag}{tag}.pth"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N_list", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n_targets", type=int, default=4)  # paired-diff = K trains/target; keep modest
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--stage0", action="store_true", help="tiny sanity: N=2, K=3, 1 target")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"],
                    help="base dataset (checkpoint models/weights-<ds>_gelu.pth); mnist = byte-identical legacy path")
    args = ap.parse_args()
    dev = args.device
    ds = args.dataset

    if args.stage0:
        print(f"=== STAGE-0 SANITY (N=8, K=12, 2 targets — exercises the whitened metric) [dataset={ds}] ===")
        r = run_for_N(8, K=12, n_targets=2, lr=args.lr, T=args.T, rank=args.rank, device=dev, tag="_stage0", dataset=ds)
        print(json.dumps(r, indent=2))
        assert math.isfinite(r["ref_dW_norm"]) and r["ref_dW_norm"] > 0, "ref ΔW degenerate"
        assert math.isfinite(r["reseed_noise"]) and r["reseed_noise"] > 0, "reseed_noise degenerate"
        assert math.isfinite(r["whitened_sensitivity"]), "whitened_sensitivity NaN (metric integration broken)"
        print("STAGE-0 OK")
        return

    all_res = []
    for N in args.N_list:
        print(f"\n===== N={N} =====", flush=True)
        r = run_for_N(N, args.K, args.n_targets, args.lr, args.T, args.rank, dev, dataset=ds)
        all_res.append(r)
        if not r["memorized"]:
            print(f"WARNING: N={N} NOT memorized (max_bce={r['ref_max_bce']:.2e} > 1e-3) — "
                  f"EXCLUDE from the 1/N exponent fit (off-convergence sensitivity is confounded).", flush=True)
        print(json.dumps(r), flush=True)
    # {K,2K} stability at the smallest N (audit S4)
    print("\n===== {K,2K} stability check at N=%d =====" % args.N_list[0], flush=True)
    r2 = run_for_N(args.N_list[0], 2 * args.K, args.n_targets, args.lr, args.T, args.rank, dev, tag="_2K", dataset=ds)
    print("reseed_noise K vs 2K:", all_res[0]["reseed_noise"], r2["reseed_noise"], flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    ds_tag = "" if ds == "mnist" else f"_{ds}"
    with open(os.path.join(RESULTS, f"arm_b_summary{ds_tag}.json"), "w") as f:
        json.dump(dict(results=all_res, k2k=r2), f, indent=2)
    print("\n=== SUMMARY (N | whitened_sensitivity | p-value | whitened_qeff | eff_rank(Σ) | mem) ===")
    for r in all_res:
        print(f"N={r['N']:>3}: whitened_sens={r['whitened_sensitivity']:.3f}  p={r['whitened_pvalue']:.3f}  "
              f"w_qeff={r['whitened_qeff']:.2f}  effrankΣ={r['sigma_eff_rank']:.1f}  "
              f"mem={r['memorized']} (max_bce={r['ref_max_bce']:.2e})", flush=True)


if __name__ == "__main__":
    main()
