#!/usr/bin/env python
"""
OOD-STYLE DIGIT INJECTION — do visually-MNIST-like digits from a DIFFERENT source (USPS,
different handwriting style / scanner pipeline) leak MORE when injected into an MNIST LoRA
fine-tune set, and is that extra leakage PREDICTED by the base-model gradient norm g0?

Program: dataset-composition-sensitivity battery (notes/dataset_sensitivity_program_plan.md).
Runs on the FIXED 3-way whitened metric (whitened_metric.py), paired-per-seed pattern of
arm_b/arm_d. mnist / gelu MLP / binary / rank 8 / T=1000 / lr 0.5 / float64. bsub-only.

SETS (N=16 default):
  baseline-D = the arm construction: get_finetuning_data(N/2, seed=42) MNIST-test set
               (identical to arm_b's build_set / arm_d's build_base pool with pool_n=N/2).
  OOD-D      = baseline-D with n_ood (default 2) of the CLASS-1 slots replaced by USPS
               digits of the SAME digit class (labels + class balance IDENTICAL, positions
               FIXED = the first n_ood class-1 slots in set order).
  USPS test images are 16x16 grayscale in [0,1]; we upsample 16->28 BILINEAR
  (align_corners=False), clamp to [0,1], float64 — same tensor format as MNIST members.
  Binary label via _get_binary_label(digit) — the SAME odd/even map as MNIST.

MEASUREMENTS (K seeds, paired per-seed diff v_j = dW(D,s_j) - dW(D_swap,s_j), init CANCELS,
everything on dW = B@A, gauge-invariant; baseline ensemble per set):
  (a) PER-IMAGE sensitivity of each OOD member measured WITHIN OOD-D: swap USPS image ->
      a HELD-OUT USPS image of the same digit. And the MATCHED MNIST member at the SAME
      position measured WITHIN baseline-D: swap MNIST -> held-out MNIST same digit
      (get_control_images_in_distribution, seed=123 — the arms' control convention).
      RATIO sens_ood/sens_mnist per position = the OOD AMPLIFICATION.
  (b) CROSS swap WITHIN OOD-D: swap the USPS member -> the ORIGINAL MNIST image of the
      same digit at that slot (style change ONLY, same digit, same slot) — is the STYLE
      itself what the adapter notices? (For n_ood=1 the cross-swapped set == baseline-D
      exactly; for n_ood>1 the other USPS members stay in place.)
  (c) PREDICTOR FIRST (before ANY training): print g0 (per-image layer-0 full-weight
      gradient norm at theta_0 — margin_vs_sensitivity's P2 quantity) and base margin m0
      for EVERY member of both sets. PREDICTION printed up front: if USPS members have
      LARGER g0 they should leak MORE; the measured ratio then tests it. Also
      Spearman(g0, sensitivity) pooled over all 2*n_ood measured members (SMALL-n, labeled).

CENTERING (mandatory): x0 = x - ds_mean with ds_mean the FROZEN MNIST reference-set mean
from _honest_target — USPS members are centered with the SAME frozen MNIST ds_mean, exactly
like any private image (the victim's pipeline doesn't know a member is OOD).

CONFOUNDS TO WEIGH (also in the saved JSON):
  * USPS global statistics differ from MNIST (stroke thickness, contrast, digit scale, and
    16->28 bilinear upsampling makes USPS members SMOOTHER / lower high-frequency energy
    than native MNIST). g0 may therefore partly track mere norm/contrast, not "style
    rarity" — we print ||x0|| (centered pixel norm) per member alongside g0 so that
    confound is visible.
  * ds_mean is MNIST's; centered USPS members carry a systematic offset (MNIST mean digit
    subtracted from a USPS digit). That offset IS the OOD-ness under study, but it means
    amplification mixes "style" with "mis-centering". Measurement (b) (style-only swap at
    fixed slot) is the cleaner read on style per se.
  * n_ood members share the set with 16-n_ood MNIST members: any amplification is measured
    in MIXED context (the intended scenario), not "pure USPS fine-tune".
"""
import os, json, math, argparse
import torch
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms

from experiments.configs import DATASETS_DIR
from experiments.jacobian_spectrum import make_activation
from experiments.data_utils import _get_binary_label, get_control_images_in_distribution
from experiments.dataset_sensitivity.arm_b_dilution import draw_B0, train_adapter
from experiments.dataset_sensitivity.arm_d_context import build_base
from experiments.dataset_sensitivity.margin_vs_sensitivity import (
    margins, layer0_grad_norms, _zero_adapter, spearman,
)
from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/ood_injection"
N_FOLDS = 5
USPS_SELECT_SEED = 777       # deterministic USPS per-digit selection
CTRL_SEED = 123              # the arms' held-out same-class MNIST control convention


# ---------------------------------------------------------------------------
# USPS loading + preprocessing (16x16 -> 28x28)
# ---------------------------------------------------------------------------
def load_usps(root=None):
    """USPS test split (16x16 grayscale, ToTensor -> [1,16,16] in [0,1]).

    LOUD guard: WEXAC compute nodes may lack internet. If this fails, pre-download ON THE
    LOGIN NODE first — the dataset is tiny (~2k test images)."""
    root = root or DATASETS_DIR
    try:
        return torchvision.datasets.USPS(
            root=root, train=False,
            transform=torchvision.transforms.ToTensor(), download=True)
    except Exception as e:
        raise RuntimeError(
            "\n" + "!" * 78 + "\n"
            "FATAL: could not load/download the USPS test set (root=%r).\n"
            "Compute nodes may have NO internet. Fix: pre-download on the LOGIN node:\n"
            "    cd /home/projects/galvardi/yoado && \\\n"
            "    PYTHONPATH=dataset_reconstruction python -c \"import torchvision; \\\n"
            "      torchvision.datasets.USPS(root='%s', train=False, download=True)\"\n"
            "then resubmit this job. Original error: %r\n" % (root, root, e)
            + "!" * 78) from e


def usps_to_mnist_format(img):
    """[1,16,16] float in [0,1] -> [1,28,28] float64 in [0,1], BILINEAR upsample
    (align_corners=False), clamped. Same tensor format as an MNIST member."""
    x = img.to(torch.float64).unsqueeze(0)                       # [1,1,16,16]
    x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
    return x.squeeze(0).clamp(0.0, 1.0)                          # [1,28,28]


def pick_usps_by_digit(usps, needed_counts, seed=USPS_SELECT_SEED):
    """Deterministically pick `needed_counts[d]` distinct USPS test images per digit d.
    Returns dict digit -> list of ([1,28,28] float64 tensor, usps_index)."""
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(usps), generator=rng)
    out = {d: [] for d in needed_counts}
    for idx in perm.tolist():
        img, digit = usps[idx]
        d = int(digit)
        if d in out and len(out[d]) < needed_counts[d]:
            out[d].append((usps_to_mnist_format(img), idx))
        if all(len(out[d]) >= needed_counts[d] for d in out):
            break
    for d, c in needed_counts.items():
        assert len(out[d]) >= c, f"USPS test set has only {len(out[d])} images of digit {d}, need {c}"
    return out


# ---------------------------------------------------------------------------
# ensembles + paired-per-seed swap measurement (arm_b/arm_d pattern)
# ---------------------------------------------------------------------------
def _ensemble(frozen, b0, seeds, x0, y, lr, T, act, rank, out_f, device):
    """Baseline ensemble dW(D, seed_j); drops non-finite draws. Returns (dict, mbce@seeds[0])."""
    dW, mbce0 = {}, None
    for s in seeds:
        _, _, mbce, d = train_adapter(frozen, b0, draw_B0(s, out_f, rank, device),
                                      x0, y, lr, T, act, rank)
        if torch.isfinite(d).all():
            dW[s] = d
            if s == seeds[0]:
                mbce0 = mbce
    assert len(dW) >= 2 * N_FOLDS, \
        f"only {len(dW)}/{len(seeds)} finite baseline draws (< 2*N_FOLDS={2 * N_FOLDS}); metric starved"
    stk = torch.stack(list(dW.values()))
    noise = ((stk - stk.mean(0)).flatten(1).norm(dim=1) ** 2).mean().sqrt().item()
    return dW, mbce0, noise


def _swap_sensitivity(dW_base, frozen, b0, x0_sw, y, lr, T, act, rank, out_f, device, seed_tag):
    """Paired per-seed whitened sensitivity of a single-image swap vs the given baseline
    ensemble (init CANCELS per pair). Mirrors arm_d's _measure_target metric block."""
    vs, vs_reseed, dropped = [], [], 0
    for s, dW_ref_s in dW_base.items():
        _, _, _, dW_sw = train_adapter(frozen, b0, draw_B0(s, out_f, rank, device),
                                       x0_sw, y, lr, T, act, rank)
        v = dW_ref_s - dW_sw
        if torch.isfinite(v).all():
            vs.append(v); vs_reseed.append(dW_ref_s)
        else:
            dropped += 1
    coherent = torch.stack(vs).mean(0).norm().item() if vs else float("nan")
    sens = pval = qeff = float("nan")
    if len(vs) >= 2 * N_FOLDS:
        ws = whitened_sensitivity([v.cpu() for v in vs], [r.cpu() for r in vs_reseed],
                                  n_folds=N_FOLDS, p_max=3, n_perm=500, seed=int(seed_tag))
        sens, pval, qeff = ws["sensitivity"], ws["pvalue"], ws["qeff_count"]
    return dict(sensitivity=sens, pvalue=pval, qeff=qeff,
                coherent_signal=coherent, n_pairs=len(vs), dropped=dropped)


# ---------------------------------------------------------------------------
# main experiment
# ---------------------------------------------------------------------------
def run(N, K, n_ood, lr, T, rank, device, stage0=False):
    tag = "_stage0" if stage0 else ""
    act = make_activation("gelu")
    mean = lambda a: (sum(a) / len(a)) if a else float("nan")
    fin_mean = lambda a: mean([v for v in a if math.isfinite(v)])

    # ---- 0) USPS FIRST (gate: fail loudly before burning GPU time on training) ----
    usps = load_usps()
    print(f"USPS test set loaded: {len(usps)} images (16x16 -> 28x28 bilinear).", flush=True)

    # ---- 1) base model + baseline-D (the arm construction; pool_n=N/2 => pool == baseline-D) ----
    frozen, b0, ds_mean, x_base, y, digits = build_base(N, lr, T, device, pool_n=N // 2)
    assert x_base.shape[0] == N and y.shape[0] == N, f"pool != baseline-D (got {x_base.shape[0]}, want {N})"
    out_f = frozen[0].shape[0]
    seeds = [1000 + j for j in range(K)]

    # OOD positions: the FIRST n_ood class-1 slots in set order (fixed)
    ood_pos = [i for i in range(N) if int(y[i]) == 1][:n_ood]
    assert len(ood_pos) == n_ood, f"need {n_ood} class-1 slots, found {len(ood_pos)}"
    ood_digits = [int(digits[p]) for p in ood_pos]
    for d in ood_digits:
        assert _get_binary_label(d) == 1, f"digit {d} at an OOD slot is not class-1 (label map broken)"

    # USPS picks: per position, [0]=INJECTED member, [1]=HELD-OUT swap target (same digit)
    needed = {}
    for d in ood_digits:
        needed[d] = needed.get(d, 0) + 2
    by_digit = pick_usps_by_digit(usps, needed)
    cursor = {d: 0 for d in needed}
    usps_inj, usps_held, usps_idx = [], [], []
    for d in ood_digits:
        (im_a, ia), (im_b, ib) = by_digit[d][cursor[d]], by_digit[d][cursor[d] + 1]
        cursor[d] += 2
        usps_inj.append(im_a.to(device)); usps_held.append(im_b.to(device))
        usps_idx.append((ia, ib))

    # OOD-D: baseline-D with the n_ood class-1 slots replaced (labels/balance IDENTICAL)
    x_ood = x_base.clone()
    for k, p in enumerate(ood_pos):
        x_ood[p] = usps_inj[k]

    # held-out MNIST controls for the matched-member swaps (arms' seed-123 convention)
    ctrl, _, _ = get_control_images_in_distribution(digits, seed=CTRL_SEED, dataset="mnist")
    ctrl = ctrl.to(torch.float64).to(device)

    # frozen-MNIST-ds_mean centering — SAME for every member incl. USPS (mandatory)
    x0_base = x_base - ds_mean
    x0_ood = x_ood - ds_mean

    # ---- 2) PREDICTOR FIRST: g0 + m0 for every member of both sets, BEFORE any training ----
    A0, B0z = _zero_adapter(frozen, rank, device)
    m0_base = margins(x0_base, y, frozen, b0, act, A0, B0z).cpu()
    g0_base = layer0_grad_norms(x0_base, y, frozen, b0, act)
    m0_ood = margins(x0_ood, y, frozen, b0, act, A0, B0z).cpu()
    g0_ood = layer0_grad_norms(x0_ood, y, frozen, b0, act)
    assert torch.isfinite(m0_ood).all() and torch.isfinite(g0_ood).all(), "base margins/gradnorms non-finite"
    xn_base = x0_base.flatten(1).norm(dim=1).cpu()   # ||x0|| — the norm/contrast confound column
    xn_ood = x0_ood.flatten(1).norm(dim=1).cpu()

    print("\n=== PREDICTOR TABLE (BEFORE any training; x0 = x - frozen MNIST ds_mean) ===")
    print("--- OOD-D members (pos | source | digit | y | g0 | m0 | ||x0||) ---")
    for i in range(N):
        src = "USPS " if i in ood_pos else "MNIST"
        print(f"  pos{i:02d} | {src} | d={int(digits[i])} y={int(y[i])} | "
              f"g0={g0_ood[i]:.4e} | m0={m0_ood[i]:+.4f} | ||x0||={xn_ood[i]:.3f}")
    print("--- MATCHED PAIRS at the OOD slots (baseline-D MNIST member vs OOD-D USPS member) ---")
    for k, p in enumerate(ood_pos):
        print(f"  pos{p:02d} d={ood_digits[k]}: MNIST g0={g0_base[p]:.4e} m0={m0_base[p]:+.4f} "
              f"||x0||={xn_base[p]:.3f}   USPS g0={g0_ood[p]:.4e} m0={m0_ood[p]:+.4f} "
              f"||x0||={xn_ood[p]:.3f}   g0 ratio={g0_ood[p] / g0_base[p]:.3f}")
    g0_usps_mean = g0_ood[ood_pos].mean().item()
    g0_mnist_mean = g0_base[ood_pos].mean().item()
    g0_ratio = g0_usps_mean / g0_mnist_mean
    pred_more = g0_ratio > 1.0
    print(f"\nPREDICTION: mean g0 USPS={g0_usps_mean:.4e} vs matched MNIST={g0_mnist_mean:.4e} "
          f"(ratio {g0_ratio:.3f}) => USPS members predicted to leak "
          f"{'MORE' if pred_more else 'LESS (direction REVERSED vs the stated hypothesis)'}; "
          f"the measured amplification ratio below tests this.", flush=True)

    # ---- 3) baseline ensembles (one per set) ----
    print(f"\n--- ensembles: K={K} seeds x 2 sets ---", flush=True)
    ens_base, mbce_base, noise_base = _ensemble(frozen, b0, seeds, x0_base, y, lr, T, act, rank, out_f, device)
    ens_ood, mbce_ood, noise_ood = _ensemble(frozen, b0, seeds, x0_ood, y, lr, T, act, rank, out_f, device)
    for name, mb in (("baseline-D", mbce_base), ("OOD-D", mbce_ood)):
        if mb is None or mb >= 1e-3:
            print(f"WARNING: {name} NOT memorized (max_bce={mb if mb is not None else float('nan'):.2e}) — "
                  f"off-convergence sensitivity is confounded.", flush=True)
    print(f"reseed_noise: baseline-D={noise_base:.4f}  OOD-D={noise_ood:.4f} "
          f"(Sigma-scale diagnostic; a large gap itself flags the injection)", flush=True)

    # ---- 4) the three swap measurements per OOD position ----
    per_member = []
    for k, p in enumerate(ood_pos):
        d = ood_digits[k]
        # (a-OOD) within OOD-D: USPS -> held-out USPS same digit
        x_sw = x_ood.clone(); x_sw[p] = usps_held[k]
        r_ood = _swap_sensitivity(ens_ood, frozen, b0, x_sw - ds_mean, y, lr, T, act, rank,
                                  out_f, device, seed_tag=20000 + p)
        # (a-MNIST matched) within baseline-D: MNIST -> held-out MNIST same digit
        x_sw = x_base.clone(); x_sw[p] = ctrl[p]
        r_mn = _swap_sensitivity(ens_base, frozen, b0, x_sw - ds_mean, y, lr, T, act, rank,
                                 out_f, device, seed_tag=30000 + p)
        # (b cross / style-only) within OOD-D: USPS -> the ORIGINAL MNIST member (same digit, same slot)
        x_sw = x_ood.clone(); x_sw[p] = x_base[p]
        r_st = _swap_sensitivity(ens_ood, frozen, b0, x_sw - ds_mean, y, lr, T, act, rank,
                                 out_f, device, seed_tag=40000 + p)
        amp = (r_ood["sensitivity"] / r_mn["sensitivity"]
               if (math.isfinite(r_ood["sensitivity"]) and math.isfinite(r_mn["sensitivity"])
                   and abs(r_mn["sensitivity"]) > 1e-12) else float("nan"))
        per_member.append(dict(
            pos=p, digit=d, usps_idx=usps_idx[k],
            g0_usps=g0_ood[p].item(), m0_usps=m0_ood[p].item(), xnorm_usps=xn_ood[p].item(),
            g0_mnist=g0_base[p].item(), m0_mnist=m0_base[p].item(), xnorm_mnist=xn_base[p].item(),
            sens_ood=r_ood["sensitivity"], p_ood=r_ood["pvalue"], qeff_ood=r_ood["qeff"],
            coh_ood=r_ood["coherent_signal"],
            sens_mnist=r_mn["sensitivity"], p_mnist=r_mn["pvalue"], qeff_mnist=r_mn["qeff"],
            coh_mnist=r_mn["coherent_signal"],
            sens_style=r_st["sensitivity"], p_style=r_st["pvalue"], qeff_style=r_st["qeff"],
            coh_style=r_st["coherent_signal"],
            amplification=amp,
            dropped=r_ood["dropped"] + r_mn["dropped"] + r_st["dropped"]))
        print(f"pos{p:02d} d={d}: sens USPS-in-OOD={r_ood['sensitivity']:.3f} (p={r_ood['pvalue']:.3f})  "
              f"matched MNIST={r_mn['sensitivity']:.3f} (p={r_mn['pvalue']:.3f})  "
              f"style-only={r_st['sensitivity']:.3f} (p={r_st['pvalue']:.3f})  amp={amp:.3f}", flush=True)

    # ---- 5) verdicts ----
    amp_mean = fin_mean([r["amplification"] for r in per_member])
    meas_more = math.isfinite(amp_mean) and amp_mean > 1.0
    # pooled Spearman(g0, sens) over ALL measured members (USPS + matched MNIST) — SMALL-n
    g0_pool = [r["g0_usps"] for r in per_member] + [r["g0_mnist"] for r in per_member]
    sens_pool = [r["sens_ood"] for r in per_member] + [r["sens_mnist"] for r in per_member]
    rho, n_rho = spearman(g0_pool, sens_pool)
    predicted = (pred_more == meas_more) if math.isfinite(amp_mean) else None
    verdict = ("NO-DATA (amplification NaN)" if predicted is None else
               f"g0 direction {'PREDICTS' if predicted else 'FAILS to predict'} the measured "
               f"amplification (g0 says {'MORE' if pred_more else 'LESS'}, measured "
               f"{'MORE' if meas_more else 'LESS/equal'}); "
               f"pooled Spearman(g0,sens)={rho:+.3f} (n={n_rho}, SMALL-n — directional only)")

    print("\n=== SUMMARY TABLE (member | source | g0 | m0 | sensitivity | p) ===")
    for r in per_member:
        print(f"  pos{r['pos']:02d} d={r['digit']} | USPS  | g0={r['g0_usps']:.4e} | m0={r['m0_usps']:+.4f} | "
              f"sens={r['sens_ood']:.3f} | p={r['p_ood']:.3f}")
        print(f"  pos{r['pos']:02d} d={r['digit']} | MNIST | g0={r['g0_mnist']:.4e} | m0={r['m0_mnist']:+.4f} | "
              f"sens={r['sens_mnist']:.3f} | p={r['p_mnist']:.3f}")
        print(f"  pos{r['pos']:02d} d={r['digit']} | style-only swap (USPS->orig MNIST, within OOD-D): "
              f"sens={r['sens_style']:.3f} | p={r['p_style']:.3f}")
    print(f"\nOOD AMPLIFICATION ratio (sens_ood/sens_mnist): per-member "
          f"{['%.3f' % r['amplification'] for r in per_member]}  mean={amp_mean:.3f}")
    print(f"STYLE swap mean sens={fin_mean([r['sens_style'] for r in per_member]):.3f} "
          f"(vs within-source USPS swap {fin_mean([r['sens_ood'] for r in per_member]):.3f}: "
          f"style-only >= within-source => the STYLE itself is what the adapter notices)")
    print(f"PREDICTOR VERDICT: {verdict}", flush=True)

    # ---- 6) save JSON + .pth (visual examples REQUIRED: the actual images used) ----
    res = dict(
        N=N, K=K, n_ood=n_ood, lr=lr, T=T, rank=rank, stage0=stage0,
        ood_pos=ood_pos, ood_digits=ood_digits, usps_indices=usps_idx,
        usps_preproc="16x16 test split -> 28x28 bilinear (align_corners=False), clamp [0,1], "
                     "float64; centered with the FROZEN MNIST ds_mean like every member",
        mbce_base=mbce_base, mbce_ood=mbce_ood,
        memorized_base=bool(mbce_base is not None and mbce_base < 1e-3),
        memorized_ood=bool(mbce_ood is not None and mbce_ood < 1e-3),
        reseed_noise_base=noise_base, reseed_noise_ood=noise_ood,
        g0_usps_mean=g0_usps_mean, g0_mnist_matched_mean=g0_mnist_mean, g0_ratio=g0_ratio,
        predicted_more_leakage=bool(pred_more),
        per_member=per_member,
        amplification_mean=amp_mean, measured_more_leakage=bool(meas_more),
        sens_style_mean=fin_mean([r["sens_style"] for r in per_member]),
        spearman_g0_sens_pooled=rho, spearman_n=n_rho,
        predictor_verdict=verdict,
        confounds="USPS global stats differ (stroke thickness/contrast/scale; 16->28 bilinear "
                  "smoothing lowers HF energy); ds_mean is MNIST's frozen mean applied to USPS "
                  "members too (mandatory, but mixes mis-centering into the amplification); "
                  "||x0|| per member saved so the norm/contrast confound is checkable; "
                  "pooled Spearman is n=%d (SMALL-n)." % n_rho)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"ood_summary{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    torch.save(dict(
        metrics=res,
        x_base=x_base.cpu(), x_ood=x_ood.cpu(), y=y.cpu(), digits=[int(d) for d in digits],
        ood_pos=ood_pos,
        usps_injected=torch.stack(usps_inj).cpu(), usps_heldout=torch.stack(usps_held).cpu(),
        mnist_ctrl_at_ood_pos=torch.stack([ctrl[p] for p in ood_pos]).cpu(),
        ds_mean=ds_mean.cpu(),
        g0_base=g0_base, m0_base=m0_base, g0_ood=g0_ood, m0_ood=m0_ood,
        xnorm_base=xn_base, xnorm_ood=xn_ood,
    ), os.path.join(RESULTS, f"ood_injection{tag}.pth"))
    print(f"saved {RESULTS}/ood_summary{tag}.json + ood_injection{tag}.pth "
          f"(incl. the actual USPS + MNIST images used)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_ood", type=int, default=2, help="# class-1 slots replaced by USPS digits")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--stage0", action="store_true",
                    help="tiny sanity: N=12, K=12, 1 OOD member; assert USPS loads + finite metric")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.stage0:
        N, K, n_ood = 12, 12, 1
        print(f"=== STAGE-0 SANITY (N={N}, K={K}, n_ood={n_ood}, rank {args.rank}) ===")
    else:
        N, K, n_ood = args.N, args.K, args.n_ood
    assert n_ood >= 1 and n_ood <= N // 2, f"n_ood={n_ood} must be in [1, N/2={N // 2}] (class-1 slots)"

    res = run(N, K, n_ood, args.lr, args.T, args.rank, args.device, stage0=args.stage0)

    if args.stage0:
        assert math.isfinite(res["g0_ratio"]) and res["g0_ratio"] > 0, "g0 predictor degenerate"
        assert math.isfinite(res["reseed_noise_ood"]) and res["reseed_noise_ood"] > 0, \
            "OOD-D reseed_noise degenerate"
        for r in res["per_member"]:
            assert math.isfinite(r["sens_ood"]), "sens_ood NaN (metric integration broken)"
            assert math.isfinite(r["sens_mnist"]), "sens_mnist NaN (metric integration broken)"
            assert math.isfinite(r["sens_style"]), "sens_style NaN (metric integration broken)"
        print("STAGE-0 OK")


if __name__ == "__main__":
    main()
