#!/usr/bin/env python
"""
Arm C — CLASS IMBALANCE: does a MINORITY-class example leave a LARGER per-example imprint
on a low-rank adapter than a MAJORITY-class example, and does that gap widen as the minority
class gets rarer? (Feldman: rare/atypical examples are memorized harder.)

Program: notes/dataset_sensitivity_program_plan.md ("BATTERY TRACKER", arm C). Runs on the
FIXED 3-way whitened metric (whitened_metric.py, 2026-08-27). Mirrors arm B (single-image
swap sensitivity) and arm E (fixed pretrained base + build_base / run_for_* / --stage0).

THE CONTRAST (fixed N, fixed set D, fixed seeds — the ONLY thing that moves is which class the
swapped image belongs to):
  Build an imbalanced binary private set of size N: (N-m) MAJORITY (class 0) + m MINORITY (class 1),
  for a sweep of m (default {1,2,4,8} at N=16 -> minority fraction 1/16 … 1/2). Fixed pretrained θ0
  (MNIST-train checkpoint), targets from the MNIST-TEST pool (genuinely held out), ds_mean FROZEN.

  PER-EXAMPLE sensitivity via a single-image LIKE-FOR-LIKE swap (arm-B style):
    minority swap → pick a class-1 image i in D, replace it with a HELD-OUT class-1 control
                    (get_control_images_in_distribution → same digit, different test instance);
                    paired per seed v_j = ΔW(D,seed_j) − ΔW(D_swap,seed_j) (init CANCELS);
                    reseed_list = ΔW(D,seed_j) (the un-swapped baseline ensemble). → sens_minority.
    majority swap → identical, but the swapped image is class 0. → sens_majority.
  Averaged over n_targets_per_class example choices per class. Both swaps replace an image with a
  HELD-OUT SAME-CLASS control (a per-example swap, NEVER a class change) inside the IDENTICAL D.

MEASURE per m, paired per seed (init CANCELS), ΔW=B·A only (gauge-invariant):
  sens_minority, sens_majority, ratio = sens_minority / sens_majority, their p-values, and the
  un-swapped baseline reseed_noise (Σ-scale diagnostic).

READ — does ratio > 1 and GROW as m shrinks? ratio(m) increasing as m↓ ⇒ the rarer class's
example is the more identifiable (memorization of rare examples). ratio ≈ 1 flat ⇒ no per-example
imbalance effect on adapter leakage.

CONFOUNDS the parent should weigh (see report): (1) at fixed m the minority/majority comparison is
per-example fair (same D, N, seeds), but the minority class has FEWER same-class peers to share the
label gradient, so "minority" is entangled with "less internally redundant" — which is the Feldman
mechanism itself, not a nuisance to remove; (2) shrinking m conflates "class rarer in the loss" with
"the swapped image is a larger fraction of its own class" (at m=1 the swap replaces the SOLE class-1
representative). A clean disentangler (vary N at fixed m, or fix minority size and grow majority) is
left to a follow-up arm.

bsub-only. mnist / gelu / binary. float64.
"""
import os, json, math, argparse
import torch

from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.data_utils import get_finetuning_data, get_control_images_in_distribution
from experiments.dataset_sensitivity.arm_b_dilution import (
    draw_B0, train_adapter, subspace_cos, build_set,
)
from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/arm_c_imbalance"
N_FOLDS = 5


def _train_dW(frozen, b0, seed, x0, y, lr, T, act, rank, out_f, device):
    """One LoRA fine-tune at init seed `seed`; returns (ΔW=B·A, max_bce)."""
    _, _, mbce, dW = train_adapter(frozen, b0, draw_B0(seed, out_f, rank, device), x0, y, lr, T, act, rank)
    return dW, mbce


def build_base(N, lr, T, device):
    """FIXED base θ0 (pretrained checkpoint; NOT fit to the private data) + ds_mean + labelled pool.
    Rank-independent, built ONCE and shared across every m and BOTH classes (arm-E convention)."""
    n_per_class = N // 2
    x_ref, y_ref, _ = build_set(n_per_class, seed=42, device=device)   # only supplies ds_mean / bias
    _, frozen, b0, _, ds_mean = _honest_target(x_ref, y_ref, T, 8, "gelu", lr, device, "mnist", num_classes=2)
    # pool: max class-0 needed = N - min(m) <= N-1; max class-1 needed = max(m) <= N-1; headroom via max(N,18)
    x_pool, y_pool, digit_pool, _ = get_finetuning_data(max(N, 18), seed=42, device=device, dataset="mnist")
    return frozen, b0, ds_mean, x_pool.to(torch.float64), y_pool.to(torch.float64), list(digit_pool)


def _build_imbalanced(pool_x, pool_y, pool_digits, N, m, device):
    """Imbalanced private set: (N-m) MAJORITY (class 0) first, then m MINORITY (class 1).
    Returns (x_D [N], y_D [N], digits_D, majority_positions, minority_positions). Indices into
    x_D are position-aligned with digits_D and with the control tensor built from digits_D."""
    assert 1 <= m <= N - 1, f"m={m} must leave >=1 image in BOTH classes at N={N}"
    n_maj = N - m
    yl = pool_y.tolist()
    c0 = [i for i, v in enumerate(yl) if v == 0.0]     # class-0 (majority) pool
    c1 = [i for i, v in enumerate(yl) if v == 1.0]     # class-1 (minority) pool
    assert len(c0) >= n_maj, f"need >= N-m={n_maj} class-0 pool imgs, have {len(c0)}"
    assert len(c1) >= m, f"need >= m={m} class-1 pool imgs, have {len(c1)}"
    idx = c0[:n_maj] + c1[:m]
    x_D = pool_x[idx].clone()
    digits_D = [pool_digits[i] for i in idx]
    y_D = torch.cat([torch.zeros(n_maj, dtype=torch.float64, device=device),
                     torch.ones(m, dtype=torch.float64, device=device)], 0)
    maj_pos = list(range(0, n_maj))
    min_pos = list(range(n_maj, N))
    return x_D, y_D, digits_D, maj_pos, min_pos


def _measure_class(target_positions, x_D, y_D, controls, dW_base, seeds,
                   frozen, b0, ds_mean, lr, T, act, rank, out_f, device, subk, seed_tag):
    """Per-example whitened sensitivity, averaged over `target_positions` (all one class).
    Single-image swap i -> held-out same-class control; paired per seed vs the un-swapped ensemble.
    dW_base: dict seed -> ΔW(D, seed) (baseline ensemble, SHARED across both classes at this m)."""
    mean = lambda a: (sum(a) / len(a)) if a else float("nan")
    w_sens, w_pval, w_d2obs, w_qeff, subs, coh = [], [], [], [], [], []
    dropped = 0
    for i in target_positions:
        x_sw = x_D.clone(); x_sw[i] = controls[i]                       # like-for-like same-class swap
        x0_sw = x_sw - ds_mean                                          # ds_mean FROZEN
        # known-init subspace diagnostic (fixed ref seed = seeds[0]) vs the un-swapped baseline
        dW_sw0, _ = _train_dW(frozen, b0, seeds[0], x0_sw, y_D, lr, T, act, rank, out_f, device)
        base0 = dW_base.get(seeds[0])
        if base0 is not None and torch.isfinite(dW_sw0).all():
            subs.append(subspace_cos(base0, dW_sw0, subk))
        # unknown-init: paired per-seed diff vs the frozen baseline ensemble
        vs, vs_reseed = [], []
        for s in seeds:
            if s not in dW_base:
                continue
            dW_sw_s, _ = _train_dW(frozen, b0, s, x0_sw, y_D, lr, T, act, rank, out_f, device)
            v = dW_base[s] - dW_sw_s
            if torch.isfinite(v).all():
                vs.append(v); vs_reseed.append(dW_base[s])             # keep v_j aligned with its reseed
            else:
                dropped += 1
        if vs:
            coh.append(torch.stack(vs).mean(0).norm().item())          # ‖E_seeds v_j‖ = coherent signal
        if len(vs) >= 2 * N_FOLDS:
            ws = whitened_sensitivity([v.cpu() for v in vs], [r.cpu() for r in vs_reseed],
                                      n_folds=N_FOLDS, p_max=3, n_perm=500, seed=int(seed_tag + i))
            w_sens.append(ws["sensitivity"]); w_pval.append(ws["pvalue"])
            w_d2obs.append(ws["d2_obs"]); w_qeff.append(ws["qeff_count"])
    return dict(sensitivity=mean(w_sens), pvalue=mean(w_pval), d2_obs=mean(w_d2obs),
                qeff=mean(w_qeff), swap_subspace_cos=mean(subs), coherent_signal=mean(coh),
                n_metric=len(w_sens), n_targets=len(target_positions), dropped=dropped)


def run_for_m(m, N, K, n_targets_per_class, lr, T, rank, device,
              frozen, b0, ds_mean, pool_x, pool_y, pool_digits, tag=""):
    """All measurements for one imbalance ratio m. Baseline ensemble (hence Σ) shared across BOTH
    classes at this m by construction: the ONLY thing differing minority vs majority is the class
    of the swapped image."""
    act = make_activation("gelu")
    out_f = frozen[0].shape[0]
    subk = min(rank, 8)
    seeds = [1000 + j for j in range(K)]

    x_D, y_D, digits_D, maj_pos, min_pos = _build_imbalanced(pool_x, pool_y, pool_digits, N, m, device)
    x0_D = x_D - ds_mean

    # held-out same-class controls, position-aligned with digits_D (control[i] shares x_D[i]'s digit)
    controls, _, _ = get_control_images_in_distribution(digits_D, seed=123, dataset="mnist")
    controls = controls.to(torch.float64).to(device)
    assert controls.shape[0] == N, f"controls {controls.shape[0]} != N={N} (aligned swap broken)"

    # un-swapped baseline ensemble ΔW(D, seed_j) — SHARED reseed_list for both classes
    dW_base, mbce_ref = {}, None
    for s in seeds:
        dW, mbce = _train_dW(frozen, b0, s, x0_D, y_D, lr, T, act, rank, out_f, device)
        if torch.isfinite(dW).all():
            dW_base[s] = dW
            if s == seeds[0]:
                mbce_ref = mbce
    base_dropped = K - len(dW_base)
    assert len(dW_base) >= 2 * N_FOLDS, \
        f"m={m}: only {len(dW_base)} finite baseline draws (< 2*N_FOLDS={2*N_FOLDS}); metric starved"
    base_stack = torch.stack(list(dW_base.values()))
    reseed_noise = ((base_stack - base_stack.mean(0)).flatten(1).norm(dim=1) ** 2).mean().sqrt().item()

    # targets: at most n_targets_per_class per class, capped by class size (m minority, N-m majority)
    min_targets = min_pos[:min(n_targets_per_class, m)]
    maj_targets = maj_pos[:min(n_targets_per_class, N - m)]
    assert len(min_targets) >= 1 and len(maj_targets) >= 1, "each class needs >=1 target"

    minr = _measure_class(min_targets, x_D, y_D, controls, dW_base, seeds,
                          frozen, b0, ds_mean, lr, T, act, rank, out_f, device, subk, seed_tag=10000)
    majr = _measure_class(maj_targets, x_D, y_D, controls, dW_base, seeds,
                          frozen, b0, ds_mean, lr, T, act, rank, out_f, device, subk, seed_tag=20000)

    sm, sM = minr["sensitivity"], majr["sensitivity"]
    ratio = (sm / sM) if (sM is not None and math.isfinite(sM) and abs(sM) > 1e-12) else float("nan")
    res = dict(
        m=m, N=N, minority_fraction=m / N, rank=rank, lr=lr, T=T, K=K,
        n_targets_per_class=n_targets_per_class,
        sens_minority=sm, sens_majority=sM, ratio=ratio,
        p_minority=minr["pvalue"], p_majority=majr["pvalue"],
        d2_minority=minr["d2_obs"], d2_majority=majr["d2_obs"],
        qeff_minority=minr["qeff"], qeff_majority=majr["qeff"],
        coh_minority=minr["coherent_signal"], coh_majority=majr["coherent_signal"],
        subcos_minority=minr["swap_subspace_cos"], subcos_majority=majr["swap_subspace_cos"],
        reseed_noise=reseed_noise, ref_max_bce=mbce_ref, memorized=bool(mbce_ref is not None and mbce_ref < 1e-3),
        base_dropped=base_dropped, min_dropped=minr["dropped"], maj_dropped=majr["dropped"],
        n_min_targets=len(min_targets), n_maj_targets=len(maj_targets),
        n_metric_min=minr["n_metric"], n_metric_maj=majr["n_metric"],
    )
    os.makedirs(RESULTS, exist_ok=True)
    torch.save(dict(metrics=res, dW_base_mean=base_stack.mean(0).cpu(), digits=digits_D,
                    m=m, N=N, rank=rank), os.path.join(RESULTS, f"armc_m{m}_N{N}{tag}.pth"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m_list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_targets_per_class", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--stage0", action="store_true",
                    help="tiny sanity: N=12, m∈{2,4}, K=12, 1 target/class, rank 8")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    if args.stage0:
        print("=== STAGE-0 SANITY (N=12, m∈{2,4}, K=12, 1 target/class, rank 8) ===")
        frozen, b0, ds_mean, px, py, pd = build_base(12, args.lr, args.T, dev)
        for m in [2, 4]:
            r = run_for_m(m, 12, 12, 1, args.lr, args.T, 8, dev, frozen, b0, ds_mean, px, py, pd, tag="_stage0")
            print(json.dumps(r, indent=2))
            assert math.isfinite(r["sens_minority"]), "sens_minority NaN (metric integration broken)"
            assert math.isfinite(r["sens_majority"]), "sens_majority NaN (metric integration broken)"
            assert math.isfinite(r["reseed_noise"]) and r["reseed_noise"] > 0, "reseed_noise degenerate"
        print("STAGE-0 OK")
        return

    frozen, b0, ds_mean, px, py, pd = build_base(args.N, args.lr, args.T, dev)
    all_res = []
    for m in args.m_list:
        if not (1 <= m <= args.N - 1):
            print(f"SKIP m={m}: needs 1<=m<=N-1={args.N-1} (a class would be empty).", flush=True)
            continue
        print(f"\n===== m={m} (minority fraction {m/args.N:.3f}) =====", flush=True)
        r = run_for_m(m, args.N, args.K, args.n_targets_per_class, args.lr, args.T, args.rank, dev,
                      frozen, b0, ds_mean, px, py, pd)
        all_res.append(r)
        if not r["memorized"]:
            print(f"WARNING: m={m} baseline NOT memorized (max_bce={r['ref_max_bce']:.2e} > 1e-3) — "
                  f"off-convergence sensitivity is confounded; interpret with care.", flush=True)
        print(json.dumps(r), flush=True)
    assert any(math.isfinite(r["sens_minority"]) and math.isfinite(r["sens_majority"]) for r in all_res), \
        "ALL m have NaN sensitivity — metric starved (too many dropped draws). Aborting."

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "arm_c_summary.json"), "w") as f:
        json.dump(dict(N=args.N, K=args.K, T=args.T, lr=args.lr, rank=args.rank,
                       m_list=args.m_list, results=all_res), f, indent=2)

    print("\n=== SUMMARY (m | frac | sens_min | sens_maj | ratio | p_min | p_maj | mem) ===")
    for r in all_res:
        print(f"m={r['m']:>2} frac={r['minority_fraction']:.3f}: "
              f"sens_min={r['sens_minority']:.3f}  sens_maj={r['sens_majority']:.3f}  "
              f"ratio={r['ratio']:.3f}  p_min={r['p_minority']:.3f}  p_maj={r['p_majority']:.3f}  "
              f"mem={r['memorized']}", flush=True)
    print("\nREAD: ratio>1 AND growing as m↓ ⇒ rarer minority example is more identifiable "
          "(Feldman memorization of rare examples); ratio≈1 flat ⇒ no per-example imbalance effect.")


if __name__ == "__main__":
    main()
