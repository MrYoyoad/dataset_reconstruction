#!/usr/bin/env python
"""
Arm D — CONTEXT RARITY (image-controlled rarity): does the SAME fixed image leak more per-example
when it is the LONE minority (rare context) than when it has many same-class peers (typical context)?

Program: notes/dataset_sensitivity_program_plan.md ("BATTERY TRACKER", arm D). Runs on the FIXED
3-way whitened metric (whitened_metric.py). Direct successor to arm C (arm_c_imbalance.py): arm C
found minority examples leak more, BUT a balanced-point control exposed a ~3.3x intrinsic CLASS-
IDENTITY asymmetry AND the rarity effect was entangled with WHICH image was swapped (a DIFFERENT
minority image at each m). Arm D removes BOTH confounds by holding the target image FIXED.

THE ONE CHANGE FROM ARM C: in arm C, at each m the code swaps a DIFFERENT minority image. In arm D
we hold ONE fixed target image T (class 1, from the MNIST-TEST pool) and its ONE fixed held-out
same-class control T', and vary only how many same-class PEERS T has.

  For each m in {1,2,4,8} (m = TOTAL class-1 count INCLUDING T) at N=16:
    private set D = (N-m) class-0 images  +  T  +  (m-1) OTHER fixed class-1 peer images.
    (m=1 -> T is the LONE class-1 = rare context; m=8 -> T has 7 peers = typical/balanced context.)
    The class-0 majority and the (m-1) peers are drawn from FIXED pools (same across m), NESTED so
    growing m ADDS peers rather than reshuffling:
        class-1(m=1)={T} ⊂ class-1(m=2)={T,p0} ⊂ class-1(m=4)={T,p0,p1,p2} ⊂ class-1(m=8)={T,p0..p6}
        class-0(m) = maj_pool[:N-m]  (nested-shrinking subset as N is fixed and m grows).
    ONLY T is the swap target; swap T -> T' (fixed) and measure paired-per-seed whitened sensitivity
    vs the un-swapped baseline ensemble ΔW(D, seed_j), EXACTLY as arm C's _measure_class.
  Averaged over n_targets DISTINCT choices of the fixed target T (a few T's, each run through the full
  m-sweep with its own fixed T'), so the result isn't a single-image fluke — but WITHIN a given T the
  swapped image is IDENTICAL across all m (the controlled comparison).

OUTPUT per m (paired per seed, init CANCELS, ΔW=B·A only, gauge-invariant):
  sens (the fixed-T swap sensitivity, averaged over the n_targets T's), p-value, coherent_signal,
  reseed_noise (Σ-scale diagnostic), memorized. Plus, per T: sens(m=1)/sens(m=8) = the RARITY GAIN.

READ — does sens(m) FALL as m grows (T less rare)? A monotone decrease = a CLEAN context-rarity
effect, free of the image-identity / class-identity confound that muddied arm C. Flat sens(m) ⇒ the
per-example imprint does NOT depend on how many same-class peers T has (arm C's effect was the
image/class confound, not context rarity).

CONFOUND to weigh (see report): with T and N fixed, growing m adds peers BUT also shifts class
BALANCE (m=1: 15 class-0 vs 1 class-1; m=8: 8 vs 8) and shrinks the class-0 majority. Unlike arm C the
swapped image is now IDENTICAL across m, so image/class identity is removed; what co-varies with peer
count is class balance — which IS the intended "context" variable (T's rarity in its own class).

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
RESULTS = "/home/projects/galvardi/yoado/results/arm_d_context"
N_FOLDS = 5


def _train_dW(frozen, b0, seed, x0, y, lr, T, act, rank, out_f, device):
    """One LoRA fine-tune at init seed `seed`; returns (ΔW=B·A, max_bce)."""
    _, _, mbce, dW = train_adapter(frozen, b0, draw_B0(seed, out_f, rank, device), x0, y, lr, T, act, rank)
    return dW, mbce


def build_base(N, lr, T, device, pool_n=None):
    """FIXED base θ0 (pretrained checkpoint; NOT fit to the private data) + ds_mean + labelled pool.
    Rank-independent, built ONCE and shared across every m and every target T (arm-C/E convention).
    `pool_n` = images requested per binary class from the MNIST-TEST pool (held out from base training);
    must cover BOTH the class-0 majority (up to N-1) AND the class-1 targets+peers (n_targets+max_m-1)."""
    n_per_class = N // 2
    x_ref, y_ref, _ = build_set(n_per_class, seed=42, device=device)   # only supplies ds_mean / bias
    _, frozen, b0, _, ds_mean = _honest_target(x_ref, y_ref, T, 8, "gelu", lr, device, "mnist", num_classes=2)
    pn = pool_n or max(N, 18)
    x_pool, y_pool, digit_pool, _ = get_finetuning_data(pn, seed=42, device=device, dataset="mnist")
    return frozen, b0, ds_mean, x_pool.to(torch.float64), y_pool.to(torch.float64), list(digit_pool)


def _split_pools(pool_y, N, max_m, n_targets, m_min):
    """Carve FIXED, DISJOINT class pools from the labelled pool (indices only), shared across ALL m:
      target_pool = first n_targets class-1 images (the distinct fixed T's),
      peer_pool   = next (max_m-1) class-1 images (the nested same-class peers; NEVER overlap a T),
      maj_pool    = first (N - m_min) class-0 images (the nested-shrinking majority; N-min(m) at its max).
    Guarantees distinct T's not overlapping the peer pool, and enough of each class."""
    yl = pool_y.tolist()
    c0 = [i for i, v in enumerate(yl) if int(v) == 0]
    c1 = [i for i, v in enumerate(yl) if int(v) == 1]
    need_c0 = N - m_min
    need_c1 = n_targets + (max_m - 1)
    assert len(c1) >= need_c1, f"need >= n_targets+max_m-1={need_c1} class-1 pool imgs, have {len(c1)}"
    assert len(c0) >= need_c0, f"need >= N-min(m)={need_c0} class-0 pool imgs, have {len(c0)}"
    target_pool = c1[:n_targets]
    peer_pool = c1[n_targets:n_targets + (max_m - 1)]
    maj_pool = c0[:need_c0]
    # sanity: targets and peers disjoint (they are, by construction — assert to catch pool drift)
    assert not (set(target_pool) & set(peer_pool)), "target/peer pools overlap (image-identity leak)"
    return target_pool, peer_pool, maj_pool


def _build_context(pool_x, pool_digits, maj_pool, peer_pool, target_idx, N, m, device):
    """Build the private set D for a FIXED target T (=pool index `target_idx`) at peer-count m.
    D order: [ (N-m) class-0 majority ] + [ T ] + [ (m-1) nested class-1 peers ]. The class-1 set is
    {T} ∪ peer_pool[:m-1] — NESTED across m (m grows -> peers ADDED). class-0 = maj_pool[:N-m] (nested-
    shrinking). Returns (x_D [N], y_D [N], digits_D, t_pos) with t_pos = position of T in D."""
    assert 1 <= m <= N - 1, f"m={m} must leave >=1 image in BOTH classes at N={N}"
    n_maj = N - m
    assert len(maj_pool) >= n_maj, f"need {n_maj} class-0, have {len(maj_pool)}"
    assert len(peer_pool) >= m - 1, f"need {m-1} peers, have {len(peer_pool)}"
    maj = maj_pool[:n_maj]
    peers = peer_pool[:m - 1]
    idx = maj + [target_idx] + peers                                   # T at position n_maj
    x_D = pool_x[idx].clone()
    digits_D = [pool_digits[i] for i in idx]
    y_D = torch.cat([torch.zeros(n_maj, dtype=torch.float64, device=device),
                     torch.ones(1 + len(peers), dtype=torch.float64, device=device)], 0)
    t_pos = n_maj
    return x_D, y_D, digits_D, t_pos


def _measure_target(target_idx, m, N, K, lr, T, rank, device, frozen, b0, ds_mean,
                    pool_x, pool_digits, maj_pool, peer_pool, act, out_f, subk, seed_tag):
    """Fixed-T context-rarity measurement for ONE target T at peer-count m. Builds the un-swapped
    baseline ensemble ΔW(D, seed_j), swaps T -> its FIXED held-out same-class control T', and returns
    the paired-per-seed whitened sensitivity vs that ensemble. T' depends ONLY on T's digit (seed 123),
    so it is IDENTICAL across all m for this T — the controlled swap."""
    seeds = [1000 + j for j in range(K)]
    x_D, y_D, digits_D, t_pos = _build_context(pool_x, pool_digits, maj_pool, peer_pool,
                                               target_idx, N, m, device)
    x0_D = x_D - ds_mean                                               # ds_mean FROZEN

    # fixed held-out same-class control T' for T's digit (identical across m for this T)
    d_T = int(pool_digits[target_idx])
    ctrl, _, _ = get_control_images_in_distribution([d_T], seed=123, dataset="mnist")
    T_prime = ctrl.to(torch.float64).to(device)[0]

    # un-swapped baseline ensemble ΔW(D, seed_j) — the reseed_list / Σ for this (T, m)
    dW_base, mbce_ref = {}, None
    for s in seeds:
        dW, mbce = _train_dW(frozen, b0, s, x0_D, y_D, lr, T, act, rank, out_f, device)
        if torch.isfinite(dW).all():
            dW_base[s] = dW
            if s == seeds[0]:
                mbce_ref = mbce
    base_dropped = K - len(dW_base)
    assert len(dW_base) >= 2 * N_FOLDS, \
        f"m={m} T={target_idx}: only {len(dW_base)} finite baseline draws (< 2*N_FOLDS={2*N_FOLDS}); metric starved"
    base_stack = torch.stack(list(dW_base.values()))
    reseed_noise = ((base_stack - base_stack.mean(0)).flatten(1).norm(dim=1) ** 2).mean().sqrt().item()

    # swap ONLY T -> T' (like-for-like same-class); ds_mean FROZEN
    x_sw = x_D.clone(); x_sw[t_pos] = T_prime
    x0_sw = x_sw - ds_mean
    # known-init subspace diagnostic (fixed ref seed) vs the un-swapped baseline
    dW_sw0, _ = _train_dW(frozen, b0, seeds[0], x0_sw, y_D, lr, T, act, rank, out_f, device)
    base0 = dW_base.get(seeds[0])
    subcos = subspace_cos(base0, dW_sw0, subk) if (base0 is not None and torch.isfinite(dW_sw0).all()) else float("nan")

    # unknown-init: paired per-seed diff v_j = ΔW(D,seed_j) - ΔW(D_swap,seed_j) (init CANCELS per pair)
    vs, vs_reseed, dropped = [], [], 0
    for s in seeds:
        if s not in dW_base:
            continue
        dW_sw_s, _ = _train_dW(frozen, b0, s, x0_sw, y_D, lr, T, act, rank, out_f, device)
        v = dW_base[s] - dW_sw_s
        if torch.isfinite(v).all():
            vs.append(v); vs_reseed.append(dW_base[s])                 # keep v_j aligned with its reseed
        else:
            dropped += 1
    coherent = torch.stack(vs).mean(0).norm().item() if vs else float("nan")

    sens = pval = d2obs = qeff = float("nan")
    n_metric = 0
    if len(vs) >= 2 * N_FOLDS:
        ws = whitened_sensitivity([v.cpu() for v in vs], [r.cpu() for r in vs_reseed],
                                  n_folds=N_FOLDS, p_max=3, n_perm=500, seed=int(seed_tag))
        sens, pval, d2obs, qeff = ws["sensitivity"], ws["pvalue"], ws["d2_obs"], ws["qeff_count"]
        assert math.isfinite(d2obs), f"m={m} T={target_idx}: d2_obs non-finite (metric broken)"
    return dict(sensitivity=sens, pvalue=pval, d2_obs=d2obs, qeff=qeff,
                coherent_signal=coherent, reseed_noise=reseed_noise, swap_subspace_cos=subcos,
                ref_max_bce=mbce_ref, memorized=bool(mbce_ref is not None and mbce_ref < 1e-3),
                digit=d_T, t_pos=t_pos, base_dropped=base_dropped, dropped=dropped,
                n_metric=(1 if math.isfinite(sens) else 0), dW_base_mean=base_stack.mean(0).cpu())


def run_for_m(m, N, K, n_targets, lr, T, rank, device, frozen, b0, ds_mean,
              pool_x, pool_digits, target_pool, peer_pool, maj_pool, tag=""):
    """All fixed-T context-rarity measurements at peer-count m, averaged over the n_targets DISTINCT
    fixed T's. target_pool/peer_pool/maj_pool are FIXED across m (built once) so ONLY T's rarity (its
    class-1 peer count / class balance) changes; the swapped image is identical across m per target."""
    act = make_activation("gelu")
    out_f = frozen[0].shape[0]
    subk = min(rank, 8)
    mean = lambda a: (sum(a) / len(a)) if a else float("nan")

    per_target = []
    for tgt_id, target_idx in enumerate(target_pool):
        r = _measure_target(target_idx, m, N, K, lr, T, rank, device, frozen, b0, ds_mean,
                            pool_x, pool_digits, maj_pool, peer_pool, act, out_f, subk,
                            seed_tag=10000 + 1000 * tgt_id + m)
        per_target.append(r)

    fin = [r for r in per_target if math.isfinite(r["sensitivity"])]
    per_sens = [r["sensitivity"] for r in per_target]                  # aligned with target_pool order (per T)
    res = dict(
        m=m, N=N, minority_fraction=m / N, rank=rank, lr=lr, T=T, K=K, n_targets=n_targets,
        sens=mean([r["sensitivity"] for r in fin]),
        pvalue=mean([r["pvalue"] for r in fin]),
        d2_obs=mean([r["d2_obs"] for r in fin]),
        qeff=mean([r["qeff"] for r in fin]),
        coherent_signal=mean([r["coherent_signal"] for r in per_target if math.isfinite(r["coherent_signal"])]),
        reseed_noise=mean([r["reseed_noise"] for r in per_target]),
        swap_subspace_cos=mean([r["swap_subspace_cos"] for r in per_target if math.isfinite(r["swap_subspace_cos"])]),
        memorized=bool(per_target and all(r["memorized"] for r in per_target)),
        ref_max_bce=mean([r["ref_max_bce"] for r in per_target if r["ref_max_bce"] is not None]),
        per_target_sens=per_sens,
        per_target_pvalue=[r["pvalue"] for r in per_target],
        per_target_digit=[r["digit"] for r in per_target],
        base_dropped=sum(r["base_dropped"] for r in per_target),
        swap_dropped=sum(r["dropped"] for r in per_target),
        n_targets_finite=len(fin),
    )
    os.makedirs(RESULTS, exist_ok=True)
    torch.save(dict(metrics=res,
                    dW_base_mean=torch.stack([r["dW_base_mean"] for r in per_target]),  # [n_targets, out, in]
                    per_target_digit=[r["digit"] for r in per_target],
                    m=m, N=N, rank=rank), os.path.join(RESULTS, f"armd_m{m}_N{N}{tag}.pth"))
    return res, per_sens


def _rarity_gains(m_list, per_sens_by_m):
    """Per-target rarity gain sens(m=min)/sens(m=max) — the LONE-minority vs typical-context ratio for
    the SAME fixed T. Reported only if both endpoints are present and finite (else nan)."""
    m_lo, m_hi = min(m_list), max(m_list)
    if m_lo == m_hi or m_lo not in per_sens_by_m or m_hi not in per_sens_by_m:
        return [], float("nan")
    lo, hi = per_sens_by_m[m_lo], per_sens_by_m[m_hi]
    gains = []
    for a, b in zip(lo, hi):
        gains.append((a / b) if (math.isfinite(a) and math.isfinite(b) and abs(b) > 1e-12) else float("nan"))
    fin = [g for g in gains if math.isfinite(g)]
    return gains, (sum(fin) / len(fin) if fin else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m_list", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="peer-counts m = TOTAL class-1 count INCLUDING the fixed target T")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_targets", type=int, default=3, help="# DISTINCT fixed targets T (each full m-sweep)")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--stage0", action="store_true",
                    help="tiny sanity: N=12, m∈{1,4}, K=12, 1 target T, rank 8")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    if args.stage0:
        N, m_list, K, n_targets, rank = 12, [1, 4], 12, 1, 8
        print(f"=== STAGE-0 SANITY (N={N}, m∈{m_list}, K={K}, {n_targets} target, rank {rank}) ===")
    else:
        N, m_list, K, n_targets, rank = args.N, args.m_list, args.K, args.n_targets, args.rank

    m_list = [m for m in m_list if 1 <= m <= N - 1]
    assert m_list, "no valid m (need 1<=m<=N-1, else a class is empty)"
    max_m, m_min = max(m_list), min(m_list)
    need_c1 = n_targets + (max_m - 1)
    pool_n = max(N - m_min, need_c1, 18)

    frozen, b0, ds_mean, px, py, pd = build_base(N, args.lr, args.T, dev, pool_n=pool_n)
    target_pool, peer_pool, maj_pool = _split_pools(py, N, max_m, n_targets, m_min)
    print(f"pools: {len(target_pool)} fixed targets, {len(peer_pool)} peers, {len(maj_pool)} majority "
          f"(target digits={[int(pd[i]) for i in target_pool]})", flush=True)

    all_res, per_sens_by_m = [], {}
    for m in m_list:
        print(f"\n===== m={m} (T + {m-1} peers; class balance {N-m}:{m}) =====", flush=True)
        r, per_sens = run_for_m(m, N, K, n_targets, args.lr, args.T, rank, dev, frozen, b0, ds_mean,
                                px, pd, target_pool, peer_pool, maj_pool,
                                tag=("_stage0" if args.stage0 else ""))
        all_res.append(r)
        per_sens_by_m[m] = per_sens
        if not r["memorized"]:
            print(f"WARNING: m={m} baseline NOT memorized (mean max_bce={r['ref_max_bce']:.2e}) — "
                  f"off-convergence sensitivity is confounded; interpret with care.", flush=True)
        print(json.dumps(r), flush=True)

    assert any(math.isfinite(r["sens"]) for r in all_res), \
        "ALL m have NaN sensitivity — metric starved (too many dropped draws). Aborting."

    gains, mean_gain = _rarity_gains(m_list, per_sens_by_m)

    if args.stage0:
        for r in all_res:
            assert math.isfinite(r["sens"]), "sens NaN (metric integration broken)"
            assert math.isfinite(r["reseed_noise"]) and r["reseed_noise"] > 0, "reseed_noise degenerate"
        print("STAGE-0 OK")
        return

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "arm_d_summary.json"), "w") as f:
        json.dump(dict(N=N, K=K, T=args.T, lr=args.lr, rank=rank, n_targets=n_targets,
                       m_list=m_list, results=all_res,
                       per_target_rarity_gain=gains, mean_rarity_gain=mean_gain), f, indent=2)

    print("\n=== SUMMARY (m | balance | sens | p | mem) ===")
    for r in all_res:
        print(f"m={r['m']:>2} bal={N-r['m']}:{r['m']:<2}: sens={r['sens']:.3f}  p={r['pvalue']:.3f}  "
              f"coh={r['coherent_signal']:.3f}  reseed={r['reseed_noise']:.3f}  mem={r['memorized']}", flush=True)
    print(f"\nRARITY GAIN sens(m={m_min})/sens(m={max_m}) per target: "
          f"{['%.3f' % g for g in gains]}  mean={mean_gain:.3f}")
    print("READ: sens(m) FALLING as m grows (T less rare) ⇒ CLEAN context-rarity effect, free of the "
          "image/class-identity confound that muddied arm C. Flat sens(m) ⇒ no context-rarity effect "
          "(arm C's gap was the image/class confound). Gain>1 ⇒ lone-minority T leaks more than typical-context T.")


if __name__ == "__main__":
    main()
