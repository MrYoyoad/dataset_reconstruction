#!/usr/bin/env python
"""
Arm E — DUPLICATION: how does an image's imprint on a low-rank adapter scale with its
copy-count k, and does the low-rank bottleneck SUPPRESS that scaling (a privacy effect)?

Program: notes/dataset_sensitivity_program_plan.md ("BATTERY TRACKER", arm E). Novel,
circularity-free. Runs on the FIXED 3-way whitened metric (whitened_metric.py, 2026-08-27).
Design hardened after an adversarial audit (2026-08-27) — see "AUDIT FIXES" below.

THE CONTRAST (FIXED-PREVALENCE, so the ONLY thing that moves with k is concentration onto T):
  Fix k_max class-1 "slots" + (N-k_max) fixed class-0 context. For copy-count k and target T:
    copies_k   = context + [T]*k + donors[k:k_max]      (k copies of T, then k_max-k shared donors)
    distinct   = context + donors[0:k_max]              (all k_max DISTINCT donors — SAME for every k)
  Both arms always hold k_max class-1 slots (prevalence FIXED across k) and an identical label
  vector; the distinct baseline is IDENTICAL for all k and all targets. So the paired per-seed
  diff v_j = ΔW(copies_k,seed_j) - ΔW(distinct,seed_j) isolates "replace k distinct donors by k
  copies of T" — pure concentration depth. k=1 is the single-swap limit (T vs one donor).

MEASURE per (rank, k), paired per seed (init CANCELS), ΔW=B·A only (gauge-invariant):
  d²(k), sensitivity(k)=d²_obs-d²_null, p(k) = whitened_sensitivity(v_list, reseed=distinct ensemble)

READ — β = log-log slope of the DEBIASED sensitivity(k) vs k, compared ACROSS rank:
  There is NO clean analytic null for β (full-batch nonlinear BCE at convergence + low-rank
  bottleneck break the naive "gradients sum ⇒ β=2" intuition). The honest null is EMPIRICAL:
    β at rank << N  (r=8, bottlenecked)   vs   β at rank >= N  (r=32, full capacity).
  β(low) < β(high)  ⇒ the low-rank bottleneck SATURATES duplication imprint (privacy-PROTECTIVE).
  β(low) ≈ β(high)  ⇒ duplication leaks the same regardless of capacity.
  (β reported on both sensitivity (headline) and d2_obs (diagnostic); d2_obs carries the
   permutation-null offset so its slope is only a cross-check.)

AUDIT FIXES folded in: (1) fixed-prevalence construction kills the minority→balanced confound;
(2) distinct baseline identical across k ⇒ Σ (denominator) frozen across k by construction, and
trained ONCE per rank; (3) headline β on debiased sensitivity, not d2_obs; (4) empirical rank-null
replaces the invalid β=2 null; (5) dropped-draw counters + finite-d² assert before fitting β.

bsub-only. mnist / gelu / binary. float64.
"""
import os, json, math, argparse
import torch

from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.data_utils import get_finetuning_data
from experiments.dataset_sensitivity.arm_b_dilution import (
    draw_B0, train_adapter, subspace_cos, build_set,
)
from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/arm_e_duplication"
N_FOLDS = 5


def _slots(pool, y_pool, N, k_max, n_targets):
    """Partition the pool into disjoint targets / donors / class-0 context. Returns index lists."""
    cls_idx = (y_pool == 1).nonzero(as_tuple=True)[0].tolist()      # class-1 images
    other_idx = (y_pool == 0).nonzero(as_tuple=True)[0].tolist()    # class-0 context pool
    assert len(cls_idx) >= n_targets + k_max, \
        f"need >= n_targets+k_max={n_targets+k_max} class-1 imgs, have {len(cls_idx)}"
    assert len(other_idx) >= N - k_max, \
        f"need >= N-k_max={N-k_max} class-0 context imgs, have {len(other_idx)}"
    targets = cls_idx[:n_targets]
    donors = cls_idx[n_targets:n_targets + k_max]                   # k_max fixed distinct donors (!= targets)
    context = other_idx[:N - k_max]                                 # fixed class-0 context (all class-0)
    return targets, donors, context


def _train_dW(frozen, b0, seed, x0, y, lr, T, act, rank, out_f, device):
    _, _, _, dW = train_adapter(frozen, b0, draw_B0(seed, out_f, rank, device), x0, y, lr, T, act, rank)
    return dW


def run_for_rank(rank, k_list, N, K, n_targets, lr, T, device, frozen, b0, ds_mean, pool, y_pool, tag="", dataset="mnist"):
    """All k for one rank. Distinct baseline (hence Σ) is fixed across k -> trained ONCE here."""
    act = make_activation("gelu")
    out_f = frozen[0].shape[0]
    k_max = max(k_list)
    mean = lambda a: (sum(a) / len(a)) if a else float("nan")
    seeds = [1000 + j for j in range(K)]
    subk = min(rank, 8)
    targets, donors, context = _slots(pool, y_pool, N, k_max, n_targets)

    ctx_x = pool[context]; ctx_y = y_pool[context]
    ones = torch.ones(k_max, dtype=torch.float64, device=device)
    y_vec = torch.cat([ctx_y, ones], 0)                            # IDENTICAL for both arms, all k

    # --- distinct baseline: context + all k_max donors (SAME for every k & target) -> Σ frozen ---
    x_dis = torch.cat([ctx_x, pool[donors]], 0)
    x0_dis = x_dis - ds_mean
    dis_ens = {}                                                   # seed -> ΔW(distinct, seed)
    for s in seeds:
        dW = _train_dW(frozen, b0, s, x0_dis, y_vec, lr, T, act, rank, out_f, device)
        if torch.isfinite(dW).all():
            dis_ens[s] = dW
    dis_dropped = K - len(dis_ens)
    dis_stack = torch.stack(list(dis_ens.values()))
    reseed_noise = ((dis_stack - dis_stack.mean(0)).flatten(1).norm(dim=1) ** 2).mean().sqrt().item()

    results = []
    for k in k_list:
        assert 0 < k <= k_max, f"k={k} out of (0,{k_max}]"
        w_sens, w_pval, w_d2obs, w_d2null, w_qeff, subs, coh = [], [], [], [], [], [], []
        cop_dropped = 0
        for t_idx in targets:
            # copies_k: context + [T]*k + donors[k:k_max]  (k_max class-1 slots; prevalence fixed)
            tgt_x = pool[[t_idx] * k]                               # T repeated k× (dim-agnostic)
            shared = pool[donors[k:k_max]]                          # k_max-k donors shared with distinct
            x_cop = torch.cat([ctx_x, tgt_x, shared], 0)
            x0_cop = x_cop - ds_mean
            # known-init subspace diagnostic (fixed ref seed)
            dW_c0 = _train_dW(frozen, b0, 0, x0_cop, y_vec, lr, T, act, rank, out_f, device)
            dW_d0 = dis_ens.get(seeds[0])
            if dW_d0 is not None:
                subs.append(subspace_cos(dW_c0, dW_d0, subk))
            # unknown-init: paired per seed vs the frozen distinct ensemble
            vs, vs_reseed = [], []
            for s in seeds:
                if s not in dis_ens:
                    continue
                dW_cop = _train_dW(frozen, b0, s, x0_cop, y_vec, lr, T, act, rank, out_f, device)
                v = dW_cop - dis_ens[s]
                if torch.isfinite(v).all():
                    vs.append(v); vs_reseed.append(dis_ens[s])
                else:
                    cop_dropped += 1
            if vs:
                coh.append(torch.stack(vs).mean(0).norm().item())
            if len(vs) >= 2 * N_FOLDS:
                ws = whitened_sensitivity([v.cpu() for v in vs], [r.cpu() for r in vs_reseed],
                                          n_folds=N_FOLDS, p_max=3, n_perm=500, seed=int(t_idx))
                w_sens.append(ws["sensitivity"]); w_pval.append(ws["pvalue"])
                w_d2obs.append(ws["d2_obs"]); w_d2null.append(ws["d2_null_mean"])
                w_qeff.append(ws["qeff_count"])
        res = dict(
            rank=rank, k=k, N=N, lr=lr, T=T, K=K, n_targets=n_targets,
            d2_obs=mean(w_d2obs), d2_null_mean=mean(w_d2null),
            whitened_sensitivity=mean(w_sens), whitened_pvalue=mean(w_pval),
            whitened_qeff=mean(w_qeff), copies_vs_distinct_subspace_cos=mean(subs),
            coherent_signal=mean(coh), reseed_noise=reseed_noise,   # Σ-scale diagnostic (frozen across k)
            dis_dropped=dis_dropped, cop_dropped=cop_dropped, n_metric=len(w_d2obs),
        )
        results.append(res)
        os.makedirs(RESULTS, exist_ok=True)
        ds_tag = "" if dataset == "mnist" else f"_{dataset}"
        torch.save(dict(metrics=res, dW_distinct_mean=dis_stack.mean(0).cpu(),
                        rank=rank, k=k, N=N, dataset=dataset),
                   os.path.join(RESULTS, f"arme_r{rank}_k{k}_N{N}{ds_tag}{tag}.pth"))
    return results


def fit_beta(ks, ys):
    """log-log slope β of y(k) vs k over POSITIVE y; returns (beta, r2, n_used, n_dropped)."""
    pts = [(math.log(k), math.log(y)) for k, y in zip(ks, ys) if y is not None and y > 0]
    n_drop = len(ks) - len(pts)
    if len(pts) < 2:
        return float("nan"), float("nan"), len(pts), n_drop
    n = len(pts); sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return float("nan"), float("nan"), n, n_drop
    beta = (n * sxy - sx * sy) / denom
    a = (sy - beta * sx) / n
    ss_tot = sum((y - sy / n) ** 2 for _, y in pts)
    ss_res = sum((y - (a + beta * x)) ** 2 for x, y in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return beta, r2, n, n_drop


def build_base(N, lr, T, device, dataset="mnist"):
    """FIXED base θ0 (pretrained checkpoint; NOT fit to the private data) + ds_mean + image pool.
    Rank-independent, so built ONCE and shared across the rank sweep."""
    n_per_class = N // 2
    x_ref, y_ref, _ = build_set(n_per_class, seed=42, device=device, dataset=dataset)   # only supplies ds_mean / bias
    _, frozen, b0, _, ds_mean = _honest_target(x_ref, y_ref, T, 8, "gelu", lr, device, dataset, num_classes=2)
    x_pool, y_pool, _, _ = get_finetuning_data(max(N, 18), seed=42, device=device, dataset=dataset)
    return frozen, b0, ds_mean, x_pool.to(torch.float64), y_pool.to(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--rank_list", type=int, nargs="+", default=[8, 32])   # empirical null: bottleneck vs full
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_targets", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--stage0", action="store_true", help="tiny sanity: N=12, k∈{1,2}, rank 8, K=12, 2 targets")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"],
                    help="base dataset (checkpoint models/weights-<ds>_gelu.pth); mnist = byte-identical legacy path")
    args = ap.parse_args()
    dev = args.device
    ds = args.dataset

    if args.stage0:
        print(f"=== STAGE-0 SANITY (N=12, k∈{{1,2}}, rank 8, K=12, 2 targets) [dataset={ds}] ===")
        frozen, b0, ds_mean, pool, y_pool = build_base(12, args.lr, args.T, dev, dataset=ds)
        rs = run_for_rank(8, [1, 2], 12, 12, 2, args.lr, args.T, dev, frozen, b0, ds_mean, pool, y_pool, tag="_stage0", dataset=ds)
        for r in rs:
            print(json.dumps(r, indent=2))
            assert math.isfinite(r["d2_obs"]), "d2_obs NaN (metric integration broken)"
            assert math.isfinite(r["coherent_signal"]) and r["coherent_signal"] > 0, "signal degenerate"
        print("STAGE-0 OK")
        return

    frozen, b0, ds_mean, pool, y_pool = build_base(args.N, args.lr, args.T, dev, dataset=ds)
    by_rank = {}
    for rank in args.rank_list:
        print(f"\n########## RANK {rank} ##########", flush=True)
        rs = run_for_rank(rank, args.k_list, args.N, args.K, args.n_targets, args.lr, args.T, dev,
                          frozen, b0, ds_mean, pool, y_pool, dataset=ds)
        for r in rs:
            print(json.dumps(r), flush=True)
        assert any(math.isfinite(r["d2_obs"]) for r in rs), \
            f"rank {rank}: ALL d2_obs NaN — metric starved (too many dropped draws). Aborting."
        by_rank[rank] = rs

    summary = dict(N=args.N, K=args.K, T=args.T, lr=args.lr, dataset=ds, k_list=args.k_list, rank_list=args.rank_list,
                   by_rank={}, scaling={})
    print("\n=== SUMMARY (rank | k | sensitivity | d2_obs | p | Σ-noise | subcos | n_metric) ===")
    for rank in args.rank_list:
        rs = by_rank[rank]
        ks = [r["k"] for r in rs]
        b_sens, r2_sens, ns, nd = fit_beta(ks, [r["whitened_sensitivity"] for r in rs])
        b_d2, r2_d2, _, _ = fit_beta(ks, [r["d2_obs"] for r in rs])
        summary["by_rank"][rank] = rs
        summary["scaling"][rank] = dict(beta_sensitivity=b_sens, r2_sensitivity=r2_sens,
                                        beta_dropped_nonpos=nd, beta_d2obs=b_d2, r2_d2obs=r2_d2)
        for r in rs:
            print(f"r={rank:>2} k={r['k']:>2}: sens={r['whitened_sensitivity']:.3f}  d2={r['d2_obs']:.3f}  "
                  f"p={r['whitened_pvalue']:.3f}  Σnoise={r['reseed_noise']:.2e}  "
                  f"subcos={r['copies_vs_distinct_subspace_cos']:.3f}  nm={r['n_metric']}", flush=True)
        print(f"   -> β(sensitivity)={b_sens:.3f} (R²={r2_sens:.3f}, dropped≤0={nd}) | "
              f"β(d2_obs)={b_d2:.3f} (R²={r2_d2:.3f})", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    ds_tag = "" if ds == "mnist" else f"_{ds}"
    with open(os.path.join(RESULTS, f"arm_e_summary{ds_tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if len(args.rank_list) >= 2:
        lo, hi = min(args.rank_list), max(args.rank_list)
        bl = summary["scaling"][lo]["beta_sensitivity"]; bh = summary["scaling"][hi]["beta_sensitivity"]
        print(f"\nEMPIRICAL-NULL READ: β(sens) r={lo} = {bl:.3f}  vs  r={hi} = {bh:.3f}")
        print("  β(low) < β(high) ⇒ low-rank bottleneck SATURATES duplication imprint (privacy-PROTECTIVE);")
        print("  β(low) ≈ β(high) ⇒ duplication leaks regardless of capacity.")


if __name__ == "__main__":
    main()
