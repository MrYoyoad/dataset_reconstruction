#!/usr/bin/env python
"""
H SPOT-CHECK (program plan §III.0) — the CHEAP de-risking gate that runs BEFORE all expensive
scaling (§III.1/§III.2): does whitened adapter-sensitivity actually predict BEHAVIORAL memorization?

WHAT IT DOES. For the EXACT 12 (image, context) cells whose whitened sensitivity is already
measured (arm D, job 245964: 3 fixed targets x m in {1,2,4,8} contexts, saved in
results/arm_d_context/arm_d_summary.json per_target_sens), compute a Feldman-style LEAVE-ONE-OUT
behavioral memorization score and rank-correlate it against the saved sensitivities.

MEM SCORE (on the margin — behavioral, ΔW enters only through the logit):
  mem(T, D) = E_seeds[ margin_T(adapter trained on D) − margin_T(adapter trained on D \\ {T}) ]
  margin_T  = (2 y_T − 1) · f(x0_T; θ0 + ΔW),   x0_T = x_T − ds_mean (ds_mean FROZEN, arms' centering)
Removing T means DROPPING it (set size N−1), NOT swapping it — the margin GAIN that T's inclusion
buys itself is the standard memorization functional. Both trainings in a pair share the SAME B0
init seed (seeds 1000+j, the arms' own ensemble seeds), so init noise partially cancels per pair.

PRE-REGISTERED (printed BEFORE any measurement): prediction = POSITIVE rank-correlation
rho(mem, sens); what-good-looks-like rho > +0.4 with a visible trend. KILL (plan §III.0): if
rho ≈ 0 or negative with a decent CI, whitened sensitivity is NOT tracking memorization and all
downstream scaling re-labels to "parametric detectability only".

ALSO: rho(mem, g0) — is behavioral memorization itself margin/gradient-predicted? (g0 = per-image
layer-0 full-weight gradient norm at θ0; only 3 distinct values across the 12 cells since g0 is a
pure image property — average-rank ties handled by spearman(); read it as a coarse check.)

KNOWN CONFOUNDS (flagged, not hidden):
 1. SET-SIZE / CONSTRUCTION MISMATCH: sensitivity was measured by SWAPPING T for a same-class T'
    (set size stays N=16); mem is measured by DROPPING T (size N−1, class balance shifts by one).
    Both perturbations are "the counterfactual without T's identity", but they are not identical
    operators. Rank correlation is the right statistic precisely because it only needs the two
    functionals to be MONOTONE in the same underlying per-image memorization load, not equal —
    and any set-size effect is a per-cell CONSTANT offset shared by all 3 targets within an m,
    which the per-m rho is immune to (reported alongside the pooled rho).
 2. m=1 DEGENERACY: at m=1 the LOO set D\\{T} has ZERO class-1 members (15 class-0 only) — the LOO
    adapter never sees class 1 at all, which inflates mem(T) mechanically. Reported: pooled rho
    WITH and WITHOUT the m=1 cells (robustness line).
 3. SMALL n: n=12 pooled → permutation-p reported and everything labeled SMALL-n honestly.

Identity guard: hard-asserts the reconstructed target digits == arm_d_summary.json
per_target_digit AND the summary config == this run's (same style as margin_vs_sensitivity).

Outputs: results/h_spotcheck/h_spotcheck.json + .pth, headline scatter (sens x, mem y, colored
by m, expected positive slope) → figures/h_spotcheck/h_spotcheck_scatter.png.
~2*12*K_loo = 240 trainings at K_loo=10 (T=1000 tiny MLP, ~1.5 s each on an L40S/DGX: ~10 min).
bsub-only. mnist / gelu / binary. float64.
"""
import os, json, math, random, argparse
import torch

from experiments.jacobian_spectrum import make_activation
from experiments.dataset_sensitivity.arm_b_dilution import draw_B0, train_adapter
from experiments.dataset_sensitivity.arm_d_context import build_base, _split_pools, _build_context
from experiments.dataset_sensitivity.margin_vs_sensitivity import (
    margins, layer0_grad_norms, spearman, _zero_adapter,
)

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/h_spotcheck"
FIGURES = "/home/projects/galvardi/yoado/figures/h_spotcheck"
ARM_D_SUMMARY = "/home/projects/galvardi/yoado/results/arm_d_context/arm_d_summary.json"

PREREG = (
    "PRE-REGISTERED (before measurement, plan §III.0):\n"
    "  PREDICTION: POSITIVE rank-correlation rho(mem, sens) — whitened adapter-sensitivity and\n"
    "    behavioral LOO memorization rank the 12 (image, context) cells the same way.\n"
    "    What-good-looks-like: rho_rank > +0.4 with a visible trend in the scatter.\n"
    "  KILL: rho(mem, sens) <= 0 (or ~0 with a decent CI) => whitened sensitivity is NOT tracking\n"
    "    memorization; do NOT scale §III.1/§III.2 until §III.3 is understood, and all downstream\n"
    "    scaling re-labels to 'parametric detectability only' — no 'leakage' language."
)


def target_margin(x_T, y_T, ds_mean, frozen, b0, act, A, B):
    """margin_T = (2 y_T - 1) * f(x_T - ds_mean; θ0 + BA). x_T: [1, ...], y_T: [1]."""
    return margins(x_T - ds_mean, y_T, frozen, b0, act, A, B)[0].item()


def loo_mem_for_cell(m, target_idx, N, K_loo, lr, T, rank, device, frozen, b0, ds_mean,
                     pool_x, pool_digits, maj_pool, peer_pool, act, out_f):
    """One (target, m) cell: K_loo paired trainings (full D vs D\\{T}, SAME B0 seed per pair).
    Returns dict with mem (mean margin gain), per-seed values, memorization diagnostics."""
    x_D, y_D, digits_D, t_pos = _build_context(pool_x, pool_digits, maj_pool, peer_pool,
                                               target_idx, N, m, device)
    x0_D = x_D - ds_mean                                              # ds_mean FROZEN
    x_T, y_T = x_D[t_pos:t_pos + 1], y_D[t_pos:t_pos + 1]
    # LOO set: DROP T (size N-1). At m=1 this leaves ZERO class-1 members (confound 2, flagged).
    x_loo = torch.cat([x_D[:t_pos], x_D[t_pos + 1:]], 0)
    y_loo = torch.cat([y_D[:t_pos], y_D[t_pos + 1:]], 0)
    x0_loo = x_loo - ds_mean

    seeds = [1000 + j for j in range(K_loo)]                          # the arms' ensemble seeds
    gains, m_full_l, m_loo_l, mbce_full_l = [], [], [], []
    dropped = 0
    for s in seeds:
        B0 = draw_B0(s, out_f, rank, device)                          # SAME init for both arms of the pair
        A_f, B_f, mbce_f, dW_f = train_adapter(frozen, b0, B0, x0_D, y_D, lr, T, act, rank)
        A_l, B_l, _, dW_l = train_adapter(frozen, b0, B0, x0_loo, y_loo, lr, T, act, rank)
        if not (torch.isfinite(dW_f).all() and torch.isfinite(dW_l).all()):
            dropped += 1
            continue
        mf = target_margin(x_T, y_T, ds_mean, frozen, b0, act, A_f, B_f)
        ml = target_margin(x_T, y_T, ds_mean, frozen, b0, act, A_l, B_l)
        if not (math.isfinite(mf) and math.isfinite(ml)):
            dropped += 1
            continue
        gains.append(mf - ml); m_full_l.append(mf); m_loo_l.append(ml); mbce_full_l.append(mbce_f)
    n_fin = len(gains)
    ok = n_fin >= max(2, K_loo // 2)
    mem = (sum(gains) / n_fin) if ok else float("nan")
    mem_std = (math.sqrt(sum((g - mem) ** 2 for g in gains) / (n_fin - 1))
               if (ok and n_fin > 1) else float("nan"))
    return dict(m=m, target_idx=target_idx, digit=int(pool_digits[target_idx]), t_pos=t_pos,
                mem=mem, mem_std=mem_std, mem_sem=(mem_std / math.sqrt(n_fin) if ok else float("nan")),
                n_finite=n_fin, dropped=dropped,
                mean_margin_full=(sum(m_full_l) / n_fin if ok else float("nan")),
                mean_margin_loo=(sum(m_loo_l) / n_fin if ok else float("nan")),
                mean_max_bce_full=(sum(mbce_full_l) / n_fin if ok else float("nan")),
                memorized=bool(ok and max(mbce_full_l) < 1e-3),
                per_seed_gain=gains, loo_has_class1=bool(m > 1))


def perm_pvalue(a, b, rho_obs, n_perm=20000, seed=0):
    """One-sided permutation p for the PRE-REGISTERED positive direction: P(rho_perm >= rho_obs)
    under random re-pairing of b against a. Honest small-n inference (n=12: exact-ish at 20k)."""
    if not math.isfinite(rho_obs):
        return float("nan")
    rng = random.Random(seed)
    bb = list(b)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(bb)
        r, _ = spearman(a, bb)
        if math.isfinite(r) and r >= rho_obs:
            count += 1
    return (1 + count) / (1 + n_perm)


def make_plot(cells, sens_by, rho, p, path):
    """HEADLINE PLOT (plan §III.0): whitened sensitivity (x) vs LOO mem score (y), one point per
    (image, context) cell, colored by m; expected shape: positive slope."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m_vals = sorted({c["m"] for c in cells})
    cmap = plt.cm.viridis                                              # version-proof (no get_cmap)
    colors = {m: cmap(i / max(len(m_vals) - 1, 1)) for i, m in enumerate(m_vals)}
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for m in m_vals:
        cs = [c for c in cells if c["m"] == m and math.isfinite(c["mem"])]
        xs = [sens_by[(c["m"], c["tgt_id"])] for c in cs]
        ys = [c["mem"] for c in cs]
        es = [c["mem_sem"] for c in cs]
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=8, capsize=3, color=colors[m],
                    label=f"m={m}" + (" (LOO set has no class-1!)" if m == 1 else ""))
        for c, x, y in zip(cs, xs, ys):
            ax.annotate(str(c["digit"]), (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=8, alpha=0.7)
    ax.set_xlabel("whitened sensitivity (arm D, measured — job 245964)")
    ax.set_ylabel("LOO mem score  E_seeds[margin(D) − margin(D\\{T})]")
    ax.set_title(f"§III.0 H spot-check — sens vs behavioral LOO memorization\n"
                 f"pooled Spearman rho={rho:+.3f}, perm-p={p:.4f} (n=12, SMALL-n; "
                 f"expect POSITIVE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K_loo", type=int, default=10, help="paired LOO seeds per cell (2 trainings each)")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--m_list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n_targets", type=int, default=3)
    ap.add_argument("--stage0", action="store_true",
                    help="tiny sanity: 1 target, m=4 only, K_loo=4, assert finite mem")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    N, T, lr, rank = args.N, args.T, args.lr, args.rank

    if args.stage0:
        m_list, n_targets_run, K_loo = [4], 1, 4
        print(f"=== STAGE-0 SANITY (1 target, m=4 only, K_loo={K_loo}) ===", flush=True)
    else:
        m_list, n_targets_run, K_loo = list(args.m_list), args.n_targets, args.K_loo

    # ---- PRE-REGISTRATION: printed BEFORE any measurement ----
    print(PREREG, flush=True)

    # ---- rebuild the EXACT arm-D pools (same calls/seeds/formula as arm_d_context.main and
    #      margin_vs_sensitivity.main — pool geometry must match the SAVED sensitivities) ----
    full_m_list = [m for m in args.m_list if 1 <= m <= N - 1]          # pool formula uses the FULL m_list
    assert full_m_list, "no valid m"
    max_m, m_min = max(full_m_list), min(full_m_list)
    need_c1 = args.n_targets + (max_m - 1)
    pool_n = max(N - m_min, need_c1, 18)                               # arm_d.main's exact formula
    frozen, b0, ds_mean, pool_x, pool_y, pool_digits = build_base(N, lr, T, dev, pool_n=pool_n)
    target_pool, peer_pool, maj_pool = _split_pools(pool_y, N, max_m, args.n_targets, m_min)
    act = make_activation("gelu")
    out_f = frozen[0].shape[0]
    my_digits = [int(pool_digits[i]) for i in target_pool]
    print(f"pools: targets={target_pool} (digits={my_digits}), {len(peer_pool)} peers, "
          f"{len(maj_pool)} majority", flush=True)

    # ---- IDENTITY GUARD (hard asserts, margin_vs_sensitivity style) ----
    assert os.path.exists(ARM_D_SUMMARY), f"{ARM_D_SUMMARY} missing — nothing to correlate against."
    with open(ARM_D_SUMMARY) as f:
        sd = json.load(f)
    cfg = dict((k, sd[k]) for k in ("N", "T", "lr", "rank", "n_targets"))
    assert (sd["N"] == N and sd["T"] == T and sd["lr"] == lr and sd["rank"] == rank
            and sd["n_targets"] == args.n_targets), (
        f"IDENTITY BROKEN: arm_d_summary config {cfg} != this run's "
        f"{dict(N=N, T=T, lr=lr, rank=rank, n_targets=args.n_targets)} — refusing to correlate.")
    sens_by = {}                                                       # (m, tgt_id) -> measured sensitivity
    for r in sd["results"]:
        assert r["per_target_digit"] == my_digits, (
            f"IDENTITY BROKEN: arm-D summary target digits {r['per_target_digit']} != "
            f"reconstructed {my_digits} (pool drift?) — refusing to correlate.")
        for t, s in enumerate(r["per_target_sens"]):
            sens_by[(r["m"], t)] = s
    print(f"identity guard OK: digits {my_digits}, config {cfg}", flush=True)

    # ---- g0 for every pool image (cheap, no training) — pure image property ----
    g0 = layer0_grad_norms(pool_x - ds_mean, pool_y, frozen, b0, act)
    A0z, B0z = _zero_adapter(frozen, rank, dev)
    m0 = margins(pool_x - ds_mean, pool_y, frozen, b0, act, A0z, B0z).cpu()
    assert torch.isfinite(g0).all() and torch.isfinite(m0).all(), "base g0/m0 non-finite"

    # ---- BEHAVIORAL LOO mem score per cell (2*K_loo trainings each) ----
    cells = []
    for m in m_list:
        for tgt_id in range(n_targets_run):
            target_idx = target_pool[tgt_id]
            c = loo_mem_for_cell(m, target_idx, N, K_loo, lr, T, rank, dev, frozen, b0, ds_mean,
                                 pool_x, pool_digits, maj_pool, peer_pool, act, out_f)
            c["tgt_id"] = tgt_id
            c["g0"] = g0[target_idx].item()
            c["m0"] = m0[target_idx].item()
            c["sens"] = sens_by.get((m, tgt_id), float("nan"))
            cells.append(c)
            print(f"cell m={m} tgt={tgt_id} (digit {c['digit']}): mem={c['mem']:+.4f} "
                  f"(+/-{c['mem_sem']:.4f} sem, {c['n_finite']}/{K_loo} finite, dropped {c['dropped']}) "
                  f"margin full={c['mean_margin_full']:.3f} loo={c['mean_margin_loo']:.3f} "
                  f"sens={c['sens']:.3f} g0={c['g0']:.3e} memorized={c['memorized']}"
                  + ("  [LOO set has NO class-1 (m=1 confound)]" if m == 1 else ""), flush=True)
            if not c["memorized"]:
                print(f"WARNING: cell m={m} tgt={tgt_id} full-set NOT memorized — mem is "
                      f"off-convergence; interpret with care.", flush=True)

    if args.stage0:
        c = cells[0]
        assert math.isfinite(c["mem"]), "STAGE-0 FAIL: mem non-finite"
        assert c["n_finite"] >= 2, "STAGE-0 FAIL: <2 finite LOO pairs"
        assert math.isfinite(c["sens"]), "STAGE-0 FAIL: saved sensitivity missing for (m=4, tgt=0)"
        print("STAGE-0 OK", flush=True)
        return

    fin = [c for c in cells if math.isfinite(c["mem"]) and math.isfinite(c["sens"])]
    assert len(fin) >= 10, f"only {len(fin)}/12 cells finite — metric starved, refusing to report rho"

    # ---- CORRELATIONS ----
    mem_v = [c["mem"] for c in fin]
    sens_v = [c["sens"] for c in fin]
    g0_v = [c["g0"] for c in fin]
    rho_sens, n_s = spearman(mem_v, sens_v)
    p_sens = perm_pvalue(mem_v, sens_v, rho_sens, seed=1)
    rho_g0, n_g = spearman(mem_v, g0_v)
    p_g0 = perm_pvalue(mem_v, g0_v, rho_g0, seed=2)
    # per-m rho + sign pattern (immune to any per-m constant offset from the N-1 confound; n=3 each!)
    per_m, signs = {}, []
    for m in m_list:
        cm = [c for c in fin if c["m"] == m]
        r_m, n_m = spearman([c["mem"] for c in cm], [c["sens"] for c in cm])
        per_m[m] = dict(rho=r_m, n=n_m)
        signs.append(f"m={m}:" + ("+" if r_m > 0 else ("-" if r_m < 0 else "0")) if math.isfinite(r_m)
                     else f"m={m}:nan")
    sign_pattern = " ".join(signs)
    # robustness: drop the degenerate m=1 cells (LOO set has zero class-1 members)
    fin_no1 = [c for c in fin if c["m"] != 1]
    rho_no1, n_no1 = spearman([c["mem"] for c in fin_no1], [c["sens"] for c in fin_no1])
    p_no1 = perm_pvalue([c["mem"] for c in fin_no1], [c["sens"] for c in fin_no1], rho_no1, seed=3)

    # ---- VERDICT vs the pre-registration ----
    if not math.isfinite(rho_sens):
        verdict = "NO-DATA"
    elif rho_sens > 0.4:
        verdict = f"PASS (rho={rho_sens:+.3f} > +0.4, perm-p={p_sens:.4f}, n={n_s} SMALL-n) — de-risked: proceed to scale §III.1/§III.2 (III.3 full gate still required)"
    elif rho_sens > 0:
        verdict = f"WEAK-POSITIVE (0 < rho={rho_sens:+.3f} <= +0.4, perm-p={p_sens:.4f}, n={n_s} SMALL-n) — trend right-signed but below the pre-registered bar; scale with caution, §III.3 URGENT"
    else:
        verdict = (f"KILL (rho={rho_sens:+.3f} <= 0, n={n_s}) — whitened sensitivity is NOT tracking "
                   f"memorization: do NOT scale §III.1/§III.2 until §III.3 is understood; all "
                   f"downstream scaling re-labels to 'parametric detectability only' — no leakage language")

    print("\n=== §III.0 H SPOT-CHECK RESULTS (n=12, SMALL-n — honest labels) ===", flush=True)
    print(f"rho(mem, sens) pooled = {rho_sens:+.3f}  perm-p(one-sided, positive) = {p_sens:.4f}  (n={n_s})")
    print(f"  per-m signs (n=3 each, offset-immune): [{sign_pattern}]")
    print(f"  robustness excl. m=1 (LOO-degenerate): rho={rho_no1:+.3f} perm-p={p_no1:.4f} (n={n_no1})")
    print(f"rho(mem, g0)   pooled = {rho_g0:+.3f}  perm-p = {p_g0:.4f}  (n={n_g}; g0 has only "
          f"{len(set(g0_v))} distinct values — image property, coarse check)")
    print(f"VERDICT: {verdict}", flush=True)

    # ---- save + headline plot ----
    os.makedirs(RESULTS, exist_ok=True)
    out = dict(
        prereg=PREREG,
        config=dict(N=N, m_list=m_list, n_targets=args.n_targets, K_loo=K_loo, lr=lr, T=T,
                    rank=rank, pool_n=pool_n, loo_construction="DROP (size N-1), not swap",
                    seeds=[1000 + j for j in range(K_loo)]),
        confounds=["swap(N) vs drop(N-1) construction mismatch — rank-corr + per-m rho mitigate",
                   "m=1 LOO set has zero class-1 members — see rho_excl_m1",
                   "n=12 SMALL-n — permutation-p reported, no CI overclaim"],
        cells=[{k: v for k, v in c.items() if k != "per_seed_gain"} for c in cells],
        correlations=dict(
            rho_mem_sens=rho_sens, perm_p_mem_sens=p_sens, n=n_s,
            per_m={str(k): v for k, v in per_m.items()}, sign_pattern=sign_pattern,
            rho_mem_sens_excl_m1=rho_no1, perm_p_excl_m1=p_no1, n_excl_m1=n_no1,
            rho_mem_g0=rho_g0, perm_p_mem_g0=p_g0),
        verdict=verdict)
    with open(os.path.join(RESULTS, "h_spotcheck.json"), "w") as f:
        json.dump(out, f, indent=2)
    torch.save(dict(cells=cells, g0=g0, m0=m0, pool_y=pool_y.cpu(),
                    pool_digits=[int(d) for d in pool_digits],
                    sens_by={f"m{m}_t{t}": s for (m, t), s in sens_by.items()},
                    summary=out), os.path.join(RESULTS, "h_spotcheck.pth"))
    fig_path = os.path.join(FIGURES, "h_spotcheck_scatter.png")
    make_plot(cells, sens_by, rho_sens, p_sens, fig_path)
    print(f"\nsaved {RESULTS}/h_spotcheck.json + .pth and {fig_path}", flush=True)


if __name__ == "__main__":
    main()
