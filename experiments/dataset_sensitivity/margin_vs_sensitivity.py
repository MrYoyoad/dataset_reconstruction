#!/usr/bin/env python
"""
Margin-vs-Sensitivity — ANALYSIS arm testing the SUPPORT-VECTOR / MAX-MARGIN hypothesis
(Haim et al. KKT) for LoRA leakage.

HYPOTHESIS: images with SMALL margin under the pretrained base theta_0 (the ones theta_0
classifies least confidently — "support-vector-like") force the adapter to do more work and
therefore leak more. Predictions:
  P1  per-image whitened sensitivity ANTI-correlates with the base margin m0_i.
  P2  sensitivity POSITIVELY correlates with the per-image layer-0 gradient norm g0_i at theta_0.
  P3  sensitivity POSITIVELY correlates with the post-fine-tune KKT dual-weight proxy
      lam_i = sigmoid(-mT_i) = |dBCE/dlogit| at theta_0 + dW.
  P4  CLASS ASYMMETRY: the class-1 examples (measured ~3.3x louder in arm C's balanced control)
      have SMALLER mean base margin and LARGER mean g0 / lam than class-0.

This arm trains NO new sensitivity measurements — it reconstructs the EXACT pools/sets the
already-run arms C and D used (by CALLING the arms' own build functions with the arms' exact
arguments) and correlates cheap per-image margin/gradient quantities against the measured
per_target_sens in results/arm_d_context/arm_d_summary.json (n=3 targets x 4 contexts = 12,
reported honestly as small-n) and the class-level sens_minority/sens_majority in
results/arm_c_imbalance/arm_c_summary*.json.

Quantities per image (x0 = x - ds_mean, ds_mean FROZEN, same centering the arms train with):
  m0_i = (2 y_i - 1) * f(x0_i; theta_0)          base margin (zero adapter — no training)
  g0_i = || d BCE(f(x0_i), y_i) / d W_0 ||_F     layer-0 full-weight per-image gradient norm at theta_0
  mT_i = (2 y_i - 1) * f(x0_i; theta_0 + dW)     post-fine-tune margin (reference adapter, seed 1000)
  lam_i = sigmoid(-mT_i)                          KKT dual-weight proxy (= |dBCE/dlogit|)
  gT_i = same as g0_i but at theta_0 + dW         post-fine-tune per-image gradient norm

Reference adapters mirror the arms exactly: seed 1000 (= the arms' seeds[0]), draw_B0 /
train_adapter from arm_b_dilution, lr 0.5, T 1000, rank 8, gelu, float64.

Outputs: results/margin_vs_sensitivity/margins.json (all numbers + verdicts) and margins.pth
(per-image tensors). No plotting. bsub-only. mnist / gelu / binary. float64.
"""
import os, json, math, argparse
import torch
import torch.nn.functional as F

from experiments.jacobian_spectrum import make_activation
from experiments.dataset_sensitivity.arm_b_dilution import (
    draw_B0, train_adapter, forward_logits,
)
from experiments.dataset_sensitivity.arm_d_context import (
    build_base, _split_pools, _build_context,
)
from experiments.dataset_sensitivity.arm_c_imbalance import _build_imbalanced

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/margin_vs_sensitivity"
ARM_D_SUMMARY = "/home/projects/galvardi/yoado/results/arm_d_context/arm_d_summary.json"
ARM_C_SUMMARIES = [  # prefer the explicit minc1 name; the deployed minc=1 run predates the suffix
    "/home/projects/galvardi/yoado/results/arm_c_imbalance/arm_c_summary_minc1.json",
    "/home/projects/galvardi/yoado/results/arm_c_imbalance/arm_c_summary.json",
]


# ---------------------------------------------------------------------------
# per-image quantities
# ---------------------------------------------------------------------------
def _zero_adapter(frozen, rank, device):
    out_f, in_f = frozen[0].shape
    A = {0: torch.zeros(rank, in_f, dtype=torch.float64, device=device)}
    B = {0: torch.zeros(out_f, rank, dtype=torch.float64, device=device)}
    return A, B


def margins(x0, y, frozen, b0, act, A, B):
    """(2y-1) * logit for every row of x0 under frozen + B@A on layer 0. No grad."""
    with torch.no_grad():
        z = forward_logits(x0, frozen, b0, A, B, act).view(-1)
    return (2.0 * y - 1.0) * z


def layer0_grad_norms(x0, y, frozen, b0, act, dW=None):
    """Per-image ||d BCE / d W_0||_F with W_0 the FULL layer-0 weight (frozen[0] [+ dW]).
    One backward per image (cheap: tiny MLP)."""
    n_layers = len(frozen)
    W_eff = frozen[0] if dW is None else (frozen[0] + dW)
    out = []
    for i in range(x0.shape[0]):
        W = W_eff.clone().detach().requires_grad_(True)
        h = x0[i:i + 1].reshape(1, -1)
        for l in range(n_layers):
            h = F.linear(h, W if l == 0 else frozen[l], b0 if l == 0 else None)
            if l < n_layers - 1:
                h = act(h)
        loss = F.binary_cross_entropy_with_logits(h.view(-1), y[i].view(-1))
        loss.backward()
        out.append(W.grad.norm().item())
    return torch.tensor(out, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Spearman rank correlation (no scipy dependency)
# ---------------------------------------------------------------------------
def _ranks(v):
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    """Spearman rho over finite pairs. Returns (rho, n). nan if n<3 or a side is constant."""
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    ra = _ranks([p[0] for p in pairs])
    rb = _ranks([p[1] for p in pairs])
    ma, mb = sum(ra) / n, sum(rb) / n
    ca = [x - ma for x in ra]
    cb = [x - mb for x in rb]
    va = sum(x * x for x in ca)
    vb = sum(x * x for x in cb)
    if va <= 0 or vb <= 0:
        return float("nan"), n
    return sum(x * y for x, y in zip(ca, cb)) / math.sqrt(va * vb), n


def _verdict(rho, expected_sign, n, strong=0.5):
    if not math.isfinite(rho):
        return "NO-DATA"
    if abs(rho) < strong:
        return f"MIXED/weak (|rho|<{strong}, n={n})"
    return (f"SUPPORTED (n={n})" if math.copysign(1.0, rho) == expected_sign
            else f"REFUTED (n={n})")


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------
def train_reference_adapter(frozen, b0, x0_D, y_D, lr, T, act, rank, out_f, device, seed=1000):
    """The arms' seeds[0]=1000 baseline member: draw_B0(1000) + train_adapter (SGD, A0=0)."""
    A, B, max_bce, dW = train_adapter(frozen, b0, draw_B0(seed, out_f, rank, device),
                                      x0_D, y_D, lr, T, act, rank)
    return A, B, max_bce, dW


def per_image_after(x_D, y_D, ds_mean, frozen, b0, act, A, B, dW):
    x0_D = x_D - ds_mean
    mT = margins(x0_D, y_D, frozen, b0, act, A, B)
    lam = torch.sigmoid(-mT)
    gT = layer0_grad_norms(x0_D, y_D, frozen, b0, act, dW=dW)
    return mT.cpu(), lam.cpu(), gT.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--m_list", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--n_targets", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--armc_m", type=int, default=8,
                    help="arm-C balanced point (m = minority count, minority_class=1)")
    ap.add_argument("--stage0", action="store_true",
                    help="tiny sanity: base margins on the pool + ONE adapter at m=4, assert finite")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    N, T, lr, rank, n_targets = args.N, args.T, args.lr, args.rank, args.n_targets

    # ---- rebuild the EXACT arm-D pools (same calls, same seeds as arm_d_context.main) ----
    m_list = [m for m in args.m_list if 1 <= m <= N - 1]
    assert m_list, "no valid m"
    max_m, m_min = max(m_list), min(m_list)
    need_c1 = n_targets + (max_m - 1)
    pool_n = max(N - m_min, need_c1, 18)               # arm_d.main's exact formula (defaults -> 18)
    if pool_n != max(N, 18):
        print(f"LOUD WARNING: pool_n={pool_n} != arm-C's max(N,18)={max(N, 18)} — the arm-C pool "
              f"would DIFFER from this reconstruction; arm-C class comparison is then invalid.",
              flush=True)
    frozen, b0, ds_mean, pool_x, pool_y, pool_digits = build_base(N, lr, T, dev, pool_n=pool_n)
    target_pool, peer_pool, maj_pool = _split_pools(pool_y, N, max_m, n_targets, m_min)
    act = make_activation("gelu")
    out_f = frozen[0].shape[0]
    print(f"pools: {len(pool_x)} images, targets={target_pool} "
          f"(digits={[int(pool_digits[i]) for i in target_pool]}), "
          f"{len(peer_pool)} peers, {len(maj_pool)} majority", flush=True)

    # ---- 1) base quantities for EVERY pool image (no training) ----
    x0_pool = pool_x - ds_mean
    A0, B0z = _zero_adapter(frozen, rank, dev)
    m0 = margins(x0_pool, pool_y, frozen, b0, act, A0, B0z).cpu()
    g0 = layer0_grad_norms(x0_pool, pool_y, frozen, b0, act)
    assert torch.isfinite(m0).all() and torch.isfinite(g0).all(), "base margins/gradnorms non-finite"
    rho_m0_g0, _ = spearman(m0.tolist(), g0.tolist())
    print(f"pool base margins: m0 in [{m0.min():.3f}, {m0.max():.3f}], "
          f"g0 in [{g0.min():.3e}, {g0.max():.3e}], spearman(m0,g0)={rho_m0_g0:.3f} "
          f"(strongly negative expected: small margin <-> large gradient)", flush=True)

    if args.stage0:
        m = 4
        x_D, y_D, digits_D, t_pos = _build_context(pool_x, pool_digits, maj_pool, peer_pool,
                                                   target_pool[0], N, m, dev)
        A, B, max_bce, dW = train_reference_adapter(frozen, b0, x_D - ds_mean, y_D,
                                                    lr, T, act, rank, out_f, dev)
        mT, lam, gT = per_image_after(x_D, y_D, ds_mean, frozen, b0, act, A, B, dW)
        assert torch.isfinite(mT).all() and torch.isfinite(lam).all() and torch.isfinite(gT).all(), \
            "post-LoRA quantities non-finite"
        assert torch.isfinite(dW).all() and dW.norm() > 0, "reference dW degenerate"
        print(f"stage0 adapter (m={m}, target {target_pool[0]}): max_bce={max_bce:.2e}, "
              f"|dW|={dW.norm():.3f}, target mT={mT[t_pos]:.3f} lam={lam[t_pos]:.3e} "
              f"gT={gT[t_pos]:.3e}", flush=True)
        print("STAGE-0 OK")
        return

    # ---- 3) AFTER-LoRA quantities: one reference adapter per arm-D context (m, target) ----
    contexts = []
    for m in m_list:
        for tgt_id, target_idx in enumerate(target_pool):
            x_D, y_D, digits_D, t_pos = _build_context(pool_x, pool_digits, maj_pool, peer_pool,
                                                       target_idx, N, m, dev)
            A, B, max_bce, dW = train_reference_adapter(frozen, b0, x_D - ds_mean, y_D,
                                                        lr, T, act, rank, out_f, dev)
            mT, lam, gT = per_image_after(x_D, y_D, ds_mean, frozen, b0, act, A, B, dW)
            c = dict(m=m, tgt_id=tgt_id, target_idx=target_idx, digit=int(pool_digits[target_idx]),
                     t_pos=t_pos, max_bce=max_bce, memorized=bool(max_bce < 1e-3),
                     dW_norm=dW.norm().item(),
                     m0_target=m0[target_idx].item(), g0_target=g0[target_idx].item(),
                     mT_target=mT[t_pos].item(), lam_target=lam[t_pos].item(),
                     gT_target=gT[t_pos].item(),
                     mT=mT.tolist(), lam=lam.tolist(), gT=gT.tolist(),
                     pool_idx=None)
            contexts.append(c)
            if not c["memorized"]:
                print(f"WARNING: context m={m} tgt={tgt_id} NOT memorized "
                      f"(max_bce={max_bce:.2e}) — lam-proxy off-convergence.", flush=True)
            print(f"context m={m} tgt={tgt_id} (digit {c['digit']}): max_bce={max_bce:.2e} "
                  f"m0={c['m0_target']:.3f} mT={c['mT_target']:.3f} "
                  f"lam={c['lam_target']:.3e} g0={c['g0_target']:.3e} gT={c['gT_target']:.3e}",
                  flush=True)

    # ---- arm-C balanced set (m=armc_m, minority_class=1): class-level lam ----
    xc, yc, digits_c, maj_pos, min_pos = _build_imbalanced(pool_x, pool_y, pool_digits,
                                                           N, args.armc_m, dev, minority_class=1)
    Ac, Bc, mbce_c, dWc = train_reference_adapter(frozen, b0, xc - ds_mean, yc,
                                                  lr, T, act, rank, out_f, dev)
    mTc, lamc, gTc = per_image_after(xc, yc, ds_mean, frozen, b0, act, Ac, Bc, dWc)
    cmean = lambda t, pos: t[pos].mean().item()
    armc_after = dict(m=args.armc_m, minority_class=1, max_bce=mbce_c,
                      memorized=bool(mbce_c < 1e-3),
                      mean_lam_c0=cmean(lamc, torch.tensor(maj_pos)),
                      mean_lam_c1=cmean(lamc, torch.tensor(min_pos)),
                      mean_mT_c0=cmean(mTc, torch.tensor(maj_pos)),
                      mean_mT_c1=cmean(mTc, torch.tensor(min_pos)),
                      mean_gT_c0=cmean(gTc, torch.tensor(maj_pos)),
                      mean_gT_c1=cmean(gTc, torch.tensor(min_pos)))
    if not armc_after["memorized"]:
        print(f"WARNING: arm-C balanced set NOT memorized (max_bce={mbce_c:.2e}).", flush=True)

    # ---- load the MEASURED sensitivities ----
    sens_by = {}          # (m, tgt_id) -> measured whitened sensitivity
    armd_ok = False
    if os.path.exists(ARM_D_SUMMARY):
        with open(ARM_D_SUMMARY) as f:
            sd = json.load(f)
        cfg_match = (sd["N"] == N and sd["T"] == T and sd["lr"] == lr and sd["rank"] == rank
                     and sd["n_targets"] == n_targets)
        if not cfg_match:
            print(f"LOUD WARNING: arm_d_summary config {dict((k, sd[k]) for k in ('N', 'T', 'lr', 'rank', 'n_targets'))} "
                  f"!= this run's — arm-D correlations SKIPPED (identity not guaranteed).", flush=True)
        else:
            my_digits = [int(pool_digits[i]) for i in target_pool]
            for r in sd["results"]:
                assert r["per_target_digit"] == my_digits, (
                    f"IDENTITY BROKEN: arm-D summary target digits {r['per_target_digit']} != "
                    f"reconstructed {my_digits} (pool drift?) — refusing to correlate.")
                if r["m"] in m_list:
                    for t, s in enumerate(r["per_target_sens"]):
                        sens_by[(r["m"], t)] = s
            armd_ok = True
    else:
        print(f"LOUD WARNING: {ARM_D_SUMMARY} missing — arm-D correlations skipped.", flush=True)

    armc_meas = None
    for p in ARM_C_SUMMARIES:
        if os.path.exists(p):
            with open(p) as f:
                sc = json.load(f)
            if sc.get("minority_class", 1) != 1:
                continue
            for r in sc["results"]:
                if r["m"] == args.armc_m:
                    armc_meas = dict(source=p, sens_minority=r["sens_minority"],
                                     sens_majority=r["sens_majority"], ratio=r["ratio"])
            break
    if armc_meas is None:
        print("LOUD WARNING: no arm-C minc1 summary with the balanced point — "
              "measured class ratio unavailable.", flush=True)

    # ---- 4) CORRELATIONS ----
    corr = dict(pooled={}, per_m={}, sign_pattern={})
    if armd_ok:
        keys = sorted(sens_by.keys())
        sens_v = [sens_by[k] for k in keys]
        m0_v = [next(c for c in contexts if (c["m"], c["tgt_id"]) == k)["m0_target"] for k in keys]
        g0_v = [next(c for c in contexts if (c["m"], c["tgt_id"]) == k)["g0_target"] for k in keys]
        lam_v = [next(c for c in contexts if (c["m"], c["tgt_id"]) == k)["lam_target"] for k in keys]
        gT_v = [next(c for c in contexts if (c["m"], c["tgt_id"]) == k)["gT_target"] for k in keys]
        for name, v in (("m0", m0_v), ("g0", g0_v), ("lam", lam_v), ("gT", gT_v)):
            rho, n = spearman(sens_v, v)
            corr["pooled"][name] = dict(rho=rho, n=n)
        for m in m_list:
            km = [k for k in keys if k[0] == m]
            sv = [sens_by[k] for k in km]
            row = {}
            for name, get in (("m0", "m0_target"), ("g0", "g0_target"), ("lam", "lam_target")):
                vv = [next(c for c in contexts if (c["m"], c["tgt_id"]) == k)[get] for k in km]
                rho, n = spearman(sv, vv)
                row[name] = dict(rho=rho, n=n)
            corr["per_m"][m] = row
        for name in ("m0", "g0", "lam"):
            corr["sign_pattern"][name] = " ".join(
                f"m={m}:{'+' if corr['per_m'][m][name]['rho'] > 0 else ('-' if corr['per_m'][m][name]['rho'] < 0 else '0')}"
                if math.isfinite(corr["per_m"][m][name]["rho"]) else f"m={m}:nan"
                for m in m_list)

    # ---- CLASS-ASYMMETRY test over the whole pool (the strongest, not small-n) ----
    c0_idx = torch.tensor([i for i, v in enumerate(pool_y.tolist()) if int(v) == 0])
    c1_idx = torch.tensor([i for i, v in enumerate(pool_y.tolist()) if int(v) == 1])
    cls = dict(n_c0=len(c0_idx), n_c1=len(c1_idx),
               mean_m0_c0=m0[c0_idx].mean().item(), mean_m0_c1=m0[c1_idx].mean().item(),
               std_m0_c0=m0[c0_idx].std().item(), std_m0_c1=m0[c1_idx].std().item(),
               mean_g0_c0=g0[c0_idx].mean().item(), mean_g0_c1=g0[c1_idx].mean().item(),
               std_g0_c0=g0[c0_idx].std().item(), std_g0_c1=g0[c1_idx].std().item(),
               armc_balanced_after=armc_after, armc_measured=armc_meas)

    # ---- 5) verdicts ----
    verdicts = {}
    if armd_ok:
        verdicts["P1_sens_anticorr_base_margin"] = _verdict(corr["pooled"]["m0"]["rho"], -1.0,
                                                            corr["pooled"]["m0"]["n"])
        verdicts["P2_sens_corr_base_gradnorm"] = _verdict(corr["pooled"]["g0"]["rho"], +1.0,
                                                          corr["pooled"]["g0"]["n"])
        verdicts["P3_sens_corr_lambda_proxy"] = _verdict(corr["pooled"]["lam"]["rho"], +1.0,
                                                         corr["pooled"]["lam"]["n"])
    margin_dir = cls["mean_m0_c1"] < cls["mean_m0_c0"]
    grad_dir = cls["mean_g0_c1"] > cls["mean_g0_c0"]
    lam_dir = armc_after["mean_lam_c1"] > armc_after["mean_lam_c0"]
    n_ok = sum([margin_dir, grad_dir, lam_dir])
    verdicts["P4_class_asymmetry"] = ("SUPPORTED" if n_ok == 3 else
                                      ("REFUTED" if n_ok == 0 else f"MIXED ({n_ok}/3 directions)"))
    verdicts["P4_detail"] = dict(class1_smaller_base_margin=bool(margin_dir),
                                 class1_larger_base_gradnorm=bool(grad_dir),
                                 class1_larger_lambda_balanced=bool(lam_dir))

    print("\n=== VERDICT BLOCK (support-vector / max-margin hypothesis) ===", flush=True)
    if armd_ok:
        print(f"P1 sens vs base-margin m0 (expect rho<0): pooled rho={corr['pooled']['m0']['rho']:+.3f} "
              f"(n={corr['pooled']['m0']['n']}, SMALL-n) per-m signs [{corr['sign_pattern']['m0']}] "
              f"-> {verdicts['P1_sens_anticorr_base_margin']}")
        print(f"P2 sens vs base-gradnorm g0 (expect rho>0): pooled rho={corr['pooled']['g0']['rho']:+.3f} "
              f"(n={corr['pooled']['g0']['n']}, SMALL-n) per-m signs [{corr['sign_pattern']['g0']}] "
              f"-> {verdicts['P2_sens_corr_base_gradnorm']}")
        print(f"P3 sens vs lambda-proxy (expect rho>0): pooled rho={corr['pooled']['lam']['rho']:+.3f} "
              f"(n={corr['pooled']['lam']['n']}, SMALL-n) per-m signs [{corr['sign_pattern']['lam']}] "
              f"-> {verdicts['P3_sens_corr_lambda_proxy']}")
    else:
        print("P1-P3: SKIPPED (arm-D summary unavailable or config mismatch).")
    print(f"P4 class asymmetry (class-1 measured "
          f"{'%.2fx' % armc_meas['ratio'] if armc_meas else '~3.3x'} louder): "
          f"mean m0 c1={cls['mean_m0_c1']:.3f} vs c0={cls['mean_m0_c0']:.3f} "
          f"({'c1 SMALLER' if margin_dir else 'c1 NOT smaller'}); "
          f"mean g0 c1={cls['mean_g0_c1']:.3e} vs c0={cls['mean_g0_c0']:.3e} "
          f"({'c1 LARGER' if grad_dir else 'c1 NOT larger'}); "
          f"mean lam(balanced) c1={armc_after['mean_lam_c1']:.3e} vs "
          f"c0={armc_after['mean_lam_c0']:.3e} ({'c1 LARGER' if lam_dir else 'c1 NOT larger'}) "
          f"-> {verdicts['P4_class_asymmetry']}", flush=True)

    # ---- save everything ----
    os.makedirs(RESULTS, exist_ok=True)
    out = dict(config=dict(N=N, m_list=m_list, n_targets=n_targets, lr=lr, T=T, rank=rank,
                           armc_m=args.armc_m, pool_n=pool_n, ref_seed=1000),
               pool=dict(digits=[int(d) for d in pool_digits], y=pool_y.tolist(),
                         m0=m0.tolist(), g0=g0.tolist(),
                         target_pool=target_pool, peer_pool=peer_pool, maj_pool=maj_pool,
                         spearman_m0_g0=rho_m0_g0),
               contexts=[{k: v for k, v in c.items() if k != "pool_idx"} for c in contexts],
               sens_measured={f"m{m}_t{t}": s for (m, t), s in sens_by.items()},
               correlations=corr, class_asymmetry=cls, verdicts=verdicts)
    with open(os.path.join(RESULTS, "margins.json"), "w") as f:
        json.dump(out, f, indent=2)
    torch.save(dict(m0=m0, g0=g0, pool_y=pool_y.cpu(), pool_digits=[int(d) for d in pool_digits],
                    contexts=[dict(m=c["m"], tgt_id=c["tgt_id"], t_pos=c["t_pos"],
                                   mT=torch.tensor(c["mT"]), lam=torch.tensor(c["lam"]),
                                   gT=torch.tensor(c["gT"])) for c in contexts],
                    armc=dict(mT=mTc, lam=lamc, gT=gTc, maj_pos=maj_pos, min_pos=min_pos),
                    summary=out),
               os.path.join(RESULTS, "margins.pth"))
    print(f"\nsaved {RESULTS}/margins.json + margins.pth", flush=True)


if __name__ == "__main__":
    main()
