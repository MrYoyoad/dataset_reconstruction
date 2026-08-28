#!/usr/bin/env python
"""
FULL-FT VALLEY COMPARISON — core wave (plan: notes/fullft_valley_comparison_plan.md,
v1.2 FINAL, unanimous audit pass). Arms C / D / E(_b0,_eps) / B2 + the ε-calibration.

MISSION: does the PARAMETERIZATION narrow the valley? Full-rank Δθ vs rank-8 BA under
matched fine-tuning from the same θ₀, same instrument (3-way whitened sensitivity),
same distance dial (similarity_ladder rungs, shared 6).

ARMS (plan §2, this module):
  calib — lr search per regime (S5 band max_bce∈[1e-4,1e-3]) + ε fixed-point calibration
          per regime (layer-0 trained reseed RMS matched to the LoRA r=8 reseed_noise,
          job 268959) + the 3-point ε-bracket {ε/3, ε, 3ε} on p0_noise (α≈−2±0.3 gate +
          curvature check; 2-point on p00/r_cross) + reseed-vs-reseed null uniformity
          (§4.1.3) + the arm-A mid-rung pre-check (S6c, CPU on job-268959 data).
  C     — FULL-RANK single layer (L0 only trainable), ε-perturb of L0.   [rank→∞ @ depth]
  D     — FULL FT all layers [PRIMARY], ε-perturb all layers, PER-LAYER readout (D4/S2).
  E_b0  — LoRA r=8, B0-reseed noise (the arm-A noise source) on the E rung subset.
  E_eps — LoRA r=8, B0 FIXED (seed 0), randomness moved to a θ₀-perturbation of L0 at
          ε-scale.  P4 noise-exchangeability control: E_b0 vs E_eps, same arm.
  B2    — §4.0 GATE: full-FT all layers under SGD MINIBATCH-ORDER noise (fixed θ₀, no ε,
          batch N/4, order seeded) on {p0_noise, mid/crossing rung, r_cross} (+p00 per
          §4.1.1).  Full-FT d*/s must agree ε↔SGD or the headline dies.

GAUGE (D2/Q5, audit-endorsed): headline Δθ = θ_T − θ₀_canonical, the ATTACKER-VISIBLE
gauge — INCLUDES the injected ε·ξ (part of Σ honestly, exactly as B0-randomness is part
of ΔW=BA). For the LoRA arms the released/measured object stays ΔW=BA only (the θ₀
perturbation in E_eps is NOT in the release; it randomizes the path, like B0).
TF7 conditionality (state on every cross-regime readout): LoRA Σ is pure path-noise;
full-FT Σ = path-noise + direct ε·ξ injection; ε-calibration matches MAGNITUDE, not
covariance STRUCTURE — cross-regime comparison is CONDITIONAL on P4 + §4.0-B2.

ε-NOISE (D2): θ₀ → θ₀ + ε·ξ per trainable layer, ξ ~ N(0,1)·std(θ₀,ℓ), seeded
(generator seed = eps_seed*1000 + ℓ, so ξ is exactly regenerable from the seed —
the ξ tensors are therefore NOT saved). ε·ξ cancels exactly at t=0 in the paired diff
v_j = Δθ(D,s_j) − Δθ(D′,s_j); through training it propagates differentially, leaving
an O(ε) residual — which is precisely what the α≈−2 linear-response gate checks.

DIAL (D3): 6 shared rungs {p00_identity, p0_noise, <mid>, r_nn, r_far, r_cross}
(EXACT similarity_ladder constructions, imported; default mid = p3_rot15, overridable
--mid_rung per the S6c arm-A pre-check, which calib prints). s(d) = sens/sens(r_cross)
with the normalizer-void gate p(r_cross)<0.05; d* = 0.1-crossing (log-log interp),
reported WITH the bracketing-rung interval + the {0.3, 0.03} robustness thresholds.
Same 2 targets as job 268959 (first n_targets class-1 slots of D, seed-42 set).

PRECISION CHOICE (D4 memory note): ALL training + the metric run in float64. Saved
Δθ/v stacks are cast to float32 ON DISK ONLY (v is computed in float64 first, so no
catastrophic cancellation is stored; per-entry float32 quantization ~1e-7 relative is
orders below the seed-noise scale). The B1 rescore must upcast float32→float64 before
calling the metric (metric input stays float64).

DESIGN LOCK: trainers swap ONLY the parameterization — forward via
arm_b_dilution.forward_logits (empty adapter ⇒ byte-identical forward), optimizer
plain SGD exactly as train_adapter; rung construction / banks / encoder / metric all
imported from similarity_ladder / whitened_metric, never re-implemented.

bsub-only (scripts/run_fullft_valley_wexac.sh + _part2). mnist/gelu/binary, N=16,
T=1000, K=50 headline. Saves per-arm .pth (metrics + per-layer Δθ/v stacks + images)
under results/fullft_valley/ + JSON summary + PNG grids (house output rules).
"""
import os
import json
import math
import argparse

import torch
import torch.nn.functional as F

from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.data_utils import get_finetuning_data
from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity
from experiments.dataset_sensitivity.arm_b_dilution import (forward_logits, draw_B0,
                                                            train_adapter)
from experiments.dataset_sensitivity.margin_vs_sensitivity import (layer0_grad_norms,
                                                                   spearman)
import experiments.dataset_sensitivity.similarity_ladder as sl

torch.set_default_dtype(torch.float64)

RESULTS = "/home/projects/galvardi/yoado/results/fullft_valley"
FIGURES = "/home/projects/galvardi/yoado/figures/fullft_valley"
SL_SUMMARY = "/home/projects/galvardi/yoado/results/similarity_ladder/similarity_ladder_summary.json"

N_FOLDS = 5
N_PERM = 500
P_MAX = 3
MEM_GATE = 1e-3                    # §4.1.4: every training gated at max_bce < 1e-3
MEM_BAND = (1e-4, 1e-3)            # S5: lr tuned into this band on the baseline
D_STAR_THRESH = (0.1, 0.3, 0.03)   # D3: headline threshold first, robustness band after
# Provenance fallback for the ε-calibration target: job 268959 (arm A, LoRA r=8,
# N=16/K=50/T=1000/lr=0.5), reseed_noise of the ΔW=BA layer-0 block, recorded in
# results/similarity_ladder/similarity_ladder_summary.json (checked 2026-08-28).
LORA_RESEED_NOISE_FALLBACK = 9.97044874151052
LORA_LR = 0.5                      # arm-A standard; verified in-band during calib
# S5 lr grid for the full-FT regimes (pre-registered; descending, 1 training each)
LR_GRID = (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)

E_RUNGS = ("p00_identity", "p0_noise", "r_far", "r_cross")
STAGE0_RUNGS = ("p00_identity", "p0_noise", "r_cross")
BRACKET_RUNGS = ("p00_identity", "p0_noise", "r_cross")
# B1 dimension-invariance coordinate-subset sizes (§4.0-B1: ~25k≈LoRA ambient → the
# full arm-D concat dim 1.785M; d* must be FLAT across the span toward LoRA's dimension)
B1_FRACTIONS = (25000, 100000, 450000, 1785000)

PREDICTIONS = {
    "calib": ("PRE-REGISTERED (D2/§4.1.2): log-log raw-sens-vs-ε on p0_noise has slope "
              "α = −2 ± 0.3 with no significant curvature (mid point within factor 1.5 "
              "of the endpoint chord); mini-d* within 2x across the bracket; reseed-vs-"
              "reseed null p-values ~uniform with qeff≈0."),
    "C": ("PRE-REGISTERED (P1b, A→C step): rank→∞ at fixed depth NARROWS the valley — "
          "d*(C) ≤ d*(A); this step crosses the r=N boundary and is labeled as such, "
          "nothing more (NO NTK-equivalence claim, TB3)."),
    "D": ("PRE-REGISTERED (P1, TB1-asymmetric): d*_full(D) < d*_LoRA(A); "
          "s_full(p0_noise) ≥ 3x s_LoRA(p0_noise). P2: per-layer L0 rises at smaller d "
          "than L1/L2 — stated on the NUMERATOR ||dmu_l|| (S2); a denominator-driven "
          "ordering does not count. HEADLINE READ ONLY AFTER §4.0 B1+B2 PASS."),
    "E_b0": ("PRE-REGISTERED (P4 control, B0 half): LoRA r=8 profile under B0-reseed "
             "noise on the E rung subset — the reference half of the exchangeability "
             "pair (compare with E_eps; subject to the §4.1.7 power gate)."),
    "E_eps": ("PRE-REGISTERED (P4 control, ε half): LoRA r=8 normalized profile under "
              "ε-noise ≈ under B0-noise (per-rung s within CIs). Mismatch ⇒ the noise "
              "analogue is NOT exchangeable ⇒ cross-regime comparison downgrades to "
              "qualitative and D2 must be re-based."),
    "B2": ("PRE-REGISTERED (§4.0-B2 GATE): full-FT d*/s under SGD minibatch-order noise "
           "CONSISTENT with ε-noise (arm D) at {p0_noise, mid, r_cross} — else "
           "'narrower valley' is a synthetic-noise artifact and the headline dies."),
    "F": ("PRE-REGISTERED (§2.1, P5): LEAVE-ONE-OUT removal footprint "
          "v_j = Δθ(D,s_j) − Δθ(D\\{i},s_j), BOTH regimes. HEADLINE P5b (S3-robust, "
          "offset-immune): per-target LOO footprints RANK-correlate across regimes "
          "(spearman ≥ +0.5 over 6–8 targets). P5a (both detectable) / P5d (normalized "
          "footprint larger in full) are DESCRIPTIVE with the N→N−1 offset caveat "
          "(removal shifts set size AND class balance 8→7 — a constant shared by all "
          "class-1 targets, cancelled by the rank comparison). Free g0 piggyback: does "
          "the base-gradient predictor (LoRA ρ=+0.78) transfer to the full regime?"),
    "B1": ("PRE-REGISTERED (§4.0-B1 GATE): dimension-invariance of d*. CPU rescore of "
           "arm-D's SAVED concat Δθ stacks at coordinate fractions {25k,100k,450k,1.79M} "
           "(fixed-seed nested random column masks). d* must be FLAT across the fractions "
           "spanning toward LoRA's 25k ambient dim — else the s(d)-normalization does NOT "
           "cancel the ambient-dimension step and the ladder trend must be dimension-"
           "corrected (§II rule 3) before the headline. NO retraining."),
}


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _flat(dtheta, sel):
    """Flatten + concat the selected layer blocks of a per-layer Δθ dict -> 1-D."""
    return torch.cat([dtheta[l].reshape(-1) for l in sel])


def _rms_spread(vecs):
    """RMS spread around the mean: sqrt(mean_j ||v_j - mean||^2). Matches the
    similarity_ladder reseed_noise formula exactly (the calibration target)."""
    stk = torch.stack([v.reshape(-1) for v in vecs])
    return ((stk - stk.mean(0)).norm(dim=1) ** 2).mean().sqrt().item()


def _readouts_for(layers):
    """Readout name -> layer-selection tuple. Single-layer arms expose 'concat' only
    (== their one block); arm D adds the per-layer decomposition (D4)."""
    if len(layers) == 1:
        return {"concat": tuple(layers)}
    d = {f"L{l}": (l,) for l in layers}
    d["concat"] = tuple(layers)
    return d


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return _jsonable(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def _save_grid(all_vis, tag):
    """similarity_ladder.save_grid, redirected into figures/fullft_valley/."""
    old = sl.FIGURES
    try:
        sl.FIGURES = FIGURES
        sl.save_grid(all_vis, tag=tag)
    finally:
        sl.FIGURES = old


# --------------------------------------------------------------------------- #
# trainers (the new piece — everything else is imported)
# --------------------------------------------------------------------------- #
def _draw_eps(frozen, layers, eps_scale, eps_seed, device):
    """Per-layer ε-noise: ε·std(θ₀,ℓ)·ξ, ξ ~ N(0,1), CPU-generator seeded with
    eps_seed*1000 + ℓ (deterministic, regenerable — ξ is never stored)."""
    out = {}
    for l in layers:
        g = torch.Generator(device="cpu").manual_seed(int(eps_seed) * 1000 + int(l))
        xi = torch.randn(frozen[l].shape, generator=g, dtype=torch.float64)
        out[l] = (eps_scale * frozen[l].std().item()) * xi.to(device)
    return out


def train_full(frozen, b0, x0, y, lr, T, act, eps_seed, eps_scale,
               trainable_layers=(0, 1, 2), batch_seed=None, batch_size=None):
    """Full-parameter SGD fine-tune from θ₀ + ε·ξ(eps_seed) (ε on TRAINABLE layers
    only; weights only, biases frozen — mirroring LoRA which trains only W).
    Full-batch BCE, T steps, float64. batch_seed != None switches to the B2 mode:
    NO ε (pass eps_scale=0), seeded minibatch-order SGD (fresh permutation per epoch,
    batches of batch_size, still T parameter updates total).

    Returns (dtheta, max_bce): dtheta = {l: θ_T[l] − θ₀[l]} over trainable layers —
    the ATTACKER-VISIBLE gauge (INCLUDES the injected ε·ξ, per D2/Q5)."""
    n_layers = len(frozen)
    eps = (_draw_eps(frozen, trainable_layers, eps_scale, eps_seed, x0.device)
           if eps_scale > 0 else {})
    W = {}
    for l in range(n_layers):
        if l in trainable_layers:
            w = frozen[l] + eps[l] if l in eps else frozen[l].clone()
            W[l] = w.detach().requires_grad_(True)
        else:
            W[l] = frozen[l]
    opt = torch.optim.SGD([W[l] for l in trainable_layers], lr=lr)  # plain SGD, as train_adapter
    N = x0.shape[0]
    g_batch = perm = None
    if batch_seed is not None:
        assert batch_size and N % batch_size == 0, "B2 needs batch_size | N"
        g_batch = torch.Generator(device="cpu").manual_seed(int(batch_seed))
        n_b = N // batch_size
    for t in range(T):
        if batch_seed is None:
            xb, yb = x0, y
        else:
            if t % n_b == 0:
                perm = torch.randperm(N, generator=g_batch).to(x0.device)
            idx = perm[(t % n_b) * batch_size:(t % n_b + 1) * batch_size]
            xb, yb = x0[idx], y[idx]
        # forward: forward_logits with an EMPTY adapter == the plain MLP forward
        out = forward_logits(xb, W, b0, {}, {}, act, target_layers=())
        loss = F.binary_cross_entropy_with_logits(out.view(-1), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = forward_logits(x0, W, b0, {}, {}, act, target_layers=()).view(-1)
        max_bce = F.binary_cross_entropy_with_logits(out, y, reduction="none").max().item()
        dtheta = {l: (W[l] - frozen[l]).detach() for l in trainable_layers}
    return dtheta, max_bce


def train_lora_eps(frozen, b0, x0, y, lr, T, act, rank, eps_seed, eps_scale):
    """Arm-E ε mode: standard LoRA (train_adapter, imported) with FIXED B0 (seed 0)
    and the randomness moved to a θ₀-perturbation of layer 0 at ε-scale.
    Measured object stays ΔW = B·A (the release; ε·ξ is NOT in it — it randomizes
    the training path, structurally like B0-reseed does)."""
    eps = _draw_eps(frozen, (0,), eps_scale, eps_seed, x0.device)
    frozen_p = dict(frozen)
    frozen_p[0] = frozen[0] + eps[0]
    out_f = frozen[0].shape[0]
    B0_fixed = draw_B0(0, out_f, rank, x0.device)
    _, _, max_bce, dW = train_adapter(frozen_p, b0, B0_fixed, x0, y, lr, T, act, rank)
    return {0: dW}, max_bce


def make_train_fn(regime, ctx, lr, eps_scale, batch_size=None):
    """Return (train_fn(seed, x0) -> (dtheta_dict, max_bce), trainable_layers).
    The seed is the regime's noise seed: ε-draw (C/D/E_eps), B0-draw (E_b0),
    minibatch-order (B2)."""
    frozen, b0, y, act, rank = ctx["frozen"], ctx["b0"], ctx["y_ft"], ctx["act"], ctx["rank"]
    T = ctx["T"]
    all_layers = tuple(range(len(frozen)))
    if regime == "C":
        fn = lambda s, x0: train_full(frozen, b0, x0, y, lr, T, act, s, eps_scale,
                                      trainable_layers=(0,))
        return fn, (0,)
    if regime == "D":
        fn = lambda s, x0: train_full(frozen, b0, x0, y, lr, T, act, s, eps_scale,
                                      trainable_layers=all_layers)
        return fn, all_layers
    if regime == "B2":
        fn = lambda s, x0: train_full(frozen, b0, x0, y, lr, T, act, 0, 0.0,
                                      trainable_layers=all_layers,
                                      batch_seed=s, batch_size=batch_size)
        return fn, all_layers
    if regime == "E_b0":
        out_f = frozen[0].shape[0]
        def fn(s, x0):
            _, _, mbce, dW = train_adapter(frozen, b0, draw_B0(s, out_f, rank, x0.device),
                                           x0, y, lr, T, act, rank)
            return {0: dW}, mbce
        return fn, (0,)
    if regime == "E_eps":
        fn = lambda s, x0: train_lora_eps(frozen, b0, x0, y, lr, T, act, rank, s, eps_scale)
        return fn, (0,)
    raise ValueError(f"unknown regime {regime}")


# --------------------------------------------------------------------------- #
# shared setup (arm-B/D construction — same D, same θ₀, same targets as job 268959)
# --------------------------------------------------------------------------- #
def setup_ctx(N, T, rank, device):
    n_per_class = N // 2
    x_ft, y_ft, digits, indices = get_finetuning_data(n_per_class, seed=42,
                                                      device=device, dataset="mnist")
    x_ft = x_ft.to(torch.float64)
    y_ft = y_ft.to(torch.float64)
    _, frozen, b0, _, ds_mean = _honest_target(x_ft, y_ft, T, rank, "gelu", LORA_LR,
                                               device, "mnist", num_classes=2)
    frozen = {l: w.detach() for l, w in frozen.items()}
    return dict(x_ft=x_ft, y_ft=y_ft, digits=digits, indices=indices,
                frozen=frozen, b0=b0.detach(), ds_mean=ds_mean,
                x0=(x_ft - ds_mean), act=make_activation("gelu"),
                T=T, rank=rank, N=N, device=device)


def target_positions(ctx, n_targets):
    """Same target selection as similarity_ladder (job 268959): first n_targets
    class-1 slots of D."""
    y = ctx["y_ft"]
    c1 = [i for i in range(ctx["N"]) if int(y[i].item()) == 1]
    assert len(c1) >= n_targets, f"need {n_targets} class-1 targets, have {len(c1)}"
    return c1[:n_targets]


def build_rung_list(ctx, t_pos, encoder, bank_n, rung_filter):
    """Rung construction via the imported similarity_ladder builders (parametric +
    encoder-retrieved; identical seeds/magnitudes ⇒ IDENTICAL T' images to job 268959).
    Returns [(name, T'[1,28,28] cpu f64, d_encoder, d_pixel)]."""
    device = ctx["device"]
    T_img = ctx["x_ft"][t_pos].detach().cpu()
    digit_T = int(ctx["digits"][t_pos])
    same_bank, cross_bank = sl.build_banks(digit_T, ctx["indices"], bank_n)
    ret, e_T = sl.retrieved_rungs(encoder, T_img, same_bank, cross_bank, device)
    par = sl.parametric_rungs(T_img)                       # [(name, T')] x6
    par_stack = torch.stack([tp for _, tp in par]).squeeze(1)
    d_par = sl.cos_dist(e_T, sl.embed_images(encoder, par_stack, device))
    rungs = ([(nm, tp, float(d_par[i])) for i, (nm, tp) in enumerate(par)] + ret)
    if rung_filter is not None:
        rungs = [r for r in rungs if r[0] in rung_filter]
    return [(nm, tp, de, (T_img - tp).norm().item()) for nm, tp, de in rungs], T_img, digit_T


# --------------------------------------------------------------------------- #
# baseline + rung measurement (the ladder loop, generalized to per-layer Δθ dicts)
# --------------------------------------------------------------------------- #
def build_baseline(train_fn, x0, K, layers, seed0=1000):
    """K reseed trainings Δθ(D, seed_j) — trained ONCE, shared by all rungs/targets.
    Gates (§4.1.4): non-finite AND non-memorized (max_bce ≥ 1e-3) draws dropped, counted."""
    base, mbces = {}, {}
    drop_nan = drop_mem = 0
    for j in range(K):
        s = seed0 + j
        dt, mbce = train_fn(s, x0)
        if not all(torch.isfinite(dt[l]).all() for l in layers):
            drop_nan += 1
            continue
        if not (mbce < MEM_GATE):
            drop_mem += 1
            continue
        base[s] = dt
        mbces[s] = mbce
    assert len(base) >= 2 * N_FOLDS, \
        f"only {len(base)}/{K} usable baseline draws (nan={drop_nan}, mem={drop_mem}); metric starved"
    noise = {name: _rms_spread([_flat(base[s], sel) for s in base])
             for name, sel in _readouts_for(layers).items()}
    return base, mbces, drop_nan, drop_mem, noise


def measure_rung(ctx, t_pos, T_prime, base, train_fn, layers, seed_tag):
    """Swap D[t_pos] -> T_prime; paired per-seed diff v_j = Δθ(D,s_j) − Δθ(D',s_j)
    (same noise seed ⇒ the injected ε·ξ / B0 / batch-order noise CANCELS at t=0);
    whitened_sensitivity per readout (per-layer + concat for arm D). Also the S2
    numerator ||dmu_l|| and per-layer denominator noise RMS, reported SEPARATELY."""
    x_sw = ctx["x_ft"].clone()
    x_sw[t_pos] = T_prime.to(x_sw.device, torch.float64)
    x0_sw = x_sw - ctx["ds_mean"]

    pairs = []                                   # (dtheta_ref, dtheta_swap), aligned seeds
    drop_nan = drop_mem = 0
    for s, dt_ref in base.items():
        dt_sw, mbce = train_fn(s, x0_sw)
        if not all(torch.isfinite(dt_sw[l]).all() for l in layers):
            drop_nan += 1
            continue
        if not (mbce < MEM_GATE):
            drop_mem += 1
            continue
        pairs.append((dt_ref, dt_sw))

    out, stacks = {}, {}
    for ri, (name, sel) in enumerate(_readouts_for(layers).items()):
        v_list = [_flat(a, sel) - _flat(b, sel) for a, b in pairs]   # float64, in-memory
        r_list = [_flat(a, sel) for a, _ in pairs]
        res = dict(sensitivity=float("nan"), pvalue=float("nan"), d2_obs=float("nan"),
                   qeff=float("nan"))
        dmu_norm = noise_rms = float("nan")
        if v_list:
            dmu_norm = torch.stack(v_list).mean(0).norm().item()     # S2 numerator
            noise_rms = _rms_spread(r_list)                          # S2 denominator scale
        if len(v_list) >= 2 * N_FOLDS:
            ws = whitened_sensitivity([v.cpu() for v in v_list], [r.cpu() for r in r_list],
                                      n_folds=N_FOLDS, p_max=P_MAX, n_perm=N_PERM,
                                      seed=int(seed_tag) * 10 + ri)
            res = dict(sensitivity=ws["sensitivity"], pvalue=ws["pvalue"],
                       d2_obs=ws["d2_obs"], qeff=ws["qeff_count"])
            assert math.isfinite(ws["d2_obs"]), f"seed_tag={seed_tag} {name}: d2_obs non-finite"
        out[name] = dict(**res, dmu_norm=dmu_norm, noise_rms=noise_rms)
    # per-layer v stacks for the B1 multi-fraction CPU rescore (float32 ON DISK ONLY;
    # v was computed in float64 above — the rescore must upcast before the metric)
    if pairs:
        for l in layers:
            stacks[f"v_L{l}"] = torch.stack([(a[l] - b[l])
                                             for a, b in pairs]).to(torch.float32).cpu()
    out_meta = dict(n_pairs=len(pairs), dropped_nan=drop_nan, dropped_mem=drop_mem)
    return out, out_meta, stacks


# --------------------------------------------------------------------------- #
# valley-width functional (D3)
# --------------------------------------------------------------------------- #
def d_star(profile, thresh):
    """profile: [(rung, d_pixel, s)] excluding p00 — sorted by d_pixel. Returns the
    0.1-crossing (log-log interp) + the bracketing-rung interval (S6c: reported as an
    INTERVAL whenever only one rung pair brackets the crossing) or a censored bound."""
    pts = sorted([(d, max(s, 1e-9), r) for r, d, s in profile
                  if r != "p00_identity" and math.isfinite(s) and d > 0])
    if len(pts) < 2:
        return dict(point=float("nan"), interval=None, censored="insufficient rungs")
    brackets = []
    for (d0, s0, r0), (d1, s1, r1) in zip(pts[:-1], pts[1:]):
        if (s0 - thresh) * (s1 - thresh) < 0:
            f = (math.log(thresh) - math.log(s0)) / (math.log(s1) - math.log(s0))
            brackets.append(dict(point=math.exp(math.log(d0) + f * (math.log(d1) - math.log(d0))),
                                 lo=d0, hi=d1, rungs=[r0, r1]))
    if not brackets:
        if all(s > thresh for _, s, _ in pts):
            return dict(point=float("nan"), interval=[0.0, pts[0][0]],
                        censored=f"all s>thresh: d* < {pts[0][0]:.3f} ({pts[0][2]})")
        if all(s < thresh for _, s, _ in pts):
            return dict(point=float("nan"), interval=[pts[-1][0], float("inf")],
                        censored=f"all s<thresh: d* > {pts[-1][0]:.3f} ({pts[-1][2]})")
        return dict(point=float("nan"), interval=None, censored="non-monotone, no bracket")
    b = brackets[0]                                # first crossing from below
    return dict(point=b["point"], interval=[b["lo"], b["hi"]], bracket_rungs=b["rungs"],
                n_bracket_pairs=len(brackets), censored=None,
                interval_only=(len(brackets) == 1))


def profile_and_dstar(per_rung, readout):
    """s(d) = sens/sens(r_cross) for one readout + d* at all thresholds + gates."""
    sens = {r["rung"]: r["readouts"][readout]["sensitivity"] for r in per_rung}
    pval = {r["rung"]: r["readouts"][readout]["pvalue"] for r in per_rung}
    dpix = {r["rung"]: r["d_pixel"] for r in per_rung}
    void = not (math.isfinite(pval.get("r_cross", float("nan")))
                and pval["r_cross"] < 0.05)        # D3 normalizer-void gate
    s_ref = sens.get("r_cross", float("nan"))
    prof = [(rg, dpix[rg], (sens[rg] / s_ref) if (s_ref and math.isfinite(s_ref)
                                                  and s_ref > 0) else float("nan"))
            for rg in sens]
    ds = {str(th): d_star(prof, th) for th in D_STAR_THRESH}
    # mini-d*: 2-point p0_noise <-> r_cross log-interp (the {K,2K} comparison object)
    mini = d_star([p for p in prof if p[0] in ("p0_noise", "r_cross")], 0.1)
    return dict(normalizer_void=void, s_profile={r: s for r, _, s in prof},
                s_ref_sens=s_ref, d_star=ds, d_star_mini=mini)


# --------------------------------------------------------------------------- #
# the dial on one arm
# --------------------------------------------------------------------------- #
def run_dial_arm(args, regime, lr, eps_scale, rung_filter, tag):
    dev = args.device
    print(f"\n{'=' * 74}\nARM {regime}  (lr={lr}, eps_scale={eps_scale}, K={args.K}, "
          f"T={args.T}, N={args.N}, rungs={sorted(rung_filter)}, tag='{tag}')\n{'=' * 74}",
          flush=True)
    print(PREDICTIONS[regime], flush=True)
    if regime in ("C", "D", "E_eps"):
        print("[TF7] full-FT/ε Σ = path-noise + direct ε·ξ injection; LoRA Σ is pure "
              "path-noise. ε-calibration matches MAGNITUDE not covariance STRUCTURE — "
              "cross-regime readouts are CONDITIONAL on P4 + §4.0-B2.", flush=True)

    ctx = setup_ctx(args.N, args.T, args.rank, dev)
    batch_size = (args.N // 4) if regime == "B2" else None
    train_fn, layers = make_train_fn(regime, ctx, lr, eps_scale, batch_size=batch_size)

    base, mbces, b_nan, b_mem, base_noise = build_baseline(train_fn, ctx["x0"], args.K, layers)
    mb = sorted(mbces.values())
    in_band = MEM_BAND[0] <= mb[-1] < MEM_BAND[1]
    print(f"[baseline] {len(base)}/{args.K} usable (nan={b_nan}, mem={b_mem}); "
          f"achieved max_bce: min={mb[0]:.2e} med={mb[len(mb) // 2]:.2e} max={mb[-1]:.2e} "
          f"(S5 band [1e-4,1e-3): worst-seed in-band={in_band}); "
          f"per-readout reseed RMS: " +
          " ".join(f"{k}={v:.4f}" for k, v in base_noise.items()), flush=True)
    if not in_band:
        print("WARNING (S5): worst-seed max_bce outside the pre-registered memorization "
              "band — report alongside every headline; consider re-tuning lr.", flush=True)

    encoder, enc_name = sl.load_encoder(dev)
    os.makedirs(RESULTS, exist_ok=True)

    tgt_pos = target_positions(ctx, args.n_targets)
    print(f"[setup] targets at positions {tgt_pos} "
          f"(digits {[int(ctx['digits'][p]) for p in tgt_pos]}), encoder={enc_name}", flush=True)

    all_targets, all_vis = [], []
    for tgt_id, t_pos in enumerate(tgt_pos):
        rungs, T_img, digit_T = build_rung_list(ctx, t_pos, encoder, args.bank, rung_filter)
        print(f"\n--- target {tgt_id}: D position {t_pos}, digit {digit_T} ---", flush=True)
        per_rung, rung_stacks = [], {}
        for i, (nm, tp, de, dp) in enumerate(rungs):
            res, meta, stacks = measure_rung(ctx, t_pos, tp, base, train_fn, layers,
                                             seed_tag=100 * (tgt_id + 1) + i)
            row = dict(rung=nm, d_pixel=dp, d_encoder=de, readouts=res, **meta)
            per_rung.append(row)
            rung_stacks[nm] = stacks
            c = res["concat"]
            print(f"[{regime} t{tgt_id} {nm:>12}] d_pix={dp:.3f} d_enc={de:.4f} "
                  f"sens={c['sensitivity']:.4g} p={c['pvalue']:.3f} qeff={c['qeff']} "
                  f"|dmu|={c['dmu_norm']:.4g} noise={c['noise_rms']:.4g} "
                  f"(pairs={meta['n_pairs']}, nan={meta['dropped_nan']}, "
                  f"mem={meta['dropped_mem']})", flush=True)
            if len(layers) > 1:
                for l in layers:
                    r = res[f"L{l}"]
                    print(f"      L{l}: sens={r['sensitivity']:.4g} p={r['pvalue']:.3f} "
                          f"|dmu|={r['dmu_norm']:.4g} noise={r['noise_rms']:.4g}", flush=True)

        # §4.1.1 artifact-kill gate on the d=0 rung
        p00 = next((r for r in per_rung if r["rung"] == "p00_identity"), None)
        p00_ok = None
        if p00 is not None:
            c = p00["readouts"]["concat"]
            p00_ok = bool((not math.isfinite(c["pvalue"])) or c["pvalue"] > 0.05)
            if not p00_ok:
                print(f"*** ARTIFACT-KILL WARNING (§4.1.1): p00_identity significantly "
                      f"nonzero (sens={c['sensitivity']:.4g}, p={c['pvalue']:.4f}) — the "
                      f"ENTIRE dial for regime {regime} is suspect. ***", flush=True)
            else:
                print(f"[gate §4.1.1] p00_identity ~0 (sens={c['sensitivity']:.4g}, "
                      f"p={c['pvalue']:.3f}) — PASS", flush=True)

        prof = {name: profile_and_dstar(per_rung, name) for name in _readouts_for(layers)}
        for name, pr in prof.items():
            if pr["normalizer_void"]:
                print(f"*** NORMALIZER-VOID (D3): p(r_cross) >= 0.05 for readout {name} — "
                      f"s(d) undefined, arm {regime} VOID for this readout. ***", flush=True)
        tgt_res = dict(tgt_id=tgt_id, t_pos=int(t_pos), digit=digit_T,
                       per_rung=per_rung, profiles=prof, p00_pass=p00_ok)
        all_targets.append(tgt_res)

        # save: images + metrics + per-layer v stacks (float32 disk; see module docstring)
        tp_stack = torch.stack([tp for _, tp, _, _ in rungs])
        torch.save(dict(T_img=T_img, T_prime_stack=tp_stack,
                        rung_names=[r[0] for r in rungs],
                        d_pixel=[r[3] for r in rungs], d_encoder=[r[2] for r in rungs],
                        sensitivity=[r["readouts"]["concat"]["sensitivity"] for r in per_rung],
                        digit=digit_T, t_pos=int(t_pos), metrics=_jsonable(tgt_res),
                        v_stacks_f32={nm: rung_stacks[nm] for nm in rung_stacks},
                        precision_note="stacks are float32 on disk; upcast to float64 "
                                       "before any metric call (B1 rescore)"),
                   os.path.join(RESULTS, f"{regime}{tag}_t{tgt_id}.pth"))
        all_vis.append(dict(T_img=T_img, T_prime_stack=tp_stack,
                            rung_names=[r[0] for r in rungs],
                            d_encoder=[r[2] for r in rungs],
                            sensitivity=[r["readouts"]["concat"]["sensitivity"] for r in per_rung],
                            digit=digit_T))

    # baseline Δθ stack (float32 disk) — the B1 rescore's reseed/noise ensemble
    torch.save(dict(seeds=sorted(base.keys()),
                    dtheta_f32={l: torch.stack([base[s][l] for s in sorted(base)]
                                               ).to(torch.float32).cpu() for l in layers},
                    mbces=mbces, layers=list(layers), lr=lr, eps_scale=eps_scale,
                    regime=regime, K=args.K, T=args.T, N=args.N,
                    eps_seed_convention="generator seed = eps_seed*1000 + layer; "
                                        "eps = eps_scale*std(theta0_l)*xi",
                    precision_note="float32 on disk; upcast to float64 before the metric"),
               os.path.join(RESULTS, f"{regime}{tag}_baseline.pth"))
    _save_grid(all_vis, tag=f"_fv_{regime}{tag}")

    summary = dict(arm=regime, tag=tag, K=args.K, N=args.N, T=args.T, rank=args.rank,
                   lr=lr, eps_scale=eps_scale, batch_size=batch_size,
                   layers=list(layers), rungs=sorted(rung_filter), encoder=enc_name,
                   gauge="attacker-visible dtheta incl eps*xi (full-FT) / dW=BA (LoRA)",
                   baseline=dict(usable=len(base), dropped_nan=b_nan, dropped_mem=b_mem,
                                 max_bce_min=mb[0], max_bce_max=mb[-1], in_band=in_band,
                                 reseed_rms=base_noise),
                   targets=[_jsonable(t) for t in all_targets])
    with open(os.path.join(RESULTS, f"{regime}{tag}_summary.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"\nsaved {RESULTS}/{regime}{tag}_summary.json", flush=True)
    return summary, all_targets


# --------------------------------------------------------------------------- #
# arm F — leave-one-out weight footprint (§2.1), full-all vs LoRA r=8 + g0 piggyback
# --------------------------------------------------------------------------- #
def _readouts_from_pairs(pairs, layers, seed_tag, save_stacks=False):
    """The measure_rung readout loop, factored for the F removal contrast (measure_rung
    itself is left untouched — design lock). pairs = [(Δθ_ref, Δθ_alt)] aligned per seed;
    per readout: whitened sensitivity + the S2 numerator ||dmu|| and denominator RMS."""
    out, stacks = {}, {}
    for ri, (name, sel) in enumerate(_readouts_for(layers).items()):
        v_list = [_flat(a, sel) - _flat(b, sel) for a, b in pairs]
        r_list = [_flat(a, sel) for a, _ in pairs]
        res = dict(sensitivity=float("nan"), pvalue=float("nan"), d2_obs=float("nan"),
                   qeff=float("nan"))
        dmu_norm = noise_rms = float("nan")
        if v_list:
            dmu_norm = torch.stack(v_list).mean(0).norm().item()
            noise_rms = _rms_spread(r_list)
        if len(v_list) >= 2 * N_FOLDS:
            ws = whitened_sensitivity([v.cpu() for v in v_list], [r.cpu() for r in r_list],
                                      n_folds=N_FOLDS, p_max=P_MAX, n_perm=N_PERM,
                                      seed=int(seed_tag) * 10 + ri)
            res = dict(sensitivity=ws["sensitivity"], pvalue=ws["pvalue"],
                       d2_obs=ws["d2_obs"], qeff=ws["qeff_count"])
        out[name] = dict(**res, dmu_norm=dmu_norm, noise_rms=noise_rms)
    if save_stacks and pairs:
        for l in layers:
            stacks[f"v_L{l}"] = torch.stack([(a[l] - b[l])
                                             for a, b in pairs]).to(torch.float32).cpu()
    return out, stacks


def measure_removal(base, loo_train, layers, x0_loo, y_loo, seed_tag, save_stacks=True):
    """LOO contrast: v_j = Δθ(D, s_j) − Δθ(D\\{i}, s_j) — SAME noise seed s_j on both
    sides (the injected ε·ξ / B0 draw CANCELS at t=0, exactly as in the swap arms).
    loo_train(s, x0, y) trains on the REDUCED set D\\{i} (size N−1, its own y)."""
    pairs = []
    drop_nan = drop_mem = 0
    for s, dt_ref in base.items():
        dt_loo, mbce = loo_train(s, x0_loo, y_loo)
        if not all(torch.isfinite(dt_loo[l]).all() for l in layers):
            drop_nan += 1
            continue
        if not (mbce < MEM_GATE):
            drop_mem += 1
            continue
        pairs.append((dt_ref, dt_loo))
    out, stacks = _readouts_from_pairs(pairs, layers, seed_tag, save_stacks)
    meta = dict(n_pairs=len(pairs), dropped_nan=drop_nan, dropped_mem=drop_mem)
    return out, meta, stacks


def run_loo_arm(args, tag):
    """Arm F (§2.1): REMOVE image i entirely (D → D\\{i}, size N−1) and measure the
    weight-space footprint in BOTH regimes (full-all-layers via train_full + ε-noise,
    LoRA r=8 via the E_b0/train_adapter path + B0-noise). one contrast per target,
    args.n_targets class-1 targets. Headline P5b = cross-regime per-target RANK
    correlation (offset-immune). Free g0 piggyback at θ₀."""
    dev = args.device
    print(f"\n{'=' * 74}\nARM F — leave-one-out (K={args.K}, T={args.T}, N={args.N}, "
          f"n_targets={args.n_targets}, tag='{tag}')\n{'=' * 74}", flush=True)
    print(PREDICTIONS["F"], flush=True)
    print("[TF7] full-FT/ε Σ carries the direct ε·ξ injection; the cross-regime P5b is a "
          "RANK statement (offset-immune) — the ONLY headline; P5a/P5d descriptive.",
          flush=True)
    ctx = setup_ctx(args.N, args.T, args.rank, dev)
    lr_full, eps = resolve_config(args, "D")            # full regime = arm-D lr + ε (frozen in calib)
    frozen, b0, y_full = ctx["frozen"], ctx["b0"], ctx["y_ft"]
    act, rank, T = ctx["act"], ctx["rank"], ctx["T"]
    all_layers = tuple(range(len(frozen)))
    out_f = frozen[0].shape[0]

    # shared full-D reseed ensembles (both regimes), reused across every removal target
    fn_full, layers_full = make_train_fn("D", ctx, lr_full, eps)
    fn_lora, layers_lora = make_train_fn("E_b0", ctx, LORA_LR, 0.0)
    base_full, mb_full, fn_nan, fn_mem, noise_full = build_baseline(fn_full, ctx["x0"],
                                                                    args.K, layers_full)
    base_lora, mb_lora, ln_nan, ln_mem, noise_lora = build_baseline(fn_lora, ctx["x0"],
                                                                    args.K, layers_lora)
    print(f"[baseline] full {len(base_full)}/{args.K} (nan={fn_nan},mem={fn_mem}); "
          f"LoRA {len(base_lora)}/{args.K} (nan={ln_nan},mem={ln_mem})", flush=True)

    # LOO trainers: same maps as the baselines but on the REDUCED set (its own y)
    def loo_full(s, x0, y):
        return train_full(frozen, b0, x0, y, lr_full, T, act, s, eps,
                          trainable_layers=all_layers)

    def loo_lora(s, x0, y):
        _, _, mbce, dW = train_adapter(frozen, b0, draw_B0(s, out_f, rank, x0.device),
                                       x0, y, LORA_LR, T, act, rank)
        return {0: dW}, mbce

    tgt = target_positions(ctx, args.n_targets)         # class-1 only (balance shift is a const)
    g0 = layer0_grad_norms(ctx["x0"], y_full, frozen, b0, act)   # free g0 piggyback @ θ₀
    print(f"[setup] LOO targets (removed positions) {tgt} "
          f"(digits {[int(ctx['digits'][i]) for i in tgt]})", flush=True)
    os.makedirs(RESULTS, exist_ok=True)

    per_target = []
    for tid, i in enumerate(tgt):
        keep = [j for j in range(args.N) if j != i]
        x0_loo = ctx["x0"][keep]
        y_loo = y_full[keep]
        out_full, meta_full, st_full = measure_removal(base_full, loo_full, layers_full,
                                                       x0_loo, y_loo, seed_tag=700 * (tid + 1))
        out_lora, meta_lora, st_lora = measure_removal(base_lora, loo_lora, layers_lora,
                                                       x0_loo, y_loo, seed_tag=700 * (tid + 1) + 7)
        row = dict(tgt_id=tid, removed_idx=int(i), digit=int(ctx["digits"][i]),
                   g0=g0[i].item(), full=out_full, lora=out_lora,
                   meta_full=meta_full, meta_lora=meta_lora)
        per_target.append(row)
        torch.save(dict(removed_idx=int(i), digit=int(ctx["digits"][i]), g0=g0[i].item(),
                        full_v_f32=st_full, lora_v_f32=st_lora,
                        readouts_full=_jsonable(out_full), readouts_lora=_jsonable(out_lora),
                        layers_full=list(layers_full),
                        precision_note="v stacks float32 on disk; upcast before any metric"),
                   os.path.join(RESULTS, f"F{tag}_t{tid}.pth"))
        cf, cl = out_full["concat"], out_lora["concat"]
        print(f"[F t{tid} rm={i} d{row['digit']}] full: sens={cf['sensitivity']:.4g} "
              f"p={cf['pvalue']:.3f} |dmu|={cf['dmu_norm']:.4g}  LoRA: "
              f"sens={cl['sensitivity']:.4g} p={cl['pvalue']:.3f}  g0={row['g0']:.3e}", flush=True)
        if len(layers_full) > 1:
            for l in layers_full:
                r = out_full[f"L{l}"]
                print(f"      full L{l}: sens={r['sensitivity']:.4g} |dmu|={r['dmu_norm']:.4g}",
                      flush=True)

    # ---- P5b HEADLINE + descriptive P5a/P5d + g0 piggyback ----
    full_fp = [t["full"]["concat"]["sensitivity"] for t in per_target]
    lora_fp = [t["lora"]["concat"]["sensitivity"] for t in per_target]
    g0_v = [t["g0"] for t in per_target]
    rho_b5, n_b5 = spearman(full_fp, lora_fp)           # OFFSET-IMMUNE headline
    rho_g0, n_g0 = spearman(full_fp, g0_v)              # SMALL-n descriptive
    p5a_full = all(math.isfinite(t["full"]["concat"]["pvalue"])
                   and t["full"]["concat"]["pvalue"] < 0.002 for t in per_target)
    p5a_lora = all(math.isfinite(t["lora"]["concat"]["pvalue"])
                   and t["lora"]["concat"]["pvalue"] < 0.002 for t in per_target)
    summary = dict(arm="F", tag=tag, K=args.K, N=args.N, T=args.T, rank=args.rank,
                   lr_full=lr_full, eps_scale=eps, lr_lora=LORA_LR, n_targets=len(tgt),
                   gauge="full: attacker-visible Δθ incl ε·ξ; LoRA: ΔW=BA",
                   P5b_cross_regime_rank=dict(rho=rho_b5, n=n_b5,
                                              note="HEADLINE, offset-immune; pass ≥ +0.5"),
                   P5a_detectable=dict(full_all_p_lt_002=bool(p5a_full),
                                       lora_all_p_lt_002=bool(p5a_lora),
                                       note="DESCRIPTIVE (N→N−1 offset caveat)"),
                   g0_piggyback=dict(rho=rho_g0, n=n_g0,
                                     note="SMALL-n descriptive; does LoRA g0-predictor transfer?"),
                   per_target=[_jsonable(t) for t in per_target])
    with open(os.path.join(RESULTS, f"F{tag}_summary.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"\nsaved {RESULTS}/F{tag}_summary.json", flush=True)

    if args.stage0:
        for t in per_target:
            assert math.isfinite(t["full"]["concat"]["sensitivity"]) and \
                math.isfinite(t["lora"]["concat"]["sensitivity"]), \
                f"stage0 F t{t['tgt_id']}: a LOO footprint is NaN (metric integration broken)"
        print("STAGE-0 F OK", flush=True)
        return summary

    print("\n=== ARM F READ (§2.1) ===", flush=True)
    print(f"P5b (HEADLINE, offset-immune): cross-regime per-target rank corr rho={rho_b5:+.3f} "
          f"(n={n_b5}) — pass ≥ +0.5 ⇒ the SAME images have the biggest removal footprint under "
          f"either parameterization.", flush=True)
    print(f"P5a (descriptive, N→N−1 caveat): removal detectable p<0.002 all targets — "
          f"full={p5a_full}, LoRA={p5a_lora}.", flush=True)
    print(f"g0 piggyback (SMALL-n descriptive): rho(full LOO footprint, g0 @ θ₀)={rho_g0:+.3f} "
          f"(n={n_g0}) — does the LoRA base-gradient predictor (ρ=+0.78) transfer to full FT?",
          flush=True)
    print("P5d (DESCRIPTIVE only): per-target full vs LoRA concat sensitivity above — a "
          "normalized-footprint magnitude comparison mixes constructions with unverified "
          "offset structure; report both, do not average.", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# arm B1 — dimension-invariance GATE (§4.0-B1): CPU rescore of arm-D's SAVED stacks
# --------------------------------------------------------------------------- #
def _concat_stack(per_layer, layers):
    """{layer: [K, out, in] f32} -> [K, Σ out*in] float64 (upcast for the metric).
    Accepts either integer layer keys (the baseline `dtheta_f32={l:...}`) or the
    per-rung `v_L{l}` string keys (the v-stacks saved by measure_rung) — the two
    savers key differently, so resolve both here."""
    def _get(l):
        return per_layer[l] if l in per_layer else per_layer[f"v_L{l}"]
    return torch.cat([_get(l).reshape(_get(l).shape[0], -1).to(torch.float64)
                      for l in layers], dim=1)


def run_b1(args, tag):
    """§4.0-B1: rescore arm-D's saved concat Δθ stacks at coordinate-subset fractions
    (fixed-seed NESTED random column masks) — NO retraining. d* must be flat across the
    fractions. Reads the arm-D files under the --arm_d_tag contract:
    D{arm_d_tag}_baseline.pth (reseed ensemble) + D{arm_d_tag}_t*.pth (per-rung v stacks)."""
    import glob
    src = args.arm_d_tag
    print(f"\n{'=' * 74}\nARM B1 — dimension-invariance rescore (arm-D tag='{src}', "
          f"fractions={list(args.b1_fractions)})\n{'=' * 74}", flush=True)
    print(PREDICTIONS["B1"], flush=True)
    bl_path = os.path.join(RESULTS, f"D{src}_baseline.pth")
    assert os.path.exists(bl_path), (
        f"B1 needs arm-D's saved baseline stack {bl_path} — run `--arm D"
        f"{' --stage0' if src == '_stage0' else ''}` FIRST (it persists the stacks).")
    bl = torch.load(bl_path, weights_only=False)
    layers = [int(l) for l in bl["layers"]]
    reseed_cat = _concat_stack(bl["dtheta_f32"], layers)           # [K_re, D] float64
    K_re, D = reseed_cat.shape
    print(f"[B1] reseed ensemble {K_re} seeds, concat dim D={D} "
          f"(upcast float32→float64 per the disk-precision stamp)", flush=True)

    tfiles = sorted(glob.glob(os.path.join(RESULTS, f"D{src}_t*.pth")))
    assert tfiles, f"no arm-D per-target stacks D{src}_t*.pth in {RESULTS}"
    fractions = list(args.b1_fractions)
    if args.stage0:                                    # rescore path only; keep it cheap
        fractions = [f for f in fractions if f <= D][:2] or [min(25000, D)]
    g = torch.Generator().manual_seed(20240828)        # ONE fixed permutation → nested masks
    perm = torch.randperm(D, generator=g)

    per_target = []
    for tf in tfiles:
        st = torch.load(tf, weights_only=False)
        rung_names = list(st["rung_names"])
        dpix = dict(zip(rung_names, st["d_pixel"]))
        vst = st["v_stacks_f32"]
        # rescore each rung at every fraction; hold ONE rung's concat v-stack at a time (memory)
        sens_by_frac = {int(min(fr, D)): {} for fr in fractions}
        pval_by_frac = {int(min(fr, D)): {} for fr in fractions}
        for nm in rung_names:
            if not vst.get(nm):
                for m in sens_by_frac:
                    sens_by_frac[m][nm] = float("nan")
                continue
            v_cat = _concat_stack(vst[nm], layers)                 # [n_pairs, D] float64
            n = min(v_cat.shape[0], K_re)
            if n != v_cat.shape[0] or n != K_re:
                print(f"[B1] NOTE {os.path.basename(tf)} {nm}: v rows={v_cat.shape[0]} vs "
                      f"reseed rows={K_re} — using first {n} (aligned only if no drops).",
                      flush=True)
            for fr in fractions:
                m = int(min(fr, D))
                cols = perm[:m]
                v_list = [v_cat[j, cols] for j in range(n)]
                r_list = [reseed_cat[j, cols] for j in range(n)]
                if len(v_list) >= 2 * N_FOLDS:
                    ws = whitened_sensitivity([x.cpu() for x in v_list], [x.cpu() for x in r_list],
                                              n_folds=N_FOLDS, p_max=P_MAX, n_perm=N_PERM, seed=m)
                    sens_by_frac[m][nm] = ws["sensitivity"]
                    pval_by_frac[m][nm] = ws["pvalue"]
                else:
                    sens_by_frac[m][nm] = float("nan")
                    pval_by_frac[m][nm] = float("nan")
            del v_cat
        per_frac = []
        for m in sorted(sens_by_frac):
            sens = sens_by_frac[m]
            s_ref = sens.get("r_cross", float("nan"))
            prof = [(nm, dpix[nm], (sens[nm] / s_ref) if (s_ref and math.isfinite(s_ref)
                                                          and s_ref > 0) else float("nan"))
                    for nm in rung_names]
            ds = d_star(prof, 0.1)
            per_frac.append(dict(n_coords=m, d_star=ds,
                                 s_profile={nm: s for nm, _, s in prof}, sens=sens))
            print(f"[B1 {os.path.basename(tf)}] coords={m:>8}: d*(0.1) point={ds.get('point')} "
                  f"interval={ds.get('interval')} bracket={ds.get('bracket_rungs')}", flush=True)
        per_target.append(dict(file=os.path.basename(tf), per_fraction=per_frac))

    # flatness verdict: spread of the finite d* points across fractions (per target)
    def _pts(pt):
        return [f["d_star"].get("point") for f in pt["per_fraction"]
                if isinstance(f["d_star"].get("point"), float) and math.isfinite(f["d_star"]["point"])]
    flat = {}
    for pt in per_target:
        pts = _pts(pt)
        flat[pt["file"]] = (max(pts) / min(pts) if len(pts) >= 2 and min(pts) > 0 else float("nan"))
    summary = dict(gate="B1", arm_d_tag=src, D_full=D, K_reseed=K_re,
                   fractions=[int(min(fr, D)) for fr in fractions],
                   targets=[_jsonable(t) for t in per_target],
                   dstar_max_over_min=_jsonable(flat))
    with open(os.path.join(RESULTS, f"B1{tag}_summary.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f"\nsaved {RESULTS}/B1{tag}_summary.json", flush=True)
    print("\n=== ARM B1 READ (§4.0-B1 GATE) ===", flush=True)
    for f, r in flat.items():
        print(f"  {f}: d*(0.1) max/min across fractions = {r} "
              f"(≈1 ⇒ dimension-INVARIANT ⇒ the s(d)-normalization earns the cross-regime "
              f"headline; large ⇒ dimension-correct the ladder trend first, §II rule 3)",
              flush=True)

    if args.stage0:
        assert per_target and per_target[0]["per_fraction"], \
            "stage0 B1: no rescored fractions (arm-D stack read path broken)"
        print("STAGE-0 B1 OK", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# arm-A (job 268959) reference — for the read blocks + the S6c pre-check
# --------------------------------------------------------------------------- #
def load_arm_a():
    with open(SL_SUMMARY) as f:
        return json.load(f)


def arm_a_profiles(rung_subset=None):
    """Per-target s(d) profiles + d* from the DONE arm-A data (rescore, no retrain)."""
    aa = load_arm_a()
    out = []
    for t in aa["results"]:
        per = t["per_rung"]
        if rung_subset is not None:
            per = [r for r in per if r["rung"] in rung_subset]
        s_ref = next(r["sensitivity"] for r in per if r["rung"] == "r_cross")
        prof = [(r["rung"], r["d_pixel"], r["sensitivity"] / s_ref) for r in per]
        out.append(dict(tgt_id=t["tgt_id"],
                        s_profile={r: s for r, _, s in prof},
                        d_star={str(th): d_star(prof, th) for th in D_STAR_THRESH},
                        d_star_mini=d_star([p for p in prof
                                            if p[0] in ("p0_noise", "r_cross")], 0.1),
                        sens={r["rung"]: r["sensitivity"] for r in per}))
    return out


def precheck_arm_a():
    """S6c pre-check (CPU, job-268959 data): do >=2 rungs bracket the s=0.1 crossing,
    and which rung should be the 'one mid' rung? Prints a recommendation."""
    print("\n--- S6c arm-A rung-bracket pre-check (job 268959, CPU rescore) ---", flush=True)
    recs = []
    for t in arm_a_profiles():
        ds = t["d_star"]["0.1"]
        # nearest-to-crossing rung, EXCLUDING the rungs already fixed in the shared
        # dial set (the mid rung is the one ADDITIONAL member, plan D3/S6c):
        fixed = ("p00_identity", "p0_noise", "r_nn", "r_far", "r_cross")
        cand = [(abs(math.log(max(s, 1e-9) / 0.1)), r) for r, s in t["s_profile"].items()
                if r not in fixed and math.isfinite(s) and s > 0]
        cand.sort()
        rec = cand[0][1] if cand else None
        recs.append(rec)
        print(f"  target {t['tgt_id']}: d*(0.1) = {ds.get('point')} "
              f"interval={ds.get('interval')} bracket={ds.get('bracket_rungs')} "
              f"censored={ds.get('censored')}  -> nearest-to-crossing rung: {rec}", flush=True)
    from collections import Counter
    top = Counter([r for r in recs if r]).most_common(1)
    rec = top[0][0] if top else "p3_rot15"
    print(f"  RECOMMENDATION: mid rung = {rec} (pass --mid_rung {rec} to the dial arms "
          f"if != the p3_rot15 default; plan D3 allows swapping the mid rung on this "
          f"pre-check).", flush=True)
    return dict(per_target=[_jsonable(t) for t in arm_a_profiles()], mid_rung_rec=rec)


# --------------------------------------------------------------------------- #
# calibration (lr search, ε fixed-point, ε-bracket, null uniformity)
# --------------------------------------------------------------------------- #
def lora_reseed_target():
    try:
        v = load_arm_a()["reseed_noise"]
        return float(v), f"loaded from {SL_SUMMARY} (job 268959)"
    except Exception as e:                          # noqa: BLE001 — provenance fallback
        return LORA_RESEED_NOISE_FALLBACK, (f"HARDCODED fallback {LORA_RESEED_NOISE_FALLBACK} "
                                            f"(job 268959; summary load failed: {e})")


def lr_search(regime, ctx, batch_size=None):
    """S5: grid over LR_GRID, 1 deterministic training each (ε=0 / batch_seed=0);
    pick the lr landing max_bce in [1e-4, 1e-3), closest to the geometric center."""
    center = math.sqrt(MEM_BAND[0] * MEM_BAND[1])
    rows = []
    for lr in LR_GRID:
        fn, _ = make_train_fn(regime, ctx, lr, 0.0, batch_size=batch_size)
        try:
            dt, mbce = fn(0, ctx["x0"])
            ok = all(torch.isfinite(dt[l]).all() for l in dt)
            mbce = mbce if ok else float("nan")
        except Exception as e:                      # noqa: BLE001 — diverged candidate
            mbce = float("nan")
            print(f"  [lr_search {regime}] lr={lr} raised {type(e).__name__}: {e}", flush=True)
        rows.append((lr, mbce))
        print(f"  [lr_search {regime}] lr={lr:<6} max_bce={mbce:.3e}", flush=True)
    in_band = [(lr, m) for lr, m in rows if math.isfinite(m) and MEM_BAND[0] <= m < MEM_BAND[1]]
    if in_band:
        lr_sel, m_sel = min(in_band, key=lambda t: abs(math.log(t[1] / center)))
        print(f"  [lr_search {regime}] SELECTED lr={lr_sel} (max_bce={m_sel:.3e}, in-band)",
              flush=True)
    else:
        fin = [(lr, m) for lr, m in rows if math.isfinite(m) and m > 0]
        assert fin, f"lr_search({regime}): no finite candidate"
        lr_sel, m_sel = min(fin, key=lambda t: abs(math.log(t[1] / center)))
        print(f"  WARNING [lr_search {regime}]: NO lr lands in the S5 band — nearest is "
              f"lr={lr_sel} (max_bce={m_sel:.3e}). Achieved value must be reported "
              f"alongside every headline.", flush=True)
    return dict(selected=lr_sel, achieved_max_bce=m_sel, grid=[[a, b] for a, b in rows],
                in_band=bool(in_band))


def calibrate_eps(regime, ctx, lr, target_rms, K_cal, max_iter=4, tol=0.10):
    """Fixed-point ε-calibration (D2): choose eps_scale so the trained LAYER-0-block
    reseed RMS (K_cal ε-seeds) matches the LoRA arm's reseed_noise (the shared readout
    surface). RMS ∝ ε in the linear regime ⇒ multiplicative update converges fast."""
    frozen = ctx["frozen"]
    eps = target_rms / (frozen[0].std().item() * math.sqrt(frozen[0].numel()))
    hist = []
    rms = float("nan")
    for it in range(max_iter):
        fn, _ = make_train_fn(regime, ctx, lr, eps)
        vecs = []
        for j in range(K_cal):
            dt, mbce = fn(5000 + j, ctx["x0"])       # 5000+ block: disjoint from dial seeds
            if torch.isfinite(dt[0]).all() and mbce < MEM_GATE:
                vecs.append(dt[0].reshape(-1))
        assert len(vecs) >= 3, f"calibrate_eps({regime}): {len(vecs)}/{K_cal} usable draws"
        rms = _rms_spread(vecs)
        hist.append([eps, rms])
        print(f"  [eps_cal {regime}] iter {it}: eps={eps:.5g} -> L0 reseed RMS={rms:.4f} "
              f"(target {target_rms:.4f})", flush=True)
        if abs(rms / target_rms - 1.0) < tol:
            break
        eps *= target_rms / rms
    converged = abs(rms / target_rms - 1.0) < tol
    if not converged:
        print(f"  WARNING [eps_cal {regime}]: fixed point NOT within {tol:.0%} after "
              f"{max_iter} iters (last RMS={rms:.4f}).", flush=True)
    return dict(eps_scale=eps, achieved_rms=rms, target_rms=target_rms,
                history=hist, converged=converged)


def run_bracket(regime, ctx, lr, eps_star, encoder, args):
    """§4.1.2 / D2 sensitivity check, per regime: 3-point {ε/3, ε, 3ε} on p0_noise
    (slope α = −2 ± 0.3, curvature: mid within factor 1.5 of the endpoint chord —
    an inline substitute for the plan's 'within its CI', flagged as such) + 2-point
    {ε, 3ε} on p00_identity/r_cross + mini-d* stability < 2x across the bracket."""
    K_br = args.K_bracket
    t_pos = target_positions(ctx, 1)[0]
    rungs, _, _ = build_rung_list(ctx, t_pos, encoder, args.bank, set(BRACKET_RUNGS))
    eps_pts = [eps_star / 3.0, eps_star, 3.0 * eps_star]
    res = {}
    for ei, eps in enumerate(eps_pts):
        rung_sub = BRACKET_RUNGS if ei > 0 else ("p0_noise",)   # 3-pt only on p0_noise
        fn, layers = make_train_fn(regime, ctx, lr, eps)
        base, _, _, _, _ = build_baseline(fn, ctx["x0"], K_br, layers, seed0=6000)
        per_eps = {}
        for i, (nm, tp, de, dp) in enumerate(rungs):
            if nm not in rung_sub:
                continue
            out, meta, _ = measure_rung(ctx, t_pos, tp, base, fn, layers,
                                        seed_tag=9000 + 100 * ei + i)
            c = out["concat"]
            per_eps[nm] = dict(sensitivity=c["sensitivity"], pvalue=c["pvalue"],
                               d_pixel=dp, **meta)
            print(f"  [bracket {regime} eps={eps:.4g} {nm:>12}] sens={c['sensitivity']:.4g} "
                  f"p={c['pvalue']:.3f} (pairs={meta['n_pairs']})", flush=True)
        res[f"eps{ei}"] = dict(eps=eps, rungs=per_eps)

    # slope fit on p0_noise (log-log, 3 points)
    xs = [math.log(r["eps"]) for r in res.values() if "p0_noise" in r["rungs"]]
    ys_raw = [r["rungs"]["p0_noise"]["sensitivity"] for r in res.values()
              if "p0_noise" in r["rungs"]]
    gate = dict(slope=float("nan"), curvature_ratio=float("nan"),
                dstar_ratio=float("nan"), passed=False, note="")
    if all(math.isfinite(y) and y > 0 for y in ys_raw) and len(ys_raw) == 3:
        ys = [math.log(y) for y in ys_raw]
        n = 3
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        # curvature: mid point vs the endpoint chord (in log-log)
        chord_mid = ys[0] + (ys[2] - ys[0]) * (xs[1] - xs[0]) / (xs[2] - xs[0])
        curv = math.exp(abs(ys[1] - chord_mid))
        # mini-d* at eps and 3eps (needs r_cross: eps1/eps2 only)
        mini = []
        for k in ("eps1", "eps2"):
            rr = res[k]["rungs"]
            if "r_cross" in rr and rr["r_cross"]["sensitivity"] > 0:
                prof = [(nm, v["d_pixel"], v["sensitivity"] / rr["r_cross"]["sensitivity"])
                        for nm, v in rr.items()]
                mini.append(d_star(prof, 0.1))
        dsr = (mini[0]["point"] / mini[1]["point"]
               if len(mini) == 2 and all(math.isfinite(m["point"]) for m in mini)
               else float("nan"))
        dsr = max(dsr, 1.0 / dsr) if math.isfinite(dsr) and dsr > 0 else float("nan")
        slope_ok = abs(slope - (-2.0)) <= 0.3
        curv_ok = curv <= 1.5
        dstar_ok = (not math.isfinite(dsr)) or dsr < 2.0
        gate = dict(slope=slope, curvature_ratio=curv, dstar_ratio=dsr,
                    passed=bool(slope_ok and curv_ok and dstar_ok),
                    slope_ok=slope_ok, curvature_ok=curv_ok, dstar_ok=dstar_ok,
                    mini_dstar=[_jsonable(m) for m in mini],
                    note="curvature tol = factor 1.5 vs endpoint chord (inline CI substitute)")
    else:
        gate["note"] = "p0_noise sensitivity non-positive/non-finite at some eps — slope unfit"
    verdict = "PASS" if gate["passed"] else "FAIL"
    print(f"  [bracket {regime}] alpha={gate['slope']:.3f} (gate -2±0.3), "
          f"curvature x{gate['curvature_ratio']:.3f} (gate ≤1.5), "
          f"mini-d* ratio {gate['dstar_ratio']:.3f} (gate <2) -> {verdict}", flush=True)
    if not gate["passed"]:
        print(f"  *** GATE FAILURE (D2): shrink eps and redo the bracket before any "
              f"{regime} rung measurement is quoted. ***", flush=True)
    return dict(points=res, gate=gate)


def null_uniformity(regime, ctx, lr, eps_scale, args):
    """§4.1.3 metric-CI gate on the NEW noise source: reseed-vs-reseed null.
    Train a pool of M ε-seeds on D (no swap), then n_redraws random disjoint pairings
    v_j = Δθ(s_a) − Δθ(s_b): p-values must be ~uniform, qeff ≈ 0."""
    M, n_redraws = args.null_pool, args.n_redraws
    fn, layers = make_train_fn(regime, ctx, lr, eps_scale)
    pool = []
    for j in range(M):
        dt, mbce = fn(7000 + j, ctx["x0"])
        if all(torch.isfinite(dt[l]).all() for l in layers) and mbce < MEM_GATE:
            pool.append(_flat(dt, tuple(layers)).cpu())
    n_pairs = len(pool) // 2
    assert n_pairs >= 2 * N_FOLDS, f"null pool starved ({len(pool)}/{M} usable)"
    import numpy as np
    pvals, qeffs = [], []
    for r in range(n_redraws):
        idx = np.random.default_rng(r).permutation(len(pool))
        v = [pool[idx[2 * i]] - pool[idx[2 * i + 1]] for i in range(n_pairs)]
        rs = [pool[idx[2 * i]] for i in range(n_pairs)]
        ws = whitened_sensitivity(v, rs, n_folds=N_FOLDS, p_max=P_MAX,
                                  n_perm=N_PERM, seed=r)
        pvals.append(ws["pvalue"])
        qeffs.append(ws["qeff_count"])
    frac05 = sum(p < 0.05 for p in pvals) / len(pvals)
    qbar = sum(qeffs) / len(qeffs)
    ok = frac05 <= 0.2 and qbar <= 0.5
    print(f"  [null {regime}] {len(pool)}/{M} usable; over {n_redraws} redraws: "
          f"frac(p<0.05)={frac05:.2f} (expect ~0.05, gate ≤0.2), mean qeff={qbar:.2f} "
          f"(gate ≤0.5), min p={min(pvals):.3f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return dict(pvalues=pvals, qeffs=qeffs, frac_p_lt_05=frac05, mean_qeff=qbar, passed=ok)


def _bracket_figure(brackets):
    """Pre-declared plot 3 (calibration panel): raw sens vs ε log-log + slope −2 guide."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for regime, br in brackets.items():
            pts = [(r["eps"], r["rungs"]["p0_noise"]["sensitivity"])
                   for r in br["points"].values() if "p0_noise" in r["rungs"]]
            pts = [(e, s) for e, s in pts if s > 0]
            if pts:
                pts.sort()
                ax.loglog([e for e, _ in pts], [s for _, s in pts], "o-", label=regime)
        anchors = [(r["eps"], r["rungs"]["p0_noise"]["sensitivity"])
                   for br in brackets.values() for r in br["points"].values()
                   if r["rungs"].get("p0_noise", {}).get("sensitivity", 0) > 0]
        if anchors:
            e0 = min(e for e, _ in anchors)
            e1 = max(e for e, _ in anchors)
            e_a, s_a = anchors[0]                    # guide through the first point
            ax.loglog([e0, e1], [s_a * (e0 / e_a) ** -2, s_a * (e1 / e_a) ** -2],
                      "k--", alpha=0.5, label="slope -2 guide")
        ax.set_xlabel("eps_scale")
        ax.set_ylabel("raw sensitivity (p0_noise)")
        ax.set_title("eps-bracket: linear-response gate (alpha = -2 ± 0.3)")
        ax.legend()
        os.makedirs(FIGURES, exist_ok=True)
        out = os.path.join(FIGURES, "calibration_bracket.png")
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[fig] saved {out}", flush=True)
    except Exception as e:                          # noqa: BLE001 — best effort
        print(f"WARNING: bracket figure failed ({type(e).__name__}: {e})", flush=True)


def run_calibration(args):
    print(f"\n{'=' * 74}\nCALIBRATION (S5 lr search + D2 eps fixed-point + eps-bracket + "
          f"§4.1.3 null gate + S6c pre-check)\n{'=' * 74}", flush=True)
    print(PREDICTIONS["calib"], flush=True)
    dev = args.device
    ctx = setup_ctx(args.N, args.T, args.rank, dev)
    target_rms, prov = lora_reseed_target()
    print(f"[target] LoRA r=8 reseed_noise = {target_rms:.6f} ({prov})", flush=True)

    # 1. lr search per full-FT regime (S5); LoRA lr verified in-band
    lrs = {}
    for regime in ("C", "D", "B2"):
        print(f"\n-- lr search: regime {regime} --", flush=True)
        bs = (args.N // 4) if regime == "B2" else None
        lrs[regime] = lr_search(regime, ctx, batch_size=bs)
    print("\n-- LoRA lr=0.5 verification (1 training) --", flush=True)
    fn_l, _ = make_train_fn("E_b0", ctx, LORA_LR, 0.0)
    _, mbce_l = fn_l(0, ctx["x0"])
    lora_in_band = MEM_BAND[0] <= mbce_l < MEM_BAND[1]
    print(f"  LoRA lr={LORA_LR}: max_bce={mbce_l:.3e} in-band={lora_in_band} "
          f"(job-268959 reference: 1.35e-04)", flush=True)
    lrs["lora"] = dict(selected=LORA_LR, achieved_max_bce=mbce_l, in_band=lora_in_band)

    # 2. eps fixed-point per ε-regime (K_cal trainings per probe)
    eps = {}
    for regime, lr_key in (("C", "C"), ("D", "D"), ("E_eps", "lora")):
        print(f"\n-- eps calibration: regime {regime} (lr={lrs[lr_key]['selected']}) --",
              flush=True)
        eps[regime] = calibrate_eps(regime, ctx, lrs[lr_key]["selected"], target_rms,
                                    args.K_cal)

    # 3. the 3-point bracket per ε-regime (p0_noise 3-pt; p00/r_cross 2-pt)
    encoder, _ = sl.load_encoder(dev)
    brackets = {}
    for regime, lr_key in (("C", "C"), ("D", "D"), ("E_eps", "lora")):
        print(f"\n-- eps-bracket: regime {regime} --", flush=True)
        lr_r = lrs[lr_key]["selected"]
        br = run_bracket(regime, ctx, lr_r, eps[regime]["eps_scale"], encoder, args)
        # ε-SHRINK LOOP (auditor rule, unanimous — yoado-6d metric + yoado-18 theory):
        # the normalized s(d)/d* is ε-INVARIANT in the linear regime (the ε⁻² cancels), so
        # magnitude FLOATS and only LINEARITY is load-bearing. If the gate fails (α∉[-2.3,-1.7]
        # or d*-ratio≥2), shrink ε and re-bracket, requiring BOTH gates at the FINAL ε. Guard 3
        # (Σ-adequacy floor): if the slope goes unfittable (signal sank into the noise floor),
        # ABORT as UNMEASURABLE rather than floor-run — the honest "no linear-AND-adequate window".
        n_shrink, MAX_SHRINK = 0, 4
        while (not br["gate"]["passed"]) and n_shrink < MAX_SHRINK:
            if not math.isfinite(br["gate"].get("slope", float("nan"))):
                br["gate"]["measurable"] = False
                br["gate"]["abort_reason"] = (
                    "slope unfittable at eps=%.5g (Σ floor-dominated) — no linear-AND-adequate window"
                    % eps[regime]["eps_scale"])
                print(f"  *** ADEQUACY FLOOR ({regime}): {br['gate']['abort_reason']} ***", flush=True)
                break
            new_eps = eps[regime]["eps_scale"] * 0.5
            n_shrink += 1
            print(f"  [ε-shrink {regime} #{n_shrink}] {eps[regime]['eps_scale']:.5g} -> {new_eps:.5g} "
                  f"(restore linearity; magnitude floats — s(d)/d* ε-invariant)", flush=True)
            eps[regime]["eps_scale"] = new_eps
            eps[regime]["shrunk_iters"] = n_shrink
            br = run_bracket(regime, ctx, lr_r, new_eps, encoder, args)
        br["gate"]["shrink_iters"] = n_shrink
        br["gate"].setdefault("measurable", bool(br["gate"]["passed"]))
        if n_shrink and br["gate"].get("measurable", False):
            eps[regime]["rms_undershoot_note"] = (
                "eps shrunk %dx below the LoRA-RMS match to restore linearity — IMMATERIAL to the "
                "normalized s(d)/d* headline (ε-invariant; magnitude floats), confirmed by d*-bracket "
                "stability (ratio=%.3f<2); the RMS-undershoot caveat attaches to the RAW-sensitivity "
                "panel only." % (n_shrink, br["gate"].get("dstar_ratio", float("nan"))))
        brackets[regime] = br
    _bracket_figure(brackets)

    # 4. §4.1.3 reseed-vs-reseed null uniformity on the NEW noise source (C and D)
    nulls = {}
    for regime in ("C", "D"):
        print(f"\n-- null uniformity: regime {regime} --", flush=True)
        nulls[regime] = null_uniformity(regime, ctx, lrs[regime]["selected"],
                                        eps[regime]["eps_scale"], args)

    # 5. S6c arm-A pre-check (CPU)
    try:
        precheck = precheck_arm_a()
    except Exception as e:                          # noqa: BLE001 — data may be absent
        print(f"WARNING: arm-A pre-check failed ({type(e).__name__}: {e})", flush=True)
        precheck = dict(error=str(e), mid_rung_rec="p3_rot15")

    calib = dict(target_rms=target_rms, target_provenance=prov,
                 lr={k: _jsonable(v) for k, v in lrs.items()},
                 eps={k: _jsonable(v) for k, v in eps.items()},
                 bracket={k: _jsonable(v) for k, v in brackets.items()},
                 null_uniformity={k: _jsonable(v) for k, v in nulls.items()},
                 arm_a_precheck=_jsonable(precheck),
                 config=dict(N=args.N, T=args.T, rank=args.rank, K_cal=args.K_cal,
                             K_bracket=args.K_bracket, null_pool=args.null_pool))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nsaved {RESULTS}/calibration.json", flush=True)

    # verdict-relevant numbers LAST (house style)
    print("\n=== CALIBRATION VERDICT ===", flush=True)
    for k in ("C", "D", "B2", "lora"):
        v = lrs[k]
        print(f"  lr[{k}] = {v['selected']} (max_bce={v['achieved_max_bce']:.3e}, "
              f"in_band={v['in_band']})", flush=True)
    for k, v in eps.items():
        print(f"  eps[{k}] = {v['eps_scale']:.5g} (L0 RMS {v['achieved_rms']:.4f} vs "
              f"target {target_rms:.4f}, converged={v['converged']})", flush=True)
    for k, v in brackets.items():
        print(f"  bracket[{k}]: alpha={v['gate']['slope']:.3f} "
              f"-> {'PASS' if v['gate']['passed'] else 'FAIL'}", flush=True)
    for k, v in nulls.items():
        print(f"  null[{k}]: frac(p<.05)={v['frac_p_lt_05']:.2f} qeff={v['mean_qeff']:.2f} "
              f"-> {'PASS' if v['passed'] else 'FAIL'}", flush=True)
    print(f"  mid-rung recommendation: {precheck.get('mid_rung_rec')}", flush=True)
    all_pass = (all(v["gate"]["passed"] for v in brackets.values())
                and all(v["passed"] for v in nulls.values()))
    print(f"  OVERALL eps-validity gates: {'PASS' if all_pass else 'FAIL'} "
          f"(FAIL ⇒ shrink eps and redo before any rung measurement is quoted)", flush=True)


# --------------------------------------------------------------------------- #
# read blocks (cross-arm comparisons; headline gated on B1+B2 by construction)
# --------------------------------------------------------------------------- #
def read_block(regime, summary, targets, args):
    print(f"\n=== ARM {regime} READ (s(d) profiles + d*) ===", flush=True)
    for t in targets:
        print(f"target {t['tgt_id']} (digit {t['digit']}):", flush=True)
        for name, pr in t["profiles"].items():
            ds = pr["d_star"]["0.1"]
            void = " [NORMALIZER-VOID]" if pr["normalizer_void"] else ""
            s_str = " ".join(f"{r}={s:.3g}" if (isinstance(s, float) and math.isfinite(s))
                             else f"{r}=nan" for r, s in pr["s_profile"].items())
            print(f"  {name}{void}: s(d): {s_str}", flush=True)
            print(f"    d*(0.1): point={ds.get('point')} interval={ds.get('interval')} "
                  f"bracket={ds.get('bracket_rungs')} censored={ds.get('censored')}; "
                  f"robustness d*(0.3)={pr['d_star']['0.3'].get('point')} "
                  f"d*(0.03)={pr['d_star']['0.03'].get('point')}; "
                  f"mini-d*(p0<->r_cross)={pr['d_star_mini'].get('point')}", flush=True)

    # cross-arm comparisons (descriptive prints; the HEADLINE is gated on B1+B2)
    if regime in ("C", "D"):
        try:
            aa = arm_a_profiles(rung_subset=set(summary["rungs"]) | {"r_cross"})
            print(f"\n-- vs arm A (LoRA r=8, job 268959; shared rungs) --", flush=True)
            for t, ta in zip(targets, aa):
                pr = t["profiles"]["concat"]
                s_p0 = pr["s_profile"].get("p0_noise", float("nan"))
                s_p0_a = ta["s_profile"].get("p0_noise", float("nan"))
                ratio = s_p0 / s_p0_a if (s_p0_a and math.isfinite(s_p0_a)) else float("nan")
                print(f"  t{t['tgt_id']}: s(p0_noise) {regime}={s_p0:.4g} A={s_p0_a:.4g} "
                      f"ratio={ratio:.2f} (P1 asks ≥3 for D); "
                      f"d*(0.1) {regime}={pr['d_star']['0.1'].get('point')} "
                      f"A={ta['d_star']['0.1'].get('point')} "
                      f"A-interval={ta['d_star']['0.1'].get('interval')}", flush=True)
            if regime == "D":
                print("  [P2/S2] per-layer ordering must be read on the NUMERATOR "
                      "||dmu_l|| columns above, NOT on per-layer d* alone (eps is "
                      "L0-calibrated then std-scaled — denominator artifact risk).",
                      flush=True)
                print("  *** HEADLINE READ ONLY AFTER §4.0 B1 (dimension-invariance "
                      "rescore of the saved stacks) + B2 (SGD-order cross-check) PASS ***",
                      flush=True)
        except Exception as e:                      # noqa: BLE001
            print(f"  (arm-A comparison unavailable: {type(e).__name__}: {e})", flush=True)
    if regime == "E_eps":
        _compare_summaries("E_b0", summary, targets,
                           "P4 exchangeability: per-rung s(eps) vs s(B0) — mismatch ⇒ "
                           "noise analogue NOT exchangeable (subject to the §4.1.7 "
                           "power gate, computed in the CPU rescore from the saved "
                           "stacks — a 'pass' without 2x-detection power is INCONCLUSIVE).")
    if regime == "B2":
        _compare_summaries("D", summary, targets,
                           "§4.0-B2: full-FT s/d* under SGD-order vs eps noise — "
                           "disagreement at the d*-determining rungs kills the headline.")


def _compare_summaries(other, summary, targets, msg):
    path = os.path.join(RESULTS, f"{other}_summary.json")
    if not os.path.exists(path):
        print(f"\n-- comparison with arm {other} pending (run it, then compare "
              f"{other}_summary.json) --", flush=True)
        return
    with open(path) as f:
        osum = json.load(f)
    print(f"\n-- vs arm {other} --\n  {msg}", flush=True)
    o_t = {t["tgt_id"]: t for t in osum["targets"]}
    for t in targets:
        if t["tgt_id"] not in o_t:
            continue
        pr, opr = t["profiles"]["concat"], o_t[t["tgt_id"]]["profiles"]["concat"]
        for rung, s in pr["s_profile"].items():
            so = opr["s_profile"].get(rung)
            if (isinstance(so, (int, float)) and isinstance(s, (int, float))
                    and math.isfinite(s) and math.isfinite(so)):
                print(f"  t{t['tgt_id']} {rung:>12}: s(this)={s:.4g} s({other})={so:.4g}",
                      flush=True)
        print(f"  t{t['tgt_id']} d*(0.1): this={pr['d_star']['0.1'].get('point')} "
              f"{other}={opr['d_star']['0.1'].get('point')}", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def resolve_config(args, regime):
    """lr / eps for a dial arm: CLI override > calibration.json > (stage0 provisional)."""
    calib = None
    if os.path.exists(args.calib):
        with open(args.calib) as f:
            calib = json.load(f)
    lr_key = {"C": "C", "D": "D", "B2": "B2", "E_b0": "lora", "E_eps": "lora"}[regime]
    needs_eps = regime in ("C", "D", "E_eps")
    lr = args.lr
    if lr is None:
        if calib is not None:
            lr = calib["lr"][lr_key]["selected"]
        elif regime in ("E_b0", "E_eps"):
            lr = LORA_LR
        elif args.stage0:
            lr = 0.05
            print(f"[stage0] PROVISIONAL lr={lr} (no calibration.json — plumbing test only)",
                  flush=True)
        else:
            raise SystemExit(f"arm {regime}: no --lr and no {args.calib} — run --arm calib first")
    eps = args.eps_scale
    if needs_eps and eps is None:
        if calib is not None:
            eps = calib["eps"][regime]["eps_scale"]
        elif args.stage0:
            eps = 1e-3
            print(f"[stage0] PROVISIONAL eps_scale={eps} (no calibration.json)", flush=True)
        else:
            raise SystemExit(f"arm {regime}: no --eps_scale and no {args.calib} — run --arm calib first")
    return lr, (eps if needs_eps else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--arm", required=True,
                    choices=["calib", "C", "D", "E_b0", "E_eps", "B2", "F", "B1"])
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--n_targets", type=int, default=2)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--bank", type=int, default=200)
    ap.add_argument("--mid_rung", default="p2_rot5",
                    help="the 'one mid' rung — DEFAULT p2_rot5 (the arm-A S6c pre-check "
                         "recommendation; calib reprints it, pass another to override)")
    ap.add_argument("--rungs", default="",
                    help="comma-separated rung-name override (else per-arm defaults)")
    ap.add_argument("--lr", type=float, default=None, help="override calibration.json")
    ap.add_argument("--eps_scale", type=float, default=None, help="override calibration.json")
    ap.add_argument("--calib", default=os.path.join(RESULTS, "calibration.json"))
    ap.add_argument("--K_cal", type=int, default=8)
    ap.add_argument("--K_bracket", type=int, default=12)
    ap.add_argument("--null_pool", type=int, default=24)
    ap.add_argument("--n_redraws", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--arm_d_tag", default="",
                    help="B1: tag suffix of the arm-D stacks to rescore ('' headline / "
                         "'_stage0' for the stage-0 stacks)")
    ap.add_argument("--b1_fractions", type=int, nargs="+", default=list(B1_FRACTIONS),
                    help="B1 coordinate-subset sizes (§4.0-B1: ~25k..1.79M)")
    ap.add_argument("--stage0", action="store_true",
                    help="tiny plumbing test: K=10, 1 target, rungs {p00,p0_noise,r_cross}")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.arm == "calib":
        run_calibration(args)
        return

    # arm F (LOO) / B1 (rescore) do NOT use the dial rung_filter path — dispatch early.
    if args.arm in ("F", "B1"):
        tag = args.tag + ("_stage0" if args.stage0 else "")
        if args.stage0:
            args.K, args.n_targets = 10, 1
            print(f"=== STAGE-0 arm {args.arm} (K={args.K}, n_targets={args.n_targets}) ===",
                  flush=True)
        (run_b1 if args.arm == "B1" else run_loo_arm)(args, tag)
        return

    tag = args.tag
    if args.stage0:
        args.K, args.n_targets = 10, 1
        rung_filter = set(STAGE0_RUNGS)
        tag = tag + "_stage0"
        print(f"=== STAGE-0 (K={args.K}, 1 target, rungs={sorted(rung_filter)}) ===", flush=True)
    elif args.rungs:
        rung_filter = set(args.rungs.split(","))
    elif args.arm in ("C", "D"):
        rung_filter = {"p00_identity", "p0_noise", args.mid_rung, "r_nn", "r_far", "r_cross"}
    elif args.arm in ("E_b0", "E_eps"):
        rung_filter = set(E_RUNGS)
    else:                                           # B2 (§4.0: +p00 per §4.1.1)
        rung_filter = {"p00_identity", "p0_noise", args.mid_rung, "r_cross"}

    lr, eps = resolve_config(args, args.arm)
    summary, targets = run_dial_arm(args, args.arm, lr, eps, rung_filter, tag)

    if args.stage0:
        for t in targets:
            for r in t["per_rung"]:
                c = r["readouts"]["concat"]
                assert math.isfinite(c["sensitivity"]), \
                    f"stage0 {r['rung']}: sensitivity NaN (metric integration broken)"
                assert math.isfinite(r["d_pixel"]) and math.isfinite(r["d_encoder"]), \
                    f"stage0 {r['rung']}: distances non-finite (rung construction broken)"
        assert all(v > 0 for v in summary["baseline"]["reseed_rms"].values()), \
            "stage0: reseed RMS degenerate (noise source dead — Σ=0, metric undefined)"
        print("STAGE-0 OK", flush=True)
        return

    read_block(args.arm, summary, targets, args)


if __name__ == "__main__":
    main()
