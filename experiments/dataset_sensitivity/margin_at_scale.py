#!/usr/bin/env python
"""
MARGIN AT SCALE (program plan §III.1) — the headline figure: WHO leaks.

Upgrades the n=12 margin MVP (margin_vs_sensitivity.py + job 260171, rho(sens,g0)=+0.857 but
CI ~ +/-0.4) to n_targets=24 (default) targets selected ACROSS the g0 spectrum, with full
statistical rigor and the mechanism diagnostics the plan mandates.

PRE-REGISTERED PREDICTION (audit C7, printed FIRST at runtime):
  rho_spearman(sens, g0) > +0.6, positive in every g0 stratum, with a bootstrap 95% CI on rho
  narrower than +/-0.15 (the WIDTH is the deliverable, not just the point estimate).
  KILL: rho < +0.3 overall, or sign flips across strata => the capstone falls.

DESIGN
  1. TARGET SELECTION across the g0 spectrum: pool of pool_per_class (default 60) class-1 +
     pool_per_class class-0 MNIST-TEST images (get_finetuning_data, seed 42); g0 computed for ALL;
     n_targets/2 targets per class picked STRATIFIED across per-class g0 quantile bins (median
     element of each bin — deterministic, no cherry-pick), so the correlation is not driven by the
     two class clusters alone. Strata recorded in the summary.
  2. PER-TARGET SENSITIVITY (arm-B/D swap-measurement pattern, 3-way whitened metric): ONE fixed
     context set D0 (N=16, 8 per class, pool images disjoint from every target) with one designated
     swap SLOT per class (the last slot of that class's block). For each target T: D_T = D0 with
     the slot of T's class replaced by T; baseline ensemble ΔW(D_T, seed_j), j=1000..1000+K-1
     (K=50); swap T -> its FIXED held-out same-class control T' (get_control_images, seed 123 —
     same digit, held out); paired per-seed diff v_j = ΔW(D_T) - ΔW(D_T,swap); whitened_sensitivity
     (n_folds=5, p_max=3, n_perm=500). NOTE: the context depends on T, so there is one baseline
     ensemble PER TARGET — ~ n_targets x 2 x K = 2400 trainings at defaults (arm-D scale x2). A
     runtime estimate is printed after the first training.
  3. STATISTICS: spearman rho(sens, g0) with (a) permutation-p over >=10000 pairings (one-sided,
     pre-registered positive direction) and (b) bootstrap 95% percentile CI over >=10000 resamples.
     PASS = rho > +0.6 AND CI half-width < 0.15. KILL = rho < +0.3. Also rho(sens, m0) and
     rho(sens, lam_T) — the mechanism comparison table (the g0-vs-lam gap is the NTK-vs-max-margin
     evidence, plan T1). Per-class rho and per-tercile signs guard the "positive in every stratum"
     clause; a split-half rho probes the plan's expected-asymptote (rho drift) line.
  4. THETA0-INDEPENDENT TYPICALITY CONTROL (audit C10, MANDATORY): scores with NO reference to
     theta_0 — (a) pixel-space distance to the class mean and (b) mean distance to the 10 nearest
     same-class neighbours (kNN density) in pixel space, both against a large held-out MNIST-TEST
     bank. PRIMARY grouping is the true DIGIT (the binary "class" mean is a blur of 5 digits, so
     distance to it measures which-digit, not atypicality); binary-class-level versions are
     reported as secondary. Combined atypicality = mean of the two z-scored digit-level components.
     Reported: rho(sens, atypicality) and the PARTIAL Spearman rho(sens, g0 | atypicality)
     (rank-transform all three, then partial Pearson on the ranks) — does g0 predict BEYOND
     image-intrinsic atypicality?
  5. LAZY/NTK DIAGNOSTIC (theory T5): reference adapter = seed 1000 trained on the UNMODIFIED
     representative context D0 (no target inside — theta_0-fair to all 24 targets equally);
     per-image g_T = gradnorm at theta_0 + ΔW for all targets -> spearman(g0, g_T) [PRIMARY
     mechanism number]. Secondary: g_T from each target's OWN D_T seed-1000 adapter (T in-set).
     CONTEXT: per-module ||ΔW_l||_F / ||W_0,l||_F — per module against its OWN base weight (never
     global, never only summed). This run has a SINGLE LoRA target module (layer 0); it is
     reported explicitly labelled as such. <0.1 = lazy/NTK regime.
  6. OUTPUT: results/margin_at_scale/summary.json + margin_at_scale.pth (per-target tensors +
     the target/control/context images). HEADLINE PLOT figures/margin_at_scale/
     margin_at_scale_headline.png: sens vs g0 scatter (colored by class, rho + CI annotated) +
     sens vs atypicality panel. Prediction printed FIRST, verdict block LAST.

CONFOUNDS (flagged, not silently absorbed):
  * Two targets sharing a digit share the same control T' (arm-D convention: T' depends only on
    the digit, seed 123). The paired diff still isolates T vs T'.
  * Sensitivity is conditional on the ONE fixed context D0 (the plan's "several contexts" is
    traded for per-target ensembles at 24 targets; context-dependence was measured WEAK in arm D).
  * g0 uses the label (BCE gradient); the typicality control is label-free by construction apart
    from same-class grouping.
  * Swapping the slot image for T changes ONE context image relative to D0; identical for all
    targets of a class, so between-target comparisons stay like-for-like within class.

bsub-only. mnist / gelu MLP / binary / rank 8 / T=1000 / lr 0.5. float64.
"""
import os, json, math, time, argparse
import numpy as np
import torch

from experiments.jacobian_spectrum import make_activation
from experiments.data_utils import get_finetuning_data, get_control_images_in_distribution
from experiments.dataset_sensitivity.arm_b_dilution import draw_B0, train_adapter
from experiments.dataset_sensitivity.arm_d_context import build_base
from experiments.dataset_sensitivity.margin_vs_sensitivity import (
    margins, layer0_grad_norms, spearman, _ranks, _zero_adapter,
)
from experiments.dataset_sensitivity.whitened_metric import whitened_sensitivity

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/margin_at_scale"
FIGURES = "/home/projects/galvardi/yoado/figures/margin_at_scale"
N_FOLDS = 5

PREREGISTERED = (
    "PRE-REGISTERED PREDICTION (plan SIII.1, audit C7): rho_spearman(sens, g0) > +0.6, positive in\n"
    "every g0 stratum, AND bootstrap 95% CI on rho with half-width < 0.15 (the width target is the\n"
    "deliverable). KILL: rho < +0.3 overall or sign flips across strata => capstone falls."
)


# ---------------------------------------------------------------------------
# statistics: permutation-p, bootstrap CI, partial Spearman (no scipy)
# ---------------------------------------------------------------------------
def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    ca = [x - ma for x in a]
    cb = [x - mb for x in b]
    va = sum(x * x for x in ca)
    vb = sum(x * x for x in cb)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum(x * y for x, y in zip(ca, cb)) / math.sqrt(va * vb)


def spearman_perm_p(a, b, n_perm=10000, seed=0):
    """One-sided (rho_perm >= rho_obs, pre-registered POSITIVE direction) permutation p for
    spearman(a,b), permuting the PAIRING (shuffle b). Add-one honest p."""
    rho_obs, n = spearman(a, b)
    if not math.isfinite(rho_obs):
        return rho_obs, float("nan"), n
    rng = np.random.default_rng(seed)
    b_arr = np.asarray(b, dtype=np.float64)
    n_ge = 0
    for _ in range(n_perm):
        rho_p, _ = spearman(a, rng.permutation(b_arr).tolist())
        if math.isfinite(rho_p) and rho_p >= rho_obs:
            n_ge += 1
    return rho_obs, (n_ge + 1) / (n_perm + 1), n


def spearman_boot_ci(a, b, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap 95% CI on spearman(a,b): resample the PAIRS with replacement.
    Degenerate resamples (constant side) return nan and are dropped (count reported)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    rhos = []
    n_bad = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho, _ = spearman(a_arr[idx].tolist(), b_arr[idx].tolist())
        if math.isfinite(rho):
            rhos.append(rho)
        else:
            n_bad += 1
    if not rhos:
        return float("nan"), float("nan"), float("nan"), n_bad
    lo, hi = np.quantile(np.asarray(rhos), [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), float((hi - lo) / 2.0), n_bad


def partial_spearman(x, y, z):
    """Partial Spearman rho(x, y | z): rank-transform all three, then the partial Pearson
    (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)) on the ranks."""
    rx, ry, rz = _ranks(list(x)), _ranks(list(y)), _ranks(list(z))
    rxy, rxz, ryz = _pearson(rx, ry), _pearson(rx, rz), _pearson(ry, rz)
    if not all(math.isfinite(v) for v in (rxy, rxz, ryz)):
        return float("nan"), dict(rxy=rxy, rxz=rxz, ryz=ryz)
    den = (1 - rxz ** 2) * (1 - ryz ** 2)
    if den <= 1e-12:
        return float("nan"), dict(rxy=rxy, rxz=rxz, ryz=ryz)
    return (rxy - rxz * ryz) / math.sqrt(den), dict(rxy=rxy, rxz=rxz, ryz=ryz)


def partial_spearman_boot_ci(x, y, z, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    xa, ya, za = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p, _ = partial_spearman(xa[idx].tolist(), ya[idx].tolist(), za[idx].tolist())
        if math.isfinite(p):
            vals.append(p)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(np.asarray(vals), [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# target selection (stratified across the g0 spectrum, balanced by class)
# ---------------------------------------------------------------------------
def select_targets_stratified(g0, pool_y, pool_digits, n_targets):
    """Per class: sort that class's pool indices by g0 ascending, cut into n_targets/2
    equal-count quantile bins, take the MEDIAN element of each bin (deterministic).
    Returns (targets, strata): targets = list of dicts (pool_idx, cls, digit, g0, stratum),
    strata = per-class bin records (g0 range + members count)."""
    assert n_targets % 2 == 0, "n_targets must be even (12 per class at the default 24)"
    n_per = n_targets // 2
    yl = [int(v) for v in pool_y.tolist()]
    targets, strata = [], []
    for cls in (0, 1):
        cls_idx = [i for i, v in enumerate(yl) if v == cls]
        assert len(cls_idx) >= n_per, f"class {cls}: pool has {len(cls_idx)} < n_targets/2={n_per}"
        order = sorted(cls_idx, key=lambda i: g0[i].item())
        bins = np.array_split(np.arange(len(order)), n_per)
        for b_id, b in enumerate(bins):
            members = [order[j] for j in b.tolist()]
            pick = members[len(members) // 2]           # median of the bin
            g_lo, g_hi = g0[members[0]].item(), g0[members[-1]].item()
            targets.append(dict(pool_idx=pick, cls=cls, digit=int(pool_digits[pick]),
                                g0=g0[pick].item(), stratum=b_id))
            strata.append(dict(cls=cls, stratum=b_id, g0_lo=g_lo, g0_hi=g_hi,
                               n_members=len(members), picked_pool_idx=pick))
    return targets, strata


def build_fixed_context(pool_y, target_idx_set, N):
    """ONE fixed context D0 (N images, N/2 per class) from pool images DISJOINT from every target,
    in pool order. Layout: [class-0 block | class-1 block]; the designated swap SLOT for class c is
    the LAST position of that class's block. Returns (ctx_idx list of N pool indices,
    slot_pos = {0: N/2-1, 1: N-1})."""
    assert N % 2 == 0
    half = N // 2
    yl = [int(v) for v in pool_y.tolist()]
    ctx = {0: [], 1: []}
    for i, v in enumerate(yl):
        if i in target_idx_set:
            continue
        if len(ctx[v]) < half:
            ctx[v].append(i)
    assert len(ctx[0]) == half and len(ctx[1]) == half, \
        f"context starved: {len(ctx[0])}/{len(ctx[1])} of {half} per class (grow --pool_per_class)"
    ctx_idx = ctx[0] + ctx[1]
    slot_pos = {0: half - 1, 1: N - 1}
    return ctx_idx, slot_pos


def build_D_for_target(pool_x, pool_digits, ctx_idx, slot_pos, tgt, N, device):
    """D_T = fixed context with the slot of T's class replaced by T. Returns
    (x_D [N], y_D [N], digits_D, t_pos)."""
    half = N // 2
    idx = list(ctx_idx)
    t_pos = slot_pos[tgt["cls"]]
    idx[t_pos] = tgt["pool_idx"]
    x_D = pool_x[idx].clone()
    digits_D = [int(pool_digits[i]) for i in idx]
    y_D = torch.cat([torch.zeros(half, dtype=torch.float64, device=device),
                     torch.ones(half, dtype=torch.float64, device=device)])
    return x_D, y_D, digits_D, t_pos


# ---------------------------------------------------------------------------
# per-target swap measurement (arm-B/D pattern: paired per-seed diffs, init cancels)
# ---------------------------------------------------------------------------
def _logits_dW(x0, frozen, b0, act, dW):
    """Logits under theta_0 + dW on layer 0 (dense dW, no adapter factors). No grad."""
    with torch.no_grad():
        h = x0.view(x0.shape[0], -1)
        for l in range(len(frozen)):
            w = frozen[0] + dW if l == 0 else frozen[l]
            h = torch.nn.functional.linear(h, w, b0 if l == 0 else None)
            if l < len(frozen) - 1:
                h = act(h)
    return h.view(-1)


def measure_target(tgt, ctrl_x, N, K, lr, T, rank, device, frozen, b0, ds_mean,
                   pool_x, pool_digits, ctx_idx, slot_pos, act, out_f, seed_tag,
                   timing_box=None, n_total_train=None):
    """Baseline ensemble ΔW(D_T, seed_j) + swap ensemble (T -> T'), paired per-seed diffs,
    whitened sensitivity. Also returns lam_T / g_T-own from the seed-1000 baseline adapter."""
    seeds = [1000 + j for j in range(K)]
    x_D, y_D, digits_D, t_pos = build_D_for_target(pool_x, pool_digits, ctx_idx, slot_pos,
                                                   tgt, N, device)
    x0_D = x_D - ds_mean                                            # ds_mean FROZEN

    T_prime = ctrl_x                                                # fixed same-digit control
    assert not torch.allclose(T_prime, x_D[t_pos]), \
        f"target {tgt['pool_idx']}: control T' identical to T (held-out control broken)"
    for j in range(N):                                              # T' must be outside D_T
        assert not torch.allclose(T_prime, x_D[j]), \
            f"target {tgt['pool_idx']}: control T' collides with D_T[{j}] (pool/control overlap)"

    # baseline ensemble
    dW_base, mbce_ref, dW_ref = {}, None, None
    for s in seeds:
        t0 = time.time()
        _, _, mbce, dW = train_adapter(frozen, b0, draw_B0(s, out_f, rank, device),
                                       x0_D, y_D, lr, T, act, rank)
        if timing_box is not None and timing_box.get("t_train") is None:
            timing_box["t_train"] = time.time() - t0
            est_h = timing_box["t_train"] * n_total_train / 3600.0
            print(f"RUNTIME ESTIMATE: {timing_box['t_train']:.2f}s/training x {n_total_train} "
                  f"trainings ~= {est_h:.1f} h (excl. metric/stats overhead)", flush=True)
        if torch.isfinite(dW).all():
            dW_base[s] = dW
            if s == seeds[0]:
                mbce_ref, dW_ref = mbce, dW
    base_dropped = K - len(dW_base)
    assert len(dW_base) >= 2 * N_FOLDS, \
        f"target {tgt['pool_idx']}: only {len(dW_base)} finite baseline draws < 2*N_FOLDS"
    if dW_ref is None:                                              # seed-1000 draw was non-finite
        s_fb = next(iter(dW_base))
        dW_ref = dW_base[s_fb]
        mbce_ref = float("nan")
        print(f"WARNING: target {tgt['pool_idx']}: seed-1000 baseline non-finite; lam/gT_own "
              f"fall back to seed {s_fb}.", flush=True)
    base_stack = torch.stack(list(dW_base.values()))
    reseed_noise = ((base_stack - base_stack.mean(0)).flatten(1).norm(dim=1) ** 2).mean().sqrt().item()

    # swap ensemble + paired per-seed diffs (init CANCELS per pair)
    x_sw = x_D.clone(); x_sw[t_pos] = T_prime
    x0_sw = x_sw - ds_mean
    vs, vs_reseed, swap_dropped = [], [], 0
    for si, s in enumerate(seeds):
        if s not in dW_base:
            continue
        _, _, _, dW_sw = train_adapter(frozen, b0, draw_B0(s, out_f, rank, device),
                                       x0_sw, y_D, lr, T, act, rank)
        v = dW_base[s] - dW_sw
        if torch.isfinite(v).all():
            vs.append(v); vs_reseed.append(dW_base[s])
        else:
            swap_dropped += 1
        if (si + 1) % 10 == 0:
            print(f"    seeds {si + 1}/{K} done", flush=True)
    assert len(vs) >= 2 * N_FOLDS, \
        f"target {tgt['pool_idx']}: only {len(vs)} finite pairs < 2*N_FOLDS (metric starved)"
    coherent = torch.stack(vs).mean(0).norm().item()

    ws = whitened_sensitivity([v.cpu() for v in vs], [r.cpu() for r in vs_reseed],
                              n_folds=N_FOLDS, p_max=3, n_perm=500, seed=int(seed_tag))
    assert math.isfinite(ws["d2_obs"]), f"target {tgt['pool_idx']}: d2_obs non-finite"

    # lam_T / g_T-own from the seed-1000 baseline adapter (T IS in D_T -> converged dual proxy)
    x0_T = (pool_x[tgt["pool_idx"]] - ds_mean).unsqueeze(0)
    y_T = torch.tensor([float(tgt["cls"])], dtype=torch.float64, device=device)
    mT = ((2.0 * y_T - 1.0) * _logits_dW(x0_T, frozen, b0, act, dW_ref)).item()
    lam = float(torch.sigmoid(torch.tensor(-mT)))
    gT_own = layer0_grad_norms(x0_T, y_T, frozen, b0, act, dW=dW_ref)[0].item()

    return dict(sensitivity=ws["sensitivity"], pvalue=ws["pvalue"], d2_obs=ws["d2_obs"],
                qeff=ws["qeff_count"], rho_selected=ws["rho_selected"],
                coherent_signal=coherent, reseed_noise=reseed_noise,
                base_dropped=base_dropped, swap_dropped=swap_dropped,
                ref_max_bce=mbce_ref, memorized=bool(mbce_ref is not None and mbce_ref < 1e-3),
                mT=mT, lam=lam, gT_own=gT_own)


# ---------------------------------------------------------------------------
# theta0-independent typicality (pixel space; digit-level PRIMARY, binary secondary)
# ---------------------------------------------------------------------------
def typicality_scores(x_targets, tgt_digits, tgt_cls, bank_x, bank_digits, bank_y, k=10):
    """For each target: (a) distance to the class mean, (b) mean distance to the k nearest
    same-class neighbours — 'class' at DIGIT level (primary) and binary level (secondary).
    Bank rows numerically identical to the target (dist < 1e-6) are excluded (the pool bank is a
    superset of the pool). NO reference to theta_0 anywhere."""
    bf = bank_x.reshape(bank_x.shape[0], -1)
    bd = [int(d) for d in bank_digits]
    by = [int(v) for v in bank_y.tolist()]
    out = dict(dist_digit_mean=[], knn_digit=[], dist_class_mean=[], knn_class=[])
    for t in range(x_targets.shape[0]):
        xt = x_targets[t].reshape(-1)
        for key_mean, key_knn, mask in (
                ("dist_digit_mean", "knn_digit",
                 [i for i, d in enumerate(bd) if d == int(tgt_digits[t])]),
                ("dist_class_mean", "knn_class",
                 [i for i, v in enumerate(by) if v == int(tgt_cls[t])])):
            rows = bf[mask]
            dists = (rows - xt.unsqueeze(0)).norm(dim=1)
            keep = dists > 1e-6                                     # drop numerical self-copies
            assert int(keep.sum()) >= k + 1, \
                f"typicality bank too small for target {t} ({key_knn}: {int(keep.sum())} <= k={k})"
            out[key_mean].append((rows[keep].mean(0) - xt).norm().item())
            out[key_knn].append(dists[keep].sort().values[:k].mean().item())
    z = lambda v: ((np.asarray(v) - np.mean(v)) / (np.std(v) + 1e-30)).tolist()
    out["atypicality"] = ((np.asarray(z(out["dist_digit_mean"]))
                           + np.asarray(z(out["knn_digit"]))) / 2.0).tolist()
    return out


# ---------------------------------------------------------------------------
# headline plot
# ---------------------------------------------------------------------------
def make_headline_plot(sens, g0_t, atyp, cls, rho, ci_lo, ci_hi, perm_p, rho_atyp, rho_partial,
                       path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, xv, xlabel, title in (
            (axes[0], g0_t, "base per-image gradient norm g0 (theta_0, layer 0)",
             "HEADLINE: whitened sensitivity vs g0"),
            (axes[1], atyp, "theta_0-independent atypicality (z-combined digit-level)",
             "CONTROL: sensitivity vs pixel-space atypicality")):
        for c in (0, 1):
            xs = [x for x, cc in zip(xv, cls) if cc == c]
            ys = [s for s, cc in zip(sens, cls) if cc == c]
            ax.scatter(xs, ys, s=55, color=colors[c], label=f"class {c}",
                       edgecolor="black", linewidth=0.5, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("whitened sensitivity (3-way, permutation-debiased)")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    axes[0].annotate(
        f"spearman rho = {rho:+.3f}\n95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}] "
        f"(half-width {((ci_hi - ci_lo) / 2):.3f})\nperm-p = {perm_p:.2e} (one-sided)\n"
        f"pre-reg: rho > +0.6, half-width < 0.15",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#fffbe6", alpha=0.9))
    axes[1].annotate(
        f"rho(sens, atypicality) = {rho_atyp:+.3f}\n"
        f"PARTIAL rho(sens, g0 | atypicality) = {rho_partial:+.3f}",
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#fffbe6", alpha=0.9))
    fig.suptitle("Margin at scale (SIII.1): who leaks — base-gradient geometry vs intrinsic "
                 "atypicality", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved headline plot -> {path}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_targets", type=int, default=24, help="total targets (half per class)")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--pool_per_class", type=int, default=60,
                    help="candidate pool size per binary class (MNIST test, seed 42)")
    ap.add_argument("--bank_per_class", type=int, default=500,
                    help="typicality reference bank size per binary class")
    ap.add_argument("--n_perm_rho", type=int, default=10000)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--stage0", action="store_true",
                    help="tiny sanity: 4 targets, K=12, no bootstrap-CI requirement, assert finite")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    print(PREREGISTERED, flush=True)                                # FIRST, per the plan

    if args.stage0:
        n_targets, K = 4, 12
        n_perm_rho = min(args.n_perm_rho, 1000)
        print(f"=== STAGE-0 SANITY ({n_targets} targets, K={K}, no bootstrap-CI requirement) ===",
              flush=True)
    else:
        n_targets, K, n_perm_rho = args.n_targets, args.K, args.n_perm_rho
    N, T, lr, rank = args.N, args.T, args.lr, args.rank
    tag = "_stage0" if args.stage0 else ""

    # ---- base theta_0 + pool (arm-D convention; pool = MNIST-TEST, seed 42) ----
    frozen, b0, ds_mean, pool_x, pool_y, pool_digits = build_base(
        N, lr, T, dev, pool_n=args.pool_per_class)
    act = make_activation("gelu")
    out_f = frozen[0].shape[0]
    print(f"pool: {pool_x.shape[0]} images ({args.pool_per_class}/class), N={N}, K={K}, "
          f"n_targets={n_targets}, rank={rank}, T={T}, lr={lr}", flush=True)

    # ---- 1) base quantities g0 / m0 for EVERY pool image (no training) ----
    x0_pool = pool_x - ds_mean
    A0, B0z = _zero_adapter(frozen, rank, dev)
    m0 = margins(x0_pool, pool_y, frozen, b0, act, A0, B0z).cpu()
    g0 = layer0_grad_norms(x0_pool, pool_y, frozen, b0, act)
    assert torch.isfinite(m0).all() and torch.isfinite(g0).all(), "pool m0/g0 non-finite"
    print(f"pool g0 in [{g0.min():.3e}, {g0.max():.3e}], m0 in [{m0.min():.3f}, {m0.max():.3f}]",
          flush=True)

    # ---- target selection: stratified across g0 quantiles, balanced by class ----
    targets, strata = select_targets_stratified(g0, pool_y, pool_digits, n_targets)
    tgt_set = {t["pool_idx"] for t in targets}
    print("targets (cls/stratum/digit/pool_idx/g0): "
          + "; ".join(f"c{t['cls']}s{t['stratum']} d{t['digit']} i{t['pool_idx']} "
                      f"g0={t['g0']:.3e}" for t in targets), flush=True)

    # ---- fixed context D0 (disjoint from all targets) + fixed controls per digit ----
    ctx_idx, slot_pos = build_fixed_context(pool_y, tgt_set, N)
    print(f"fixed context D0 pool indices: {ctx_idx} (swap slots: class0 -> pos {slot_pos[0]}, "
          f"class1 -> pos {slot_pos[1]})", flush=True)
    uniq_digits = sorted({t["digit"] for t in targets})
    ctrl_x_all, _, _ = get_control_images_in_distribution(uniq_digits, seed=123, dataset="mnist")
    ctrl_by_digit = {d: ctrl_x_all[i].to(torch.float64).to(dev)
                     for i, d in enumerate(uniq_digits)}

    # ---- 2) per-target sensitivity: one baseline+swap ensemble PER target ----
    n_total_train = n_targets * 2 * K + 1
    timing_box = {"t_train": None}
    per_target = []
    t_start = time.time()
    for ti, tgt in enumerate(targets):
        print(f"\n===== target {ti + 1}/{n_targets}: class {tgt['cls']}, digit {tgt['digit']}, "
              f"stratum {tgt['stratum']}, g0={tgt['g0']:.3e} =====", flush=True)
        r = measure_target(tgt, ctrl_by_digit[tgt["digit"]], N, K, lr, T, rank, dev,
                           frozen, b0, ds_mean, pool_x, pool_digits, ctx_idx, slot_pos,
                           act, out_f, seed_tag=20000 + ti,
                           timing_box=timing_box, n_total_train=n_total_train)
        r.update(pool_idx=tgt["pool_idx"], cls=tgt["cls"], digit=tgt["digit"],
                 stratum=tgt["stratum"], g0=tgt["g0"], m0=m0[tgt["pool_idx"]].item())
        per_target.append(r)
        if not r["memorized"]:
            print(f"WARNING: target {ti} baseline NOT memorized (max_bce={r['ref_max_bce']:.2e}) "
                  f"— lam proxy off-convergence.", flush=True)
        print(f"  -> sens={r['sensitivity']:.3f} p={r['pvalue']:.3f} coh={r['coherent_signal']:.3f} "
              f"reseed={r['reseed_noise']:.3f} lam={r['lam']:.3e} "
              f"[elapsed {(time.time() - t_start) / 60:.1f} min]", flush=True)

    sens = [r["sensitivity"] for r in per_target]
    g0_t = [r["g0"] for r in per_target]
    m0_t = [r["m0"] for r in per_target]
    lam_t = [r["lam"] for r in per_target]
    cls_t = [r["cls"] for r in per_target]
    assert all(math.isfinite(s) for s in sens), "non-finite sensitivity (metric broken)"

    # ---- 5) lazy/NTK diagnostic: representative adapter (seed 1000) on the UNMODIFIED D0 ----
    x_D0 = pool_x[ctx_idx].clone()
    y_D0 = torch.cat([torch.zeros(N // 2, dtype=torch.float64, device=dev),
                      torch.ones(N // 2, dtype=torch.float64, device=dev)])
    _, _, mbce_rep, dW_rep = train_adapter(frozen, b0, draw_B0(1000, out_f, rank, dev),
                                           (x_D0 - ds_mean), y_D0, lr, T, act, rank)
    x_targets = pool_x[[t["pool_idx"] for t in targets]].clone()
    y_targets = torch.tensor([float(t["cls"]) for t in targets], dtype=torch.float64, device=dev)
    gT_rep = layer0_grad_norms(x_targets - ds_mean, y_targets, frozen, b0, act, dW=dW_rep)
    rho_g0_gT, _ = spearman(g0_t, gT_rep.tolist())
    rho_g0_gTown, _ = spearman(g0_t, [r["gT_own"] for r in per_target])
    # per-module ||dW||_F / ||W0||_F — per module against ITS OWN base weight. This run trains a
    # SINGLE LoRA module (layer 0); reported per-module (here: one entry), labelled, never global.
    module_ratios = {"layer0 (the single LoRA target module)":
                     dW_rep.norm().item() / frozen[0].norm().item()}
    lazy_all = all(v < 0.1 for v in module_ratios.values())

    # ---- 4) theta_0-independent typicality (large held-out bank, pixel space) ----
    bank_n = max(args.bank_per_class, args.pool_per_class)
    bank_x, bank_y, bank_digits, _ = get_finetuning_data(bank_n, seed=42, device=dev,
                                                         dataset="mnist")
    bank_x = bank_x.to(torch.float64)
    typ = typicality_scores(x_targets, [t["digit"] for t in targets], cls_t,
                            bank_x, bank_digits, bank_y, k=args.knn_k)
    atyp = typ["atypicality"]
    assert all(math.isfinite(v) for v in atyp), "atypicality non-finite"

    # ---- 3) statistics ----
    rho, perm_p, n_pairs = spearman_perm_p(sens, g0_t, n_perm=n_perm_rho, seed=7)
    if args.stage0:
        ci_lo = ci_hi = halfwidth = float("nan"); n_bad_boot = 0
    else:
        ci_lo, ci_hi, halfwidth, n_bad_boot = spearman_boot_ci(sens, g0_t,
                                                               n_boot=args.n_boot, seed=8)
    rho_m0, perm_p_m0, _ = spearman_perm_p(sens, [-v for v in m0_t], n_perm=n_perm_rho, seed=9)
    rho_m0 = -rho_m0                                                # report signed rho(sens, m0)
    rho_lam, perm_p_lam, _ = spearman_perm_p(sens, lam_t, n_perm=n_perm_rho, seed=10)
    rho_atyp, _ = spearman(sens, atyp)
    rho_typ_dm, _ = spearman(sens, typ["dist_digit_mean"])
    rho_typ_knn, _ = spearman(sens, typ["knn_digit"])
    rho_typ_cm, _ = spearman(sens, typ["dist_class_mean"])
    rho_g0_atyp, _ = spearman(g0_t, atyp)
    rho_partial, partial_parts = partial_spearman(sens, g0_t, atyp)
    if args.stage0:
        part_lo = part_hi = float("nan")
    else:
        part_lo, part_hi = partial_spearman_boot_ci(sens, g0_t, atyp, n_boot=args.n_boot, seed=11)

    # per-class rho (the "not driven by the class clusters alone" guard)
    per_class_rho = {}
    for c in (0, 1):
        sc = [s for s, cc in zip(sens, cls_t) if cc == c]
        gc = [g for g, cc in zip(g0_t, cls_t) if cc == c]
        per_class_rho[c], _ = spearman(sc, gc)
    # per-tercile signs (pooled g0 terciles — the "positive in every stratum" clause)
    order = sorted(range(len(sens)), key=lambda i: g0_t[i])
    terciles = np.array_split(np.asarray(order), 3)
    tercile_rhos = []
    for tb in terciles:
        rt, _ = spearman([sens[i] for i in tb.tolist()], [g0_t[i] for i in tb.tolist()])
        tercile_rhos.append(rt)
    sign_flip = any(math.isfinite(rt) and rt < 0 for rt in tercile_rhos)
    # split-half drift (expected-asymptote line: rho should be stable across halves)
    rho_even, _ = spearman(sens[0::2], g0_t[0::2])
    rho_odd, _ = spearman(sens[1::2], g0_t[1::2])

    # ---- verdict (pre-registered) ----
    if args.stage0:
        verdict = "STAGE-0 (no verdict: CI requirement waived, n=4)"
    elif math.isfinite(rho) and rho < 0.3:
        verdict = f"KILL (rho={rho:+.3f} < +0.3 — the capstone falls; demote to an anecdote)"
    elif math.isfinite(rho) and rho > 0.6 and math.isfinite(halfwidth) and halfwidth < 0.15 \
            and not sign_flip:
        verdict = (f"PASS (rho={rho:+.3f} > +0.6, CI half-width {halfwidth:.3f} < 0.15, "
                   f"no tercile sign flip)")
    else:
        why = []
        if not (math.isfinite(rho) and rho > 0.6):
            why.append(f"rho={rho:+.3f} not > +0.6")
        if not (math.isfinite(halfwidth) and halfwidth < 0.15):
            why.append(f"CI half-width {halfwidth:.3f} not < 0.15")
        if sign_flip:
            why.append("tercile sign flip")
        verdict = "INDETERMINATE (" + "; ".join(why) + ") — above KILL, below PASS"

    # ---- 6) save summary.json + .pth + headline plot ----
    os.makedirs(RESULTS, exist_ok=True)
    summary = dict(
        config=dict(n_targets=n_targets, N=N, K=K, lr=lr, T=T, rank=rank,
                    pool_per_class=args.pool_per_class, bank_per_class=bank_n,
                    n_perm_rho=n_perm_rho, n_boot=(0 if args.stage0 else args.n_boot),
                    knn_k=args.knn_k, n_folds=N_FOLDS, metric_n_perm=500, seeds0=1000,
                    stage0=bool(args.stage0)),
        preregistered=PREREGISTERED,
        strata=strata,
        context=dict(ctx_idx=ctx_idx, slot_pos={str(k): v for k, v in slot_pos.items()},
                     ctx_digits=[int(pool_digits[i]) for i in ctx_idx]),
        per_target=[{k: v for k, v in r.items()} for r in per_target],
        headline=dict(rho_sens_g0=rho, perm_p=perm_p, n=n_pairs,
                      ci95=[ci_lo, ci_hi], ci_halfwidth=halfwidth, n_bad_boot=n_bad_boot,
                      per_class_rho=per_class_rho, tercile_rhos=tercile_rhos,
                      sign_flip_across_terciles=bool(sign_flip),
                      rho_split_even=rho_even, rho_split_odd=rho_odd),
        mechanism_table=dict(rho_sens_g0=rho, rho_sens_m0=rho_m0, rho_sens_lam=rho_lam,
                             perm_p_g0=perm_p, perm_p_m0_neg=perm_p_m0, perm_p_lam=perm_p_lam,
                             g0_vs_lam_gap=(rho - rho_lam) if all(map(math.isfinite, (rho, rho_lam)))
                             else float("nan")),
        typicality_control=dict(rho_sens_atypicality=rho_atyp,
                                rho_sens_dist_digit_mean=rho_typ_dm,
                                rho_sens_knn_digit=rho_typ_knn,
                                rho_sens_dist_class_mean_secondary=rho_typ_cm,
                                rho_g0_atypicality=rho_g0_atyp,
                                partial_rho_sens_g0_given_atyp=rho_partial,
                                partial_parts=partial_parts,
                                partial_ci95=[part_lo, part_hi]),
        lazy_diagnostic=dict(spearman_g0_gT_representative=rho_g0_gT,
                             spearman_g0_gT_own_context_secondary=rho_g0_gTown,
                             per_module_dW_over_W0=module_ratios,
                             lazy_all_modules=bool(lazy_all),
                             rep_max_bce=mbce_rep, rep_memorized=bool(mbce_rep < 1e-3)),
        verdict=verdict,
    )
    with open(os.path.join(RESULTS, f"summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(dict(summary=summary,
                    x_targets=x_targets.cpu(), y_targets=y_targets.cpu(),
                    x_controls=torch.stack([ctrl_by_digit[t["digit"]] for t in targets]).cpu(),
                    x_context=x_D0.cpu(), ds_mean=ds_mean.cpu(),
                    g0_pool=g0, m0_pool=m0, pool_y=pool_y.cpu(),
                    pool_digits=[int(d) for d in pool_digits],
                    sens=torch.tensor(sens), g0_targets=torch.tensor(g0_t),
                    m0_targets=torch.tensor(m0_t), lam_targets=torch.tensor(lam_t),
                    gT_rep=gT_rep, gT_own=torch.tensor([r["gT_own"] for r in per_target]),
                    atypicality=torch.tensor(atyp),
                    dist_digit_mean=torch.tensor(typ["dist_digit_mean"]),
                    knn_digit=torch.tensor(typ["knn_digit"]),
                    dW_rep=dW_rep.cpu()),
               os.path.join(RESULTS, f"margin_at_scale{tag}.pth"))
    print(f"\nsaved {RESULTS}/summary{tag}.json + margin_at_scale{tag}.pth", flush=True)

    make_headline_plot(sens, g0_t, atyp, cls_t, rho, ci_lo, ci_hi, perm_p, rho_atyp, rho_partial,
                       os.path.join(FIGURES, f"margin_at_scale_headline{tag}.png"))

    # ---- VERDICT BLOCK — LAST, per the plan ----
    print("\n=== VERDICT BLOCK (margin at scale, SIII.1 pre-registered) ===", flush=True)
    print(f"HEADLINE rho(sens, g0) = {rho:+.3f}  (n={n_pairs}, one-sided perm-p={perm_p:.2e}, "
          f"{n_perm_rho} perms)")
    print(f"  bootstrap 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]  half-width={halfwidth:.3f} "
          f"(target < 0.15){'' if not n_bad_boot else f'  [{n_bad_boot} degenerate resamples dropped]'}")
    print(f"  per-class rho: c0={per_class_rho[0]:+.3f}  c1={per_class_rho[1]:+.3f}  "
          f"(class-cluster guard)")
    print(f"  g0-tercile rhos: {['%+.3f' % r if math.isfinite(r) else 'nan' for r in tercile_rhos]} "
          f"sign-flip={sign_flip}")
    print(f"  split-half stability: rho_even={rho_even:+.3f} rho_odd={rho_odd:+.3f} "
          f"(drifting rho => strata not exchangeable)")
    print(f"MECHANISM TABLE: rho(sens,g0)={rho:+.3f} | rho(sens,m0)={rho_m0:+.3f} | "
          f"rho(sens,lam_T)={rho_lam:+.3f}  ->  g0-vs-lam gap = "
          f"{summary['mechanism_table']['g0_vs_lam_gap']:+.3f} "
          f"(gap>0: NTK/gradient-recording over max-margin, plan T1)")
    print(f"TYPICALITY CONTROL (theta_0-independent, pixel space): "
          f"rho(sens,atyp)={rho_atyp:+.3f} [dist-to-digit-mean {rho_typ_dm:+.3f}, "
          f"kNN{args.knn_k} {rho_typ_knn:+.3f}; binary-class-mean secondary {rho_typ_cm:+.3f}]")
    print(f"  rho(g0,atyp)={rho_g0_atyp:+.3f};  PARTIAL rho(sens, g0 | atyp) = {rho_partial:+.3f} "
          f"(95% CI [{part_lo:+.3f}, {part_hi:+.3f}]) — g0 predicting BEYOND intrinsic atypicality"
          f"{' (YES)' if math.isfinite(rho_partial) and rho_partial > 0.3 else ''}")
    print(f"LAZY/NTK DIAGNOSTIC: spearman(g0, gT_rep)={rho_g0_gT:+.3f} [PRIMARY mechanism number; "
          f"own-context secondary {rho_g0_gTown:+.3f}]")
    for name, v in module_ratios.items():
        print(f"  per-module ||dW||_F/||W0||_F: {name} = {v:.4f} "
              f"({'lazy (<0.1)' if v < 0.1 else 'NOT lazy (>=0.1) — needs a feature-learning story'})")
    print(f"VERDICT: {verdict}", flush=True)

    if args.stage0:
        assert math.isfinite(rho), "stage0: rho non-finite"
        assert all(math.isfinite(v) for v in (rho_g0_gT, rho_atyp)), \
            "stage0: diagnostic correlations non-finite"
        print("STAGE-0 OK")


if __name__ == "__main__":
    main()
