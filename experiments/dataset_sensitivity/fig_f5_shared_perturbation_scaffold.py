#!/usr/bin/env python
"""
FIGURE F5 — RECOVER THE SHARED PERTURBATION  (COMPUTE APPROVED, WIRED).

######################################################################
#  Compute was APPROVED by the user (Gal). The former refuse-to-run  #
#  --approved gate and the NotImplementedError stub are REMOVED; the #
#  fine-tune + known-recipe ΔW-matching attack is now implemented.   #
#  bsub-only (WEXAC); NEVER run locally.                             #
######################################################################

THREAT MODEL (a DIFFERENT, EASIER target than per-image reconstruction):
Apply a fixed SHARED transform T_p (a common rotation theta [deg], or a common
gaussian-blur sigma) to ALL N private training images with ONE true parameter
p_true, fine-tune a LoRA adapter, then attack to recover the transform's scalar
PARAMETER p (theta or sigma) — NOT the images. Readout: recovery error of the
shared-transform parameter vs its true value, normalized against a transform-
BLIND baseline (recovery SKILL), with a bootstrap CI over independent p_true draws.

ATTACK — KNOWN-RECIPE ΔW-MATCHING (BEYOND the weakest per-image attacker):
The attacker KNOWS the recipe (frozen base θ0 checkpoint, lr, T, rank, activation)
and holds a PROXY image set — DIFFERENT images, SAME distribution (a disjoint MNIST
subset), NOT the private images. For each candidate p on a grid, the attacker applies
p to the PROXY images, fine-tunes an adapter under the IDENTICAL recipe -> ΔW_proxy(p),
and recovers  p_hat = argmax_p  cosine(ΔW_obs, ΔW_proxy(p)),  where ΔW_obs = B·A is the
observed gauge-invariant adapter product on the private (transformed) set. B0-init noise
is averaged over K seeds for both ΔW_obs and each ΔW_proxy(p). The proxy grid is an
attacker asset -> computed ONCE and reused across all p_true draws.

WHY THIS IS AN EXTENSION, NOT A CONTRADICTION (framing, mandatory):
Recovering ONE scalar shared by N images is a much weaker / often easier target than
reconstructing the per-image pixels the weakest-attacker valley measures. A recovery
SKILL significantly > 0 here is STRUCTURAL leakage — an extension PAST the weakest-
attacker (prior-free, adapter-only, per-image) target, NOT a contradiction of it. This
is a KNOWN-RECIPE attacker, strictly stronger than the weakest per-image attacker.

HONEST FAILURE MODE (flagged, not hidden): ΔW_obs carries private-IMAGE content AND the
transform; ΔW_proxy(p) carries PROXY-image content AND the transform. If image-content
mismatch dominates the adapter direction, cosine(ΔW_obs, ΔW_proxy(p)) is FLAT in p (no
peak at p_true) -> skill ~ 0 -> the honest null "not recoverable above blind". Report it,
do not spin it.

Reuses the arm_b_dilution fine-tune machinery (build_set / train_adapter / draw_B0;
mnist / gelu / binary / rank 8 / T=1000 / float64) and the similarity_ladder shared-
transform harness (_rotate, _gauss_blur) so the common transform is byte-identical to
the per-image rungs. bsub-only.
"""
import os
import sys
import math
import json
import random
import argparse

import torch

# NOTE: the shared-transform harness (_rotate, _gauss_blur) is imported LAZILY inside
# apply_shared_transform, and the heavy fine-tune stack (build_set / train_adapter /
# draw_B0 / _honest_target / make_activation, all WEXAC-only) is imported LAZILY inside
# run_shared_perturbation_experiment, so this module's metric functions stay importable
# (and ast-parseable) without torchvision/timm/checkpoints.

FIGURES = "/home/projects/galvardi/yoado/figures/shared_perturbation"
MEETING_FIG = "/home/projects/galvardi/yoado/figures/meeting/f5_shared_perturbation.png"
RESULTS = "/home/projects/galvardi/yoado/results/shared_perturbation"

# private set uses seed 42 (the arm-B/D convention); the proxy is a DISJOINT subset.
PRIVATE_BASE_SEED = 42
PROXY_SEED = 20240829

CAPTION = (
    "F5 - Recover the SHARED transform parameter (known-recipe ΔW-matching attack). "
    "OBSERVE (not conclude): recovering ONE scalar shared by N images is a DIFFERENT, "
    "often EASIER target than the per-image pixels the weakest attacker measures - a "
    "STRUCTURAL-leakage EXTENSION beyond the weakest attacker, NOT a contradiction. "
    "This is a KNOWN-RECIPE attacker (strictly stronger than the weakest per-image one). "
    "skill = 1 - |p_hat - p_true| / (transform-blind error); RECOVERABLE only if the "
    "bootstrap-CI lower bound of skill > 0 (skill significantly above blind), else an "
    "honest null. CI is bootstrapped over INDEPENDENT p_true draws (the N images share p "
    "- a CI over them would be pseudo-replication)."
)


# --------------------------------------------------------------------------- #
# SHARED TRANSFORM — apply a SINGLE parameter to ALL N images (not per-image). #
# --------------------------------------------------------------------------- #
def apply_shared_transform(images, kind, param):
    """images: [N,1,28,28] float in [0,1]. Apply the SAME transform (kind, param) to
    EVERY image. kind in {'rot' (deg), 'blur' (sigma)}. Reuses the ladder harness so the
    common transform is byte-identical to the per-image rungs. Computed on CPU (the harness
    builds its affine grid / kernel on CPU) then left on the input images' CPU device; the
    caller moves the result to the compute device."""
    from experiments.dataset_sensitivity.similarity_ladder import _rotate, _gauss_blur
    out = []
    for i in range(images.shape[0]):
        img = images[i].detach().to("cpu", torch.float64)
        if kind == "rot":
            out.append(_rotate(img, deg=param))
        elif kind == "blur":
            out.append(_gauss_blur(img, sigma=param))
        else:
            raise ValueError(f"unknown shared-transform kind {kind!r}")
    return torch.stack(out, dim=0)


# --------------------------------------------------------------------------- #
# RECOVERY-ERROR METRIC (yoado-6d spec — defined; do NOT redefine downstream). #
# --------------------------------------------------------------------------- #
def recovery_error(kind, theta_hat, theta_true):  # yoado-6d defines this
    """Absolute PARAMETER error in native units for ONE recovery.

    yoado-6d spec:
      * rotation (kind='rot'): CIRCULAR error = min(|theta_hat-theta| mod 360,
        360 - that). Degrees.
      * blur (kind='blur'): plain |sigma_hat - sigma|. Sigma units.
    """
    if kind == "rot":
        d = abs(theta_hat - theta_true) % 360.0
        return min(d, 360.0 - d)
    if kind == "blur":
        return abs(theta_hat - theta_true)
    raise ValueError(f"unknown kind {kind!r}")


def blind_prior_mean(kind, grid):
    """The transform-BLIND constant guess = the prior MEAN over the pinned prior support
    (the candidate grid). yoado numbers-audit refinement:
      * rotation: the prior support is PINNED to the (symmetric) grid range [gmin,gmax];
        the optimal constant guess under a symmetric restricted range is the prior mean,
        which is mean(grid)=0 here. (A FULL-circle uniform prior would DEGENERATE - all
        constant guesses tie by symmetry at err=90deg - which is exactly why we PIN the
        support to the restricted grid range rather than the full circle.)
      * blur: sigma is linear/non-periodic; the prior-mean guess = mean(grid) is well-defined.
    """
    return sum(grid) / len(grid)


def blind_baseline_error(kind, theta_true_draws, prior_mean):
    """Transform-BLIND guesser: always guess the prior MEAN of the parameter. Returns the
    list of per-draw baseline errors (one per draw) so the bootstrap resamples the SAME
    independent p_true-draws for both the recovery errors and the baseline errors."""
    return [recovery_error(kind, prior_mean, tt) for tt in theta_true_draws]


def analytic_blind_error(kind, grid):
    """Documentation-only: the EXPECTED transform-blind error if p_true is uniform over the
    pinned prior support (the grid). rotation: E|theta| over the symmetric range (e.g.
    [-40,40] -> 20deg). blur: E|sigma - mean(grid)| over the range. The reported err_baseline
    in the skill is the per-DRAW version above (matched to the bootstrap draws); this analytic
    value is printed alongside so the two agree up to draw noise."""
    pm = blind_prior_mean(kind, grid)
    return sum(recovery_error(kind, pm, g) for g in grid) / len(grid)


def recovery_skill(err, err_baseline):
    """Normalized dimensionless headline: skill = 1 - err/err_baseline.
    1 = perfect, 0 = no better than the transform-blind guess, <0 = worse.
    Meaningful recovery REQUIRES err significantly below err_baseline (skill>0)."""
    if err_baseline == 0:
        return float("nan")
    return 1.0 - err / err_baseline


def bootstrap_skill_ci(errs, baseline_errs, n_boot=10000, alpha=0.05, seed=0):
    """Bootstrap 95% CI for the recovery SKILL.

    GUARD (yoado-6d): resample over INDEPENDENT p_true-DRAWS (each draw = a fresh shared-p
    dataset + one full recovery), n = number of draws. Do NOT bootstrap over the N images —
    they SHARE p, so a CI over N is pseudo-replication. `errs` / `baseline_errs` must therefore
    each hold ONE value per p_true-draw.
    """
    assert len(errs) == len(baseline_errs), "errs and baseline_errs must align by draw"
    m = len(errs)
    if m == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    point = recovery_skill(sum(errs) / m, sum(baseline_errs) / m)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(m) for _ in range(m)]  # resample DRAWS, not images
        e = sum(errs[i] for i in idx) / m
        b = sum(baseline_errs[i] for i in idx) / m
        s = recovery_skill(e, b)
        if math.isfinite(s):
            boots.append(s)
    if not boots:
        return (point, float("nan"), float("nan"))
    boots.sort()
    lo = boots[int((alpha / 2) * len(boots))]
    hi = boots[int((1 - alpha / 2) * len(boots)) - 1]
    return (point, lo, hi)


def skill_verdict(lo):
    """The meaningful-recovery GATE (F5 analogue of the trivial-baseline gate): recovery is
    REAL only if the bootstrap-CI LOWER bound of skill > 0 (skill significantly above blind),
    not merely the point estimate."""
    if not math.isfinite(lo):
        return "INDETERMINATE (CI undefined)"
    return "RECOVERABLE (CI>0)" if lo > 0 else "NOT above blind (CI straddles 0 - honest null)"


# --------------------------------------------------------------------------- #
# fine-tune helpers (reuse arm_b_dilution machinery; ΔW = B·A only)
# --------------------------------------------------------------------------- #
def _cos_flat(a, b):
    """Cosine similarity between two flattened ΔW tensors (gauge-invariant products)."""
    import torch.nn.functional as F
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1).item()


def _mean_dW(train_adapter, draw_B0, frozen, b0, x0, y, lr, T, act, rank, out_f, seeds, device):
    """Fine-tune the adapter K times (one per B0 seed) and return the seed-AVERAGED ΔW=B·A
    (init-noise reduced) plus the mean max-BCE (convergence check). Non-finite draws dropped."""
    dWs, mbces = [], []
    for s in seeds:
        B0 = draw_B0(s, out_f, rank, device)
        _, _, mbce, dW = train_adapter(frozen, b0, B0, x0, y, lr, T, act, rank)
        if torch.isfinite(dW).all():
            dWs.append(dW)
            mbces.append(mbce)
    if not dWs:
        return None, float("nan"), 0
    return torch.stack(dWs).mean(0), (sum(mbces) / len(mbces)), len(dWs)


def _load_disjoint_sets(get_finetuning_data, N, device):
    """Private set (seed 42) + a DISJOINT proxy set (same distribution, different images).
    Both balanced binary. Redraws the proxy seed until its test-set indices are disjoint from
    the private ones (bounded tries). Returns cpu-float64 image tensors + labels + index sets."""
    n_per_class = N // 2
    xp, yp, dp, ip = get_finetuning_data(n_per_class, seed=PRIVATE_BASE_SEED, device="cpu", dataset="mnist")
    priv_idx = set(int(i) for i in ip)
    seed = PROXY_SEED
    for _ in range(50):
        xq, yq, dq, iq = get_finetuning_data(n_per_class, seed=seed, device="cpu", dataset="mnist")
        proxy_idx = set(int(i) for i in iq)
        if priv_idx.isdisjoint(proxy_idx):
            return (xp.to(torch.float64), yp.to(torch.float64), priv_idx,
                    xq.to(torch.float64), yq.to(torch.float64), proxy_idx)
        seed += 1
    raise RuntimeError("could not draw a proxy set disjoint from the private set in 50 tries")


def _private_draw_images(get_finetuning_data, N, draw_idx, forbid_idx):
    """A FRESH private image subset for one independent p_true-draw (different seed per draw ->
    genuinely independent draws for the bootstrap, not pseudo-replicated re-labelings of one set).
    Kept disjoint from the proxy index set. Returns (images, labels) as cpu float64; the labels
    are returned WITH the images so a seed advanced for disjointness never desyncs x from y."""
    n_per_class = N // 2
    seed = PRIVATE_BASE_SEED + 1 + 101 * draw_idx
    for _ in range(50):
        x, y, d, idx = get_finetuning_data(n_per_class, seed=seed, device="cpu", dataset="mnist")
        if forbid_idx.isdisjoint(set(int(i) for i in idx)):
            return x.to(torch.float64), y.to(torch.float64)
        seed += 1
    raise RuntimeError(f"draw {draw_idx}: could not draw a private subset disjoint from the proxy")


# --------------------------------------------------------------------------- #
# COMPUTE PATH — known-recipe ΔW-matching recovery of the shared parameter.     #
# --------------------------------------------------------------------------- #
def run_shared_perturbation_experiment(kind, grid, n_draws, K_seeds, N, lr, T, rank,
                                       device, n_boot=10000, seed=0):
    """Recover the shared transform parameter by known-recipe ΔW-matching.

    Pipeline (per transform `kind`, grid of candidate parameters `grid`):
      1. Load frozen base θ0 (checkpoint) + a DISJOINT proxy image set.
      2. PROXY GRID (attacker asset, computed ONCE): for each candidate p, transform the
         PROXY images by p, center by that set's ds_mean, fine-tune K adapters (K B0 seeds)
         under the identical recipe -> seed-averaged ΔW_proxy(p).
      3. For each of n_draws INDEPENDENT p_true-draws: draw p_true uniformly from the grid,
         draw a FRESH private image subset, transform it by p_true, fine-tune K adapters ->
         seed-averaged ΔW_obs, then p_hat = argmax_p cosine(ΔW_obs, ΔW_proxy(p)).
      4. Score recovery_error, skill vs the transform-blind baseline, bootstrap the skill CI
         over draws, and emit the RECOVERABLE / honest-null verdict.

    Returns a result dict (grid, per-draw p_true/p_hat/err, cosine curves, skill+CI, verdict).
    """
    import torch.nn.functional as F  # noqa: F401 (used via _cos_flat)
    from experiments.jacobian_spectrum import _honest_target, make_activation
    from experiments.data_utils import get_finetuning_data
    from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0

    torch.set_default_dtype(torch.float64)
    act = make_activation("gelu")
    seeds = [1000 + j for j in range(K_seeds)]

    # 1) disjoint private/proxy sets + honest frozen θ0 (checkpoint; set-independent).
    (x_priv0, y_priv0, priv_idx, x_proxy, y_proxy, proxy_idx) = _load_disjoint_sets(
        get_finetuning_data, N, device)
    # frozen/b0 come from the checkpoint (independent of the fine-tune set); derive once.
    _, frozen, b0, _, _ = _honest_target(x_priv0.to(device), y_priv0.to(device),
                                         T, rank, "gelu", lr, device, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    y_proxy_d = y_proxy.to(device)

    print(f"[F5 {kind}] N={N} grid={grid} K_seeds={K_seeds} n_draws={n_draws} "
          f"T={T} lr={lr} rank={rank}", flush=True)
    print(f"[F5 {kind}] private/proxy index overlap = {len(priv_idx & proxy_idx)} (must be 0)",
          flush=True)

    # 2) proxy grid ΔW_proxy(p) — computed ONCE (attacker asset), reused across draws.
    proxy_dW = []
    proxy_mbce = []
    for p in grid:
        xp = apply_shared_transform(x_proxy, kind, p).to(device)   # [N,1,28,28]
        dsm = xp.mean(dim=0, keepdim=True)
        x0 = xp.reshape(xp.shape[0], -1) - dsm.reshape(1, -1)
        dW, mbce, nfin = _mean_dW(train_adapter, draw_B0, frozen, b0, x0, y_proxy_d,
                                  lr, T, act, rank, out_f, seeds, device)
        assert dW is not None, f"proxy candidate p={p}: all {K_seeds} seeds non-finite"
        proxy_dW.append(dW)
        proxy_mbce.append(mbce)
    print(f"[F5 {kind}] proxy grid built ({len(grid)} candidates); "
          f"mean proxy max_bce={sum(proxy_mbce)/len(proxy_mbce):.2e}", flush=True)

    # 3) independent p_true-draws.
    rng = random.Random(seed)
    p_true_draws, p_hat_draws, errs, cos_curves, obs_mbce = [], [], [], [], []
    for d in range(n_draws):
        p_true = rng.choice(list(grid))
        x_priv, y_priv = _private_draw_images(get_finetuning_data, N, d, proxy_idx)
        y_priv_d = y_priv.to(device)

        xt = apply_shared_transform(x_priv, kind, p_true).to(device)
        dsm = xt.mean(dim=0, keepdim=True)
        x0 = xt.reshape(xt.shape[0], -1) - dsm.reshape(1, -1)
        dW_obs, mbce, nfin = _mean_dW(train_adapter, draw_B0, frozen, b0, x0, y_priv_d,
                                      lr, T, act, rank, out_f, seeds, device)
        assert dW_obs is not None, f"draw {d}: all {K_seeds} obs seeds non-finite"
        curve = [_cos_flat(dW_obs, dWp) for dWp in proxy_dW]
        j_hat = max(range(len(grid)), key=lambda j: curve[j])
        p_hat = grid[j_hat]
        err = recovery_error(kind, p_hat, p_true)
        p_true_draws.append(p_true)
        p_hat_draws.append(p_hat)
        errs.append(err)
        cos_curves.append(curve)
        obs_mbce.append(mbce)
        print(f"[F5 {kind}] draw {d}: p_true={p_true:+.2f} p_hat={p_hat:+.2f} "
              f"err={err:.3f} peak_cos={curve[j_hat]:.4f} (obs max_bce={mbce:.2e})", flush=True)

    # 4) skill + bootstrap CI over draws (the pinned-prior transform-blind baseline).
    p_blind = blind_prior_mean(kind, grid)
    base_errs = blind_baseline_error(kind, p_true_draws, p_blind)
    point, lo, hi = bootstrap_skill_ci(errs, base_errs, n_boot=n_boot, alpha=0.05, seed=seed)
    verdict = skill_verdict(lo)
    analytic_base = analytic_blind_error(kind, grid)

    res = dict(
        kind=kind, grid=list(grid), N=N, K_seeds=K_seeds, n_draws=n_draws, T=T, lr=lr, rank=rank,
        prior_support=[min(grid), max(grid)], prior_mean=p_blind,
        p_true_draws=p_true_draws, p_hat_draws=p_hat_draws,
        recovery_errors=errs, baseline_errors=base_errs, cos_curves=cos_curves,
        mean_error=sum(errs) / len(errs), mean_baseline_error=sum(base_errs) / len(base_errs),
        analytic_blind_error=analytic_base,
        skill=point, skill_lo=lo, skill_hi=hi, verdict=verdict,
        obs_mean_max_bce=sum(obs_mbce) / len(obs_mbce),
        proxy_mean_max_bce=sum(proxy_mbce) / len(proxy_mbce),
    )
    print(f"\n[F5 {kind}] === SKILL {point:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  "
          f"(mean_err={res['mean_error']:.3f} vs blind={res['mean_baseline_error']:.3f}, "
          f"analytic_blind={analytic_base:.3f}) ===", flush=True)
    print(f"[F5 {kind}] VERDICT: {verdict}", flush=True)
    print(f"[F5 {kind}] HONEST CAVEAT: a flat cosine-vs-p curve (no peak at p_true) means "
          f"proxy-vs-private IMAGE mismatch swamped the transform signal -> skill~0 null.",
          flush=True)
    return res


# --------------------------------------------------------------------------- #
# figure — (a) cosine-vs-p curve, (b) skill±CI per transform, (c) recovered-vs-true scatter
# --------------------------------------------------------------------------- #
def make_figure(results_by_kind, out_path):
    """results_by_kind: {kind: result-dict}. Best-effort (a plotting failure must not kill
    the metrics — the result .pth is saved regardless)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        kinds = [k for k in ("rot", "blur") if k in results_by_kind]
        nrows = max(len(kinds), 1)
        fig = plt.figure(figsize=(15, 4.4 * nrows + 1.4))
        gs = GridSpec(nrows, 3, figure=fig, hspace=0.42, wspace=0.30,
                      left=0.06, right=0.985, top=0.90, bottom=0.20)
        unit = {"rot": "deg", "blur": "sigma"}
        for r, kind in enumerate(kinds):
            res = results_by_kind[kind]
            grid = res["grid"]
            # representative draw = the one with the MEDIAN recovery error (typical, not cherry-picked).
            errs = res["recovery_errors"]
            order = sorted(range(len(errs)), key=lambda i: errs[i])
            rep = order[len(order) // 2]
            curve = res["cos_curves"][rep]
            p_true = res["p_true_draws"][rep]
            p_hat = res["p_hat_draws"][rep]

            # (a) cosine(ΔW_obs, ΔW_proxy(p)) vs p — representative draw.
            ax = fig.add_subplot(gs[r, 0])
            ax.plot(grid, curve, "-o", color="#1f77b4", ms=4, lw=1.6)
            ax.axvline(p_true, color="#2ca02c", ls="--", lw=1.6, label=f"p_true={p_true:g}")
            j_hat = max(range(len(grid)), key=lambda j: curve[j])
            ax.plot([grid[j_hat]], [curve[j_hat]], "*", color="#d62728", ms=15,
                    label=f"p_hat={p_hat:g}")
            ax.set_xlabel(f"candidate p ({unit[kind]})")
            ax.set_ylabel("cosine(ΔW_obs, ΔW_proxy(p))")
            ax.set_title(f"({'ab'[r]}) {kind}: ΔW-match cosine vs p (rep. draw)", fontsize=10)
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.3)

            # (c) recovered-vs-true scatter across p_true draws.
            ax2 = fig.add_subplot(gs[r, 2])
            ax2.plot([min(grid), max(grid)], [min(grid), max(grid)], "k--", lw=1, alpha=0.6,
                     label="perfect")
            ax2.scatter(res["p_true_draws"], res["p_hat_draws"], color="#9467bd",
                        s=45, alpha=0.8, edgecolor="k", linewidth=0.4)
            ax2.set_xlabel(f"p_true ({unit[kind]})")
            ax2.set_ylabel(f"p_hat ({unit[kind]})")
            ax2.set_title(f"({'cd'[r]}) {kind}: recovered vs true "
                          f"(n={res['n_draws']} draws)", fontsize=10)
            ax2.legend(fontsize=8, loc="best")
            ax2.grid(alpha=0.3)

        # (b) skill ± CI per transform — spanning the middle column.
        axb = fig.add_subplot(gs[:, 1])
        xs = list(range(len(kinds)))
        pts = [results_by_kind[k]["skill"] for k in kinds]
        los = [results_by_kind[k]["skill_lo"] for k in kinds]
        his = [results_by_kind[k]["skill_hi"] for k in kinds]
        yerr = [[max(p - lo, 0) for p, lo in zip(pts, los)],
                [max(hi - p, 0) for p, hi in zip(pts, his)]]
        colors = ["#2ca02c" if lo > 0 else "#7f7f7f" for lo in los]
        axb.axhline(0.0, color="#d62728", ls="--", lw=1.4, label="blind baseline (skill=0)")
        axb.errorbar(xs, pts, yerr=yerr, fmt="none", ecolor="k", elinewidth=1.6, capsize=6, zorder=2)
        axb.scatter(xs, pts, c=colors, s=120, zorder=3, edgecolor="k", linewidth=0.6)
        for x, k in zip(xs, kinds):
            res = results_by_kind[k]
            axb.annotate(res["verdict"].split(" (")[0], (x, res["skill"]),
                         textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8)
        axb.set_xticks(xs)
        axb.set_xticklabels([f"{k}\n{unit[k]}" for k in kinds])
        axb.set_ylabel("recovery skill = 1 - err/blind   (95% bootstrap CI over p_true draws)")
        axb.set_title("(b) recovery SKILL ± CI per transform\n(RECOVERABLE iff CI lower bound > 0)",
                      fontsize=10)
        axb.set_ylim(min(-1.05, min(los) - 0.1), max(1.05, max(his) + 0.1))
        axb.legend(fontsize=8, loc="lower right")
        axb.grid(alpha=0.3)

        fig.suptitle("F5 - Recovering the SHARED transform parameter (known-recipe ΔW-matching)",
                     fontsize=13, y=0.965)
        fig.text(0.5, 0.015, CAPTION, ha="center", va="bottom", fontsize=8.0, wrap=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] saved {out_path}", flush=True)
    except Exception as e:  # noqa: BLE001 — best effort; tensors/JSON persist regardless
        print(f"WARNING: F5 figure failed ({type(e).__name__}: {e}) — "
              "results JSON/pth are still saved.", flush=True)


def _default_grid(kind):
    if kind == "rot":
        return [float(v) for v in range(-40, 41, 5)]          # symmetric -> prior mean 0
    return [round(0.5 + 0.5 * i, 3) for i in range(8)]         # sigma 0.5..4.0 (sigma>0: blur is defined)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="F5 shared-perturbation recovery (known-recipe ΔW-matching; bsub-only).")
    ap.add_argument("--transform", choices=["rotation", "blur", "both"], default="both",
                    help="which shared transform(s) to recover.")
    ap.add_argument("--grid", type=float, nargs="+", default=None,
                    help="candidate-parameter grid (deg for rotation, sigma>0 for blur). "
                         "Default: rotation -40..40 step5; blur 0.5..4.0 step0.5.")
    ap.add_argument("--n_draws", type=int, default=8,
                    help="INDEPENDENT p_true-draws (the bootstrap unit; >=8 for the full run).")
    ap.add_argument("--K_seeds", type=int, default=3,
                    help="B0-init seeds averaged for ΔW_obs and each ΔW_proxy(p).")
    ap.add_argument("--N", type=int, default=16, help="private (and proxy) set size.")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage0", action="store_true",
                    help="TINY gate: N=6, 1 p_true-draw, coarse grid, K_seeds=2. Asserts a finite "
                         "skill and a real peak in the ΔW-cosine curve, then exits.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=MEETING_FIG)
    args = ap.parse_args()

    print("[F5] caption:", flush=True)
    print("     " + CAPTION, flush=True)

    kinds = {"rotation": ["rot"], "blur": ["blur"], "both": ["rot", "blur"]}[args.transform]

    if args.stage0:
        # TINY: exercise the full pipeline cheaply on ONE transform + coarse grid.
        kind = kinds[0]
        grid = args.grid or ([-40.0, -20.0, 0.0, 20.0, 40.0] if kind == "rot"
                             else [0.5, 1.5, 2.5, 3.5])
        print(f"=== STAGE-0 (TINY): kind={kind} N=6 n_draws=1 K_seeds=2 grid={grid} ===", flush=True)
        res = run_shared_perturbation_experiment(
            kind=kind, grid=grid, n_draws=1, K_seeds=2, N=6, lr=args.lr, T=args.T,
            rank=args.rank, device=args.device, n_boot=200, seed=args.seed)
        # gate assertions
        assert math.isfinite(res["skill"]), "STAGE-0: skill non-finite"
        curve = res["cos_curves"][0]
        assert all(math.isfinite(c) for c in curve), "STAGE-0: cosine curve has non-finite entries"
        peak = max(curve)
        mean_c = sum(curve) / len(curve)
        assert peak > mean_c + 1e-6, (
            f"STAGE-0: ΔW-cosine curve has NO peak (max={peak:.4f} <= mean={mean_c:.4f}); "
            "the matching statistic is flat — pipeline/metric broken.")
        assert res["cos_curves"][0].index(peak) == max(
            range(len(curve)), key=lambda j: curve[j]), "STAGE-0: argmax bookkeeping inconsistent"
        print(f"STAGE-0 OK (skill={res['skill']:+.3f}, peak_cos={peak:.4f} > mean={mean_c:.4f})",
              flush=True)
        return

    os.makedirs(RESULTS, exist_ok=True)
    results_by_kind = {}
    for kind in kinds:
        grid = args.grid or _default_grid(kind)
        res = run_shared_perturbation_experiment(
            kind=kind, grid=grid, n_draws=args.n_draws, K_seeds=args.K_seeds, N=args.N,
            lr=args.lr, T=args.T, rank=args.rank, device=args.device,
            n_boot=args.n_boot, seed=args.seed)
        results_by_kind[kind] = res
        torch.save(res, os.path.join(RESULTS, f"f5_{kind}.pth"))
        with open(os.path.join(RESULTS, f"f5_{kind}.json"), "w") as f:
            json.dump({k: v for k, v in res.items() if k != "cos_curves"}, f, indent=2)

    make_figure(results_by_kind, args.out)

    print("\n=== F5 SUMMARY ===", flush=True)
    for kind, res in results_by_kind.items():
        print(f"  {kind:>5}: skill={res['skill']:+.3f} CI[{res['skill_lo']:+.3f},"
              f"{res['skill_hi']:+.3f}]  {res['verdict']}", flush=True)
    print("READ (OBSERVE, not conclude): skill CI lower bound > 0 => the SHARED transform IS "
          "recoverable by the known-recipe attacker (STRUCTURAL leakage beyond the weakest "
          "per-image attacker — an EXTENSION, not a contradiction). CI straddling 0 => NOT "
          "recoverable above the transform-blind baseline (honest null; report it). Likely null "
          "driver to flag: proxy-vs-private image mismatch swamping the transform signal in ΔW.",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
