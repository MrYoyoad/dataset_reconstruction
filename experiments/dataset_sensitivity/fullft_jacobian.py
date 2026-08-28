#!/usr/bin/env python
"""Arm G — the Jacobian comparison J_full vs J_LoRA (full-FT valley-width plan §2.2).

Measures the LOCAL differential geometry of the fine-tuning map, in BOTH
parameterizations, on the SAME secret tangent directions ``a``:

    J_LoRA = ∂ vec(A_T, B_T) / ∂ a        (the existing jacobian_spectrum object)
    J_full = ∂ vec(Δθ_full) / ∂ a         (THIS arm — the full-parameter analogue)

where the private data is hidden inside realistic image variations
``x_i(a_i) = x_i^0 + U_i a_i`` and the two maps share the SAME θ₀ (`_honest_target`
checkpoint), the SAME data D, and the SAME tangent basis U — the ONLY change is the
parameterization (all-layer / layer-0 full-rank Δθ vs rank-8 single-layer BA). This
is the like-for-like companion of the finite-distance distance-dial (arm A/C/D):
J's small singular directions APPROXIMATE the flat valley under local-linearity
(plan TF6), so the spectra of J_full vs J_LoRA are the infinitesimal version of the
s(d) profiles.

READOUT PRIORITY (plan §2.2 / audit S4 / TB4):
  * P7 LEADS — the RAW (noise-free) singular spectra of J_full vs J_LoRA plus the
    targeted ratio ‖J·a_nn‖/‖J·a_far‖ per regime (a_nn = tangent toward the
    near-duplicate, a_far = toward the far anchor: the linearized s(d)). RAW is the
    PRIMARY, noise-free comparison.
  * P6 DEMOTED, EXPLICITLY T=5-CONDITIONAL — r_J (hard rank) is a CONSISTENCY check,
    NOT a ≥2× claim: at converged T both saturate at Nk (count set by the DATA, §I.4);
    any full>LoRA gap at T=5 is an underfit artifact (eff_rank ~9-13 < 32).
  * SNR-whitened q_eff is P4-CONDITIONAL SECONDARY (uses each regime's own ε/B0
    noise Σ) and inherits the D2/TF7 conditionality — NOT built here (the raw
    readouts are the make-or-break primary; q_eff is a downstream rescore that reuses
    jacobian_spectrum.snr_spectrum / estimate_sigma_seed once P4 clears).

MANDATORY T-SWEEP T={1,5,20} (audit S4/Q8): the spectral SHAPE and the a_nn/a_far
ratio must be T-stable before any small-T J is trusted against the T=1000 valley.
Every readout carries the caveat **"early-training Jacobian, not converged-valley."**
max_bce is reported at each T.

Reduced config (plan §2.2): N=4, k=8 (Nk=32), **layer-0 J_full PRIMARY** (like-for-
like with J_LoRA on the same weight block), all-layer J_full as the stretch/optional.
GELU MANDATORY (the create_graph double-backward through the unroll needs C^∞ — never
modified_relu, the standing constraint). float64.

Usage (WEXAC, bsub-only — NEVER run locally):
    python -u -m experiments.dataset_sensitivity.fullft_jacobian --stage0 --device cuda
    python -u -m experiments.dataset_sensitivity.fullft_jacobian --tsweep --save --device cuda
    python -u -m experiments.dataset_sensitivity.fullft_jacobian --compare --T 5 --save --device cuda
    #   --all_layers  → the all-layer J_full stretch (heavier; see the cost note below).

Coordination with the sibling fullft_valley.py: this module implements its OWN minimal
full-param unrolled training (`unrolled_full_theta`) because arm G needs the unroll
differentiable w.r.t. the input-perturbation coords a (create_graph=True through the T
steps), which fullft_valley.train_full (plain torch.optim.SGD) is not. The two MUST be
the SAME operator at a=0 / create_graph=False, or the cross-arm comparison is void —
this is enforced by `_parity_gate()`: a HARD on-node assert run at the TOP of stage0()
(NOT a comment). It runs both trainers on a tiny ε-off/batch-off config and asserts
allclose(atol=1e-9, rtol=1e-7). They agree to machine precision (both = w−lr·grad, BCE,
W init at frozen.clone(), biases frozen; train_full's forward_logits(empty adapter) ==
_full_forward), so no reconciliation of _full_forward was needed. TODO(unify): fold
unrolled_full_theta onto a shared create_graph core with train_full (make parity
structural, not just tested).
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'dataset_reconstruction'))

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

from experiments.configs import RESULTS_DIR, FIGURES_DIR, TRAIN_LR
from experiments.gate_matrix_test import effective_rank
from experiments.jacobian_spectrum import (
    _mnist_ctx, exact_jacobian, make_images,
    unrolled_lora_AB, finetune_metrics,
)

RESULTS = os.path.join(RESULTS_DIR, 'fullft_jacobian')


# ---------------------------------------------------------------------------
# 1. Differentiable full-batch FULL-PARAM unrolled training (returns Δθ)
#    The full-parameter analogue of jacobian_spectrum.unrolled_lora_AB.
# ---------------------------------------------------------------------------
def _full_forward(frozen, W, b0, x, act, trainable_layers):
    """MLP forward with FULL weights: trainable layers use the live W[l], the rest
    stay frozen. Layer 0 carries the (frozen) bias; deeper layers bias-free;
    activation after every layer except the last (a raw logit / [N,K] out).

    Mirrors _partial_lora_forward, but with direct weights instead of frozen+B@A —
    this IS the single-variable change (parameterization) the arm measures.
    """
    n_layers = len(frozen)
    h = x.view(x.shape[0], -1)
    for l in range(n_layers):
        w = W[l] if l in trainable_layers else frozen[l]
        bias = b0 if l == 0 else None
        h = F.linear(h, w, bias)
        if l < n_layers - 1:
            h = act(h)
    return h


def unrolled_full_theta(frozen, b0, x, y, lr, T, act, trainable_layers,
                        num_classes=2, create_graph=True):
    """Unroll T full-batch SGD steps on the FULL weight matrices; RETURN Δθ dict.

    The full-parameter sibling of unrolled_lora_AB. How it DIFFERS from the LoRA
    unroll (the reference), point-by-point:
      * INIT: LoRA starts A=0, B=B0 (ΔW=0, function-preserving at t=0); here W
        starts AT θ₀ (frozen[l].clone()) so Δθ=0 at t=0 trivially. Biases stay
        FROZEN (weights only, mirroring LoRA which trains only W — plan D4).
      * PARAMS differentiated: LoRA optimizes A,B (rank·(in+out) params ⇒ the
        low-rank projection Bᵀ∇_W / ∇_W Aᵀ of the gradient); here we optimize the
        full W[l] (out·in params ⇒ the RAW ∇_W). This is the whole point.
      * OUTPUT: LoRA returns (A_T,B_T) ⇒ vecY dim rank·(in+out); here Δθ =
        W_T − frozen ⇒ dim Σ_ℓ out_ℓ·in_ℓ (784k layer-0, 1.79M all-layer).
      * NOISE for Σ (not used by the RAW primary): LoRA reseeds B0 (path
        randomization); full-FT would perturb θ₀ by ε·ξ (plan D2) — a separate
        P4-conditional secondary, not built here.

    create_graph=True keeps Δθ twice-differentiable w.r.t. a (needed by
    exact_jacobian_full); False = value-only single-backward (identical Δθ values,
    far cheaper) for max_bce / Σ sampling.
    """
    W = {l: frozen[l].clone().requires_grad_(True) for l in trainable_layers}
    for _ in range(T):
        out = _full_forward(frozen, W, b0, x, act, trainable_layers)
        if num_classes <= 2:
            loss = F.binary_cross_entropy_with_logits(out.view(-1), y)
        else:
            assert not torch.is_floating_point(y) or (y == y.round()).all(), \
                "multi-class y must be integer class indices"
            loss = F.cross_entropy(out, y.long())
        params = [W[l] for l in trainable_layers]
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
        W = {l: W[l] - lr * grads[i] for i, l in enumerate(trainable_layers)}
    # Δθ = θ_T − θ₀ (attacker-visible gauge; frozen[l] is a leaf constant here).
    return {l: W[l] - frozen[l] for l in trainable_layers}


def build_theta_index(frozen, trainable_layers):
    """Ascending-layer flatten layout of Δθ. Returns list of (layer, shape, slice)
    so a flat J_full row can be decomposed back into per-layer blocks."""
    index, off = [], 0
    for l in trainable_layers:
        n = frozen[l].numel()
        index.append((l, tuple(frozen[l].shape), slice(off, off + n)))
        off += n
    return index


def flatten_theta(dtheta, theta_index):
    """Flatten Δθ to a single vector following theta_index (ascending layer)."""
    return torch.cat([dtheta[l].reshape(-1) for (l, _s, _sl) in theta_index])


# ---------------------------------------------------------------------------
# 2. The forward map a -> vec(Δθ_full) and its exact Jacobian
# ---------------------------------------------------------------------------
class CtxFull:
    """Everything forward_theta needs, fixed at a=0. Shares frozen/b0/x0_centered/U/y
    with the paired LoRA ctx so J_full and J_LoRA live on the SAME tangents."""
    def __init__(self, frozen, b0, x0_centered, U, y, lr, T, act,
                 trainable_layers, theta_index, num_classes=2):
        self.frozen = frozen
        self.b0 = b0
        self.x0_centered = x0_centered
        self.U = U
        self.y = y
        self.lr = lr
        self.T = T
        self.act = act
        self.trainable_layers = trainable_layers
        self.theta_index = theta_index
        self.num_classes = num_classes


def forward_theta(a_flat, ctx, create_graph=True):
    """The ℝ^{Nk} -> ℝ^{dimΘ} closure. a_flat is [N*k] (row-major over images).
    create_graph=False = value-only unroll (same Δθ values, far cheaper)."""
    N, k = ctx.U.shape[0], ctx.U.shape[2]
    a = a_flat.view(N, k)
    x = make_images(ctx.x0_centered, ctx.U, a)
    dtheta = unrolled_full_theta(ctx.frozen, ctx.b0, x, ctx.y, ctx.lr, ctx.T,
                                 ctx.act, ctx.trainable_layers,
                                 num_classes=ctx.num_classes,
                                 create_graph=create_graph)
    return flatten_theta(dtheta, ctx.theta_index)


def exact_jacobian_full(a0, ctx):
    """J_full = ∂vec(Δθ)/∂a at a0, shape [dimΘ, Nk], via the SAME forward-over-reverse
    jvp_double pattern as jacobian_spectrum.exact_jacobian (two autograd.grad calls,
    composing with the create_graph unroll). One cheap backward per a-coordinate;
    dimΘ >> Nk so forward-mode wins.

    GELU guard (the standing constraint): modifiedrelu has no double-backward → the
    exact J is invalid. GELU is C^∞.
    """
    from CreateModel import ModifiedRelu
    if isinstance(ctx.act, ModifiedRelu):
        raise ValueError("modifiedrelu has no double-backward → no exact J_full. "
                         "Use gelu (the mandatory activation for this arm).")
    a = a0.clone().detach().requires_grad_(True)
    Y = forward_theta(a, ctx)
    dimY, Nk = Y.numel(), a.numel()
    u = torch.zeros(dimY, dtype=a.dtype, device=a.device, requires_grad=True)
    (JtU,) = torch.autograd.grad(Y, a, grad_outputs=u, create_graph=True)
    cols, eye = [], torch.eye(Nk, dtype=a.dtype, device=a.device)
    for j in range(Nk):
        (col,) = torch.autograd.grad(JtU, u, grad_outputs=eye[j],
                                     retain_graph=(j < Nk - 1))
        cols.append(col.detach())
    return torch.stack(cols, dim=1)                    # [dimΘ, Nk]


def fd_jacobian_full(a0, ctx, coords, eps=1e-5):
    """Central-difference columns of J_full for the given coords (the FD gate).
    No torch.no_grad(): forward_theta runs the inner SGD via create_graph autograd;
    we detach the value only (same pattern as jacobian_spectrum.finite_difference_jacobian)."""
    cols = {}
    for j in coords:
        e = torch.zeros_like(a0); e[j] = 1.0
        Yp = forward_theta(a0 + eps * e, ctx, create_graph=False).detach()
        Ym = forward_theta(a0 - eps * e, ctx, create_graph=False).detach()
        cols[j] = (Yp - Ym) / (2 * eps)
    return cols


# ---------------------------------------------------------------------------
# 3. Readouts: hard rank, per-layer col(J) energy, the local valley ratio
# ---------------------------------------------------------------------------
def hard_rank(J, tol=1e-8):
    """r_J = # singular values > tol·σ_max (the plan's hard-rank definition)."""
    sv = torch.linalg.svdvals(J.double())
    return int((sv > tol * sv[0]).sum()), sv


def per_layer_energy(J, theta_index, n_vecs=None, tol=1e-8):
    """Energy of each LEFT singular vector of J across the per-layer Δθ blocks.

    Returns a [r, L] matrix: row i = the fraction of ‖u_i‖² carried by each layer
    block, for the top-r left singular vectors (r = hard rank, or n_vecs). Only
    informative when >1 layer is trainable (all-layer J_full); trivial (all energy
    in L0) for the layer-0-only PRIMARY.
    """
    U, s, _ = torch.linalg.svd(J.double(), full_matrices=False)
    r = n_vecs or int((s > tol * s[0]).sum())
    r = max(1, min(r, U.shape[1]))
    layers = [l for (l, _s, _sl) in theta_index]
    E = torch.zeros(r, len(layers), dtype=torch.float64)
    for i in range(r):
        ui = U[:, i]
        for c, (l, _s, sl) in enumerate(theta_index):
            E[i, c] = ui[sl].pow(2).sum()
        E[i] /= (E[i].sum() + 1e-30)
    return E, s[:r], layers


def build_valley_basis(x0_centered, y, digits, near_sigma=0.02, seed=0):
    """Per-image [d,2] tangent basis: column 0 = a_nn (unit direction toward the
    near-duplicate), column 1 = a_far (unit direction toward the far anchor). The
    differential version of the dial's near/far rungs (plan §2.2, P7).

    Encoder-free, deterministic construction (no r_nn/r_far retrieval — that needs
    the similarity_ladder encoder; the sibling can later supply retrieved directions):
      * near-duplicate direction = the p0_noise perturbation (x + near_sigma·N(0,1),
        the dial's explicit 'near-duplicate anchor'); as a unit direction the clip is
        immaterial. Points along a tiny-pixel-distance (fine) move.
      * far-anchor direction = toward the farthest SAME-binary-class DIFFERENT-digit
        image in the batch (the r_cross construction: a different digit of the same
        parity, label-safe). Fallbacks keep the N=2 stage-0 well-defined.

    Directions are offset-invariant, so centering (x0_centered) vs raw is irrelevant.
    Column index for image i: near = i*2+0, far = i*2+1 (a.view(N,2) row-major) ⇒
    ‖J·a_nn(i)‖ = ‖column (2i) of J‖, ‖J·a_far(i)‖ = ‖column (2i+1)‖.
    """
    N = x0_centered.shape[0]
    Xf = x0_centered.reshape(N, -1).double()
    d = Xf.shape[1]
    y = y.view(-1)
    U = torch.zeros(N, d, 2, dtype=torch.float64)
    meta = []
    for i in range(N):
        g = torch.Generator().manual_seed(int(seed) + 1000 + i)
        near_dir = near_sigma * torch.randn(d, generator=g, dtype=torch.float64)
        # far anchor: same binary class, different digit, farthest in pixel space.
        same = [j for j in range(N) if j != i and float(y[j]) == float(y[i])
                and digits[j] != digits[i]]
        if not same:
            same = [j for j in range(N) if j != i and float(y[j]) == float(y[i])]
        if not same:                                    # N=2 one-per-class fallback
            same = [j for j in range(N) if j != i]
        j_far = max((( (Xf[j] - Xf[i]).norm().item()), j) for j in same)[1]
        far_dir = Xf[j_far] - Xf[i]
        U[i, :, 0] = near_dir / (near_dir.norm() + 1e-30)
        U[i, :, 1] = far_dir / (far_dir.norm() + 1e-30)
        meta.append({'i': i, 'j_far': j_far, 'y': float(y[i]),
                     'digit': int(digits[i]), 'digit_far': int(digits[j_far])})
    return U.to(x0_centered.device), meta


def valley_ratio(J):
    """Given J built on the valley basis ([dimY, 2N], columns [near0,far0,near1,...]),
    return (mean_ratio, per_image_ratio, near_norms, far_norms) where
    per_image_ratio[i] = ‖J·a_nn(i)‖ / ‖J·a_far(i)‖ — the linearized near/far dial."""
    Nk = J.shape[1]
    N = Nk // 2
    near = torch.stack([J[:, 2 * i].norm() for i in range(N)])
    far = torch.stack([J[:, 2 * i + 1].norm() for i in range(N)])
    ratio = near / (far + 1e-30)
    return float(ratio.mean()), ratio, near, far


# ---------------------------------------------------------------------------
# 4. Context builders (shared θ₀/D/U across the two regimes) + max_bce
# ---------------------------------------------------------------------------
def build_contexts(N, k, T, rank=8, activation='gelu', seed=42, lr=TRAIN_LR,
                   device='cpu', dataset='mnist', tangent_method='qr',
                   trainable_layers=(0,)):
    """Build the paired (ctx_lora, ctx_full) on the SAME θ₀, D, and tangent basis U.

    The LoRA ctx is built by jacobian_spectrum._mnist_ctx (the single source of the
    honest θ₀ path); the full-FT ctx reuses that ctx's frozen/b0/x0_centered/U/y so
    the ONLY difference between J_LoRA and J_full is the parameterization. Returns
    (ctx_lora, ctx_full, digits, ds_mean).
    """
    ctx_lora, col_scales, digits, ds_mean = _mnist_ctx(
        N=N, k=k, T=T, rank=rank, activation=activation, seed=seed, lr=lr,
        device=device, tangent_method=tangent_method, dataset=dataset)
    theta_index = build_theta_index(ctx_lora.frozen, trainable_layers)
    ctx_full = CtxFull(ctx_lora.frozen, ctx_lora.b0, ctx_lora.x0_centered,
                       ctx_lora.U, ctx_lora.y, ctx_lora.lr, T, ctx_lora.act,
                       trainable_layers, theta_index)
    return ctx_lora, ctx_full, digits, ds_mean


def full_ft_max_bce(ctx_full):
    """max per-sample BCE of the full-FT fine-tune at a=0 (the memorization gate /
    the 'is this T underfit?' readout). Value-only (create_graph=False)."""
    N, k = ctx_full.U.shape[0], ctx_full.U.shape[2]
    a0 = torch.zeros(N * k, dtype=torch.float64, device=ctx_full.U.device)
    x = make_images(ctx_full.x0_centered, ctx_full.U, a0.view(N, k))
    # re-run the unroll value-only, then read per-sample loss at the endpoint.
    W = {l: ctx_full.frozen[l].clone() for l in ctx_full.trainable_layers}
    with torch.no_grad():
        for _ in range(ctx_full.T):
            # value-only manual SGD (no graph) — mirrors unrolled_full_theta.
            with torch.enable_grad():
                Wg = {l: W[l].clone().requires_grad_(True)
                      for l in ctx_full.trainable_layers}
                out = _full_forward(ctx_full.frozen, Wg, ctx_full.b0, x,
                                    ctx_full.act, ctx_full.trainable_layers)
                loss = F.binary_cross_entropy_with_logits(out.view(-1), ctx_full.y)
                grads = torch.autograd.grad(
                    loss, [Wg[l] for l in ctx_full.trainable_layers])
            W = {l: W[l] - ctx_full.lr * grads[i]
                 for i, l in enumerate(ctx_full.trainable_layers)}
        out = _full_forward(ctx_full.frozen, W, ctx_full.b0, x, ctx_full.act,
                            ctx_full.trainable_layers)
        per = F.binary_cross_entropy_with_logits(out.view(-1), ctx_full.y,
                                                 reduction='none')
    return float(per.max()), float(per.mean())


def lora_max_bce(ctx_lora):
    """max per-sample BCE of the LoRA fine-tune at a=0 (paired readout)."""
    N, k = ctx_lora.U.shape[0], ctx_lora.U.shape[2]
    a0 = torch.zeros(N, k, dtype=torch.float64, device=ctx_lora.U.device)
    x = make_images(ctx_lora.x0_centered, ctx_lora.U, a0)
    A, B = unrolled_lora_AB(ctx_lora.frozen, ctx_lora.b0, ctx_lora.B0, x, ctx_lora.y,
                            ctx_lora.lr, ctx_lora.T, ctx_lora.scaling, ctx_lora.act,
                            ctx_lora.target_layers, create_graph=False)
    m = finetune_metrics(ctx_lora.frozen, ctx_lora.b0,
                         {l: A[l].detach() for l in A}, {l: B[l].detach() for l in B},
                         ctx_lora.x0_centered, ctx_lora.y, ctx_lora.scaling,
                         ctx_lora.act, ctx_lora.target_layers)
    return m['max_bce'], m['mean_bce']


# ---------------------------------------------------------------------------
# 5. Pre-registered predictions (printed FIRST, verbatim)
# ---------------------------------------------------------------------------
def print_predictions():
    print("=" * 78)
    print("ARM G — PRE-REGISTERED PREDICTIONS (printed BEFORE any number; plan §3)")
    print("=" * 78)
    print("[P7 — LEADS] The singular spectrum of J_full decays SLOWER along")
    print("  similarity directions than J_LoRA's; concretely ‖J·a_nn‖/‖J·a_far‖ is")
    print("  LARGER for J_full (full resolves finer image-distinctions). J's small")
    print("  singular dirs approximate the flat valley UNDER LOCAL-LINEARITY over the")
    print("  valley scale — which the dial-vs-J comparison itself tests (TF6).")
    print("  KILL: spectra identical up to global scale ⇒ the valley difference is NOT")
    print("  local-geometric (nonlinear/finite-distance); dial (P1) and J disagree —")
    print("  THIS DISSOCIATION IS ITSELF THE FINDING (Fisher-bridge §I.3 inherits it).")
    print("[P6 — DEMOTED, T=5-CONDITIONAL consistency check] In the early-training")
    print("  rank-deficiency window (eff_rank ≈ 9-13 < Nk=32) r_J(full) MAY exceed")
    print("  r_J(LoRA). The LIKELY CONVERGED outcome is BOTH saturating at Nk — count")
    print("  set by the DATA, not the parameterization (§I.4). NO ≥2× claim; any")
    print("  full>LoRA gap at T=5 is an underfit artifact. A converged r_J<Nk in ONE")
    print("  regime only would be the sole surprising count result — flag it.")
    print("[CAVEAT — on EVERY G readout] early-training Jacobian (T∈{1,5,20}),")
    print("  NOT the converged (T=1000) valley. Trust a small-T J only after the")
    print("  T-sweep shows the SHAPE and the a_nn/a_far ratio are T-stable.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# 6. Compare at one T, and the mandatory T-sweep
# ---------------------------------------------------------------------------
def run_compare(N=4, k=8, T=5, rank=8, activation='gelu', device='cuda',
                dataset='mnist', seed=42, lr=TRAIN_LR, trainable_layers=(0,),
                tol=1e-8, verbose=True):
    """Build J_full and J_LoRA on the SAME qr tangents (spectrum, r_J, per-layer
    col(J_full) energy) AND on the valley basis (the a_nn/a_far ratio). Returns a
    dict; RAW / noise-free throughout (the P7 primary)."""
    ctx_lora, ctx_full, digits, ds_mean = build_contexts(
        N, k, T, rank=rank, activation=activation, seed=seed, lr=lr,
        device=device, dataset=dataset, tangent_method='qr',
        trainable_layers=trainable_layers)
    Nk = N * k
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)

    # --- spectra on the shared qr basis ---
    J_lora = exact_jacobian(a0, ctx_lora, method='jvp_double')
    J_full = exact_jacobian_full(a0, ctx_full)
    rJ_lora, sv_lora = hard_rank(J_lora, tol)
    rJ_full, sv_full = hard_rank(J_full, tol)
    er_lora, er_full = effective_rank(J_lora), effective_rank(J_full)

    # --- per-layer col(J_full) energy (informative for all-layer) ---
    E_full, s_top, layers = per_layer_energy(J_full, ctx_full.theta_index, tol=tol)

    # --- the local valley ratio ‖J·a_nn‖/‖J·a_far‖ on the valley basis ---
    U_val, val_meta = build_valley_basis(ctx_full.x0_centered, ctx_full.y, digits,
                                         seed=seed)
    ctx_lora_v, ctx_full_v, _dig, _dm = build_contexts(
        N, k, T, rank=rank, activation=activation, seed=seed, lr=lr,
        device=device, dataset=dataset, tangent_method='qr',
        trainable_layers=trainable_layers)
    # swap in the valley tangents (same θ₀/D/y; ONLY U changes) on both ctx. The
    # LoRA AB index is U-independent (input-side), so it carries over unchanged.
    ctx_lora_v.U = U_val
    ctx_full_v.U = U_val
    a0v = torch.zeros(2 * N, dtype=torch.float64, device=device)
    Jv_lora = exact_jacobian(a0v, ctx_lora_v, method='jvp_double')
    Jv_full = exact_jacobian_full(a0v, ctx_full_v)
    r_lora_mean, r_lora_pi, nn_l, far_l = valley_ratio(Jv_lora)
    r_full_mean, r_full_pi, nn_f, far_f = valley_ratio(Jv_full)

    mb_full, mmb_full = full_ft_max_bce(ctx_full)
    mb_lora, mmb_lora = lora_max_bce(ctx_lora)

    if verbose:
        print(f"\n[G] T={T} N={N} k={k} Nk={Nk} layers={tuple(trainable_layers)} "
              f"dataset={dataset} (RAW / noise-free; early-training Jacobian)")
        print(f"[G]   J_full  shape={tuple(J_full.shape)} r_J={rJ_full}/{Nk} "
              f"eff_rank={er_full:.2f}  σ:[{sv_full[0]:.2e},{sv_full[-1]:.2e}] "
              f"max_bce={mb_full:.2e}")
        print(f"[G]   J_LoRA  shape={tuple(J_lora.shape)} r_J={rJ_lora}/{Nk} "
              f"eff_rank={er_lora:.2f}  σ:[{sv_lora[0]:.2e},{sv_lora[-1]:.2e}] "
              f"max_bce={mb_lora:.2e}")
        print(f"[G]   P6 (T=5-cond consistency): r_J full={rJ_full} vs LoRA={rJ_lora} "
              f"(both→Nk={Nk} at converged T; gap here = underfit artifact)")
        print(f"[G]   P7 (LEADS) valley ratio ‖J·a_nn‖/‖J·a_far‖:  "
              f"full={r_full_mean:.3f}  LoRA={r_lora_mean:.3f}  "
              f"(P7 ⇒ full > LoRA)")
        if len(layers) > 1:
            print(f"[G]   per-layer col(J_full) energy (top singular vec): "
                  + "  ".join(f"L{l}={E_full[0, c]:.3f}"
                              for c, l in enumerate(layers)))

    return {
        'T': T, 'N': N, 'k': k, 'rank': rank, 'dataset': dataset, 'seed': seed,
        'trainable_layers': tuple(trainable_layers), 'digits': digits,
        'J_full_shape': tuple(J_full.shape), 'J_lora_shape': tuple(J_lora.shape),
        'sv_full': sv_full.cpu(), 'sv_lora': sv_lora.cpu(),
        'r_J_full': rJ_full, 'r_J_lora': rJ_lora,
        'eff_rank_full': er_full, 'eff_rank_lora': er_lora,
        'per_layer_energy': E_full.cpu(), 'per_layer_layers': layers,
        'valley_ratio_full': r_full_mean, 'valley_ratio_lora': r_lora_mean,
        'valley_ratio_full_perimg': r_full_pi.cpu(),
        'valley_ratio_lora_perimg': r_lora_pi.cpu(),
        'valley_meta': val_meta,
        'max_bce_full': mb_full, 'max_bce_lora': mb_lora,
        'mean_bce_full': mmb_full, 'mean_bce_lora': mmb_lora,
    }


def run_tsweep(N=4, k=8, Ts=(1, 5, 20), rank=8, activation='gelu', device='cuda',
               dataset='mnist', seed=42, lr=TRAIN_LR, trainable_layers=(0,),
               save=False, tag=None):
    """MANDATORY T-sweep (audit S4/Q8): r_J, spectral shape, and the a_nn/a_far ratio
    at each T + max_bce — establishes SHAPE/RATIO T-stability BEFORE trusting the
    small-T J against the T=1000 valley."""
    print_predictions()
    print(f"\n[G-Tsweep] Ts={list(Ts)} N={N} k={k} layers={tuple(trainable_layers)} "
          f"dataset={dataset}  (early-training Jacobian, NOT converged-valley)")
    rows = []
    for T in Ts:
        r = run_compare(N=N, k=k, T=T, rank=rank, activation=activation,
                        device=device, dataset=dataset, seed=seed, lr=lr,
                        trainable_layers=trainable_layers)
        rows.append(r)
    print(f"\n[G-Tsweep] SUMMARY (T-stability of shape & ratio):")
    print(f"[G-Tsweep] {'T':>4s} {'rJ_full':>7s} {'rJ_LoRA':>7s} "
          f"{'ratio_full':>10s} {'ratio_LoRA':>10s} "
          f"{'mbce_full':>9s} {'mbce_LoRA':>9s}")
    for r in rows:
        print(f"[G-Tsweep] {r['T']:4d} {r['r_J_full']:7d} {r['r_J_lora']:7d} "
              f"{r['valley_ratio_full']:10.3f} {r['valley_ratio_lora']:10.3f} "
              f"{r['max_bce_full']:9.2e} {r['max_bce_lora']:9.2e}")
    print("[G-Tsweep] READ: ratio/rJ T-stable ⇒ the small-T J is a trustworthy proxy "
          "for the converged valley; T-drift ⇒ report as underfit-only (P6/caveat).")
    out = {'rows': rows, 'N': N, 'k': k, 'Ts': list(Ts), 'rank': rank,
           'dataset': dataset, 'seed': seed,
           'trainable_layers': tuple(trainable_layers)}
    if save:
        lay = 'L0' if tuple(trainable_layers) == (0,) else 'all'
        tag = tag or f"{dataset}_N{N}_k{k}_r{rank}_{activation}_{lay}_s{seed}"
        os.makedirs(RESULTS, exist_ok=True)
        path = os.path.join(RESULTS, f"fullft_jacobian_tsweep_{tag}.pth")
        torch.save(out, path)
        print(f"[G-Tsweep] saved -> {path}")
        _plot_tsweep(out, os.path.join(FIGURES_DIR, 'fullft_jacobian',
                                       f"tsweep_{tag}.png"))
    return out


def _plot_tsweep(out, save_path):
    """Overlaid singular spectra of J_full vs J_LoRA (per T) + the r_J bars and the
    valley-ratio-vs-T panel (plan plot 6)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = out['rows']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=130)
    for r in rows:
        svf, svl = r['sv_full'], r['sv_lora']
        axes[0].semilogy(range(1, len(svf) + 1), svf, 'o-', ms=3,
                         label=f"full T={r['T']}")
        axes[0].semilogy(range(1, len(svl) + 1), svl, 's--', ms=3,
                         label=f"LoRA T={r['T']}")
    axes[0].set_xlabel('index i'); axes[0].set_ylabel(r'$\sigma_i(J)$')
    axes[0].set_title(f"J spectra: full vs LoRA (Nk={out['N']*out['k']})")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=6)
    Ts = [r['T'] for r in rows]
    axes[1].plot(Ts, [r['r_J_full'] for r in rows], 'o-', label='r_J full')
    axes[1].plot(Ts, [r['r_J_lora'] for r in rows], 's--', label='r_J LoRA')
    axes[1].axhline(out['N'] * out['k'], color='k', ls=':', label='Nk ceiling')
    axes[1].set_xlabel('T'); axes[1].set_ylabel('hard rank r_J')
    axes[1].set_title('r_J vs T (P6: both→Nk at converged T)')
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)
    axes[2].plot(Ts, [r['valley_ratio_full'] for r in rows], 'o-', label='full')
    axes[2].plot(Ts, [r['valley_ratio_lora'] for r in rows], 's--', label='LoRA')
    axes[2].set_xlabel('T'); axes[2].set_ylabel(r'$\|J a_{nn}\|/\|J a_{far}\|$')
    axes[2].set_title('P7 valley ratio vs T (full > LoRA?)')
    axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=7)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight',
                                    facecolor='white'); plt.close()
    print(f"[G-Tsweep] figure -> {save_path}")


# ---------------------------------------------------------------------------
# 7. Parity gate + stage-0 gate (tiny: N=2, k=8, T=5, layer-0)
# ---------------------------------------------------------------------------
def _parity_gate(device='cpu', verbose=True):
    """HARD GATE (audit): arm G's training map MUST be the SAME operator as arms C/D's
    (`fullft_valley.train_full`) — the whole cross-arm comparison is VOID otherwise.
    Both are plain SGD (w←w−lr·grad, BCE, W init at frozen.clone(), biases frozen), but
    unrolled_full_theta (manual w−lr·grad + _full_forward) and train_full
    (torch.optim.SGD + forward_logits with an EMPTY adapter) are INDEPENDENT
    implementations — this asserts they agree byte-for-byte at a=0/create_graph=False.

    Tiny config: N=2, layer-0 only, T=5, GELU, eps_scale=0, batch_seed=None (no ε, no
    minibatch noise), identical θ₀/b0/x0/y/lr. Aborts on mismatch (reconcile
    _full_forward to forward_logits before submit).
    """
    from experiments.dataset_sensitivity.fullft_valley import train_full
    N, k, T = 2, 8, 5
    _ctx_lora, ctx_full, _digits, _ds = build_contexts(
        N, k, T, rank=8, activation='gelu', seed=42, device=device,
        dataset='mnist', tangent_method='qr', trainable_layers=(0,))
    x0 = ctx_full.x0_centered              # a=0 ⇒ x = x0_centered (make_images identity)
    l0 = ctx_full.trainable_layers[0]
    # arm-G unroll (value-only) vs the arms-C/D trainer (ε OFF, minibatch noise OFF).
    dtheta_unroll = unrolled_full_theta(
        ctx_full.frozen, ctx_full.b0, x0, ctx_full.y, ctx_full.lr, ctx_full.T,
        ctx_full.act, ctx_full.trainable_layers, create_graph=False)
    dtheta_train, _mb = train_full(
        ctx_full.frozen, ctx_full.b0, x0, ctx_full.y, ctx_full.lr, ctx_full.T,
        ctx_full.act, eps_seed=0, eps_scale=0.0,
        trainable_layers=ctx_full.trainable_layers, batch_seed=None)
    max_abs = (dtheta_unroll[l0] - dtheta_train[l0]).abs().max().item()
    ok = torch.allclose(dtheta_unroll[l0], dtheta_train[l0], atol=1e-9, rtol=1e-7)
    if verbose:
        print("=== ARM-G PARITY GATE (unrolled_full_theta vs fullft_valley.train_full) ===")
        print(f"  config: N={N}, layer-0, T={T}, gelu, eps_scale=0, batch_seed=None")
        print(f"  max |Δθ_unroll − Δθ_train| = {max_abs:.3e}  (atol=1e-9, rtol=1e-7)")
        print(f"  {'PASSED — SAME operator as arms C/D' if ok else 'FAILED — operators DIFFER, arm G VOID'}")
    assert ok, (f"arm-G training map DIFFERS from fullft_valley.train_full (max abs "
                f"diff {max_abs:.3e} > tol) — reconcile _full_forward to forward_logits "
                f"before submit; the cross-arm comparison is VOID otherwise.")
    return {'passed': ok, 'max_abs_diff': max_abs}


def stage0(device='cpu', verbose=True):
    """Tiny gate: parity-with-arms-C/D FIRST, then assert J_full finite + FD spot-check
    the jvp double-backward (like jacobian_spectrum.toy_ad_gate). N=2, k=8, T=5,
    layer-0, GELU, float64."""
    # Parity gate BEFORE the FD gate: arm G must train the SAME full-FT operator as
    # arms C/D (fullft_valley.train_full), else the cross-arm comparison is void.
    _parity_gate(device=device, verbose=verbose)
    N, k, T = 2, 8, 5
    ctx_lora, ctx_full, digits, ds_mean = build_contexts(
        N, k, T, rank=8, activation='gelu', seed=42, device=device,
        dataset='mnist', tangent_method='qr', trainable_layers=(0,))
    Nk = N * k
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)

    J = exact_jacobian_full(a0, ctx_full)
    finite = bool(torch.isfinite(J).all())
    # FD spot-check on 3 coords (rel err < 1e-4, the real-smoke bar — full-FT Δθ is
    # far larger dynamic range than the toy, so 1e-4 not 1e-6).
    coords = [0, Nk // 2, Nk - 1]
    fd = fd_jacobian_full(a0, ctx_full, coords, eps=1e-5)
    fd_rel = max((J[:, j] - fd[j]).norm().item() /
                 (fd[j].norm().item() + 1e-30) for j in coords)
    r_J, sv = hard_rank(J)
    er = effective_rank(J)
    mb_full, _ = full_ft_max_bce(ctx_full)

    # valley basis smoke (2N cols) — the a_nn/a_far readout must also build.
    U_val, meta = build_valley_basis(ctx_full.x0_centered, ctx_full.y, digits, seed=42)
    ctx_full.U = U_val
    a0v = torch.zeros(2 * N, dtype=torch.float64, device=device)
    Jv = exact_jacobian_full(a0v, ctx_full)
    vr, _pi, _nn, _far = valley_ratio(Jv)

    ok = finite and (fd_rel < 1e-4) and bool(torch.isfinite(Jv).all())
    if verbose:
        print("=== ARM-G STAGE-0 GATE (N=2, k=8, T=5, layer-0, gelu, float64) ===")
        print(f"  digits={digits}  J_full shape={tuple(J.shape)} (dimΘ={J.shape[0]}, Nk={Nk})")
        print(f"  [gate] J finite               : {finite}")
        print(f"  [gate] FD rel err (<1e-4)     : {fd_rel:.3e}")
        print(f"  [gate] valley J finite        : {bool(torch.isfinite(Jv).all())}")
        print(f"  [diag] r_J(J_full)            : {r_J}/{Nk}  eff_rank={er:.2f}")
        print(f"  [diag] full-FT max_bce (T=5)  : {mb_full:.2e} (underfit expected)")
        print(f"  [diag] valley ratio (full)    : {vr:.3f}")
        print(f"  {'PASSED' if ok else 'FAILED'}")
    return {'passed': ok, 'finite': finite, 'fd_rel': fd_rel, 'r_J': r_J,
            'eff_rank': er, 'Nk': Nk, 'max_bce': mb_full, 'valley_ratio': vr}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage0', action='store_true',
                   help='tiny FD gate (N=2,k=8,T=5,layer-0); assert J finite + FD spot-check')
    p.add_argument('--compare', action='store_true',
                   help='single-T J_full vs J_LoRA comparison')
    p.add_argument('--tsweep', action='store_true',
                   help='mandatory T-sweep T={1,5,20} (shape/ratio T-stability)')
    p.add_argument('--all_layers', action='store_true',
                   help='all-layer J_full stretch (heavier; default = layer-0 PRIMARY)')
    p.add_argument('--N', type=int, default=4)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--T', type=int, default=5)
    p.add_argument('--Ts', type=int, nargs='+', default=[1, 5, 20])
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--activation', type=str, default='gelu')
    p.add_argument('--dataset', type=str, default='mnist',
                   choices=['mnist', 'fashion', 'flowers'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=TRAIN_LR)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--save', action='store_true')
    p.add_argument('--tag', type=str, default=None)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # layer-0 PRIMARY per plan §2.2; all-layer is the optional stretch.
    trainable_layers = (0, 1, 2) if args.all_layers else (0,)

    if args.stage0:
        g = stage0(device=device)
        if not g['passed']:
            print("FATAL: arm-G stage-0 gate failed — do NOT submit. Inspect J_full.")
            sys.exit(1)

    if args.compare:
        print_predictions()
        run_compare(N=args.N, k=args.k, T=args.T, rank=args.rank,
                    activation=args.activation, device=device,
                    dataset=args.dataset, seed=args.seed, lr=args.lr,
                    trainable_layers=trainable_layers)

    if args.tsweep:
        run_tsweep(N=args.N, k=args.k, Ts=tuple(args.Ts), rank=args.rank,
                   activation=args.activation, device=device,
                   dataset=args.dataset, seed=args.seed, lr=args.lr,
                   trainable_layers=trainable_layers, save=args.save, tag=args.tag)

    print("=== Done ===")
