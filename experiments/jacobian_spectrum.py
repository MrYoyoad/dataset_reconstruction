"""Phase J0/J1 — the data-latent Jacobian of LoRA fine-tuning.

Implements the falsifiable program in
``notes/jacobian_leakage_experiment_plan.md``. The central object is the
**data-latent Jacobian**

    J = ∂ vec(A_T, B_T) / ∂ a          (shape [dimY, Nk])

where the private data is hidden inside realistic image variations

    x_i(a_i) = x_i^0 + U_i a_i          (U_i orthonormal tangent directions),

and ``(A_T, B_T)`` is the LoRA adapter obtained by *deterministically*
fine-tuning θ₀ on ``{x_i(a_i)}`` for T unrolled SGD steps. J0 asks: can we
recover the private coordinates ``a`` from the released adapter, and does the
spectrum of ``J`` predict *which* coordinates survive?

J1 adds training-seed randomness, estimates the seed-noise covariance
``Σ_seed``, whitens the Jacobian (``J_SNR = Σ_seed^{-1/2} J``, the square root of
the Fisher information ``F = Jᵀ Σ_seed^{-1} J``), and computes the effective
recoverable dimension ``q_eff(ε) = #{i : ε·σ_i(J_SNR) > 1}``.

Design notes (see the plan's audit section):
- **GELU only.** The Jacobian is computed by differentiating *through* the
  unrolled training gradients (a third-order autograd path). ``modified_relu``
  has no double-backward and silently corrupts J. GELU is C^∞.
- **float64 throughout** — the FD gate requires ``<1e-6`` relative error.
- **ds_mean is frozen at a=0.** ``forward_Y`` subtracts a fixed dataset mean;
  it never recomputes the mean from ``x(a)`` (that would couple the batch and
  pollute J).
- **Single LoRA module** first (``target_layers=(0,)``); ``scaling = 1``
  (alpha = rank), matching ``lora_wrapper.LoRALinear``.
- **generate_target applies LoRA to all 3 layers**; its ``θ_T`` is NOT the
  single-module target. We reuse it only for ``frozen / b0 / B0 / ds_mean`` and
  define the target as ``Y0 := forward_Y(0)``.

Usage (WEXAC):
    python -u -m experiments.jacobian_spectrum --smoke --device cuda
    python -u -m experiments.jacobian_spectrum --j0 --N 4 --k 8 --T 5 --save --device cuda
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

from experiments.configs import RESULTS_DIR, FIGURES_DIR, TRAIN_LR
from experiments.direct_inversion import generate_target
from experiments.run_experiment_b import make_activation
from experiments.data_utils import get_finetuning_data
from experiments.gate_matrix_test import effective_rank


# ---------------------------------------------------------------------------
# 1. Tangent-basis construction (§2 of the plan: controlled k)
# ---------------------------------------------------------------------------
def build_tangents(d, k, N, seed=0, method='qr', decay=0.5, device='cpu'):
    """Per-image orthonormal (or deliberately rank-deficient) tangent bases.

    Returns:
        U: [N, d, k] tensor. Column j of U[i] is a pixel-space direction; the
           private coordinate a_{ij} moves image i along it.
        col_scales: [N, k] the norm injected into each column (all-ones for
           'qr'; a geometric decay for 'svd'). This is the *known* ground truth
           the J0 spectrum-prediction test checks σ_i(J) against.

    method:
        'qr'  — orthonormal random tangents (clean rank k). The realistic case:
                every private coordinate perturbs the image equally.
        'svd' — orthonormal directions scaled by a geometric decay
                (decay**j). Injects a *known* rank deficiency so we can verify
                the measured spectrum σ_i(J) tracks it (the falsification test).
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    U = torch.empty(N, d, k, dtype=torch.float64)
    col_scales = torch.empty(N, k, dtype=torch.float64)
    for i in range(N):
        M = torch.randn(d, k, generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(M)            # [d, k], orthonormal columns
        Q = Q[:, :k]
        if method == 'qr':
            s = torch.ones(k, dtype=torch.float64)
        elif method == 'svd':
            s = torch.tensor([decay ** j for j in range(k)], dtype=torch.float64)
        else:
            raise ValueError(f"unknown tangent method {method!r}")
        U[i] = Q * s.unsqueeze(0)
        col_scales[i] = s
    return U.to(device), col_scales.to(device)


def make_images(x0_centered, U, a):
    """x_i = x0_i + U_i a_i, differentiable in a. float64.

    Args:
        x0_centered: [N, C, H, W] mean-subtracted base images (a = 0 point).
        U: [N, d, k] tangent bases (d = C*H*W).
        a: [N, k] private coordinates.
    Returns:
        x: [N, C, H, W]
    """
    N = x0_centered.shape[0]
    shape = x0_centered.shape
    x0f = x0_centered.reshape(N, -1)
    # (U_i a_i) for every i:  [N, d, k] x [N, k] -> [N, d]
    delta = torch.einsum('ndk,nk->nd', U, a)
    return (x0f + delta).reshape(shape)


# ---------------------------------------------------------------------------
# 2. Differentiable single-module LoRA fine-tuning (returns A, B)
# ---------------------------------------------------------------------------
def _a_shape(frozen, B0, l):
    """A_l has shape [rank, in_features_l]. rank from B0_l=[out,rank];
    in_features from frozen_l=[out,in]. (Not direct_inversion.A_rank_shape,
    which hardcodes the MNIST dims and would break the toy net.)"""
    rank = B0[l].shape[1]
    in_features = frozen[l].shape[1]
    return (rank, in_features)



def _draw_B0(frozen, rank, target_layers, seed, device='cpu'):
    """Seeded LoRA B init — the ordinary-training randomness source for Σ_seed.

    Matches lora_wrapper.LoRALinear._init_lora_weights (kaiming_uniform_ with
    a=√5), which for a [out, rank] weight is U(−1/√rank, 1/√rank). Drawn from a
    seeded generator so each seed gives a reproducible, distinct init (A stays 0).
    """
    g = torch.Generator().manual_seed(int(seed))
    B0 = {}
    bound = 1.0 / (rank ** 0.5)
    for l in target_layers:
        out = frozen[l].shape[0]
        B = (torch.rand(out, rank, generator=g, dtype=torch.float64) * 2 - 1) * bound
        B0[l] = B.to(device)
    return B0


def _partial_lora_forward(frozen, A, B, b0, x, scaling, act, target_layers):
    """Forward of the MLP with LoRA on ``target_layers`` only.

    Mirrors direct_inversion._lora_forward but leaves non-target layers as plain
    frozen linears. Layer 0 carries the (frozen) bias; layers 1,2 are bias-free.
    """
    h = x.view(x.shape[0], -1)
    for l in (0, 1, 2):
        if l in target_layers:
            w = frozen[l] + scaling * (B[l] @ A[l])
        else:
            w = frozen[l]
        bias = b0 if l == 0 else None
        h = F.linear(h, w, bias)
        if l < 2:
            h = act(h)
    return h


def unrolled_lora_AB(frozen, b0, B0, x, y, lr, T, scaling, act,
                     target_layers=(0,)):
    """Unroll T full-batch SGD steps on the LoRA params; RETURN the (A, B) dicts.

    A local variant of direct_inversion.unrolled_finetune_lora that (a) trains
    only ``target_layers`` and (b) returns the adapter matrices instead of the
    composed θ_T (J0 observes the adapter, not the merged weight). Starts from
    the known LoRA init A = 0, B = B0. create_graph=True keeps (A_T, B_T) twice
    differentiable w.r.t. x (hence a).
    """
    A = {l: torch.zeros(_a_shape(frozen, B0, l), dtype=x.dtype, device=x.device,
                        requires_grad=True) for l in target_layers}
    B = {l: B0[l].clone().requires_grad_(True) for l in target_layers}

    for _ in range(T):
        logits = _partial_lora_forward(frozen, A, B, b0, x, scaling, act,
                                       target_layers).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        params = [A[l] for l in target_layers] + [B[l] for l in target_layers]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        n = len(target_layers)
        gA, gB = grads[:n], grads[n:]
        A = {l: A[l] - lr * gA[i] for i, l in enumerate(target_layers)}
        B = {l: B[l] - lr * gB[i] for i, l in enumerate(target_layers)}
    return A, B


def build_ab_index(frozen, B0, target_layers):
    """Fixed A-then-B, ascending-layer flatten layout. Returns list of
    (kind, layer, shape) for provenance / unflatten."""
    index = []
    for l in target_layers:
        index.append(('A', l, _a_shape(frozen, B0, l)))
    for l in target_layers:
        index.append(('B', l, tuple(B0[l].shape)))
    return index


def flatten_AB(A, B, index):
    """Flatten the adapter to vecY following ``index`` (A block then B block)."""
    parts = []
    for kind, l, _shape in index:
        parts.append((A[l] if kind == 'A' else B[l]).reshape(-1))
    return torch.cat(parts)


# ---------------------------------------------------------------------------
# 3. The forward map a -> Y and its exact Jacobian
# ---------------------------------------------------------------------------
class Ctx:
    """Everything forward_Y needs, fixed at a=0 (ds_mean frozen here)."""
    def __init__(self, frozen, b0, B0, x0_centered, U, y, lr, T, scaling, act,
                 target_layers, index):
        self.frozen = frozen
        self.b0 = b0
        self.B0 = B0
        self.x0_centered = x0_centered
        self.U = U
        self.y = y
        self.lr = lr
        self.T = T
        self.scaling = scaling
        self.act = act
        self.target_layers = target_layers
        self.index = index


def forward_Y(a_flat, ctx):
    """The ℝ^{Nk} -> ℝ^{dimY} closure. a_flat is [N*k] (row-major over images)."""
    N, k = ctx.U.shape[0], ctx.U.shape[2]
    a = a_flat.view(N, k)
    x = make_images(ctx.x0_centered, ctx.U, a)
    A, B = unrolled_lora_AB(ctx.frozen, ctx.b0, ctx.B0, x, ctx.y,
                            ctx.lr, ctx.T, ctx.scaling, ctx.act,
                            ctx.target_layers)
    return flatten_AB(A, B, ctx.index)


def exact_jacobian(a0, ctx, method='jvp_double'):
    """J = ∂Y/∂a at a0, shape [dimY, Nk].

    'jvp_double' — forward-over-reverse JVP built from two ``autograd.grad``
    calls (pure autograd, composes with the create_graph unroll). One column per
    a-coordinate; Nk cheap backward passes over one retained graph. This is the
    production path (dimY >> Nk, so forward-mode wins).

    'reverse_loop' — one reverse pass per *output* row (dimY passes). Only
    tractable on the toy; used to cross-check 'jvp_double'.
    """
    a = a0.clone().detach().requires_grad_(True)
    Y = forward_Y(a, ctx)
    dimY, Nk = Y.numel(), a.numel()

    if method == 'jvp_double':
        u = torch.zeros(dimY, dtype=a.dtype, device=a.device, requires_grad=True)
        # JtU = Jᵀ u  (a function of u, linear); create_graph so we can d/du it.
        (JtU,) = torch.autograd.grad(Y, a, grad_outputs=u, create_graph=True)
        cols = []
        eye = torch.eye(Nk, dtype=a.dtype, device=a.device)
        for j in range(Nk):
            # d(Jᵀu)/du · e_j = J e_j = column j of J. Retain the graph for
            # every column except the last, which frees it eagerly.
            (col,) = torch.autograd.grad(JtU, u, grad_outputs=eye[j],
                                         retain_graph=(j < Nk - 1))
            cols.append(col.detach())
        return torch.stack(cols, dim=1)          # [dimY, Nk]

    elif method == 'reverse_loop':
        rows = []
        eye = torch.eye(dimY, dtype=a.dtype, device=a.device)
        for i in range(dimY):
            (row,) = torch.autograd.grad(Y, a, grad_outputs=eye[i],
                                         retain_graph=True)
            rows.append(row.detach())
        return torch.stack(rows, dim=0)          # [dimY, Nk]

    raise ValueError(f"unknown jacobian method {method!r}")


def finite_difference_jacobian(a0, ctx, coords, eps=1e-5):
    """Central-difference columns of J for the given coordinate indices."""
    cols = {}
    for j in coords:
        e = torch.zeros_like(a0); e[j] = 1.0
        # NB: no torch.no_grad() — forward_Y runs the inner SGD via
        # autograd.grad(create_graph=True), which needs grad tracking to build
        # loss.grad_fn. We only want the *value* of Y here, so detach the result.
        Yp = forward_Y(a0 + eps * e, ctx).detach()
        Ym = forward_Y(a0 - eps * e, ctx).detach()
        cols[j] = (Yp - Ym) / (2 * eps)
    return cols


# ---------------------------------------------------------------------------
# 4. Spectrum & recovery
# ---------------------------------------------------------------------------
def spectrum(J):
    """Singular values + entropy effective rank of J."""
    svals = torch.linalg.svdvals(J.double())
    return svals, effective_rank(J)


def recover_a(J, Y_target, Y0, metric_isqrt=None, rcond=1e-10):
    """(whitened) least-squares recovery of a from the observed adapter change.

    Solves min_a ‖ W(J a) − W(Y_target − Y0) ‖² with W = metric_isqrt (identity
    for J0), returning the min-norm solution. Returns â [Nk].

    Uses the SVD pseudo-inverse (not torch.linalg.lstsq): J is often
    rank-deficient (that is the whole point — some private directions are
    unrecoverable), and lstsq's default CUDA driver ('gels') assumes full rank
    and is unstable there. pinv handles the rank-deficient case correctly and
    returns the min-norm (row-space) solution.
    """
    dY = (Y_target - Y0).reshape(-1, 1)
    A = J
    if metric_isqrt is not None:
        A = metric_isqrt @ J
        dY = metric_isqrt @ dY
    sol = torch.linalg.pinv(A, rcond=rcond) @ dY
    return sol.reshape(-1)


# ---------------------------------------------------------------------------
# 5. J1 — seed noise, whitening, Fisher / q_eff  (buildable now; gated on J0)
# ---------------------------------------------------------------------------
def estimate_sigma_seed(ctx_factory, S, a0):
    """Sample vecY at a=0 over S training seeds; return the [S, dimY] matrix of
    mean-centered samples (rows). Never forms the dimY×dimY covariance.

    ctx_factory(seed) -> Ctx must vary only the training seed (B0 init / any
    stochasticity), holding the data at a0 fixed.
    """
    samples = []
    for s in range(S):
        ctx = ctx_factory(s)
        # no torch.no_grad(): forward_Y's inner SGD needs grad tracking; we
        # detach the value-only result (see finite_difference_jacobian).
        samples.append(forward_Y(a0, ctx).detach())
    Ys = torch.stack(samples, dim=0)             # [S, dimY]
    return Ys - Ys.mean(dim=0, keepdim=True)


def snr_spectrum(J, centered_samples, shrinkage=1e-3):
    """Generalized singular values σ_i(J_SNR) with J_SNR = Σ_seed^{-1/2} J.

    Works in the S-sample subspace: Σ_seed ≈ (1/(S-1)) Mᵀ M with M =
    centered_samples [S, dimY], regularized by ``shrinkage`` (diagonal loading /
    Ledoit–Wolf shrink toward isotropic). The Fisher matrix F = Jᵀ Σ^{-1} J is
    only Nk×Nk, so we form and eig-decompose that directly. Returns
    sqrt(eig(F)) = σ_i(J_SNR), descending.

    Σ^{-1} v via Woodbury: with Σ = ρ I + (1/(S-1)) MᵀM,
        Σ^{-1} = ρ^{-1}[ I − Mᵀ (ρ(S-1) I + M Mᵀ)^{-1} M ].
    Only the S×S system (M Mᵀ) is inverted; dimY×dimY is never formed.
    """
    S, dimY = centered_samples.shape
    M = centered_samples
    rho = shrinkage * (M.pow(2).sum() / (S * dimY) + 1e-30)  # scale-aware floor
    # Σ^{-1} J, column by column, via Woodbury.
    MJ = M @ J                                   # [S, Nk]
    Gram = M @ M.t()                             # [S, S]
    K = rho * (S - 1) * torch.eye(S, dtype=J.dtype, device=J.device) + Gram
    corr = M.t() @ torch.linalg.solve(K, MJ)     # [dimY, Nk]
    Sigma_inv_J = (J - corr) / rho               # [dimY, Nk]
    Fisher = J.t() @ Sigma_inv_J                 # [Nk, Nk], = Jᵀ Σ^{-1} J
    Fisher = 0.5 * (Fisher + Fisher.t())
    eig = torch.linalg.eigvalsh(Fisher).clamp_min(0)
    sigma_snr = eig.flip(0).sqrt()               # descending
    return sigma_snr, Fisher


def q_eff(sigma_snr, eps):
    """Effective recoverable dimension at perturbation scale eps."""
    return int((eps * sigma_snr > 1.0).sum().item())


def q_eff_colspace(J, centered, eps_list, tol=1e-8):
    """SOUND whitening restricted to col(J) (yoado-29's fix for the SGD phase).

    The energy-overlap metric is chance-baseline-confounded and the full Σ_seed is
    unmeasurable at feasible S. Instead estimate the noise ONLY where the signal
    lives: let Q = orthonormal basis of col(J) (dim r_J ≤ Nk); project the noise
    samples n_s = Qᵀ(Y_s − Ȳ); estimate the small Σ_J = Cov(n_s) [r_J×r_J]; and
    whiten inside col(J): F = (QᵀJ)ᵀ Σ_J^{-1} (QᵀJ). This is estimable at
    S ≳ few·r_J and needs no shrinkage floor for unsampled directions.

    Returns dict: r_J, sigma_snr, q_eff(eps), tr(Σ_J), and the isotropy check
    tr(Σ_J) / (μ·r_J) — whether the noise genuinely HAS variance in col(J)
    (≈1 isotropic-in-col(J); ≈0 the noise avoids the signal directions).
    """
    U, s, _ = torch.linalg.svd(J, full_matrices=False)
    r_J = int((s > tol * s[0]).sum())
    Q = U[:, :r_J]                                   # [dimY, r_J], spans col(J)
    Nproj = centered @ Q                             # [S, r_J]
    S = centered.shape[0]
    Sigma_J = (Nproj.t() @ Nproj) / (S - 1)          # [r_J, r_J]
    Jc = Q.t() @ J                                    # [r_J, Nk]
    eig_noise = torch.linalg.eigvalsh(Sigma_J).clamp_min(0)
    # regularize only against numerical singularity (a floor tiny vs the trace)
    ridge = 1e-12 * (Sigma_J.diagonal().mean() + 1e-30)
    Sigma_J_reg = Sigma_J + ridge * torch.eye(r_J, dtype=J.dtype, device=J.device)
    F = Jc.t() @ torch.linalg.solve(Sigma_J_reg, Jc)
    F = 0.5 * (F + F.t())
    sigma_snr = torch.linalg.eigvalsh(F).clamp_min(0).flip(0).sqrt()
    qeffs = {eps: int((eps * sigma_snr > 1).sum().item()) for eps in eps_list}
    mu = (centered.double().pow(2).sum() / ((S - 1) * centered.shape[1])).item()
    tr_SigmaJ = float(Sigma_J.diagonal().sum())
    iso_ratio = tr_SigmaJ / (mu * r_J + 1e-30)       # ≈1 isotropic; ≈0 avoids col(J)
    return {'r_J': r_J, 'sigma_snr': sigma_snr, 'q_eff': qeffs,
            'tr_Sigma_J': tr_SigmaJ, 'iso_ratio': iso_ratio,
            'noise_eig_min': float(eig_noise[0]),
            'noise_eig_max': float(eig_noise[-1])}


def noise_subspace_energy(J, centered):
    """Fraction of J's energy that lies inside the *measured* seed-noise subspace.

    The Fisher Jᵀ Σ_seed^{-1} J only needs Σ_seed restricted to J's column space,
    so the adequacy ratio is Nk-vs-S, not dimY-vs-S. Where J's columns fall in
    span(the S noise samples), q_eff rests on measured noise (trustworthy); where
    they fall in the orthogonal complement, their "noise" is only the shrinkage
    floor ρμ — a ρ-artifact, which is exactly why those directions are ρ-sensitive.

    Returns ‖P·J‖_F² / ‖J‖_F² with P = projection onto row-space(centered).
    """
    M = centered                                  # [S, dimY], rows = noise samples
    Gram = M @ M.t()                              # [S, S]
    MJ = M @ J                                     # [S, Nk]
    PJ = M.t() @ (torch.linalg.pinv(Gram) @ MJ)   # projection of J's cols
    return (PJ.norm() ** 2 / (J.norm() ** 2 + 1e-30)).item()


# ---------------------------------------------------------------------------
# 6. Contexts: toy (self-test) and real MNIST single-module
# ---------------------------------------------------------------------------
def _toy_ctx(seed=0, N=2, k=4, T=5, d_in=6, d_h=5, rank=2, lr=0.1,
             tangent_method='qr'):
    """Tiny synthetic net for the AD unit test — CPU float64, no data files.

    Architecture matches the 3-layer MLP shape (d_in - d_h - d_h - 1) so the
    single-module LoRA path exercises the exact same code as MNIST.
    """
    g = torch.Generator().manual_seed(seed)
    frozen = {
        0: torch.randn(d_h, d_in, generator=g, dtype=torch.float64) * 0.3,
        1: torch.randn(d_h, d_h, generator=g, dtype=torch.float64) * 0.3,
        2: torch.randn(1, d_h, generator=g, dtype=torch.float64) * 0.3,
    }
    b0 = torch.randn(d_h, generator=g, dtype=torch.float64) * 0.1
    B0 = {0: torch.randn(d_h, rank, generator=g, dtype=torch.float64) * 0.2}
    x0 = torch.randn(N, 1, 1, d_in, generator=g, dtype=torch.float64)
    x0_centered = x0 - x0.mean(dim=0, keepdim=True)
    y = torch.tensor([0.0, 1.0] * (N // 2) + [0.0] * (N % 2),
                     dtype=torch.float64)[:N]
    U, col_scales = build_tangents(d_in, k, N, seed=seed + 1,
                                   method=tangent_method)
    act = make_activation('gelu')
    target_layers = (0,)
    index = build_ab_index(frozen, B0, target_layers)
    ctx = Ctx(frozen, b0, B0, x0_centered, U, y, lr, T, 1.0, act,
              target_layers, index)
    return ctx, col_scales


def _mnist_ctx(N=2, k=8, T=5, rank=2, activation='gelu', lr=TRAIN_LR,
               seed=42, device='cpu', tangent_method='qr'):
    """Real single-module MNIST context via generate_target.

    generate_target trains an all-layer LoRA; we reuse only frozen / b0 / B0[0]
    / ds_mean from it, then define the single-module target as Y0 = forward_Y(0).
    """
    n_per_class = N // 2
    x_ft, y_ft, digits, _ = get_finetuning_data(n_per_class, seed=seed,
                                                device=device)
    _theta_T_all, frozen, b0, B0_all, ds_mean = generate_target(
        x_ft, y_ft, T, rank, activation, lr, device)
    target_layers = (0,)
    B0 = {l: B0_all[l] for l in target_layers}
    x0_centered = (x_ft - ds_mean) if ds_mean is not None else x_ft
    d = x0_centered.reshape(N, -1).shape[1]
    U, col_scales = build_tangents(d, k, N, seed=seed + 1,
                                   method=tangent_method, device=device)
    act = make_activation(activation)
    index = build_ab_index(frozen, B0, target_layers)
    ctx = Ctx(frozen, b0, B0, x0_centered, U, y_ft, lr, T, 1.0, act,
              target_layers, index)
    return ctx, col_scales, digits, ds_mean


# ---------------------------------------------------------------------------
# 7. Self-tests (the gate) and J0 run
# ---------------------------------------------------------------------------
def toy_ad_gate(verbose=True):
    """AD-correctness gate. Returns dict of the diagnostic errors.

    Pass conditions test only whether J and the linear model are *correct* — NOT
    whether the map is invertible (rank deficiency is a finding, not a failure,
    and recovery quality is conditioning-dependent science measured in run_j0):
    (1) central-FD check of J (rel err < 1e-6) — J is the right Jacobian;
    (2) jvp_double vs reverse_loop agreement (< 1e-8) — the two J routines agree;
    (3) linearization residual ‖(Y_t−Y0) − J·a_true‖/‖Y_t−Y0‖ < 1e-3 at a tiny
        a_true — J is the correct local linear model (no inversion, so this is
        rank/conditioning-independent).

    Reported diagnostics (NOT gated): eff_rank(J); row-space recovery
    â vs P_row(a_true); full-a_true recovery. A rank-deficient toy J
    (eff_rank < Nk) is exactly the identifiability signal the program measures —
    it must not fail the correctness gate. Must pass before any WEXAC sweep.
    """
    ctx, col_scales = _toy_ctx()
    Nk = ctx.U.shape[0] * ctx.U.shape[2]
    a0 = torch.zeros(Nk, dtype=torch.float64)

    J = exact_jacobian(a0, ctx, method='jvp_double')
    J_rev = exact_jacobian(a0, ctx, method='reverse_loop')
    rev_gap = (J - J_rev).abs().max().item()

    coords = list(range(min(4, Nk)))
    fd = finite_difference_jacobian(a0, ctx, coords, eps=1e-5)
    fd_rel = max((J[:, j] - fd[j]).norm().item() /
                 (fd[j].norm().item() + 1e-30) for j in coords)

    # tiny known a_true → linearization must dominate; then report recovery.
    torch.manual_seed(0)
    a_true = 1e-6 * torch.randn(Nk, dtype=torch.float64)
    Y0 = forward_Y(a0, ctx).detach()
    Y_t = forward_Y(a_true, ctx).detach()
    dY = Y_t - Y0
    lin_res = (dY - J @ a_true).norm().item() / (dY.norm().item() + 1e-30)

    a_hat = recover_a(J, Y_t, Y0)
    a_row = torch.linalg.pinv(J, rcond=1e-10) @ (J @ a_true)   # P_row(a_true)
    rec_row_rel = (a_hat - a_row).norm().item() / (a_row.norm().item() + 1e-30)
    rec_full_rel = (a_hat - a_true).norm().item() / (a_true.norm().item() + 1e-30)

    svals, er = spectrum(J)
    ok = (fd_rel < 1e-6 and rev_gap < 1e-8 and lin_res < 1e-3)
    if verbose:
        print("=== TOY-AD GATE ===")
        print(f"  J shape                     : {tuple(J.shape)}")
        print(f"  [gate] FD rel err  (<1e-6)  : {fd_rel:.3e}")
        print(f"  [gate] jvp vs reverse(<1e-8): {rev_gap:.3e}")
        print(f"  [gate] lin residual (<1e-3) : {lin_res:.3e}")
        print(f"  [diag] eff_rank(J)          : {er:.3f}  (Nk={Nk})  "
              f"<Nk ⟺ rank deficiency (a finding)")
        print(f"  [diag] row-space rec rel    : {rec_row_rel:.3e}")
        print(f"  [diag] full-a_true rec rel  : {rec_full_rel:.3e}")
        print(f"  {'PASSED' if ok else 'FAILED'}")
    return {'fd_rel': fd_rel, 'rev_gap': rev_gap, 'lin_res': lin_res,
            'rec_row_rel': rec_row_rel, 'rec_full_rel': rec_full_rel,
            'eff_rank': er, 'Nk': Nk, 'passed': ok}


def real_smoke(N=2, k=8, T=5, rank=2, device='cpu', verbose=True):
    """Real single-module MNIST smoke: FD on 3 coords (<1e-4), print spectrum."""
    ctx, col_scales, digits, ds_mean = _mnist_ctx(
        N=N, k=k, T=T, rank=rank, device=device)
    Nk = N * k
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)
    J = exact_jacobian(a0, ctx, method='jvp_double')
    coords = [0, Nk // 2, Nk - 1]
    fd = finite_difference_jacobian(a0, ctx, coords, eps=1e-5)
    fd_rel = max((J[:, j] - fd[j]).norm().item() /
                 (fd[j].norm().item() + 1e-30) for j in coords)
    svals, er = spectrum(J)
    if verbose:
        print("=== REAL MNIST SINGLE-MODULE SMOKE ===")
        print(f"  digits={digits}  J shape={tuple(J.shape)} (dimY={J.shape[0]}, Nk={Nk})")
        print(f"  FD rel err (max of 3 coords, <1e-4): {fd_rel:.3e}")
        print(f"  σ(J): max={svals[0]:.3e} min={svals[-1]:.3e} "
              f"cond={svals[0] / (svals[-1] + 1e-30):.3e}")
        print(f"  eff_rank(J) = {er:.3f}  (Nk={Nk})")
        print(f"  {'PASSED' if fd_rel < 1e-4 else 'FAILED'}")
    return {'fd_rel': fd_rel, 'eff_rank': er, 'svals': svals,
            'J_shape': tuple(J.shape)}


def run_j0(N=4, k=8, T=5, rank=8, activation='gelu', device='cuda',
           tangent_method='qr', eps_list=(1e-3, 1e-2, 1e-1, 1.0), seed=42,
           save=False, tag=None):
    """Phase J0: build J, its spectrum, and coordinate recovery vs eps.

    For each eps we set a_true = eps * (unit random direction per coord),
    fine-tune, observe Y, recover â by LSQ, and record per-coordinate error.
    Deterministic training (no seed noise) — the predictor is σ_i(J).
    """
    ctx, col_scales, digits, ds_mean = _mnist_ctx(
        N=N, k=k, T=T, rank=rank, activation=activation, seed=seed,
        device=device, tangent_method=tangent_method)
    Nk = N * k
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)

    print(f"[J0] building J  (N={N}, k={k}, T={T}, rank={rank}, "
          f"tangent={tangent_method})")
    J = exact_jacobian(a0, ctx, method='jvp_double')
    svals, er = spectrum(J)
    print(f"[J0] J shape={tuple(J.shape)}  eff_rank={er:.3f}  "
          f"σ_max={svals[0]:.3e}  σ_min={svals[-1]:.3e}")

    torch.manual_seed(seed)
    direction = torch.randn(Nk, dtype=torch.float64, device=device)
    direction = direction / direction.norm()

    Jpinv = torch.linalg.pinv(J.to(device), rcond=1e-10)
    eps_results = {}
    for eps in eps_list:
        a_true = eps * direction
        Y0 = forward_Y(a0, ctx).detach()
        Y_t = forward_Y(a_true, ctx).detach()
        a_hat = recover_a(J.to(device), Y_t, Y0)
        a_row = Jpinv @ (J.to(device) @ a_true)         # recoverable component
        per_coord_err = (a_hat - a_true).abs()
        # rel_err_full includes the unrecoverable null space (a floor set by
        # rank deficiency); rel_err_row isolates how the recovery of the
        # *recoverable* part degrades as ε leaves the linear regime (the teeth).
        rel_full = (a_hat - a_true).norm().item() / (a_true.norm().item() + 1e-30)
        rel_row = (a_hat - a_row).norm().item() / (a_row.norm().item() + 1e-30)
        eps_results[eps] = {
            'a_true': a_true.cpu(), 'a_hat': a_hat.cpu(),
            'a_row': a_row.cpu(), 'per_coord_err': per_coord_err.cpu(),
            'rel_err': rel_full, 'rel_err_row': rel_row,
        }
        print(f"[J0] eps={eps:<6g}  rel_err_full={rel_full:.3e}  "
              f"rel_err_row={rel_row:.3e}")

    out = {
        'J': J.cpu(), 'svals': svals.cpu(), 'eff_rank': er,
        'col_scales': col_scales.cpu(), 'eps_results': eps_results,
        'N': N, 'k': k, 'T': T, 'rank': rank, 'activation': activation,
        'tangent_method': tangent_method, 'digits': digits,
    }
    if save:
        tag = tag or f"N{N}_k{k}_T{T}_r{rank}_{activation}_{tangent_method}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"jacobian_j0_{tag}.pth")
        torch.save(out, path)
        print(f"[J0] saved -> {path}")
        _plot_j0(out, os.path.join(FIGURES_DIR, 'jacobian_spectrum',
                                   f"j0_{tag}.png"))
    return out


def run_j0_T_sweep(N=4, k=8, rank=8, activation='gelu', device='cuda',
                   tangent_method='qr', Ts=(5, 20, 50), seed=42):
    """De-confound the deterministic eff_rank readout (yoado-29's caution): if
    eff_rank(J) climbs toward Nk as T grows, the low rank at T=5 was
    *underfitting* (directions not yet moved); if it plateaus below Nk, the
    deficiency is *structural* (a rank-r module genuinely cannot express them).
    Does NOT recover coordinates — just builds J and reports its spectrum per T.
    """
    Nk = N * k
    print(f"[J0-Tsweep] N={N} k={k} rank={rank} tangent={tangent_method} Nk={Nk}")
    rows = []
    for T in Ts:
        ctx, _cs, digits, _dm = _mnist_ctx(
            N=N, k=k, T=T, rank=rank, activation=activation, seed=seed,
            device=device, tangent_method=tangent_method)
        a0 = torch.zeros(Nk, dtype=torch.float64, device=device)
        J = exact_jacobian(a0, ctx, method='jvp_double')
        svals, er = spectrum(J)
        rows.append((T, er, svals[0].item(), svals[-1].item()))
        print(f"[J0-Tsweep] T={T:<3d}  eff_rank={er:.3f}/{Nk}  "
              f"σ_max={svals[0]:.3e}  σ_min={svals[-1]:.3e}  "
              f"cond={svals[0] / (svals[-1] + 1e-30):.2e}")
    verdict = ("UNDERFITTING (eff_rank climbs with T)"
               if rows[-1][1] > rows[0][1] + 0.5
               else "STRUCTURAL (eff_rank plateaus < Nk)")
    print(f"[J0-Tsweep] verdict: {verdict}")
    return rows


def run_j1(N=2, k=8, T=5, rank=8, activation='gelu', device='cuda',
           tangent_method='qr', S_list=(16, 32, 64),
           eps_list=(0.01, 0.1, 0.3, 1.0), shrink_list=(1e-4, 1e-2, 1e-1),
           seed=42, save=False, tag=None):
    """Phase J1: seed-whiten the Jacobian and compute q_eff (the FIRST
    privacy-meaningful number).

    Σ_seed = Cov over LoRA-B0 init draws (ordinary-training randomness; full
    batch ⇒ B0 is the main stochasticity). J is taken at a reference seed; the
    generalized spectrum σ_i(J_SNR) with J_SNR = Σ_seed^{-1/2} J gives
    q_eff(ε) = #{i : ε σ_i(J_SNR) > 1}. Reports q_eff over a range of shrinkage
    ρ and ε (small σ(J_SNR) are ρ-sensitive — this must be shown, not hidden).
    """
    base_ctx, col_scales, digits, ds_mean = _mnist_ctx(
        N=N, k=k, T=T, rank=rank, activation=activation, seed=seed,
        device=device, tangent_method=tangent_method)
    target_layers = base_ctx.target_layers
    Nk = N * k
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)

    def make_ctx(b0_seed):
        B0 = _draw_B0(base_ctx.frozen, rank, target_layers, b0_seed, device)
        index = build_ab_index(base_ctx.frozen, B0, target_layers)
        return Ctx(base_ctx.frozen, base_ctx.b0, B0, base_ctx.x0_centered,
                   base_ctx.U, base_ctx.y, base_ctx.lr, T, base_ctx.scaling,
                   base_ctx.act, target_layers, index)

    ref_ctx = make_ctx(seed)
    J = exact_jacobian(a0, ref_ctx, method='jvp_double')
    svals, er = spectrum(J)
    print(f"[J1] N={N} k={k} T={T} rank={rank}  J={tuple(J.shape)}  "
          f"raw eff_rank={er:.3f}/{Nk}  σ_max={svals[0]:.3e} σ_min={svals[-1]:.3e}")

    results = {}
    for S in S_list:
        centered = estimate_sigma_seed(lambda s: make_ctx(10_000 + s), S, a0)
        # anisotropy of the seed-noise cloud (motivates whitening)
        svd = torch.linalg.svd(centered.double(), full_matrices=False)
        noise_sv = svd.S
        # mean-centering removes one dof, so noise_sv[S-1]≈0 structurally; report
        # the smallest *non-trivial* singular value (S-2) for a meaningful ratio.
        smin_idx = max(0, min(S - 2, len(noise_sv) - 1))
        print(f"[J1] S={S}  Σ_seed noise svals: max={noise_sv[0]:.3e} "
              f"min*={noise_sv[smin_idx]:.3e}  "
              f"(anisotropy {noise_sv[0] / (noise_sv[smin_idx] + 1e-30):.2e}; "
              f"*smallest non-trivial)")
        # Gaussianity eyeball (check a): moments of the noise cloud along its top
        # direction. B0 varies GLOBALLY while J is LOCAL in a, so the CRLB/Gaussian
        # framing is only approximate if this cloud is heavy-tailed/multimodal.
        proj = centered.double() @ svd.Vh[0]                  # [S]
        proj = (proj - proj.mean()) / (proj.std() + 1e-30)
        skew = (proj ** 3).mean().item()
        exkurt = (proj ** 4).mean().item() - 3.0
        gflag = ("NON-Gaussian → Σ_seed crude, q_eff approximate"
                 if abs(skew) > 1 or abs(exkurt) > 2 else "≈Gaussian")
        print(f"[J1] S={S}  noise top-dir moments: skew={skew:+.2f} "
              f"excess_kurt={exkurt:+.2f}  ({gflag})")
        # Reliability: adequacy ratio (Nk vs S, NOT dimY vs S) + fraction of J's
        # energy actually spanned by the measured noise (yoado-29). q_eff is
        # trustworthy only for the J-energy the noise subspace supports.
        adequacy = S / Nk
        jenergy = noise_subspace_energy(J, centered)
        aflag = "OK" if adequacy >= 4 else ("MARGINAL" if adequacy >= 2 else "UNDER-SAMPLED")
        chance = (min(S - 1, Nk) / centered.shape[1])           # overlap baseline
        print(f"[J1] S={S}  adequacy S/Nk={adequacy:.1f} ({aflag})  |  "
              f"J-energy in measured noise subspace = {100 * jenergy:.1f}%  "
              f"(chance baseline {100 * chance:.2f}% — only meaningful ABOVE it)")
        # Decisive dimensionality check (yoado-29): eff_rank(Σ_seed) and whether it
        # GROWS with S. Growing/flat-spectrum ⇒ high-dim noise undersampled ⇒ the
        # low overlap is a dimensionality artifact, not orthogonality. λ_i = sv_i².
        lam = (noise_sv[:smin_idx + 1].double()) ** 2
        p = lam / (lam.sum() + 1e-30)
        cov_eff_rank = float(torch.exp(-(p * p.log()).sum()))
        # Honest fallback under an ISOTROPIC init-noise model: floor = measured mean
        # variance μ = trace(Σ)/dimY (NOT the shrinkage ρμ). σ_i(J_SNR)=σ_i(J)/√μ.
        mu = (centered.double().pow(2).sum()
              / ((S - 1) * centered.shape[1])).item()
        sigma_iso = svals.to(centered.device) / (mu ** 0.5)
        qeff_iso = {eps: int((eps * sigma_iso > 1).sum().item()) for eps in eps_list}
        isostr = "  ".join(f"ε={e:g}:{qeff_iso[e]}/{Nk}" for e in eps_list)
        print(f"[J1] S={S}  eff_rank(Σ_seed)={cov_eff_rank:.1f} "
              f"(track vs S: growing ⇒ undersampled high-dim)  |  μ={mu:.3e}  "
              f"q_eff|iso: {isostr}")
        # SOUND diagnostic: whiten INSIDE col(J) — measures noise where the signal
        # lives, sidestepping the high-dim undersampling. iso_ratio≈0 ⇒ noise avoids
        # col(J) (init doesn't mask); ≈1 ⇒ isotropic-in-col(J) (masks). (yoado-29)
        cs = q_eff_colspace(J, centered, eps_list)
        csstr = "  ".join(f"ε={e:g}:{cs['q_eff'][e]}/{Nk}" for e in eps_list)
        print(f"[J1] S={S}  col(J) r_J={cs['r_J']}  tr(Σ_J)/(μ·r_J)="
              f"{cs['iso_ratio']:.3f} (≈0 noise avoids col(J); ≈1 isotropic-there)  "
              f"|  q_eff|col(J): {csstr}")
        for shrink in shrink_list:
            sigma_snr, Fisher = snr_spectrum(J, centered, shrinkage=shrink)
            qeffs = {eps: q_eff(sigma_snr, eps) for eps in eps_list}
            qstr = "  ".join(f"ε={e:g}:q_eff={qeffs[e]}/{Nk}" for e in eps_list)
            print(f"[J1] S={S} ρ={shrink:<6g}  σ_SNR max={sigma_snr[0]:.3e} "
                  f"min={sigma_snr[-1]:.3e}  |  {qstr}")
            results[(S, shrink)] = {
                'sigma_snr': sigma_snr.cpu(), 'q_eff': qeffs,
                'noise_svals': noise_sv.cpu(),
                'noise_skew': skew, 'noise_exkurt': exkurt,
                'adequacy_S_over_Nk': adequacy, 'j_noise_energy_frac': jenergy,
                'energy_chance_baseline': chance, 'cov_eff_rank': cov_eff_rank,
                'iso_mu': mu, 'q_eff_iso': qeff_iso,
                'colspace': {'r_J': cs['r_J'], 'iso_ratio': cs['iso_ratio'],
                             'q_eff': cs['q_eff'], 'tr_Sigma_J': cs['tr_Sigma_J'],
                             'sigma_snr': cs['sigma_snr'].cpu()},
            }

    # The leakage bracket (yoado-29): the deterministic raw eff_rank is the
    # KNOWN-init attacker (Σ_seed→0, upper bound); q_eff is the UNKNOWN-init
    # attacker (marginalizes over the init they don't know — conservative/robust).
    print(f"[J1] BRACKET  known-init upper bound ≈ raw eff_rank {er:.2f}/{Nk}  |  "
          f"unknown-init q_eff = the q_eff table above (conservative)")

    out = {
        'J': J.cpu(), 'svals': svals.cpu(), 'raw_eff_rank': er,
        'results': results, 'N': N, 'k': k, 'T': T, 'rank': rank,
        'activation': activation, 'tangent_method': tangent_method,
        'S_list': list(S_list), 'eps_list': list(eps_list),
        'shrink_list': list(shrink_list), 'digits': digits,
    }
    if save:
        tag = tag or f"N{N}_k{k}_T{T}_r{rank}_{activation}_{tangent_method}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"jacobian_j1_{tag}.pth")
        torch.save(out, path)
        print(f"[J1] saved -> {path}")
        _plot_j1(out, os.path.join(FIGURES_DIR, 'jacobian_spectrum',
                                   f"j1_{tag}.png"))
    return out


def _plot_j1(out, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Nk = out['N'] * out['k']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    # left: whitened spectra σ_i(J_SNR) for each (S, shrink)
    for (S, shrink), r in out['results'].items():
        ss = r['sigma_snr']
        axes[0].semilogy(range(1, len(ss) + 1), ss, 'o-', ms=3,
                         label=f"S={S}, ρ={shrink:g}")
    axes[0].set_xlabel('index i'); axes[0].set_ylabel(r'$\sigma_i(J_{SNR})$')
    axes[0].set_title(f"whitened spectrum (Nk={Nk})")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=6)
    # right: q_eff/Nk vs eps for the largest S, across shrink
    S_max = max(out['S_list'])
    eps = sorted(out['eps_list'])
    for shrink in out['shrink_list']:
        q = [out['results'][(S_max, shrink)]['q_eff'][e] / Nk for e in eps]
        axes[1].semilogx(eps, q, 's-', label=f"ρ={shrink:g}")
    axes[1].set_xlabel(r'perturbation scale $\epsilon$')
    axes[1].set_ylabel(r'$q_{eff}/q$')
    axes[1].set_title(f"recoverable fraction vs ε (S={S_max})")
    axes[1].set_ylim(-0.02, 1.02); axes[1].grid(True, alpha=0.3, which='both')
    axes[1].legend(fontsize=7)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight',
                                    facecolor='white'); plt.close()
    print(f"[J1] figure -> {save_path}")


def _plot_j0(out, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    svals = out['svals']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    axes[0].semilogy(range(1, len(svals) + 1), svals, 'o-', color='#1f77b4')
    axes[0].set_xlabel('index i'); axes[0].set_ylabel(r'$\sigma_i(J)$')
    axes[0].set_title(f"J spectrum (eff_rank={out['eff_rank']:.2f}, "
                      f"Nk={out['N'] * out['k']})")
    axes[0].grid(True, alpha=0.3)
    eps = sorted(out['eps_results'])
    rel_full = [out['eps_results'][e]['rel_err'] for e in eps]
    rel_row = [out['eps_results'][e].get('rel_err_row', float('nan'))
               for e in eps]
    axes[1].loglog(eps, rel_full, 's-', color='#d62728',
                   label='full (incl. null space)')
    axes[1].loglog(eps, rel_row, 'o--', color='#1f77b4',
                   label='row space (recoverable)')
    axes[1].set_xlabel(r'perturbation scale $\epsilon$')
    axes[1].set_ylabel('LSQ recovery rel error')
    axes[1].set_title('coordinate recovery vs ε (linear→nonlinear)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, which='both')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight',
                                    facecolor='white'); plt.close()
    print(f"[J0] figure -> {save_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='toy-AD gate + real MNIST single-module smoke')
    p.add_argument('--j0', action='store_true', help='run Phase J0')
    p.add_argument('--j1', action='store_true', help='run Phase J1 (whitening/q_eff)')
    p.add_argument('--T_sweep', action='store_true',
                   help='eff_rank(J) vs T — underfitting vs structural de-confound')
    p.add_argument('--Ts', type=int, nargs='+', default=[5, 20, 50])
    p.add_argument('--S_list', type=int, nargs='+', default=[16, 32, 64])
    p.add_argument('--shrink_list', type=float, nargs='+',
                   default=[1e-4, 1e-2, 1e-1])
    p.add_argument('--N', type=int, default=4)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--T', type=int, default=5)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--activation', type=str, default='gelu')
    p.add_argument('--tangent', type=str, default='qr', choices=['qr', 'svd'])
    p.add_argument('--eps_list', type=float, nargs='+',
                   default=[1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--save', action='store_true')
    p.add_argument('--tag', type=str, default=None)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.smoke:
        gate = toy_ad_gate()
        if not gate['passed']:
            print("FATAL: toy-AD gate failed — do NOT submit. Inspect J.")
            sys.exit(1)
        real_smoke(device=device)

    if args.j0:
        run_j0(N=args.N, k=args.k, T=args.T, rank=args.rank,
               activation=args.activation, device=device,
               tangent_method=args.tangent, eps_list=tuple(args.eps_list),
               seed=args.seed, save=args.save, tag=args.tag)

    if args.T_sweep:
        run_j0_T_sweep(N=args.N, k=args.k, rank=args.rank,
                       activation=args.activation, device=device,
                       tangent_method=args.tangent, Ts=tuple(args.Ts),
                       seed=args.seed)

    if args.j1:
        run_j1(N=args.N, k=args.k, T=args.T, rank=args.rank,
               activation=args.activation, device=device,
               tangent_method=args.tangent, S_list=tuple(args.S_list),
               eps_list=tuple(args.eps_list), shrink_list=tuple(args.shrink_list),
               seed=args.seed, save=args.save, tag=args.tag)

    print("=== Done ===")