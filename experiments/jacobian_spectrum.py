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
def _pca_basis(k, n_bank=4000, seed=0, device='cpu', dataset='mnist', tail=False):
    """Principal directions of a PUBLIC data bank (the Gen-L generator).

    A linear generator x = μ + U z has a CONSTANT Jacobian U, so its top-k
    well-conditioned directions are just the top-k principal components of the
    data — the real axes natural images vary along, not random noise. The bank
    is the TRAIN split (public proxy), disjoint from the private TEST images.

    Args:
        dataset: which bank to build the PCA from — MUST match the private data's
            dataset (Blocker-2: a prior bug hard-coded 'mnist' here).
        tail: if True return the BOTTOM-k right singular vectors (the low-variance
            / high-frequency directions where natural images carry ~no energy —
            OFF-manifold; a contrast, not a privacy-relevant basis).

    Returns:
        dirs: [d, k] orthonormal principal directions (right singular vectors).
        sv:   [k] their singular values (the real variance profile).
    """
    from experiments.data_utils import _load_dataset
    ds = _load_dataset(dataset, train=True)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_bank].tolist()
    X = torch.stack([ds[i][0] for i in idx]).reshape(len(idx), -1).double()
    Xc = X - X.mean(dim=0, keepdim=True)
    # principal axes = right singular vectors of the centered data matrix
    _U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    if tail:
        dirs = Vh[-k:].t().contiguous()       # bottom-k (off-manifold contrast)
        sv = S[-k:]
    else:
        dirs = Vh[:k].t().contiguous()        # top-k
        sv = S[:k]
    return dirs.to(device), sv.to(device)


def _private_diff_basis(x0_centered):
    """Discriminative tangents from the PRIVATE set's own inter-image differences.

    x0_centered is already {x_i − x̄} (mean over the N fine-tuning images), so its
    right singular vectors are the directions that DISTINGUISH the private images
    — the privacy-relevant, on-manifold discriminative subspace. It is inherently
    **rank ≤ N−1** (N mean-centered points): N=2 → 1 direction, N=4 → ≤3. That
    low-dimensionality IS the finding (see plan H1, Blocker-1).

    Returns:
        dirs: [d, r] orthonormal difference directions, r = rank ≤ N−1.
        sv:   [r] their singular values.
    """
    N = x0_centered.shape[0]
    X = x0_centered.reshape(N, -1).double()   # rows = (x_i − x̄)
    _U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    r = int((S > 1e-9 * S[0]).sum())          # numerical rank ≤ N−1
    return Vh[:r].t().contiguous(), S[:r]


def build_tangents(d, k, N, seed=0, method='qr', decay=0.5, device='cpu',
                   pca_dirs=None, pca_sv=None, priv_dirs=None):
    """Per-image tangent bases. Column j of U[i] moves image i along direction j.

    Returns U [N, d, k] and col_scales [N, k]. NOTE for 'difference' the effective
    k is priv_dirs' rank (≤ N−1), so read the true k from U.shape[2] downstream.

    method:
        'qr'         — orthonormal RANDOM tangents (arbitrary pixel directions).
        'svd'        — random orthonormal scaled by a geometric decay (control).
        'pca'        — top-k PRINCIPAL directions of the data (Gen-L), unit-norm.
                       On-manifold but the SHARED (least-private) modes → collinear.
        'pca_scaled' — pca scaled by the real data spectrum (sv/sv[0]).
        'pca_tail'   — BOTTOM-k principal directions (pass tail pca_dirs). OFF-manifold
                       contrast: low-variance/high-freq, images carry ~no energy there.
        'difference' — top-(N−1) PCs of the PRIVATE set's own {x_i−x̄} (priv_dirs).
                       On-manifold AND privacy-relevant (distinguishes the images);
                       inherently rank ≤ N−1 — the star H1 method.
        'residual'   — random qr directions with span(pca_dirs) projected out
                       (off-manifold-ish contrast: what's left after the shared modes).
    """
    # ---- shared-basis methods (same U for every image) ----
    if method in ('pca', 'pca_scaled', 'pca_tail', 'difference'):
        if method == 'difference':
            if priv_dirs is None:
                raise ValueError("method='difference' requires priv_dirs")
            Q = priv_dirs.double()
            k = Q.shape[1]                    # effective k = rank ≤ N−1
            s = torch.ones(k, dtype=torch.float64, device=Q.device)
        else:
            if pca_dirs is None:
                raise ValueError(f"method={method!r} requires pca_dirs")
            Q = pca_dirs.double()
            if method == 'pca_scaled':
                s = (pca_sv[:k].double() / (pca_sv[0].double() + 1e-30))
            else:                             # pca, pca_tail: unit-norm
                s = torch.ones(k, dtype=torch.float64, device=Q.device)
        Ui = Q * s.unsqueeze(0)               # [d, k], shared across images
        U = Ui.unsqueeze(0).repeat(N, 1, 1)   # [N, d, k]
        col_scales = s.unsqueeze(0).repeat(N, 1).cpu()
        return U.to(device), col_scales.to(device)

    # ---- per-image random methods (qr / svd / residual) ----
    if method == 'residual' and pca_dirs is None:
        raise ValueError("method='residual' requires pca_dirs (subspace to remove)")
    P = pca_dirs.double().to('cpu') if method == 'residual' else None
    g = torch.Generator(device='cpu').manual_seed(seed)
    U = torch.empty(N, d, k, dtype=torch.float64)
    col_scales = torch.empty(N, k, dtype=torch.float64)
    for i in range(N):
        M = torch.randn(d, k, generator=g, dtype=torch.float64)
        if method == 'residual':
            M = M - P @ (P.t() @ M)           # remove the shared-PCA subspace
        Q, _ = torch.linalg.qr(M)            # [d, k], orthonormal columns
        Q = Q[:, :k]
        if method in ('qr', 'residual'):
            s = torch.ones(k, dtype=torch.float64)
        elif method == 'svd':
            s = torch.tensor([decay ** j for j in range(k)], dtype=torch.float64)
        else:
            raise ValueError(f"unknown tangent method {method!r}")
        U[i] = Q * s.unsqueeze(0)
        col_scales[i] = s
    return U.to(device), col_scales.to(device)


def subspace_overlap(U_a, U_b):
    """Principal-angle overlap between two tangent sets — the guardrail against the
    invariance no-op (job 993396). Returns (mean_cos, energy_frac): cosines of the
    principal angles between the orthonormalized column spans, and the fraction of
    U_a's energy captured by proj onto span(U_b). LOW overlap ⇒ genuinely different
    subspace. Works in whatever space the columns live in — call it in INPUT space
    (d-dim) AND in col(J)/Y-space, since the invariance theorem lives in col(J).

    U_a: [d, k_a], U_b: [d, k_b] (any matching row dim).
    """
    def _orth(M):
        Q, _ = torch.linalg.qr(M.double())
        return Q[:, :M.shape[1]]
    Qa, Qb = _orth(U_a), _orth(U_b)
    M = Qa.t() @ Qb                            # [k_a, k_b]
    cos = torch.linalg.svdvals(M).clamp(0, 1)  # principal-angle cosines
    energy = (M.pow(2).sum() / Qa.shape[1]).item()   # ‖P_b Qa‖²/k_a
    return float(cos.mean()), energy


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

    Generalized to ANY layer count (audit: the old `for l in (0,1,2)` silently
    dropped layers 3+ on the CIFAR "monster" net → wrong θ_T/J/accuracy, no crash).
    Layer 0 carries the (frozen) bias; deeper layers are bias-free; activation
    after every layer except the last (a raw logit out).
    """
    n_layers = len(frozen)
    h = x.view(x.shape[0], -1)
    for l in range(n_layers):
        if l in target_layers:
            w = frozen[l] + scaling * (B[l] @ A[l])
        else:
            w = frozen[l]
        bias = b0 if l == 0 else None
        h = F.linear(h, w, bias)
        if l < n_layers - 1:
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


@torch.no_grad()
def finetune_metrics(frozen, b0, A, B, x, y, scaling, act, target_layers,
                     x_held=None, y_held=None):
    """R4 always-on metrics for a fine-tuned adapter (audit: measure, don't assume).

    Returns per-sample BCE on the private set (the MEMORIZATION signal — "it
    should help memorize the images"), private-set accuracy, and — if held-out
    data is given — the composed model's held-out accuracy (does it generalize or
    just memorize?). NB: value-only, so torch.no_grad is fine here (unlike the
    unrolled forward, which contains an autograd.grad).
    """
    logits = _partial_lora_forward(frozen, A, B, b0, x, scaling, act,
                                   target_layers).view(-1)
    per_sample_bce = F.binary_cross_entropy_with_logits(
        logits, y, reduction='none')                     # [N] memorization
    priv_acc = ((logits > 0).double() == y).double().mean().item()
    out = {'per_sample_bce': per_sample_bce.detach().cpu(),
           'mean_bce': per_sample_bce.mean().item(),
           'max_bce': per_sample_bce.max().item(),
           'private_acc': priv_acc}
    if x_held is not None and y_held is not None:
        hlog = _partial_lora_forward(frozen, A, B, b0, x_held, scaling, act,
                                     target_layers).view(-1)
        out['held_acc'] = ((hlog > 0).double() == y_held).double().mean().item()
    return out


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
    # Guard (rigor): modifiedrelu is a custom autograd.Function with only a
    # first backward → NO double-backward → the exact J is invalid (it would
    # error cryptically). modifiedrelu is ACCURACY-ONLY; use gelu (exact) or relu
    # (within-cell first-order). Single chokepoint for all J computation.
    from CreateModel import ModifiedRelu
    if isinstance(ctx.act, ModifiedRelu):
        raise ValueError(
            "modifiedrelu has no double-backward → no exact Jacobian (accuracy-only). "
            "Use gelu (exact J) or relu (within-cell first-order J).")
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
    # per-direction noise level relative to isotropic μ (descending). A flat ≈iso_ratio
    # profile ⇒ uniform attenuation; a split (some ≈1, some ≈0) ⇒ structured masking
    # (e.g. col(J) A-block dirs see attenuated B-block init noise). (yoado-29)
    eig_over_mu = (eig_noise.flip(0) / (mu + 1e-30))
    return {'r_J': r_J, 'sigma_snr': sigma_snr, 'q_eff': qeffs,
            'tr_Sigma_J': tr_SigmaJ, 'iso_ratio': iso_ratio,
            'eig_over_mu': eig_over_mu.cpu(),
            'noise_eig_min': float(eig_noise[0]),
            'noise_eig_max': float(eig_noise[-1])}


def _coord_transforms(J, N, k):
    """Linear reparametrizations of the Nk coordinates a → M·a', giving J' = J·M.

    Returns list of (name, J', is_restriction). RELABELINGS (full-rank M) leave
    col(J) — hence the hard rank and I(a;Y) — invariant but rescale eff_rank/q_eff
    (those are coordinate-dependent). RESTRICTIONS drop coordinates → genuinely
    smaller subspace. Coord order is [img0:k][img1:k]…  (a.view(N,k) row-major).
    """
    dev, dt = J.device, J.dtype
    Nk = N * k
    out = [('identity', J, False)]

    # response-whiten: M = V Σ^{-1} on the top-r_J subspace ⇒ J·M has orthonormal
    # columns (flat spectrum). Cosmetic: same col(J), maximal eff_rank.
    U, s, Vh = torch.linalg.svd(J, full_matrices=False)
    r_J = int((s > 1e-8 * s[0]).sum())
    Mw = (Vh[:r_J].t() / s[:r_J])                    # [Nk, r_J]
    out.append(('response_white', J @ Mw, False))    # J' = U_r (orthonormal)

    # cross-image sum/diff per PCA direction (N=2 only): orthogonal relabeling.
    if N == 2:
        M = torch.zeros(Nk, Nk, dtype=dt, device=dev)
        r2 = 1.0 / (2 ** 0.5)
        for j in range(k):
            c, d = j, k + j                          # new coords: common_j, diff_j
            M[0 * k + j, c] = r2;  M[1 * k + j, c] = r2      # common → both +
            M[0 * k + j, d] = r2;  M[1 * k + j, d] = -r2     # diff   → opposite
        Jci = J @ M
        out.append(('crossimg_sumdiff', Jci, False))         # relabel (invariant col)
        out.append(('crossimg_diffONLY', Jci[:, k:2 * k], True))  # restrict to diff
    return out


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


def _honest_target(x_ft, y_ft, T, rank, activation, lr, device, base_dataset):
    """R0b: honest θ₀ target-generator (replaces the swap). Loads the base model
    trained WITH `activation` on `base_dataset` (models/weights-<base>_<act>.pth),
    builds the net UNDER that activation (no monkey-patch), and returns
    frozen/b0/B0/ds_mean generalized to any layer count. Geometry-asserted."""
    from experiments.configs import DATASET_SPECS, MODELS_DIR
    from experiments.run_experiment_b import load_pretrained
    from experiments.ntk_steps import compute_multi_step_update_lora
    spec = DATASET_SPECS[base_dataset]
    ckpt = os.path.join(MODELS_DIR, f"weights-{base_dataset}_{activation}.pth")
    assert os.path.exists(ckpt), (
        f"honest θ₀ checkpoint missing: {ckpt} — train it first (base-retrain job). "
        f"NOT falling back to the swap (that's the bug we're fixing).")
    assert x_ft.reshape(x_ft.shape[0], -1).shape[1] == spec['input_dim'], (
        f"input_dim mismatch: x has {x_ft.reshape(x_ft.shape[0],-1).shape[1]}, "
        f"{base_dataset} spec expects {spec['input_dim']}")
    model = load_pretrained(device=device, pretrained_path=ckpt,
                            input_dim=spec['input_dim'], hidden=spec['hidden'],
                            activation_name=activation)
    # model already carries the honest activation ⇒ activation_name=None (NO swap).
    upd = compute_multi_step_update_lora(model, x_ft.clone(), y_ft.clone(),
                                         lr=lr, n_steps=T, rank=rank,
                                         activation_name=None)
    theta_0 = upd['theta_0']
    n_layers = len(spec['hidden']) + 1
    assert f'layers.{n_layers-1}.weight' in theta_0 and \
        f'layers.{n_layers}.weight' not in theta_0, \
        f"n_layers geometry mismatch for {base_dataset} (expected {n_layers})"
    frozen = {i: theta_0[f'layers.{i}.weight'] for i in range(n_layers)}
    b0 = theta_0['layers.0.bias']
    B0 = {i: upd['lora_B0'][f'layers.{i}.weight'] for i in range(n_layers)}
    return upd['theta_T'], frozen, b0, B0, upd['ds_mean']


def _mnist_ctx(N=2, k=8, T=5, rank=2, activation='gelu', lr=TRAIN_LR,
               seed=42, device='cpu', tangent_method='qr', dataset='mnist',
               anchor_alpha=0.0, b0_seed=None, base_dataset=None):
    """Real single-module context via generate_target (784-dim θ₀).

    generate_target trains an all-layer LoRA; we reuse only frozen / b0 / B0[0]
    / ds_mean from it, then define the single-module target as Y0 = forward_Y(0).

    dataset: private-image source (784-dim track: 'mnist'/'fashion'/'flowers',
        all reuse the MNIST θ₀ — the 28×28 transfer-attack cookbook).
    anchor_alpha: linearize/train the LoRA map from a shifted work point
        θ_anchor = (1−α)θ₀ + α·θ_T (θ_T = the composed fine-tuned endpoint).
    b0_seed: if given, override the LoRA B0 with a DETERMINISTIC seeded draw
        (_draw_B0). Needed when comparing col(J) ACROSS tangent methods (H1) —
        generate_target's B0 is RNG-order-dependent, so two ctx built in sequence
        would otherwise get different B0 and their col(J) would differ for reasons
        unrelated to the tangents. Fixed B0 ⇒ the col(J) difference isolates U.
    """
    n_per_class = N // 2
    x_ft, y_ft, digits, _ = get_finetuning_data(n_per_class, seed=seed,
                                                device=device, dataset=dataset)
    # R0b honest θ₀: load the base model trained WITH `activation` (no swap).
    # base_dataset != dataset ⇒ cross-dataset transfer (base on A, private from B).
    base_dataset = base_dataset or dataset
    theta_T_all, frozen, b0, B0_all, ds_mean = _honest_target(
        x_ft, y_ft, T, rank, activation, lr, device, base_dataset)
    # Anchor: move the base weights toward the fine-tuned endpoint (a different
    # area of parameter space) before defining the a↦Y map.
    if anchor_alpha != 0.0:
        frozen = {l: (1 - anchor_alpha) * frozen[l]
                  + anchor_alpha * theta_T_all[f'layers.{l}.weight']
                  for l in frozen}
    target_layers = (0,)
    if b0_seed is not None:
        B0 = _draw_B0(frozen, rank, target_layers, b0_seed, device)
    else:
        B0 = {l: B0_all[l] for l in target_layers}
    x0_centered = (x_ft - ds_mean) if ds_mean is not None else x_ft
    d = x0_centered.reshape(N, -1).shape[1]
    # Basis dispatch (H1). ALWAYS pass dataset=dataset to _pca_basis (Blocker-2:
    # a prior bug hard-coded 'mnist', making fashion/flowers pca rows compare
    # against an mnist basis). 'residual' removes the top-m shared PCA subspace.
    pca_dirs, pca_sv, priv_dirs = (None, None, None)
    if tangent_method in ('pca', 'pca_scaled'):
        pca_dirs, pca_sv = _pca_basis(k, seed=seed + 7, device=device, dataset=dataset)
    elif tangent_method == 'pca_tail':
        pca_dirs, pca_sv = _pca_basis(k, seed=seed + 7, device=device,
                                      dataset=dataset, tail=True)
    elif tangent_method == 'residual':
        m = min(64, d)                        # shared subspace to project out
        pca_dirs, pca_sv = _pca_basis(m, seed=seed + 7, device=device, dataset=dataset)
    elif tangent_method == 'difference':
        priv_dirs, _pv = _private_diff_basis(x0_centered)   # [d, ≤N−1]
        priv_dirs = priv_dirs.to(device)
    U, col_scales = build_tangents(d, k, N, seed=seed + 1,
                                   method=tangent_method, device=device,
                                   pca_dirs=pca_dirs, pca_sv=pca_sv,
                                   priv_dirs=priv_dirs)
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
           tangent_method='qr', eps_list=(1e-4, 1e-3, 1e-2, 1e-1), seed=42,
           save=False, tag=None, dataset='mnist', anchor_alpha=0.0):
    """Phase J0: build J, its spectrum, and coordinate recovery vs eps.

    For each eps we set a_true = eps * (unit random direction per coord),
    fine-tune, observe Y, recover â by LSQ, and record per-coordinate error AND
    the LOCALITY residual lin_res = ‖(Y(a)−Y0) − J·a‖/‖Y(a)−Y0‖ — recovery is
    only trustworthy where lin_res ≪ 1 (the linear regime actually holds).
    Deterministic training (no seed noise) — the predictor is σ_i(J).
    """
    ctx, col_scales, digits, ds_mean = _mnist_ctx(
        N=N, k=k, T=T, rank=rank, activation=activation, seed=seed,
        device=device, tangent_method=tangent_method, dataset=dataset,
        anchor_alpha=anchor_alpha)
    Nk = N * k
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)

    print(f"[J0] building J  (dataset={dataset}, N={N}, k={k}, T={T}, rank={rank}, "
          f"tangent={tangent_method}, anchor_alpha={anchor_alpha}, seed={seed})")
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
        dY = Y_t - Y0
        # LOCALITY: is the linear model J still valid at this ε? (regime check)
        lin_res = (dY - J.to(device) @ a_true).norm().item() / (dY.norm().item() + 1e-30)
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
            'rel_err': rel_full, 'rel_err_row': rel_row, 'lin_res': lin_res,
        }
        local = 'LOCAL' if lin_res < 0.05 else ('marginal' if lin_res < 0.2 else 'NON-LOCAL')
        print(f"[J0] eps={eps:<6g}  lin_res={lin_res:.3e} ({local})  "
              f"rel_err_full={rel_full:.3e}  rel_err_row={rel_row:.3e}")

    out = {
        'J': J.cpu(), 'svals': svals.cpu(), 'eff_rank': er,
        'col_scales': col_scales.cpu(), 'eps_results': eps_results,
        'N': N, 'k': k, 'T': T, 'rank': rank, 'activation': activation,
        'tangent_method': tangent_method, 'digits': digits,
        'dataset': dataset, 'anchor_alpha': anchor_alpha, 'seed': seed,
    }
    if save:
        tag = tag or (f"{dataset}_N{N}_k{k}_T{T}_r{rank}_{activation}_"
                      f"{tangent_method}_a{anchor_alpha}_s{seed}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"jacobian_j0_{tag}.pth")
        torch.save(out, path)
        print(f"[J0] saved -> {path}")
        _plot_j0(out, os.path.join(FIGURES_DIR, 'jacobian_spectrum',
                                   f"j0_{tag}.png"))
    return out


def run_coord_transforms(N=2, k=8, T=5, rank=8, activation='gelu', device='cuda',
                         tangent_method='pca', dataset='mnist', seed=42,
                         anchor_alpha=0.0, S=128, eps_list=(0.1, 1.0, 10.0)):
    """Test the user's idea: subtract linear parts of the PCA tangents (across
    images / response-whiten) and re-measure. Shows which metrics are INVARIANT
    (hard rank of col(J), the true leakage) vs COORDINATE-DEPENDENT (eff_rank,
    q_eff), and that only a genuine subspace RESTRICTION (diff-only) changes the
    hard rank. Builds J and Σ_seed once, then applies each transform post-hoc.
    """
    base_ctx, col_scales, digits, ds_mean = _mnist_ctx(
        N=N, k=k, T=T, rank=rank, activation=activation, seed=seed,
        device=device, tangent_method=tangent_method, dataset=dataset,
        anchor_alpha=anchor_alpha)
    target_layers = base_ctx.target_layers
    Nk = N * k
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)

    def make_ctx(b0_seed):
        B0 = _draw_B0(base_ctx.frozen, rank, target_layers, b0_seed, device)
        index = build_ab_index(base_ctx.frozen, B0, target_layers)
        return Ctx(base_ctx.frozen, base_ctx.b0, B0, base_ctx.x0_centered,
                   base_ctx.U, base_ctx.y, base_ctx.lr, T, base_ctx.scaling,
                   base_ctx.act, target_layers, index)

    J = exact_jacobian(a0, make_ctx(seed), method='jvp_double')
    centered = estimate_sigma_seed(lambda s: make_ctx(10_000 + s), S, a0)
    hard_rank0 = int((torch.linalg.svdvals(J) > 1e-8 * torch.linalg.svdvals(J)[0]).sum())
    print(f"[COORD] dataset={dataset} N={N} k={k} tangent={tangent_method} "
          f"seed={seed} anchor={anchor_alpha}  Nk={Nk}  hard_rank(col J)={hard_rank0}")
    print(f"[COORD] {'transform':20s} {'ncoord':>6s} {'hard_rank':>9s} {'eff_rank':>8s} "
          f"{'iso_ratio':>9s}  q_eff|col(J) @ε=" + ",".join(f"{e:g}" for e in eps_list))
    for name, Jp, is_restrict in _coord_transforms(J, N, k):
        sv = torch.linalg.svdvals(Jp)
        hr = int((sv > 1e-8 * sv[0]).sum())
        er = effective_rank(Jp)
        cs = q_eff_colspace(Jp, centered, eps_list)
        qs = ",".join(str(cs['q_eff'][e]) for e in eps_list)
        tag = name + (" (restrict)" if is_restrict else "")
        print(f"[COORD] {tag:20s} {Jp.shape[1]:6d} {hr:9d} {er:8.2f} "
              f"{cs['iso_ratio']:9.3f}  {qs}")
    print("[COORD] NOTE: relabelings (white, sumdiff) keep hard_rank fixed = the "
          "true leakage is invariant; eff_rank/q_eff shift (coordinate-dependent). "
          "diffONLY restricts the subspace (hard_rank ≤ k) — a real change, not a fix.")


def _ctx_reseed_b0(ctx, rank, b0_seed, device):
    """Clone a ctx with a fresh B0 draw (Σ_seed over init draws for THIS J)."""
    B0 = _draw_B0(ctx.frozen, rank, ctx.target_layers, b0_seed, device)
    index = build_ab_index(ctx.frozen, B0, ctx.target_layers)
    return Ctx(ctx.frozen, ctx.b0, B0, ctx.x0_centered, ctx.U, ctx.y, ctx.lr,
               ctx.T, ctx.scaling, ctx.act, ctx.target_layers, index)


def _colj_basis(J, tol=1e-8):
    """Orthonormal basis of col(J) (left singular vectors up to numerical rank)."""
    U, s, _ = torch.linalg.svd(J, full_matrices=False)
    r = int((s > tol * s[0]).sum())
    return U[:, :r], r


# On/off-manifold tag per H1 method (plan MAJOR-c): only 'difference' is a
# privacy-relevant on-manifold discriminative basis; pca is shared/on-manifold;
# the rest are off-manifold contrasts.
_H1_MANIFOLD = {'pca': 'on(shared)', 'pca_scaled': 'on(shared)',
                'difference': 'ON(privacy)', 'pca_tail': 'OFF(deep)',
                'residual': 'OFF(resid)', 'qr': 'off(random)'}


def run_h1(methods=('pca', 'difference', 'pca_tail', 'residual', 'qr'),
           N=2, k=8, T=5, rank=8, activation='gelu', device='cuda',
           dataset='mnist', seed=42, anchor_alpha=0.0, S=64,
           eps_list=(0.001, 0.01, 0.1), save=False, tag=None):
    """H1: do discriminative tangent bases change col(J) vs PCA-shared-modes?

    Reports the INVARIANT fraction hard_rank(col J)/Nk and iso_ratio, the
    coordinate-dependent eff_rank / q_eff|col(J), and — the guardrail against the
    invariance no-op — principal-angle overlap vs the matched-k PCA basis in BOTH
    input-space AND col(J)/Y-space (the invariance theorem lives in col(J)).
    'difference' runs at k=N−1 (rank ≤ N−1 by construction — the finding); every
    row is tagged on/off-manifold (only 'difference' is privacy-relevant).
    """
    pca_cache = {}

    def pca_ref(kk):
        if kk not in pca_cache:
            ctxp, _cs, _d, _m = _mnist_ctx(
                N=N, k=kk, T=T, rank=rank, activation=activation, seed=seed,
                device=device, tangent_method='pca', dataset=dataset,
                anchor_alpha=anchor_alpha, b0_seed=seed)
            a0p = torch.zeros(N * ctxp.U.shape[2], dtype=torch.float64, device=device)
            pca_cache[kk] = (ctxp, exact_jacobian(a0p, ctxp))
        return pca_cache[kk]

    print(f"[H1] dataset={dataset} N={N} T={T} rank={rank} seed={seed} "
          f"anchor={anchor_alpha}  (difference runs at k=N−1={N-1})")
    print(f"[H1] {'method':11s} {'manifold':11s} {'k':>2s} {'Nk':>3s} {'hard':>4s} "
          f"{'frac':>5s} {'effrk':>6s} {'iso':>5s} {'in_ovlp':>7s} {'colJ_ovlp':>9s}  "
          f"q_eff|col(J) @ε=" + ",".join(f"{e:g}" for e in eps_list))
    results = {}
    for method in methods:
        km = (N - 1) if method == 'difference' else k
        ctx, cs, digits, dsm = _mnist_ctx(
            N=N, k=km, T=T, rank=rank, activation=activation, seed=seed,
            device=device, tangent_method=method, dataset=dataset,
            anchor_alpha=anchor_alpha, b0_seed=seed)
        k_eff = ctx.U.shape[2]
        Nk = N * k_eff
        a0 = torch.zeros(Nk, dtype=torch.float64, device=device)
        J = exact_jacobian(a0, ctx)
        sv = torch.linalg.svdvals(J)
        hard = int((sv > 1e-8 * sv[0]).sum())
        frac = hard / Nk
        er = effective_rank(J)
        centered = estimate_sigma_seed(
            lambda s: _ctx_reseed_b0(ctx, rank, 10_000 + s, device), S, a0)
        csr = q_eff_colspace(J, centered, eps_list)
        # overlap vs matched-k PCA (input-space uses image-0's tangents; col(J)
        # uses matched # of columns) — MAJOR-b guardrail.
        ctxp, Jp = pca_ref(k_eff)
        in_cos, _ie = subspace_overlap(ctx.U[0], ctxp.U[0])
        Qm, rm = _colj_basis(J)
        Qp, rp = _colj_basis(Jp)
        rr = min(rm, rp)
        cj_cos, _ce = subspace_overlap(Qm[:, :rr], Qp[:, :rr])
        qs = ",".join(str(csr['q_eff'][e]) for e in eps_list)
        print(f"[H1] {method:11s} {_H1_MANIFOLD.get(method,'?'):11s} {k_eff:2d} {Nk:3d} "
              f"{hard:4d} {frac:5.2f} {er:6.2f} {csr['iso_ratio']:5.2f} "
              f"{in_cos:7.3f} {cj_cos:9.3f}  {qs}")
        results[method] = {
            'k': k_eff, 'Nk': Nk, 'hard_rank': hard, 'frac': frac,
            'eff_rank': er, 'iso_ratio': csr['iso_ratio'],
            'input_overlap_vs_pca': in_cos, 'colJ_overlap_vs_pca': cj_cos,
            'q_eff': csr['q_eff'], 'manifold': _H1_MANIFOLD.get(method, '?'),
            'svals': sv.cpu(), 'U0': ctx.U[0].cpu(),
            'x0_centered': ctx.x0_centered.cpu(), 'ds_mean': dsm.cpu() if dsm is not None else None,
        }
    print("[H1] READ: 'difference' is the only on-manifold privacy-relevant basis "
          "(rank≤N−1 by construction). A method 'beats' pca only if colJ_ovlp is "
          "LOW (genuinely different col(J)) — low in_ovlp alone is the invariance no-op.")
    out = {'results': results, 'N': N, 'k': k, 'T': T, 'rank': rank,
           'dataset': dataset, 'seed': seed, 'anchor_alpha': anchor_alpha,
           'methods': list(methods), 'digits': digits}
    if save:
        tag = tag or f"{dataset}_N{N}_k{k}_T{T}_r{rank}_a{anchor_alpha}_s{seed}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"jacobian_h1_{tag}.pth")
        torch.save(out, path)
        print(f"[H1] saved -> {path}")
        _plot_h1(out, os.path.join(FIGURES_DIR, 'jacobian_spectrum', f"h1_{tag}.png"))
    return out


def _plot_h1(out, save_path):
    """Visual grid: each method's tangent-direction images + a perturbed x(a)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = out['methods']
    ncol = 4
    fig, axes = plt.subplots(len(methods), ncol, figsize=(2.2 * ncol, 2.2 * len(methods)),
                             dpi=120, squeeze=False)
    for r, m in enumerate(methods):
        res = out['results'][m]
        U0 = res['U0']                         # [d, k_m]
        d = U0.shape[0]
        side = int(round(d ** 0.5))            # 784-dim track ⇒ 28×28
        for c in range(ncol):
            ax = axes[r][c]; ax.axis('off')
            if c < U0.shape[1] and side * side == d:
                ax.imshow(U0[:, c].reshape(side, side), cmap='RdBu')
                ax.set_title(f"{m} dir{c}" if c == 0 else f"dir{c}", fontsize=6)
    fig.suptitle(f"H1 tangent directions — {out['dataset']} N={out['N']} seed={out['seed']}",
                 fontsize=9)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[H1] figure -> {save_path}")


def recover_a_nonlinear(ctx, Y_target, a_init=None, n_restarts=8, outer_iters=400,
                        adam_lr=0.05, grad_clip=1.0, device='cuda', seed=0):
    """Nonlinear inverse: optimize â over the full nonlinear forward_Y to match
    Y_target (Adam on â). Mirrors direct_inversion.run_direct_inversion but on a
    COORDINATE vector — NO pixel-box penalty (box_weight=0; â is not image-shaped).
    Second-order (forward_Y's inner SGD is create_graph=True; one backward of the
    L2 loss uses ≤2nd derivatives — same profile as direct inversion). grad-clip +
    many restarts (unrolled landscapes are chaotic at large ε). Returns best â, loss.
    """
    Nk = (a_init.numel() if a_init is not None
          else ctx.U.shape[0] * ctx.U.shape[2])
    best_loss, best_a = float('inf'), None
    for r in range(n_restarts):
        torch.manual_seed(seed * 100 + r)
        if r == 0 and a_init is not None:
            a = a_init.clone().detach().to(device).double().requires_grad_(True)
        else:
            a = (0.01 * torch.randn(Nk, dtype=torch.float64, device=device)
                 ).requires_grad_(True)
        opt = torch.optim.Adam([a], lr=adam_lr)
        last = float('inf')
        for _it in range(outer_iters):
            opt.zero_grad()
            Y = forward_Y(a, ctx)
            loss = (Y - Y_target).pow(2).sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([a], grad_clip)
            opt.step()
            last = loss.item()
        if last < best_loss:
            best_loss, best_a = last, a.detach().clone()
    return best_a, best_loss


def run_h2(N=2, k=8, T=5, rank=8, tangent='pca', activation='gelu', device='cuda',
           dataset='mnist', seed=42, anchor_alpha=0.0,
           eps_list=(0.01, 0.1, 0.3, 1.0, 3.0), n_restarts=8, outer_iters=400,
           save=False, tag=None):
    """H2: does a nonlinear inverse recover a coordinate in the LINEAR NULL SPACE
    of J (past the first-order q_eff ceiling)? Targets a_true = ε·v_min (the
    smallest-σ right singular vector of J — the collinear/near-null direction) and
    reports the null-component recovery, with the three-way verdict (yoado-15 #3):
      loss(â)≫0 → optimizer failure;  loss(â)≈0 & null-match → nonlinear WIN;
      loss(â)≈0 & null-mismatch → genuine collision (Y-match, wrong a).
    Local (warm-start from linear) vs global (random init). ε capped below NaN.
    Framed as the known-init UPPER bound (distinct from the SGD-noise phase).
    """
    ctx, cs, digits, dsm = _mnist_ctx(
        N=N, k=k, T=T, rank=rank, activation=activation, seed=seed,
        device=device, tangent_method=tangent, dataset=dataset,
        anchor_alpha=anchor_alpha)
    Nk = N * ctx.U.shape[2]
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)
    J = exact_jacobian(a0, ctx)
    sv = torch.linalg.svdvals(J)
    _U, s, Vh = torch.linalg.svd(J, full_matrices=False)
    v_min = Vh[-1].to(device)                  # a-space dir J flattens most (near-null)
    hard = int((s > 1e-8 * s[0]).sum())
    print(f"[H2] dataset={dataset} tangent={tangent} N={N} k={k} T={T}  J={tuple(J.shape)} "
          f"hard_rank={hard}/{Nk}  σ_min/σ_max={sv[-1]/sv[0]:.2e} (near-null dir = v_min)")
    Y0 = forward_Y(a0, ctx).detach()
    print(f"[H2] {'eps':>6s} {'lin_null_err':>12s} {'nl_null_err':>12s} "
          f"{'nl_relloss':>10s} {'verdict':>22s} {'loc<glob':>9s}")
    results = {}
    for eps in eps_list:
        a_true = eps * v_min
        Y_t = forward_Y(a_true, ctx).detach()
        if not torch.isfinite(Y_t).all():
            print(f"[H2] eps={eps:<6g}  Y non-finite — ε too large, capping the grid here")
            break
        dY_scale = (Y_t - Y0).norm().item() + 1e-30
        # linear baseline (should MISS the null coordinate)
        a_lin = recover_a(J, Y_t, Y0)
        lin_null = ((a_lin - a_true) @ v_min).abs().item() / (eps + 1e-30)
        # nonlinear: local (warm-start) vs global (random)
        a_loc, l_loc = recover_a_nonlinear(ctx, Y_t, a_init=a_lin, device=device,
                                           n_restarts=1, outer_iters=outer_iters, seed=seed)
        a_glob, l_glob = recover_a_nonlinear(ctx, Y_t, a_init=None, device=device,
                                             n_restarts=n_restarts, outer_iters=outer_iters,
                                             seed=seed)
        a_nl, l_nl = (a_loc, l_loc) if l_loc <= l_glob else (a_glob, l_glob)
        nl_null = ((a_nl - a_true) @ v_min).abs().item() / (eps + 1e-30)
        rel_loss = (l_nl ** 0.5) / dY_scale
        # Verdict (yoado-15): a WIN requires nonlinear to beat a GENUINELY-FAILED
        # linear (relative + gated), not just the optimizer converging. Absolute
        # nl<thr fires spuriously whenever Adam converges. Since here linear is
        # never blind (no hard null), the corrected criterion ~never fires — which
        # is the CORRECT answer for this ill-conditioned (not rank-deficient) regime.
        if rel_loss > 0.1:
            verdict = 'optimizer-failure'      # Y not matched — inconclusive
        elif lin_null < 0.3:
            verdict = 'both-recover(no-null)'  # linear already got it; no ceiling
        elif nl_null < 0.5 * lin_null:
            verdict = 'NONLINEAR-WIN'          # nl beats a failed linear
        else:
            verdict = 'collision(Y=,a≠)'       # Y-match, both miss ⇒ non-identifiable
        loc_better = 'yes' if l_loc <= l_glob else 'no'
        print(f"[H2] {eps:6g} {lin_null:12.3e} {nl_null:12.3e} {rel_loss:10.3e} "
              f"{verdict:>22s} {loc_better:>9s}")
        results[eps] = {
            'a_true': a_true.cpu(), 'a_nl': a_nl.cpu(), 'a_lin': a_lin.cpu(),
            'lin_null_err': lin_null, 'nl_null_err': nl_null, 'rel_loss': rel_loss,
            'l_local': l_loc, 'l_global': l_glob, 'verdict': verdict,
        }
    print("[H2] READ: lin_null_err≈1 (linear blind to the near-null coord); "
          "NONLINEAR-WIN ⇒ the nonlinearity recovers past the first-order q_eff ceiling; "
          "collision ⇒ genuinely non-identifiable; optimizer-failure ⇒ inconclusive.")
    out = {'results': results, 'N': N, 'k': k, 'T': T, 'rank': rank,
           'tangent': tangent, 'dataset': dataset, 'seed': seed,
           'v_min': v_min.cpu(), 'svals': sv.cpu(), 'hard_rank': hard,
           'x0_centered': ctx.x0_centered.cpu(),
           'ds_mean': dsm.cpu() if dsm is not None else None,
           'U0': ctx.U[0].cpu(), 'digits': digits}
    if save:
        tag = tag or f"{dataset}_{tangent}_N{N}_k{k}_T{T}_r{rank}_s{seed}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"jacobian_h2_{tag}.pth")
        torch.save(out, path)
        print(f"[H2] saved -> {path}")
    return out


def run_rigor(N=4, k=8, rank=8, activation='gelu', device='cuda',
              tangent_method='qr', dataset='mnist', Ts=(5, 20, 50, 100, 200),
              seed=42, memorize_thresh=1e-3, lr=TRAIN_LR, save=False, tag=None):
    """R3+R4: leakage AND memorization/accuracy across T (underfit→converged→
    overtrained), on the HONEST θ₀. At each T: eff_rank/hard_rank(J) (leakage
    geometry) + the fine-tune's per-sample BCE on the ACTUAL private images (the
    memorization signal — "it should memorize the images") + private-set accuracy.
    A row is MEMORIZED when max per-sample BCE < memorize_thresh. Answers "does the
    fine-tune actually work?" and "how does leakage move as it converges/overtrains?"
    """
    Nk = N * k
    print(f"[RIGOR] dataset={dataset} act={activation} N={N} k={k} rank={rank} "
          f"tangent={tangent_method} seed={seed} lr={lr}  (leakage + memorization vs T)")
    print(f"[RIGOR] {'T':>4s} {'eff_rank':>8s} {'hard':>4s} {'mean_bce':>9s} "
          f"{'max_bce':>9s} {'priv_acc':>8s}  memorized")
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)
    rows = []
    T_converge = None
    for T in Ts:
        ctx, cs, digits, dsm = _mnist_ctx(
            N=N, k=k, T=T, rank=rank, activation=activation, seed=seed,
            device=device, tangent_method=tangent_method, dataset=dataset, lr=lr)
        J = exact_jacobian(a0, ctx, method='jvp_double')
        svals, er = spectrum(J)
        hard = int((svals > 1e-8 * svals[0]).sum())
        # the ACTUAL fine-tuned adapter on the private images (a=0 ⇒ x = x0)
        x_priv = make_images(ctx.x0_centered, ctx.U,
                             torch.zeros(N, ctx.U.shape[2], dtype=torch.float64, device=device))
        A, B = unrolled_lora_AB(ctx.frozen, ctx.b0, ctx.B0, x_priv, ctx.y,
                                ctx.lr, T, ctx.scaling, ctx.act, ctx.target_layers)
        A = {l: A[l].detach() for l in A}
        B = {l: B[l].detach() for l in B}
        m = finetune_metrics(ctx.frozen, ctx.b0, A, B, ctx.x0_centered, ctx.y,
                             ctx.scaling, ctx.act, ctx.target_layers)
        memorized = m['max_bce'] < memorize_thresh
        if memorized and T_converge is None:
            T_converge = T
        print(f"[RIGOR] {T:4d} {er:8.2f} {hard:4d} {m['mean_bce']:9.2e} "
              f"{m['max_bce']:9.2e} {m['private_acc']:8.2f}  {'YES' if memorized else 'no'}")
        rows.append({'T': T, 'eff_rank': er, 'hard_rank': hard,
                     'mean_bce': m['mean_bce'], 'max_bce': m['max_bce'],
                     'private_acc': m['private_acc'],
                     'per_sample_bce': m['per_sample_bce'], 'svals': svals.cpu()})
    verdict = (f"converges (memorized) at T={T_converge}" if T_converge
               else "NOT memorized in the T grid (underfit — extend Ts / raise lr)")
    print(f"[RIGOR] verdict: {verdict}; "
          f"eff_rank {rows[0]['eff_rank']:.1f}→{rows[-1]['eff_rank']:.1f} over T "
          f"({'rises (underfit)' if rows[-1]['eff_rank'] > rows[0]['eff_rank']+0.5 else 'plateau'})")
    out = {'rows': rows, 'N': N, 'k': k, 'rank': rank, 'activation': activation,
           'dataset': dataset, 'tangent_method': tangent_method, 'seed': seed,
           'Ts': list(Ts), 'T_converge': T_converge, 'digits': digits}
    if save:
        tag = tag or f"{dataset}_{activation}_N{N}_k{k}_r{rank}_{tangent_method}_s{seed}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"jacobian_rigor_{tag}.pth")
        torch.save(out, path)
        print(f"[RIGOR] saved -> {path}")
    return out


def run_schemes(N=4, k=8, T=50, rank=8, activation='gelu', device='cuda',
                tangent_method='qr', dataset='mnist', seed=42, save=False, tag=None):
    """R2: how does the perturbation ASSIGNMENT across images change leakage?
    Each scheme is a coordinate map P applied to the DIFFERENT Jacobian (J_s = J·P):
      DIFFERENT : P = I_Nk         — per-image independent secrets (Nk coords).
      SAME      : P = 1_N ⊗ I_k    — all N images get the IDENTICAL k-dim secret
                  (a genuine restriction to k reinforced-across-images coords).
      MIXTURE   : P = M ⊗ I_k with a RANK-DEFICIENT blend M (audit: a full-rank M
                  is just a relabel ≡ DIFFERENT — the invariance no-op; only a
                  rank-deficient blend genuinely changes the recoverable set).
    Reports eff_rank/hard_rank(J_s) + deterministic recovery per scheme (honest θ₀).
    """
    ctx, cs, digits, dsm = _mnist_ctx(
        N=N, k=k, T=T, rank=rank, activation=activation, seed=seed,
        device=device, tangent_method=tangent_method, dataset=dataset)
    Nk = N * k
    a0 = torch.zeros(Nk, dtype=torch.float64, device=device)
    J = exact_jacobian(a0, ctx, method='jvp_double')        # DIFFERENT
    Ik = torch.eye(k, dtype=torch.float64, device=device)
    # rank-deficient blend for MIXTURE: average the N images in pairs (rank ⌈N/2⌉)
    g = torch.Generator(device='cpu').manual_seed(seed)
    Mblend = torch.zeros(N, N, dtype=torch.float64)
    r_mix = max(1, N // 2)
    Q = torch.linalg.qr(torch.randn(N, r_mix, generator=g, dtype=torch.float64))[0]
    Mblend = (Q @ Q.t()).to(device)                          # rank r_mix projector
    schemes = {
        'different': torch.eye(Nk, dtype=torch.float64, device=device),
        'same': torch.kron(torch.ones(N, 1, dtype=torch.float64, device=device), Ik),
        'mixture': torch.kron(Mblend, Ik),
    }
    print(f"[SCHEMES] dataset={dataset} act={activation} N={N} k={k} T={T} rank={rank} "
          f"Nk={Nk}  (leakage vs assignment; MIXTURE rank={r_mix})")
    print(f"[SCHEMES] {'scheme':10s} {'ncoord':>6s} {'hard_rank':>9s} {'eff_rank':>8s} "
          f"{'rec_rel@e=.1':>12s}")
    torch.manual_seed(seed)
    results = {}
    for name, P in schemes.items():
        Js = J @ P
        sv = torch.linalg.svdvals(Js)
        hard = int((sv > 1e-8 * sv[0]).sum())
        er = effective_rank(Js)
        # deterministic recovery of a small random secret in the scheme coords
        m = P.shape[1]
        direction = torch.randn(m, dtype=torch.float64, device=device)
        direction = direction / direction.norm()
        a_s = 0.1 * direction
        Y0 = forward_Y(a0, ctx).detach()
        Y_t = forward_Y((P @ a_s), ctx).detach()
        a_hat = torch.linalg.pinv(Js, rcond=1e-10) @ (Y_t - Y0)
        rec = (a_hat - a_s).norm().item() / (a_s.norm().item() + 1e-30)
        print(f"[SCHEMES] {name:10s} {m:6d} {hard:9d} {er:8.2f} {rec:12.3e}")
        results[name] = {'ncoord': m, 'hard_rank': hard, 'eff_rank': er,
                         'rec_rel': rec, 'svals': sv.cpu()}
    print("[SCHEMES] READ: SAME restricts to k reinforced coords; MIXTURE (rank-def) "
          "restricts to ~r_mix·k; DIFFERENT is the full Nk. hard_rank = recoverable dirs.")
    out = {'results': results, 'N': N, 'k': k, 'T': T, 'rank': rank,
           'activation': activation, 'dataset': dataset, 'seed': seed,
           'r_mix': r_mix, 'digits': digits}
    if save:
        tag = tag or f"{dataset}_{activation}_N{N}_k{k}_T{T}_r{rank}_s{seed}"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"jacobian_schemes_{tag}.pth")
        torch.save(out, path)
        print(f"[SCHEMES] saved -> {path}")
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
           seed=42, save=False, tag=None, dataset='mnist', anchor_alpha=0.0):
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
        device=device, tangent_method=tangent_method, dataset=dataset,
        anchor_alpha=anchor_alpha)
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
    print(f"[J1] dataset={dataset} N={N} k={k} T={T} rank={rank} "
          f"tangent={tangent_method} anchor={anchor_alpha} seed={seed}  "
          f"J={tuple(J.shape)}  raw eff_rank={er:.3f}/{Nk}  "
          f"σ_max={svals[0]:.3e} σ_min={svals[-1]:.3e}")

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
        # per-direction Σ_J spectrum relative to μ — flat ⇒ uniform attenuation;
        # split (some ≈1, some ≈0) ⇒ structured masking within col(J). (yoado-29)
        eom = cs['eig_over_mu']
        n_above = int((eom > 0.5).sum().item())
        print(f"[J1] S={S}  Σ_J/μ spectrum: max={eom[0]:.3f} med={eom[len(eom)//2]:.3f} "
              f"min={eom[-1]:.3f}  #(>0.5·μ)={n_above}/{cs['r_J']} "
              f"(how many col(J) dirs carry near-isotropic noise)")
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
        'dataset': dataset, 'anchor_alpha': anchor_alpha, 'seed': seed,
    }
    if save:
        tag = tag or (f"{dataset}_N{N}_k{k}_T{T}_r{rank}_{activation}_"
                      f"{tangent_method}_a{anchor_alpha}_s{seed}")
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
    p.add_argument('--coord_transforms', action='store_true',
                   help='subtract linear parts (across-image diff / response-whiten) & re-measure')
    p.add_argument('--rigor', action='store_true',
                   help='R3+R4: leakage + memorization/accuracy vs T (honest θ₀)')
    p.add_argument('--schemes', action='store_true',
                   help='R2: leakage vs perturbation assignment (DIFFERENT/SAME/MIXTURE)')
    p.add_argument('--h1', action='store_true',
                   help='H1: discriminative tangents (difference/pca_tail/residual vs pca)')
    p.add_argument('--h2', action='store_true',
                   help='H2: nonlinear recovery of the near-null coordinate vs ε')
    p.add_argument('--h1_methods', type=str, nargs='+',
                   default=['pca', 'difference', 'pca_tail', 'residual', 'qr'])
    p.add_argument('--n_restarts', type=int, default=8)
    p.add_argument('--outer_iters', type=int, default=400)
    p.add_argument('--Ts', type=int, nargs='+', default=[5, 20, 50])
    p.add_argument('--S_list', type=int, nargs='+', default=[16, 32, 64])
    p.add_argument('--shrink_list', type=float, nargs='+',
                   default=[1e-4, 1e-2, 1e-1])
    p.add_argument('--N', type=int, default=4)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--T', type=int, default=5)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--activation', type=str, default='gelu')
    p.add_argument('--tangent', type=str, default='qr',
                   choices=['qr', 'svd', 'pca', 'pca_scaled', 'pca_tail',
                            'difference', 'residual'])
    p.add_argument('--eps_list', type=float, nargs='+',
                   default=[1e-3, 1e-2, 1e-1, 1.0])
    p.add_argument('--dataset', type=str, default='mnist',
                   choices=['mnist', 'fashion', 'flowers'],
                   help='784-dim track; all reuse the MNIST θ₀')
    p.add_argument('--anchor_alpha', type=float, default=0.0,
                   help='linearize from θ_anchor=(1−α)θ₀+αθ_T (work point)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=TRAIN_LR,
                   help='fine-tune learning rate (rigor: lr-sanity + reach memorization)')
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
               seed=args.seed, save=args.save, tag=args.tag,
               dataset=args.dataset, anchor_alpha=args.anchor_alpha)

    if args.T_sweep:
        run_j0_T_sweep(N=args.N, k=args.k, rank=args.rank,
                       activation=args.activation, device=device,
                       tangent_method=args.tangent, Ts=tuple(args.Ts),
                       seed=args.seed)

    if args.rigor:
        run_rigor(N=args.N, k=args.k, rank=args.rank, activation=args.activation,
                  device=device, tangent_method=args.tangent, dataset=args.dataset,
                  Ts=tuple(args.Ts), seed=args.seed, lr=args.lr,
                  save=args.save, tag=args.tag)

    if args.schemes:
        run_schemes(N=args.N, k=args.k, T=args.T, rank=args.rank,
                    activation=args.activation, device=device,
                    tangent_method=args.tangent, dataset=args.dataset,
                    seed=args.seed, save=args.save, tag=args.tag)

    if args.j1:
        run_j1(N=args.N, k=args.k, T=args.T, rank=args.rank,
               activation=args.activation, device=device,
               tangent_method=args.tangent, S_list=tuple(args.S_list),
               eps_list=tuple(args.eps_list), shrink_list=tuple(args.shrink_list),
               seed=args.seed, save=args.save, tag=args.tag,
               dataset=args.dataset, anchor_alpha=args.anchor_alpha)

    if args.coord_transforms:
        run_coord_transforms(N=args.N, k=args.k, T=args.T, rank=args.rank,
                             activation=args.activation, device=device,
                             tangent_method=args.tangent, dataset=args.dataset,
                             seed=args.seed, anchor_alpha=args.anchor_alpha,
                             eps_list=tuple(args.eps_list))

    if args.h1:
        run_h1(methods=tuple(args.h1_methods), N=args.N, k=args.k, T=args.T,
               rank=args.rank, activation=args.activation, device=device,
               dataset=args.dataset, seed=args.seed, anchor_alpha=args.anchor_alpha,
               eps_list=tuple(args.eps_list), save=args.save, tag=args.tag)

    if args.h2:
        run_h2(N=args.N, k=args.k, T=args.T, rank=args.rank, tangent=args.tangent,
               activation=args.activation, device=device, dataset=args.dataset,
               seed=args.seed, anchor_alpha=args.anchor_alpha,
               eps_list=tuple(args.eps_list), n_restarts=args.n_restarts,
               outer_iters=args.outer_iters, save=args.save, tag=args.tag)

    print("=== Done ===")