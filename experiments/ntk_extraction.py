"""NTK reconstruction: recover private training data from weight changes.

In the NTK regime, the weight update ΔW = -η Σ_i c_i ∇_θ f(θ₀; x_i)
where c_i = (σ(f(θ₀; x_i)) - y_i) / N.

Two modes:
- Oracle mode: coefficients c_i are passed as FIXED constants computed
  from the true private data x. This is an upper bound on attack quality.
- Free-coefficient mode: c_i are optimized jointly with x, mirroring
  Haim et al.'s treatment of Lagrange multipliers λ_i. This is the
  realistic attack scenario.

Key differences from get_kkt_loss in extraction.py:
- Target is ΔW (weight change), not W (full weights)
- Features evaluated at θ₀ (model is frozen at init)
"""

import torch
import torch.nn as nn
from experiments.configs import (
    EXTRACTION_LR, EXTRACTION_INIT_SCALE,
    EXTRACTION_EPOCHS, EXTRACTION_EVAL_EVERY,
    COEFF_LR, COEFF_BOX_WEIGHT, COEFF_CONSISTENCY_WEIGHT,
    COEFF_INIT, COEFF_SIGN_WEIGHT, COEFF_MIN_MAGNITUDE,
)


def _project_to_lora_subspace(matrix, B0):
    """Project a matrix onto the column space of B0.

    Computes P @ matrix where P = B0 @ (B0^T @ B0)^{-1} @ B0^T
    is the orthogonal projection onto col(B0).

    Args:
        matrix: [d_out, d_in] matrix to project
        B0: [d_out, r] LoRA B matrix at initialization

    Returns:
        projected: [d_out, d_in] matrix projected onto col(B0)
    """
    # Solve (B0^T B0) C = B0^T matrix for C, then return B0 @ C
    BtB = B0.T @ B0  # [r, r]
    Bt_M = B0.T @ matrix  # [r, d_in]
    C = torch.linalg.solve(BtB, Bt_M)  # [r, d_in]
    return B0 @ C  # [d_out, d_in]


def compute_lstsq_coefficients(model_at_theta0, delta_w, x, lr, n_steps, lora_B0=None):
    """Analytic least-squares coefficients (ANA-GIA / R-GAP family).

    ΔW = -lr*T * Σ_i c_i ∇_θ f(θ₀; x_i). For a FIXED x the per-sample gradients ∇f(x_i) are constant,
    so this is a *linear* least-squares in c: c* = argmin_c ‖vec(ΔW) - G c‖², G[:,i] = -lr*T·vec(∇f(x_i)).
    Solving c in closed form removes the SGD-on-c failure modes entirely — no sign-flip, no coefficient
    collapse, no local minima. Alternated with the x-optimization this is analytic coefficient recovery.
    Returns c detached (used as fixed coefficients until the next refresh).
    """
    N = x.shape[0]
    params = list(model_at_theta0.parameters())
    names = [n for n, _ in model_at_theta0.named_parameters()]
    cols = []
    for i in range(N):
        out_i = model_at_theta0(x[i:i+1]).sum()               # scalar f(θ₀; x_i)
        grads = torch.autograd.grad(out_i, params, retain_graph=True, create_graph=False,
                                    allow_unused=True)
        parts = []
        for name, g in zip(names, grads):
            if name in delta_w and g is not None:
                gg = -lr * n_steps * g
                if lora_B0 is not None and name in lora_B0:
                    gg = _project_to_lora_subspace(gg, lora_B0[name])
                parts.append(gg.reshape(-1))
        cols.append(torch.cat(parts))
    G = torch.stack(cols, dim=1)                               # [P, N]
    tparts = []
    for name in names:
        if name in delta_w:
            t = delta_w[name]
            if lora_B0 is not None and name in lora_B0:
                t = _project_to_lora_subspace(t, lora_B0[name])
            tparts.append(t.reshape(-1))
    b = torch.cat(tparts).reshape(-1, 1)                        # [P, 1]
    sol = torch.linalg.lstsq(G, b).solution                    # [N, 1]
    return sol.reshape(-1).detach()


def svd_subspace_init(delta_w, n, input_shape, first_layer_key='layers.0.weight',
                      scale=0.5, noise=0.15, seed=None, device='cpu'):
    """SPEAR-inspired initialization for the N>1 superposition problem.

    The first-layer weight change ΔW₁ = -lr·Σ_i c_i g_i x_iᵀ has rank ≤ N, and (SPEAR Thm 3.1–3.2)
    the inputs are a linear mixture of its right-singular-vectors: Xᵀ = Q⁻¹R with ΔW₁ = L·R (SVD).
    So the N images live in the row-space of R. Initializing the reconstructions from the top-N right
    singular vectors places them in the correct low-rank subspace from the start — far better than
    random noise, which is what lets the joint optimization collapse to superpositions. The
    optimization (+ a repulsion/diversity penalty) then finds the unmixing Q within that subspace.

    Returns x_init: [N, *input_shape].
    """
    W = delta_w[first_layer_key].detach().to(device)          # [hidden, input_dim]
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)        # Vh: [k, input_dim], rows span inputs
    k = min(n, Vh.shape[0])
    dirs = Vh[:k]                                              # top-k input-space directions
    dirs = dirs / (dirs.abs().amax(dim=1, keepdim=True) + 1e-8) * scale   # to a sane image scale
    if k < n:                                                 # pad if fewer components than samples
        dirs = torch.cat([dirs, torch.randn(n - k, W.shape[1], device=device) * scale], dim=0)
    x0 = dirs.reshape(n, *input_shape)
    if noise > 0:
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        x0 = x0 + torch.randn(x0.shape, device=device, generator=gen) * (noise * scale)
    return x0


def get_ntk_loss(model_at_theta0, delta_w, x, coefficients, lr, n_steps,
                 lora_B0=None, loss_type='l2'):
    """Compute the NTK reconstruction loss.

    Two loss types:
    - 'l2': Σ_l ||ΔW_l - predicted_ΔW_l||² (original, scale-sensitive)
    - 'cosine': 1 - cos(ΔW_flat, predicted_flat) (scale-invariant, direction-only)
      From ImpMIA (Smorodinsky et al.) — focuses on gradient direction.

    where predicted_ΔW_l = -lr * n_steps * Σ_i c_i ∇_{W_l} f(θ₀; x_i)

    Works with both oracle (fixed) and free (learnable) coefficients.

    Args:
        model_at_theta0: model frozen at θ₀ (must be in eval mode but allow grad)
        delta_w: dict of target weight changes (θ_T - θ₀)
        x: [N, C, H, W] reconstructed images (requires_grad=True)
        coefficients: [N] coefficients c_i (fixed tensor or requires_grad=True)
        lr: learning rate used during training
        n_steps: number of gradient steps T
        lora_B0: optional dict mapping layer name -> B₀ matrix [d_out, r]
        loss_type: 'l2' or 'cosine'

    Returns:
        loss: scalar
    """
    # Forward pass with coefficient weighting
    values = model_at_theta0(x).squeeze()
    output = (values * coefficients).sum()

    # Compute ∇_θ f weighted by coefficients = Σ_i c_i ∇_θ f(θ₀; x_i)
    params = list(model_at_theta0.parameters())
    grad = torch.autograd.grad(
        outputs=output,
        inputs=params,
        create_graph=True,
        retain_graph=True,
    )

    # After 1 step: ΔW = -lr * Σ_i c_i ∇_θ f(θ₀; x_i)
    # After T steps (NTK approx): ΔW ≈ -lr * T * Σ_i c̄_i ∇_θ f(θ₀; x_i)

    # Collect target and predicted parts
    target_parts = []
    predicted_parts = []

    for (name, param), g in zip(model_at_theta0.named_parameters(), grad):
        if name in delta_w:
            target = delta_w[name]
            predicted = -lr * n_steps * g

            # Project both into LoRA subspace if B₀ provided for this layer
            if lora_B0 is not None and name in lora_B0:
                B0 = lora_B0[name]
                target = _project_to_lora_subspace(target, B0)
                predicted = _project_to_lora_subspace(predicted, B0)

            target_parts.append(target.view(-1))
            predicted_parts.append(predicted.view(-1))

    if loss_type == 'cosine':
        target_flat = torch.cat(target_parts)
        predicted_flat = torch.cat(predicted_parts)
        cos_sim = torch.nn.functional.cosine_similarity(
            target_flat.unsqueeze(0), predicted_flat.unsqueeze(0))
        return 1 - cos_sim.squeeze()
    else:
        # L2 loss (original)
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for t, p in zip(target_parts, predicted_parts):
            loss = loss + (t - p).pow(2).sum()
        return loss


def get_ntk_verify_loss(x):
    """Constraint loss: keep x in valid range [-1, 1]."""
    loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    loss = loss + (x - 1).relu().pow(2).sum()
    loss = loss + (-1 - x).relu().pow(2).sum()
    return loss


def get_coeff_penalty(coefficients, model_at_theta0=None, x=None, y=None,
                      box_weight=COEFF_BOX_WEIGHT,
                      consistency_weight=COEFF_CONSISTENCY_WEIGHT,
                      sign_weight=0.0,
                      min_coeff=COEFF_MIN_MAGNITUDE):
    """Penalty on free coefficients c, mirroring Haim et al.'s λ regularization.

    Three terms:
    1. Box constraint: c ∈ [-1, 1] (two-sided, since c = (σ-y)/N can be ±)
       Haim et al. use weight=5 for their one-sided λ ≥ 0.05 constraint.
    2. Sign enforcement: c > min_coeff for y=0, c < -min_coeff for y=1.
       Mirrors Haim's `5 * (-λ + min_lambda).relu().pow(2).sum()` but adapted
       for our formulation where c carries the sign (y ∈ {0,1}, not {-1,+1}).
       Haim's best runs tune min_lambda to 0.42-0.50 (near max).
    3. Self-consistency: |c - (σ(f(θ₀; x)) - y)/N|² (optional, soft)
       Uses no_grad on the predicted c to avoid re-coupling x and c.

    Args:
        coefficients: [N] learnable coefficients (requires_grad=True)
        model_at_theta0: model at θ₀ (needed for consistency penalty)
        x: [N, C, H, W] current x estimate (needed for consistency penalty)
        y: [N] binary labels (needed for consistency and sign penalty)
        box_weight: weight for box constraint (default: 5.0)
        consistency_weight: weight for consistency penalty (default: 0.0)
        sign_weight: weight for sign enforcement (default: 0.0 = off)
        min_coeff: minimum |c| magnitude for sign enforcement (default: 0.05)

    Returns:
        loss: scalar penalty
    """
    loss = torch.tensor(0.0, device=coefficients.device, dtype=coefficients.dtype)

    # Box: penalize |c| > 1
    loss = loss + box_weight * (coefficients - 1).relu().pow(2).sum()
    loss = loss + box_weight * (-1 - coefficients).relu().pow(2).sum()

    # Sign enforcement: c > min_coeff for y=0, c < -min_coeff for y=1
    # expected_sign: +1 for y=0, -1 for y=1
    # Penalizes: sign_weight * max(0, -expected_sign * c + min_coeff)²
    if sign_weight > 0 and y is not None:
        expected_sign = 1 - 2 * y  # +1 for y=0, -1 for y=1
        sign_violation = (-expected_sign * coefficients + min_coeff).relu()
        loss = loss + sign_weight * sign_violation.pow(2).sum()

    # Self-consistency (soft constraint — no_grad on target to decouple)
    if consistency_weight > 0 and model_at_theta0 is not None and x is not None and y is not None:
        with torch.no_grad():
            logits = model_at_theta0(x).view(-1)
            c_predicted = (torch.sigmoid(logits) - y) / x.shape[0]
        loss = loss + consistency_weight * (coefficients - c_predicted).pow(2).sum()

    return loss


def run_ntk_extraction(model_at_theta0, delta_w, coefficients,
                       lr_train, n_steps, n_per_class,
                       extraction_lr=EXTRACTION_LR,
                       extraction_epochs=EXTRACTION_EPOCHS,
                       init_scale=EXTRACTION_INIT_SCALE,
                       eval_every=EXTRACTION_EVAL_EVERY,
                       optimizer_type='lbfgs',
                       lora_B0=None,
                       free_coefficients=False,
                       y=None,
                       coeff_lr=COEFF_LR,
                       coeff_box_weight=COEFF_BOX_WEIGHT,
                       coeff_consistency_weight=COEFF_CONSISTENCY_WEIGHT,
                       coeff_optimizer_type='sgd',
                       coeff_init=COEFF_INIT,
                       coeff_sign_weight=COEFF_SIGN_WEIGHT,
                       coeff_min_magnitude=COEFF_MIN_MAGNITUDE,
                       loss_type='l2',
                       verify_weight=1.0,
                       x_init=None,
                       init_seed=None,
                       input_shape=(1, 28, 28),
                       closed_form_coeff=False,
                       coeff_refresh_every=500,
                       diversity_weight=0.0,
                       tv_weight=0.0,
                       pixel_box=False,
                       ds_mean=None,
                       device='cpu', verbose=True):
    """Run NTK-based reconstruction.

    Two modes:
    - Oracle (free_coefficients=False): coefficients are fixed constants
      computed from the true private data. Upper bound on attack quality.
    - Free (free_coefficients=True): coefficients are optimized jointly
      with x, mirroring Haim et al.'s λ optimization. Realistic attack.

    For L-BFGS: x and c are optimized jointly in one optimizer.
    For SGD/Adam: separate optimizers with different learning rates
    (c gets smaller lr, matching Haim et al.'s extraction_lambda_lr).

    Args:
        model_at_theta0: model frozen at θ₀
        delta_w: dict of target weight changes
        coefficients: [N] coefficients — oracle values if free_coefficients=False,
                      or oracle reference for diagnostics if free_coefficients=True
        lr_train: learning rate used during training
        n_steps: number of training steps T
        n_per_class: samples per class to reconstruct
        extraction_lr: learning rate for x reconstruction
        extraction_epochs: number of reconstruction iterations
        init_scale: initialization scale for x
        optimizer_type: 'lbfgs', 'sgd', or 'adam'
        lora_B0: optional dict {layer_name: B₀ tensor} for LoRA projection
        free_coefficients: if True, optimize c alongside x
        y: [N] binary labels (needed for consistency penalty in free mode)
        coeff_lr: learning rate for c optimizer (free mode, SGD/Adam only)
        coeff_box_weight: weight for c ∈ [-1, 1] box constraint
        coeff_consistency_weight: weight for consistency penalty
        verify_weight: weight for pixel box constraint L_verify (default: 1.0)
        x_init: optional [N, 1, 28, 28] warm-start for x (e.g., from previous T)
        init_seed: optional random seed for x initialization (for restarts)
        device: computation device

    Returns:
        x_recon: [N, 1, 28, 28] reconstructed images
        results: dict with loss history and (if free mode) coefficient info
    """
    model_at_theta0.eval()
    extraction_amount = n_per_class * 2

    # Initialize x
    if x_init is not None:
        x = x_init.clone().to(device)
        if verbose:
            print(f"  Warm-starting x from provided x_init")
    else:
        if init_seed is not None:
            torch.manual_seed(init_seed)
        # input_shape defaults to MNIST (1,28,28); flowers32=(3,32,32), flowers64=(3,64,64).
        x = torch.randn(extraction_amount, *input_shape, device=device) * init_scale
    x.requires_grad_(True)

    # Proper [0,1] pixel box operates on the DISPLAYED image (x + ds_mean); coerce ds_mean to a
    # broadcastable device tensor once (shape (1,C,H,W) or (C,H,W)).
    if pixel_box and ds_mean is not None:
        if not torch.is_tensor(ds_mean):
            ds_mean = torch.as_tensor(ds_mean)
        ds_mean = ds_mean.to(device=device, dtype=x.dtype).detach()

    # Initialize coefficients
    if free_coefficients:
        if coeff_init == 'sign_aware' and y is not None:
            # c > 0 for y=0, c < 0 for y=1; init at midpoint of valid range
            signs = 1 - 2 * y  # +1 for y=0, -1 for y=1
            c = (signs * 0.25).clone().to(device=device, dtype=x.dtype)
        elif coeff_init == 'uniform' and y is not None:
            # Haim-style: random magnitude in [0.1, 0.5] with correct sign
            signs = 1 - 2 * y
            magnitudes = torch.rand(extraction_amount, device=device, dtype=x.dtype) * 0.4 + 0.1
            c = (signs * magnitudes).clone()
        else:
            # 'zeros' — original behavior
            c = torch.zeros(extraction_amount, device=device, dtype=x.dtype)
        c.requires_grad_(True)
        oracle_c = coefficients  # save oracle reference for diagnostics
        if verbose:
            init_desc = coeff_init if coeff_init != 'zeros' else 'zeros'
            print(f"  Free-coefficient mode: {extraction_amount} coefficients, "
                  f"init={init_desc}, sign_weight={coeff_sign_weight}, "
                  f"min_coeff={coeff_min_magnitude}")
    else:
        c = coefficients  # fixed, no grad
        oracle_c = coefficients

    # Analytic closed-form coefficients (ANA-GIA): c is NOT a free SGD variable — it is re-solved by
    # least-squares from the current x each refresh interval. Overrides free_coefficients bookkeeping.
    if closed_form_coeff:
        free_coefficients = False
        oracle_c = coefficients
        c = compute_lstsq_coefficients(model_at_theta0, delta_w, x.detach(), lr_train, n_steps,
                                       lora_B0=lora_B0)

    # Build optimizers
    # For free-coefficient mode, c always gets its own separate optimizer with small lr
    # (mirroring Haim et al.'s separate opt_l with lr=1e-4 for λ).
    # L-BFGS handles x only; c is decoupled to avoid L-BFGS overwhelming c updates.
    def _make_coeff_opt(c_param, lr, opt_type='sgd'):
        if opt_type == 'adam':
            return torch.optim.Adam([c_param], lr=lr)
        return torch.optim.SGD([c_param], lr=lr, momentum=0.9)

    if optimizer_type == 'lbfgs':
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=20,
                                history_size=50, line_search_fn='strong_wolfe')
        opt_c = _make_coeff_opt(c, coeff_lr, coeff_optimizer_type) if free_coefficients else None
    elif optimizer_type == 'adam':
        opt = torch.optim.Adam([x], lr=extraction_lr)
        opt_c = _make_coeff_opt(c, coeff_lr, coeff_optimizer_type) if free_coefficients else None
    else:
        opt = torch.optim.SGD([x], lr=extraction_lr, momentum=0.9)
        opt_c = _make_coeff_opt(c, coeff_lr, coeff_optimizer_type) if free_coefficients else None

    loss_history = []
    ntk_loss_history = []
    coeff_history = []  # track c convergence

    if optimizer_type == 'lbfgs':
        _last_ntk = [0.0]
        _last_verify = [0.0]
        _last_coeff = [0.0]

        for epoch in range(extraction_epochs):
            def closure():
                opt.zero_grad()
                if opt_c is not None:
                    opt_c.zero_grad()
                ntk_loss = get_ntk_loss(
                    model_at_theta0, delta_w, x, c, lr_train, n_steps,
                    lora_B0=lora_B0, loss_type=loss_type,
                )
                verify_loss = get_ntk_verify_loss(x)
                loss = ntk_loss + verify_weight * verify_loss

                if diversity_weight > 0:  # SPEAR-spirit source separation: repel superpositions
                    loss = loss + diversity_weight * get_diversity_penalty(x)
                if tv_weight > 0:         # natural-image TV prior ("the lever is the prior")
                    loss = loss + tv_weight * get_tv_penalty(x)
                if pixel_box and ds_mean is not None:  # proper [0,1] box on the IMAGE (x+ds_mean)
                    loss = loss + verify_weight * get_pixel_box_loss(x, ds_mean)

                if free_coefficients:
                    coeff_loss = get_coeff_penalty(
                        c, model_at_theta0, x, y,
                        box_weight=coeff_box_weight,
                        consistency_weight=coeff_consistency_weight,
                        sign_weight=coeff_sign_weight,
                        min_coeff=coeff_min_magnitude,
                    )
                    loss = loss + coeff_loss
                    _last_coeff[0] = coeff_loss.item()

                loss.backward()
                _last_ntk[0] = ntk_loss.item()
                _last_verify[0] = verify_loss.item()
                return loss

            loss_val = opt.step(closure)
            # Step the separate c optimizer (uses gradients computed in closure)
            if opt_c is not None:
                opt_c.step()
            # Analytic coefficient refresh: re-solve c by least-squares from the current x.
            if closed_form_coeff and (epoch % coeff_refresh_every == 0):
                new_c = compute_lstsq_coefficients(model_at_theta0, delta_w, x.detach(),
                                                   lr_train, n_steps, lora_B0=lora_B0)
                with torch.no_grad():
                    c.copy_(new_c)

            if _last_ntk[0] != _last_ntk[0]:  # NaN check
                if verbose:
                    print(f"NaN at epoch {epoch}, stopping.")
                break

            loss_history.append(loss_val.item())
            ntk_loss_history.append(_last_ntk[0])

            if free_coefficients:
                coeff_history.append(c.detach().clone().cpu().tolist())

            if verbose and epoch % eval_every == 0:
                msg = (f"  NTK extraction epoch {epoch}: ntk={_last_ntk[0]:.4e} "
                       f"verify={_last_verify[0]:.4e}")
                if free_coefficients or closed_form_coeff:
                    c_vals = c.detach().cpu().tolist()
                    c_err = (c.detach() - oracle_c.to(c.device)).abs().mean().item() if (oracle_c is not None and oracle_c.shape[0] == c.shape[0]) else 0
                    msg += f" coeff_pen={_last_coeff[0]:.4e} c={[f'{v:.3f}' for v in c_vals]} c_err={c_err:.4f}"
                print(msg)
    else:
        # First-order optimizers (SGD, Adam)
        for epoch in range(extraction_epochs):
            ntk_loss = get_ntk_loss(
                model_at_theta0, delta_w, x, c, lr_train, n_steps,
                lora_B0=lora_B0, loss_type=loss_type,
            )
            verify_loss = get_ntk_verify_loss(x)
            loss = ntk_loss + verify_weight * verify_loss

            if diversity_weight > 0:  # SPEAR-spirit source separation: repel superpositions
                loss = loss + diversity_weight * get_diversity_penalty(x)
            if tv_weight > 0:         # natural-image TV prior ("the lever is the prior")
                loss = loss + tv_weight * get_tv_penalty(x)
            if pixel_box and ds_mean is not None:  # proper [0,1] box on the IMAGE (x+ds_mean)
                loss = loss + verify_weight * get_pixel_box_loss(x, ds_mean)

            if free_coefficients:
                coeff_loss = get_coeff_penalty(
                    c, model_at_theta0, x, y,
                    box_weight=coeff_box_weight,
                    consistency_weight=coeff_consistency_weight,
                    sign_weight=coeff_sign_weight,
                    min_coeff=coeff_min_magnitude,
                )
                loss = loss + coeff_loss

            if torch.isnan(ntk_loss):
                if verbose:
                    print(f"NaN at epoch {epoch}, stopping.")
                break

            opt.zero_grad()
            if opt_c is not None:
                opt_c.zero_grad()
            loss.backward()
            opt.step()
            if opt_c is not None:
                opt_c.step()
            # Analytic coefficient refresh: re-solve c by least-squares from the current x.
            if closed_form_coeff and (epoch % coeff_refresh_every == 0):
                new_c = compute_lstsq_coefficients(model_at_theta0, delta_w, x.detach(),
                                                   lr_train, n_steps, lora_B0=lora_B0)
                with torch.no_grad():
                    c.copy_(new_c)

            loss_history.append(loss.item())
            ntk_loss_history.append(ntk_loss.item())

            if free_coefficients:
                coeff_history.append(c.detach().clone().cpu().tolist())

            if verbose and epoch % eval_every == 0:
                msg = (f"  NTK extraction epoch {epoch}: ntk={ntk_loss.item():.4e} "
                       f"verify={verify_loss.item():.4e}")
                if free_coefficients or closed_form_coeff:
                    c_vals = c.detach().cpu().tolist()
                    c_err = (c.detach() - oracle_c.to(c.device)).abs().mean().item() if (oracle_c is not None and oracle_c.shape[0] == c.shape[0]) else 0
                    msg += f" coeff_pen={coeff_loss.item():.4e} c={[f'{v:.3f}' for v in c_vals]} c_err={c_err:.4f}"
                print(msg)

    # Log final loss component magnitudes
    if verbose and loss_history:
        final_ntk = ntk_loss_history[-1] if ntk_loss_history else 0
        print(f"  Final loss components: ntk={final_ntk:.4e}, "
              f"verify={_last_verify[0] if optimizer_type == 'lbfgs' else verify_loss.item():.4e}")
        if free_coefficients:
            print(f"    coeff_pen={_last_coeff[0] if optimizer_type == 'lbfgs' else coeff_loss.item():.4e}, "
                  f"verify_weight={verify_weight}")

    x_recon = x.detach().clone()
    result_dict = {
        'loss_history': loss_history,
        'ntk_loss_history': ntk_loss_history,
        'verify_weight': verify_weight,
    }

    if free_coefficients:
        result_dict['final_coefficients'] = c.detach().clone().cpu()
        result_dict['oracle_coefficients'] = oracle_c.cpu() if oracle_c is not None else None
        result_dict['coeff_history'] = coeff_history
        if oracle_c is not None and oracle_c.shape[0] == c.shape[0]:
            result_dict['coeff_error'] = (c.detach().cpu() - oracle_c.cpu()).abs().mean().item()

    return x_recon, result_dict


def get_pixel_box_loss(x, ds_mean):
    """Proper [0,1] pixel box: the reconstruction lives in centered space, so the DISPLAYED image is
    x + ds_mean. The default box only bounds x to [-1,1]; it lets x + ds_mean escape [0,1] and clip
    (novel Q-B clipped 50%, tanking its raw SSIM). This penalizes the *image* leaving [0,1] directly."""
    img = x + ds_mean
    return (img - 1.0).relu().pow(2).sum() + (-img).relu().pow(2).sum()


def get_tv_penalty(x):
    """Total-variation prior for NATURAL images — the dominant, highest-leverage prior in gradient
    inversion ("the lever is the prior, not the gradient match"; the thesis ViT work went 0.02->0.55
    with TV). Penalizes mean |neighbouring-pixel difference|. Unlike SVD-init this is an x-space prior,
    so it applies to the LoRA case too. x: [N, C, H, W]."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def get_diversity_penalty(x, min_dist=0.5):
    """Repulsive penalty for reconstructed images closer than min_dist.

    Adapted from Haim et al.'s diversity_loss (extraction.py:19-27).
    Uses sigmoid to create a smooth penalty that activates when nearest-
    neighbor distance falls below min_dist.

    Args:
        x: [N, C, H, W] reconstructed images
        min_dist: distance threshold below which penalty activates

    Returns:
        penalty: scalar (0 if no pair is too close)
    """
    flat_x = x.view(x.shape[0], -1)
    # Pairwise L2 distances
    D = torch.cdist(flat_x, flat_x, p=2)
    D.fill_diagonal_(float('inf'))
    nn_dist = D.min(dim=1).values
    relevant_nns = nn_dist[nn_dist < min_dist]
    if relevant_nns.shape[0] > 0:
        return relevant_nns.mul(-20).sigmoid().mean()
    return torch.tensor(0.0, device=x.device, dtype=x.dtype)


def run_sequential_peeling(model_at_theta0, delta_w, n, lr, n_steps, input_shape,
                           extraction_epochs=6000, lr_x=0.05, init_scale=0.03,
                           tv_weight=0.1, verify_weight=1.0, lora_B0=None,
                           pixel_box=False, ds_mean=None,
                           seed=0, device='cpu', verbose=False):
    """Greedy source peeling for the N>1 superposition wall (LoRA-native — an x-space method).

    Reconstruct ONE source (single image x_j + its scalar coefficient c_j) from the current residual
    ΔW, subtract its NTK contribution (-lr·T·c_j·∇f(x_j)) from the residual, and repeat n times. Each
    step is a single-source problem, so the N sources can't blend into a superposition. A TV prior
    steadies each single-image recovery. Returns x_recon [n, *input_shape] and a small results dict.
    """
    residual = {k: v.detach().clone() for k, v in delta_w.items()}
    params = list(model_at_theta0.parameters())
    if pixel_box and ds_mean is not None:                 # box the IMAGE (x+ds_mean) to [0,1]
        if not torch.is_tensor(ds_mean):
            ds_mean = torch.as_tensor(ds_mean)
        ds_mean = ds_mean.to(device=device, dtype=params[0].dtype).detach()
    names = [nm for nm, _ in model_at_theta0.named_parameters()]
    recons, coeffs = [], []
    for j in range(n):
        g = torch.Generator(device=device).manual_seed(seed + 1000 * j)
        x = (torch.randn(1, *input_shape, device=device, generator=g) * init_scale).requires_grad_(True)
        c = torch.zeros(1, device=device, requires_grad=True)              # single free coefficient
        opt = torch.optim.Adam([x, c], lr=lr_x)
        for _ in range(extraction_epochs):
            opt.zero_grad()
            loss = get_ntk_loss(model_at_theta0, residual, x, c, lr, n_steps, lora_B0=lora_B0)
            loss = loss + verify_weight * get_ntk_verify_loss(x)
            if pixel_box and ds_mean is not None:
                loss = loss + verify_weight * get_pixel_box_loss(x, ds_mean)
            if tv_weight > 0:
                loss = loss + tv_weight * get_tv_penalty(x)
            loss.backward()
            opt.step()
        recons.append(x.detach())
        coeffs.append(float(c.detach()))
        # Subtract this source's contribution from the residual (compute grad WITH autograd, edit under no_grad).
        out = model_at_theta0(x.detach()).sum()
        grads = torch.autograd.grad(out, params, retain_graph=False, allow_unused=True)
        with torch.no_grad():
            for name, gg in zip(names, grads):
                if name in residual and gg is not None:
                    contrib = -lr * n_steps * c.detach() * gg
                    if lora_B0 is not None and name in lora_B0:
                        contrib = _project_to_lora_subspace(contrib, lora_B0[name])
                    residual[name] = residual[name] - contrib
        if verbose:
            print(f"  peel {j+1}/{n}: c={coeffs[-1]:.3f}")
    x_recon = torch.cat(recons, dim=0)
    return x_recon, {'loss_history': [], 'ntk_loss_history': [], 'peel_coefficients': coeffs,
                     'verify_weight': verify_weight}


def run_ntk_extraction_with_restarts(n_restarts=1, base_seed=0, **kwargs):
    """Run extraction multiple times with different init seeds, keep best.

    Args:
        n_restarts: number of restarts (1 = no restart, just run once)
        base_seed: base seed for generating per-restart init seeds
        **kwargs: all arguments to run_ntk_extraction

    Returns:
        best_x: reconstructed images from the best restart
        best_results: results dict from the best restart
    """
    if n_restarts <= 1:
        return run_ntk_extraction(**kwargs)

    best_loss = float('inf')
    best_x = None
    best_results = None
    verbose = kwargs.get('verbose', True)

    for i in range(n_restarts):
        init_seed = base_seed + i * 1000
        if verbose:
            print(f"\n  --- Restart {i+1}/{n_restarts} (init_seed={init_seed}) ---")
        kwargs_i = dict(kwargs, init_seed=init_seed)
        x_recon, results = run_ntk_extraction(**kwargs_i)

        final_loss = results['ntk_loss_history'][-1] if results['ntk_loss_history'] else float('inf')
        if verbose:
            print(f"    Final NTK loss: {final_loss:.4e}")

        if final_loss < best_loss:
            best_loss = final_loss
            best_x = x_recon
            best_results = results

    if verbose:
        print(f"\n  Best restart: loss={best_loss:.4e}")

    return best_x, best_results


def run_ntk_extraction_n_sweep(model_at_theta0, delta_w, lr_train, n_steps,
                               n_per_class_candidates,
                               oracle_coefficients=None,
                               oracle_y=None,
                               free_coefficients=True,
                               device='cpu', verbose=True, **kwargs):
    """Try multiple N values and return the best reconstruction.

    Like Haim et al., the attacker doesn't know the true dataset size.
    Try each candidate n_per_class, run extraction, keep the one with
    lowest final NTK loss.

    For each N, labels are assumed balanced binary: y = [0]*N + [1]*N.

    Args:
        model_at_theta0: model frozen at θ₀
        delta_w: dict of target weight changes
        lr_train: learning rate used during training
        n_steps: number of training steps T
        n_per_class_candidates: list of n_per_class values to try
        oracle_coefficients: [true_N] oracle coefficients (for diagnostics)
        oracle_y: [true_N] true labels (for diagnostics)
        free_coefficients: whether to use free-coefficient mode
        device: computation device
        **kwargs: passed to run_ntk_extraction

    Returns:
        best_x: reconstructed images from the best N
        best_results: results dict from the best N
        best_n_per_class: the winning N value
        all_results: dict mapping n_per_class -> (x_recon, results)
    """
    best_loss = float('inf')
    best_x = None
    best_results = None
    best_n = None
    all_results = {}

    for n_pc in n_per_class_candidates:
        if verbose:
            print(f"\n  --- N sweep: trying n_per_class={n_pc} ({n_pc*2} images) ---")

        # For free-coefficient mode, we don't pass oracle c
        # For oracle mode, oracle_coefficients must match extraction_amount
        if free_coefficients:
            # Pass None for coefficients — oracle_c has wrong shape when
            # n_pc != n_per_class_train and is only used for diagnostics
            coeffs = None
            y = torch.cat([
                torch.zeros(n_pc, device=device, dtype=torch.float64),
                torch.ones(n_pc, device=device, dtype=torch.float64),
            ])
        else:
            # Oracle mode: can only use if N matches
            if oracle_coefficients is not None and len(oracle_coefficients) == n_pc * 2:
                coeffs = oracle_coefficients
            else:
                if verbose:
                    print(f"    Skipping N={n_pc*2}: oracle coefficients don't match (have {len(oracle_coefficients) if oracle_coefficients is not None else 0})")
                continue
            y = oracle_y

        x_recon, results = run_ntk_extraction(
            model_at_theta0, delta_w, coeffs,
            lr_train=lr_train, n_steps=n_steps, n_per_class=n_pc,
            free_coefficients=free_coefficients,
            y=y,
            device=device, verbose=verbose, **kwargs,
        )

        all_results[n_pc] = (x_recon, results)

        # Pick best by final NTK loss
        final_loss = results['ntk_loss_history'][-1] if results['ntk_loss_history'] else float('inf')
        if verbose:
            print(f"    N={n_pc*2}: final NTK loss = {final_loss:.4e}")
        if final_loss < best_loss:
            best_loss = final_loss
            best_x = x_recon
            best_results = results
            best_n = n_pc

    if verbose:
        print(f"\n  N sweep winner: n_per_class={best_n} ({best_n*2} images), loss={best_loss:.4e}")

    return best_x, best_results, best_n, all_results
