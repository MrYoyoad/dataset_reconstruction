"""Modified KKT reconstruction loss for the NTK regime.

In the NTK regime, the weight update ΔW = -η Σ_i c_i ∇_θ f(θ₀; x_i)
where c_i are known coefficients. We optimize x to match this target.

Key differences from get_kkt_loss in extraction.py:
- Target is ΔW (weight change), not W (full weights)
- Coefficients c_i are fixed (not optimized) — eliminates λ unknowns
- Only optimize over x (images), not x + λ
- Features evaluated at θ₀ (model is frozen at init)
"""

import torch
import torch.nn as nn
from experiments.configs import (
    EXTRACTION_LR, EXTRACTION_INIT_SCALE,
    EXTRACTION_EPOCHS, EXTRACTION_EVAL_EVERY,
)


def get_ntk_loss(model_at_theta0, delta_w, x, coefficients, lr, n_steps):
    """Compute the NTK reconstruction loss.

    Loss = Σ_l ||ΔW_l - predicted_ΔW_l||²

    where predicted_ΔW_l = -lr * n_steps * (1/N) Σ_i c_i ∇_{W_l} f(θ₀; x_i)

    This is exact at T=1, approximate at T>1 (NTK approximation).

    Args:
        model_at_theta0: model frozen at θ₀ (must be in eval mode but allow grad)
        delta_w: dict of target weight changes (θ_T - θ₀)
        x: [N, C, H, W] reconstructed images (requires_grad=True)
        coefficients: [N] known coefficients c_i
        lr: learning rate used during training
        n_steps: number of gradient steps T

    Returns:
        loss: scalar, sum of per-layer ||ΔW - predicted_ΔW||²
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

    # predicted ΔW = -lr * n_steps * grad
    # (The grad already contains the 1/N from the coefficient computation,
    #  but coefficients = (σ(f)-y)/N so they include the 1/N factor.
    #  Actually the forward computes Σ c_i f(x_i) and autograd gives
    #  Σ c_i ∇f(x_i), then predicted ΔW = -lr * this * n_steps.)
    #
    # Wait — the actual SGD step is:
    #   θ_{t+1} = θ_t - lr * ∇_θ L(θ_t)
    #   where ∇_θ L = (1/N) Σ_i (σ(f(x_i)) - y_i) ∇_θ f(x_i)
    #              = Σ_i c_i ∇_θ f(x_i)   [since c_i = (σ(f) - y)/N]
    #
    # So after 1 step: ΔW = -lr * Σ_i c_i ∇_θ f(θ₀; x_i)
    # After T steps (NTK approx): ΔW ≈ -lr * T * Σ_i c̄_i ∇_θ f(θ₀; x_i)
    #
    # Here we use the T=1 coefficients for all steps (approximation).

    loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    param_names = [n for n, _ in model_at_theta0.named_parameters()]

    for (name, param), g in zip(model_at_theta0.named_parameters(), grad):
        # Find matching delta_w key
        if name in delta_w:
            target = delta_w[name]
            predicted = -lr * n_steps * g
            loss = loss + (target - predicted).pow(2).sum()

    return loss


def get_ntk_verify_loss(x):
    """Constraint loss: keep x in valid range [-1, 1]."""
    loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    loss = loss + (x - 1).relu().pow(2).sum()
    loss = loss + (-1 - x).relu().pow(2).sum()
    return loss


def run_ntk_extraction(model_at_theta0, delta_w, coefficients,
                       lr_train, n_steps, n_per_class,
                       extraction_lr=EXTRACTION_LR,
                       extraction_epochs=EXTRACTION_EPOCHS,
                       init_scale=EXTRACTION_INIT_SCALE,
                       eval_every=EXTRACTION_EVAL_EVERY,
                       device='cpu', verbose=True):
    """Run NTK-based reconstruction.

    Unlike KKT reconstruction, we don't optimize λ — coefficients are known.

    Args:
        model_at_theta0: model frozen at θ₀
        delta_w: dict of target weight changes
        coefficients: [N] known coefficients
        lr_train: learning rate used during training
        n_steps: number of training steps T
        n_per_class: samples per class to reconstruct
        extraction_lr: learning rate for reconstruction
        extraction_epochs: number of reconstruction iterations
        init_scale: initialization scale for x
        device: computation device

    Returns:
        x_recon: [N, 1, 28, 28] reconstructed images
        results: dict with loss history
    """
    model_at_theta0.eval()
    extraction_amount = n_per_class * 2

    # Initialize x
    x = torch.randn(extraction_amount, 1, 28, 28, device=device) * init_scale
    x.requires_grad_(True)

    opt = torch.optim.SGD([x], lr=extraction_lr, momentum=0.9)

    loss_history = []
    ntk_loss_history = []

    for epoch in range(extraction_epochs):
        ntk_loss = get_ntk_loss(
            model_at_theta0, delta_w, x, coefficients, lr_train, n_steps
        )
        verify_loss = get_ntk_verify_loss(x)
        loss = ntk_loss + verify_loss

        if torch.isnan(ntk_loss):
            if verbose:
                print(f"NaN at epoch {epoch}, stopping.")
            break

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        ntk_loss_history.append(ntk_loss.item())

        if verbose and epoch % eval_every == 0:
            print(f"  NTK extraction epoch {epoch}: ntk={ntk_loss.item():.4e} "
                  f"verify={verify_loss.item():.4e}")

    x_recon = x.detach().clone()
    return x_recon, {
        'loss_history': loss_history,
        'ntk_loss_history': ntk_loss_history,
    }
