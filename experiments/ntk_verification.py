"""NTK regime verification diagnostics.

Verifies whether the NTK approximation holds at a given step count T:
1. Relative weight change: ||θ_T - θ₀|| / ||θ₀|| (want < 0.01)
2. Feature cosine similarity: cos(∇_θ f(θ₀; x), ∇_θ f(θ_T; x)) (want > 0.99)
3. Coefficient drift: how much c_i changed from step 0 to step T
"""

import torch
import torch.nn.functional as F


def _get_per_sample_gradients(model, x):
    """Compute ∇_θ f(θ; x_i) for each sample individually.

    Returns a list of dicts, one per sample, each dict mapping param_name → gradient.
    Also returns flattened gradient vectors [N, total_params].
    """
    model.eval()
    grad_list = []
    flat_grads = []

    for i in range(x.shape[0]):
        xi = x[i:i+1]
        model.zero_grad()
        out = model(xi).squeeze()
        out.backward()

        grads = {}
        flat_parts = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
                flat_parts.append(param.grad.clone().view(-1))
            else:
                grads[name] = torch.zeros_like(param)
                flat_parts.append(torch.zeros(param.numel(), dtype=param.dtype, device=param.device))

        grad_list.append(grads)
        flat_grads.append(torch.cat(flat_parts))

    return grad_list, torch.stack(flat_grads)


def compute_relative_weight_change(theta_0, theta_T):
    """Compute ||θ_T - θ₀|| / ||θ₀|| per layer and overall.

    Args:
        theta_0: dict of parameter tensors at init
        theta_T: dict of parameter tensors after T steps

    Returns:
        dict with:
            'per_layer': dict mapping key → relative change
            'overall': float, global relative change
    """
    per_layer = {}
    total_delta_sq = 0.0
    total_theta0_sq = 0.0

    for key in theta_0:
        if key not in theta_T:
            continue
        delta = theta_T[key] - theta_0[key]
        theta0_norm = theta_0[key].norm().item()
        delta_norm = delta.norm().item()

        if theta0_norm > 0:
            per_layer[key] = delta_norm / theta0_norm
        else:
            per_layer[key] = float('inf') if delta_norm > 0 else 0.0

        total_delta_sq += delta.pow(2).sum().item()
        total_theta0_sq += theta_0[key].pow(2).sum().item()

    overall = (total_delta_sq ** 0.5) / (total_theta0_sq ** 0.5) if total_theta0_sq > 0 else float('inf')

    return {'per_layer': per_layer, 'overall': overall}


def compute_feature_stability(model_at_theta0, model_at_theta_T, x):
    """Compute cosine similarity between features at θ₀ and θ_T.

    Features = ∇_θ f(θ; x_i) (the Jacobian of the network output w.r.t. parameters).

    Args:
        model_at_theta0: model loaded with θ₀ weights
        model_at_theta_T: model loaded with θ_T weights
        x: [N, C, H, W] input data

    Returns:
        dict with:
            'per_sample': [N] cosine similarities
            'mean': float, average cosine similarity
    """
    _, flat_grads_0 = _get_per_sample_gradients(model_at_theta0, x)
    _, flat_grads_T = _get_per_sample_gradients(model_at_theta_T, x)

    cos_sims = F.cosine_similarity(flat_grads_0, flat_grads_T, dim=1)

    return {
        'per_sample': cos_sims,
        'mean': cos_sims.mean().item(),
    }


def compute_coefficient_drift(model_at_theta0, model_at_theta_T, x, y):
    """Compute how much the coefficients c_i changed between θ₀ and θ_T.

    c_i(θ) = (σ(f(θ; x_i)) - y_i) / N

    Args:
        model_at_theta0: model at θ₀
        model_at_theta_T: model at θ_T
        x: [N, C, H, W]
        y: [N] binary labels

    Returns:
        dict with:
            'c_init': [N] coefficients at θ₀
            'c_final': [N] coefficients at θ_T
            'abs_drift': [N] |c_T - c_0| per sample
            'relative_drift': [N] |c_T - c_0| / |c_0| per sample
            'mean_abs_drift': float
    """
    from experiments.ntk_steps import compute_known_coefficients

    c_init = compute_known_coefficients(model_at_theta0, x, y)
    c_final = compute_known_coefficients(model_at_theta_T, x, y)

    abs_drift = (c_final - c_init).abs()
    relative_drift = abs_drift / (c_init.abs() + 1e-10)

    return {
        'c_init': c_init,
        'c_final': c_final,
        'abs_drift': abs_drift,
        'relative_drift': relative_drift,
        'mean_abs_drift': abs_drift.mean().item(),
    }


def verify_ntk_at_step(theta_0, theta_T, model_class_fn, x, y):
    """Full NTK verification at a given step count.

    Args:
        theta_0: state_dict at init
        theta_T: state_dict after T steps
        model_class_fn: callable() that creates a fresh model
        x: [N, C, H, W] training data
        y: [N] labels

    Returns:
        dict with all diagnostics:
            'weight_change': relative weight change diagnostics
            'feature_stability': feature cosine similarity diagnostics
            'coefficient_drift': coefficient drift diagnostics
    """
    # Create two models with θ₀ and θ_T
    model_0 = model_class_fn()
    model_0.load_state_dict(theta_0)
    model_0.eval()

    model_T = model_class_fn()
    model_T.load_state_dict(theta_T)
    model_T.eval()

    weight_change = compute_relative_weight_change(theta_0, theta_T)
    feature_stability = compute_feature_stability(model_0, model_T, x)
    coefficient_drift = compute_coefficient_drift(model_0, model_T, x, y)

    return {
        'weight_change': weight_change,
        'feature_stability': feature_stability,
        'coefficient_drift': coefficient_drift,
    }
