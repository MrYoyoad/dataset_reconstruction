"""Multi-step gradient computation and NTK coefficient extraction.

For Experiment B: take T gradient steps, compute ΔW = θ_T - θ₀,
and extract the known coefficients c_i = (σ(f(θ₀; x_i)) - y_i) / N.
"""

import copy
import torch
import torch.nn.functional as F


def compute_known_coefficients(model, x, y):
    """Compute the known NTK coefficients at the current model state.

    At initialization (or any fixed θ):
        c_i = (σ(f(θ; x_i)) - y_i) / N

    where σ is the sigmoid function (from BCEWithLogitsLoss gradient).

    Args:
        model: NeuralNetwork at θ (eval mode)
        x: [N, C, H, W] input images
        y: [N] binary labels in {0, 1}

    Returns:
        coefficients: [N] tensor, the c_i values
    """
    model.eval()
    with torch.no_grad():
        logits = model(x).view(-1)
        probs = torch.sigmoid(logits)
        N = x.shape[0]
        coefficients = (probs - y) / N
    return coefficients


def compute_multi_step_update(model, x, y, lr, n_steps, reduce_mean=True):
    """Take n_steps full-batch SGD steps and compute the weight change.

    Args:
        model: NeuralNetwork (will be modified in-place)
        x: [N, C, H, W] training data
        y: [N] binary labels
        lr: learning rate
        n_steps: number of gradient steps
        reduce_mean: whether to subtract dataset mean

    Returns:
        dict with:
            'theta_0': state_dict at initialization (before any steps)
            'theta_T': state_dict after T steps
            'delta_w': dict of (theta_T - theta_0) per parameter
            'coefficients_at_init': [N] known coefficients at θ₀
            'ds_mean': dataset mean (or None)
            'loss_history': list of loss values per step
    """
    # Mean subtraction
    ds_mean = None
    if reduce_mean:
        ds_mean = x.mean(dim=0, keepdim=True)
        x = x - ds_mean

    # Save θ₀
    theta_0 = {k: v.clone() for k, v in model.state_dict().items()}

    # Compute coefficients at init
    coefficients_at_init = compute_known_coefficients(model, x, y)

    # Run T steps of full-batch SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_history = []

    model.train()
    for step in range(n_steps):
        logits = model(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    # Save θ_T
    theta_T = {k: v.clone() for k, v in model.state_dict().items()}

    # Compute ΔW = θ_T - θ₀
    delta_w = {}
    for key in theta_0:
        delta_w[key] = theta_T[key] - theta_0[key]

    return {
        'theta_0': theta_0,
        'theta_T': theta_T,
        'delta_w': delta_w,
        'coefficients_at_init': coefficients_at_init,
        'ds_mean': ds_mean,
        'loss_history': loss_history,
    }


def compute_multi_step_update_lora(model, x, y, lr, n_steps, rank=8,
                                    alpha=None, reduce_mean=True):
    """Same as compute_multi_step_update but trains only LoRA parameters.

    Returns the same dict plus the composed delta_w (effective weight change
    from LoRA composition).
    """
    from experiments.lora_wrapper import apply_lora, compose_state_dict

    # Mean subtraction
    ds_mean = None
    if reduce_mean:
        ds_mean = x.mean(dim=0, keepdim=True)
        x = x - ds_mean

    # Save θ₀ (before LoRA applied)
    theta_0 = {k: v.clone() for k, v in model.state_dict().items()}

    # Apply LoRA
    lora_params = apply_lora(model, rank=rank, alpha=alpha)

    # Compute coefficients at init (LoRA starts at W₀ since A=0)
    coefficients_at_init = compute_known_coefficients(model, x, y)

    # Run T steps of LoRA SGD
    optimizer = torch.optim.SGD(lora_params, lr=lr)
    loss_history = []

    model.train()
    for step in range(n_steps):
        logits = model(x).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    # Composed state dict after training
    theta_T_composed = compose_state_dict(model)

    # Compute ΔW = composed(θ_T) - θ₀
    delta_w = {}
    for key in theta_0:
        if key in theta_T_composed:
            delta_w[key] = theta_T_composed[key] - theta_0[key]

    return {
        'theta_0': theta_0,
        'theta_T': theta_T_composed,
        'delta_w': delta_w,
        'coefficients_at_init': coefficients_at_init,
        'ds_mean': ds_mean,
        'loss_history': loss_history,
    }
