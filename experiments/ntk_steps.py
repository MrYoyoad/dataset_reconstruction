"""Multi-step gradient computation and NTK coefficient extraction.

For Experiment B: take T gradient steps, compute ΔW = θ_T - θ₀,
and extract the known coefficients c_i = (σ(f(θ₀; x_i)) - y_i) / N.
"""

import copy
import math
import torch
import torch.nn.functional as F


WARMUP_FRACTION = 0.1  # 10% warmup, matching HuggingFace default


def get_step_lr(lr, step, n_steps, schedule):
    """Compute per-step learning rate for a given schedule.

    Args:
        lr: base (peak) learning rate
        step: current step index (0-based)
        n_steps: total number of steps
        schedule: one of 'constant', 'cosine', 'linear', 'cosine_warmup'

    Returns:
        learning rate for this step
    """
    if schedule == 'constant':
        return lr
    elif schedule == 'cosine':
        return lr * 0.5 * (1 + math.cos(math.pi * step / max(n_steps, 1)))
    elif schedule == 'linear':
        return lr * (1 - step / max(n_steps, 1))
    elif schedule == 'cosine_warmup':
        if n_steps <= 1:
            return lr  # no warmup possible with 1 step
        warmup_steps = max(1, int(n_steps * WARMUP_FRACTION))
        if step < warmup_steps:
            return lr * (step + 1) / (warmup_steps + 1)
        else:
            progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
            return lr * 0.5 * (1 + math.cos(math.pi * progress))
    else:
        return lr


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
        out = model(x)
        # Multi-class base (Tier B): the binary sigmoid-coefficient c_i=(σ(f)-y)/N
        # is undefined for a [N,K>1] head, and every caller that USES this value is
        # binary-only (the multi-class path in _honest_target discards it — it calls
        # this helper only via compute_multi_step_update_lora with n_steps=0, for the
        # loss-independent frozen/b0/B0/ds_mean). Return zeros to avoid the shape
        # crash without touching the binary path (out=[N,1] falls through unchanged).
        if out.dim() > 1 and out.shape[1] > 1:
            return torch.zeros(x.shape[0], device=out.device, dtype=out.dtype)
        logits = out.view(-1)
        probs = torch.sigmoid(logits)
        N = x.shape[0]
        coefficients = (probs - y) / N
    return coefficients


FINETUNE_OPTIMIZER_CHOICES = ['sgd', 'adamw']


def _make_finetune_optimizer(params, lr, finetune_optimizer='sgd',
                              weight_decay=0.01):
    """Create the optimizer used for the fine-tuning (forward) step.

    Args:
        params: iterable of parameters to optimize
        lr: learning rate
        finetune_optimizer: 'sgd' or 'adamw'
        weight_decay: weight decay for AdamW (ignored for SGD)

    Returns:
        torch.optim.Optimizer
    """
    if finetune_optimizer == 'sgd':
        return torch.optim.SGD(params, lr=lr)
    elif finetune_optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown finetune_optimizer: {finetune_optimizer}. "
                         f"Choose from {FINETUNE_OPTIMIZER_CHOICES}")


def compute_multi_step_update(model, x, y, lr, n_steps, reduce_mean=True,
                               activation_name=None, relu_alpha=None,
                               lr_schedule='constant',
                               finetune_optimizer='sgd',
                               weight_decay=0.01):
    """Take n_steps full-batch gradient steps and compute the weight change.

    Args:
        model: NeuralNetwork (will be modified in-place)
        x: [N, C, H, W] training data
        y: [N] binary labels
        lr: learning rate (base LR; for decaying schedules, this is the peak LR)
        n_steps: number of gradient steps
        reduce_mean: whether to subtract dataset mean
        activation_name: if provided, swap model activation before training.
            One of 'relu', 'leaky_relu', 'modified_relu'.
        relu_alpha: alpha for ModifiedRelu (only used if activation_name='modified_relu')
        lr_schedule: one of 'constant', 'cosine', 'linear', 'cosine_warmup'.
            See get_step_lr() for formulas.
        finetune_optimizer: 'sgd' (default) or 'adamw'. Controls which optimizer
            is used for the fine-tuning step. SGD is required by implicit bias
            theory; AdamW tests generalization to real-world LoRA practice.
        weight_decay: weight decay for AdamW (default 0.01, ignored for SGD).

    Returns:
        dict with:
            'theta_0': state_dict at initialization (before any steps)
            'theta_T': state_dict after T steps
            'delta_w': dict of (theta_T - theta_0) per parameter
            'coefficients_at_init': [N] known coefficients at θ₀
            'ds_mean': dataset mean (or None)
            'loss_history': list of loss values per step
    """
    # Optionally swap activation for ablation
    if activation_name is not None:
        from experiments.run_experiment_b import make_activation
        model.activation = make_activation(activation_name, relu_alpha or 149.87)
    # Mean subtraction
    ds_mean = None
    if reduce_mean:
        ds_mean = x.mean(dim=0, keepdim=True)
        x = x - ds_mean

    # Save θ₀
    theta_0 = {k: v.clone() for k, v in model.state_dict().items()}

    # Compute coefficients at init
    coefficients_at_init = compute_known_coefficients(model, x, y)

    # Run T steps of full-batch optimization
    optimizer = _make_finetune_optimizer(model.parameters(), lr,
                                         finetune_optimizer, weight_decay)
    loss_history = []

    model.train()
    for step in range(n_steps):
        # Per-step LR scheduling
        if lr_schedule != 'constant':
            step_lr = get_step_lr(lr, step, n_steps, lr_schedule)
            for pg in optimizer.param_groups:
                pg['lr'] = step_lr

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
                                    alpha=None, reduce_mean=True,
                                    activation_name=None, relu_alpha=None,
                                    lr_schedule='constant',
                                    finetune_optimizer='sgd',
                                    weight_decay=0.01):
    """Same as compute_multi_step_update but trains only LoRA parameters.

    Returns the same dict plus the composed delta_w (effective weight change
    from LoRA composition).

    Args:
        lr_schedule: one of 'constant', 'cosine', 'linear', 'cosine_warmup'.
            See get_step_lr() for formulas.
        finetune_optimizer: 'sgd' (default) or 'adamw'. Controls which optimizer
            is used for the fine-tuning step.
        weight_decay: weight decay for AdamW (default 0.01, ignored for SGD).
    """
    # Optionally swap activation for ablation
    if activation_name is not None:
        from experiments.run_experiment_b import make_activation
        model.activation = make_activation(activation_name, relu_alpha or 149.87)
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

    # Save B₀ matrices (at init) for optional projection in NTK loss.
    # The LoRA update ΔW = B₀A₁ lives in the column space of B₀.
    # Projecting the NTK loss into this subspace removes the irreducible
    # residual from full-rank gradient components outside col(B₀).
    from experiments.lora_wrapper import LoRALinear
    lora_B0 = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer, LoRALinear):
            lora_B0[f'layers.{i}.weight'] = layer.lora_B.data.clone()

    # Compute coefficients at init (LoRA starts at W₀ since A=0)
    coefficients_at_init = compute_known_coefficients(model, x, y)

    # Run T steps of LoRA optimization
    optimizer = _make_finetune_optimizer(lora_params, lr,
                                         finetune_optimizer, weight_decay)
    loss_history = []

    model.train()
    for step in range(n_steps):
        # Per-step LR scheduling
        if lr_schedule != 'constant':
            step_lr = get_step_lr(lr, step, n_steps, lr_schedule)
            for pg in optimizer.param_groups:
                pg['lr'] = step_lr

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
        'lora_B0': lora_B0,
    }
