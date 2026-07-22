"""NTK regime verification diagnostics.

Verifies whether the NTK approximation holds at a given step count T:
1. Relative weight change: ||θ_T - θ₀|| / ||θ₀|| (want < 0.01)
2. Feature cosine similarity: cos(∇_θ f(θ₀; x), ∇_θ f(θ_T; x)) (want > 0.99)
3. Coefficient drift: how much c_i changed from step 0 to step T

Additional diagnostics (logged but don't block execution):
- ΔW effective rank (expect ≤ N under NTK)
- NTK Gram matrix condition number (high = numerically unstable)
- Numerical health (NaN/Inf detection)
"""

import torch
import torch.nn.functional as F

from experiments.configs import NTK_WEIGHT_CHANGE_THRESHOLD, NTK_FEATURE_COS_THRESHOLD


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


def check_numerical_health(delta_w, coefficients=None, label=""):
    """Check for NaN/Inf in ΔW and coefficients. Returns True if healthy."""
    prefix = f"  [{label}] " if label else "  "
    healthy = True
    for key, val in delta_w.items():
        if torch.isnan(val).any():
            print(f"{prefix}ERROR: NaN detected in delta_w['{key}']!")
            healthy = False
        if torch.isinf(val).any():
            print(f"{prefix}ERROR: Inf detected in delta_w['{key}']!")
            healthy = False
    if coefficients is not None:
        if torch.isnan(coefficients).any():
            print(f"{prefix}ERROR: NaN detected in coefficients!")
            healthy = False
        if torch.isinf(coefficients).any():
            print(f"{prefix}ERROR: Inf detected in coefficients!")
            healthy = False
    return healthy


def compute_delta_w_effective_rank(delta_w, n_samples, verbose=True):
    """Compute effective rank of each layer's ΔW. Under NTK, expect rank ≤ N.

    For NTK: ΔW_l = -lr*T * Σ c_i ∇_{W_l} f(θ₀;x_i) is a sum of N rank-1
    matrices, so each layer should have rank ≤ N. Reports per-layer ranks.

    Returns:
        dict with 'effective_rank' (max across layers), 'n_samples'
    """
    max_rank = 0
    for key in sorted(delta_w.keys()):
        v = delta_w[key]
        if v.dim() >= 2:
            sv = torch.linalg.svdvals(v.float())
            layer_rank = (sv > sv[0] * 1e-6).sum().item()
            max_rank = max(max_rank, layer_rank)
            if verbose and layer_rank > n_samples:
                print(f"    ΔW[{key}] rank={layer_rank} > N={n_samples}")

    if verbose:
        print(f"  ΔW max layer rank: {max_rank} (expect ≤ N={n_samples})")
        if max_rank > n_samples * 2:
            print(f"    NOTE: rank >> N — NTK approximation may be poor")

    return {'effective_rank': max_rank, 'n_samples': n_samples}


def compute_gram_condition_number(model_at_theta0, x, verbose=True):
    """Compute condition number of the NTK Gram matrix K.

    K_{ij} = <∇f(θ₀;x_i), ∇f(θ₀;x_j)>

    High condition number means numerically unstable reconstruction.

    Returns:
        dict with 'condition_number', 'gram_matrix'
    """
    _, flat_grads = _get_per_sample_gradients(model_at_theta0, x)
    K = flat_grads @ flat_grads.T  # [N, N]
    cond_K = torch.linalg.cond(K.float()).item()

    if verbose:
        print(f"  NTK Gram cond number: {cond_K:.2e}")
        if cond_K > 1e6:
            print(f"    NOTE: High condition number — reconstruction may be numerically unstable")

    return {'condition_number': cond_K, 'gram_matrix': K.cpu()}


def compute_linearization_error(model_at_theta0, delta_w, x, y, lr, n_steps):
    """Compute the NTK linearization error using oracle data.

    Measures: ||ΔW_actual - ΔW_predicted|| / ||ΔW_actual||
    where ΔW_predicted = -lr * T * Σ c_i ∇f(θ₀; x_i)

    This is the ground truth NTK approximation quality — independent
    of the reconstruction optimizer. If this is high, the NTK approximation
    itself is failing, not just the extraction.

    Args:
        model_at_theta0: model loaded with θ₀ weights
        delta_w: dict of actual weight changes (θ_T - θ₀)
        x: [N, C, H, W] true training data
        y: [N] true labels
        lr: learning rate used during fine-tuning
        n_steps: number of gradient steps T

    Returns:
        dict with:
            'relative_error': ||ΔW - ΔW_pred|| / ||ΔW|| (overall)
            'per_layer': dict mapping key → relative error per layer
            'absolute_error': ||ΔW - ΔW_pred|| (overall)
    """
    from experiments.ntk_steps import compute_known_coefficients

    model_at_theta0.eval()
    c = compute_known_coefficients(model_at_theta0, x, y)

    # Compute predicted ΔW = -lr * T * Σ c_i ∇f(θ₀; x_i)
    # Forward pass with coefficient weighting
    values = model_at_theta0(x).squeeze()
    output = (values * c).sum()

    params = list(model_at_theta0.parameters())
    grad = torch.autograd.grad(outputs=output, inputs=params,
                                create_graph=False, retain_graph=False)

    per_layer = {}
    total_err_sq = 0.0
    total_dw_sq = 0.0

    for (name, param), g in zip(model_at_theta0.named_parameters(), grad):
        if name in delta_w:
            predicted = -lr * n_steps * g
            actual = delta_w[name]
            err = (actual - predicted).pow(2).sum().item()
            dw_norm_sq = actual.pow(2).sum().item()

            total_err_sq += err
            total_dw_sq += dw_norm_sq

            if dw_norm_sq > 0:
                per_layer[name] = (err ** 0.5) / (dw_norm_sq ** 0.5)
            else:
                per_layer[name] = 0.0

    overall_err = total_err_sq ** 0.5
    overall_dw = total_dw_sq ** 0.5
    relative = overall_err / overall_dw if overall_dw > 0 else 0.0

    return {
        'relative_error': relative,
        'per_layer': per_layer,
        'absolute_error': overall_err,
    }


def compute_function_space_lin_error(model_at_anchor, model_at_theta_T, x, delta):
    """Function-space Taylor residual of the network output around an anchor.

    Measures how linear the network *function* Φ is over the segment from the
    anchor to θ_T, in the plan's exact terms (experiment_plan.md "Addition 3"):

        ||Φ(θ_T; x) − [Φ(θ_anchor; x) + ∇Φ(θ_anchor; x)·δ]|| / ||Φ(θ_T; x) − Φ(θ_anchor; x)||

    where δ = θ_T − θ_anchor is the (known) displacement. This is pure
    approximation quality — it does NOT reconstruct x_i and is independent of
    the extraction optimizer. It is the headline linearization-error(α) curve:
    as the anchor α→1 the segment shrinks (δ→0) so the residual should drop.
    Complements the weight-space compute_linearization_error above.

    Args:
        model_at_anchor: model loaded with θ_anchor weights (real activation, not
            the ModifiedRelu extraction surrogate — this is the true Φ).
        model_at_theta_T: model loaded with θ_T weights (same architecture).
        x: [N, C, H, W] evaluation inputs (mean-subtracted, as during training).
        delta: dict param_name → (θ_T − θ_anchor)[param], the Taylor direction.

    Returns:
        dict with 'relative_error', 'absolute_error', 'delta_output_norm'.
    """
    model_at_anchor.eval()
    model_at_theta_T.eval()

    with torch.no_grad():
        f_anchor = model_at_anchor(x).view(-1)
        f_T = model_at_theta_T(x).view(-1)

    # First-order term: directional derivative ∇Φ(θ_anchor; x_i)·δ per sample.
    # Reuse the per-sample gradient machinery, then contract with δ.
    grad_list, _ = _get_per_sample_gradients(model_at_anchor, x)
    jvp = torch.zeros_like(f_anchor)
    for i, grads in enumerate(grad_list):
        acc = torch.zeros((), dtype=f_anchor.dtype, device=f_anchor.device)
        for name, g in grads.items():
            if name in delta:
                acc = acc + (g * delta[name]).sum()
        jvp[i] = acc

    predicted = f_anchor + jvp
    abs_err = (f_T - predicted).norm().item()
    delta_out = (f_T - f_anchor).norm().item()
    relative = abs_err / delta_out if delta_out > 0 else 0.0

    return {
        'relative_error': relative,
        'absolute_error': abs_err,
        'delta_output_norm': delta_out,
    }


def ntk_smoke_test(theta_0, theta_T, model_class_fn, x, y, delta_w=None,
                   verbose=True, label=""):
    """Comprehensive NTK regime check before extraction.

    Runs:
    1. NTK regime check (feature stability + weight change) — PASS/FAIL
    2. Numerical health (NaN/Inf) — flags errors
    3. ΔW effective rank — logged
    4. Gram matrix condition number — logged

    Does NOT abort — just prints warnings so we can decide.

    Returns:
        passed: bool (NTK regime satisfied)
        diagnostics: dict with all measurements
    """
    prefix = f"[{label}] " if label else ""

    # 1. Full NTK verification (weight change, feature stability, coeff drift)
    diag = verify_ntk_at_step(theta_0, theta_T, model_class_fn, x, y)
    wc = diag['weight_change']['overall']
    fs = diag['feature_stability']['mean']
    cd = diag['coefficient_drift']['mean_abs_drift']

    passed = (wc < NTK_WEIGHT_CHANGE_THRESHOLD and fs > NTK_FEATURE_COS_THRESHOLD)

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"  {prefix}NTK smoke test: {status}")
        print(f"    weight_change={wc:.6f} (threshold <{NTK_WEIGHT_CHANGE_THRESHOLD})")
        print(f"    feature_cos={fs:.6f} (threshold >{NTK_FEATURE_COS_THRESHOLD})")
        print(f"    coeff_drift={cd:.6f}")
        if not passed:
            print(f"    WARNING: NTK regime NOT satisfied. Results may be unreliable.")

    diagnostics = {
        'weight_change': wc,
        'feature_stability': fs,
        'coefficient_drift': cd,
        'ntk_passed': passed,
    }

    # 2. Numerical health
    if delta_w is not None:
        coeffs = diag['coefficient_drift']['c_init']
        healthy = check_numerical_health(delta_w, coeffs, label=label)
        diagnostics['numerical_healthy'] = healthy

    # 3. ΔW effective rank
    if delta_w is not None:
        n_samples = x.shape[0]
        rank_info = compute_delta_w_effective_rank(delta_w, n_samples, verbose=verbose)
        diagnostics['delta_w_effective_rank'] = rank_info['effective_rank']

    # 4. Gram matrix condition number
    model_0 = model_class_fn()
    model_0.load_state_dict(theta_0)
    model_0.eval()
    gram_info = compute_gram_condition_number(model_0, x, verbose=verbose)
    diagnostics['gram_condition_number'] = gram_info['condition_number']

    return passed, diagnostics
