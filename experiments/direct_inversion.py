"""DI-Phase 0: direct weight inversion toy (experiment_plan.md Part B).

Treat fine-tuning as a deterministic differentiable map θ_T = F(θ₀, {x_i}) and
recover the private data by minimising ‖θ_T − F(θ₀, {x̂_i})‖² with Adam on x̂,
autograd backpropagating through an *unrolled* SGD implementation of F.

Scope (the spec): MNIST MLP, N=4 (n_per_class=2), LoRA r=8, GELU, full batch,
T ∈ {1,2,5,10,20}, Regime A (endpoint matching), x̂ init from noise. Deliverable:
SSIM-vs-T curve. Sanity: T=1 must reproduce the single-gradient NTK baseline
(experiments.run_experiment_b free-coefficient LoRA).

Why GELU: unrolling SGD differentiates *through* the gradient (second order), so
the activation must have a well-defined double-backward. GELU is C^∞; ModifiedRelu
(a custom autograd.Function with no double-backward) must never be used here.

Usage (WEXAC):
    python -m experiments.direct_inversion --sweep_T --save --device cuda
    python -m experiments.direct_inversion --sanity_t1 --device cuda
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

from experiments.configs import (
    RESULTS_DIR, FIGURES_DIR, TRAIN_LR, EXTRACTION_INIT_SCALE,
)
from experiments.run_experiment_b import create_model, load_pretrained, make_activation
from experiments.ntk_steps import compute_multi_step_update_lora
from experiments.ntk_extraction import get_ntk_verify_loss
from experiments.data_utils import get_finetuning_data
from experiments.metrics import compute_ssim

# The MLP is [Linear(784,1000,bias), Linear(1000,1000), Linear(1000,1)]; LoRA is
# applied to all three. Only layer 0 carries a bias (frozen). Keys match the
# composed state_dict emitted by lora_wrapper.compose_state_dict.
_LAYER_KEYS = ['layers.0.weight', 'layers.1.weight', 'layers.2.weight']
_IN_FEATURES = {0: 784, 1: 1000, 2: 1000}


def _lora_forward(frozen, A, B, b0, x, scaling, act):
    """Functional forward of the LoRA-composed MLP (differentiable in x, A, B).

    Mirrors CreateModel.NeuralNetwork.forward exactly: activation after layers 0
    and 1, raw logit out of layer 2. W_eff_l = frozen_l + scaling·(B_l @ A_l).
    """
    h = x.view(x.shape[0], -1)
    w0 = frozen[0] + scaling * (B[0] @ A[0])
    h = act(F.linear(h, w0, b0))
    w1 = frozen[1] + scaling * (B[1] @ A[1])
    h = act(F.linear(h, w1, None))
    w2 = frozen[2] + scaling * (B[2] @ A[2])
    return F.linear(h, w2, None)


def unrolled_finetune_lora(frozen, b0, B0, x_hat, y, lr, T, scaling, act):
    """F: unroll T full-batch SGD steps on the LoRA params, return composed θ_T.

    Starts from the known LoRA init A=0, B=B0 (the recipe is known — Regime A), and
    matches compute_multi_step_update_lora's plain SGD (no momentum). Uses
    create_graph=True so the returned weights are twice-differentiable w.r.t. x_hat.
    """
    A = {k: torch.zeros(A_rank_shape(B0, k), dtype=x_hat.dtype, device=x_hat.device,
                        requires_grad=True) for k in (0, 1, 2)}
    B = {k: B0[k].clone().requires_grad_(True) for k in (0, 1, 2)}

    for _ in range(T):
        logits = _lora_forward(frozen, A, B, b0, x_hat, scaling, act).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        params = [A[0], A[1], A[2], B[0], B[1], B[2]]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gA, gB = grads[:3], grads[3:]
        A = {k: A[k] - lr * gA[i] for i, k in enumerate((0, 1, 2))}
        B = {k: B[k] - lr * gB[i] for i, k in enumerate((0, 1, 2))}

    theta_T = {
        'layers.0.weight': frozen[0] + scaling * (B[0] @ A[0]),
        'layers.0.bias': b0,
        'layers.1.weight': frozen[1] + scaling * (B[1] @ A[1]),
        'layers.2.weight': frozen[2] + scaling * (B[2] @ A[2]),
    }
    return theta_T


def A_rank_shape(B0, k):
    """A_k has shape [rank, in_features_k]; rank is read from B0_k = [out, rank]."""
    rank = B0[k].shape[1]
    return (rank, _IN_FEATURES[k])


def di_endpoint_loss(theta_T_hat, theta_T_target):
    """Regime A endpoint loss: Σ_l ‖θ_T_hat_l − θ_T_target_l‖² over weight keys.

    Equivalent to matching ΔW (θ₀ cancels). layers.0.bias is frozen and identical,
    so it contributes zero — included for completeness.
    """
    total = None
    for k in theta_T_target:
        if k in theta_T_hat:
            term = (theta_T_hat[k] - theta_T_target[k]).pow(2).sum()
            total = term if total is None else total + term
    return total


def generate_target(x_ft, y_ft, T, rank, activation, lr, device):
    """Produce the true θ_T = F(θ₀, x_true) via the existing (non-differentiable) recipe."""
    model = load_pretrained(device=device)
    upd = compute_multi_step_update_lora(
        model, x_ft.clone(), y_ft.clone(), lr=lr, n_steps=T, rank=rank,
        activation_name=activation,
    )
    theta_0 = upd['theta_0']
    frozen = {i: theta_0[f'layers.{i}.weight'] for i in (0, 1, 2)}
    b0 = theta_0['layers.0.bias']
    B0 = {i: upd['lora_B0'][f'layers.{i}.weight'] for i in (0, 1, 2)}
    return upd['theta_T'], frozen, b0, B0, upd['ds_mean']


def run_direct_inversion(T, N=4, rank=8, activation='gelu', lr=TRAIN_LR,
                         outer_iters=2000, adam_lr=0.05, n_restarts=4,
                         box_weight=1.0, init_scale=EXTRACTION_INIT_SCALE,
                         x_init=None, seed=42, device='cuda', verbose=True):
    """Recover x̂ ≈ x_true by minimising ‖θ_T − F(θ₀, x̂)‖² with Adam on x̂.

    Returns dict with x_recon (centered space), ground truth, SSIM, final loss.
    """
    torch.manual_seed(seed)
    n_per_class = N // 2
    x_ft, y_ft, digits, idx = get_finetuning_data(n_per_class, seed=seed, device=device)

    theta_T_target, frozen, b0, B0, ds_mean = generate_target(
        x_ft, y_ft, T, rank, activation, lr, device)
    x_true_centered = x_ft - ds_mean if ds_mean is not None else x_ft
    scaling = 1.0  # alpha=rank default → alpha/rank = 1
    act = make_activation(activation)

    best_loss, best_x = float('inf'), None
    for r in range(n_restarts):
        torch.manual_seed(seed * 100 + r)
        if x_init is not None and r == 0:
            x_hat = x_init.clone().to(device).detach().requires_grad_(True)
        else:
            x_hat = (init_scale * torch.randn(N, 1, 28, 28, dtype=torch.float64,
                                              device=device)).requires_grad_(True)
        opt = torch.optim.Adam([x_hat], lr=adam_lr)
        for it in range(outer_iters):
            opt.zero_grad()
            theta_T_hat = unrolled_finetune_lora(frozen, b0, B0, x_hat, y_ft,
                                                 lr, T, scaling, act)
            loss = di_endpoint_loss(theta_T_hat, theta_T_target)
            if box_weight > 0:
                loss = loss + box_weight * get_ntk_verify_loss(x_hat)
            loss.backward()
            opt.step()
            if verbose and (it % max(1, outer_iters // 5) == 0 or it == outer_iters - 1):
                print(f"    [T={T} restart {r}] iter {it}: loss={loss.item():.4e}")
        fl = loss.item()
        if fl < best_loss:
            best_loss, best_x = fl, x_hat.detach().clone()

    ssim_per, ssim_mean = compute_ssim(best_x, x_true_centered, ds_mean)
    if verbose:
        print(f"  T={T}: SSIM={ssim_mean:.4f}  final_loss={best_loss:.4e}")
    return {
        'T': T, 'x_recon': best_x, 'x_train': x_ft, 'ds_mean': ds_mean,
        'ssim': ssim_mean, 'ssim_per': ssim_per, 'final_loss': best_loss,
        'digits': digits, 'N': N, 'rank': rank, 'activation': activation,
    }


def sanity_check_T1(N=4, rank=8, activation='gelu', device='cuda',
                    extraction_epochs=20000, outer_iters=2000):
    """DI at T=1 vs the single-gradient NTK ORACLE baseline (Experiment B, LoRA).

    Both recover the same digits. Compare against ORACLE (fixed coefficients),
    not free-coefficient: free-coeff LBFGS needs ~50k epochs to converge, so at a
    sanity budget it under-reports and gives a spurious gap. DI does *exact*
    endpoint matching at T=1 while the NTK method is linearized, so DI is expected
    to be at or slightly above the oracle NTK SSIM (the difference is the NTK
    linearization error DI avoids). Flags a large gap rather than hard-failing.
    """
    from experiments.run_experiment_b import run_single_config
    print("=== T=1 sanity: DI vs Experiment B oracle NTK (LoRA) ===")
    di = run_direct_inversion(T=1, N=N, rank=rank, activation=activation,
                              outer_iters=outer_iters, n_restarts=2,
                              device=device, verbose=True)
    b = run_single_config(
        n_steps=1, rank=rank, n_per_class=N // 2, seed=42, run_baseline=False,
        free_coefficients=False, finetune_activation=activation,
        extraction_epochs=extraction_epochs, optimizer_type='lbfgs',
        device=device, verbose=False)
    b_ssim = b.get('lora_metrics', {}).get('ssim')
    print(f"  DI-T1 SSIM              = {di['ssim']:.4f}")
    print(f"  Exp-B T=1 oracle SSIM   = {b_ssim}")
    if b_ssim is not None:
        gap = di['ssim'] - b_ssim
        print(f"  DI − oracle = {gap:+.4f}  ->",
              "OK (DI ≳ oracle, as expected — exact vs linearized)"
              if gap > -0.15 else "UNEXPECTED — DI below oracle, inspect F")
    return di, b


def sweep_T(Ts=(1, 2, 5, 10, 20), N=4, rank=8, activation='gelu',
            outer_iters=2000, n_restarts=4, box_weight=1.0, device='cuda',
            warm_start=True):
    """Run direct inversion across T, warm-starting x̂ from the previous T."""
    results = {}
    x_prev = None
    for T in Ts:
        print(f"\n{'='*60}\nDirect inversion T={T}\n{'='*60}")
        res = run_direct_inversion(
            T=T, N=N, rank=rank, activation=activation, outer_iters=outer_iters,
            n_restarts=n_restarts, box_weight=box_weight,
            x_init=(x_prev if warm_start else None),
            device=device, verbose=True)
        results[T] = res
        x_prev = res['x_recon']
    return results


def _plot_ssim_vs_T(results, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Ts = sorted(results)
    ss = [results[T]['ssim'] for T in Ts]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(Ts, ss, marker='o', linewidth=2, color='#1f77b4')
    for T, s in zip(Ts, ss):
        ax.annotate(f'{s:.3f}', (T, s), textcoords='offset points',
                    xytext=(0, 8), fontsize=9, ha='center')
    ax.set_xlabel('unrolled SGD steps T')
    ax.set_ylabel('reconstruction SSIM')
    ax.set_title('Direct weight inversion: SSIM vs T (MNIST, N=4, LoRA r=8, GELU)')
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def _save_results(results, save_path):
    save_dict = {}
    for T, res in results.items():
        save_dict[T] = {
            'x_recon': res['x_recon'].cpu(),
            'x_train': res['x_train'].cpu(),
            'ds_mean': res['ds_mean'].cpu() if res['ds_mean'] is not None else None,
            'ssim': res['ssim'], 'final_loss': res['final_loss'],
            'digits': res['digits'],
        }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Saved tensors to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_T', action='store_true')
    parser.add_argument('--sanity_t1', action='store_true')
    parser.add_argument('--Ts', type=int, nargs='+', default=[1, 2, 5, 10, 20])
    parser.add_argument('--N', type=int, default=4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--outer_iters', type=int, default=2000)
    parser.add_argument('--n_restarts', type=int, default=4)
    parser.add_argument('--box_weight', type=float, default=1.0,
                        help='Weight on the soft [-1,1] box penalty (default 1.0 '
                             '= current behaviour). Raise to curb [0,1] saturation.')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.sanity_t1:
        sanity_check_T1(N=args.N, rank=args.rank, activation=args.activation,
                        device=device, outer_iters=args.outer_iters)

    if args.sweep_T:
        results = sweep_T(Ts=tuple(args.Ts), N=args.N, rank=args.rank,
                          activation=args.activation, outer_iters=args.outer_iters,
                          n_restarts=args.n_restarts, box_weight=args.box_weight,
                          device=device)
        print("\n=== SSIM vs T ===")
        for T in sorted(results):
            print(f"  T={T:>3d}: SSIM={results[T]['ssim']:.4f}  "
                  f"loss={results[T]['final_loss']:.3e}")
        tag = args.tag or f"N{args.N}_r{args.rank}_{args.activation}"
        _plot_ssim_vs_T(results, os.path.join(FIGURES_DIR, 'direct_inversion',
                                              f"di_ssim_vs_T_{tag}.png"))
        if args.save:
            _save_results(results, os.path.join(RESULTS_DIR,
                                                f"direct_inversion_{tag}.pth"))
            from experiments.plotting import plot_reconstruction_grid
            for T, res in results.items():
                plot_reconstruction_grid(
                    res['x_train'], x_recon_lora=res['x_recon'],
                    ssim_lora=res['ssim_per'] if 'ssim_per' in res else None,
                    ds_mean=res['ds_mean'],
                    save_path=os.path.join(FIGURES_DIR, 'direct_inversion',
                                           f"di_grid_{tag}_T{T}.png"))

    print("\n=== Done ===")
