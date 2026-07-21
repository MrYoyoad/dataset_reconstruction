"""Experiment B: Multi-Step NTK Reconstruction from Pre-Trained Weights.

Attack scenario: pre-trained model (θ₀) is fine-tuned on private data
for T gradient steps. Attacker observes ΔW = θ_T - θ₀ and reconstructs
the private fine-tuning data.

Two extraction modes:
- Oracle: coefficients c_i computed from true x (upper bound, cheating)
- Free: coefficients c_i optimized alongside x (realistic attack)

Usage:
    # Oracle mode (default, backward-compatible)
    python -m experiments.run_experiment_b --n_steps 1 --rank 8

    # Free-coefficient mode (realistic attack)
    python -m experiments.run_experiment_b --n_steps 1 --rank 8 --free_coefficients

    # Free-coefficient with N sweep (attacker doesn't know dataset size)
    python -m experiments.run_experiment_b --n_steps 1 --free_coefficients --n_sweep

    # Ablation: consistency penalty weight
    python -m experiments.run_experiment_b --free_coefficients --consistency_weight 1.0
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from CreateModel import NeuralNetwork, ModifiedRelu

from experiments.configs import (
    INPUT_DIM, OUTPUT_DIM, MODEL_HIDDEN_LIST,
    EXTRACTION_EPOCHS, EXTRACTION_EVAL_EVERY, RESULTS_DIR, TRAIN_LR,
    PRETRAINED_MNIST_PATH, EXTRACTION_RELU_ALPHA,
    COEFF_LR, COEFF_BOX_WEIGHT, COEFF_CONSISTENCY_WEIGHT,
    COEFF_INIT, COEFF_SIGN_WEIGHT, COEFF_MIN_MAGNITUDE,
    N_PER_CLASS_EXTRACTION_SWEEP, FIGURES_DIR,
    ACTIVATION_CHOICES, LR_SCHEDULE_CHOICES,
)
from experiments.data_utils import (
    get_finetuning_data, get_control_images_in_distribution,
)
from experiments.ntk_steps import (
    compute_multi_step_update, compute_multi_step_update_lora,
    FINETUNE_OPTIMIZER_CHOICES,
)
from experiments.ntk_extraction import run_ntk_extraction, run_ntk_extraction_n_sweep
from experiments.ntk_verification import ntk_smoke_test, compute_linearization_error
from experiments.metrics import compute_all_metrics


def make_activation(activation_name, relu_alpha=EXTRACTION_RELU_ALPHA):
    """Create an activation module from a name string.

    Args:
        activation_name: one of ACTIVATION_CHOICES
        relu_alpha: alpha for ModifiedRelu (default: 149.87)

    Returns:
        nn.Module activation
    """
    if activation_name == 'relu':
        return nn.ReLU()
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU(0.01)
    elif activation_name == 'modified_relu':
        return ModifiedRelu(relu_alpha)
    # Smooth (C^inf) activations — Addition 2. GELU is the deployment-realistic
    # one (real ViTs/BERT/GPT); silu/softplus give the smoothness ordering.
    elif activation_name == 'gelu':
        return nn.GELU()
    elif activation_name == 'silu':
        return nn.SiLU()
    elif activation_name == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError(f"Unknown activation: {activation_name}")


def create_model(device='cpu', extraction=False, relu_alpha=EXTRACTION_RELU_ALPHA,
                 activation_name=None):
    """Create a NeuralNetwork matching the MNIST architecture.

    Args:
        device: computation device
        extraction: if True and activation_name is None, use ModifiedRelu
            (smooth backward pass). If False and activation_name is None,
            use standard ReLU.
        relu_alpha: alpha parameter for ModifiedRelu (default: 149.87 from
            Haim et al. W&B sweep).
        activation_name: explicit activation override. If provided, ignores
            the extraction flag. One of 'relu', 'leaky_relu', 'modified_relu'.
    """
    if activation_name is not None:
        activation = make_activation(activation_name, relu_alpha)
    else:
        activation = ModifiedRelu(relu_alpha) if extraction else nn.ReLU()
    model = NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=MODEL_HIDDEN_LIST,
        output_dim=OUTPUT_DIM,
        activation=activation,
        use_bias=False,
    )
    return model.to(device).double()


def load_pretrained(device='cpu', pretrained_path=None):
    """Load the pre-trained MNIST odd/even model as θ₀."""
    path = pretrained_path or PRETRAINED_MNIST_PATH
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = create_model(device=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def _run_extraction(model_theta0, delta_w, oracle_coefficients, y_ft,
                    lr, n_steps, n_per_class,
                    free_coefficients, coeff_lr, coeff_box_weight,
                    coeff_consistency_weight, coeff_optimizer_type,
                    coeff_init=COEFF_INIT,
                    coeff_sign_weight=COEFF_SIGN_WEIGHT,
                    coeff_min_magnitude=COEFF_MIN_MAGNITUDE,
                    loss_type='l2',
                    n_sweep=False,
                    extraction_epochs=EXTRACTION_EPOCHS,
                    optimizer_type='lbfgs', lora_B0=None,
                    verify_weight=1.0,
                    x_init=None, init_seed=None,
                    device='cpu', verbose=True):
    """Run NTK extraction in oracle, free, or N-sweep mode."""
    extra_kwargs = dict(
        coeff_init=coeff_init,
        coeff_sign_weight=coeff_sign_weight,
        coeff_min_magnitude=coeff_min_magnitude,
        loss_type=loss_type,
    )
    if n_sweep and free_coefficients:
        # N sweep: try multiple dataset sizes
        x_recon, extract_res, best_n, all_results = run_ntk_extraction_n_sweep(
            model_theta0, delta_w,
            lr_train=lr, n_steps=n_steps,
            n_per_class_candidates=N_PER_CLASS_EXTRACTION_SWEEP,
            oracle_coefficients=oracle_coefficients,
            oracle_y=y_ft,
            free_coefficients=True,
            extraction_epochs=extraction_epochs,
            optimizer_type=optimizer_type,
            lora_B0=lora_B0,
            coeff_lr=coeff_lr,
            coeff_box_weight=coeff_box_weight,
            coeff_consistency_weight=coeff_consistency_weight,
            coeff_optimizer_type=coeff_optimizer_type,
            verify_weight=verify_weight,
            device=device, verbose=verbose,
            **extra_kwargs,
        )
        extract_res['n_sweep_winner'] = best_n
        extract_res['n_sweep_all'] = {
            n: r['ntk_loss_history'][-1] if r['ntk_loss_history'] else float('inf')
            for n, (_, r) in all_results.items()
        }
    else:
        # Single N extraction (oracle or free)
        x_recon, extract_res = run_ntk_extraction(
            model_theta0, delta_w, oracle_coefficients,
            lr_train=lr, n_steps=n_steps, n_per_class=n_per_class,
            extraction_epochs=extraction_epochs,
            optimizer_type=optimizer_type,
            lora_B0=lora_B0,
            free_coefficients=free_coefficients,
            y=y_ft,
            coeff_lr=coeff_lr,
            coeff_box_weight=coeff_box_weight,
            coeff_consistency_weight=coeff_consistency_weight,
            coeff_optimizer_type=coeff_optimizer_type,
            verify_weight=verify_weight,
            x_init=x_init,
            init_seed=init_seed,
            device=device, verbose=verbose,
            **extra_kwargs,
        )
    return x_recon, extract_res


def run_single_config(n_steps, rank=None, n_per_class=1, seed=42,
                      lr=TRAIN_LR, pretrained_path=None,
                      extraction_epochs=EXTRACTION_EPOCHS,
                      run_baseline=True,
                      free_coefficients=False,
                      coeff_lr=COEFF_LR,
                      coeff_box_weight=COEFF_BOX_WEIGHT,
                      coeff_consistency_weight=COEFF_CONSISTENCY_WEIGHT,
                      coeff_optimizer_type='sgd',
                      coeff_init=COEFF_INIT,
                      coeff_sign_weight=COEFF_SIGN_WEIGHT,
                      coeff_min_magnitude=COEFF_MIN_MAGNITUDE,
                      loss_type='l2',
                      n_sweep=False,
                      optimizer_type='lbfgs',
                      relu_alpha=EXTRACTION_RELU_ALPHA,
                      verify_weight=1.0,
                      save_results=False,
                      finetune_activation=None,
                      lr_schedule='constant',
                      finetune_optimizer='sgd',
                      weight_decay=0.01,
                      x_init=None,
                      init_seed=None,
                      device='cpu', verbose=True):
    """Run Experiment B for one (n_steps, rank) configuration.

    Loads pre-trained weights as θ₀, fine-tunes on held-out MNIST test
    samples, then reconstructs using NTK loss.

    If rank is None, runs full-model (no LoRA) only.

    Returns dict with all results and metrics.
    """
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)

    # Load held-out fine-tuning data (from MNIST TEST set, not train)
    x_ft, y_ft, digits, indices = get_finetuning_data(
        n_per_class, seed=seed, device=device
    )
    mode_str = "free-coefficient" if free_coefficients else "oracle"
    if verbose:
        print(f"Fine-tuning digits: {digits}, indices: {indices}")
        print(f"n_steps={n_steps}, rank={rank}, lr={lr}, mode={mode_str}")
        print(f"  relu_alpha={relu_alpha} (ModifiedRelu for extraction)")
        if free_coefficients:
            print(f"  coeff_lr={coeff_lr}, box_weight={coeff_box_weight}, "
                  f"consistency_weight={coeff_consistency_weight}, n_sweep={n_sweep}")

    # Learning rate sanity check
    effective_lr = lr * n_steps * (n_per_class * 2)
    if effective_lr > 0.1 and verbose:
        print(f"  WARNING: effective lr*T*N={effective_lr:.4f} > 0.1 — NTK may not hold")

    # Determine effective LR (with schedule)
    # For inv_sqrt_T / inv_T: pre-compute a flat reduced LR (all steps use same LR)
    # For per-step schedules (cosine, linear, cosine_warmup): ntk_steps handles decay
    from experiments.ntk_steps import get_step_lr
    if lr_schedule == 'inv_sqrt_T':
        effective_lr_train = lr / (n_steps ** 0.5) if n_steps > 1 else lr
        training_lr = effective_lr_train
        training_schedule = 'constant'
    elif lr_schedule == 'inv_T':
        effective_lr_train = lr / n_steps if n_steps > 1 else lr
        training_lr = effective_lr_train
        training_schedule = 'constant'
    elif lr_schedule in ('cosine', 'linear', 'cosine_warmup'):
        # Per-step scheduling handled by ntk_steps; compute average for reporting
        lr_sum = sum(get_step_lr(lr, t, n_steps, lr_schedule)
                     for t in range(max(n_steps, 1)))
        effective_lr_train = lr_sum / max(n_steps, 1)
        training_lr = lr  # base LR; ntk_steps handles per-step decay
        training_schedule = lr_schedule
    else:
        effective_lr_train = lr
        training_lr = lr
        training_schedule = 'constant'

    # Determine extraction activation: if finetune_activation is set,
    # use the same activation for extraction (consistency principle).
    # If not set, use the default behavior (ReLU finetune, ModifiedRelu extract).
    if finetune_activation is not None:
        extract_activation = finetune_activation
    else:
        extract_activation = None  # use default extraction=True behavior

    results = {'n_steps': n_steps, 'rank': rank, 'n_per_class': n_per_class,
               'seed': seed, 'digits': digits, 'mode': mode_str,
               'relu_alpha': relu_alpha, 'finetune_activation': finetune_activation,
               'lr_schedule': lr_schedule, 'effective_lr': effective_lr_train,
               'finetune_optimizer': finetune_optimizer}

    if verbose and finetune_activation:
        print(f"  finetune_activation={finetune_activation} (same for extraction)")
    if verbose and finetune_optimizer != 'sgd':
        print(f"  finetune_optimizer={finetune_optimizer} (weight_decay={weight_decay})")
    if verbose and lr_schedule != 'constant':
        print(f"  lr_schedule={lr_schedule}, effective_lr={effective_lr_train:.6f}")

    # --- Full model (no LoRA) baseline ---
    if run_baseline:
        if verbose:
            print(f"\n--- Full model fine-tuning, T={n_steps} steps ({mode_str}) ---")

        model_full = load_pretrained(device=device, pretrained_path=pretrained_path)
        update_result = compute_multi_step_update(
            model_full, x_ft.clone(), y_ft.clone(),
            lr=training_lr, n_steps=n_steps,
            activation_name=finetune_activation, relu_alpha=relu_alpha,
            lr_schedule=training_schedule,
            finetune_optimizer=finetune_optimizer,
            weight_decay=weight_decay,
        )

        x_centered = x_ft - update_result['ds_mean'] if update_result['ds_mean'] is not None else x_ft

        # NTK smoke test (full model)
        def _make_model():
            return create_model(device=device,
                                activation_name=finetune_activation,
                                relu_alpha=relu_alpha)

        ntk_passed, full_diag = ntk_smoke_test(
            update_result['theta_0'], update_result['theta_T'],
            _make_model, x_centered, y_ft,
            delta_w=update_result['delta_w'],
            verbose=verbose, label="Full",
        )
        results['full_diagnostics'] = full_diag

        # Linearization error (how good is the NTK approximation itself?)
        model_lin = create_model(device=device,
                                 activation_name=finetune_activation,
                                 relu_alpha=relu_alpha)
        model_lin.load_state_dict(update_result['theta_0'])
        model_lin.eval()
        lin_error = compute_linearization_error(
            model_lin, update_result['delta_w'], x_centered, y_ft,
            lr=effective_lr_train, n_steps=n_steps,
        )
        results['full_linearization_error'] = lin_error['relative_error']
        if verbose:
            print(f"  Linearization error: {lin_error['relative_error']:.6f}")

        # Create extraction model at θ₀
        model_theta0 = load_pretrained(device=device, pretrained_path=pretrained_path)
        if extract_activation is not None:
            # Consistent activation: use same for extraction
            model_theta0_extract = create_model(
                device=device, activation_name=extract_activation,
                relu_alpha=relu_alpha)
        else:
            # Default: ModifiedRelu for extraction
            model_theta0_extract = create_model(
                device=device, extraction=True, relu_alpha=relu_alpha)
        model_theta0_extract.load_state_dict(model_theta0.state_dict())
        model_theta0_extract.eval()

        # NTK reconstruction
        x_recon_full, extract_res = _run_extraction(
            model_theta0_extract, update_result['delta_w'],
            update_result['coefficients_at_init'], y_ft,
            lr=effective_lr_train, n_steps=n_steps, n_per_class=n_per_class,
            free_coefficients=free_coefficients,
            coeff_lr=coeff_lr, coeff_box_weight=coeff_box_weight,
            coeff_consistency_weight=coeff_consistency_weight,
            coeff_optimizer_type=coeff_optimizer_type,
            coeff_init=coeff_init,
            coeff_sign_weight=coeff_sign_weight,
            coeff_min_magnitude=coeff_min_magnitude,
            loss_type=loss_type,
            n_sweep=n_sweep,
            extraction_epochs=extraction_epochs,
            optimizer_type=optimizer_type,
            lora_B0=None,
            verify_weight=verify_weight,
            x_init=x_init, init_seed=init_seed,
            device=device, verbose=verbose,
        )

        # Only compute ground-truth metrics when reconstruction N matches training N
        # (N-sweep may return more images than the training set had)
        if x_recon_full.shape[0] == x_centered.shape[0]:
            metrics_full = compute_all_metrics(x_recon_full, x_centered, update_result['ds_mean'])
            results['full_metrics'] = {k: v['mean'] for k, v in metrics_full.items()}
        else:
            if verbose:
                print(f"  N-sweep returned {x_recon_full.shape[0]} images vs {x_centered.shape[0]} ground truth — skipping SSIM")
            results['full_metrics'] = {}
        results['full_extract_res'] = extract_res
        results['x_recon_full'] = x_recon_full
        if verbose:
            ssim_val = results['full_metrics'].get('ssim', None)
            if ssim_val is not None:
                print(f"  Full model reconstruction: SSIM={ssim_val:.4f}")
            if free_coefficients and 'coeff_error' in extract_res:
                print(f"  Coefficient error (|c_free - c_oracle|): {extract_res['coeff_error']:.4f}")

    # --- LoRA version ---
    if rank is not None:
        if verbose:
            print(f"\n--- LoRA rank={rank}, T={n_steps} steps ({mode_str}) ---")

        model_lora = load_pretrained(device=device, pretrained_path=pretrained_path)
        update_result_lora = compute_multi_step_update_lora(
            model_lora, x_ft.clone(), y_ft.clone(), lr=training_lr,
            n_steps=n_steps, rank=rank,
            activation_name=finetune_activation, relu_alpha=relu_alpha,
            lr_schedule=training_schedule,
            finetune_optimizer=finetune_optimizer,
            weight_decay=weight_decay,
        )

        x_centered = x_ft - update_result_lora['ds_mean'] if update_result_lora['ds_mean'] is not None else x_ft

        # NTK smoke test (LoRA)
        def _make_model_lora():
            return create_model(device=device,
                                activation_name=finetune_activation,
                                relu_alpha=relu_alpha)

        lora_ntk_passed, lora_diag = ntk_smoke_test(
            update_result_lora['theta_0'], update_result_lora['theta_T'],
            _make_model_lora, x_centered, y_ft,
            delta_w=update_result_lora['delta_w'],
            verbose=verbose, label=f"LoRA r={rank}",
        )
        results['lora_diagnostics'] = lora_diag

        # Linearization error (LoRA)
        model_lin_lora = create_model(device=device,
                                      activation_name=finetune_activation,
                                      relu_alpha=relu_alpha)
        model_lin_lora.load_state_dict(update_result_lora['theta_0'])
        model_lin_lora.eval()
        lin_error_lora = compute_linearization_error(
            model_lin_lora, update_result_lora['delta_w'], x_centered, y_ft,
            lr=effective_lr_train, n_steps=n_steps,
        )
        results['lora_linearization_error'] = lin_error_lora['relative_error']
        if verbose:
            print(f"  LoRA linearization error: {lin_error_lora['relative_error']:.6f}")

        # Create extraction model at θ₀
        model_theta0_lora = load_pretrained(device=device, pretrained_path=pretrained_path)
        if extract_activation is not None:
            model_theta0_lora_extract = create_model(
                device=device, activation_name=extract_activation,
                relu_alpha=relu_alpha)
        else:
            model_theta0_lora_extract = create_model(
                device=device, extraction=True, relu_alpha=relu_alpha)
        model_theta0_lora_extract.load_state_dict(model_theta0_lora.state_dict())
        model_theta0_lora_extract.eval()

        # NTK reconstruction
        x_recon_lora, extract_res_lora = _run_extraction(
            model_theta0_lora_extract, update_result_lora['delta_w'],
            update_result_lora['coefficients_at_init'], y_ft,
            lr=effective_lr_train, n_steps=n_steps, n_per_class=n_per_class,
            free_coefficients=free_coefficients,
            coeff_lr=coeff_lr, coeff_box_weight=coeff_box_weight,
            coeff_consistency_weight=coeff_consistency_weight,
            coeff_optimizer_type=coeff_optimizer_type,
            coeff_init=coeff_init,
            coeff_sign_weight=coeff_sign_weight,
            coeff_min_magnitude=coeff_min_magnitude,
            loss_type=loss_type,
            n_sweep=n_sweep,
            extraction_epochs=extraction_epochs,
            optimizer_type=optimizer_type,
            lora_B0=None,
            verify_weight=verify_weight,
            x_init=x_init, init_seed=init_seed,
            device=device, verbose=verbose,
        )

        if x_recon_lora.shape[0] == x_centered.shape[0]:
            metrics_lora = compute_all_metrics(x_recon_lora, x_centered, update_result_lora['ds_mean'])
            results['lora_metrics'] = {k: v['mean'] for k, v in metrics_lora.items()}
        else:
            if verbose:
                print(f"  N-sweep returned {x_recon_lora.shape[0]} images vs {x_centered.shape[0]} ground truth — skipping SSIM")
            results['lora_metrics'] = {}
        results['lora_extract_res'] = extract_res_lora
        results['x_recon_lora'] = x_recon_lora
        if verbose:
            ssim_val = results['lora_metrics'].get('ssim', None)
            if ssim_val is not None:
                print(f"  LoRA reconstruction: SSIM={ssim_val:.4f}")
            if free_coefficients and 'coeff_error' in extract_res_lora:
                print(f"  Coefficient error (|c_free - c_oracle|): {extract_res_lora['coeff_error']:.4f}")

    # --- Control images ---
    if run_baseline:
        ds_mean = update_result['ds_mean']
    elif rank is not None:
        ds_mean = update_result_lora['ds_mean']
    else:
        raise ValueError("Nothing to run: rank is None and run_baseline is False")
    x_ctrl, y_ctrl, ctrl_digits = get_control_images_in_distribution(digits, device=device)
    x_ctrl_centered = x_ctrl - ds_mean if ds_mean is not None else x_ctrl

    recon_for_ctrl = results.get('x_recon_full', results.get('x_recon_lora'))
    if recon_for_ctrl is not None and recon_for_ctrl.shape[0] == x_ctrl_centered.shape[0]:
        metrics_ctrl = compute_all_metrics(recon_for_ctrl, x_ctrl_centered, ds_mean)
        results['control_metrics'] = {k: v['mean'] for k, v in metrics_ctrl.items()}
    else:
        results['control_metrics'] = {}

    results['x_train'] = x_ft
    results['x_ctrl'] = x_ctrl
    results['ds_mean'] = ds_mean

    return results


if __name__ == '__main__':
    from experiments.configs import get_device

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--n_per_class', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=TRAIN_LR)
    parser.add_argument('--no_baseline', action='store_true')
    parser.add_argument('--extraction_epochs', type=int, default=EXTRACTION_EPOCHS,
                        help='Number of extraction optimization iterations')
    parser.add_argument('--optimizer', type=str, default='lbfgs',
                        choices=['lbfgs', 'sgd', 'adam'])
    parser.add_argument('--device', type=str, default=None,
                        help='cpu/cuda/mps (auto-detect if omitted)')

    # Free-coefficient options
    parser.add_argument('--free_coefficients', action='store_true',
                        help='Optimize c alongside x (realistic attack)')
    parser.add_argument('--coeff_lr', type=float, default=COEFF_LR,
                        help='Learning rate for coefficient optimizer')
    parser.add_argument('--box_weight', type=float, default=COEFF_BOX_WEIGHT,
                        help='Weight for c ∈ [-1,1] box constraint')
    parser.add_argument('--consistency_weight', type=float,
                        default=COEFF_CONSISTENCY_WEIGHT,
                        help='Weight for |c - c_predicted(x)|² penalty')
    parser.add_argument('--coeff_optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'],
                        help='Optimizer type for free coefficients (default: sgd)')
    parser.add_argument('--coeff_init', type=str, default=COEFF_INIT,
                        choices=['zeros', 'sign_aware', 'uniform'],
                        help='Coefficient initialization strategy')
    parser.add_argument('--sign_weight', type=float, default=COEFF_SIGN_WEIGHT,
                        help='Weight for sign enforcement penalty (0=off)')
    parser.add_argument('--min_coeff', type=float, default=COEFF_MIN_MAGNITUDE,
                        help='Minimum |c| for sign constraint')
    parser.add_argument('--loss_type', type=str, default='l2',
                        choices=['l2', 'cosine'],
                        help='NTK loss type: l2 or cosine')

    # N sweep
    parser.add_argument('--n_sweep', action='store_true',
                        help='Try multiple N values (attacker guesses dataset size)')

    # Extraction activation & loss weights
    parser.add_argument('--relu_alpha', type=float, default=EXTRACTION_RELU_ALPHA,
                        help='Alpha for ModifiedRelu extraction activation (default: 149.87)')
    parser.add_argument('--verify_weight', type=float, default=1.0,
                        help='Weight for pixel box constraint L_verify (default: 1.0)')

    # Activation ablation (Sprint 2b)
    parser.add_argument('--finetune_activation', type=str, default=None,
                        choices=ACTIVATION_CHOICES,
                        help='Activation for both fine-tuning and extraction (default: ReLU fine-tune, ModifiedRelu extract)')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        choices=LR_SCHEDULE_CHOICES,
                        help='LR schedule: constant, inv_sqrt_T, inv_T, cosine, linear, cosine_warmup')

    # Fine-tuning optimizer (Sprint 2c realism probe)
    parser.add_argument('--finetune_optimizer', type=str, default='sgd',
                        choices=['sgd', 'adamw'],
                        help='Optimizer for fine-tuning step: sgd (theory) or adamw (realistic)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW fine-tuning (ignored for SGD)')

    # Output control
    parser.add_argument('--save_results', action='store_true',
                        help='Save tensors (.pth) and figure (.png) to results/')

    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Using device: {device}")

    results = run_single_config(
        n_steps=args.n_steps,
        rank=args.rank,
        n_per_class=args.n_per_class,
        seed=args.seed,
        lr=args.lr,
        extraction_epochs=args.extraction_epochs,
        run_baseline=not args.no_baseline,
        free_coefficients=args.free_coefficients,
        coeff_lr=args.coeff_lr,
        coeff_box_weight=args.box_weight,
        coeff_consistency_weight=args.consistency_weight,
        coeff_optimizer_type=args.coeff_optimizer,
        coeff_init=args.coeff_init,
        coeff_sign_weight=args.sign_weight,
        coeff_min_magnitude=args.min_coeff,
        loss_type=args.loss_type,
        finetune_activation=args.finetune_activation,
        lr_schedule=args.lr_schedule,
        finetune_optimizer=args.finetune_optimizer,
        weight_decay=args.weight_decay,
        n_sweep=args.n_sweep,
        optimizer_type=args.optimizer,
        relu_alpha=args.relu_alpha,
        verify_weight=args.verify_weight,
        save_results=args.save_results,
        device=device,
    )

    print("\n=== Final Results ===")
    mode = "FREE-COEFFICIENT" if args.free_coefficients else "ORACLE"
    print(f"Mode: {mode}")
    if 'full_metrics' in results:
        print(f"Full model (T={args.n_steps}): {results['full_metrics']}")
    if 'lora_metrics' in results:
        print(f"LoRA rank={args.rank} (T={args.n_steps}): {results['lora_metrics']}")
    if 'full_diagnostics' in results:
        print(f"Full NTK diagnostics: {results['full_diagnostics']}")
    if 'lora_diagnostics' in results:
        print(f"LoRA NTK diagnostics: {results['lora_diagnostics']}")
    if 'control_metrics' in results:
        print(f"Control: {results['control_metrics']}")

    # Coefficient diagnostics
    for key in ['full_extract_res', 'lora_extract_res']:
        if key in results and 'coeff_error' in results[key]:
            label = 'Full' if 'full' in key else 'LoRA'
            print(f"{label} coeff error: {results[key]['coeff_error']:.4f}")
            print(f"{label} final c: {results[key]['final_coefficients'].tolist()}")
            if results[key]['oracle_coefficients'] is not None:
                print(f"{label} oracle c: {results[key]['oracle_coefficients'].tolist()}")

    # N sweep results
    for key in ['full_extract_res', 'lora_extract_res']:
        if key in results and 'n_sweep_winner' in results[key]:
            label = 'Full' if 'full' in key else 'LoRA'
            print(f"{label} N sweep winner: n_per_class={results[key]['n_sweep_winner']}")
            print(f"{label} N sweep losses: {results[key]['n_sweep_all']}")

    # Generate figure
    from experiments.plotting import generate_experiment_b_figure
    generate_experiment_b_figure(results)

    # Save results if requested
    if args.save_results:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)

        # Build descriptive filename
        parts = [f"exp_b_T{args.n_steps}"]
        if args.rank is not None:
            parts.append(f"r{args.rank}")
        else:
            parts.append("full")
        if args.free_coefficients:
            parts.append("free")
        parts.append(f"s{args.seed}")
        parts.append(f"a{int(args.relu_alpha)}")
        base_name = "_".join(parts)

        # Save tensors (.pth)
        save_dict = {}
        for k in ['x_recon_full', 'x_recon_lora', 'x_train', 'x_ctrl', 'ds_mean',
                   'full_metrics', 'lora_metrics', 'control_metrics',
                   'full_diagnostics', 'lora_diagnostics',
                   'full_extract_res', 'lora_extract_res']:
            if k in results:
                save_dict[k] = results[k]
        # Compute actual param counts for figure labels
        from experiments.lora_wrapper import apply_lora, get_lora_param_count
        _m = load_pretrained(device='cpu')
        full_params = sum(p.numel() for p in _m.parameters())
        lora_params = None
        if args.rank is not None:
            apply_lora(_m, rank=args.rank)
            lora_params = get_lora_param_count(_m)
        del _m

        save_dict['config'] = {
            'n_steps': args.n_steps, 'rank': args.rank,
            'n_per_class': args.n_per_class, 'seed': args.seed,
            'lr': args.lr, 'mode': mode,
            'relu_alpha': args.relu_alpha,
            'optimizer': args.optimizer,
            'full_params': full_params,
            'lora_params': lora_params,
        }
        pth_path = os.path.join(RESULTS_DIR, f"{base_name}.pth")
        torch.save(save_dict, pth_path)
        print(f"\nSaved tensors to {pth_path}")

        # Save figure (.png)
        sprint1_dir = os.path.join(FIGURES_DIR, 'sprint1')
        generate_experiment_b_figure(results, save_dir=sprint1_dir)
        print(f"Saved figure to {sprint1_dir}/")
