"""Experiment B: Multi-Step NTK Reconstruction from Pre-Trained Weights.

Attack scenario: pre-trained model (θ₀) is fine-tuned on private data
for T gradient steps. Attacker observes ΔW = θ_T - θ₀ and reconstructs
the private fine-tuning data using the NTK loss with known coefficients.

Usage:
    conda run -n rec python -m experiments.run_experiment_b --n_steps 1 --rank 8 --n_per_class 1
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from CreateModel import NeuralNetwork

from experiments.configs import (
    INPUT_DIM, OUTPUT_DIM, MODEL_HIDDEN_LIST,
    EXTRACTION_EPOCHS, EXTRACTION_EVAL_EVERY, RESULTS_DIR, TRAIN_LR,
    PRETRAINED_MNIST_PATH,
)
from experiments.data_utils import (
    get_finetuning_data, get_control_images_in_distribution,
)
from experiments.ntk_steps import compute_multi_step_update, compute_multi_step_update_lora
from experiments.ntk_extraction import run_ntk_extraction
from experiments.ntk_verification import verify_ntk_at_step
from experiments.metrics import compute_all_metrics


def create_model(device='cpu'):
    """Create a NeuralNetwork matching the MNIST architecture (ReLU, no bias)."""
    model = NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=MODEL_HIDDEN_LIST,
        output_dim=OUTPUT_DIM,
        activation=nn.ReLU(),
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


def run_single_config(n_steps, rank=None, n_per_class=1, seed=42,
                      lr=TRAIN_LR, pretrained_path=None,
                      extraction_epochs=EXTRACTION_EPOCHS,
                      run_baseline=True,
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
    if verbose:
        print(f"Fine-tuning digits: {digits}, indices: {indices}")
        print(f"n_steps={n_steps}, rank={rank}, lr={lr}")

    results = {'n_steps': n_steps, 'rank': rank, 'n_per_class': n_per_class,
               'seed': seed, 'digits': digits}

    # --- Full model (no LoRA) baseline ---
    if run_baseline:
        if verbose:
            print(f"\n--- Full model fine-tuning, T={n_steps} steps ---")

        model_full = load_pretrained(device=device, pretrained_path=pretrained_path)
        update_result = compute_multi_step_update(
            model_full, x_ft.clone(), y_ft.clone(), lr=lr, n_steps=n_steps,
        )

        # Create model at θ₀ for NTK features
        model_theta0 = load_pretrained(device=device, pretrained_path=pretrained_path)
        model_theta0.eval()

        # NTK verification
        def _make_model():
            return create_model(device=device)

        x_centered = x_ft - update_result['ds_mean'] if update_result['ds_mean'] is not None else x_ft

        diagnostics = verify_ntk_at_step(
            update_result['theta_0'], update_result['theta_T'],
            _make_model, x_centered, y_ft,
        )
        results['full_diagnostics'] = {
            'weight_change': diagnostics['weight_change']['overall'],
            'feature_stability': diagnostics['feature_stability']['mean'],
            'coefficient_drift': diagnostics['coefficient_drift']['mean_abs_drift'],
        }
        if verbose:
            d = results['full_diagnostics']
            print(f"  NTK diag: weight_change={d['weight_change']:.6f}, "
                  f"feature_cos={d['feature_stability']:.6f}, "
                  f"coeff_drift={d['coefficient_drift']:.6f}")

        # NTK reconstruction
        x_recon_full, extract_res = run_ntk_extraction(
            model_theta0, update_result['delta_w'],
            update_result['coefficients_at_init'],
            lr_train=lr, n_steps=n_steps, n_per_class=n_per_class,
            extraction_epochs=extraction_epochs, device=device, verbose=verbose,
        )

        metrics_full = compute_all_metrics(x_recon_full, x_centered, update_result['ds_mean'])
        results['full_metrics'] = {k: v['mean'] for k, v in metrics_full.items()}
        results['x_recon_full'] = x_recon_full
        if verbose:
            print(f"  Full model reconstruction: SSIM={metrics_full['ssim']['mean']:.4f}")

    # --- LoRA version ---
    if rank is not None:
        if verbose:
            print(f"\n--- LoRA rank={rank}, T={n_steps} steps ---")

        model_lora = load_pretrained(device=device, pretrained_path=pretrained_path)
        update_result_lora = compute_multi_step_update_lora(
            model_lora, x_ft.clone(), y_ft.clone(), lr=lr,
            n_steps=n_steps, rank=rank,
        )

        # Model at θ₀ for NTK features
        model_theta0_lora = load_pretrained(device=device, pretrained_path=pretrained_path)
        model_theta0_lora.eval()

        x_centered = x_ft - update_result_lora['ds_mean'] if update_result_lora['ds_mean'] is not None else x_ft

        # NTK reconstruction
        x_recon_lora, extract_res_lora = run_ntk_extraction(
            model_theta0_lora, update_result_lora['delta_w'],
            update_result_lora['coefficients_at_init'],
            lr_train=lr, n_steps=n_steps, n_per_class=n_per_class,
            extraction_epochs=extraction_epochs, device=device, verbose=verbose,
        )

        metrics_lora = compute_all_metrics(x_recon_lora, x_centered, update_result_lora['ds_mean'])
        results['lora_metrics'] = {k: v['mean'] for k, v in metrics_lora.items()}
        results['x_recon_lora'] = x_recon_lora
        if verbose:
            print(f"  LoRA reconstruction: SSIM={metrics_lora['ssim']['mean']:.4f}")

    # --- Control images ---
    ds_mean = update_result['ds_mean'] if run_baseline else update_result_lora['ds_mean']
    x_ctrl, y_ctrl, ctrl_digits = get_control_images_in_distribution(digits, device=device)
    x_ctrl_centered = x_ctrl - ds_mean if ds_mean is not None else x_ctrl

    recon_for_ctrl = results.get('x_recon_full', results.get('x_recon_lora'))
    if recon_for_ctrl is not None:
        metrics_ctrl = compute_all_metrics(recon_for_ctrl, x_ctrl_centered, ds_mean)
        results['control_metrics'] = {k: v['mean'] for k, v in metrics_ctrl.items()}

    results['x_train'] = x_ft
    results['x_ctrl'] = x_ctrl
    results['ds_mean'] = ds_mean

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=1)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--n_per_class', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=TRAIN_LR)
    parser.add_argument('--no_baseline', action='store_true')
    args = parser.parse_args()

    results = run_single_config(
        n_steps=args.n_steps,
        rank=args.rank,
        n_per_class=args.n_per_class,
        seed=args.seed,
        lr=args.lr,
        run_baseline=not args.no_baseline,
    )

    print("\n=== Final Results ===")
    if 'full_metrics' in results:
        print(f"Full model (T={args.n_steps}): {results['full_metrics']}")
    if 'lora_metrics' in results:
        print(f"LoRA rank={args.rank} (T={args.n_steps}): {results['lora_metrics']}")
    if 'full_diagnostics' in results:
        print(f"NTK diagnostics: {results['full_diagnostics']}")
    if 'control_metrics' in results:
        print(f"Control: {results['control_metrics']}")
