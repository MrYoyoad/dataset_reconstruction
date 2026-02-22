"""Experiment A: Convergence + LoRA → Compose → Existing Reconstruction Pipeline.

Train FCN with LoRA to convergence on few-shot MNIST, compose W = W₀ + BA,
feed into the UNCHANGED existing reconstruction pipeline.

Usage:
    conda run -n rec python -m experiments.run_experiment_a --rank 8 --n_per_class 1
"""

import sys
import os
import argparse
import copy
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from CreateModel import NeuralNetwork, get_activation
from extraction import get_trainable_params, calc_extraction_loss, evaluate_extraction
from common_utils.common import load_weights

from experiments.configs import (
    INPUT_DIM, OUTPUT_DIM, MODEL_HIDDEN_LIST, MODEL_INIT_LIST,
    EXTRACTION_LR, EXTRACTION_LAMBDA_LR, EXTRACTION_INIT_SCALE,
    EXTRACTION_MIN_LAMBDA, EXTRACTION_RELU_ALPHA, EXTRACTION_EPOCHS,
    EXTRACTION_EVAL_EVERY, RESULTS_DIR,
)
from experiments.data_utils import (
    get_few_shot_mnist, get_control_images_in_distribution,
)
from experiments.train_lora import train_lora, train_full_finetune
from experiments.lora_wrapper import compose_state_dict, save_composed_weights, get_lora_param_count
from experiments.metrics import compute_all_metrics


def create_fresh_model(init_scale=None, device='cpu'):
    """Create a NeuralNetwork matching the MNIST architecture with proper init."""
    activation = nn.ReLU()
    model = NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=MODEL_HIDDEN_LIST,
        output_dim=OUTPUT_DIM,
        activation=activation,
        use_bias=False,
    )
    model = model.to(device)
    if init_scale is not None:
        model.layers[0].weight.data.normal_().mul_(init_scale)
        if model.layers[0].bias is not None:
            model.layers[0].bias.data.normal_().mul_(init_scale)
    torch.set_default_dtype(torch.float64)
    return model


def create_extraction_model(device='cpu'):
    """Create model with ModifiedReLU activation for the extraction phase."""
    from CreateModel import ModifiedRelu
    activation = ModifiedRelu(EXTRACTION_RELU_ALPHA)
    model = NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=MODEL_HIDDEN_LIST,
        output_dim=OUTPUT_DIM,
        activation=activation,
        use_bias=False,
    )
    return model.to(device)


def run_extraction(model, x0, y0, ds_mean, n_per_class,
                   extraction_epochs=EXTRACTION_EPOCHS, device='cpu'):
    """Run the KKT reconstruction, mirroring Main.py's data_extraction().

    Args:
        model: trained model loaded with weights (eval mode)
        x0: ground truth images [N, C, H, W] (for evaluation only)
        y0: ground truth labels [N] (for evaluation only)
        ds_mean: dataset mean tensor
        n_per_class: samples to reconstruct per class

    Returns:
        x_recon: reconstructed images tensor
        extraction_results: dict with loss history and final metrics
    """
    model.eval()

    # Create extraction labels: first half -1, second half +1 (matching Main.py)
    extraction_amount = n_per_class * 2
    y_extract = torch.zeros(extraction_amount, device=device)
    y_extract[:extraction_amount // 2] = -1
    y_extract[extraction_amount // 2:] = 1

    # Initialize x and lambda
    n, c, h, w = x0.shape
    x = torch.randn(extraction_amount, c, h, w, device=device) * EXTRACTION_INIT_SCALE
    x.requires_grad_(True)
    lam = torch.rand(extraction_amount, 1, device=device)
    lam.requires_grad_(True)

    opt_x = torch.optim.SGD([x], lr=EXTRACTION_LR, momentum=0.9)
    opt_l = torch.optim.SGD([lam], lr=EXTRACTION_LAMBDA_LR, momentum=0.9)

    # Create a namespace mimicking args for calc_extraction_loss
    class Args:
        extraction_loss_type = 'kkt'
        extraction_min_lambda = EXTRACTION_MIN_LAMBDA
        extraction_data_amount = extraction_amount
    args = Args()

    loss_history = []
    best_x = None
    best_score = float('inf')

    for epoch in range(extraction_epochs):
        values = model(x).squeeze()
        loss, kkt_loss, loss_verify = calc_extraction_loss(
            args, lam, model, values, x, y_extract
        )

        if torch.isnan(kkt_loss):
            print(f"NaN at epoch {epoch}, stopping extraction.")
            break

        opt_x.zero_grad()
        opt_l.zero_grad()
        loss.backward()
        opt_x.step()
        opt_l.step()

        loss_history.append(kkt_loss.item())

        if epoch % EXTRACTION_EVAL_EVERY == 0:
            with torch.no_grad():
                print(f"  Extraction epoch {epoch}: kkt={kkt_loss.item():.4e} "
                      f"verify={loss_verify.item():.4e}")

    x_recon = x.detach().clone()
    return x_recon, {'loss_history': loss_history}


def run_single_config(rank, n_per_class, seed=42, run_baseline=True,
                      init_scale=None, device='cpu', verbose=True):
    """Run Experiment A for one (rank, N) configuration.

    Returns dict with all results and metrics.
    """
    if init_scale is None:
        init_scale = MODEL_INIT_LIST[0]

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)

    # Load data
    x_train, y_train, digits, indices = get_few_shot_mnist(
        n_per_class, seed=seed, device=device
    )
    if verbose:
        print(f"Training digits: {digits}, indices: {indices}")
        print(f"x_train shape: {x_train.shape}, y_train: {y_train.tolist()}")

    results = {'rank': rank, 'n_per_class': n_per_class, 'seed': seed,
               'digits': digits, 'indices': indices}

    # --- LoRA training ---
    if verbose:
        print(f"\n--- Training LoRA rank={rank} ---")
    model_lora = create_fresh_model(init_scale=init_scale, device=device)
    # Save init state for reference
    init_sd = {k: v.clone() for k, v in model_lora.state_dict().items()}

    train_result = train_lora(
        model_lora, x_train.clone(), y_train.clone(), rank=rank,
        verbose=verbose, eval_every=10000,
    )
    results['lora_train'] = {
        'final_loss': train_result['final_loss'],
        'epochs': train_result['epochs_trained'],
        'converged': train_result['converged'],
        'lora_param_count': get_lora_param_count(model_lora),
    }
    ds_mean = train_result['ds_mean']

    # Compose and save
    composed_sd = compose_state_dict(model_lora)
    composed_path = os.path.join(
        RESULTS_DIR, f'lora_r{rank}_n{n_per_class}_s{seed}_composed.pth'
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_composed_weights(model_lora, composed_path)
    if verbose:
        print(f"Saved composed weights to {composed_path}")

    # Load into extraction model and reconstruct
    extraction_model = create_extraction_model(device=device)
    extraction_model.load_state_dict(composed_sd)
    extraction_model.eval()

    x_centered = x_train - ds_mean if ds_mean is not None else x_train
    x_recon_lora, extract_res = run_extraction(
        extraction_model, x_centered, y_train, ds_mean, n_per_class, device=device,
    )

    # Metrics against training data
    metrics_lora = compute_all_metrics(x_recon_lora, x_centered, ds_mean)
    results['lora_metrics'] = {k: v['mean'] for k, v in metrics_lora.items()}
    results['x_recon_lora'] = x_recon_lora
    if verbose:
        print(f"LoRA reconstruction: SSIM={metrics_lora['ssim']['mean']:.4f}, "
              f"DSSIM={metrics_lora['dssim']['mean']:.4f}")

    # --- Full fine-tuning baseline ---
    if run_baseline:
        if verbose:
            print(f"\n--- Training full fine-tuning baseline ---")
        model_full = create_fresh_model(init_scale=init_scale, device=device)
        # Use same init as LoRA model
        model_full.load_state_dict(init_sd)

        train_result_full = train_full_finetune(
            model_full, x_train.clone(), y_train.clone(),
            verbose=verbose, eval_every=10000,
        )
        results['full_ft_train'] = {
            'final_loss': train_result_full['final_loss'],
            'epochs': train_result_full['epochs_trained'],
            'converged': train_result_full['converged'],
        }

        extraction_model_full = create_extraction_model(device=device)
        extraction_model_full.load_state_dict(model_full.state_dict())
        extraction_model_full.eval()

        x_recon_full, _ = run_extraction(
            extraction_model_full, x_centered, y_train,
            train_result_full['ds_mean'], n_per_class, device=device,
        )

        metrics_full = compute_all_metrics(
            x_recon_full, x_centered, train_result_full['ds_mean']
        )
        results['full_ft_metrics'] = {k: v['mean'] for k, v in metrics_full.items()}
        results['x_recon_full'] = x_recon_full
        if verbose:
            print(f"Full FT reconstruction: SSIM={metrics_full['ssim']['mean']:.4f}, "
                  f"DSSIM={metrics_full['dssim']['mean']:.4f}")

    # --- Control images ---
    x_ctrl, y_ctrl, ctrl_digits = get_control_images_in_distribution(
        digits, device=device
    )
    x_ctrl_centered = x_ctrl - ds_mean if ds_mean is not None else x_ctrl
    metrics_ctrl = compute_all_metrics(x_recon_lora, x_ctrl_centered, ds_mean)
    results['control_metrics'] = {k: v['mean'] for k, v in metrics_ctrl.items()}
    if verbose:
        print(f"Control comparison: SSIM={metrics_ctrl['ssim']['mean']:.4f}, "
              f"DSSIM={metrics_ctrl['dssim']['mean']:.4f}")

    results['x_train'] = x_train
    results['x_ctrl'] = x_ctrl
    results['ds_mean'] = ds_mean

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--n_per_class', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_baseline', action='store_true')
    args = parser.parse_args()

    results = run_single_config(
        rank=args.rank,
        n_per_class=args.n_per_class,
        seed=args.seed,
        run_baseline=not args.no_baseline,
    )
    print("\n=== Final Results ===")
    print(f"LoRA (rank={args.rank}): {results['lora_metrics']}")
    if 'full_ft_metrics' in results:
        print(f"Full FT baseline: {results['full_ft_metrics']}")
    print(f"Control comparison: {results['control_metrics']}")
