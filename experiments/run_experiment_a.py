"""Experiment A: Convergence + LoRA → Compose → Existing Reconstruction Pipeline.

Attack scenario: pre-trained model (W₀) is fine-tuned with LoRA on private
held-out data to convergence. Compose W = W₀ + BA, feed into the UNCHANGED
existing KKT reconstruction pipeline.

Usage:
    conda run -n rec python -m experiments.run_experiment_a --rank 8 --n_per_class 1
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from CreateModel import NeuralNetwork, ModifiedRelu
from extraction import calc_extraction_loss

from experiments.configs import (
    INPUT_DIM, OUTPUT_DIM, MODEL_HIDDEN_LIST,
    EXTRACTION_LR, EXTRACTION_LAMBDA_LR, EXTRACTION_INIT_SCALE,
    EXTRACTION_MIN_LAMBDA, EXTRACTION_RELU_ALPHA, EXTRACTION_EPOCHS,
    EXTRACTION_EVAL_EVERY, RESULTS_DIR, PRETRAINED_MNIST_PATH,
    TRAIN_LR, TRAIN_EPOCHS,
)
from experiments.data_utils import (
    get_finetuning_data, get_control_images_in_distribution,
)
from experiments.train_lora import train_lora, train_full_finetune
from experiments.lora_wrapper import compose_state_dict, save_composed_weights, get_lora_param_count
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
    """Load the pre-trained MNIST odd/even model as W₀."""
    path = pretrained_path or PRETRAINED_MNIST_PATH
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = create_model(device=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def create_extraction_model(device='cpu'):
    """Create model with ModifiedReLU activation for the extraction phase."""
    activation = ModifiedRelu(EXTRACTION_RELU_ALPHA)
    model = NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=MODEL_HIDDEN_LIST,
        output_dim=OUTPUT_DIM,
        activation=activation,
        use_bias=False,
    )
    return model.to(device).double()


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
                      pretrained_path=None, extraction_epochs=EXTRACTION_EPOCHS,
                      extraction_n_per_class=None, fine_tune_lr=TRAIN_LR,
                      fine_tune_epochs=TRAIN_EPOCHS,
                      device='cpu', verbose=True):
    """Run Experiment A for one (rank, N) configuration.

    Loads pre-trained weights as W₀, fine-tunes (LoRA or full) on held-out
    MNIST test data to convergence, then reconstructs via KKT.

    Args:
        extraction_n_per_class: N for extraction (default: same as n_per_class).
            Set differently to test wrong-N hypothesis.
        fine_tune_lr: learning rate for fine-tuning (default: TRAIN_LR=0.01).
        fine_tune_epochs: max epochs for fine-tuning (default: TRAIN_EPOCHS=1M).

    Returns dict with all results and metrics.
    """
    if extraction_n_per_class is None:
        extraction_n_per_class = n_per_class

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)

    # Load held-out fine-tuning data (MNIST TEST set)
    x_ft, y_ft, digits, indices = get_finetuning_data(
        n_per_class, seed=seed, device=device
    )
    if verbose:
        print(f"Fine-tuning digits: {digits}, indices: {indices}")
        print(f"x_ft shape: {x_ft.shape}, y_ft: {y_ft.tolist()}")

    results = {'rank': rank, 'n_per_class': n_per_class,
               'extraction_n_per_class': extraction_n_per_class,
               'fine_tune_lr': fine_tune_lr, 'fine_tune_epochs': fine_tune_epochs,
               'seed': seed, 'digits': digits, 'indices': indices}

    # --- LoRA training ---
    if verbose:
        print(f"\n--- Training LoRA rank={rank} from pre-trained weights ---")
    model_lora = load_pretrained(device=device, pretrained_path=pretrained_path)
    # Save init state for reference
    init_sd = {k: v.clone() for k, v in model_lora.state_dict().items()}

    train_result = train_lora(
        model_lora, x_ft.clone(), y_ft.clone(), rank=rank,
        lr=fine_tune_lr, epochs=fine_tune_epochs,
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

    x_centered = x_ft - ds_mean if ds_mean is not None else x_ft
    x_recon_lora, extract_res = run_extraction(
        extraction_model, x_centered, y_ft, ds_mean, extraction_n_per_class,
        extraction_epochs=extraction_epochs, device=device,
    )

    # Metrics against fine-tuning data (only when shapes match)
    results['x_recon_lora'] = x_recon_lora
    results['lora_final_kkt_loss'] = extract_res['loss_history'][-1] if extract_res['loss_history'] else float('inf')
    if x_recon_lora.shape[0] == x_centered.shape[0]:
        metrics_lora = compute_all_metrics(x_recon_lora, x_centered, ds_mean)
        results['lora_metrics'] = {k: v['mean'] for k, v in metrics_lora.items()}
        if verbose:
            print(f"LoRA reconstruction: SSIM={metrics_lora['ssim']['mean']:.4f}, "
                  f"DSSIM={metrics_lora['dssim']['mean']:.4f}")
    else:
        if verbose:
            print(f"LoRA extraction N={extraction_n_per_class} != fine-tune N={n_per_class}, "
                  f"skipping SSIM. Final KKT loss={results['lora_final_kkt_loss']:.4e}")

    # --- Full fine-tuning baseline ---
    if run_baseline:
        if verbose:
            print(f"\n--- Training full fine-tuning baseline ---")
        model_full = load_pretrained(device=device, pretrained_path=pretrained_path)

        train_result_full = train_full_finetune(
            model_full, x_ft.clone(), y_ft.clone(),
            lr=fine_tune_lr, epochs=fine_tune_epochs,
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

        x_recon_full, extract_res_full = run_extraction(
            extraction_model_full, x_centered, y_ft,
            train_result_full['ds_mean'], extraction_n_per_class,
            extraction_epochs=extraction_epochs, device=device,
        )

        results['x_recon_full'] = x_recon_full
        results['full_final_kkt_loss'] = extract_res_full['loss_history'][-1] if extract_res_full['loss_history'] else float('inf')
        if x_recon_full.shape[0] == x_centered.shape[0]:
            metrics_full = compute_all_metrics(
                x_recon_full, x_centered, train_result_full['ds_mean']
            )
            results['full_ft_metrics'] = {k: v['mean'] for k, v in metrics_full.items()}
            if verbose:
                print(f"Full FT reconstruction: SSIM={metrics_full['ssim']['mean']:.4f}, "
                      f"DSSIM={metrics_full['dssim']['mean']:.4f}")
        else:
            if verbose:
                print(f"Full FT extraction N={extraction_n_per_class} != fine-tune N={n_per_class}, "
                      f"skipping SSIM. Final KKT loss={results['full_final_kkt_loss']:.4e}")

    # --- Control images (only when extraction N matches fine-tuning N) ---
    if extraction_n_per_class == n_per_class:
        x_ctrl, y_ctrl, ctrl_digits = get_control_images_in_distribution(
            digits, device=device
        )
        x_ctrl_centered = x_ctrl - ds_mean if ds_mean is not None else x_ctrl
        metrics_ctrl = compute_all_metrics(x_recon_lora, x_ctrl_centered, ds_mean)
        results['control_metrics'] = {k: v['mean'] for k, v in metrics_ctrl.items()}
        results['x_ctrl'] = x_ctrl
        if verbose:
            print(f"Control comparison: SSIM={metrics_ctrl['ssim']['mean']:.4f}, "
                  f"DSSIM={metrics_ctrl['dssim']['mean']:.4f}")

    results['x_train'] = x_ft
    results['ds_mean'] = ds_mean

    return results


if __name__ == '__main__':
    from experiments.configs import get_device

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--n_per_class', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_baseline', action='store_true')
    parser.add_argument('--extraction_epochs', type=int, default=EXTRACTION_EPOCHS)
    parser.add_argument('--extraction_n_per_class', type=int, default=None,
                        help='N for extraction (default: same as n_per_class)')
    parser.add_argument('--fine_tune_lr', type=float, default=TRAIN_LR,
                        help='Learning rate for fine-tuning (default: 0.01)')
    parser.add_argument('--fine_tune_epochs', type=int, default=TRAIN_EPOCHS,
                        help='Max epochs for fine-tuning (default: 1M)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Using device: {device}")

    results = run_single_config(
        rank=args.rank,
        n_per_class=args.n_per_class,
        seed=args.seed,
        run_baseline=not args.no_baseline,
        extraction_epochs=args.extraction_epochs,
        extraction_n_per_class=args.extraction_n_per_class,
        fine_tune_lr=args.fine_tune_lr,
        fine_tune_epochs=args.fine_tune_epochs,
        device=device,
    )
    print("\n=== Final Results ===")
    if 'lora_metrics' in results:
        print(f"LoRA (rank={args.rank}): {results['lora_metrics']}")
    if 'lora_final_kkt_loss' in results:
        print(f"LoRA final KKT loss: {results['lora_final_kkt_loss']:.4e}")
    if 'full_ft_metrics' in results:
        print(f"Full FT baseline: {results['full_ft_metrics']}")
    if 'full_final_kkt_loss' in results:
        print(f"Full FT final KKT loss: {results['full_final_kkt_loss']:.4e}")
    if 'control_metrics' in results:
        print(f"Control comparison: {results['control_metrics']}")
