"""Anchor α-sweep orchestrator (meeting Addition 3).

For each α in the sweep, run Experiment B with the linearization anchor
θ_anchor = (1-α)·θ₀ + α·θ_T on BOTH the full-FT and LoRA paths, collect
SSIM(α) and linearization-error(α) (function-space headline + weight-space
companion), save per-α tensors + reconstruction grids, and emit the two-curve
validation plot per path.

The scientific read (experiment_plan.md "Addition 3"): a legitimate win is when
lin-error bottoms out where SSIM peaks (improvement explained by better
linearization). A red flag is SSIM continuing to climb past the lin-error
minimum — the anchor is leaking x_i, not approximating better.

Usage (WEXAC):
    # full sweep, both paths (~50 GPU-min at default extraction_epochs)
    python -m experiments.run_anchor_sweep --device cuda --n_steps 10 --rank 8 \
        --finetune_activation gelu --save
    # fast smoke (few epochs, 2 alphas) — validates the pipeline end to end
    python -m experiments.run_anchor_sweep --device cuda --n_steps 5 --rank 8 \
        --alphas 0 0.5 --extraction_epochs 300 --finetune_activation gelu
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

import torch

torch.set_default_dtype(torch.float64)

from experiments.configs import (
    RESULTS_DIR, FIGURES_DIR, EXTRACTION_EPOCHS, ANCHOR_ALPHA_SWEEP,
)


def run_alpha_sweep(alphas, n_steps, rank, n_per_class, seed,
                    finetune_activation, extraction_epochs, device, verbose=True,
                    verify_weight=1.0):
    """Run Experiment B once per α (both full-FT and LoRA paths). Returns {α: results}."""
    from experiments.run_experiment_b import run_single_config

    all_results = {}
    for i, a in enumerate(alphas):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(alphas)}] anchor_alpha={a}, T={n_steps}, rank={rank}, "
              f"act={finetune_activation}, verify_weight={verify_weight}")
        print(f"{'='*60}")
        res = run_single_config(
            n_steps=n_steps, rank=rank, n_per_class=n_per_class, seed=seed,
            run_baseline=True, anchor_alpha=a,
            finetune_activation=finetune_activation,
            extraction_epochs=extraction_epochs,
            verify_weight=verify_weight,
            optimizer_type='lbfgs', device=device, verbose=verbose,
        )
        all_results[a] = res
        fm, lm = res.get('full_metrics', {}), res.get('lora_metrics', {})
        print(f"  α={a}: full SSIM={fm.get('ssim')}, LoRA SSIM={lm.get('ssim')} | "
              f"lin_fs full={res.get('full_lin_error_fs')}, "
              f"lin_fs lora={res.get('lora_lin_error_fs')}")
    return all_results


def collect_curves(all_results, alphas):
    """Extract per-α SSIM + linearization-error curves for both paths."""
    def metric(res, key, sub=None):
        v = res.get(key)
        if sub is not None and isinstance(v, dict):
            return v.get(sub)
        return v

    return {
        'alphas': list(alphas),
        'full_ssim': [metric(all_results[a], 'full_metrics', 'ssim') for a in alphas],
        'lora_ssim': [metric(all_results[a], 'lora_metrics', 'ssim') for a in alphas],
        'full_lin_fs': [metric(all_results[a], 'full_lin_error_fs') for a in alphas],
        'lora_lin_fs': [metric(all_results[a], 'lora_lin_error_fs') for a in alphas],
        'full_lin_ws': [metric(all_results[a], 'full_linearization_error') for a in alphas],
        'lora_lin_ws': [metric(all_results[a], 'lora_linearization_error') for a in alphas],
    }


def save_sweep(all_results, curves, save_path):
    """Persist all image tensors + metrics + curves (CLAUDE.md: save visuals, not just numbers)."""
    save_dict = {'curves': curves, 'per_alpha': {}}
    for a, res in all_results.items():
        entry = {
            'anchor_alpha': a,
            'x_train': res['x_train'].cpu(),
            'ds_mean': res.get('ds_mean', torch.zeros(1)).cpu(),
            'digits': res.get('digits', []),
        }
        for k in ['x_recon_full', 'x_recon_lora', 'x_ctrl',
                  'full_metrics', 'lora_metrics', 'control_metrics',
                  'full_lin_error_fs', 'lora_lin_error_fs',
                  'full_linearization_error', 'lora_linearization_error']:
            if k in res:
                v = res[k]
                entry[k] = v.cpu() if torch.is_tensor(v) else v
        save_dict['per_alpha'][a] = entry
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Saved sweep tensors to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphas', type=float, nargs='+', default=None,
                        help='anchor alphas (default: configs.ANCHOR_ALPHA_SWEEP)')
    parser.add_argument('--n_steps', type=int, default=10,
                        help='fine-tuning steps T (anchor matters most for multi-step)')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--n_per_class', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetune_activation', type=str, default='gelu')
    parser.add_argument('--extraction_epochs', type=int, default=EXTRACTION_EPOCHS)
    parser.add_argument('--verify_weight', type=float, default=1.0,
                        help='soft box-constraint weight passed to run_single_config '
                             '(raise to reduce [0,1] pixel saturation; default 1.0 = baseline)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save', action='store_true',
                        help='save per-α tensors + grids and the aggregate sweep .pth')
    parser.add_argument('--tag', type=str, default=None, help='filename tag override')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    alphas = args.alphas if args.alphas is not None else list(ANCHOR_ALPHA_SWEEP)
    print(f"Anchor alphas: {alphas}")

    all_results = run_alpha_sweep(
        alphas, args.n_steps, args.rank, args.n_per_class, args.seed,
        args.finetune_activation, args.extraction_epochs, device,
        verify_weight=args.verify_weight)

    curves = collect_curves(all_results, alphas)
    print("\n=== Anchor sweep curves (α: SSIM / lin-error function-space) ===")
    for j, a in enumerate(alphas):
        print(f"  α={a}: full SSIM={curves['full_ssim'][j]} lin_fs={curves['full_lin_fs'][j]} | "
              f"LoRA SSIM={curves['lora_ssim'][j]} lin_fs={curves['lora_lin_fs'][j]}")

    from experiments.plotting import generate_experiment_b_figure, plot_anchor_two_curve
    tag = args.tag or f"T{args.n_steps}_r{args.rank}_{args.finetune_activation}_s{args.seed}"
    fig_dir = os.path.join(FIGURES_DIR, 'anchor_sweep')

    if args.save:
        for a, res in all_results.items():
            generate_experiment_b_figure(res, save_dir=fig_dir,
                                         base_name=f"anchor_{tag}_a{a:g}")
        save_sweep(all_results, curves,
                   os.path.join(RESULTS_DIR, f"anchor_sweep_{tag}.pth"))

    # Two-curve validation plots (always emitted — the headline deliverable)
    os.makedirs(fig_dir, exist_ok=True)
    if any(v is not None for v in curves['full_ssim']):
        plot_anchor_two_curve(
            alphas, curves['full_ssim'], curves['full_lin_fs'], curves['full_lin_ws'],
            save_path=os.path.join(fig_dir, f"anchor_two_curve_full_{tag}.png"),
            path_label='Full-FT')
    if any(v is not None for v in curves['lora_ssim']):
        plot_anchor_two_curve(
            alphas, curves['lora_ssim'], curves['lora_lin_fs'], curves['lora_lin_ws'],
            save_path=os.path.join(fig_dir, f"anchor_two_curve_lora_r{args.rank}_{tag}.png"),
            path_label=f'LoRA r={args.rank}')

    print("\n=== Done ===")
