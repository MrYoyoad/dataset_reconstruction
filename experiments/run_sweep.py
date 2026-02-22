"""Sweep driver for both experiments across rank, N, and step count.

Experiment A sweep: rank × N (convergence regime)
Experiment B sweep: rank × T (NTK regime, multi-step)

Usage:
    conda run -n rec python -m experiments.run_sweep --experiment a
    conda run -n rec python -m experiments.run_sweep --experiment b
"""

import sys
import os
import argparse
import csv
import json
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from experiments.configs import (
    RANK_SWEEP, N_PER_CLASS_SWEEP, STEP_SWEEP, RANK_SWEEP_EXP_B,
    RESULTS_DIR, EXTRACTION_EPOCHS,
)


def run_experiment_a_sweep(ranks=None, ns=None, seed=42, device='cpu',
                           extraction_epochs=EXTRACTION_EPOCHS):
    """Run Experiment A across rank × N grid."""
    from experiments.run_experiment_a import run_single_config

    ranks = ranks or RANK_SWEEP
    ns = ns or N_PER_CLASS_SWEEP

    results_list = []
    csv_path = os.path.join(RESULTS_DIR, f'experiment_a_sweep_{datetime.now():%Y%m%d_%H%M%S}.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Experiment A Sweep: {len(ranks)} ranks × {len(ns)} N values ===")

    for n in ns:
        for rank in ranks:
            print(f"\n--- rank={rank}, N={n} ---")
            try:
                results = run_single_config(
                    rank=rank, n_per_class=n, seed=seed,
                    run_baseline=(rank == ranks[0]),  # baseline only once per N
                    device=device, verbose=True,
                )

                row = {
                    'rank': rank,
                    'n_per_class': n,
                    'seed': seed,
                    'lora_ssim': results['lora_metrics'].get('ssim', None),
                    'lora_dssim': results['lora_metrics'].get('dssim', None),
                    'lora_ncc': results['lora_metrics'].get('ncc', None),
                    'lora_l2': results['lora_metrics'].get('l2', None),
                    'control_ssim': results['control_metrics'].get('ssim', None),
                    'control_dssim': results['control_metrics'].get('dssim', None),
                    'lora_converged': results['lora_train']['converged'],
                    'lora_final_loss': results['lora_train']['final_loss'],
                    'lora_epochs': results['lora_train']['epochs'],
                }

                if 'full_ft_metrics' in results:
                    row['full_ft_ssim'] = results['full_ft_metrics'].get('ssim', None)
                    row['full_ft_dssim'] = results['full_ft_metrics'].get('dssim', None)

                results_list.append(row)
                print(f"  SSIM={row['lora_ssim']:.4f}, DSSIM={row['lora_dssim']:.4f}")

            except Exception as e:
                print(f"  FAILED: {e}")
                results_list.append({
                    'rank': rank, 'n_per_class': n, 'seed': seed, 'error': str(e)
                })

    # Save CSV
    if results_list:
        keys = results_list[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\nResults saved to {csv_path}")

    return results_list


def run_experiment_b_sweep(ranks=None, steps=None, seed=42, device='cpu',
                           extraction_epochs=EXTRACTION_EPOCHS):
    """Run Experiment B across rank × step count grid."""
    from experiments.run_experiment_b import run_single_config

    ranks = ranks or RANK_SWEEP_EXP_B
    steps = steps or STEP_SWEEP

    results_list = []
    csv_path = os.path.join(RESULTS_DIR, f'experiment_b_sweep_{datetime.now():%Y%m%d_%H%M%S}.csv')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"=== Experiment B Sweep: {len(ranks)} ranks × {len(steps)} step counts ===")

    for n_steps in steps:
        for rank in [None] + ranks:  # None = full model baseline
            label = f"full" if rank is None else f"rank={rank}"
            print(f"\n--- T={n_steps}, {label} ---")
            try:
                results = run_single_config(
                    n_steps=n_steps, rank=rank, n_per_class=1, seed=seed,
                    run_baseline=(rank is None or rank == ranks[0]),
                    extraction_epochs=extraction_epochs,
                    device=device, verbose=True,
                )

                row = {
                    'n_steps': n_steps,
                    'rank': rank if rank is not None else 'full',
                    'seed': seed,
                }

                if 'full_metrics' in results:
                    row['full_ssim'] = results['full_metrics'].get('ssim', None)
                    row['full_dssim'] = results['full_metrics'].get('dssim', None)

                if 'lora_metrics' in results:
                    row['lora_ssim'] = results['lora_metrics'].get('ssim', None)
                    row['lora_dssim'] = results['lora_metrics'].get('dssim', None)

                if 'full_diagnostics' in results:
                    row['weight_change'] = results['full_diagnostics'].get('weight_change', None)
                    row['feature_stability'] = results['full_diagnostics'].get('feature_stability', None)
                    row['coefficient_drift'] = results['full_diagnostics'].get('coefficient_drift', None)

                if 'control_metrics' in results:
                    row['control_ssim'] = results['control_metrics'].get('ssim', None)
                    row['control_dssim'] = results['control_metrics'].get('dssim', None)

                results_list.append(row)

            except Exception as e:
                print(f"  FAILED: {e}")
                results_list.append({
                    'n_steps': n_steps, 'rank': rank, 'seed': seed, 'error': str(e)
                })

    # Save CSV
    if results_list:
        all_keys = set()
        for r in results_list:
            all_keys.update(r.keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(results_list)
        print(f"\nResults saved to {csv_path}")

    return results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['a', 'b', 'both'], default='both')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extraction_epochs', type=int, default=EXTRACTION_EPOCHS)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    if args.experiment in ('a', 'both'):
        run_experiment_a_sweep(seed=args.seed, extraction_epochs=args.extraction_epochs)

    if args.experiment in ('b', 'both'):
        run_experiment_b_sweep(seed=args.seed, extraction_epochs=args.extraction_epochs)
