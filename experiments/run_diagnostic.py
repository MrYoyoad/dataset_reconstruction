"""Diagnostic experiments: targeted follow-ups to Sprint 2c findings.

1. N=6 with 500K epochs — does B5 failure come from optimization or information?
2. B6 multi-seed — validate AdamW T=10 LoRA result across 5 seeds.
3. seed_fix_ablation — ablate sign-aware init, sign constraint, min_coeff threshold.

Usage (on WEXAC):
    python -u -m experiments.run_diagnostic --device cuda
    python -u -m experiments.run_diagnostic --task n6_diagnostic --device cuda
    python -u -m experiments.run_diagnostic --task b6_seeds --device cuda
    python -u -m experiments.run_diagnostic --task seed_fix_ablation --device cuda
"""

import sys
import os
import argparse
import csv
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

torch.set_default_dtype(torch.float64)

from experiments.configs import RESULTS_DIR, get_device
from experiments.run_experiment_b import run_single_config as run_exp_b


def save_csv(results_list, csv_path):
    """Save results list to CSV."""
    if not results_list:
        return
    all_keys = set()
    for r in results_list:
        all_keys.update(r.keys())
    keys = sorted(all_keys)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results_list:
            writer.writerow({k: r.get(k, '') for k in keys})
    print(f"Saved {len(results_list)} rows to {csv_path}")


def extract_row(res, extra):
    """Extract flat row from Experiment B results."""
    row = {
        'n_steps': res.get('n_steps'),
        'rank': res.get('rank', 'full'),
        'seed': res.get('seed'),
        'n_per_class': res.get('n_per_class'),
        'activation': res.get('finetune_activation') or 'relu',
    }
    if 'full_metrics' in res:
        row['full_ssim'] = res['full_metrics'].get('ssim')
    if 'lora_metrics' in res:
        row['lora_ssim'] = res['lora_metrics'].get('ssim')
    if 'control_metrics' in res:
        row['control_ssim'] = res['control_metrics'].get('ssim')
    if 'full_extract_res' in res and 'coeff_error' in res['full_extract_res']:
        row['full_coeff_error'] = res['full_extract_res']['coeff_error']
    if 'lora_extract_res' in res and 'coeff_error' in res['lora_extract_res']:
        row['lora_coeff_error'] = res['lora_extract_res']['coeff_error']
    row.update(extra)
    return row


def run_n6_diagnostic(device, seed=42):
    """N=6 with 500K extraction epochs.

    B5 showed SSIM collapses at N=6 (0.370 full, 0.353 LoRA r=8) with 50K epochs.
    This diagnostic tests whether 10x more optimization helps, answering:
    - If 500K epochs don't help → information-theoretic limit (ΔW from N=6 is too diluted)
    - If 500K epochs help significantly → optimization bottleneck (attack works but needs compute)
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: N=6 with 500K extraction epochs")
    print("=" * 70)

    configs = [
        # Baseline: N=2 at 50K (should match B5 result)
        {'n_per_class': 1, 'n_steps': 1, 'extraction_epochs': 50_000, 'label': 'N=2 50K (baseline)'},
        # N=6 at 50K (should match B5 result)
        {'n_per_class': 3, 'n_steps': 1, 'extraction_epochs': 50_000, 'label': 'N=6 50K (B5 repro)'},
        # N=6 at 500K (the actual diagnostic)
        {'n_per_class': 3, 'n_steps': 1, 'extraction_epochs': 500_000, 'label': 'N=6 500K (diagnostic)'},
        # Also test T=10 with inv_T (more realistic)
        {'n_per_class': 3, 'n_steps': 10, 'extraction_epochs': 500_000, 'label': 'N=6 T=10 500K'},
    ]

    results_list = []

    for i, cfg in enumerate(configs, 1):
        for rank in [None, 8]:
            rank_label = "full" if rank is None else f"r={rank}"
            print(f"\n[{i}/{len(configs)}] {cfg['label']}, {rank_label}")
            lr_schedule = 'inv_T' if cfg['n_steps'] > 1 else 'constant'
            try:
                res = run_exp_b(
                    n_steps=cfg['n_steps'], rank=rank,
                    n_per_class=cfg['n_per_class'], seed=seed,
                    run_baseline=(rank is None),
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type='sgd',
                    finetune_activation='leaky_relu',
                    extraction_epochs=cfg['extraction_epochs'],
                    lr_schedule=lr_schedule,
                    device=device, verbose=True,
                )
                row = extract_row(res, {
                    'task': 'n6_diagnostic',
                    'extraction_epochs': cfg['extraction_epochs'],
                    'lr_schedule': lr_schedule,
                    'label': cfg['label'],
                })
                results_list.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_per_class': cfg['n_per_class'],
                    'n_steps': cfg['n_steps'], 'rank': rank or 'full',
                    'extraction_epochs': cfg['extraction_epochs'],
                    'task': 'n6_diagnostic', 'error': str(e),
                })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'diagnostic_n6_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_b6_multi_seed(device, seeds=None):
    """B6 AdamW probe with multiple seeds.

    Original B6 (seed=42) showed:
    - SGD T=1: full=1.000, LoRA=0.830
    - AdamW T=1: full=0.461, LoRA=0.540
    - AdamW T=10: full=0.605, LoRA=0.785  ← surprisingly good!

    The T=10 LoRA+AdamW result (0.785) is close to SGD (0.780).
    Need multiple seeds to know if this is robust or a lucky seed.
    """
    if seeds is None:
        seeds = [42, 7, 123, 999, 2026]

    print("\n" + "=" * 70)
    print(f"B6 MULTI-SEED: AdamW Fine-Tuning Probe ({len(seeds)} seeds)")
    print(f"Seeds: {seeds}")
    print("=" * 70)
    finetune_optimizers = ['sgd', 'adamw']
    steps = [1, 10]
    ranks = [None, 8]
    lr_by_optimizer = {'sgd': 0.01, 'adamw': 1e-3}

    results_list = []
    total = len(seeds) * len(finetune_optimizers) * len(steps) * len(ranks)
    i = 0

    for seed in seeds:
        for ft_opt in finetune_optimizers:
            for T in steps:
                for rank in ranks:
                    i += 1
                    label = "full" if rank is None else f"r={rank}"
                    ft_lr = lr_by_optimizer[ft_opt]
                    lr_schedule = 'inv_T' if T > 1 else 'constant'
                    print(f"\n[{i}/{total}] seed={seed}, {ft_opt}, T={T}, "
                          f"{label}, lr={ft_lr}")
                    try:
                        res = run_exp_b(
                            n_steps=T, rank=rank, n_per_class=1, seed=seed,
                            lr=ft_lr,
                            run_baseline=(rank is None),
                            free_coefficients=True,
                            coeff_consistency_weight=1.0,
                            optimizer_type='sgd',
                            finetune_activation='leaky_relu',
                            finetune_optimizer=ft_opt,
                            lr_schedule=lr_schedule,
                            extraction_epochs=50_000,
                            device=device, verbose=True,
                        )
                        row = extract_row(res, {
                            'task': 'b6_seeds',
                            'finetune_optimizer': ft_opt,
                            'finetune_lr': ft_lr,
                            'lr_schedule': lr_schedule,
                        })
                        results_list.append(row)
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        import traceback
                        traceback.print_exc()
                        results_list.append({
                            'seed': seed, 'n_steps': T,
                            'rank': rank or 'full',
                            'finetune_optimizer': ft_opt,
                            'task': 'b6_seeds', 'error': str(e),
                        })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'diagnostic_b6_seeds_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_seed_fix_ablation(device, seeds=None):
    """Ablation: sign-aware init, sign constraint, min_coeff threshold.

    Two ablation tables:
    A. Feature ablation — which fix matters most?
    B. min_coeff threshold sweep — how aggressive should the constraint be?

    Runs 15 seeds × 2 models (full + LoRA r=8) per config.
    See plan for rationale and predictions.
    """
    if seeds is None:
        seeds = [42, 7, 123, 999, 2026, 1, 13, 37, 77, 256, 314, 500, 777, 1234, 2025]

    # Ablation A: feature ablation (which fix matters most?)
    ablation_a = [
        {'name': 'baseline',               'coeff_init': 'zeros',      'coeff_sign_weight': 0.0, 'min_coeff': 0.0,  'loss_type': 'l2'},
        {'name': 'sign_init_only',          'coeff_init': 'sign_aware', 'coeff_sign_weight': 0.0, 'min_coeff': 0.0,  'loss_type': 'l2'},
        {'name': 'sign_constraint_only',    'coeff_init': 'zeros',      'coeff_sign_weight': 5.0, 'min_coeff': 0.05, 'loss_type': 'l2'},
        {'name': 'init+constraint',         'coeff_init': 'sign_aware', 'coeff_sign_weight': 5.0, 'min_coeff': 0.05, 'loss_type': 'l2'},
        {'name': 'init+constraint+cosine',  'coeff_init': 'sign_aware', 'coeff_sign_weight': 5.0, 'min_coeff': 0.05, 'loss_type': 'cosine'},
    ]

    # Ablation B: min_coeff threshold sweep (with sign_aware init + sign_weight=5.0)
    ablation_b = [
        {'name': 'min_coeff=0.01', 'coeff_init': 'sign_aware', 'coeff_sign_weight': 5.0, 'min_coeff': 0.01, 'loss_type': 'l2'},
        # min_coeff=0.05 is already in ablation_a as 'init+constraint'
        {'name': 'min_coeff=0.10', 'coeff_init': 'sign_aware', 'coeff_sign_weight': 5.0, 'min_coeff': 0.10, 'loss_type': 'l2'},
        {'name': 'min_coeff=0.25', 'coeff_init': 'sign_aware', 'coeff_sign_weight': 5.0, 'min_coeff': 0.25, 'loss_type': 'l2'},
    ]

    all_configs = ablation_a + ablation_b

    print("\n" + "=" * 70)
    print(f"SEED FIX ABLATION: {len(all_configs)} configs × {len(seeds)} seeds × 2 models")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    results_list = []
    total = len(all_configs) * len(seeds) * 2  # 2 = full + LoRA r=8
    i = 0

    for cfg in all_configs:
        for seed in seeds:
            for rank in [None, 8]:
                i += 1
                rank_label = "full" if rank is None else f"r={rank}"
                print(f"\n[{i}/{total}] {cfg['name']}, seed={seed}, {rank_label}")
                try:
                    res = run_exp_b(
                        n_steps=1, rank=rank, n_per_class=1, seed=seed,
                        run_baseline=(rank is None and cfg is all_configs[0] and seed == seeds[0]),
                        free_coefficients=True,
                        coeff_consistency_weight=1.0,
                        optimizer_type='sgd',
                        finetune_activation='leaky_relu',
                        extraction_epochs=50_000,
                        coeff_init=cfg['coeff_init'],
                        coeff_sign_weight=cfg['coeff_sign_weight'],
                        coeff_min_magnitude=cfg['min_coeff'],
                        loss_type=cfg['loss_type'],
                        device=device, verbose=True,
                    )
                    row = extract_row(res, {
                        'task': 'seed_fix_ablation',
                        'config_name': cfg['name'],
                        'coeff_init': cfg['coeff_init'],
                        'coeff_sign_weight': cfg['coeff_sign_weight'],
                        'min_coeff': cfg['min_coeff'],
                        'loss_type': cfg['loss_type'],
                    })
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'seed': seed, 'rank': rank or 'full',
                        'config_name': cfg['name'],
                        'coeff_init': cfg['coeff_init'],
                        'coeff_sign_weight': cfg['coeff_sign_weight'],
                        'min_coeff': cfg['min_coeff'],
                        'loss_type': cfg['loss_type'],
                        'task': 'seed_fix_ablation', 'error': str(e),
                    })

                # Save incrementally every 10 runs
                if i % 10 == 0:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_path = os.path.join(RESULTS_DIR, f'seed_fix_ablation_partial_{ts}.csv')
                    save_csv(results_list, csv_path)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'seed_fix_ablation_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnostic experiments')
    parser.add_argument('--task', type=str, nargs='+',
                        default=['n6_diagnostic', 'b6_seeds'],
                        choices=['n6_diagnostic', 'b6_seeds', 'seed_fix_ablation'],
                        help='Which diagnostics to run')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Custom seed list (overrides defaults)')
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    task_fns = {
        'n6_diagnostic': lambda: run_n6_diagnostic(device, args.seed),
        'b6_seeds': lambda: run_b6_multi_seed(device, seeds=args.seeds),
        'seed_fix_ablation': lambda: run_seed_fix_ablation(device, seeds=args.seeds),
    }

    for t in args.task:
        task_fns[t]()

    print("\n=== Diagnostics complete ===")
