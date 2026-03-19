"""Sprint 2b: Push NTK reconstruction to larger T.

Unified sweep driver for all Sprint 2b experiment phases.
Phases can be selected via CLI flags.

Phases:
    0: Activation function ablation
    1: SGD + free-c baseline (T × rank)
    2: Random restarts
    3: LR scaling (constant, inv_sqrt_T, inv_T)
    4: Progressive warm-start
    5: LR magnitude ablation (constant SGD at multiple LR values)
    6: Cosine SGD schedule (per-step cosine annealing)
    7: Linear LR schedule (per-step linear decay)

Usage (on WEXAC):
    python -u -m experiments.run_sprint2b_sweep --phase 0 --device cuda
    python -u -m experiments.run_sprint2b_sweep --phase 5 6 --device cuda --finetune_activation leaky_relu
    python -u -m experiments.run_sprint2b_sweep --phase 7 --device cuda --finetune_activation leaky_relu
"""

import sys
import os
import argparse
import csv
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

torch.set_default_dtype(torch.float64)

from experiments.configs import (
    RESULTS_DIR, FIGURES_DIR, EXTRACTION_EPOCHS,
    TRAIN_LR, STEP_SWEEP, RANK_SWEEP_EXP_B,
    EXTRACTION_RELU_ALPHA,
    COEFF_LR, COEFF_BOX_WEIGHT,
)
from experiments.run_experiment_b import run_single_config


def save_csv(results_list, csv_path):
    """Save results list to CSV, handling varying keys."""
    if not results_list:
        return
    all_keys = set()
    for r in results_list:
        all_keys.update(r.keys())
    # Sort keys for deterministic output
    keys = sorted(all_keys)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_list)
    print(f"Results saved to {csv_path}")


def extract_row(results, extra_fields=None):
    """Extract a flat row dict from run_single_config results."""
    row = {
        'n_steps': results.get('n_steps'),
        'rank': results.get('rank', 'full'),
        'seed': results.get('seed'),
        'activation': results.get('finetune_activation') or 'relu',
        'lr_schedule': results.get('lr_schedule', 'constant'),
        'effective_lr': results.get('effective_lr'),
    }
    if 'full_metrics' in results:
        row['full_ssim'] = results['full_metrics'].get('ssim')
    if 'lora_metrics' in results:
        row['lora_ssim'] = results['lora_metrics'].get('ssim')
    if 'full_diagnostics' in results:
        row['weight_change'] = results['full_diagnostics'].get('weight_change')
        row['feature_stability'] = results['full_diagnostics'].get('feature_stability')
    if 'full_linearization_error' in results:
        row['full_linearization_error'] = results['full_linearization_error']
    if 'lora_linearization_error' in results:
        row['lora_linearization_error'] = results['lora_linearization_error']
    if 'control_metrics' in results:
        row['control_ssim'] = results['control_metrics'].get('ssim')
    if extra_fields:
        row.update(extra_fields)
    return row


def run_phase0(device, extraction_epochs, seed=42):
    """Phase 0: Activation function ablation.

    3 activations × T={1, 5, 10, 50, 100} × full model only.
    """
    print("\n" + "=" * 70)
    print("PHASE 0: Activation Function Ablation")
    print("=" * 70)

    activations = [None, 'leaky_relu', 'modified_relu']  # None = default (ReLU+ModifiedRelu)
    steps = [1, 5, 10, 50, 100]

    results_list = []
    total = len(activations) * len(steps)
    i = 0

    for act in activations:
        act_label = act or 'relu_default'
        for T in steps:
            i += 1
            print(f"\n[{i}/{total}] activation={act_label}, T={T}")
            try:
                res = run_single_config(
                    n_steps=T, rank=None, n_per_class=1, seed=seed,
                    run_baseline=True,
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type='sgd',
                    extraction_epochs=extraction_epochs,
                    finetune_activation=act,
                    device=device, verbose=True,
                )
                row = extract_row(res, {'phase': 0})
                row['activation'] = act_label
                results_list.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_steps': T, 'activation': act_label,
                    'phase': 0, 'error': str(e),
                })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2b_phase0_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_phase1(device, extraction_epochs, seed=42, finetune_activation=None):
    """Phase 1: SGD + free-coefficient baseline across T × rank.

    T: {1, 2, 5, 10, 20, 50, 100, 500, 1000} × rank: {full, 1, 4, 8, 32, 64}
    """
    print("\n" + "=" * 70)
    print("PHASE 1: SGD + Free-Coefficient Baseline")
    print("=" * 70)

    steps = STEP_SWEEP
    ranks = [None] + RANK_SWEEP_EXP_B  # None = full

    results_list = []
    total = len(steps) * len(ranks)
    i = 0

    for T in steps:
        for rank in ranks:
            i += 1
            label = "full" if rank is None else f"r={rank}"
            print(f"\n[{i}/{total}] T={T}, {label}")
            try:
                res = run_single_config(
                    n_steps=T, rank=rank, n_per_class=1, seed=seed,
                    run_baseline=(rank is None or rank == ranks[1]),
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type='sgd',
                    extraction_epochs=extraction_epochs,
                    finetune_activation=finetune_activation,
                    device=device, verbose=True,
                )
                row = extract_row(res, {'phase': 1})
                results_list.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_steps': T, 'rank': rank or 'full',
                    'phase': 1, 'error': str(e),
                })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2b_phase1_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_phase2(device, extraction_epochs, seed=42, finetune_activation=None):
    """Phase 2: Random restarts.

    T: {1, 5, 10, 20, 100} × rank: {full, 8, 32} × 3 restarts.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Random Restarts")
    print("=" * 70)

    steps = [1, 5, 10, 20, 100]
    ranks = [None, 8, 32]
    n_restarts = 3

    results_list = []
    total = len(steps) * len(ranks) * n_restarts
    i = 0

    for T in steps:
        for rank in ranks:
            for restart in range(n_restarts):
                i += 1
                label = "full" if rank is None else f"r={rank}"
                print(f"\n[{i}/{total}] T={T}, {label}, restart={restart}")
                try:
                    res = run_single_config(
                        n_steps=T, rank=rank, n_per_class=1, seed=seed,
                        run_baseline=(rank is None or rank == 8),
                        free_coefficients=True,
                        coeff_consistency_weight=1.0,
                        optimizer_type='sgd',
                        extraction_epochs=extraction_epochs,
                        finetune_activation=finetune_activation,
                        init_seed=seed + restart * 1000,
                        device=device, verbose=True,
                    )
                    row = extract_row(res, {'phase': 2, 'restart': restart,
                                            'init_seed': seed + restart * 1000})
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'n_steps': T, 'rank': rank or 'full',
                        'restart': restart, 'phase': 2, 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2b_phase2_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_phase3(device, extraction_epochs, seed=42, finetune_activation=None,
               steps_override=None):
    """Phase 3: LR scaling.

    T: {1, 5, 10, 20, 50, 100, 500, 1000} × 3 LR schedules × full model only.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: LR Scaling")
    print("=" * 70)

    steps = steps_override or STEP_SWEEP
    schedules = ['constant', 'inv_sqrt_T', 'inv_T']

    results_list = []
    total = len(steps) * len(schedules)
    i = 0

    for T in steps:
        for sched in schedules:
            i += 1
            print(f"\n[{i}/{total}] T={T}, schedule={sched}")
            try:
                res = run_single_config(
                    n_steps=T, rank=None, n_per_class=1, seed=seed,
                    run_baseline=True,
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type='sgd',
                    extraction_epochs=extraction_epochs,
                    finetune_activation=finetune_activation,
                    lr_schedule=sched,
                    device=device, verbose=True,
                )
                row = extract_row(res, {'phase': 3})
                results_list.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_steps': T, 'lr_schedule': sched,
                    'phase': 3, 'error': str(e),
                })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2b_phase3_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_phase4(device, extraction_epochs, seed=42, finetune_activation=None,
               steps_override=None):
    """Phase 4: Progressive warm-start.

    Chain: T=1→2→5→10→20→50→100→500→1000 for {full, LoRA r=8}.
    Initialize x at T=k from solution at T=k-1.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Progressive Warm-Start")
    print("=" * 70)

    steps = steps_override or [1, 2, 5, 10, 20, 50, 100, 500, 1000]
    ranks = [None, 8]

    results_list = []
    total = len(steps) * len(ranks)
    i = 0

    for rank in ranks:
        label = "full" if rank is None else f"r={rank}"
        prev_x = None  # warm-start chain

        for T in steps:
            i += 1
            print(f"\n[{i}/{total}] T={T}, {label}, warm_start={'yes' if prev_x is not None else 'no'}")
            try:
                res = run_single_config(
                    n_steps=T, rank=rank, n_per_class=1, seed=seed,
                    run_baseline=(rank is None or rank == 8),
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type='sgd',
                    extraction_epochs=extraction_epochs,
                    finetune_activation=finetune_activation,
                    x_init=prev_x,
                    device=device, verbose=True,
                )
                row = extract_row(res, {
                    'phase': 4,
                    'warm_start': prev_x is not None,
                })
                results_list.append(row)

                # Update warm-start x for next T
                if rank is None and 'x_recon_full' in res:
                    prev_x = res['x_recon_full'].detach().clone()
                elif rank is not None and 'x_recon_lora' in res:
                    prev_x = res['x_recon_lora'].detach().clone()

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_steps': T, 'rank': rank or 'full',
                    'phase': 4, 'error': str(e),
                })
                prev_x = None  # reset chain on failure

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2b_phase4_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_phase5(device, extraction_epochs, seed=42, finetune_activation=None,
               steps_override=None):
    """Phase 5: LR magnitude ablation.

    LR ∈ {0.001, 0.005, 0.01, 0.05} × T ∈ {1, 5, 10, 50, 100} × {full, LoRA r=8, r=32}.
    All with constant LR schedule. Tests whether the attack works when the
    victim uses a different (but still constant) learning rate.
    """
    from experiments.configs import LR_MAGNITUDE_SWEEP

    print("\n" + "=" * 70)
    print("PHASE 5: LR Magnitude Ablation")
    print("=" * 70)

    steps = steps_override or [1, 5, 10, 50, 100]
    lrs = LR_MAGNITUDE_SWEEP
    ranks = [None, 8, 32]

    results_list = []
    total = len(lrs) * len(steps) * len(ranks)
    i = 0

    for lr_val in lrs:
        for T in steps:
            for rank in ranks:
                i += 1
                label = "full" if rank is None else f"r={rank}"
                print(f"\n[{i}/{total}] lr={lr_val}, T={T}, {label}")
                try:
                    res = run_single_config(
                        n_steps=T, rank=rank, n_per_class=1, seed=seed,
                        lr=lr_val,
                        run_baseline=(rank is None or rank == 8),
                        free_coefficients=True,
                        coeff_consistency_weight=1.0,
                        optimizer_type='sgd',
                        extraction_epochs=extraction_epochs,
                        finetune_activation=finetune_activation,
                        lr_schedule='constant',
                        device=device, verbose=True,
                    )
                    row = extract_row(res, {'phase': 5, 'lr': lr_val})
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'n_steps': T, 'rank': rank or 'full',
                        'lr': lr_val, 'phase': 5, 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2b_phase5_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_phase6(device, extraction_epochs, seed=42, finetune_activation=None,
               steps_override=None):
    """Phase 6: Cosine SGD schedule.

    T ∈ {10, 50, 100, 500} × {full, LoRA r=8, r=32}.
    Per-step cosine annealing: lr_t = lr * 0.5 * (1 + cos(pi * t / T)).
    Tests what happens when LR decays to ~0 during training (theory says
    suboptimal for max-margin, but the attacker may face a victim who uses it).
    """
    print("\n" + "=" * 70)
    print("PHASE 6: Cosine SGD Schedule")
    print("=" * 70)

    steps = steps_override or [10, 50, 100, 500]
    ranks = [None, 8, 32]

    results_list = []
    total = len(steps) * len(ranks)
    i = 0

    for T in steps:
        for rank in ranks:
            i += 1
            label = "full" if rank is None else f"r={rank}"
            print(f"\n[{i}/{total}] T={T}, {label}, schedule=cosine")
            try:
                res = run_single_config(
                    n_steps=T, rank=rank, n_per_class=1, seed=seed,
                    run_baseline=(rank is None or rank == 8),
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type='sgd',
                    extraction_epochs=extraction_epochs,
                    finetune_activation=finetune_activation,
                    lr_schedule='cosine',
                    device=device, verbose=True,
                )
                row = extract_row(res, {'phase': 6})
                results_list.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_steps': T, 'rank': rank or 'full',
                    'lr_schedule': 'cosine', 'phase': 6, 'error': str(e),
                })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2b_phase6_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_phase7(device, extraction_epochs, seed=42, finetune_activation=None,
               steps_override=None):
    """Phase 7: Linear LR schedule.

    T ∈ {1, 2, 5, 10, 20, 50, 100, 500, 1000} × {full model only}.
    Per-step linear decay: lr_t = lr * (1 - t/T).
    Completes the schedule comparison from Phase 3 (constant, inv_sqrt_T, inv_T).
    Expected: falls between constant and inv_sqrt_T (Σ lr ≈ lr·T/2).
    """
    print("\n" + "=" * 70)
    print("PHASE 7: Linear LR Schedule")
    print("=" * 70)

    steps = steps_override or STEP_SWEEP
    ranks = [None, 8]

    results_list = []
    total = len(steps) * len(ranks)
    i = 0

    for T in steps:
        for rank in ranks:
            i += 1
            label = "full" if rank is None else f"r={rank}"
            print(f"\n[{i}/{total}] T={T}, {label}, schedule=linear")
            try:
                res = run_single_config(
                    n_steps=T, rank=rank, n_per_class=1, seed=seed,
                    run_baseline=(rank is None),
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type='sgd',
                    extraction_epochs=extraction_epochs,
                    finetune_activation=finetune_activation,
                    lr_schedule='linear',
                    device=device, verbose=True,
                )
                row = extract_row(res, {'phase': 7, 'lr_schedule': 'linear'})
                results_list.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_steps': T, 'rank': rank or 'full',
                    'lr_schedule': 'linear', 'phase': 7, 'error': str(e),
                })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2b_phase7_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


if __name__ == '__main__':
    from experiments.configs import get_device, ACTIVATION_CHOICES

    parser = argparse.ArgumentParser(description='Sprint 2b sweep')
    parser.add_argument('--phase', type=int, nargs='+', default=[0],
                        help='Which phases to run (0-7)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--extraction_epochs', type=int, default=EXTRACTION_EPOCHS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetune_activation', type=str, default=None,
                        choices=ACTIVATION_CHOICES,
                        help='Override activation for phases 1-4 (phase 0 tests all)')
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated step values to override STEP_SWEEP (e.g. "1,5,10")')
    args = parser.parse_args()

    device = args.device or get_device()
    steps_override = [int(s) for s in args.steps.split(',')] if args.steps else None
    print(f"Using device: {device}")
    print(f"Extraction epochs: {args.extraction_epochs}")
    print(f"Phases to run: {args.phase}")
    if steps_override:
        print(f"Steps override: {steps_override}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    phase_fns = {
        0: lambda: run_phase0(device, args.extraction_epochs, args.seed),
        1: lambda: run_phase1(device, args.extraction_epochs, args.seed,
                              args.finetune_activation),
        2: lambda: run_phase2(device, args.extraction_epochs, args.seed,
                              args.finetune_activation),
        3: lambda: run_phase3(device, args.extraction_epochs, args.seed,
                              args.finetune_activation,
                              steps_override=steps_override),
        4: lambda: run_phase4(device, args.extraction_epochs, args.seed,
                              args.finetune_activation,
                              steps_override=steps_override),
        5: lambda: run_phase5(device, args.extraction_epochs, args.seed,
                              args.finetune_activation,
                              steps_override=steps_override),
        6: lambda: run_phase6(device, args.extraction_epochs, args.seed,
                              args.finetune_activation,
                              steps_override=steps_override),
        7: lambda: run_phase7(device, args.extraction_epochs, args.seed,
                              args.finetune_activation,
                              steps_override=steps_override),
    }

    for p in args.phase:
        if p in phase_fns:
            phase_fns[p]()
        else:
            print(f"Unknown phase: {p}")

    print("\n=== Sprint 2b sweep complete ===")
