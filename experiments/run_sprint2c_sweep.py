"""Sprint 2c: KKT & NTK Reconstruction Ablations.

Sweep driver for Sprint 2c experiments. Organized into two tracks:
- Track A: Experiment A (KKT) — fine-tuning LR × epochs × N sweep
- Track B: Experiment B (NTK) — loss ratio, optimizer×activation, N sweep,
  few-shot N, AdamW probe, consistency weight ablation

Usage (on WEXAC):
    # Track A: KKT with fine-tuning LR and N sweep
    python -u -m experiments.run_sprint2c_sweep --track A --device cuda

    # Track B ablations
    python -u -m experiments.run_sprint2c_sweep --track B2 --device cuda
    python -u -m experiments.run_sprint2c_sweep --track B3a --device cuda
    python -u -m experiments.run_sprint2c_sweep --track B4 --device cuda
    python -u -m experiments.run_sprint2c_sweep --track B5 --device cuda   # N fine-tuning sweep
    python -u -m experiments.run_sprint2c_sweep --track B6 --device cuda   # AdamW probe
    python -u -m experiments.run_sprint2c_sweep --track B7 --device cuda   # consistency weight × T
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
    RESULTS_DIR, EXTRACTION_EPOCHS, TRAIN_LR, TRAIN_EPOCHS,
    STEP_SWEEP, RANK_SWEEP_EXP_B,
)
from experiments.run_experiment_a import run_single_config as run_exp_a
from experiments.run_experiment_b import run_single_config as run_exp_b


def save_csv(results_list, csv_path):
    """Save results list to CSV, handling varying keys."""
    if not results_list:
        return
    all_keys = set()
    for r in results_list:
        all_keys.update(r.keys())
    keys = sorted(all_keys)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results_list)
    print(f"Results saved to {csv_path}")


def extract_row_a(results, extra_fields=None):
    """Extract a flat row dict from Experiment A results."""
    row = {
        'rank': results.get('rank'),
        'n_per_class': results.get('n_per_class'),
        'extraction_n_per_class': results.get('extraction_n_per_class'),
        'fine_tune_lr': results.get('fine_tune_lr'),
        'fine_tune_epochs': results.get('fine_tune_epochs'),
        'seed': results.get('seed'),
    }
    if 'lora_train' in results:
        row['lora_final_loss'] = results['lora_train'].get('final_loss')
        row['lora_converged'] = results['lora_train'].get('converged')
        row['lora_epochs'] = results['lora_train'].get('epochs')
    if 'lora_metrics' in results:
        row['lora_ssim'] = results['lora_metrics'].get('ssim')
        row['lora_dssim'] = results['lora_metrics'].get('dssim')
    if 'lora_final_kkt_loss' in results:
        row['lora_kkt_loss'] = results['lora_final_kkt_loss']
    if 'full_ft_train' in results:
        row['full_final_loss'] = results['full_ft_train'].get('final_loss')
        row['full_converged'] = results['full_ft_train'].get('converged')
    if 'full_ft_metrics' in results:
        row['full_ssim'] = results['full_ft_metrics'].get('ssim')
        row['full_dssim'] = results['full_ft_metrics'].get('dssim')
    if 'full_final_kkt_loss' in results:
        row['full_kkt_loss'] = results['full_final_kkt_loss']
    if 'control_metrics' in results:
        row['control_ssim'] = results['control_metrics'].get('ssim')
    if extra_fields:
        row.update(extra_fields)
    return row


def extract_row_b(results, extra_fields=None):
    """Extract a flat row dict from Experiment B results."""
    row = {
        'n_steps': results.get('n_steps'),
        'rank': results.get('rank', 'full'),
        'seed': results.get('seed'),
        'activation': results.get('finetune_activation') or 'relu',
        'optimizer': extra_fields.get('optimizer', 'lbfgs') if extra_fields else 'lbfgs',
    }
    if 'full_metrics' in results:
        row['full_ssim'] = results['full_metrics'].get('ssim')
    if 'lora_metrics' in results:
        row['lora_ssim'] = results['lora_metrics'].get('ssim')
    if 'control_metrics' in results:
        row['control_ssim'] = results['control_metrics'].get('ssim')
    if extra_fields:
        row.update(extra_fields)
    return row


def run_track_a(device, extraction_epochs, seed=42):
    """Track A: Experiment A — Fine-tuning LR × epochs × N sweep.

    Grid: fine_tune_lr ∈ {0.001, 0.003, 0.01} × epochs ∈ {1M, 5M}
          × {full FT, LoRA r=8}
    For each: N sweep with extraction_n_per_class ∈ {1, 5, 10, 25, 50, 100, 250, 251}
    """
    print("\n" + "=" * 70)
    print("TRACK A: Experiment A — KKT with Fine-Tuning LR × N Sweep")
    print("=" * 70)

    fine_tune_lrs = [0.001, 0.003, 0.01]
    fine_tune_epoch_list = [1_000_000, 5_000_000]
    ranks = [8, None]  # LoRA r=8 and full FT (None = full for LoRA, baseline handles full FT)
    n_sweep = [1, 5, 10, 25, 50, 100, 250, 251]

    results_list = []
    total = len(fine_tune_lrs) * len(fine_tune_epoch_list) * len(n_sweep)
    # Each (lr, epochs) pair trains one LoRA model and one full FT model,
    # then runs extraction with each N value.
    i = 0

    for ft_lr in fine_tune_lrs:
        for ft_epochs in fine_tune_epoch_list:
            for ext_n in n_sweep:
                i += 1
                print(f"\n[{i}/{total}] ft_lr={ft_lr}, ft_epochs={ft_epochs}, "
                      f"extraction_n={ext_n}")
                try:
                    res = run_exp_a(
                        rank=8, n_per_class=1, seed=seed,
                        run_baseline=True,
                        extraction_epochs=extraction_epochs,
                        extraction_n_per_class=ext_n,
                        fine_tune_lr=ft_lr,
                        fine_tune_epochs=ft_epochs,
                        device=device, verbose=True,
                    )
                    row = extract_row_a(res, {'track': 'A'})
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'rank': 8, 'fine_tune_lr': ft_lr,
                        'fine_tune_epochs': ft_epochs,
                        'extraction_n_per_class': ext_n,
                        'track': 'A', 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_a_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_track_b2(device, extraction_epochs, seed=42):
    """Track B2: Loss Ratio Ablation (verify_weight).

    verify_weight ∈ {0.01, 0.1, 1.0, 10.0} × T ∈ {1, 10} × {full, LoRA r=8}
    """
    print("\n" + "=" * 70)
    print("TRACK B2: Loss Ratio Ablation (verify_weight)")
    print("=" * 70)

    verify_weights = [0.01, 0.1, 1.0, 10.0]
    steps = [1, 10]
    ranks = [None, 8]

    results_list = []
    total = len(verify_weights) * len(steps) * len(ranks)
    i = 0

    for vw in verify_weights:
        for T in steps:
            for rank in ranks:
                i += 1
                label = "full" if rank is None else f"r={rank}"
                print(f"\n[{i}/{total}] verify_weight={vw}, T={T}, {label}")
                try:
                    res = run_exp_b(
                        n_steps=T, rank=rank, n_per_class=1, seed=seed,
                        run_baseline=(rank is None),
                        free_coefficients=True,
                        coeff_consistency_weight=1.0,
                        optimizer_type='sgd',
                        finetune_activation='leaky_relu',
                        verify_weight=vw,
                        extraction_epochs=extraction_epochs,
                        device=device, verbose=True,
                    )
                    row = extract_row_b(res, {
                        'track': 'B2', 'verify_weight': vw,
                        'optimizer': 'sgd',
                    })
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'n_steps': T, 'rank': rank or 'full',
                        'verify_weight': vw,
                        'track': 'B2', 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_b2_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_track_b3a(device, extraction_epochs, seed=42):
    """Track B3a: Optimizer × Activation for LoRA (grid search at T=1).

    optimizer ∈ {lbfgs, sgd, adam} × activation ∈ {leaky_relu, relu}
    × rank ∈ {8, 32}
    """
    print("\n" + "=" * 70)
    print("TRACK B3a: Optimizer × Activation for LoRA")
    print("=" * 70)

    optimizers = ['lbfgs', 'sgd', 'adam']
    activations = ['leaky_relu', 'relu']
    ranks = [8, 32]

    results_list = []
    total = len(optimizers) * len(activations) * len(ranks)
    i = 0

    for opt in optimizers:
        for act in activations:
            for rank in ranks:
                i += 1
                print(f"\n[{i}/{total}] optimizer={opt}, activation={act}, rank={rank}")
                try:
                    res = run_exp_b(
                        n_steps=1, rank=rank, n_per_class=1, seed=seed,
                        run_baseline=False,
                        free_coefficients=True,
                        coeff_consistency_weight=1.0,
                        optimizer_type=opt,
                        finetune_activation=act if act != 'relu' else None,
                        extraction_epochs=extraction_epochs,
                        device=device, verbose=True,
                    )
                    row = extract_row_b(res, {
                        'track': 'B3a', 'optimizer': opt,
                    })
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'n_steps': 1, 'rank': rank,
                        'optimizer': opt, 'activation': act,
                        'track': 'B3a', 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_b3a_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_track_b3b(device, extraction_epochs, seed=42,
                  best_optimizer='lbfgs', best_activation=None):
    """Track B3b: Scale best LoRA combo across T (depends on B3a results).

    Best combo at T ∈ {1, 5, 10, 20, 100} × rank ∈ {8, 32}
    """
    print("\n" + "=" * 70)
    print("TRACK B3b: Best LoRA Combo Across T")
    print(f"  Using optimizer={best_optimizer}, activation={best_activation}")
    print("=" * 70)

    steps = [1, 5, 10, 20, 100]
    ranks = [8, 32]

    results_list = []
    total = len(steps) * len(ranks)
    i = 0

    for T in steps:
        for rank in ranks:
            i += 1
            print(f"\n[{i}/{total}] T={T}, rank={rank}")
            try:
                res = run_exp_b(
                    n_steps=T, rank=rank, n_per_class=1, seed=seed,
                    run_baseline=False,
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type=best_optimizer,
                    finetune_activation=best_activation,
                    extraction_epochs=extraction_epochs,
                    device=device, verbose=True,
                )
                row = extract_row_b(res, {
                    'track': 'B3b', 'optimizer': best_optimizer,
                })
                results_list.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_steps': T, 'rank': rank,
                    'optimizer': best_optimizer,
                    'track': 'B3b', 'error': str(e),
                })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_b3b_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_track_b4(device, extraction_epochs, seed=42):
    """Track B4: N Sweep (NTK).

    T ∈ {1, 5, 10, 20} × {full, LoRA r=8} with --n_sweep.
    """
    print("\n" + "=" * 70)
    print("TRACK B4: N Sweep (NTK)")
    print("=" * 70)

    steps = [1, 5, 10, 20]
    ranks = [None, 8]

    results_list = []
    total = len(steps) * len(ranks)
    i = 0

    for T in steps:
        for rank in ranks:
            i += 1
            label = "full" if rank is None else f"r={rank}"
            print(f"\n[{i}/{total}] T={T}, {label}")
            try:
                res = run_exp_b(
                    n_steps=T, rank=rank, n_per_class=1, seed=seed,
                    run_baseline=(rank is None),
                    free_coefficients=True,
                    coeff_consistency_weight=1.0,
                    optimizer_type='sgd',
                    finetune_activation='leaky_relu',
                    n_sweep=True,
                    extraction_epochs=extraction_epochs,
                    device=device, verbose=True,
                )
                row = extract_row_b(res, {
                    'track': 'B4', 'optimizer': 'sgd',
                })
                # Add N sweep results
                for key in ['full_extract_res', 'lora_extract_res']:
                    if key in res and 'n_sweep_winner' in res[key]:
                        prefix = 'full' if 'full' in key else 'lora'
                        row[f'{prefix}_n_sweep_winner'] = res[key]['n_sweep_winner']
                        row[f'{prefix}_n_sweep_losses'] = str(res[key].get('n_sweep_all', {}))
                results_list.append(row)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results_list.append({
                    'n_steps': T, 'rank': rank or 'full',
                    'track': 'B4', 'error': str(e),
                })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_b4_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_track_b5(device, extraction_epochs, seed=42):
    """Track B5: N Fine-Tuning Sweep (realistic few-shot sizes).

    Tests whether the NTK attack works with more fine-tuning data (N>2).
    n_per_class ∈ {1, 3, 5, 10} → N ∈ {2, 6, 10, 20} total.
    T ∈ {1, 10} × {full, LoRA r=8}.
    """
    print("\n" + "=" * 70)
    print("TRACK B5: N Fine-Tuning Sweep (Few-Shot Realism)")
    print("=" * 70)

    n_per_class_values = [1, 3, 5, 10]
    steps = [1, 10]
    ranks = [None, 8]

    results_list = []
    total = len(n_per_class_values) * len(steps) * len(ranks)
    i = 0

    for npc in n_per_class_values:
        for T in steps:
            for rank in ranks:
                i += 1
                label = "full" if rank is None else f"r={rank}"
                # Use inv_T for T>1 to keep linearization bounded
                schedule = 'inv_T' if T > 1 else 'constant'
                print(f"\n[{i}/{total}] n_per_class={npc} (N={2*npc}), "
                      f"T={T}, {label}, schedule={schedule}")
                try:
                    res = run_exp_b(
                        n_steps=T, rank=rank, n_per_class=npc, seed=seed,
                        run_baseline=(rank is None),
                        free_coefficients=True,
                        coeff_consistency_weight=1.0,
                        optimizer_type='sgd',
                        finetune_activation='leaky_relu',
                        lr_schedule=schedule,
                        extraction_epochs=extraction_epochs,
                        device=device, verbose=True,
                    )
                    row = extract_row_b(res, {
                        'track': 'B5', 'optimizer': 'sgd',
                        'n_per_class': npc,
                        'n_total': 2 * npc,
                        'lr_schedule': schedule,
                    })
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'n_steps': T, 'rank': rank or 'full',
                        'n_per_class': npc, 'n_total': 2 * npc,
                        'track': 'B5', 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_b5_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_track_b6(device, extraction_epochs, seed=42):
    """Track B6: AdamW Fine-Tuning Probe (realism test).

    Tests whether NTK reconstruction works when the victim fine-tunes
    with AdamW instead of SGD. AdamW is the standard optimizer in real
    LoRA practice but breaks the implicit bias theory guarantees.

    Configs: {sgd, adamw} × T ∈ {1, 10} × {full, LoRA r=8}
    """
    print("\n" + "=" * 70)
    print("TRACK B6: AdamW Fine-Tuning Probe")
    print("=" * 70)

    finetune_optimizers = ['sgd', 'adamw']
    steps = [1, 10]
    ranks = [None, 8]

    # AdamW typically uses lower LR than SGD
    lr_by_optimizer = {'sgd': 0.01, 'adamw': 1e-3}

    results_list = []
    total = len(finetune_optimizers) * len(steps) * len(ranks)
    i = 0

    for ft_opt in finetune_optimizers:
        for T in steps:
            for rank in ranks:
                i += 1
                label = "full" if rank is None else f"r={rank}"
                ft_lr = lr_by_optimizer[ft_opt]
                print(f"\n[{i}/{total}] finetune_optimizer={ft_opt}, "
                      f"T={T}, {label}, lr={ft_lr}")
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
                        extraction_epochs=extraction_epochs,
                        device=device, verbose=True,
                    )
                    row = extract_row_b(res, {
                        'track': 'B6', 'optimizer': 'sgd',
                        'finetune_optimizer': ft_opt,
                        'finetune_lr': ft_lr,
                    })
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'n_steps': T, 'rank': rank or 'full',
                        'finetune_optimizer': ft_opt,
                        'track': 'B6', 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_b6_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_track_b7(device, extraction_epochs, seed=42):
    """Track B7: Consistency Weight × T Ablation.

    Tests whether the consistency penalty (coupling free coefficients to model
    predictions at θ₀) needs to be tuned as T grows.  At T=1, α=1 is exact
    because the coefficients are computed at θ₀ ≈ θ_T.  At large T, θ_T
    drifts from θ₀, so the consistency target becomes stale — higher α may
    over-constrain, while lower α may lose the sign-disambiguation benefit.

    Grid: α ∈ {0, 0.1, 1.0, 5.0} × T ∈ {1, 10, 100, 1000}
          × {full, LoRA r=8} = 32 configs.
    Uses inv_T schedule for all T (keeps ||ΔW|| bounded at high T).
    """
    print("\n" + "=" * 70)
    print("TRACK B7: Consistency Weight × T Ablation")
    print("=" * 70)

    consistency_weights = [0.0, 0.1, 1.0, 5.0]
    steps = [1, 10, 100, 1000]
    ranks = [None, 8]

    results_list = []
    total = len(consistency_weights) * len(steps) * len(ranks)
    i = 0

    for alpha in consistency_weights:
        for T in steps:
            for rank in ranks:
                i += 1
                label = "full" if rank is None else f"r={rank}"
                schedule = 'inv_T' if T > 1 else 'constant'
                print(f"\n[{i}/{total}] consistency_weight={alpha}, T={T}, "
                      f"{label}, schedule={schedule}")
                try:
                    res = run_exp_b(
                        n_steps=T, rank=rank, n_per_class=1, seed=seed,
                        run_baseline=(rank is None),
                        free_coefficients=True,
                        coeff_consistency_weight=alpha,
                        optimizer_type='sgd',
                        finetune_activation='leaky_relu',
                        lr_schedule=schedule,
                        extraction_epochs=extraction_epochs,
                        device=device, verbose=True,
                    )
                    row = extract_row_b(res, {
                        'track': 'B7', 'optimizer': 'sgd',
                        'consistency_weight': alpha,
                        'lr_schedule': schedule,
                    })
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'n_steps': T, 'rank': rank or 'full',
                        'consistency_weight': alpha,
                        'track': 'B7', 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_b7_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


def run_track_b8(device, extraction_epochs, seed=42):
    """Track B8: Loss Weight Ablation (box_weight + extended consistency_weight).

    Completes the loss weight picture started by B2 (verify_weight) and B7
    (consistency_weight × T).  B7 sweeps CW ∈ {0, 0.1, 1, 5} but never
    varies box_weight from the default 5.0.

    Sub-sweep 1: finer CW grid — CW ∈ {0.01, 0.5, 2.0, 10.0, 100.0}, BW=5.0
    Sub-sweep 2: box_weight   — BW ∈ {0, 0.5, 1.0, 5.0, 20.0}, CW=1.0
    Sub-sweep 3: bare baseline — CW=0, BW=0

    Each at {T=1, T=10} × {full, LoRA r=8}.  ~53 configs total.
    Settings: leaky_relu, SGD, free-c, inv_T for T>1.
    """
    print("\n" + "=" * 70)
    print("TRACK B8: Loss Weight Ablation (box_weight + extended CW)")
    print("=" * 70)

    steps = [1, 10]
    ranks = [None, 8]
    results_list = []
    i = 0

    # --- Sub-sweep 1: finer CW grid (BW=5.0 default) ---
    cw_extra = [0.01, 0.5, 2.0, 10.0, 100.0]
    bw_default = 5.0

    # --- Sub-sweep 2: box_weight sweep (CW=1.0) ---
    bw_values = [0.0, 0.5, 1.0, 5.0, 20.0]
    cw_best = 1.0

    # --- Sub-sweep 3: bare baseline ---
    # CW=0, BW=0 — already covered if CW=0 appears in sub-sweep 1 AND
    # BW=0 appears in sub-sweep 2, but we need both zero simultaneously.
    # We'll add it explicitly.

    # Build config list: (cw, bw, sub_sweep_label)
    configs = []
    for cw in cw_extra:
        configs.append((cw, bw_default, 'CW_sweep'))
    for bw in bw_values:
        configs.append((cw_best, bw, 'BW_sweep'))
    configs.append((0.0, 0.0, 'bare'))

    # Deduplicate (CW=1.0, BW=5.0) appears in both sub-sweeps
    seen = set()
    unique_configs = []
    for cw, bw, label in configs:
        key = (cw, bw)
        if key not in seen:
            seen.add(key)
            unique_configs.append((cw, bw, label))

    total = len(unique_configs) * len(steps) * len(ranks)
    print(f"Total configs: {total} "
          f"({len(unique_configs)} weight combos × {len(steps)} T × {len(ranks)} ranks)")

    for cw, bw, label in unique_configs:
        for T in steps:
            for rank in ranks:
                i += 1
                rlabel = "full" if rank is None else f"r={rank}"
                schedule = 'inv_T' if T > 1 else 'constant'
                print(f"\n[{i}/{total}] CW={cw}, BW={bw} ({label}), "
                      f"T={T}, {rlabel}, schedule={schedule}")
                try:
                    res = run_exp_b(
                        n_steps=T, rank=rank, n_per_class=1, seed=seed,
                        run_baseline=(rank is None),
                        free_coefficients=True,
                        coeff_consistency_weight=cw,
                        coeff_box_weight=bw,
                        optimizer_type='sgd',
                        finetune_activation='leaky_relu',
                        lr_schedule=schedule,
                        extraction_epochs=extraction_epochs,
                        device=device, verbose=True,
                    )
                    row = extract_row_b(res, {
                        'track': 'B8', 'sub_sweep': label,
                        'optimizer': 'sgd',
                        'consistency_weight': cw,
                        'box_weight': bw,
                        'lr_schedule': schedule,
                    })
                    results_list.append(row)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    results_list.append({
                        'n_steps': T, 'rank': rank or 'full',
                        'consistency_weight': cw, 'box_weight': bw,
                        'sub_sweep': label,
                        'track': 'B8', 'error': str(e),
                    })

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'sprint2c_track_b8_{ts}.csv')
    save_csv(results_list, csv_path)
    return results_list


if __name__ == '__main__':
    from experiments.configs import get_device, ACTIVATION_CHOICES

    parser = argparse.ArgumentParser(description='Sprint 2c sweep')
    parser.add_argument('--track', type=str, nargs='+', default=['A'],
                        choices=['A', 'B2', 'B3a', 'B3b', 'B4', 'B5', 'B6', 'B7', 'B8'],
                        help='Which tracks to run')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--extraction_epochs', type=int, default=EXTRACTION_EPOCHS)
    parser.add_argument('--seed', type=int, default=42)
    # B3b options (depends on B3a results)
    parser.add_argument('--best_optimizer', type=str, default='lbfgs',
                        choices=['lbfgs', 'sgd', 'adam'],
                        help='Best optimizer from B3a (for B3b)')
    parser.add_argument('--best_activation', type=str, default=None,
                        choices=ACTIVATION_CHOICES,
                        help='Best activation from B3a (for B3b)')
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Using device: {device}")
    print(f"Extraction epochs: {args.extraction_epochs}")
    print(f"Tracks to run: {args.track}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    track_fns = {
        'A': lambda: run_track_a(device, args.extraction_epochs, args.seed),
        'B2': lambda: run_track_b2(device, args.extraction_epochs, args.seed),
        'B3a': lambda: run_track_b3a(device, args.extraction_epochs, args.seed),
        'B3b': lambda: run_track_b3b(device, args.extraction_epochs, args.seed,
                                     args.best_optimizer, args.best_activation),
        'B4': lambda: run_track_b4(device, args.extraction_epochs, args.seed),
        'B5': lambda: run_track_b5(device, args.extraction_epochs, args.seed),
        'B6': lambda: run_track_b6(device, args.extraction_epochs, args.seed),
        'B7': lambda: run_track_b7(device, args.extraction_epochs, args.seed),
        'B8': lambda: run_track_b8(device, args.extraction_epochs, args.seed),
    }

    for t in args.track:
        if t in track_fns:
            track_fns[t]()
        else:
            print(f"Unknown track: {t}")

    print("\n=== Sprint 2c sweep complete ===")
