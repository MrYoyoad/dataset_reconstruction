"""Phase 0 D1: Post-hoc comparison of 4 independent config runs.

After the 4 D1 jobs complete (A/B/C/D), this script loads the .pth files,
generates the combined comparison figure and cos_sim overlay.

Usage:
    python -m experiments.phase0_d1_compare
    # or: bash scripts/run_phase0_d1_wexac.sh compare
"""

import os
import sys
import glob
import csv
import torch
import numpy as np
from datetime import datetime


def find_latest_pth(pattern):
    """Find the most recent .pth file matching a glob pattern."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[-1]  # sorted by name → latest timestamp last


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures', 'phase0')
    os.makedirs(figures_dir, exist_ok=True)

    # Expected config naming from run_phase0: phase0_full_r8_n1_s42_*.pth
    # We need to identify which .pth belongs to which config.
    # The individual jobs save as standard phase0_full_*.pth — we'll use the
    # 'optimizer' and 'tv_weight' fields inside the .pth to identify them.

    # Find all Phase 0 full-model results from today or recent
    all_pths = sorted(glob.glob(os.path.join(results_dir, 'phase0_full_r8_n1_s42_*.pth')))

    if not all_pths:
        print("No phase0_full_r8_n1_s42_*.pth files found in results/")
        sys.exit(1)

    # Load all and group by (optimizer, tv_weight)
    configs = {}
    for path in all_pths:
        data = torch.load(path, map_location='cpu', weights_only=False)
        opt = data.get('optimizer', 'unknown')
        tv = data.get('tv_weight', -1)
        key = (opt, tv)
        # Keep the latest run per config
        configs[key] = (path, data)

    # Map to D1 config names
    config_map = {
        ('Adam', 1e-4): 'A_adam_tv1e-4',
        ('signAdam', 1e-4): 'B_signAdam_tv1e-4',
        ('Adam', 1e-2): 'C_adam_tv1e-2',
        ('signAdam', 1e-2): 'D_signAdam_tv1e-2',
    }

    d1_results = []
    missing = []
    for key, name in config_map.items():
        if key in configs:
            path, data = configs[key]
            metrics = data['metrics']
            d1_results.append({
                'name': name,
                'optimizer': key[0],
                'tv_weight': key[1],
                'ssim': metrics['ssim'],
                'psnr': metrics['psnr'],
                'mse': metrics['mse'],
                'best_cos_sim': metrics.get('best_cos_sim', -1),
                'inversion_time': data.get('inversion_time', -1),
                'x_recon': data['x_recon'],
                'x_true': data['x_true'],
                'loss_history': data.get('loss_history', None),
                'path': path,
            })
            print(f"  {name}: loaded from {os.path.basename(path)}")
        else:
            missing.append(name)

    if missing:
        print(f"\nWARNING: Missing configs: {missing}")
        print(f"Available (optimizer, tv_weight) combos: {list(configs.keys())}")
        if not d1_results:
            sys.exit(1)

    # Sort by config name
    d1_results.sort(key=lambda r: r['name'])

    # Print summary table
    print(f"\n{'='*70}")
    print("D1 COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'Optimizer':<10s} {'TV':<8s} "
          f"{'SSIM':>6s} {'PSNR':>6s} {'CosSim':>7s} {'Time':>6s}")
    print("-" * 70)
    for r in d1_results:
        t = f"{r['inversion_time']:.0f}s" if r['inversion_time'] > 0 else "N/A"
        print(f"{r['name']:<25s} {r['optimizer']:<10s} {r['tv_weight']:<8.0e} "
              f"{r['ssim']:>6.4f} {r['psnr']:>6.1f} "
              f"{r['best_cos_sim']:>7.4f} {t:>6s}")

    # Save CSV
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(results_dir, f'phase0_d1_comparison_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        fields = ['name', 'optimizer', 'tv_weight', 'ssim', 'psnr', 'mse',
                  'best_cos_sim', 'inversion_time']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in d1_results:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nCSV: {csv_path}")

    # Generate comparison figure
    _save_comparison_figure(d1_results, figures_dir)

    # Generate cos_sim overlay (if loss_history available)
    has_history = all(r['loss_history'] is not None for r in d1_results)
    if has_history:
        _save_cossim_overlay(d1_results, figures_dir)
    else:
        print("Skipping cos_sim overlay (some configs missing loss_history)")


def _save_comparison_figure(results, figures_dir):
    """GT + N reconstructions side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    denorm_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    denorm_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    gt = results[0]['x_true']
    gt_np = (gt * denorm_std + denorm_mean).clamp(0, 1)[0].numpy().transpose(1, 2, 0)

    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    axes[0].imshow(gt_np)
    axes[0].set_title('Ground Truth', fontsize=11)
    axes[0].axis('off')

    for i, r in enumerate(results):
        recon_np = (r['x_recon'] * denorm_std + denorm_mean).clamp(0, 1)[0] \
                   .numpy().transpose(1, 2, 0)
        axes[i + 1].imshow(recon_np)
        cos_str = f"{r['best_cos_sim']:.3f}" if r['best_cos_sim'] > 0 else "N/A"
        axes[i + 1].set_title(
            f"{r['name']}\nSSIM={r['ssim']:.3f}  cos={cos_str}",
            fontsize=9
        )
        axes[i + 1].axis('off')

    fig.suptitle('Phase 0 D1: Optimizer x TV Comparison (full-model gradient)',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(figures_dir, 'phase0_d1_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def _save_cossim_overlay(results, figures_dir):
    """Overlay cos_sim + total loss curves from best restart of each config."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, r in enumerate(results):
        cos_curves = r['loss_history']['cos_sim']
        best_r = max(range(len(cos_curves)),
                     key=lambda ri: max(cos_curves[ri]))
        c = colors[i % len(colors)]

        ax1.plot(cos_curves[best_r], color=c, linewidth=1.2,
                 label=f"{r['name']} (best={max(cos_curves[best_r]):.4f})")

        total_curves = r['loss_history']['total']
        ax2.plot(total_curves[best_r], color=c, linewidth=1.2,
                 label=r['name'])

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Best Restart: Cos Sim Convergence')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Best Restart: Total Loss Convergence')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(figures_dir, 'phase0_d1_cossim_overlay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
