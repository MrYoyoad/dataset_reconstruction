"""Phase 0 D2: Post-hoc comparison of sweep results.

Generates:
  1. Summary CSV sorted by SSIM
  2. Heatmap: tv_weight x lr, panels for 10K / 30K iters
  3. Top-5 comparison figure (GT + 5 best reconstructions)
  4. Cos_sim overlay for top-5

Usage:
    python -m experiments.phase0_d2_compare
    # or: bash scripts/run_phase0_d2_wexac.sh compare
"""

import os
import glob
import csv
import torch
import numpy as np
from datetime import datetime


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures',
                               'phase0', 'd2_sweep')
    os.makedirs(figures_dir, exist_ok=True)

    # Find all D2 result files
    pths = sorted(glob.glob(os.path.join(results_dir, 'phase0_d2_*.pth')))
    if not pths:
        print("No phase0_d2_*.pth files found in results/")
        return

    # Load all results
    all_results = []
    for path in pths:
        d = torch.load(path, map_location='cpu', weights_only=False)
        m = d['metrics']
        all_results.append({
            'path': path,
            'config_name': d.get('d2_config_name', os.path.basename(path)),
            'config_index': d.get('d2_config_index', -1),
            'tv_weight': d.get('tv_weight', -1),
            'lr': d.get('lr', -1),
            'n_iters': d.get('n_iters', -1),
            'n_restarts': d.get('n_restarts', -1),
            'ssim': m['ssim'],
            'psnr': m['psnr'],
            'mse': m['mse'],
            'best_cos_sim': m.get('best_cos_sim', -1),
            'inversion_time': d.get('inversion_time', -1),
            'freq_weight': d.get('freq_weight', 0),
            'lpips_weight': d.get('lpips_weight', 0),
            'x_true': d['x_true'],
            'x_recon': d['x_recon'],
            'loss_history': d.get('loss_history'),
        })

    # Keep latest result per (tv_weight, lr, n_iters) combo
    seen = {}
    for r in all_results:
        key = (r['tv_weight'], r['lr'], r['n_iters'])
        seen[key] = r  # last one wins (sorted by filename = timestamp)
    unique = sorted(seen.values(), key=lambda r: -r['ssim'])

    # Print summary
    print(f"\n{'='*80}")
    print(f"D2 SWEEP RESULTS ({len(unique)} configs)")
    print(f"{'='*80}")
    print(f"{'Rank':<5s} {'Config':<30s} {'TV':<8s} {'LR':<6s} "
          f"{'Iters':<7s} {'SSIM':>7s} {'PSNR':>6s} {'CosSim':>7s}")
    print("-" * 80)
    for i, r in enumerate(unique):
        print(f"{i+1:<5d} {r['config_name']:<30s} {r['tv_weight']:<8.0e} "
              f"{r['lr']:<6.3f} {r['n_iters']:<7d} {r['ssim']:>7.4f} "
              f"{r['psnr']:>6.1f} {r['best_cos_sim']:>7.4f}")

    # Save CSV
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(results_dir, f'phase0_d2_comparison_{ts}.csv')
    fields = ['config_name', 'config_index', 'tv_weight', 'lr', 'n_iters',
              'n_restarts', 'ssim', 'psnr', 'mse', 'best_cos_sim',
              'inversion_time', 'freq_weight', 'lpips_weight']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in unique:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nCSV: {csv_path}")

    # Generate figures
    _save_heatmap(unique, figures_dir)
    _save_top_comparison(unique[:5], figures_dir)
    _save_cossim_overlay(unique[:5], figures_dir)
    _save_cossim_overlay_by_tv(unique, figures_dir)


def _save_heatmap(results, figures_dir):
    """SSIM heatmap: tv_weight (y) x lr (x), separate panels for 10K/30K."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from experiments.phase0_vit_inversion import D2_TV_WEIGHTS, D2_LRS, D2_ITERS

    fig, axes = plt.subplots(1, len(D2_ITERS), figsize=(7 * len(D2_ITERS), 5))
    if len(D2_ITERS) == 1:
        axes = [axes]

    for panel_idx, n_iters in enumerate(D2_ITERS):
        ax = axes[panel_idx]
        grid = np.full((len(D2_TV_WEIGHTS), len(D2_LRS)), np.nan)

        for r in results:
            if r['n_iters'] != n_iters:
                continue
            try:
                tv_idx = D2_TV_WEIGHTS.index(r['tv_weight'])
                lr_idx = D2_LRS.index(r['lr'])
                grid[tv_idx, lr_idx] = r['ssim']
            except ValueError:
                continue

        im = ax.imshow(grid, cmap='RdYlGn', aspect='auto',
                       vmin=0, vmax=max(0.3, np.nanmax(grid) * 1.1))
        ax.set_xticks(range(len(D2_LRS)))
        ax.set_xticklabels([f'{lr}' for lr in D2_LRS])
        ax.set_yticks(range(len(D2_TV_WEIGHTS)))
        ax.set_yticklabels([f'{tv:.0e}' for tv in D2_TV_WEIGHTS])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('TV Weight')
        ax.set_title(f'{n_iters//1000}K iterations')

        # Annotate cells
        for tv_i in range(len(D2_TV_WEIGHTS)):
            for lr_i in range(len(D2_LRS)):
                val = grid[tv_i, lr_i]
                if not np.isnan(val):
                    color = 'white' if val < 0.1 else 'black'
                    ax.text(lr_i, tv_i, f'{val:.3f}', ha='center',
                            va='center', fontsize=9, fontweight='bold',
                            color=color)

        plt.colorbar(im, ax=ax, label='SSIM', shrink=0.8)

    fig.suptitle('D2 Sweep: SSIM Heatmap (signAdam, Flowers102)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(figures_dir, 'phase0_d2_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def _save_top_comparison(top_results, figures_dir):
    """GT + top-N reconstructions side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    denorm_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    denorm_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    n = len(top_results)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    gt = top_results[0]['x_true']
    gt_np = (gt * denorm_std + denorm_mean).clamp(0, 1)[0].numpy().transpose(1, 2, 0)
    axes[0].imshow(gt_np)
    axes[0].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for i, r in enumerate(top_results):
        recon_np = (r['x_recon'] * denorm_std + denorm_mean).clamp(0, 1)[0] \
                   .numpy().transpose(1, 2, 0)
        axes[i + 1].imshow(recon_np)
        axes[i + 1].set_title(
            f"#{i+1}: tv={r['tv_weight']:.0e} lr={r['lr']}\n"
            f"SSIM={r['ssim']:.3f}  cos={r['best_cos_sim']:.3f}",
            fontsize=9, fontweight='bold' if i == 0 else 'normal',
            color='#1a7a2e' if i == 0 else 'black'
        )
        axes[i + 1].axis('off')

    fig.suptitle(f'D2 Top-{n} Reconstructions (signAdam, full gradient)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(figures_dir, 'phase0_d2_top5_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def _save_cossim_overlay(top_results, figures_dir):
    """Cos_sim convergence overlay for top-N configs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, r in enumerate(top_results):
        lh = r.get('loss_history')
        if not lh:
            continue
        cos_curves = lh['cos_sim']
        best_r = max(range(len(cos_curves)),
                     key=lambda ri: max(cos_curves[ri]))
        c = colors[i % len(colors)]
        label = f"tv={r['tv_weight']:.0e} lr={r['lr']} (SSIM={r['ssim']:.3f})"

        ax1.plot(cos_curves[best_r], color=c, linewidth=1.2, label=label)
        total_curves = lh['total']
        ax2.plot(total_curves[best_r], color=c, linewidth=1.2, label=label)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Best Restart: Cos Sim')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Best Restart: Total Loss')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(figures_dir, 'phase0_d2_cossim_overlay.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def _save_cossim_overlay_by_tv(results, figures_dir):
    """Cos_sim overlay with one curve per TV level (best config at that TV).

    D2's dominant lever is tv_weight, so this view shows convergence
    behavior for each qualitatively-different regime — the analog of
    the D1 overlay (which had 4 curves for 4 qualitatively-different
    optimizer × TV combinations).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.phase0_vit_inversion import D2_TV_WEIGHTS

    # Best result per TV level (highest SSIM, must have loss_history)
    best_per_tv = {}
    for r in results:
        if not r.get('loss_history'):
            continue
        tv = r['tv_weight']
        if tv not in best_per_tv or r['ssim'] > best_per_tv[tv]['ssim']:
            best_per_tv[tv] = r

    # Keep TV order consistent with D2_TV_WEIGHTS (low → high)
    ordered = [best_per_tv[tv] for tv in D2_TV_WEIGHTS if tv in best_per_tv]
    if not ordered:
        print("No D2 results with loss_history — skipping by-TV overlay")
        return

    # Cool→warm gradient so reader sees TV increasing visually
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, len(ordered) - 1)) for i in range(len(ordered))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for r, c in zip(ordered, colors):
        cos_curves = r['loss_history']['cos_sim']
        best_r = max(range(len(cos_curves)),
                     key=lambda ri: max(cos_curves[ri]))
        label = (f"tv={r['tv_weight']:.0e}  best lr={r['lr']}  "
                 f"(SSIM={r['ssim']:.3f})")
        ax1.plot(cos_curves[best_r], color=c, linewidth=1.4, label=label)
        ax2.plot(r['loss_history']['total'][best_r], color=c, linewidth=1.4,
                 label=label)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Best Restart per TV Level: Cos Sim Convergence')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Best Restart per TV Level: Total Loss Convergence')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(figures_dir, 'phase0_d2_cossim_overlay_by_tv.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
