"""Figure generation for LoRA reconstruction experiments.

Generates publication-quality plots for the thesis and supervisor email.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from experiments.configs import FIGURES_DIR


def _setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    return plt


def plot_reconstruction_grid(x_train, x_recon_lora, x_recon_full=None,
                             x_ctrl=None, x_ood=None,
                             ssim_lora=None, ssim_full=None,
                             ssim_ctrl=None, ssim_ood=None,
                             ds_mean=None, save_path=None, title=None):
    """Plot reconstruction comparison grid.

    Rows: Original | Full FT recon | LoRA recon | In-dist control | OOD control
    """
    plt = _setup_matplotlib()

    rows = ['Training\n(ground truth)']
    images = [x_train]
    scores = [None]

    if x_recon_full is not None:
        rows.append('Full FT\nreconstruction')
        images.append(x_recon_full)
        scores.append(ssim_full)

    rows.append('LoRA (r=8)\nreconstruction')
    images.append(x_recon_lora)
    scores.append(ssim_lora)

    if x_ctrl is not None:
        rows.append('In-dist\ncontrol')
        images.append(x_ctrl)
        scores.append(ssim_ctrl)

    if x_ood is not None:
        rows.append('OOD\ncontrol')
        images.append(x_ood)
        scores.append(ssim_ood)

    n_rows = len(rows)
    n_cols = images[0].shape[0]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r in range(n_rows):
        for c in range(n_cols):
            img = images[r][c].squeeze().cpu().numpy()
            if ds_mean is not None and r == 0:
                img = img + ds_mean.squeeze().cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[r, c].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[r, c].axis('off')
            if c == 0:
                axes[r, c].set_ylabel(rows[r], rotation=0, labelpad=80, va='center')
            if scores[r] is not None:
                axes[r, c].set_title(f'SSIM={scores[r]:.3f}', fontsize=10)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()
    return fig


def plot_ntk_diagnostics(step_counts, ssim_values, feature_stability,
                         coefficient_drift, weight_change=None,
                         save_path=None, title=None):
    """Plot NTK diagnostic figure: SSIM + stability + drift vs step count T.

    This is the key figure showing WHY reconstruction degrades.
    """
    plt = _setup_matplotlib()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Panel 1: Reconstruction SSIM
    ax1.plot(step_counts, ssim_values, 'o-', color='#2196F3', linewidth=2, markersize=6)
    ax1.set_ylabel('SSIM')
    ax1.set_title('Reconstruction Quality' if not title else title)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Feature stability
    ax2.plot(step_counts, feature_stability, 's-', color='#4CAF50', linewidth=2, markersize=6)
    ax2.set_ylabel('Feature Cosine Sim')
    ax2.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, label='NTK threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Coefficient drift
    ax3.plot(step_counts, coefficient_drift, '^-', color='#FF9800', linewidth=2, markersize=6)
    ax3.set_ylabel('Coefficient Drift')
    ax3.set_xlabel('Number of gradient steps T')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()
    return fig


def plot_rank_sweep(ranks, ssim_values, labels=None,
                    baseline_ssim=None, control_ssim=None,
                    save_path=None, title=None):
    """Plot SSIM vs LoRA rank.

    Multiple lines for different experiments/step counts.
    """
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))

    if isinstance(ssim_values[0], (list, np.ndarray)):
        for i, (ssim, label) in enumerate(zip(ssim_values, labels or [f'Config {i}' for i in range(len(ssim_values))])):
            ax.plot(ranks, ssim, 'o-', linewidth=2, markersize=6, label=label)
    else:
        ax.plot(ranks, ssim_values, 'o-', linewidth=2, markersize=6,
                label=labels[0] if labels else 'LoRA')

    if baseline_ssim is not None:
        ax.axhline(y=baseline_ssim, color='red', linestyle='--', alpha=0.7,
                    label=f'Full FT baseline ({baseline_ssim:.3f})')

    if control_ssim is not None:
        ax.axhline(y=control_ssim, color='gray', linestyle=':', alpha=0.5,
                    label=f'Control ({control_ssim:.3f})')

    ax.set_xlabel('LoRA Rank')
    ax.set_xscale('log', base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_ylabel('SSIM')
    ax.set_title(title or 'Reconstruction Quality vs LoRA Rank')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()
    return fig


def plot_rank_n_heatmap(ranks, ns, ssim_matrix,
                        save_path=None, title=None):
    """Plot rank × N heatmap for Experiment A."""
    plt = _setup_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ssim_matrix, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([str(n) for n in ns])
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('Samples per Class (N)')
    ax.set_title(title or 'Reconstruction SSIM: Rank × N')

    # Add text annotations
    for i in range(len(ns)):
        for j in range(len(ranks)):
            text = ax.text(j, i, f'{ssim_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=10)

    plt.colorbar(im, ax=ax, label='SSIM')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close()
    return fig


def generate_experiment_b_figure(results, save_dir=None):
    """Generate the Experiment B reconstruction grid from results dict."""
    save_dir = save_dir or FIGURES_DIR
    x_train = results['x_train']
    ds_mean = results.get('ds_mean')

    x_recon_full = results.get('x_recon_full')
    x_recon_lora = results.get('x_recon_lora')

    ssim_full = results.get('full_metrics', {}).get('ssim')
    ssim_lora = results.get('lora_metrics', {}).get('ssim')
    ctrl_ssim = results.get('control_metrics', {}).get('ssim')

    plot_reconstruction_grid(
        x_train=x_train,
        x_recon_lora=x_recon_lora if x_recon_lora is not None else x_recon_full,
        x_recon_full=x_recon_full,
        x_ctrl=results.get('x_ctrl'),
        ssim_lora=ssim_lora,
        ssim_full=ssim_full,
        ssim_ctrl=ctrl_ssim,
        ds_mean=ds_mean,
        save_path=os.path.join(save_dir, 'experiment_b_grid.png'),
        title='Experiment B: 1-Step NTK Reconstruction',
    )
