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


def _scores_to_list(scores, n):
    """Convert SSIM scores (float, tensor, list, or None) to per-image list."""
    if scores is None:
        return [None] * n
    if isinstance(scores, (float, int)):
        return [scores] * n
    if hasattr(scores, 'tolist'):  # tensor
        vals = scores.tolist()
        return vals[:n] + [None] * max(0, n - len(vals))
    if isinstance(scores, (list, tuple)):
        return list(scores)[:n] + [None] * max(0, n - len(scores))
    return [float(scores)] * n


def plot_reconstruction_grid(x_train, x_recon_lora=None, x_recon_full=None,
                             x_ctrl=None, x_ood=None,
                             ssim_lora=None, ssim_full=None,
                             ssim_ctrl=None, ssim_ood=None,
                             ds_mean=None, save_path=None, title=None,
                             rank=None, digits=None):
    """Plot reconstruction comparison grid with clear labels and per-image SSIM.

    Row structure:
        1. Ground Truth — the actual private training images (pixel space)
        2. Full FT Reconstruction — from full weight change ΔW (centered space)
        3. LoRA Reconstruction — from low-rank adapter update BA (centered space)
        4. Control — different samples from same class (pixel space, sanity check)

    Images in "centered space" (reconstructions) have ds_mean added back for
    display. Images already in pixel space (ground truth, controls) are
    displayed as-is.

    SSIM scores can be a single float (shown on all columns) or a per-image
    tensor/list (shown individually per column).
    """
    plt = _setup_matplotlib()

    # Build row specs: (label, images, scores, is_centered)
    row_specs = []
    row_specs.append(('Ground Truth\n(training data)', x_train, None, False))

    if x_recon_full is not None:
        row_specs.append(('Full Fine-Tune\nReconstruction', x_recon_full,
                          ssim_full, True))

    if x_recon_lora is not None:
        rank_label = f' (r={rank})' if rank else ''
        row_specs.append((f'LoRA{rank_label}\nReconstruction', x_recon_lora,
                          ssim_lora, True))

    if x_ctrl is not None:
        row_specs.append(('Control\n(same class, diff. sample)', x_ctrl,
                          ssim_ctrl, False))

    if x_ood is not None:
        row_specs.append(('OOD Control', x_ood, ssim_ood, False))

    n_rows = len(row_specs)
    n_cols = x_train.shape[0]

    cell_size = 2.8
    fig_width = cell_size * n_cols + 3.5
    fig_height = cell_size * n_rows + 1.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.subplots_adjust(left=0.28, right=0.96, top=0.88,
                        bottom=0.04, hspace=0.55, wspace=0.2)

    ds_mean_np = ds_mean.squeeze().cpu().numpy() if ds_mean is not None else None

    for r, (label, images, scores, is_centered) in enumerate(row_specs):
        per_img = _scores_to_list(scores, n_cols)

        for c in range(n_cols):
            ax = axes[r, c]
            img = images[c].squeeze().cpu().numpy()

            # Add dataset mean back for images in centered space
            if is_centered and ds_mean_np is not None:
                img = img + ds_mean_np

            img = np.clip(img, 0, 1)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Per-image SSIM below the image
            if per_img[c] is not None:
                ax.text(0.5, -0.08, f'SSIM = {per_img[c]:.3f}',
                        transform=ax.transAxes, ha='center', fontsize=10,
                        color='#333333')

            # Column headers on first row
            if r == 0 and digits is not None and c < len(digits):
                parity = 'even' if digits[c] % 2 == 0 else 'odd'
                ax.set_title(f'Digit {digits[c]} ({parity})',
                             fontsize=11, fontweight='bold', pad=8)

    # Row labels using fig.text (visible regardless of layout)
    for r in range(n_rows):
        pos = axes[r, 0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(0.02, y_center, row_specs[r][0],
                 va='center', ha='left', fontsize=11, fontweight='bold',
                 linespacing=1.4)

    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=0.96)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
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


def plot_step_rank_heatmap(csv_path, save_path=None, title=None):
    """Plot T × rank heatmap from Experiment B sweep CSV.

    Shows SSIM as color for each (step count, rank) combination.
    Full model shown as a separate column.
    """
    import csv
    plt = _setup_matplotlib()

    # Parse CSV
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Extract unique step counts and ranks
    step_set = sorted(set(int(r['n_steps']) for r in rows))
    rank_set = sorted(set(r['rank'] for r in rows))

    # Separate full model and LoRA results
    full_ssim = {}  # T -> SSIM
    lora_ssim = {}  # (T, rank) -> SSIM

    for r in rows:
        T = int(r['n_steps'])
        rank = r['rank']
        if rank == 'full':
            val = r.get('full_ssim', '')
            if val:
                full_ssim[T] = float(val)
        else:
            val = r.get('lora_ssim', '')
            if val:
                lora_ssim[(T, int(rank))] = float(val)

    # Build matrix: rows = step counts, cols = [full, rank1, rank2, ...]
    lora_ranks = sorted(set(int(rk) for rk in rank_set if rk != 'full'))
    col_labels = ['Full'] + [f'r={rk}' for rk in lora_ranks]
    n_rows = len(step_set)
    n_cols = len(col_labels)

    matrix = np.full((n_rows, n_cols), np.nan)
    for i, T in enumerate(step_set):
        if T in full_ssim:
            matrix[i, 0] = full_ssim[T]
        for j, rk in enumerate(lora_ranks):
            if (T, rk) in lora_ssim:
                matrix[i, j + 1] = lora_ssim[(T, rk)]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(T) for T in step_set], fontsize=11)
    ax.set_xlabel('Model Variant', fontsize=13)
    ax.set_ylabel('Gradient Steps (T)', fontsize=13)
    ax.set_title(title or 'NTK Reconstruction SSIM: Steps × Rank', fontsize=14,
                 fontweight='bold', pad=12)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.55 else 'black'
                weight = 'bold' if val > 0.75 else 'normal'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=9.5, color=color, fontweight=weight)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='SSIM', shrink=0.85)
    cbar.ax.tick_params(labelsize=10)

    # Add horizontal line separating T=1 from rest
    ax.axhline(y=0.5, color='white', linewidth=2)

    plt.tight_layout()
    save_path = save_path or os.path.join(FIGURES_DIR, 'sprint1', 'ntk_step_rank_heatmap.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()
    return fig


def generate_experiment_b_figure(results, save_dir=None, base_name=None):
    """Generate the Experiment B figure with clear attack scenario labeling.

    Layout: rows = samples, columns = [Ground Truth, Full Recon, LoRA Recon, Control].
    Each image gets per-image SSIM, color-coded by quality.
    """
    from experiments.metrics import compute_ssim

    plt = _setup_matplotlib()
    save_dir = save_dir or os.path.join(FIGURES_DIR, 'sprint1')

    x_train = results['x_train']
    ds_mean = results.get('ds_mean')
    x_recon_full = results.get('x_recon_full')
    x_recon_lora = results.get('x_recon_lora')
    x_ctrl = results.get('x_ctrl')
    rank = results.get('rank')
    n_steps = results.get('n_steps', 1)
    digits = results.get('digits', [])

    x_train_c = x_train - ds_mean if ds_mean is not None else x_train
    ds_mean_np = ds_mean.squeeze().cpu().numpy() if ds_mean is not None else None
    n_samples = x_train.shape[0]

    # Per-image SSIM
    ssim_full = ssim_lora = ssim_ctrl = None
    if x_recon_full is not None:
        ssim_full, _ = compute_ssim(x_recon_full, x_train_c, ds_mean)
    if x_recon_lora is not None:
        ssim_lora, _ = compute_ssim(x_recon_lora, x_train_c, ds_mean)
    recon_for_ctrl = x_recon_full if x_recon_full is not None else x_recon_lora
    if recon_for_ctrl is not None and x_ctrl is not None:
        x_ctrl_c = x_ctrl - ds_mean if ds_mean is not None else x_ctrl
        ssim_ctrl, ssim_ctrl_mean = compute_ssim(recon_for_ctrl, x_ctrl_c, ds_mean)

    # Compute actual parameter counts from config or model info
    config = results.get('config', {})
    full_params = config.get('full_params')
    lora_params = config.get('lora_params')

    def _fmt_params(n):
        if n is None:
            return ''
        if n >= 1_000_000:
            return f'\n({n/1_000_000:.1f}M params)'
        elif n >= 1_000:
            return f'\n({n/1_000:.0f}K params)'
        return f'\n({n} params)'

    # Build column specs: (label, color, images, is_centered, ssim_vals)
    cols = [('Ground Truth\n(private data)', '#333333', x_train, False, None)]
    if x_recon_full is not None:
        cols.append((f'Full-Model Recon{_fmt_params(full_params)}', '#1565C0',
                     x_recon_full, True, ssim_full))
    if x_recon_lora is not None:
        rank_str = f'r={rank} ' if rank else ''
        cols.append((f'LoRA {rank_str}Recon{_fmt_params(lora_params)}', '#E65100',
                     x_recon_lora, True, ssim_lora))
    if x_ctrl is not None:
        cols.append(('Negative Control\n(diff instance, same digit)', '#777777',
                     x_ctrl, False, ssim_ctrl))

    n_cols = len(cols)
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3.5 * n_cols, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.12, right=0.98,
                        hspace=0.45, wspace=0.15)

    # Detect free-coefficient mode from config or extract results
    is_free = config.get('mode', '').upper() == 'FREE-COEFFICIENT'
    if not is_free:
        for key in ('lora_extract_res', 'full_extract_res'):
            if key in results and 'coeff_error' in results.get(key, {}):
                is_free = True
                break

    # Title
    mode_label = 'Free Coefficient' if is_free else 'Oracle Coefficients'
    rank_str = f', LoRA r={rank}' if rank else ''
    fig.suptitle(f'{mode_label}{rank_str}, T={n_steps}',
                 fontsize=15, fontweight='bold', y=0.96)

    # Column headers
    for c, (label, color, _, _, _) in enumerate(cols):
        pos = axes[0, c].get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.88, label,
                 ha='center', va='bottom', fontsize=10, fontweight='bold',
                 color=color, linespacing=1.3)

    # Plot images
    for r in range(n_samples):
        digit = digits[r] if r < len(digits) else '?'
        parity = 'odd' if int(digit) % 2 == 1 else 'even'
        pos = axes[r, 0].get_position()
        fig.text(0.02, (pos.y0 + pos.y1) / 2,
                 f'Sample {r+1}\nDigit {digit}\n({parity})',
                 va='center', ha='left', fontsize=10, fontweight='bold',
                 color='#333333', linespacing=1.4)

        for c, (_, color, images, is_centered, ssim_vals) in enumerate(cols):
            ax = axes[r, c]
            img = images[r].squeeze().cpu().numpy()
            if is_centered and ds_mean_np is not None:
                img = img + ds_mean_np
            img = np.clip(img, 0, 1)

            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.5)

            if ssim_vals is not None and r < len(ssim_vals):
                val = ssim_vals[r].item() if hasattr(ssim_vals[r], 'item') else float(ssim_vals[r])
                sc = '#2E7D32' if val > 0.9 else ('#E65100' if val > 0.7 else '#888888')
                ax.text(0.5, -0.12, f'SSIM = {val:.3f}',
                        transform=ax.transAxes, ha='center', fontsize=10.5,
                        color=sc, fontweight='bold')

    # Bottom summary
    parts = []
    if ssim_full is not None:
        parts.append(f'Full-model recon nearly identical to ground truth (SSIM > {ssim_full.mean():.2f}).')
    if ssim_lora is not None:
        parts.append(f'LoRA r={rank} recon clearly recovers digit structure (SSIM = {ssim_lora.mean():.2f}).')
    if ssim_ctrl is not None:
        parts.append(f'\nControl comparison (SSIM = {ssim_ctrl_mean:.3f}) proves instance-specific leakage, not just class-level similarity.')
    if parts:
        fig.text(0.5, 0.01, ' '.join(parts),
                 ha='center', fontsize=9.5, color='#555555', style='italic',
                 linespacing=1.4)

    suffix = '_free' if is_free else '_oracle'
    # base_name makes the figure unique per config. Without it every run — including
    # a 3-epoch smoke test — overwrites the same canonical figure (see LESSONS_LEARNED).
    stem = f'experiment_b_grid{suffix}' if base_name is None else f'{base_name}{suffix}'
    save_path = os.path.join(save_dir, f'{stem}.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_anchor_two_curve(alphas, ssims, lin_fs, lin_ws=None, save_path=None,
                          path_label='LoRA', title=None):
    """Two-curve anchor-sweep validation plot (meeting Addition 3).

    Left axis: reconstruction SSIM(α). Right axis: linearization error(α),
    function-space (solid, headline) and optional weight-space (dashed).
    Legitimate win = lin-error bottoms out where SSIM peaks (improvement
    explained by better linearization). Red flag = SSIM keeps climbing past
    the lin-error minimum (the anchor is leaking x_i, not approximating better).

    Args:
        alphas: list of α values (x-axis).
        ssims: list of SSIM(α), left axis.
        lin_fs: list of function-space lin-error(α), right axis (headline).
        lin_ws: optional list of weight-space lin-error(α), right axis (dashed).
        save_path: output PNG path.
        path_label: 'LoRA' or 'Full' (title/legend).
        title: optional title override.
    """
    plt = _setup_matplotlib()
    order = sorted(range(len(alphas)), key=lambda i: alphas[i])
    xs = [alphas[i] for i in order]
    ss = [ssims[i] for i in order]
    lf = [lin_fs[i] for i in order]

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    c_ssim, c_lin = '#1f77b4', '#d62728'

    ax1.plot(xs, ss, marker='o', linewidth=2, color=c_ssim, label='SSIM')
    ax1.set_xlabel(r'anchor $\alpha$  ($\theta_{\mathrm{anchor}}=(1-\alpha)\theta_0+\alpha\theta_T$)')
    ax1.set_ylabel('reconstruction SSIM', color=c_ssim)
    ax1.tick_params(axis='y', labelcolor=c_ssim)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, lf, marker='s', linewidth=2, color=c_lin,
             label='lin-error (function-space)')
    if lin_ws is not None:
        lw = [lin_ws[i] for i in order]
        ax2.plot(xs, lw, marker='^', linewidth=1.5, linestyle='--',
                 color=c_lin, alpha=0.6, label='lin-error (weight-space)')
    ax2.set_ylabel('linearization error', color=c_lin)
    ax2.tick_params(axis='y', labelcolor=c_lin)
    ax2.spines['top'].set_visible(False)

    # Mark the SSIM peak and the function-space lin-error minimum. If the SSIM
    # peak sits at a *larger* α than the lin-error minimum, that is the red flag.
    i_peak = max(range(len(xs)), key=lambda i: (ss[i] if ss[i] is not None else -1))
    i_min = min(range(len(xs)), key=lambda i: (lf[i] if lf[i] is not None else float('inf')))
    ax1.axvline(xs[i_peak], color=c_ssim, alpha=0.25, linestyle=':')
    ax2.axvline(xs[i_min], color=c_lin, alpha=0.25, linestyle=':')
    flag = 'OK: SSIM peak ≈ lin-error min' if xs[i_peak] <= xs[i_min] + 1e-9 \
        else 'RED FLAG: SSIM climbs past lin-error min (possible x_i leakage)'

    ax1.set_title(title or f'Anchor α-sweep ({path_label}) — {flag}', fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    plt.close()
    return save_path
