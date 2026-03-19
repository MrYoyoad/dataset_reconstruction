"""Re-run best-rank extraction for each T and save image tensors + PDF.

Reads the sweep CSV to find the best LoRA rank per T, re-runs extraction
for those configs, saves all image tensors to a .pth, and generates a PDF
grid of examples.

Usage (on WEXAC):
    python -m experiments.gen_t_sweep_examples --device cuda
    python -m experiments.gen_t_sweep_examples --device cuda --csv results/experiment_b_sweep_20260222_115334.csv
"""
import sys
import os
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

import torch
import numpy as np

torch.set_default_dtype(torch.float64)

from experiments.configs import RESULTS_DIR, FIGURES_DIR, EXTRACTION_EPOCHS


def parse_best_per_t(csv_path):
    """Parse sweep CSV and return best LoRA rank per T."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    lora_ssim = {}
    full_ssim = {}
    for r in rows:
        T = int(r['n_steps'])
        rank = r['rank']
        if rank == 'full':
            if r.get('full_ssim'):
                full_ssim[T] = float(r['full_ssim'])
        else:
            if r.get('lora_ssim'):
                lora_ssim[(T, int(rank))] = float(r['lora_ssim'])

    step_set = sorted(set(int(r['n_steps']) for r in rows))
    lora_ranks = sorted(set(int(rk) for rk in set(r['rank'] for r in rows) if rk != 'full'))

    best_per_t = {}
    for T in step_set:
        best_val, best_r = -1, None
        for rk in lora_ranks:
            v = lora_ssim.get((T, rk), -1)
            if v > best_val:
                best_val, best_r = v, rk
        best_per_t[T] = (best_r, best_val)

    return step_set, best_per_t, full_ssim


def run_all_extractions(step_set, best_per_t, device, extraction_epochs):
    """Run extraction for each T with its best rank. Returns dict of results."""
    from experiments.run_experiment_b import run_single_config

    all_results = {}
    for i, T in enumerate(step_set):
        best_rank = best_per_t[T][0]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(step_set)}] T={T}, best LoRA rank={best_rank}")
        print(f"{'='*60}")

        results = run_single_config(
            n_steps=T, rank=best_rank, n_per_class=1, seed=42,
            run_baseline=True,
            extraction_epochs=extraction_epochs,
            optimizer_type='lbfgs',
            device=device, verbose=True,
        )
        all_results[T] = results

        # Progress report
        fm = results.get('full_metrics', {})
        lm = results.get('lora_metrics', {})
        full_s = fm.get('ssim', None)
        lora_s = lm.get('ssim', None)
        full_str = f"{full_s:.4f}" if full_s is not None else "N/A"
        lora_str = f"{lora_s:.4f}" if lora_s is not None else "N/A"
        print(f"  Full SSIM={full_str}, LoRA r={best_rank} SSIM={lora_str}")

    return all_results


def save_results(all_results, best_per_t, save_path):
    """Save image tensors and metrics for all T values to a .pth file."""
    save_dict = {}
    for T, res in all_results.items():
        entry = {
            'x_train': res['x_train'].cpu(),
            'ds_mean': res.get('ds_mean', torch.zeros(1)).cpu(),
            'digits': res.get('digits', []),
            'best_rank': best_per_t[T][0],
        }
        if 'x_recon_full' in res:
            entry['x_recon_full'] = res['x_recon_full'].cpu()
            entry['full_metrics'] = res.get('full_metrics', {})
        if 'x_recon_lora' in res:
            entry['x_recon_lora'] = res['x_recon_lora'].cpu()
            entry['lora_metrics'] = res.get('lora_metrics', {})
        if 'x_ctrl' in res:
            entry['x_ctrl'] = res['x_ctrl'].cpu()
            entry['control_metrics'] = res.get('control_metrics', {})
        save_dict[T] = entry

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    print(f"Saved results to {save_path}")


def generate_pdf(all_results, best_per_t, full_ssim_sweep, step_set, pdf_path):
    """Generate a multi-page PDF with reconstruction examples for each T."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'sans-serif',
    })

    def to_display(img_tensor, ds_mean=None, centered=False):
        img = img_tensor.squeeze().cpu().numpy()
        if centered and ds_mean is not None:
            img = img + ds_mean.squeeze().cpu().numpy()
        return np.clip(img, 0, 1)

    n_T = len(step_set)
    n_samples = 2

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        # ── Page 1: Main grid ──
        fig, axes = plt.subplots(n_T, 2 * 3, figsize=(15, 2.5 * n_T))

        fig.suptitle('Best Reconstruction per Training Step Count (T)\n'
                     'Oracle coefficients, seed=42, MNIST odd/even',
                     fontsize=14, fontweight='bold', y=0.995)

        col_headers = ['Ground Truth', 'Full Model\nReconstruction',
                       'Best LoRA\nReconstruction']
        for s in range(n_samples):
            for c, header in enumerate(col_headers):
                axes[0, s * 3 + c].set_title(header, fontsize=10,
                                              fontweight='bold', pad=6)

        first_res = all_results[step_set[0]]
        digits = first_res.get('digits', ['?', '?'])
        fig.text(0.25, 0.98, f'Sample 1 (digit {digits[0]})',
                 ha='center', fontsize=11, fontweight='bold', color='#1565C0')
        fig.text(0.75, 0.98, f'Sample 2 (digit {digits[1]})',
                 ha='center', fontsize=11, fontweight='bold', color='#E65100')

        for row_idx, T in enumerate(step_set):
            res = all_results[T]
            ds_mean = res.get('ds_mean')
            best_rank = best_per_t[T][0]

            x_train = res['x_train']
            x_recon_full = res.get('x_recon_full')
            x_recon_lora = res.get('x_recon_lora')
            fm = res.get('full_metrics', {})
            lm = res.get('lora_metrics', {})

            for s in range(n_samples):
                # Ground truth
                ax = axes[row_idx, s * 3]
                ax.imshow(to_display(x_train[s]), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])

                # Full model recon
                ax = axes[row_idx, s * 3 + 1]
                if x_recon_full is not None:
                    ax.imshow(to_display(x_recon_full[s], ds_mean, centered=True),
                              cmap='gray', vmin=0, vmax=1)
                    ssim_val = fm.get('ssim')
                    if ssim_val is not None:
                        ax.text(0.5, -0.15, f'SSIM={ssim_val:.3f}',
                                transform=ax.transAxes, ha='center', fontsize=8,
                                color='#1565C0', fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                            ha='center', va='center')
                ax.set_xticks([]); ax.set_yticks([])

                # Best LoRA recon
                ax = axes[row_idx, s * 3 + 2]
                if x_recon_lora is not None:
                    ax.imshow(to_display(x_recon_lora[s], ds_mean, centered=True),
                              cmap='gray', vmin=0, vmax=1)
                    ssim_val = lm.get('ssim')
                    if ssim_val is not None:
                        ax.text(0.5, -0.15, f'r={best_rank}, SSIM={ssim_val:.3f}',
                                transform=ax.transAxes, ha='center', fontsize=8,
                                color='#E65100', fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                            ha='center', va='center')
                ax.set_xticks([]); ax.set_yticks([])

            # Row label
            axes[row_idx, 0].set_ylabel(f'T={T}', fontsize=11, fontweight='bold',
                                        rotation=0, labelpad=35, va='center')

        fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.02,
                            hspace=0.35, wspace=0.08)
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')
        plt.close()

        # ── Page 2: Summary table ──
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.axis('off')

        table_data = []
        for T in step_set:
            rk, ssim_lora_sweep = best_per_t[T]
            fs_sweep = full_ssim_sweep.get(T, float('nan'))
            res = all_results[T]
            actual_full = res.get('full_metrics', {}).get('ssim', float('nan'))
            actual_lora = res.get('lora_metrics', {}).get('ssim', float('nan'))
            table_data.append([
                str(T), f'r={rk}',
                f'{actual_full:.4f}' if actual_full == actual_full else 'N/A',
                f'{actual_lora:.4f}' if actual_lora == actual_lora else 'N/A',
                f'{fs_sweep:.4f}' if fs_sweep == fs_sweep else 'N/A',
                f'{ssim_lora_sweep:.4f}',
            ])

        table = ax2.table(
            cellText=table_data,
            colLabels=['T', 'Best Rank', 'Full SSIM\n(this run)',
                       'LoRA SSIM\n(this run)', 'Full SSIM\n(sweep)',
                       'LoRA SSIM\n(sweep)'],
            loc='center', cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#E0E0E0')
                cell.set_text_props(fontweight='bold')

        fig2.suptitle('T-Sweep Summary: SSIM Values',
                      fontsize=13, fontweight='bold')
        pdf.savefig(fig2, bbox_inches='tight', facecolor='white')
        plt.close()

    print(f"PDF saved: {pdf_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str,
                        default=os.path.join(RESULTS_DIR, 'experiment_b_sweep_20260222_115334.csv'))
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--extraction_epochs', type=int, default=1000)
    parser.add_argument('--output_pth', type=str, default=None)
    parser.add_argument('--output_pdf', type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
    print(f"Using device: {args.device}")

    # Parse sweep CSV
    step_set, best_per_t, full_ssim_sweep = parse_best_per_t(args.csv)

    print("Best LoRA rank per T:")
    for T in step_set:
        rk, ssim = best_per_t[T]
        fs = full_ssim_sweep.get(T, float('nan'))
        print(f"  T={T:>4d}: rank={rk:>2d} (SSIM={ssim:.4f}), full SSIM={fs:.4f}")

    # Run extractions
    all_results = run_all_extractions(step_set, best_per_t, args.device,
                                       args.extraction_epochs)

    # Save results
    pth_path = args.output_pth or os.path.join(RESULTS_DIR, 't_sweep_examples.pth')
    save_results(all_results, best_per_t, pth_path)

    # Generate PDF
    pdf_path = args.output_pdf or os.path.join(FIGURES_DIR, 't_sweep_examples.pdf')
    generate_pdf(all_results, best_per_t, full_ssim_sweep, step_set, pdf_path)

    print("\n=== Done ===")
