"""Regenerate the clip-free Q-B seen-vs-novel example grids from the pixel-box .pth tensors.

The old figures/pdf_examples/FREEC_QB_*.png show the stale CLIPPED first pass. This rebuilds them
from results/exp_b_T1_flowers32_r8_free_s4{2,3,4}_a10000_vw5_{seen,novel}_pbox.pth (job 952081),
one 3-row grid per arm (GT / LoRA recon / control) across the 3 seeds x 2 images = 6 columns.

Run on WEXAC (bsub), never the login node. CPU-only (tensor load + matplotlib), no GPU needed.
"""
import os
import torch

from experiments.plotting import plot_reconstruction_grid
from experiments.metrics import compute_all_metrics

RESULTS = 'results'
OUTDIR = 'figures/pdf_examples'
SEEDS = [42, 43, 44]
ARMS = ['seen', 'novel']


def _path(arm, seed):
    return os.path.join(RESULTS, f'exp_b_T1_flowers32_r8_free_s{seed}_a10000_vw5_{arm}_pbox.pth')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for arm in ARMS:
        xs_gt, xs_rec, xs_ctrl, sn_list = [], [], [], []
        ds_mean = None
        for seed in SEEDS:
            d = torch.load(_path(arm, seed), map_location='cpu', weights_only=False)
            if d.get('ds_mean') is not None:
                ds_mean = d['ds_mean']
            xt, xr, xc = d['x_train'], d['x_recon_lora'], d['x_ctrl']
            xs_gt.append(xt); xs_rec.append(xr); xs_ctrl.append(xc)
            tgt = xt - ds_mean if ds_mean is not None else xt
            m = compute_all_metrics(xr, tgt, ds_mean)
            sn_list.append(float(m['ssim_norm']['mean']))
        x_train = torch.cat(xs_gt, 0)
        x_recon = torch.cat(xs_rec, 0)
        x_ctrl = torch.cat(xs_ctrl, 0)
        arm_sn = sum(sn_list) / len(sn_list)
        title = (f"Q-B {arm.upper()} (flowers32, free-c, r=8, [0,1] pixel box) - "
                 f"seeds {SEEDS} - mean ssim_norm={arm_sn:.3f}")
        out = os.path.join(OUTDIR, f'FREEC_QB_{arm}_pbox.png')
        plot_reconstruction_grid(
            x_train=x_train, x_recon_lora=x_recon, x_ctrl=x_ctrl,
            ssim_lora=arm_sn, ds_mean=ds_mean, save_path=out, title=title,
            rank=8, class_label='Species-parity')
        print(f"wrote {out}  (mean ssim_norm={arm_sn:.3f}, {x_train.shape[0]} cols)")


if __name__ == '__main__':
    main()
