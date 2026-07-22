"""Re-score saved reconstruction runs without re-running any GPU work.

Every run saves its tensors (x_train, x_recon_*, x_ctrl, ds_mean) to results/*.pth, so new or
corrected metrics can be applied retroactively to completed and in-flight sweeps. This is why a
metric change does not require killing running jobs or recomputing hours of extraction.

Tensor-space convention (matches run_experiment_b.py):
    x_train, x_ctrl, ds_mean  -> pixel space
    x_recon_full/lora         -> centered space (i.e. compared against x_train - ds_mean)

Usage:
    python -m experiments.recompute_metrics
    python -m experiments.recompute_metrics --glob 'results/exp_b_T1_*.pth' --out /tmp/rescored.csv
"""

import argparse
import csv
import glob as globlib
import os

import torch

from experiments.configs import RESULTS_DIR
from experiments.metrics import compute_all_metrics

RECON_KEYS = ['x_recon_full', 'x_recon_lora']

# Scalar metric columns pulled from compute_all_metrics for the CSV.
METRIC_COLS = ['ssim', 'ssim11', 'ssim_norm', 'ssim_norm11', 'ssim_mean_baseline',
               'ncc', 'l2', 'clipped_fraction', 'pre_clamp_min', 'pre_clamp_max']


def _means(metrics):
    return {k: (metrics[k]['mean'] if k in metrics else None) for k in METRIC_COLS}


def rescore_file(path):
    """Recompute metrics for one saved run. Returns a list of CSV row dicts."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    x_train = d.get('x_train')
    ds_mean = d.get('ds_mean')
    x_ctrl = d.get('x_ctrl')
    if x_train is None:
        return []

    x_centered = x_train - ds_mean if ds_mean is not None else x_train
    x_ctrl_centered = (x_ctrl - ds_mean) if (x_ctrl is not None and ds_mean is not None) else x_ctrl

    rows = []
    for key in RECON_KEYS:
        recon = d.get(key)
        if recon is None:
            continue
        row = {'file': os.path.basename(path), 'recon': key.replace('x_recon_', '')}
        row.update(_means(compute_all_metrics(recon, x_centered, ds_mean)))

        # Control uses the same semantics as run_experiment_b.py:557 —
        # SSIM(reconstruction, control image), NOT SSIM(control, ground truth).
        if x_ctrl_centered is not None and recon.shape[0] == x_ctrl_centered.shape[0]:
            ctrl = compute_all_metrics(recon, x_ctrl_centered, ds_mean)
            row['ctrl_ssim'] = ctrl['ssim']['mean']
            row['ctrl_ssim11'] = ctrl['ssim11']['mean']
            row['ctrl_ssim_norm'] = ctrl['ssim_norm']['mean']

        cfg = d.get('config') or {}
        for c in ('n_steps', 'rank', 'seed', 'finetune_activation', 'lr',
                  'n_per_class', 'loss_type', 'anchor_alpha'):
            if c in cfg:
                row[c] = cfg[c]
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--glob', default=os.path.join(RESULTS_DIR, 'exp_b_*.pth'))
    p.add_argument('--out', default=None, help='CSV output path')
    args = p.parse_args()

    files = sorted(globlib.glob(args.glob))
    if not files:
        print(f"No files matched {args.glob}")
        return

    rows = []
    for f in files:
        try:
            rows.extend(rescore_file(f))
        except Exception as e:  # a corrupt/partial .pth must not abort the sweep re-scoring
            print(f"  SKIP {os.path.basename(f)}: {type(e).__name__}: {e}")

    if not rows:
        print("No rescorable runs found.")
        return

    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)

    out = args.out or os.path.join(RESULTS_DIR, 'metrics_recomputed.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Rescored {len(rows)} reconstructions from {len(files)} files -> {out}")
    hdr = f"{'file':46s} {'recon':5s} {'ssim':>7s} {'ssim11':>7s} {'norm':>7s} {'base':>7s} {'clip%':>7s}"
    print(hdr); print('-' * len(hdr))
    for r in rows:
        def f(k):
            v = r.get(k)
            return f"{v:7.4f}" if isinstance(v, float) else f"{'-':>7s}"
        print(f"{r['file'][:46]:46s} {r['recon']:5s} {f('ssim')} {f('ssim11')} "
              f"{f('ssim_norm')} {f('ssim_mean_baseline')} {f('clipped_fraction')}")


if __name__ == '__main__':
    main()
