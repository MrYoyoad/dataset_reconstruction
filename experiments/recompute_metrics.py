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
import re

import torch

from experiments.configs import ACTIVATION_CHOICES, RESULTS_DIR
from experiments.metrics import compute_all_metrics

RECON_KEYS = ['x_recon_full', 'x_recon_lora']

# Scalar metric columns pulled from compute_all_metrics for the CSV.
METRIC_COLS = ['ssim', 'ssim11', 'ssim_norm', 'ssim_norm11', 'ssim_mean_baseline',
               'ncc', 'l2', 'clipped_fraction', 'pre_clamp_min', 'pre_clamp_max']

# Diagnostic columns pulled from {full,lora}_diagnostics — an SSIM is meaningless without
# these (near-zero weight_change or effective_rank << r ⇒ the fine-tune barely happened).
DIAG_COLS = ['weight_change', 'delta_w_effective_rank', 'feature_stability', 'ntk_passed']

# Known activations, longest-first, so 'gelu_tanh'/'softplus_b0.5' are matched before their
# prefixes 'gelu'/'softplus'. Used to recover finetune_activation from the filename, since
# run_experiment_b.py does NOT save it into config (only encodes it in base_name).
_ACTS_LONGEST_FIRST = sorted(ACTIVATION_CHOICES, key=len, reverse=True)
_ACT_RE = re.compile(r'_a\d+_(' + '|'.join(re.escape(a) for a in _ACTS_LONGEST_FIRST) + r')(?=_|$)')


def _activation_from_filename(fname):
    """Recover the fine-tune activation from a base_name like
    'exp_b_T1_r8_s42_a149_gelu_tanh_lr0.1'. Returns None if no known activation token is present
    (e.g. a default-relu run 'exp_b_T1_r8_s42_a149_npc2')."""
    m = _ACT_RE.search(os.path.splitext(fname)[0])
    return m.group(1) if m else None


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
        recon_type = key.replace('x_recon_', '')  # 'full' or 'lora'
        row = {'file': os.path.basename(path), 'recon': recon_type}
        row.update(_means(compute_all_metrics(recon, x_centered, ds_mean)))

        # Control uses the same semantics as run_experiment_b.py:557 —
        # SSIM(reconstruction, control image), NOT SSIM(control, ground truth).
        # ctrl_margin (ssim − ctrl_ssim) is the instance-leakage bar: it cancels the shared
        # clipping/background, so it is trustworthy at small N where absolute SSIM is not.
        if x_ctrl_centered is not None and recon.shape[0] == x_ctrl_centered.shape[0]:
            ctrl = compute_all_metrics(recon, x_ctrl_centered, ds_mean)
            row['ctrl_ssim'] = ctrl['ssim']['mean']
            row['ctrl_ssim11'] = ctrl['ssim11']['mean']
            row['ctrl_ssim_norm'] = ctrl['ssim_norm']['mean']
            row['ctrl_margin'] = row['ssim'] - row['ctrl_ssim']
            row['ctrl_margin_norm'] = row['ssim_norm'] - row['ctrl_ssim_norm']

        # NTK diagnostics — never trust an SSIM without them (weight_change / effective_rank).
        diag = d.get(f'{recon_type}_diagnostics') or {}
        for dk in DIAG_COLS:
            if dk in diag:
                row[dk] = diag[dk]

        cfg = d.get('config') or {}
        # finetune_activation / loss_type are not saved in config — recover from the filename.
        if 'finetune_activation' not in cfg:
            act = _activation_from_filename(os.path.basename(path))
            if act is not None:
                row['finetune_activation'] = act
        if 'loss_type' not in cfg:
            row['loss_type'] = 'cosine' if '_cosine' in os.path.basename(path) else 'l2'
        for c in ('n_steps', 'rank', 'seed', 'finetune_activation', 'lr',
                  'n_per_class', 'loss_type', 'anchor_alpha', 'dataset', 'source',
                  'finetune_optimizer', 'verify_weight'):
            if c in cfg:
                row[c] = cfg[c]
        # Fallback: recover dataset from the filename prefix (exp_b_T1_<dataset>_...) for older
        # runs whose config predates the 'dataset' key. 'mnist' carries no prefix token.
        if 'dataset' not in row:
            base = os.path.basename(path)
            for ds in ('flowers32', 'flowers64', 'flowers', 'fashion'):
                if f'_{ds}_' in base:
                    row['dataset'] = ds
                    break
            else:
                row['dataset'] = 'mnist'
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
    hdr = (f"{'activation':13s} {'recon':5s} {'ssim':>7s} {'norm':>7s} {'base':>7s} "
           f"{'margin':>7s} {'wchg':>7s} {'erank':>5s} {'clip%':>6s}")
    print(hdr); print('-' * len(hdr))
    for r in rows:
        def f(k, w=7):
            v = r.get(k)
            if isinstance(v, bool):
                return f"{str(v):>{w}s}"
            if isinstance(v, int):
                return f"{v:{w}d}"
            return f"{v:{w}.4f}" if isinstance(v, float) else f"{'-':>{w}s}"
        act = (r.get('finetune_activation') or 'relu?')
        print(f"{act[:13]:13s} {r['recon']:5s} {f('ssim')} {f('ssim_norm')} "
              f"{f('ssim_mean_baseline')} {f('ctrl_margin')} {f('weight_change')} "
              f"{f('delta_w_effective_rank', 5)} {f('clipped_fraction', 6)}")


if __name__ == '__main__':
    main()
