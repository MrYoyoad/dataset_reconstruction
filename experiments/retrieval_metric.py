"""Instance-level identifiability metric: retrieval accuracy.

Why this exists (STATUS.md 2026-07-22): absolute SSIM on mostly-black MNIST is dominated by the
shared background, so a blurry `ds_mean` already scores ~0.76 and SSIM cannot tell a real
reconstruction from a blur. Retrieval sidesteps that by asking a *relative* question:

    among all N training images, is the reconstruction of image i most similar to image i?

The background is common to every candidate, so it cancels in the comparison. This measures
*instance-level* leakage (did we recover THIS image, not merely "a 3") and has a clean random
baseline of 1/N — so it gets stronger, not weaker, as N grows.

Distances are computed in several spaces. `feature_fn` (e.g. a classifier's penultimate layer) plugs
in the classifier-based ranking idea; without it, pixel / NCC / SSIM spaces already give a
background-robust instance-level signal.
"""

import argparse
import glob as globlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from evaluations import ncc_dist              # noqa: E402
from common_utils.image import get_ssim_all   # noqa: E402
from experiments.configs import RESULTS_DIR   # noqa: E402

METRIC_SPACES = ('pixel', 'ncc', 'ssim')


def _to_pixel(recon, x_train, ds_mean):
    """Put both reconstruction and target in clamped pixel space.

    x_recon_* are stored in centered space (target = x_train - ds_mean); x_train is pixel space.
    """
    a = recon + ds_mean if ds_mean is not None else recon
    a = a.clamp(0, 1).float().cpu()
    b = x_train.clamp(0, 1).float().cpu()
    return a, b


def similarity_matrix(a, b, space='pixel', feature_fn=None):
    """[N_recon, N_target] similarity (higher = more similar), row i = reconstruction i."""
    if feature_fn is not None:
        fa, fb = feature_fn(a), feature_fn(b)
        fa = fa.reshape(fa.shape[0], -1)
        fb = fb.reshape(fb.shape[0], -1)
        fa = fa / (fa.norm(dim=1, keepdim=True) + 1e-8)
        fb = fb / (fb.norm(dim=1, keepdim=True) + 1e-8)
        return fa @ fb.t()                       # cosine similarity in feature space
    if space == 'pixel':
        return -torch.cdist(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))  # -L2
    if space == 'ncc':
        return -ncc_dist(a, b)                   # ncc_dist is a distance -> negate
    if space == 'ssim':
        return get_ssim_all(a, b)                # S[i,j] = ssim(recon_i, target_j)
    raise ValueError(f"unknown space: {space}")


def retrieval_scores(sim):
    """Given an [N,N] similarity matrix, score how often the correct target is retrieved.

    Returns top-1 accuracy, mean rank of the true target (1 = best), mean reciprocal rank, and the
    1/N random baseline. Ranking is done per reconstruction (per row).
    """
    n = sim.shape[0]
    # rank of the diagonal (true target) within its row: how many candidates beat or tie it.
    diag = sim.diag().unsqueeze(1)
    rank = (sim > diag).sum(dim=1) + 1                 # 1 = true target is the single best
    rank = rank.float()
    return {
        'n': n,
        'top1_acc': (rank == 1).float().mean().item(),
        'mean_rank': rank.mean().item(),
        'mrr': (1.0 / rank).mean().item(),
        'random_top1': 1.0 / n,
    }


def score_file(path, feature_fn=None):
    """Retrieval scores for each reconstruction in one saved run. N<2 is skipped (undefined)."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    x_train, ds_mean = d.get('x_train'), d.get('ds_mean')
    if x_train is None or x_train.shape[0] < 2:
        return []
    rows = []
    for key in ('x_recon_full', 'x_recon_lora'):
        recon = d.get(key)
        if recon is None:
            continue
        a, b = _to_pixel(recon, x_train, ds_mean)
        row = {'file': os.path.basename(path), 'recon': key.replace('x_recon_', '')}
        for space in METRIC_SPACES:
            s = retrieval_scores(similarity_matrix(a, b, space=space))
            row[f'top1_{space}'] = s['top1_acc']
            row[f'rank_{space}'] = s['mean_rank']
        if feature_fn is not None:
            s = retrieval_scores(similarity_matrix(a, b, feature_fn=feature_fn))
            row['top1_feat'] = s['top1_acc']
            row['rank_feat'] = s['mean_rank']
        row['n'] = x_train.shape[0]
        row['random_top1'] = 1.0 / x_train.shape[0]
        cfg = d.get('config') or {}
        for c in ('rank', 'n_per_class', 'seed', 'finetune_activation'):
            if c in cfg:
                row[c] = cfg[c]
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--glob', default=os.path.join(RESULTS_DIR, 'exp_b_*.pth'))
    args = p.parse_args()
    files = sorted(globlib.glob(args.glob))
    if not files:
        print(f"No files matched {args.glob}")
        return

    rows = []
    for f in files:
        try:
            rows.extend(score_file(f))
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {type(e).__name__}: {e}")

    hdr = f"{'file':44s} {'rec':4s} {'N':>3s} {'rand':>5s} {'top1_px':>7s} {'top1_ncc':>8s} {'top1_ssim':>9s}"
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f"{r['file'][:44]:44s} {r['recon']:4s} {r['n']:3d} {r['random_top1']:5.2f} "
              f"{r['top1_pixel']:7.2f} {r['top1_ncc']:8.2f} {r['top1_ssim']:9.2f}")
    print(f"\n{len(rows)} reconstructions scored. top1 above 'rand' = instance-level leakage.")


if __name__ == '__main__':
    main()
