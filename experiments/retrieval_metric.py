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
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from evaluations import ncc_dist              # noqa: E402
from common_utils.image import get_ssim_all   # noqa: E402
from CreateModel import NeuralNetwork          # noqa: E402
from experiments.configs import (              # noqa: E402
    RESULTS_DIR, INPUT_DIM, MODEL_HIDDEN_LIST, OUTPUT_DIM, PRETRAINED_MNIST_PATH,
)

METRIC_SPACES = ('pixel', 'ncc', 'ssim')


def base_classifier_feature_fn(model_path=None, device='cpu'):
    """Feature extractor from the EXISTING base MNIST classifier (theta_0), no training needed.

    Uses the penultimate layer (1000-dim, the input to the final linear) as an instance-discriminative
    feature space. The attacker already has theta_0, so ranking in its feature space is a fair,
    self-contained "classifier-based" retrieval — and it is far more digit-aware than raw pixels.
    """
    path = model_path or PRETRAINED_MNIST_PATH
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NeuralNetwork(INPUT_DIM, MODEL_HIDDEN_LIST, OUTPUT_DIM,
                          activation=nn.ReLU(), use_bias=False).to(device).double()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    @torch.no_grad()
    def fn(x):
        feats = x.to(device).double().reshape(x.shape[0], -1)
        for layer in model.layers[:-1]:          # all but the final classification layer
            feats = model.activation(layer(feats))
        return feats.detach().cpu()              # [N, 1000] penultimate features
    return fn


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
    p.add_argument('--classifier', action='store_true',
                   help='Also rank in the base MNIST classifier feature space (adds top1_feat)')
    args = p.parse_args()
    files = sorted(globlib.glob(args.glob))
    if not files:
        print(f"No files matched {args.glob}")
        return

    feature_fn = base_classifier_feature_fn() if args.classifier else None

    rows = []
    for f in files:
        try:
            rows.extend(score_file(f, feature_fn=feature_fn))
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {type(e).__name__}: {e}")

    feat = ' top1_feat' if feature_fn is not None else ''
    hdr = f"{'file':40s} {'rec':4s} {'N':>3s} {'rand':>5s} {'px':>5s} {'ncc':>5s} {'ssim':>5s}{feat}"
    print(hdr); print('-' * len(hdr))
    for r in rows:
        line = (f"{r['file'][:40]:40s} {r['recon']:4s} {r['n']:3d} {r['random_top1']:5.2f} "
                f"{r['top1_pixel']:5.2f} {r['top1_ncc']:5.2f} {r['top1_ssim']:5.2f}")
        if feature_fn is not None:
            line += f" {r.get('top1_feat', float('nan')):9.2f}"
        print(line)
    print(f"\n{len(rows)} reconstructions scored. top1 above 'rand' = instance-level leakage.")


if __name__ == '__main__':
    main()
