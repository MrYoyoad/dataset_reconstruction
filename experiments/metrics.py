"""Metrics wrapper around the existing evaluations.py."""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from evaluations import viz_nns, ncc_dist, l2_dist
from common_utils.image import get_ssim_pairs_kornia


def compute_ssim(x_recon, x_target, ds_mean=None):
    """Compute per-pair SSIM between reconstructed and target images.

    Args:
        x_recon: [N, C, H, W] reconstructed images
        x_target: [N, C, H, W] target images
        ds_mean: dataset mean to add back (if data was mean-subtracted)

    Returns:
        ssim_per_pair: tensor [N], SSIM values per pair
        mean_ssim: float
    """
    with torch.no_grad():
        a = x_recon.clone()
        b = x_target.clone()
        if ds_mean is not None:
            a = a + ds_mean
            b = b + ds_mean
        # Clamp to [0, 1] for SSIM
        a = a.clamp(0, 1).float().cpu()
        b = b.clamp(0, 1).float().cpu()
        ssim_vals = get_ssim_pairs_kornia(a, b)
    return ssim_vals, ssim_vals.mean().item()


def compute_dssim(x_recon, x_target, ds_mean=None):
    """Compute DSSIM = (1 - SSIM) / 2. Lower is better."""
    ssim_vals, mean_ssim = compute_ssim(x_recon, x_target, ds_mean)
    dssim_vals = (1 - ssim_vals) / 2
    return dssim_vals, dssim_vals.mean().item()


def compute_ncc(x_recon, x_target):
    """Compute NCC distance between reconstructed and target (lower = better)."""
    with torch.no_grad():
        # ncc_dist expects 4D tensors (N, C, H, W) for normalize_batch
        r = x_recon.float().cpu()
        t = x_target.float().cpu()
        d = ncc_dist(r, t)
        ncc_vals = d.diag()
    return ncc_vals, ncc_vals.mean().item()


def compute_all_metrics(x_recon, x_target, ds_mean=None):
    """Compute all reconstruction metrics.

    Returns dict with 'ssim', 'dssim', 'ncc', 'l2', each containing
    'per_pair' (tensor) and 'mean' (float).
    """
    ssim_vals, ssim_mean = compute_ssim(x_recon, x_target, ds_mean)
    dssim_vals, dssim_mean = compute_dssim(x_recon, x_target, ds_mean)
    ncc_vals, ncc_mean = compute_ncc(x_recon, x_target)

    with torch.no_grad():
        flat_r = x_recon.reshape(x_recon.shape[0], -1).float().cpu()
        flat_t = x_target.reshape(x_target.shape[0], -1).float().cpu()
        l2_vals = (flat_r - flat_t).pow(2).sum(dim=1).sqrt()
        l2_mean = l2_vals.mean().item()

    return {
        'ssim': {'per_pair': ssim_vals, 'mean': ssim_mean},
        'dssim': {'per_pair': dssim_vals, 'mean': dssim_mean},
        'ncc': {'per_pair': ncc_vals, 'mean': ncc_mean},
        'l2': {'per_pair': l2_vals, 'mean': l2_mean},
    }


def match_reconstructions_to_targets(x_recon, x_target, metric='ncc'):
    """Find nearest-neighbor matching from reconstructions to targets.

    Uses existing viz_nns. Returns matched pairs sorted by distance.
    """
    with torch.no_grad():
        qq, v = viz_nns(
            x_recon.float().cpu(),
            x_target.float().cpu(),
            max_per_nn=1,
            metric=metric,
        )
    return qq, v
