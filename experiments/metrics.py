"""Metrics wrapper around the existing evaluations.py.

SSIM reporting notes (see LESSONS_LEARNED.md 2026-07-22)
--------------------------------------------------------
- ``ssim`` uses **window_size=3**, matching ``get_ssim_pairs_kornia`` and therefore every
  historical number in STATUS.md. Keep using it for continuity with our own past runs.
- ``ssim11`` uses **window_size=11**, the Wang et al. (2004) standard that SimuDy / Haim et al.
  report. Use this — and only this — when comparing against published numbers.
- ``ssim_norm`` matches each reconstruction's mean/std to its target before scoring, removing the
  luminance and contrast penalty and leaving structure. Raw SSIM is scale/contrast sensitive, so a
  perfectly recognisable reconstruction with a global brightness shift scores badly under ``ssim``.
- ``ssim_mean_baseline`` is what the *trivial* predictor (the dataset mean) scores. A result at or
  below this carries no instance-specific information, however good the raw SSIM looks.
- ``clipped_fraction`` reports how much of ``x_recon + ds_mean`` fell outside [0, 1] and was
  silently saturated by the clamp below. Extraction only *softly* constrains x to [-1, 1]
  (ntk_extraction.get_ntk_verify_loss), so this can be non-zero and it corrupts SSIM when it is.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dataset_reconstruction'))

from evaluations import viz_nns, ncc_dist, l2_dist
from common_utils.image import get_ssim_pairs_kornia

# Window sizes: 3 = our historical setting, 11 = literature standard (Wang et al. 2004).
SSIM_WINDOW_LEGACY = 3
SSIM_WINDOW_STANDARD = 11


def _ssim_pairs(a, b, window_size=SSIM_WINDOW_LEGACY):
    """Per-pair SSIM with an explicit window size.

    Mirrors ``common_utils.image.get_ssim_pairs_kornia`` (which hardcodes window_size=3) but lets
    the caller choose the window. Implemented here rather than by editing the vendored
    dataset_reconstruction/ helper, which lives in a separate git repo.
    """
    from kornia import metrics as kornia_metrics
    return kornia_metrics.ssim(a, b, window_size=window_size).reshape(a.shape[0], -1).mean(dim=1)


def _prepare_pair(x_recon, x_target, ds_mean=None):
    """Add ds_mean back, clamp to [0, 1], and report how much was clipped.

    Returns (a, b, clip_info) where clip_info describes the *pre-clamp* range of the
    reconstruction so silent saturation is visible instead of invisible.
    """
    a = x_recon.clone()
    b = x_target.clone()
    if ds_mean is not None:
        a = a + ds_mean
        b = b + ds_mean

    out_of_range = ((a < 0) | (a > 1)).float()
    clip_info = {
        'clipped_fraction': out_of_range.mean().item(),
        'pre_clamp_min': a.min().item(),
        'pre_clamp_max': a.max().item(),
    }

    a = a.clamp(0, 1).float().cpu()
    b = b.clamp(0, 1).float().cpu()
    return a, b, clip_info


def _match_moments(a, b, eps=1e-8):
    """Rescale each image of `a` to match the per-image mean/std of `b`.

    Removes the luminance and contrast terms of SSIM, leaving the structural comparison.
    """
    dims = tuple(range(1, a.dim()))
    a_mean, a_std = a.mean(dim=dims, keepdim=True), a.std(dim=dims, keepdim=True)
    b_mean, b_std = b.mean(dim=dims, keepdim=True), b.std(dim=dims, keepdim=True)
    return (a - a_mean) / (a_std + eps) * b_std + b_mean


def compute_ssim(x_recon, x_target, ds_mean=None, window_size=SSIM_WINDOW_LEGACY):
    """Compute per-pair SSIM between reconstructed and target images.

    Args:
        x_recon: [N, C, H, W] reconstructed images
        x_target: [N, C, H, W] target images
        ds_mean: dataset mean to add back (if data was mean-subtracted)
        window_size: 3 (historical, default) or 11 (literature standard)

    Returns:
        ssim_per_pair: tensor [N], SSIM values per pair
        mean_ssim: float
    """
    with torch.no_grad():
        a, b, _ = _prepare_pair(x_recon, x_target, ds_mean)
        ssim_vals = _ssim_pairs(a, b, window_size=window_size)
    return ssim_vals, ssim_vals.mean().item()


def compute_ssim_normalized(x_recon, x_target, ds_mean=None,
                            window_size=SSIM_WINDOW_LEGACY):
    """SSIM after matching each reconstruction's mean/std to its target.

    Scale/offset invariant: answers "is the *structure* right?" independently of a global
    brightness or contrast mismatch.
    """
    with torch.no_grad():
        a, b, _ = _prepare_pair(x_recon, x_target, ds_mean)
        a = _match_moments(a, b).clamp(0, 1)
        ssim_vals = _ssim_pairs(a, b, window_size=window_size)
    return ssim_vals, ssim_vals.mean().item()


def compute_mean_baseline_ssim(x_target, ds_mean, window_size=SSIM_WINDOW_LEGACY):
    """SSIM of the trivial predictor (the dataset mean) against the targets.

    This is the floor. A reconstruction scoring at or below it has learned nothing
    instance-specific, no matter how respectable the absolute number looks. Returns None when
    ds_mean is unavailable.
    """
    if ds_mean is None:
        return None, None
    with torch.no_grad():
        b = (x_target + ds_mean).clamp(0, 1).float().cpu()
        a = ds_mean.expand_as(x_target).clamp(0, 1).float().cpu()
        ssim_vals = _ssim_pairs(a, b, window_size=window_size)
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

    Returns dict with 'ssim', 'dssim', 'ncc', 'l2' (each with 'per_pair' and 'mean'), plus the
    diagnostic entries 'ssim11', 'ssim_norm', 'ssim_norm11', 'ssim_mean_baseline' and the scalar
    clipping diagnostics. Existing keys keep their exact previous meaning so historical .pth files
    and downstream code stay valid.
    """
    ssim_vals, ssim_mean = compute_ssim(x_recon, x_target, ds_mean)
    ssim11_vals, ssim11_mean = compute_ssim(x_recon, x_target, ds_mean,
                                            window_size=SSIM_WINDOW_STANDARD)
    norm_vals, norm_mean = compute_ssim_normalized(x_recon, x_target, ds_mean)
    norm11_vals, norm11_mean = compute_ssim_normalized(x_recon, x_target, ds_mean,
                                                       window_size=SSIM_WINDOW_STANDARD)
    base_vals, base_mean = compute_mean_baseline_ssim(x_target, ds_mean)
    dssim_vals, dssim_mean = compute_dssim(x_recon, x_target, ds_mean)
    ncc_vals, ncc_mean = compute_ncc(x_recon, x_target)

    with torch.no_grad():
        _, _, clip_info = _prepare_pair(x_recon, x_target, ds_mean)
        flat_r = x_recon.reshape(x_recon.shape[0], -1).float().cpu()
        flat_t = x_target.reshape(x_target.shape[0], -1).float().cpu()
        l2_vals = (flat_r - flat_t).pow(2).sum(dim=1).sqrt()
        l2_mean = l2_vals.mean().item()

    out = {
        'ssim': {'per_pair': ssim_vals, 'mean': ssim_mean},
        'ssim11': {'per_pair': ssim11_vals, 'mean': ssim11_mean},
        'ssim_norm': {'per_pair': norm_vals, 'mean': norm_mean},
        'ssim_norm11': {'per_pair': norm11_vals, 'mean': norm11_mean},
        'dssim': {'per_pair': dssim_vals, 'mean': dssim_mean},
        'ncc': {'per_pair': ncc_vals, 'mean': ncc_mean},
        'l2': {'per_pair': l2_vals, 'mean': l2_mean},
    }
    if base_mean is not None:
        out['ssim_mean_baseline'] = {'per_pair': base_vals, 'mean': base_mean}
    for k, v in clip_info.items():
        out[k] = {'per_pair': None, 'mean': v}
    return out


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
