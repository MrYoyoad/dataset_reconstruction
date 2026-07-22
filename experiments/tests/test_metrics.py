"""Tests for experiments/metrics.py.

Each test here corresponds to a failure that actually bit us (see LESSONS_LEARNED.md 2026-07-22):
an identical config scored SSIM 0.358 at 5 extraction epochs but 0.041 at 50,000, and nothing in
the pipeline could distinguish "the reconstruction is bad" from "the metric is being distorted by
saturation / contrast mismatch" or from "this is just the dataset mean".

CPU-only, per experiments/tests/README.md.
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from experiments.metrics import (  # noqa: E402
    compute_all_metrics, compute_ssim, compute_ssim_normalized,
    compute_mean_baseline_ssim, _ssim_pairs,
    SSIM_WINDOW_LEGACY, SSIM_WINDOW_STANDARD,
)


def _blob(n=4, size=28, seed=0):
    """Smooth-ish [0,1] images (pure noise makes SSIM degenerate and tests meaningless)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, size, size, generator=g)
    x = torch.nn.functional.avg_pool2d(x, 5, stride=1, padding=2)  # decorrelate high freq
    return (x - x.amin()) / (x.amax() - x.amin())


class TestBasics:
    def test_identical_is_one(self):
        x = _blob()
        _, m = compute_ssim(x, x.clone())
        assert m > 0.99

    def test_window3_matches_legacy_helper(self):
        """The local helper must reproduce common_utils.image.get_ssim_pairs_kornia exactly.

        If this ever fails, every historical SSIM in STATUS.md silently changed meaning.
        """
        from common_utils.image import get_ssim_pairs_kornia
        a, b = _blob(seed=1), _blob(seed=2)
        legacy = get_ssim_pairs_kornia(a, b)
        local = _ssim_pairs(a, b, window_size=SSIM_WINDOW_LEGACY)
        assert torch.allclose(legacy, local, atol=1e-6)

    def test_window_size_changes_value(self):
        """window=3 (ours, historical) != window=11 (Wang et al., what SimuDy reports).

        Pins the fact that our numbers are NOT directly comparable to published ones.
        """
        a, b = _blob(seed=1), _blob(seed=2)
        s3 = _ssim_pairs(a, b, window_size=SSIM_WINDOW_LEGACY).mean().item()
        s11 = _ssim_pairs(a, b, window_size=SSIM_WINDOW_STANDARD).mean().item()
        assert abs(s3 - s11) > 1e-3, "windows 3 and 11 should differ; cross-paper comparison is unsafe"


class TestNormalization:
    def test_normalized_ssim_is_scale_invariant(self):
        """A global contrast/brightness change must not move ssim_norm, but does move raw ssim.

        This is the core claim behind reporting ssim_norm: a perfectly recognisable
        reconstruction with a global photometric shift should not be scored as a failure.
        """
        target = _blob(seed=3)
        recon = target.clone()
        shifted = (recon * 0.5 + 0.2).clamp(0, 1)   # contrast squash + brightness offset

        _, raw = compute_ssim(shifted, target)
        _, norm = compute_ssim_normalized(shifted, target)

        assert raw < 0.99, "raw SSIM should be penalised by the photometric shift"
        assert norm > raw, "normalized SSIM should recover most of that penalty"
        assert norm > 0.9, "structure is identical, so normalized SSIM should be near 1"


class TestDiagnostics:
    def test_clipping_fraction_detected(self):
        """Out-of-range reconstructions are silently saturated by the [0,1] clamp.

        Extraction only *softly* constrains x to [-1,1], and saved runs show x_recon pinned at
        exactly +/-1, so after adding ds_mean a large fraction can fall outside [0,1].
        """
        target = _blob(seed=4)
        in_range = target.clone()
        out_range = target.clone() * 4.0 - 1.5   # deliberately far outside [0,1]

        m_ok = compute_all_metrics(in_range, target)
        m_bad = compute_all_metrics(out_range, target)

        assert m_ok['clipped_fraction']['mean'] == pytest.approx(0.0, abs=1e-6)
        assert m_bad['clipped_fraction']['mean'] > 0.1
        assert m_bad['pre_clamp_max']['mean'] > 1.0 or m_bad['pre_clamp_min']['mean'] < 0.0

    def test_mean_baseline_is_reported_and_nontrivial(self):
        """The dataset mean is a *trivial* predictor and still scores well above zero.

        Any reconstruction at or below this number carries no instance-specific information,
        which is the most likely explanation of the 0.358-at-5-epochs result.
        """
        target_pixel = _blob(seed=5)
        ds_mean = target_pixel.mean(dim=0, keepdim=True)
        target_centered = target_pixel - ds_mean

        _, base = compute_mean_baseline_ssim(target_centered, ds_mean)
        assert base is not None and base > 0.05, "the trivial predictor should score meaningfully > 0"

        m = compute_all_metrics(torch.zeros_like(target_centered), target_centered, ds_mean)
        assert 'ssim_mean_baseline' in m
        # x_recon = 0 in centered space IS the dataset mean, so it must match the baseline.
        assert m['ssim']['mean'] == pytest.approx(base, abs=1e-5)

    def test_mean_baseline_none_without_ds_mean(self):
        vals, mean = compute_mean_baseline_ssim(_blob(), None)
        assert vals is None and mean is None


class TestSemantics:
    def test_control_is_recon_vs_control_not_control_vs_truth(self):
        """Pin the control semantics (run_experiment_b.py:557).

        Control is SSIM(reconstruction, control image) — a Haim-style "does the reconstruction
        match the true sample more than a different same-class sample?" test. It therefore
        legitimately changes when the reconstruction changes; that is NOT evidence of a bug.
        """
        truth = _blob(seed=6)
        ctrl = _blob(seed=7)
        good = truth.clone()
        bad = ctrl.clone()

        good_vs_truth = compute_all_metrics(good, truth)['ssim']['mean']
        good_vs_ctrl = compute_all_metrics(good, ctrl)['ssim']['mean']
        bad_vs_truth = compute_all_metrics(bad, truth)['ssim']['mean']

        assert good_vs_truth > good_vs_ctrl, "a good reconstruction beats the control comparison"
        assert good_vs_truth > bad_vs_truth
        # Control depends on the reconstruction by construction:
        assert compute_all_metrics(bad, ctrl)['ssim']['mean'] != pytest.approx(good_vs_ctrl, abs=1e-3)


class TestRegression:
    def test_near_init_vs_structured(self):
        """Regression for the 5-epoch vs 50k-epoch anomaly, in miniature.

        A near-dataset-mean 'reconstruction' can out-score a structurally-correct but
        photometrically-shifted one under raw SSIM. The baseline + normalized metrics must expose
        that, so a near-mean output is never mistaken for a good reconstruction.
        """
        target_pixel = _blob(seed=8)
        ds_mean = target_pixel.mean(dim=0, keepdim=True)
        target_centered = target_pixel - ds_mean

        near_mean = torch.zeros_like(target_centered)              # == predicting ds_mean
        structured = (target_centered * 2.2).clamp(-1, 1)          # right structure, wrong scale

        m_mean = compute_all_metrics(near_mean, target_centered, ds_mean)
        m_struct = compute_all_metrics(structured, target_centered, ds_mean)
        baseline = m_mean['ssim_mean_baseline']['mean']

        # The near-mean output must be indistinguishable from the trivial baseline...
        assert m_mean['ssim']['mean'] == pytest.approx(baseline, abs=1e-5)
        # ...while the structured one is genuinely better once scale is normalised away.
        assert m_struct['ssim_norm']['mean'] > m_mean['ssim_norm']['mean'], (
            "normalized SSIM must rank real structure above the trivial mean predictor")
