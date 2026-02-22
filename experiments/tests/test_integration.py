"""Integration tests: end-to-end pipeline smoke tests."""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from experiments.run_experiment_a import (
    create_fresh_model, create_extraction_model, run_extraction, run_single_config,
)
from experiments.train_lora import train_lora, train_full_finetune
from experiments.lora_wrapper import compose_state_dict, save_composed_weights
from experiments.data_utils import get_few_shot_mnist
from experiments.metrics import compute_all_metrics
from experiments.configs import MODEL_INIT_LIST

torch.set_default_dtype(torch.float64)


class TestComposeLoadRoundtrip:
    def test_save_load_matches_output(self, tmp_path):
        """Save composed weights → load into fresh model → output matches."""
        model = create_fresh_model(init_scale=MODEL_INIT_LIST[0])
        x, y, _, _ = get_few_shot_mnist(1, seed=42)
        train_lora(model, x.clone(), y.clone(), rank=8,
                   epochs=200, threshold=0, verbose=False)

        # Get output from LoRA model
        model.eval()
        x_test = torch.randn(2, 1, 28, 28)
        out_lora = model(x_test).detach()

        # Save and load
        path = str(tmp_path / 'test.pth')
        save_composed_weights(model, path)

        fresh = create_fresh_model()
        from common_utils.common import load_weights
        load_weights(fresh, path, device='cpu')
        fresh.eval()
        out_fresh = fresh(x_test).detach()

        torch.testing.assert_close(out_lora, out_fresh)


class TestFullPipelineSmoke:
    def test_no_crashes(self):
        """Train LoRA → compose → load → extract (short) → no crashes, finite loss."""
        model = create_fresh_model(init_scale=MODEL_INIT_LIST[0])
        x, y, _, _ = get_few_shot_mnist(1, seed=42)

        result = train_lora(
            model, x.clone(), y.clone(), rank=8,
            epochs=200, threshold=0, verbose=False,
        )
        ds_mean = result['ds_mean']

        composed_sd = compose_state_dict(model)
        extraction_model = create_extraction_model()
        extraction_model.load_state_dict(composed_sd)

        x_centered = x - ds_mean if ds_mean is not None else x
        x_recon, extract_res = run_extraction(
            extraction_model, x_centered, y, ds_mean, n_per_class=1,
            extraction_epochs=100,
        )

        # Basic sanity
        assert x_recon.shape == x_centered.shape
        assert torch.isfinite(x_recon).all()
        assert len(extract_res['loss_history']) > 0
        assert all(torch.isfinite(torch.tensor(v)) for v in extract_res['loss_history'])


class TestMetricsSanity:
    def test_perfect_ssim_for_identical(self):
        """SSIM should be ~1.0 for identical images."""
        x = torch.rand(2, 1, 28, 28)
        metrics = compute_all_metrics(x, x.clone())
        assert metrics['ssim']['mean'] > 0.99
        assert metrics['dssim']['mean'] < 0.01
        assert metrics['l2']['mean'] < 0.01

    def test_low_ssim_for_random(self):
        """SSIM should be low for completely different random images."""
        x1 = torch.rand(2, 1, 28, 28)
        x2 = torch.rand(2, 1, 28, 28)
        metrics = compute_all_metrics(x1, x2)
        assert metrics['ssim']['mean'] < 0.5


class TestExtractionModelActivation:
    def test_has_modified_relu(self):
        """Extraction model should use ModifiedReLU, not standard ReLU."""
        from CreateModel import ModifiedRelu
        model = create_extraction_model()
        has_modified = False
        for module in model.modules():
            if isinstance(module, ModifiedRelu):
                has_modified = True
                break
        assert has_modified, "Extraction model doesn't have ModifiedReLU!"
