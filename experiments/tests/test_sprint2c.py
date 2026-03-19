"""Tests for Sprint 2c code changes."""

import sys
import os
import copy
import inspect
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

torch.set_default_dtype(torch.float64)


class TestSprint2cSweepImports:
    def test_sweep_driver_imports(self):
        """Sprint 2c sweep driver should import without errors."""
        from experiments.run_sprint2c_sweep import (
            run_track_a, run_track_b2, run_track_b3a, run_track_b3b, run_track_b4,
            run_track_b5, run_track_b6, run_track_b7,
            save_csv, extract_row_a, extract_row_b,
        )

    def test_experiment_a_new_params(self):
        """Experiment A should accept new parameters."""
        from experiments.run_experiment_a import run_single_config
        sig = inspect.signature(run_single_config)
        assert 'extraction_n_per_class' in sig.parameters
        assert 'fine_tune_lr' in sig.parameters
        assert 'fine_tune_epochs' in sig.parameters


class TestStepsOverride:
    def test_phase3_accepts_steps_override(self):
        """run_phase3 should accept steps_override parameter."""
        from experiments.run_sprint2b_sweep import run_phase3
        sig = inspect.signature(run_phase3)
        assert 'steps_override' in sig.parameters

    def test_phase4_accepts_steps_override(self):
        """run_phase4 should accept steps_override parameter."""
        from experiments.run_sprint2b_sweep import run_phase4
        sig = inspect.signature(run_phase4)
        assert 'steps_override' in sig.parameters


# === Finetune optimizer tests (EXP-19: AdamW probe) ===

class TestFinetuneOptimizerSignature:
    def test_ntk_steps_accepts_finetune_optimizer(self):
        """compute_multi_step_update should accept finetune_optimizer param."""
        from experiments.ntk_steps import compute_multi_step_update
        sig = inspect.signature(compute_multi_step_update)
        assert 'finetune_optimizer' in sig.parameters
        assert sig.parameters['finetune_optimizer'].default == 'sgd'

    def test_ntk_steps_lora_accepts_finetune_optimizer(self):
        """compute_multi_step_update_lora should accept finetune_optimizer."""
        from experiments.ntk_steps import compute_multi_step_update_lora
        sig = inspect.signature(compute_multi_step_update_lora)
        assert 'finetune_optimizer' in sig.parameters
        assert sig.parameters['finetune_optimizer'].default == 'sgd'

    def test_run_experiment_b_accepts_finetune_optimizer(self):
        """run_single_config should accept finetune_optimizer param."""
        from experiments.run_experiment_b import run_single_config
        sig = inspect.signature(run_single_config)
        assert 'finetune_optimizer' in sig.parameters
        assert 'weight_decay' in sig.parameters

    def test_finetune_optimizer_choices_in_configs(self):
        """FINETUNE_OPTIMIZER_CHOICES should be defined in configs."""
        from experiments.configs import FINETUNE_OPTIMIZER_CHOICES
        assert 'sgd' in FINETUNE_OPTIMIZER_CHOICES
        assert 'adamw' in FINETUNE_OPTIMIZER_CHOICES


class TestFinetuneOptimizerFactory:
    def test_sgd_default(self):
        """_make_finetune_optimizer('sgd') returns SGD."""
        from experiments.ntk_steps import _make_finetune_optimizer
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        opt = _make_finetune_optimizer(params, lr=0.01, finetune_optimizer='sgd')
        assert isinstance(opt, torch.optim.SGD)

    def test_adamw(self):
        """_make_finetune_optimizer('adamw') returns AdamW."""
        from experiments.ntk_steps import _make_finetune_optimizer
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        opt = _make_finetune_optimizer(params, lr=1e-3, finetune_optimizer='adamw')
        assert isinstance(opt, torch.optim.AdamW)

    def test_invalid_raises(self):
        """Invalid optimizer name should raise ValueError."""
        from experiments.ntk_steps import _make_finetune_optimizer
        params = [torch.nn.Parameter(torch.randn(3, 3))]
        with pytest.raises(ValueError, match="Unknown finetune_optimizer"):
            _make_finetune_optimizer(params, lr=0.01, finetune_optimizer='rmsprop')


def _make_small_model():
    """Small model for fast testing."""
    from CreateModel import NeuralNetwork
    from experiments.configs import INPUT_DIM, OUTPUT_DIM
    return NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=[32],  # tiny for speed
        output_dim=OUTPUT_DIM,
        activation=nn.LeakyReLU(0.01),
        use_bias=False,
    )


def _get_test_data(n_per_class=1):
    """Load test data."""
    from experiments.data_utils import get_few_shot_mnist
    return get_few_shot_mnist(n_per_class, seed=42)


class TestAdamWProducesFiniteUpdate:
    def test_adamw_full_model(self):
        """AdamW fine-tuning should produce finite ΔW."""
        from experiments.ntk_steps import compute_multi_step_update
        model = _make_small_model()
        x, y, _, _ = _get_test_data()
        result = compute_multi_step_update(
            model, x.clone(), y.clone(), lr=1e-3, n_steps=3,
            finetune_optimizer='adamw',
        )
        for key, dw in result['delta_w'].items():
            assert torch.isfinite(dw).all(), f"Non-finite ΔW for {key}"
        assert len(result['loss_history']) == 3
        # Loss should decrease (at least not be NaN)
        assert all(not torch.isnan(torch.tensor(l)) for l in result['loss_history'])

    def test_adamw_lora(self):
        """AdamW LoRA fine-tuning should produce finite ΔW."""
        from experiments.ntk_steps import compute_multi_step_update_lora
        model = _make_small_model()
        x, y, _, _ = _get_test_data()
        result = compute_multi_step_update_lora(
            model, x.clone(), y.clone(), lr=1e-3, n_steps=3, rank=4,
            finetune_optimizer='adamw',
        )
        for key, dw in result['delta_w'].items():
            assert torch.isfinite(dw).all(), f"Non-finite ΔW for {key}"
        assert 'lora_B0' in result

    def test_sgd_adamw_produce_different_updates(self):
        """SGD and AdamW should produce different ΔW from the same init."""
        from experiments.ntk_steps import compute_multi_step_update
        from experiments.configs import INPUT_DIM, OUTPUT_DIM

        # Use identical models
        model_sgd = _make_small_model()
        model_adam = _make_small_model()
        model_adam.load_state_dict(copy.deepcopy(model_sgd.state_dict()))

        x, y, _, _ = _get_test_data()

        res_sgd = compute_multi_step_update(
            model_sgd, x.clone(), y.clone(), lr=0.01, n_steps=5,
            finetune_optimizer='sgd',
        )
        res_adam = compute_multi_step_update(
            model_adam, x.clone(), y.clone(), lr=0.01, n_steps=5,
            finetune_optimizer='adamw',
        )

        # At least one layer's ΔW should differ
        any_diff = False
        for key in res_sgd['delta_w']:
            if not torch.allclose(res_sgd['delta_w'][key], res_adam['delta_w'][key]):
                any_diff = True
                break
        assert any_diff, "SGD and AdamW produced identical ΔW"


# === N fine-tuning sweep tests (EXP-20) ===

class TestNPerClassData:
    def test_n_per_class_5(self):
        """Loading n_per_class=5 should give 10 total samples."""
        from experiments.data_utils import get_finetuning_data
        x, y, digits, indices = get_finetuning_data(5, seed=42)
        assert x.shape[0] == 10
        assert y.shape[0] == 10
        # Should have 5 of each class
        assert (y == 0).sum().item() == 5
        assert (y == 1).sum().item() == 5

    def test_n_per_class_10(self):
        """Loading n_per_class=10 should give 20 total samples."""
        from experiments.data_utils import get_finetuning_data
        x, y, digits, indices = get_finetuning_data(10, seed=42)
        assert x.shape[0] == 20
        assert y.shape[0] == 20

    def test_larger_n_runs_multi_step(self):
        """compute_multi_step_update should work with N>2."""
        from experiments.ntk_steps import compute_multi_step_update
        from experiments.data_utils import get_few_shot_mnist
        model = _make_small_model()
        x, y, _, _ = get_few_shot_mnist(5, seed=42)  # N=10
        result = compute_multi_step_update(
            model, x.clone(), y.clone(), lr=0.01, n_steps=2,
        )
        # Coefficients should have N=10 entries
        assert result['coefficients_at_init'].shape[0] == 10
        for key, dw in result['delta_w'].items():
            assert torch.isfinite(dw).all()


# === Track B5/B6 integration ===

class TestTrackB5B6Imports:
    def test_track_b5_exists(self):
        """Track B5 (N fine-tuning sweep) should be importable."""
        from experiments.run_sprint2c_sweep import run_track_b5
        sig = inspect.signature(run_track_b5)
        assert 'device' in sig.parameters
        assert 'extraction_epochs' in sig.parameters

    def test_track_b6_exists(self):
        """Track B6 (AdamW probe) should be importable."""
        from experiments.run_sprint2c_sweep import run_track_b6
        sig = inspect.signature(run_track_b6)
        assert 'device' in sig.parameters
        assert 'extraction_epochs' in sig.parameters

    def test_track_b7_exists(self):
        """Track B7 (consistency weight × T) should be importable."""
        from experiments.run_sprint2c_sweep import run_track_b7
        sig = inspect.signature(run_track_b7)
        assert 'device' in sig.parameters
        assert 'extraction_epochs' in sig.parameters


# === Phase 7 (linear LR schedule) tests ===

class TestPhase7LinearSchedule:
    def test_phase7_exists(self):
        """Phase 7 should be importable from sprint2b sweep."""
        from experiments.run_sprint2b_sweep import run_phase7
        sig = inspect.signature(run_phase7)
        assert 'steps_override' in sig.parameters

    def test_get_step_lr_linear(self):
        """Linear schedule: lr_t = lr * (1 - t/T)."""
        from experiments.ntk_steps import get_step_lr
        lr = 0.01
        n_steps = 10
        # step 0: full lr
        assert get_step_lr(lr, 0, n_steps, 'linear') == lr
        # step 5: half lr
        assert abs(get_step_lr(lr, 5, n_steps, 'linear') - lr * 0.5) < 1e-10
        # step 10: zero lr
        assert abs(get_step_lr(lr, 10, n_steps, 'linear')) < 1e-10

    def test_linear_schedule_runs(self):
        """compute_multi_step_update with linear schedule should run."""
        from experiments.ntk_steps import compute_multi_step_update
        model = _make_small_model()
        x, y, _, _ = _get_test_data()
        result = compute_multi_step_update(
            model, x.clone(), y.clone(), lr=0.01, n_steps=5,
            lr_schedule='linear',
        )
        for key, dw in result['delta_w'].items():
            assert torch.isfinite(dw).all()
        assert len(result['loss_history']) == 5


# === Consistency weight tests (B7) ===

class TestConsistencyWeightVariation:
    def test_different_weights_different_results(self):
        """Different consistency weights should produce different extraction."""
        from experiments.ntk_steps import compute_multi_step_update
        from experiments.ntk_extraction import run_ntk_extraction

        model = _make_small_model()
        x, y, _, _ = _get_test_data()
        result = compute_multi_step_update(
            model, x.clone(), y.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_small_model()
        model_0.load_state_dict(result['theta_0'])

        results_by_weight = {}
        for alpha in [0.0, 1.0]:
            x_recon, res = run_ntk_extraction(
                model_0, result['delta_w'], result['coefficients_at_init'],
                lr_train=0.01, n_steps=1, n_per_class=1,
                extraction_epochs=30, optimizer_type='sgd',
                free_coefficients=True, y=y,
                coeff_consistency_weight=alpha,
                verbose=False,
            )
            results_by_weight[alpha] = x_recon.detach().clone()

        # alpha=0 and alpha=1 should produce different reconstructions
        assert not torch.allclose(
            results_by_weight[0.0], results_by_weight[1.0], atol=1e-6
        ), "Different consistency weights produced identical results"


# === N-sweep bug fix test (B4) ===

class TestNSweepNoSizeMismatch:
    """Regression test: N-sweep with n_per_class > training N must not crash."""

    def test_n_sweep_runs_without_error(self):
        """run_ntk_extraction_n_sweep should handle n_per_class > training N."""
        from experiments.ntk_steps import compute_multi_step_update
        from experiments.ntk_extraction import run_ntk_extraction_n_sweep

        model = _make_small_model()
        x, y, _, _ = _get_test_data(n_per_class=1)  # N=2
        result = compute_multi_step_update(
            model, x.clone(), y.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_small_model()
        model_0.load_state_dict(result['theta_0'])

        # Sweep includes n_per_class > 1 (training used n_per_class=1)
        best_x, best_res, best_n, all_results = run_ntk_extraction_n_sweep(
            model_0, result['delta_w'],
            lr_train=0.01, n_steps=1,
            n_per_class_candidates=[1, 2, 3],
            oracle_coefficients=result['coefficients_at_init'],  # shape [2]
            oracle_y=y,
            free_coefficients=True,
            extraction_epochs=10,
            optimizer_type='sgd',
            coeff_consistency_weight=1.0,
            verbose=False,
        )
        assert best_x is not None
        assert best_n in [1, 2, 3]
        # All candidates should have results
        assert len(all_results) == 3
        # Shape must match the winning n_per_class
        assert best_x.shape[0] == best_n * 2

    def test_run_exp_b_with_n_sweep(self):
        """Full run_single_config with n_sweep=True must not crash."""
        from experiments.run_experiment_b import run_single_config
        # This used to crash with "size of tensor a (4) must match b (2)"
        res = run_single_config(
            n_steps=1, rank=None, n_per_class=1, seed=42,
            run_baseline=True,
            free_coefficients=True,
            n_sweep=True,
            extraction_epochs=10,
            optimizer_type='sgd',
            finetune_activation='leaky_relu',
            verbose=False,
        )
        assert 'full_extract_res' in res
        assert 'n_sweep_winner' in res['full_extract_res']
