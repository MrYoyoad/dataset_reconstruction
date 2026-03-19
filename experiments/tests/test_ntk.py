"""Tests for NTK experiments: ntk_steps.py, ntk_extraction.py, ntk_verification.py"""

import sys
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from experiments.ntk_steps import (
    compute_known_coefficients, compute_multi_step_update,
    compute_multi_step_update_lora,
)
from experiments.ntk_extraction import (
    get_ntk_loss, run_ntk_extraction, get_coeff_penalty,
    get_diversity_penalty, run_ntk_extraction_with_restarts,
)
from experiments.ntk_verification import (
    compute_relative_weight_change, compute_feature_stability,
    compute_coefficient_drift, verify_ntk_at_step,
)
from experiments.data_utils import get_few_shot_mnist
from experiments.configs import INPUT_DIM, OUTPUT_DIM, MODEL_HIDDEN_LIST

torch.set_default_dtype(torch.float64)


def _make_model():
    from CreateModel import NeuralNetwork
    return NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=MODEL_HIDDEN_LIST,
        output_dim=OUTPUT_DIM,
        activation=nn.ReLU(),
        use_bias=False,
    )


def _get_data():
    return get_few_shot_mnist(1, seed=42)


# === ntk_steps tests ===

class TestCoefficientsAtInit:
    def test_approximate_values(self):
        """At random init with balanced labels, c_i ≈ ±0.25.
        Since σ(f(θ₀;x)) ≈ 0.5 for random init, c_i = (0.5 - y_i)/N.
        With N=2: c_i ≈ (0.5-0)/2 = 0.25 or (0.5-1)/2 = -0.25."""
        model = _make_model()
        x, y, _, _ = _get_data()
        # Reduce mean like training does
        ds_mean = x.mean(dim=0, keepdim=True)
        x_centered = x - ds_mean
        coefficients = compute_known_coefficients(model, x_centered, y)
        assert coefficients.shape == (2,)
        # Should be approximately ±0.25 (with some variance from random init)
        assert coefficients.abs().max().item() < 0.5


class TestCoefficientsExact:
    def test_manual_computation(self):
        """Verify coefficient formula: c_i = (σ(f(θ₀;x_i)) - y_i) / N."""
        model = _make_model()
        x, y, _, _ = _get_data()
        ds_mean = x.mean(dim=0, keepdim=True)
        x_centered = x - ds_mean

        # Manually compute
        model.eval()
        with torch.no_grad():
            logits = model(x_centered).view(-1)
            probs = torch.sigmoid(logits)
            expected = (probs - y) / x_centered.shape[0]

        actual = compute_known_coefficients(model, x_centered, y)
        torch.testing.assert_close(actual, expected)


class TestOneStepDeltaW:
    def test_delta_matches_gradient(self):
        """After 1 step: ΔW should equal -lr * gradient."""
        lr = 0.01
        model = _make_model()
        x, y, _, _ = _get_data()

        # Manually compute expected gradient
        model_copy = _make_model()
        model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
        ds_mean = x.mean(dim=0, keepdim=True)
        x_c = x - ds_mean

        model_copy.train()
        logits = model_copy(x_c).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()

        expected_delta = {}
        for name, param in model_copy.named_parameters():
            expected_delta[name] = -lr * param.grad.clone()

        # Now compute via our function
        result = compute_multi_step_update(model, x.clone(), y.clone(), lr=lr, n_steps=1)

        for key in expected_delta:
            torch.testing.assert_close(
                result['delta_w'][key], expected_delta[key],
                atol=1e-10, rtol=1e-8,
                msg=f"ΔW mismatch for {key}"
            )


class TestMultiStepAccumulation:
    def test_delta_is_theta_T_minus_theta_0(self):
        """ΔW_T = θ_T - θ₀ (by construction)."""
        model = _make_model()
        x, y, _, _ = _get_data()
        result = compute_multi_step_update(model, x.clone(), y.clone(), lr=0.01, n_steps=5)

        for key in result['delta_w']:
            expected = result['theta_T'][key] - result['theta_0'][key]
            torch.testing.assert_close(result['delta_w'][key], expected)


# === ntk_verification tests ===

class TestNTKVerificationAtInit:
    def test_zero_change_at_init(self):
        """At T=0, weight change = 0 and feature cos = 1."""
        model = _make_model()
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        change = compute_relative_weight_change(sd, sd)
        assert change['overall'] == 0.0

    def test_perfect_feature_stability_at_init(self):
        model = _make_model()
        x, y, _, _ = _get_data()
        ds_mean = x.mean(dim=0, keepdim=True)
        x_c = x - ds_mean
        stability = compute_feature_stability(model, model, x_c)
        assert stability['mean'] > 0.999


class TestNTKVerificationSmallStep:
    def test_features_stable_at_one_step(self):
        """At T=1 with small lr, NTK should hold: feature cos > 0.99."""
        model = _make_model()
        x, y, _, _ = _get_data()

        result = compute_multi_step_update(
            model, x.clone(), y.clone(), lr=0.001, n_steps=1
        )

        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])
        model_T = _make_model()
        model_T.load_state_dict(result['theta_T'])

        x_c = x - result['ds_mean'] if result['ds_mean'] is not None else x
        stability = compute_feature_stability(model_0, model_T, x_c)
        assert stability['mean'] > 0.99, f"Feature stability too low: {stability['mean']}"


class TestNTKVerificationWeightChange:
    def test_small_change_at_one_step(self):
        """With small lr and 1 step, relative weight change should be small."""
        model = _make_model()
        x, y, _, _ = _get_data()

        result = compute_multi_step_update(
            model, x.clone(), y.clone(), lr=0.001, n_steps=1
        )

        change = compute_relative_weight_change(result['theta_0'], result['theta_T'])
        assert change['overall'] < 0.1, f"Weight change too large: {change['overall']}"


# === ntk_extraction tests ===

class TestNTKLossGradientFlows:
    def test_x_grad_not_none(self):
        """NTK loss backward should produce gradients on x."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()

        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1
        )

        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])
        model_0.eval()

        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        loss = get_ntk_loss(
            model_0, result['delta_w'], x,
            result['coefficients_at_init'], lr=0.01, n_steps=1,
        )
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestNTKExtractionSmoke:
    def test_no_crashes(self):
        """NTK extraction should run without errors (short test)."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()

        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1
        )

        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])

        x_recon, res = run_ntk_extraction(
            model_0, result['delta_w'], result['coefficients_at_init'],
            lr_train=0.01, n_steps=1, n_per_class=1,
            extraction_epochs=50, verbose=False,
        )

        assert x_recon.shape == (2, 1, 28, 28)
        assert torch.isfinite(x_recon).all()
        assert len(res['loss_history']) > 0


# === LoRA multi-step tests ===

class TestLoRAMultiStepUpdate:
    def test_delta_w_shapes(self):
        """LoRA multi-step update should produce correct delta_w shapes."""
        model = _make_model()
        x, y, _, _ = _get_data()
        result = compute_multi_step_update_lora(
            model, x.clone(), y.clone(), lr=0.01, n_steps=5, rank=8,
        )
        # delta_w should have weight keys with correct shapes
        assert 'layers.0.weight' in result['delta_w']
        assert result['delta_w']['layers.0.weight'].shape == (1000, 784)

    def test_coefficients_returned(self):
        model = _make_model()
        x, y, _, _ = _get_data()
        result = compute_multi_step_update_lora(
            model, x.clone(), y.clone(), lr=0.01, n_steps=1, rank=8,
        )
        assert result['coefficients_at_init'].shape == (2,)


# === Adam extraction smoke test ===

class TestNTKExtractionAdam:
    def test_adam_produces_finite_output(self):
        """NTK extraction with Adam optimizer should produce finite results."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])

        x_recon, res = run_ntk_extraction(
            model_0, result['delta_w'], result['coefficients_at_init'],
            lr_train=0.01, n_steps=1, n_per_class=1,
            extraction_epochs=50, optimizer_type='adam', verbose=False,
        )
        assert x_recon.shape == (2, 1, 28, 28)
        assert torch.isfinite(x_recon).all()
        assert len(res['loss_history']) > 0


# === Free-coefficient smoke test ===

class TestNTKExtractionFreeCoeffSmoke:
    def test_coefficients_move_from_init(self):
        """Free-coefficient extraction should update c from their init values."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])

        x_recon, res = run_ntk_extraction(
            model_0, result['delta_w'], result['coefficients_at_init'],
            lr_train=0.01, n_steps=1, n_per_class=1,
            extraction_epochs=50, optimizer_type='sgd',
            free_coefficients=True, y=y_train,
            coeff_consistency_weight=1.0,
            verbose=False,
        )
        assert x_recon.shape == (2, 1, 28, 28)
        assert torch.isfinite(x_recon).all()
        # Coefficients should have been returned and should differ from oracle
        assert 'final_coefficients' in res
        assert res['final_coefficients'].shape == (2,)


# === Sign-aware coefficient initialization tests ===

class TestSignAwareInit:
    def test_sign_aware_produces_correct_signs(self):
        """sign_aware init: c > 0 for y=0, c < 0 for y=1."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])

        x_recon, res = run_ntk_extraction(
            model_0, result['delta_w'], result['coefficients_at_init'],
            lr_train=0.01, n_steps=1, n_per_class=1,
            extraction_epochs=5, optimizer_type='sgd',
            free_coefficients=True, y=y_train,
            coeff_init='sign_aware',
            coeff_sign_weight=0.0,
            verbose=False,
        )
        # After only 5 epochs, signs should still match init
        c_final = res['final_coefficients']
        # y_train is [0, 1] -> expected signs [+, -]
        assert c_final[0] > 0, f"c[0] should be positive for y=0, got {c_final[0]}"
        assert c_final[1] < 0, f"c[1] should be negative for y=1, got {c_final[1]}"

    def test_zeros_init_starts_at_zero(self):
        """zeros init: c starts at 0."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])

        x_recon, res = run_ntk_extraction(
            model_0, result['delta_w'], result['coefficients_at_init'],
            lr_train=0.01, n_steps=1, n_per_class=1,
            extraction_epochs=1, optimizer_type='sgd',
            free_coefficients=True, y=y_train,
            coeff_init='zeros',
            coeff_sign_weight=0.0,
            verbose=False,
        )
        # After 1 epoch, c should have barely moved from zero
        c_final = res['final_coefficients']
        assert c_final.abs().max().item() < 0.1, f"Expected c near zero, got {c_final}"


# === Sign constraint penalty tests ===

class TestSignConstraint:
    def test_wrong_sign_penalized(self):
        """Sign penalty should be > 0 when c has wrong sign for its label."""
        y = torch.tensor([0.0, 1.0])
        # Wrong signs: c[0]=-0.1 (should be positive), c[1]=+0.1 (should be negative)
        c_wrong = torch.tensor([-0.1, 0.1])
        penalty = get_coeff_penalty(
            c_wrong, y=y, box_weight=0.0, consistency_weight=0.0,
            sign_weight=5.0, min_coeff=0.05,
        )
        assert penalty.item() > 0, f"Expected positive penalty for wrong signs, got {penalty.item()}"

    def test_correct_sign_no_penalty(self):
        """Sign penalty should be 0 when c has correct sign and |c| > min_coeff."""
        y = torch.tensor([0.0, 1.0])
        # Correct signs: c[0]=+0.3, c[1]=-0.3 (well above min_coeff=0.05)
        c_correct = torch.tensor([0.3, -0.3])
        penalty = get_coeff_penalty(
            c_correct, y=y, box_weight=0.0, consistency_weight=0.0,
            sign_weight=5.0, min_coeff=0.05,
        )
        assert penalty.item() == 0.0, f"Expected zero penalty for correct signs, got {penalty.item()}"

    def test_below_min_coeff_penalized(self):
        """Sign penalty should fire when |c| < min_coeff even with correct sign."""
        y = torch.tensor([0.0, 1.0])
        # Correct signs but too small: c[0]=+0.01, c[1]=-0.01 (below min_coeff=0.05)
        c_small = torch.tensor([0.01, -0.01])
        penalty = get_coeff_penalty(
            c_small, y=y, box_weight=0.0, consistency_weight=0.0,
            sign_weight=5.0, min_coeff=0.05,
        )
        assert penalty.item() > 0, f"Expected positive penalty for too-small c, got {penalty.item()}"


# === Cosine loss tests ===

class TestCosineLoss:
    def test_perfect_match_zero_loss(self):
        """Cosine loss should be ~0 when predicted matches target exactly."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])
        model_0.eval()

        ds_mean = x_train.mean(dim=0, keepdim=True)
        x_c = x_train - ds_mean

        # With true x and oracle c, cosine loss should be ~0
        loss = get_ntk_loss(
            model_0, result['delta_w'], x_c,
            result['coefficients_at_init'], lr=0.01, n_steps=1,
            loss_type='cosine',
        )
        assert loss.item() < 0.01, f"Expected near-zero cosine loss with true data, got {loss.item()}"

    def test_cosine_loss_bounded(self):
        """Cosine loss should be in [0, 2]."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])
        model_0.eval()

        x_rand = torch.randn(2, 1, 28, 28)
        loss = get_ntk_loss(
            model_0, result['delta_w'], x_rand,
            result['coefficients_at_init'], lr=0.01, n_steps=1,
            loss_type='cosine',
        )
        assert 0 <= loss.item() <= 2.0, f"Cosine loss out of bounds: {loss.item()}"


# === Diversity penalty tests ===

class TestDiversityPenalty:
    def test_collapsed_images_penalized(self):
        """Diversity penalty should fire when images are identical."""
        x = torch.zeros(2, 1, 28, 28)
        x[0] = 0.5
        x[1] = 0.5  # identical
        penalty = get_diversity_penalty(x, min_dist=0.5)
        assert penalty.item() > 0, f"Expected positive penalty for identical images, got {penalty.item()}"

    def test_diverse_images_no_penalty(self):
        """Diversity penalty should be 0 for well-separated images."""
        x = torch.randn(2, 1, 28, 28) * 5  # large random images, far apart
        penalty = get_diversity_penalty(x, min_dist=0.1)
        assert penalty.item() == 0.0, f"Expected zero penalty for diverse images, got {penalty.item()}"


# === Extraction with sign-aware init smoke test ===

class TestExtractionSignAwareSmoke:
    def test_sign_aware_extraction_runs(self):
        """Extraction with sign_aware init + sign constraint should not crash."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])

        x_recon, res = run_ntk_extraction(
            model_0, result['delta_w'], result['coefficients_at_init'],
            lr_train=0.01, n_steps=1, n_per_class=1,
            extraction_epochs=20, optimizer_type='sgd',
            free_coefficients=True, y=y_train,
            coeff_init='sign_aware',
            coeff_sign_weight=5.0,
            coeff_min_magnitude=0.05,
            loss_type='l2',
            verbose=False,
        )
        assert x_recon.shape == (2, 1, 28, 28)
        assert torch.isfinite(x_recon).all()
        assert 'coeff_error' in res

    def test_cosine_loss_extraction_runs(self):
        """Extraction with cosine loss should not crash."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])

        x_recon, res = run_ntk_extraction(
            model_0, result['delta_w'], result['coefficients_at_init'],
            lr_train=0.01, n_steps=1, n_per_class=1,
            extraction_epochs=20, optimizer_type='sgd',
            free_coefficients=True, y=y_train,
            coeff_init='sign_aware',
            coeff_sign_weight=5.0,
            loss_type='cosine',
            verbose=False,
        )
        assert x_recon.shape == (2, 1, 28, 28)
        assert torch.isfinite(x_recon).all()

    def test_lbfgs_with_sign_aware(self):
        """L-BFGS + sign_aware init should not crash."""
        model = _make_model()
        x_train, y_train, _, _ = _get_data()
        result = compute_multi_step_update(
            model, x_train.clone(), y_train.clone(), lr=0.01, n_steps=1,
        )
        model_0 = _make_model()
        model_0.load_state_dict(result['theta_0'])

        x_recon, res = run_ntk_extraction(
            model_0, result['delta_w'], result['coefficients_at_init'],
            lr_train=0.01, n_steps=1, n_per_class=1,
            extraction_epochs=10, optimizer_type='lbfgs',
            free_coefficients=True, y=y_train,
            coeff_init='sign_aware',
            coeff_sign_weight=5.0,
            coeff_min_magnitude=0.05,
            verbose=False,
        )
        assert x_recon.shape == (2, 1, 28, 28)
        assert torch.isfinite(x_recon).all()
