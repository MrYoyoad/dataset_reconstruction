"""Tests for experiments/train_lora.py"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from experiments.train_lora import train_lora, train_full_finetune, train_loop
from experiments.lora_wrapper import LoRALinear, apply_lora, compose_state_dict
from experiments.data_utils import get_few_shot_mnist
from experiments.configs import INPUT_DIM, OUTPUT_DIM, MODEL_HIDDEN_LIST

torch.set_default_dtype(torch.float64)


def _make_model():
    from CreateModel import NeuralNetwork
    activation = nn.ReLU()
    return NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=MODEL_HIDDEN_LIST,
        output_dim=OUTPUT_DIM,
        activation=activation,
        use_bias=False,
    )


def _get_data():
    """Get 2 easy-to-separate samples (1 per class)."""
    return get_few_shot_mnist(1, seed=42)


class TestLoRALossDecreases:
    def test_loss_decreases_over_100_steps(self):
        model = _make_model()
        x, y, _, _ = _get_data()
        result = train_lora(
            model, x.clone(), y.clone(), rank=8,
            epochs=100, threshold=0, eval_every=200, verbose=False,
        )
        assert result['loss_history'][-1] < result['loss_history'][0]


class TestFullFTLossDecreases:
    def test_loss_decreases_over_100_steps(self):
        model = _make_model()
        x, y, _, _ = _get_data()
        result = train_full_finetune(
            model, x.clone(), y.clone(),
            epochs=100, threshold=0, eval_every=200, verbose=False,
        )
        assert result['loss_history'][-1] < result['loss_history'][0]


class TestLoRAConvergenceTrivial:
    def test_converges_within_10k_steps(self):
        """On 2 very separable samples, LoRA rank 8 should converge to very low loss."""
        model = _make_model()
        x, y, _, _ = _get_data()
        result = train_lora(
            model, x.clone(), y.clone(), rank=8,
            epochs=10_000, threshold=1e-10, eval_every=5000, verbose=False,
        )
        # May not hit 1e-10 in 10K steps, but loss should be very small
        assert result['final_loss'] < 1e-3, f"Loss too high: {result['final_loss']}"


class TestFullFTConvergenceTrivial:
    def test_converges_within_10k_steps(self):
        model = _make_model()
        x, y, _, _ = _get_data()
        result = train_full_finetune(
            model, x.clone(), y.clone(),
            epochs=10_000, threshold=1e-10, eval_every=5000, verbose=False,
        )
        assert result['final_loss'] < 1e-3, f"Loss too high: {result['final_loss']}"


class TestOnlyLoRAParamsChange:
    def test_frozen_weights_unchanged(self):
        """After training, original W₀ weights should be unchanged."""
        model = _make_model()
        # Save original weights
        orig_weights = {
            i: model.layers[i].weight.data.clone()
            for i in range(len(model.layers))
            if isinstance(model.layers[i], nn.Linear)
        }

        x, y, _, _ = _get_data()
        train_lora(
            model, x.clone(), y.clone(), rank=8,
            epochs=500, threshold=0, eval_every=1000, verbose=False,
        )

        # Check frozen weights didn't change
        for i, layer in enumerate(model.layers):
            if isinstance(layer, LoRALinear):
                torch.testing.assert_close(
                    layer.frozen_weight.data, orig_weights[i],
                    msg=f"Layer {i} frozen weight changed during LoRA training!"
                )


class TestComposedWeightsDifferFromInit:
    def test_weights_changed_after_training(self):
        model = _make_model()
        init_sd = {k: v.clone() for k, v in model.state_dict().items()}

        x, y, _, _ = _get_data()
        train_lora(
            model, x.clone(), y.clone(), rank=8,
            epochs=500, threshold=0, eval_every=1000, verbose=False,
        )

        composed_sd = compose_state_dict(model)
        # At least one weight should have changed
        any_changed = False
        for key in composed_sd:
            if key in init_sd and not torch.allclose(composed_sd[key], init_sd[key]):
                any_changed = True
                break
        assert any_changed, "No weights changed after LoRA training!"


class TestDatasetMeanReturned:
    def test_ds_mean_shape(self):
        model = _make_model()
        x, y, _, _ = _get_data()
        result = train_lora(
            model, x.clone(), y.clone(), rank=8,
            epochs=10, threshold=0, verbose=False,
        )
        assert result['ds_mean'] is not None
        assert result['ds_mean'].shape == (1, 1, 28, 28)


class TestNoReduceMean:
    def test_ds_mean_none_when_disabled(self):
        model = _make_model()
        x, y, _, _ = _get_data()
        result = train_lora(
            model, x.clone(), y.clone(), rank=8,
            epochs=10, threshold=0, reduce_mean=False, verbose=False,
        )
        assert result['ds_mean'] is None
