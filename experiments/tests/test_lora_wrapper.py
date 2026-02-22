"""Tests for experiments/lora_wrapper.py"""

import sys
import os
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from experiments.lora_wrapper import LoRALinear, apply_lora, compose_state_dict, save_composed_weights
from experiments.configs import INPUT_DIM, OUTPUT_DIM, MODEL_HIDDEN_LIST

torch.set_default_dtype(torch.float64)

# Expected state_dict keys for 784-1000-1000-1 architecture
EXPECTED_KEYS = ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.2.weight']
EXPECTED_SHAPES = {
    'layers.0.weight': (1000, 784),
    'layers.0.bias': (1000,),
    'layers.1.weight': (1000, 1000),
    'layers.2.weight': (1, 1000),
}


def _make_model():
    """Create a NeuralNetwork matching the MNIST architecture."""
    from CreateModel import NeuralNetwork
    activation = nn.ReLU()
    model = NeuralNetwork(
        input_dim=INPUT_DIM,
        hidden_dim_list=MODEL_HIDDEN_LIST,
        output_dim=OUTPUT_DIM,
        activation=activation,
        use_bias=False,
    )
    return model


def _make_input(batch_size=2):
    return torch.randn(batch_size, 1, 28, 28)


class TestLoRAInitPreservesOutput:
    def test_output_unchanged_after_apply(self):
        model = _make_model()
        x = _make_input()
        out_before = model(x).detach().clone()
        apply_lora(model, rank=8)
        out_after = model(x).detach()
        torch.testing.assert_close(out_before, out_after)


class TestComposeStateDictKeys:
    def test_keys_match(self):
        model = _make_model()
        apply_lora(model, rank=8)
        sd = compose_state_dict(model)
        assert sorted(sd.keys()) == sorted(EXPECTED_KEYS)


class TestComposeStateDictShapes:
    def test_shapes_match(self):
        model = _make_model()
        apply_lora(model, rank=8)
        sd = compose_state_dict(model)
        for key, expected_shape in EXPECTED_SHAPES.items():
            assert sd[key].shape == expected_shape, f"{key}: {sd[key].shape} != {expected_shape}"


class TestComposeStateDictDtype:
    def test_dtype_float64(self):
        model = _make_model()
        apply_lora(model, rank=8)
        sd = compose_state_dict(model)
        for key, tensor in sd.items():
            assert tensor.dtype == torch.float64, f"{key}: {tensor.dtype}"


class TestOnlyABTrainable:
    def test_frozen_params(self):
        model = _make_model()
        apply_lora(model, rank=8)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        for name in trainable:
            assert 'lora_A' in name or 'lora_B' in name, f"Unexpected trainable: {name}"

    def test_has_trainable_params(self):
        model = _make_model()
        apply_lora(model, rank=8)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestLoRAForwardWithNonzeroA:
    def test_output_changes(self):
        model = _make_model()
        x = _make_input()
        out_before = model(x).detach().clone()
        apply_lora(model, rank=8)
        # Manually set A to nonzero
        for layer in model.layers:
            if isinstance(layer, LoRALinear):
                layer.lora_A.data.fill_(0.1)
                break
        out_after = model(x).detach()
        assert not torch.allclose(out_before, out_after)


class TestComposeMatchesForward:
    def test_compose_numerically_matches(self):
        model = _make_model()
        x = _make_input()
        apply_lora(model, rank=8)
        # Set nonzero A so LoRA contributes
        for layer in model.layers:
            if isinstance(layer, LoRALinear):
                layer.lora_A.data.normal_()
        out_model = model(x).detach()
        # Load composed weights into a fresh model
        sd = compose_state_dict(model)
        fresh = _make_model()
        fresh.load_state_dict(sd)
        fresh.eval()
        out_fresh = fresh(x).detach()
        torch.testing.assert_close(out_model, out_fresh)


class TestDifferentRanks:
    @pytest.mark.parametrize("rank", [1, 4, 8, 64])
    def test_a_b_shapes(self, rank):
        model = _make_model()
        apply_lora(model, rank=rank)
        for layer in model.layers:
            if isinstance(layer, LoRALinear):
                assert layer.lora_A.shape[0] == rank
                assert layer.lora_B.shape[1] == rank


class TestSaveLoadRoundtrip:
    def test_save_load(self, tmp_path):
        model = _make_model()
        apply_lora(model, rank=8)
        # Train a tiny bit so weights differ
        for layer in model.layers:
            if isinstance(layer, LoRALinear):
                layer.lora_A.data.normal_(std=0.01)
        path = str(tmp_path / 'test_weights.pth')
        save_composed_weights(model, path)
        # Load into fresh model
        from common_utils.common import load_weights
        fresh = _make_model()
        load_weights(fresh, path, device='cpu')
        x = _make_input()
        torch.testing.assert_close(model(x).detach(), fresh(x).detach())
