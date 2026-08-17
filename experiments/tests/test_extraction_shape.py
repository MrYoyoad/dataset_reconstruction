"""Part: ntk_extraction threads input_shape into the x-hat init (RGB-safe); default unchanged."""
import torch
import torch.nn as nn

from CreateModel import NeuralNetwork
from experiments.ntk_extraction import run_ntk_extraction

torch.set_default_dtype(torch.float64)


def _tiny_model(input_dim):
    return NeuralNetwork(input_dim=input_dim, hidden_dim_list=[8], output_dim=1,
                         activation=nn.GELU(), use_bias=False).double()


def _zero_deltas(model):
    return {name: torch.zeros_like(p) for name, p in model.named_parameters()}


def test_custom_input_shape():
    m = _tiny_model(12)                     # 3*2*2 = 12
    x, _ = run_ntk_extraction(
        m, _zero_deltas(m), torch.zeros(2),
        lr_train=0.01, n_steps=1, n_per_class=1,
        extraction_epochs=2, optimizer_type='adam',
        input_shape=(3, 2, 2), device='cpu', verbose=False)
    assert x.shape == (2, 3, 2, 2)


def test_default_shape_regression():
    m = _tiny_model(784)                    # MNIST default
    x, _ = run_ntk_extraction(
        m, _zero_deltas(m), torch.zeros(2),
        lr_train=0.01, n_steps=1, n_per_class=1,
        extraction_epochs=2, optimizer_type='adam',
        device='cpu', verbose=False)
    assert x.shape == (2, 1, 28, 28)
