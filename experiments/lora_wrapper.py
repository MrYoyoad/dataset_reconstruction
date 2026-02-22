"""LoRA (Low-Rank Adaptation) wrapper for the existing NeuralNetwork FCN.

Provides LoRALinear module, apply_lora() to inject adapters, and
compose_state_dict() to merge W₀ + BA into a standard state_dict
compatible with the existing reconstruction pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank adapter BA.

    Effective weight: W_eff = W₀ + (alpha/rank) * B @ A
    At init: A = 0, so W_eff = W₀ (model starts unchanged).

    Args:
        in_features: input dimension
        out_features: output dimension
        rank: LoRA rank r
        alpha: scaling factor (default: rank, so scaling = 1)
        bias: whether to include bias (frozen from original)
    """

    def __init__(self, in_features, out_features, rank, alpha=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank

        # Frozen original weight
        self.frozen_weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.frozen_bias = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter('frozen_bias', None)

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))

        self._init_lora_weights()

    def _init_lora_weights(self):
        """Standard LoRA init: A = 0, B = Kaiming uniform."""
        nn.init.zeros_(self.lora_A)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x):
        w = self.frozen_weight + self.scaling * (self.lora_B @ self.lora_A)
        return F.linear(x, w, self.frozen_bias)

    def composed_weight(self):
        """Return the effective weight W₀ + scaling * B @ A."""
        return self.frozen_weight + self.scaling * (self.lora_B @ self.lora_A)

    def delta_weight(self):
        """Return the LoRA contribution: scaling * B @ A."""
        return self.scaling * (self.lora_B @ self.lora_A)


def apply_lora(model, rank, target_layer_indices=None, alpha=None):
    """Replace specified linear layers with LoRALinear wrappers.

    Freezes all original parameters and injects trainable A, B matrices.

    Args:
        model: NeuralNetwork instance (must have model.layers ModuleList)
        rank: LoRA rank
        target_layer_indices: which layers to adapt (default: all)
        alpha: LoRA alpha (default: rank)

    Returns:
        list of trainable LoRA parameters (for the optimizer)
    """
    if target_layer_indices is None:
        target_layer_indices = list(range(len(model.layers)))

    lora_params = []

    for idx in target_layer_indices:
        orig_layer = model.layers[idx]
        assert isinstance(orig_layer, nn.Linear), (
            f"Layer {idx} is {type(orig_layer)}, expected nn.Linear"
        )

        has_bias = orig_layer.bias is not None
        device = orig_layer.weight.device
        lora_layer = LoRALinear(
            in_features=orig_layer.in_features,
            out_features=orig_layer.out_features,
            rank=rank,
            alpha=alpha,
            bias=has_bias,
        )

        # Copy original weights (frozen)
        lora_layer.frozen_weight.data.copy_(orig_layer.weight.data)
        if has_bias:
            lora_layer.frozen_bias.data.copy_(orig_layer.bias.data)

        # Move to same device as original layer
        lora_layer = lora_layer.to(device)

        # Replace in the model
        model.layers[idx] = lora_layer
        lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad_(False)

    return lora_params


def compose_state_dict(model):
    """Merge LoRA weights into a standard state_dict.

    Walks model.layers. For LoRALinear layers, composes W = W₀ + scaling*B@A.
    For regular nn.Linear layers, copies weights as-is.

    Returns dict with keys matching the original NeuralNetwork format:
    'layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.2.weight', ...
    """
    state_dict = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer, LoRALinear):
            state_dict[f'layers.{i}.weight'] = layer.composed_weight().detach().clone()
            if layer.frozen_bias is not None:
                state_dict[f'layers.{i}.bias'] = layer.frozen_bias.detach().clone()
        elif isinstance(layer, nn.Linear):
            state_dict[f'layers.{i}.weight'] = layer.weight.detach().clone()
            if layer.bias is not None:
                state_dict[f'layers.{i}.bias'] = layer.bias.detach().clone()
        else:
            raise TypeError(f"Unexpected layer type at index {i}: {type(layer)}")
    return state_dict


def get_lora_param_count(model):
    """Count total trainable LoRA parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_composed_weights(model, path, epoch=None):
    """Save composed weights in the format expected by common_utils.load_weights()."""
    sd = compose_state_dict(model)
    torch.save({'state_dict': sd, 'epoch': epoch, 'batch': None}, path)
