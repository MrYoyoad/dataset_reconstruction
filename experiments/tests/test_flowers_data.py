"""Part: data_utils flowers32/flowers64 loaders (RGB native dims) + source filter guard.

Loads a couple of real Flowers102 test images (already on disk) — light, CPU-only.
"""
import torch
import pytest

from experiments.data_utils import (
    get_finetuning_data, get_control_images_in_distribution,
)

torch.set_default_dtype(torch.float64)


def test_flowers32_shape_and_labels():
    x, y, species, idx = get_finetuning_data(1, seed=42, dataset='flowers32')
    assert x.shape == (2, 3, 32, 32)
    assert y.shape == (2,)
    assert (y == 0).sum().item() == 1 and (y == 1).sum().item() == 1
    # binary label is species-index parity
    for sp, lbl in zip(species, y.tolist()):
        assert int(sp) % 2 == int(lbl)


def test_flowers64_shape():
    x, y, species, idx = get_finetuning_data(1, seed=42, dataset='flowers64')
    assert x.shape == (2, 3, 64, 64)


def test_controls_same_species_rgb():
    x, y, species, idx = get_finetuning_data(1, seed=7, dataset='flowers32')
    xc, yc, cs = get_control_images_in_distribution(species, dataset='flowers32')
    assert xc.shape == (2, 3, 32, 32)
    assert list(cs) == list(species)   # same species, returned in training order


def test_source_requires_holdout():
    with pytest.raises(ValueError):
        get_finetuning_data(1, dataset='flowers32', source='seen')  # no holdout_species
