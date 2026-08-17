"""Part: flowers102_parity base-trainer logic (labels + transform).

The full `Main.py --run_mode=train` end-to-end smoke is run separately as a Bash/GPU
Stage-0 check (it pays a torch-import + Flowers102-decode cost unsuitable for the unit suite).
Here we unit-test the pure, cheap pieces the trainer relies on.
"""
import os
import sys
import types
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from problems.flowers102_parity import create_labels, _flowers_transform

torch.set_default_dtype(torch.float64)


def test_create_labels_parity():
    y = create_labels([0, 1, 2, 3, 58, 73])
    assert y.tolist() == [0, 1, 0, 1, 0, 1]


def test_transform_rgb_native():
    args = types.SimpleNamespace(flowers_hw=32, flowers_gray=False)
    t = _flowers_transform(args)(Image.new('RGB', (50, 40)))
    assert tuple(t.shape) == (3, 32, 32)


def test_transform_rgb_64():
    args = types.SimpleNamespace(flowers_hw=64, flowers_gray=False)
    t = _flowers_transform(args)(Image.new('RGB', (100, 80)))
    assert tuple(t.shape) == (3, 64, 64)


def test_transform_grayscale():
    args = types.SimpleNamespace(flowers_hw=28, flowers_gray=True)
    t = _flowers_transform(args)(Image.new('RGB', (50, 40)))
    assert tuple(t.shape) == (1, 28, 28)
