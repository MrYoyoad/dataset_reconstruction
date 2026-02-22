"""Tests for experiments/data_utils.py"""

import torch
import pytest

from experiments.data_utils import (
    get_few_shot_mnist,
    get_control_images_in_distribution,
    get_control_images_ood,
)

torch.set_default_dtype(torch.float64)


class TestFewShotShapes:
    def test_n1_shapes(self):
        x, y, digits, indices = get_few_shot_mnist(1)
        assert x.shape == (2, 1, 28, 28)
        assert y.shape == (2,)

    def test_n4_shapes(self):
        x, y, digits, indices = get_few_shot_mnist(4)
        assert x.shape == (8, 1, 28, 28)
        assert y.shape == (8,)


class TestFewShotLabels:
    def test_balanced_binary(self):
        x, y, digits, indices = get_few_shot_mnist(1)
        assert (y == 0).sum().item() == 1
        assert (y == 1).sum().item() == 1

    def test_balanced_binary_n4(self):
        x, y, digits, indices = get_few_shot_mnist(4)
        assert (y == 0).sum().item() == 4
        assert (y == 1).sum().item() == 4


class TestFewShotDeterministic:
    def test_same_seed_same_data(self):
        x1, y1, d1, i1 = get_few_shot_mnist(1, seed=42)
        x2, y2, d2, i2 = get_few_shot_mnist(1, seed=42)
        torch.testing.assert_close(x1, x2)
        assert d1 == d2
        assert i1 == i2


class TestFewShotDifferentSeeds:
    def test_different_seed_different_data(self):
        x1, _, _, i1 = get_few_shot_mnist(1, seed=42)
        x2, _, _, i2 = get_few_shot_mnist(1, seed=99)
        assert i1 != i2


class TestControlInDistSameDigit:
    def test_same_digit_labels(self):
        _, _, training_digits, _ = get_few_shot_mnist(1, seed=42)
        x_ctrl, y_ctrl, ctrl_digits = get_control_images_in_distribution(training_digits)
        assert sorted(ctrl_digits) == sorted(training_digits)


class TestControlInDistDifferentInstances:
    def test_not_identical(self):
        x_train, _, training_digits, _ = get_few_shot_mnist(1, seed=42)
        x_ctrl, _, _ = get_control_images_in_distribution(training_digits)
        # Controls should not be pixel-identical to training images
        for i in range(len(x_train)):
            assert not torch.allclose(x_train[i], x_ctrl[i]), "Control is identical to training!"


class TestControlInDistFromTestSet:
    def test_from_test_split(self):
        """Controls come from test set — verified by construction in the function
        (train=False in _load_mnist). Just verify we get valid data."""
        _, _, training_digits, _ = get_few_shot_mnist(1, seed=42)
        x_ctrl, y_ctrl, _ = get_control_images_in_distribution(training_digits)
        assert x_ctrl.shape[1:] == (1, 28, 28)
        assert x_ctrl.dtype == torch.float64


class TestControlOODSameDigit:
    def test_same_digit_labels(self):
        _, _, training_digits, _ = get_few_shot_mnist(1, seed=42)
        x_ctrl, y_ctrl, ctrl_digits = get_control_images_ood(training_digits)
        assert sorted(ctrl_digits) == sorted(training_digits)


class TestControlOODCorrectShape:
    def test_shape_28x28(self):
        _, _, training_digits, _ = get_few_shot_mnist(1, seed=42)
        x_ctrl, _, _ = get_control_images_ood(training_digits)
        assert x_ctrl.shape[1:] == (1, 28, 28)
        assert x_ctrl.dtype == torch.float64


class TestControlOODDifferentFromMNIST:
    def test_not_too_similar(self):
        x_train, _, training_digits, _ = get_few_shot_mnist(1, seed=42)
        x_ood, _, _ = get_control_images_ood(training_digits)
        # OOD should be meaningfully different from training (different dataset)
        for i in range(len(x_train)):
            l2 = (x_train[i] - x_ood[i]).pow(2).sum().sqrt().item()
            assert l2 > 0.5, f"OOD control too similar to training: L2={l2}"
