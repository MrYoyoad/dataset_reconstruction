"""Tests for the Addition-2 smoothness axis (make_activation / ACTIVATION_CHOICES).

The sweep compares activations to test "does smoothness improve NTK survival and reconstruction?".
That only means anything if (a) every declared activation actually constructs, (b) the Softplus
beta knob really interpolates smooth -> ReLU, and (c) the "smooth" ones are genuinely smooth while
the controls are genuinely not. These tests pin all three.

CPU-only, per experiments/tests/README.md.
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from experiments.configs import ACTIVATION_CHOICES, SOFTPLUS_BETA_SWEEP, _softplus_name  # noqa: E402
from experiments.run_experiment_b import make_activation  # noqa: E402

# Activations smooth at 0 (bounded curvature). modified_relu is a custom sigmoid-modulated
# backward pass, excluded from the analytic check. NB: hardswish is smooth AT 0 (locally the
# quadratic x(x+3)/6) and only kinks at x=+/-3 — see its dedicated test. softplus_b50 has large
# but bounded curvature (beta/4 = 12.5), so only betas <= 10 are checked here.
SMOOTH_AT_0 = ['gelu', 'silu', 'softplus', 'mish', 'gelu_tanh', 'elu', 'tanh'] + \
              [_softplus_name(b) for b in SOFTPLUS_BETA_SWEEP if b <= 10]
KINKED_AT_0 = ['relu', 'leaky_relu']

# At h=1e-3 the symmetric second difference is ~1/h (~1000) at a true corner and O(1) where the
# function is twice-differentiable — a ~75x gap, so any threshold in between separates cleanly.
KINK_CURVATURE = 200.0    # kinked: second-difference magnitude above this
SMOOTH_CURVATURE = 50.0   # smooth: below this


def _second_diff(act, x0=0.0, h=1e-3):
    """|f(x0+h) - 2 f(x0) + f(x0-h)| / h^2 — a corner makes this ~1/h; smooth points stay O(1)."""
    x = torch.tensor([x0 - h, x0, x0 + h], dtype=torch.double)
    y = act(x)
    return ((y[2] - 2 * y[1] + y[0]) / (h * h)).abs().item()


class TestConstruction:
    def test_every_declared_activation_constructs(self):
        """A name in ACTIVATION_CHOICES that argparse accepts but make_activation rejects would
        fail only after the job reaches the GPU queue."""
        for name in ACTIVATION_CHOICES:
            act = make_activation(name)
            assert isinstance(act, torch.nn.Module), name
            out = act(torch.linspace(-3, 3, 7, dtype=torch.double))
            assert torch.isfinite(out).all(), f"{name} produced non-finite output"

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            make_activation('definitely_not_an_activation')

    def test_softplus_names_are_all_declared(self):
        for b in SOFTPLUS_BETA_SWEEP:
            assert _softplus_name(b) in ACTIVATION_CHOICES


class TestSoftplusKnob:
    def test_beta_parsed_correctly(self):
        for b in SOFTPLUS_BETA_SWEEP:
            act = make_activation(_softplus_name(b))
            assert float(act.beta) == pytest.approx(float(b))

    def test_large_beta_approaches_relu(self):
        """beta -> inf must converge to ReLU, otherwise the 'smoothness knob' framing is wrong."""
        x = torch.linspace(-3, 3, 61, dtype=torch.double)
        relu = torch.relu(x)
        errs = [(make_activation(_softplus_name(b))(x) - relu).abs().max().item()
                for b in sorted(SOFTPLUS_BETA_SWEEP)]
        assert errs == sorted(errs, reverse=True), f"error should shrink as beta grows: {errs}"
        assert errs[-1] < 0.05, "largest beta should closely approximate ReLU"

    def test_small_beta_is_far_from_relu(self):
        x = torch.linspace(-3, 3, 61, dtype=torch.double)
        err = (make_activation(_softplus_name(min(SOFTPLUS_BETA_SWEEP)))(x) - torch.relu(x)).abs().max()
        assert err.item() > 0.2, "smallest beta should be clearly smoother than ReLU"


class TestSmoothnessClassification:
    @pytest.mark.parametrize('name', SMOOTH_AT_0)
    def test_smooth_activations_have_bounded_curvature(self, name):
        assert _second_diff(make_activation(name)) < SMOOTH_CURVATURE, f"{name} looks kinked at 0"

    @pytest.mark.parametrize('name', KINKED_AT_0)
    def test_kinked_controls_have_blowup_curvature(self, name):
        """These are the controls. If a 'kinked' control were secretly smooth, the sweep could not
        attribute any effect to smoothness."""
        assert _second_diff(make_activation(name)) > KINK_CURVATURE, f"{name} is not kinked at 0"

    def test_hardswish_is_smooth_at_0_but_kinked_at_3(self):
        """hardswish is the shape-matched non-smooth control, but its corners are at +/-3, not 0.

        Pinning this stops anyone (including me — I got it wrong first) from treating it as
        kinked-at-0 in the smoothness analysis.
        """
        act = make_activation('hardswish')
        assert _second_diff(act, x0=0.0) < SMOOTH_CURVATURE, "hardswish is smooth at 0"
        assert _second_diff(act, x0=3.0) > KINK_CURVATURE, "hardswish should kink at x=3"

    def test_tanh_is_bounded_and_gelu_is_not(self):
        """tanh is the control separating smoothness from unboundedness."""
        x = torch.linspace(-50, 50, 101, dtype=torch.double)
        assert make_activation('tanh')(x).abs().max().item() <= 1.0 + 1e-6
        assert make_activation('gelu')(x).max().item() > 10.0

    def test_gelu_exact_and_tanh_approx_are_close_but_distinct(self):
        x = torch.linspace(-3, 3, 61, dtype=torch.double)
        diff = (make_activation('gelu')(x) - make_activation('gelu_tanh')(x)).abs().max().item()
        assert 0 < diff < 0.01, "tanh-approx GELU should be close to, but not identical to, exact"
