"""Unit tests for the face-structure prior."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.face_prior import (  # noqa: E402
    compute_face_prior,
    face_detection_score,
    face_prior_ramp,
    load_face_prior,
    _layout_penalty,
)


_DEVICE = 'cpu'
_FACE1 = os.path.join(os.path.dirname(__file__), '..', '..',
                      'data', 'faces', 'face1.jpg')


@pytest.fixture(scope='module')
def prior():
    return load_face_prior(model='auto', device=_DEVICE)


def _load_face_pixel(path: str) -> torch.Tensor:
    """Load JPG, resize to 224x224, return [1, 3, 224, 224] in [0, 1]."""
    from PIL import Image
    import torchvision.transforms as T
    img = Image.open(path).convert('RGB').resize((224, 224))
    return T.ToTensor()(img).unsqueeze(0)


# ---------------- 1. Loader ----------------

def test_loader_returns_models(prior):
    assert 'detector' in prior
    assert prior['detector'] is not None
    assert isinstance(prior['detector'], torch.nn.Module)
    # frozen and in eval mode
    assert not prior['detector'].training
    for p in prior['detector'].parameters():
        assert p.requires_grad is False


# ---------------- 2/3. Real face vs noise ----------------

def test_real_face_low_loss(prior):
    if not os.path.exists(_FACE1):
        pytest.skip(f"missing {_FACE1}")
    x = _load_face_pixel(_FACE1)
    out = compute_face_prior(x, prior)
    assert out['total'].item() < 0.3, (
        f"real face should give low prior loss, got {out['total'].item():.4f}"
    )
    # Presence loss should be near 0 (c1 ~= 1.0 -> -log(1) = 0)
    assert out['presence'].item() < 0.1


def test_random_noise_high_loss(prior):
    torch.manual_seed(0)
    x = torch.rand(1, 3, 224, 224)
    out = compute_face_prior(x, prior)
    assert out['total'].item() > 0.5, (
        f"random noise should give high prior loss, "
        f"got {out['total'].item():.4f}"
    )


# ---------------- 4. Differentiability ----------------

def test_face_loss_differentiable(prior):
    if not os.path.exists(_FACE1):
        pytest.skip(f"missing {_FACE1}")
    x = _load_face_pixel(_FACE1)
    x.requires_grad_(True)
    out = compute_face_prior(x, prior)
    out['total'].backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0, "no gradient flowed back to x"


# ---------------- 5. Detection score has no grad ----------------

def test_face_det_score_no_grad(prior):
    if not os.path.exists(_FACE1):
        pytest.skip(f"missing {_FACE1}")
    x = _load_face_pixel(_FACE1).requires_grad_(True)
    score = face_detection_score(x, prior)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # x.grad should still be None — face_detection_score must not pollute graph
    assert x.grad is None


# ---------------- 6. Warmup ramp ----------------

def test_warmup_ramp():
    warm, ramp = 100, 50
    assert face_prior_ramp(0, warm, ramp) == 0.0
    assert face_prior_ramp(warm - 1, warm, ramp) == 0.0
    assert face_prior_ramp(warm, warm, ramp) == 0.0
    # mid-ramp
    mid = face_prior_ramp(warm + 25, warm, ramp)
    assert abs(mid - 0.5) < 1e-6
    # end of ramp
    assert face_prior_ramp(warm + ramp, warm, ramp) == 1.0
    assert face_prior_ramp(warm + ramp + 100, warm, ramp) == 1.0
    # zero ramp -> immediate switch
    assert face_prior_ramp(warm, warm, 0) == 1.0


# ---------------- 7/8. Layout penalty unit tests ----------------

def _canonical_detection() -> torch.Tensor:
    """Synthesize a 15-d detection row for a canonical face inside a 100x100 bbox."""
    # bbox [10, 10, 110, 110], width=100, height=100
    eyes_y = 10 + 35   # at 0.35 in bbox
    nose_y = 10 + 50
    mouth_y = 10 + 75
    eye_l_x = 10 + 35
    eye_r_x = 10 + 65
    nose_x = 10 + 50
    mouth_l_x = 10 + 40
    mouth_r_x = 10 + 60
    return torch.tensor([
        10.0, 10.0, 110.0, 110.0,                # bbox
        eye_l_x, eyes_y, eye_r_x, eyes_y,        # eyes
        nose_x, nose_y,                           # nose
        mouth_l_x, mouth_y, mouth_r_x, mouth_y,  # mouth corners
        0.99,                                    # confidence
    ])


def test_layout_penalty_zero_on_canonical_face():
    box = _canonical_detection()
    p = _layout_penalty(box).item()
    assert p < 1e-5, f"canonical face should give 0 layout penalty, got {p:.4f}"


def test_layout_penalty_positive_on_eye_mouth_swap():
    box = _canonical_detection()
    # Swap eye and mouth y-coords (mouth above eyes)
    box = box.clone()
    box[5] = 75.0   # eye_l_y
    box[7] = 75.0   # eye_r_y
    box[11] = 35.0  # mouth_l_y
    box[13] = 35.0  # mouth_r_y
    p = _layout_penalty(box).item()
    assert p > 0.1, (
        f"swapped layout should give significant penalty, got {p:.4f}"
    )


# ---------------- 9. Pipeline plumbing smoke test ----------------

def test_pipeline_accepts_face_args():
    """invert_gradient accepts the face-prior kwargs without errors."""
    import inspect
    from experiments.phase0_vit_inversion import invert_gradient, run_phase0
    sig = inspect.signature(invert_gradient)
    for k in ('face_weight', 'face_prior', 'face_layout_weight',
             'face_sym_weight', 'face_warmup_iters', 'face_ramp_iters',
             'cos_weight', 'partial_save_fn'):
        assert k in sig.parameters, f"invert_gradient missing {k}"
    sig2 = inspect.signature(run_phase0)
    for k in ('face_weight', 'face_layout_weight', 'face_sym_weight',
             'face_warmup_iters', 'face_ramp_iters', 'face_model',
             'cos_weight'):
        assert k in sig2.parameters, f"run_phase0 missing {k}"


# ---------------- 10. Partial save callback ----------------

def test_partial_save_fn_called_each_restart(tmp_path):
    """invert_gradient calls partial_save_fn once per completed restart.

    Uses a 2-layer MLP toy model so the test stays fast on CPU. Verifies the
    callback is invoked, gets a valid x tensor + best_cos float, and that a
    file written by the callback persists on disk.
    """
    import torch
    import torch.nn as nn
    from experiments.phase0_vit_inversion import invert_gradient

    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 4), nn.ReLU(),
                          nn.Linear(4, 2)).eval()
    labels = torch.tensor([[1.0]])
    x = torch.randn(1, 3, 8, 8)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
    grads = {n: g.detach() for n, g in
             zip([n for n, _ in model.named_parameters()],
                 torch.autograd.grad(loss, list(model.parameters())))}
    for p in model.parameters():
        p.requires_grad_(True)

    calls = []
    save_path = tmp_path / 'partial.pth'

    def cb(r, x_best, cos_best, hist):
        calls.append((r, float(cos_best), tuple(x_best.shape)))
        torch.save({'restart': r, 'x_recon': x_best.cpu(),
                    'best_cos_sim': cos_best}, save_path)

    invert_gradient(
        model, grads, labels, x.shape,
        n_iters=5, n_restarts=3, lr=0.01,
        tv_weight=0.0, partial_save_fn=cb,
        device='cpu', verbose=False,
    )
    assert len(calls) == 3, f"expected 3 callback invocations, got {len(calls)}"
    assert all(c[2] == (1, 3, 8, 8) for c in calls)
    assert save_path.exists()
    loaded = torch.load(save_path, map_location='cpu', weights_only=False)
    assert loaded['restart'] == 2  # last restart index
    assert loaded['x_recon'].shape == (1, 3, 8, 8)
