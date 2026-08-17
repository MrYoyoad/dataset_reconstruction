"""Part: create_model / load_pretrained honor input_dim + hidden (flowers-native dims)."""
import torch

from experiments.run_experiment_b import create_model, load_pretrained

torch.set_default_dtype(torch.float64)


def test_forward_at_flowers32_dim():
    m = create_model(input_dim=3072, hidden=[16, 16])
    out = m(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 1)


def test_forward_at_flowers64_dim():
    m = create_model(input_dim=12288, hidden=[16, 16])
    out = m(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 1)


def test_default_is_mnist():
    m = create_model()
    out = m(torch.randn(3, 1, 28, 28))
    assert out.shape == (3, 1)


def test_load_pretrained_roundtrip(tmp_path):
    m = create_model(input_dim=3072, hidden=[16, 16])
    ckpt = tmp_path / 'fake.pth'
    torch.save({'state_dict': m.state_dict()}, ckpt)
    m2 = load_pretrained(pretrained_path=str(ckpt), input_dim=3072, hidden=[16, 16])
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.allclose(v1, v2)
