"""Part: run_single_config end-to-end on flowers32 (fake theta_0, tiny arch, CPU).

Exercises the whole plumbing — dim/theta_0 threading, LoRA + full paths, anchor, metrics,
diagnostics, RGB shapes — without needing the real trained base model.
"""
import torch

from experiments import configs
from experiments.run_experiment_b import run_single_config, create_model

torch.set_default_dtype(torch.float64)


def test_flowers32_end_to_end(tmp_path, monkeypatch):
    # Shrink the flowers32 arch so the CPU smoke is fast; keep input_dim=3072 so the real
    # flowers test images (3x32x32) still flatten in.
    small = dict(configs.DATASET_SPECS['flowers32'])
    small['hidden'] = [32, 32]
    monkeypatch.setitem(configs.DATASET_SPECS, 'flowers32', small)

    m = create_model(input_dim=small['input_dim'], hidden=small['hidden'])
    ckpt = tmp_path / 'fake_flowers32.pth'
    torch.save({'state_dict': m.state_dict()}, ckpt)

    res = run_single_config(
        n_steps=1, rank=4, n_per_class=1, seed=42,
        dataset='flowers32', pretrained_path=str(ckpt),
        extraction_epochs=2, device='cpu', verbose=False)

    assert res['dataset'] == 'flowers32'
    assert tuple(res['input_shape']) == (3, 32, 32)
    assert res['x_recon_lora'].shape == (2, 3, 32, 32)
    assert res['x_recon_full'].shape == (2, 3, 32, 32)
    assert 'ssim' in res['lora_metrics']
    assert 'weight_change' in res['lora_diagnostics']
    assert 'anchor_alpha' in res
