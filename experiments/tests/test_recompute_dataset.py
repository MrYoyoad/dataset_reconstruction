"""Part: recompute_metrics carries the 'dataset' column through to the CSV rows."""
import torch

from experiments.recompute_metrics import rescore_file

torch.set_default_dtype(torch.float64)


def _save_fake_run(path, dataset):
    n, c, h, w = 2, 3, 32, 32
    torch.save({
        'x_train': torch.rand(n, c, h, w),
        'x_ctrl': torch.rand(n, c, h, w),
        'ds_mean': torch.rand(1, c, h, w),
        'x_recon_lora': torch.rand(n, c, h, w),
        'lora_diagnostics': {'weight_change': 0.02, 'delta_w_effective_rank': 2,
                             'feature_stability': 0.99, 'ntk_passed': True},
        'config': {'n_steps': 1, 'rank': 8, 'seed': 42, 'lr': 0.01,
                   'dataset': dataset, 'anchor_alpha': 0.0},
    }, path)


def test_dataset_column_from_config(tmp_path):
    p = tmp_path / 'exp_b_T1_flowers32_r8_s42_a149_gelu.pth'
    _save_fake_run(str(p), 'flowers32')
    rows = rescore_file(str(p))
    assert rows and rows[0]['dataset'] == 'flowers32'
    assert rows[0]['weight_change'] == 0.02


def test_dataset_column_fallback_from_filename(tmp_path):
    # config without 'dataset' -> recovered from the filename prefix
    p = tmp_path / 'exp_b_T1_flowers32_r8_s42_a149_silu.pth'
    torch.save({
        'x_train': torch.rand(2, 3, 32, 32),
        'ds_mean': torch.rand(1, 3, 32, 32),
        'x_recon_lora': torch.rand(2, 3, 32, 32),
        'config': {'n_steps': 1, 'rank': 8, 'seed': 42},
    }, str(p))
    rows = rescore_file(str(p))
    assert rows and rows[0]['dataset'] == 'flowers32'
