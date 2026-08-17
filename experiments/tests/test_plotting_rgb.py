"""Part: plotting renders RGB (3-channel) grids and keeps the grayscale path working."""
import os
import torch

from experiments.plotting import generate_experiment_b_figure

torch.set_default_dtype(torch.float64)


def _results(shape, dataset):
    n, c, h, w = 2, shape[0], shape[1], shape[2]
    g = lambda: torch.rand(n, c, h, w)
    return {
        'x_train': g(), 'x_recon_full': g(), 'x_recon_lora': g(), 'x_ctrl': g(),
        'ds_mean': torch.rand(1, c, h, w),
        'rank': 4, 'n_steps': 1, 'digits': [3, 58], 'dataset': dataset,
        'config': {'dataset': dataset, 'mode': 'ORACLE',
                   'full_params': 1000, 'lora_params': 100},
    }


def test_rgb_figure_written(tmp_path):
    out = generate_experiment_b_figure(_results((3, 32, 32), 'flowers32'),
                                       save_dir=str(tmp_path), base_name='rgb_test')
    assert any(f.endswith('.png') for f in os.listdir(tmp_path))


def test_grayscale_figure_written(tmp_path):
    out = generate_experiment_b_figure(_results((1, 28, 28), 'mnist'),
                                       save_dir=str(tmp_path), base_name='gray_test')
    assert any(f.endswith('.png') for f in os.listdir(tmp_path))
