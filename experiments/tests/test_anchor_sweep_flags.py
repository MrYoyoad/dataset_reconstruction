"""Part: run_anchor_sweep threads dataset + lr (Phase-C flowers anchor two-curve)."""
import inspect

from experiments.run_anchor_sweep import run_alpha_sweep


def test_run_alpha_sweep_has_dataset_and_lr():
    params = inspect.signature(run_alpha_sweep).parameters
    assert 'dataset' in params
    assert 'lr' in params
    assert params['dataset'].default == 'mnist'   # default preserves MNIST behavior
