"""Part: configs.DATASET_SPECS — well-formed geometry + resolvable base-checkpoint paths."""
import os
from experiments.configs import DATASET_SPECS


def test_input_dim_matches_shape():
    for name, spec in DATASET_SPECS.items():
        c, h, w = spec['shape']
        assert spec['input_dim'] == c * h * w, f"{name}: input_dim != prod(shape)"


def test_flowers_native_specs():
    assert DATASET_SPECS['flowers32']['shape'] == (3, 32, 32)
    assert DATASET_SPECS['flowers32']['input_dim'] == 3072
    assert DATASET_SPECS['flowers64']['shape'] == (3, 64, 64)
    assert DATASET_SPECS['flowers64']['input_dim'] == 12288


def test_pretrained_paths_present():
    for name, spec in DATASET_SPECS.items():
        assert isinstance(spec['pretrained'], str) and spec['pretrained']
    assert DATASET_SPECS['flowers32']['pretrained'].endswith('weights-flowers32.pth')
    assert DATASET_SPECS['flowers64']['pretrained'].endswith('weights-flowers64.pth')
    # mnist/fashion/flowers all reuse the MNIST base
    assert os.path.basename(DATASET_SPECS['mnist']['pretrained']).startswith('weights-mnist')
    assert DATASET_SPECS['fashion']['pretrained'] == DATASET_SPECS['mnist']['pretrained']
