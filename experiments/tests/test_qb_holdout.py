"""Part: Phase-D Q-B — species holdout in the base trainer + seen/novel fine-tune filter."""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from experiments.data_utils import get_finetuning_data
from problems.flowers102_parity import get_balanced_data

torch.set_default_dtype(torch.float64)

HOLDOUT = [10, 11, 12, 13]   # two even + two odd species


def test_finetune_source_seen_excludes_holdout():
    _, _, species, _ = get_finetuning_data(2, seed=1, dataset='flowers32',
                                           source='seen', holdout_species=HOLDOUT)
    assert all(int(s) not in HOLDOUT for s in species)


def test_finetune_source_novel_only_holdout():
    _, _, species, _ = get_finetuning_data(1, seed=1, dataset='flowers32',
                                           source='novel', holdout_species=HOLDOUT)
    assert all(int(s) in HOLDOUT for s in species)


def test_base_trainer_holdout_excludes_species():
    # Synthetic dataset: image is filled with float(species) so we can read the species back.
    species_seq = [s for s in range(8) for _ in range(4)]   # 0..7 each x4
    ds = [(torch.full((3, 4, 4), float(s)), s) for s in species_seq]
    holdout = {2, 3}
    x0, y0 = get_balanced_data(None, ds, data_amount=4, holdout=holdout)
    got_species = {int(x0[i].flatten()[0].item()) for i in range(x0.shape[0])}
    assert got_species.isdisjoint(holdout)
    assert (y0 == 0).sum().item() == 2 and (y0 == 1).sum().item() == 2
