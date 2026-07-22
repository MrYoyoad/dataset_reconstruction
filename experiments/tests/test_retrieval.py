"""Tests for experiments/retrieval_metric.py.

The retrieval metric is our instance-level, background-robust leakage measure. These pin the two
things it must get right: a correct reconstruction retrieves the correct target (top-1 = 1), and a
mismatched one does not (top-1 ~ chance). CPU-only.
"""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

from experiments.retrieval_metric import (  # noqa: E402
    retrieval_scores, similarity_matrix, METRIC_SPACES,
)


def _blob(n=6, size=28, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, size, size, generator=g)
    x = torch.nn.functional.avg_pool2d(x, 5, stride=1, padding=2)
    return (x - x.amin()) / (x.amax() - x.amin())


class TestRetrievalScores:
    def test_perfect_matrix_scores_one(self):
        # diagonal is the max in every row -> top-1 = 1, mean rank = 1
        sim = torch.eye(5) * 2 - 1
        s = retrieval_scores(sim)
        assert s['top1_acc'] == 1.0
        assert s['mean_rank'] == 1.0
        assert s['random_top1'] == pytest.approx(0.2)

    def test_worst_matrix_scores_low(self):
        # diagonal is the minimum in every row -> true target ranked last
        sim = torch.ones(4, 4) - torch.eye(4)
        s = retrieval_scores(sim)
        assert s['top1_acc'] == 0.0
        assert s['mean_rank'] == 4.0

    def test_mrr_between_zero_and_one(self):
        sim = torch.randn(7, 7)
        s = retrieval_scores(sim)
        assert 0.0 < s['mrr'] <= 1.0


class TestSimilaritySpaces:
    @pytest.mark.parametrize('space', METRIC_SPACES)
    def test_identical_reconstruction_retrieves_itself(self, space):
        """A perfect reconstruction must score top-1 = 1 in every distance space."""
        b = _blob(seed=1)
        a = b.clone()                      # perfect reconstruction
        s = retrieval_scores(similarity_matrix(a, b, space=space))
        assert s['top1_acc'] == 1.0, f"{space}: perfect recon should retrieve itself"

    @pytest.mark.parametrize('space', METRIC_SPACES)
    def test_shuffled_reconstruction_fails(self, space):
        """If recon i is actually target i+1, retrieval must NOT credit it."""
        b = _blob(seed=2)
        a = torch.roll(b, shifts=1, dims=0)  # recon i == target i+1
        s = retrieval_scores(similarity_matrix(a, b, space=space))
        assert s['top1_acc'] < 0.5, f"{space}: shuffled recon should not retrieve the true target"

    def test_background_does_not_inflate_retrieval(self):
        """The whole point: a shared constant background must not create false retrieval.

        Two distinct digits on the same black background: a *mean blob* (identical for every image)
        should retrieve at chance, NOT be credited just because backgrounds match.
        """
        b = _blob(n=4, seed=3)
        mean_blob = b.mean(dim=0, keepdim=True).expand_as(b)  # same image for every row
        s = retrieval_scores(similarity_matrix(mean_blob, b, space='pixel'))
        # every row is identical, so argmax is the same column for all -> at most 1/N correct
        assert s['top1_acc'] <= 1.0 / b.shape[0] + 1e-6


class TestFeatureFn:
    def test_feature_fn_is_used(self):
        b = _blob(seed=4)
        identity_feat = lambda x: x            # trivial feature map
        s = retrieval_scores(similarity_matrix(b.clone(), b, feature_fn=identity_feat))
        assert s['top1_acc'] == 1.0
