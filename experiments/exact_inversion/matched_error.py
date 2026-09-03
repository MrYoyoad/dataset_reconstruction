#!/usr/bin/env python3
"""Matched-permutation image error from the saved cell tensors (post-hoc, no solve).

Two private columns with the same label carry the same one-hot, so exchanging them leaves the release exactly
invariant -- a genuine discrete symmetry of the forward map.  The column-wise error the cells report would score
a permuted-but-correct reconstruction as a failure.  This reads every saved {x_real, x_chart, x_hat, meta} and
reports, per cell: column-wise max error, the one-to-one matched error (assign_err, any permutation), and the
label-respecting matched error (permutations within a label only -- the symmetry that actually exists) when the
row carries the labels.  Against x_chart for cell (a), against x_real otherwise.

  python -m experiments.exact_inversion.matched_error results/exact_inversion/step48_*/*.pth
"""
import json, sys
import torch
from experiments.exact_inversion.lora_exact_inversion import assign_err

torch.set_default_dtype(torch.float64)


def errors(x_hat, x_true, labels=None):
    D = torch.cdist(x_hat.T, x_true.T) / torch.linalg.norm(x_true, dim=0)[None, :]        # D[i, j] = |hat_i - true_j| / |true_j|
    col = float(torch.diagonal(D).max()); matched = float(assign_err(D).max())
    within = None
    if labels is not None:
        Dl = D.clone(); lab = torch.tensor(labels)
        Dl[lab[:, None] != lab[None, :]] = float("inf")
        within = float(assign_err(Dl).max())
    return col, matched, within


def main():
    for p in sys.argv[1:]:
        d = torch.load(p, map_location="cpu", weights_only=False); meta = d["meta"]
        ref = d["x_chart"] if meta.get("cell") == "a" else d["x_real"]
        col, matched, within = errors(d["x_hat"], ref, meta.get("y"))
        print(json.dumps(dict(file=p, chart=meta.get("chart"), cell=meta.get("cell"), encoder=meta.get("encoder"),
                              labels=meta.get("y"), err_colwise_max=col, err_matched_any_max=matched,
                              err_matched_within_label_max=within, residual=meta.get("residual"))))


if __name__ == "__main__":
    main()
