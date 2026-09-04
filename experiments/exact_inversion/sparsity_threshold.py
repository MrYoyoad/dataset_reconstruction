#!/usr/bin/env python3
"""Where does the compressed-sensing boundary actually sit? Measure the sparsity instead of assuming it.

Two lanes gave different thresholds for the same cell: `r ~ 256` (yoado-cd/81) and `~137` (yoado-b9). The whole
difference is the assumed sparsity `s` in `m >~ s.log(n/s)` -- s = 150 gives 248, s = 50 gives 137. Which outcome
of the structured-recovery cell is INFORMATIVE depends on which is right, so this measures `s` on the actual
private batch rather than picking a side.

`s` is not a single number, so three definitions are reported and the threshold is bracketed by all of them:
  * the count of pixels above a small absolute value (the naive reading, and the one that inflates `s`);
  * the best `s`-term approximation: the smallest `s` whose top-`s` coefficients carry 95% and 99% of the energy,
    which is the quantity the compressed-sensing bound is actually about;
  * the same in the basis a natural-image cell would use (DCT), since pixel sparsity is MNIST-specific.
PRE-REGISTRATION SUPPORT: the cell supplies `r - N'` conditions. Printing the predicted threshold beside that
count fixes, before the run, whether the cell sits ABOVE the boundary (where success restates arithmetic and only
FAILURE is a finding) or NEAR it (where either outcome informs).

  python -u -m experiments.exact_inversion.sparsity_threshold
"""
import argparse, json, math, socket, sys
import numpy as np
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import read_idx

torch.set_default_dtype(torch.float64)


def s_term(x, frac):
    """Smallest s whose top-s coefficients carry `frac` of the energy -- the quantity the CS bound is about."""
    e = np.sort(x ** 2)[::-1]
    c = np.cumsum(e) / max(e.sum(), 1e-300)
    return int(np.searchsorted(c, frac) + 1)


def dct2(x):
    from scipy.fft import dctn
    return dctn(x.reshape(28, 28), norm="ortho").ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--conditions", nargs="*", type=int, default=[8, 56, 248],
                    help="conditions supplied at r = 16, 64, 256 with N' = 8")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    X, y = read_idx(a.data_root, "test")
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(len(X), a.N, replace=False)
    B = X[idx]
    n = 784

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    defs = {}
    defs["nonzero_gt_0.05"] = [int((b > 0.05).sum()) for b in B]
    defs["s95_pixel"] = [s_term(b, 0.95) for b in B]
    defs["s99_pixel"] = [s_term(b, 0.99) for b in B]
    try:
        defs["s95_dct"] = [s_term(dct2(b), 0.95) for b in B]
        defs["s99_dct"] = [s_term(dct2(b), 0.99) for b in B]
    except Exception as e:
        print(f"# DCT unavailable ({e}); pixel bases only", flush=True)

    print(f"# sparsity of {a.N} MNIST test digits, n = {n}   git={git_hash()} host={socket.gethostname()}",
          flush=True)
    print(f"# {'definition':18s} {'median s':>9s} {'threshold s.log(n/s)':>21s}", flush=True)
    rows = {}
    for k, v in defs.items():
        s = int(np.median(v))
        thr = int(round(s * math.log(n / max(s, 1))))
        rows[k] = dict(s_median=s, s_all=v, threshold=thr)
        print(f"  {k:18s} {s:9d} {thr:21d}", flush=True)

    print(f"\n# conditions supplied, against each threshold:", flush=True)
    for c in a.conditions:
        pos = {k: ("ABOVE" if c >= 1.2 * r0["threshold"] else "NEAR" if c >= 0.8 * r0["threshold"] else "BELOW")
               for k, r0 in rows.items()}
        print(f"  {c:4d} conditions:  " + "  ".join(f"{k}={pos[k]}" for k in rows), flush=True)
        emit(dict(part="CS_THRESHOLD", conditions=c, n=n, N=a.N, by_definition=rows, position=pos,
                  informative_outcome={k: ("FAILURE only -- success would restate arithmetic" if p == "ABOVE"
                                           else "either outcome informs" if p == "NEAR"
                                           else "SUCCESS only -- failure would restate arithmetic")
                                       for k, p in pos.items()},
                  note="which outcome of the structured-recovery cell is a FINDING depends on where the cell sits "
                       "relative to the boundary; that position is fixed here, before the run",
                  git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
