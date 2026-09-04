#!/usr/bin/env python3
"""Publish the DOSE before the response: best s-term error per (dataset, basis), from PUBLIC data only.

yoado-7e/b9: a compressibility dose-response is only meaningful if the axis is committed BEFORE the outcomes, and
it is only scoreable if the cells span at least an order of magnitude in best-s-term error. Both are properties of
the data and the basis alone -- no release, no attack, no private set -- so they can and must be fixed now.

The primary axis is BASIS AT FIXED DATA (pixel / DCT / a public PCA dictionary on the same images), because a
dataset axis is confounded: changing the dataset changes the base model's classification of it, hence the margins,
hence the imprints, hence whether the certificate records anything at all. The dataset arm is kept and LABELLED
CONFOUNDED; the basis axis carries the claim.

Everything here is computed on the PUBLIC split. The private batch is never touched, so nothing in this table can
be tuned on an outcome.

  python -u -m experiments.exact_inversion.dose_table
"""
import argparse, json, math, socket, sys
import numpy as np

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import read_idx


def load_public(name, root, n):
    """28x28 greyscale in [0,1], PUBLIC split only."""
    if name == "mnist":
        return read_idx(root, "train")[0][:n]
    if name == "fashion":
        with open(f"{root}/FashionMNIST/raw/train-images-idx3-ubyte", "rb") as f:
            f.read(16); a = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
        return a.astype(np.float64)[:n] / 255.0
    if name == "emnist":
        with open(f"{root}/EMNIST/raw/emnist-letters-train-images-idx3-ubyte", "rb") as f:
            f.read(16); a = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
        return a.astype(np.float64)[:n] / 255.0
    if name == "cifar_grey":
        import pickle, os
        d = os.path.join(root, "cifar-10-batches-py", "data_batch_1")
        with open(d, "rb") as f: b = pickle.load(f, encoding="bytes")
        x = b[b"data"][:n].reshape(-1, 3, 32, 32).astype(np.float64) / 255.0
        g = x.mean(1)                                                   # greyscale, then centre-crop to 28x28
        return g[:, 2:30, 2:30].reshape(len(g), 784)
    raise SystemExit(f"unknown dataset {name}")


def transform(X, basis, fit=None):
    if basis == "pixel":
        return X, None
    if basis == "dct":
        from scipy.fft import dctn
        return np.stack([dctn(x.reshape(28, 28), norm="ortho").ravel() for x in X]), None
    if basis == "pca":
        if fit is None:
            Xc = X - X.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            fit = (X.mean(0), Vt)
        mu, Vt = fit
        return (X - mu) @ Vt.T, fit
    raise SystemExit(basis)


def s_term_error(coef, s):
    """Relative L2 error of the best s-term approximation -- the quantity the CS bound is about."""
    e = np.sort(coef ** 2, axis=1)[:, ::-1]
    tot = e.sum(1) + 1e-300
    keep = e[:, :s].sum(1)
    return np.sqrt(np.maximum(tot - keep, 0) / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["mnist", "emnist", "fashion", "cifar_grey"])
    ap.add_argument("--bases", nargs="*", default=["pixel", "dct", "pca"])
    ap.add_argument("--n-public", type=int, default=5000)
    ap.add_argument("--s", type=int, default=50, help="the fixed s at which the dose is quoted")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    print(f"# DOSE table, PUBLIC data only, committed before any attack.   s = {a.s}   "
          f"git={git_hash()} host={socket.gethostname()}", flush=True)
    print(f"# {'dataset':11s} {'basis':6s} {'err@s':>9s} {'s95':>6s} {'s99':>6s} {'thr(s95)':>9s}", flush=True)
    cells = []
    for ds in a.datasets:
        X = load_public(ds, a.data_root, a.n_public)
        for b in a.bases:
            coef, _ = transform(X, b)
            err = float(np.median(s_term_error(coef, a.s)))
            e = np.sort(coef ** 2, axis=1)[:, ::-1]
            c = np.cumsum(e, axis=1) / (e.sum(1, keepdims=True) + 1e-300)
            s95 = int(np.median([np.searchsorted(row, 0.95) + 1 for row in c]))
            s99 = int(np.median([np.searchsorted(row, 0.99) + 1 for row in c]))
            thr = int(round(s95 * math.log(784 / max(s95, 1))))
            cells.append(dict(dataset=ds, basis=b, err_at_s=err, s95=s95, s99=s99, cs_threshold_from_s95=thr))
            print(f"  {ds:11s} {b:6s} {err:9.4f} {s95:6d} {s99:6d} {thr:9d}", flush=True)
    errs = [c["err_at_s"] for c in cells if c["err_at_s"] > 0]
    span = (max(errs) / min(errs)) if errs else 0.0
    scoreable = bool(len(cells) >= 4 and span >= 10)
    print(f"\n# span in best-{a.s}-term error across cells: x{span:.1f} over {len(cells)} cells  -> "
          f"dose-response {'SCOREABLE' if scoreable else 'NOT SCOREABLE (needs >=4 cells spanning >=1 order)'}",
          flush=True)
    emit(dict(part="DOSE", s=a.s, cells=cells, span_ratio=span, n_cells=len(cells),
              dose_response_scoreable=scoreable,
              primary_axis="basis at fixed data (pixel/dct/pca); the DATASET axis is kept but CONFOUNDED, since "
                           "changing the data changes the base model's margins, hence the imprints, hence whether "
                           "the certificate records anything at all",
              committed_before_attack=True, public_split_only=True,
              git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))


if __name__ == "__main__":
    main()
