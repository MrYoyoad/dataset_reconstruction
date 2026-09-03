#!/usr/bin/env python3
"""Blur control for Step 19's "the MNIST chart draws the foreign sets BETTER than MNIST" (0.317 / 0.383 vs 0.518).

optdigits are 8x8 material and fonts are clean glyphs: both are intrinsically low-bandwidth, and a 16-component
linear chart represents smooth images better whatever it was fitted on.  Decisive control: give the SAME MNIST
control digits optdigits' effective resolution (8x8 area-downsample -> 20x20 bilinear -> 28 canvas, the optdigits
pipeline) or a Gaussian blur, and re-measure chart_repr_err on the SAME PCA basis.  No release, no solve.
If blurred MNIST lands near 0.32, the chart half of Step 19's two-sided prediction was not tested by provenance.

  python -m experiments.exact_inversion.blur_control
"""
import argparse, json, socket, sys
import torch, torch.nn.functional as F

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import font_digits, optdigits, mnist_like

torch.set_default_dtype(torch.float64)


def repr_err(chart, X):
    W = chart.coords_of(X); Xo = chart.psi(W)
    e = torch.linalg.norm(Xo - X, dim=0) / torch.linalg.norm(X, dim=0)
    return float(e.median()), float(e.max())


def gaussian_blur(X, sigma):
    if sigma <= 0: return X
    r = int(3 * sigma) + 1; t = torch.arange(-r, r + 1, dtype=torch.float64); g = torch.exp(-t ** 2 / (2 * sigma ** 2)); g /= g.sum()
    k2 = (g[:, None] * g[None, :])[None, None]
    return F.conv2d(X.T.reshape(-1, 1, 28, 28), k2.to(X.device), padding=r).reshape(-1, 784).T


def down_up(X, lo):
    """area-downsample the 20-px content box to lo x lo, bilinear back to 20 x 20, centred on a 28 canvas (the optdigits path)."""
    out = []
    for i in range(X.shape[1]):
        im = X[:, i].reshape(1, 1, 28, 28)
        box = im[:, :, 4:24, 4:24]
        small = F.interpolate(box, size=(lo, lo), mode="area")
        up = F.interpolate(small, size=(20, 20), mode="bilinear", align_corners=False)
        canvas = torch.zeros(28, 28, device=X.device); canvas[4:24, 4:24] = up[0, 0]
        out.append(mnist_like(canvas.cpu().numpy()).to(X.device))
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=8); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-fit", type=int, default=50000); ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--optdigits", default="data/ood_digits/optdigits.tes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev)
    chart = PCAChart(Xtr_t, a.k, dev)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    idx, seen = [], set()
    for i in perm.tolist():
        if int(yte[i]) not in seen: idx.append(i); seen.add(int(yte[i]))
        if len(idx) == a.N: break
    X_mn = Xte_t[torch.tensor(idx, device=dev)].T.contiguous()
    labels = [0, 3, 5, 1, 9, 6, 7, 4]
    sets = {"mnist_control": X_mn, "font": font_digits(labels, a.seed)[0].to(dev), "optdigits": optdigits(labels, a.optdigits, a.seed).to(dev)}
    for sg in (0.5, 1.0, 1.5, 2.0): sets[f"mnist_blur_sigma{sg}"] = gaussian_blur(X_mn, sg)
    for lo in (14, 10, 8, 6): sets[f"mnist_down{lo}_up20"] = down_up(X_mn, lo)
    # a whole-test-set reference so the 8-digit draw is not the only in-distribution number
    Xall = Xte_t[:2000].T.contiguous(); sets["mnist_test2000"] = Xall; sets["mnist_test2000_down8_up20"] = down_up(Xall, 8)
    print(f"# blur control  k={a.k}  git={git_hash()}", flush=True)
    for name, X in sets.items():
        med, mx = repr_err(chart, X)
        row = dict(part="blur_control", set=name, n=int(X.shape[1]), k=a.k, chart="mnist_pca_train", chart_repr_err_median=med, chart_repr_err_max=mx,
                   git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
