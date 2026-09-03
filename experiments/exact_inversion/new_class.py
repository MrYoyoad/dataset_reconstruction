#!/usr/bin/env python3
"""Adding a NEW CLASS by LoRA: a CIFAR-10 model gets an 11th output, "flower", and is fine-tuned on eight flowers.

The extreme case of the mechanism (Step 18): every fine-tuning image is one the model has never seen the like
of, so every one has an O(1) residual and should be recorded in full -- and all eight share ONE label, the worst
conditioning case measured so far (Step 18, repeated labels).  The attacker who sees "an adapter that adds a
flower class" builds a FLOWER chart from public flowers: PCA on the CIFAR-100 train flower superclass (orchid,
poppy, rose, sunflower, tulip; 2,500 images), against a generic CIFAR-10 chart as the control.  Private flowers
come from the CIFAR-100 TEST split (unseen by the backbone, which is CIFAR-10-only, and by the chart).

  head     W0 extended by a zero row for class 10 (m = 11); LoRA B A on the extended head, B0 = 0.
  batches  flowers (8 x label 10) · cifar10 control (8 test images, their labels, m = 10) · mixed (4 + 4, m = 11).
  reads    margins / residuals at W0, imprints ||C_i||, rank B_T, spectrum; sigma_min(J) at the on-chart truth;
           cells (a) on-chart and (b) raw with each chart; image grids (32 x 32 x 3) saved.
  line     k < m + r - N = 19 (m = 11) / 18 (m = 10) at r = 16, N = 8.

  python -m experiments.exact_inversion.new_class --ks 16
"""
import argparse, json, math, os, pickle, socket, sys, time
import numpy as np, torch

from experiments.exact_inversion.lora_exact_inversion import train_release, qr_canon, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart
from experiments.exact_inversion.train_cifar_backbone import load_cifar10
from experiments.exact_inversion.vae_chart import invert_cell
from experiments.exact_inversion.margin_check import traced_release
from experiments.exact_inversion.subset_and_ood import margins_of

torch.set_default_dtype(torch.float64)
FLOWER_FINE = [54, 62, 70, 82, 92]                       # orchid, poppy, rose, sunflower, tulip (coarse class 2, "flowers")


def load_cifar100_flowers(root):
    out = {}
    for split in ("train", "test"):
        b = pickle.load(open(os.path.join(root, "cifar-100-python", split), "rb"), encoding="bytes")
        cl = np.array(b[b"coarse_labels"]); X = b[b"data"].astype(np.float64) / 255.0
        out[split] = (X[cl == 2], np.array(b[b"fine_labels"])[cl == 2])
    return out


class ExtendedHead:
    """The CIFAR-10 backbone with its head extended by a zero row for the new class (m = 11)."""
    def __init__(self, bb, n_new=1):
        self.W1, self.b1, self.W2, self.act = bb.W1, bb.b1, bb.W2, bb.act
        self.W0 = torch.cat([bb.W0, torch.zeros(n_new, bb.W0.shape[1], device=bb.W0.device)], 0)
        self.m, self.n = self.W0.shape
    phi = TrainedBackbone.phi
    logits = TrainedBackbone.logits


def grid(files, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(files) * 3, 8, figsize=(8, 3.2 * len(files)))
    for fi, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for r, key in enumerate(["x_real", "x_chart", "x_hat"]):
            X = d[key]
            for c in range(8):
                ax = axes[fi * 3 + r, c]; ax.axis("off")
                if c < X.shape[1]: ax.imshow(X[:, c].reshape(3, 32, 32).permute(1, 2, 0).clamp(0, 1).numpy())
            axes[fi * 3 + r, 0].set_title(f"{os.path.basename(f)[:-4]} / {key}", fontsize=6, loc="left")
    plt.tight_layout(); plt.savefig(path, dpi=110); print(f"# figure {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/cifar10_mlp.pth")
    ap.add_argument("--ks", nargs="*", type=int, default=[16]); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=2); ap.add_argument("--lm-iters", type=int, default=600)
    ap.add_argument("--batches", nargs="*", default=["flowers", "cifar10", "mixed"])
    ap.add_argument("--charts", nargs="*", default=["flower_pca", "cifar_pca"])
    ap.add_argument("--cells", nargs="*", default=["a", "b"])
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None); ap.add_argument("--fig", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, ytr, Xte, yte = load_cifar10(a.data_root); fl = load_cifar100_flowers(a.data_root)
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    Ftr_t = torch.tensor(fl["train"][0], device=dev); Fte_t = torch.tensor(fl["test"][0], device=dev); fte_fine = fl["test"][1]
    base = TrainedBackbone(a.model, dev, "gelu")
    with torch.no_grad():
        acc = float((base.logits(Xte_t.T).argmax(0) == yte_t).double().mean())
    ext = ExtendedHead(base)
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(base.n)
    g = torch.Generator().manual_seed(a.seed + 7)
    pf = torch.randperm(Fte_t.shape[0], generator=g)[:a.N]; pc = torch.randperm(Xte_t.shape[0], generator=g)[:a.N]
    flowers = (Fte_t[pf].T.contiguous(), torch.full((a.N,), 10, device=dev), [int(v) for v in fte_fine[pf.numpy()]])
    cifar = (Xte_t[pc].T.contiguous(), yte_t[pc], None)
    mixed = (torch.cat([flowers[0][:, :a.N // 2], cifar[0][:, :a.N // 2]], 1), torch.cat([flowers[1][:a.N // 2], cifar[1][:a.N // 2]]), None)
    batches = dict(flowers=(flowers, ext), cifar10=(cifar, base), mixed=(mixed, ext))
    print(f"# new_class  CIFAR-10 backbone acc {acc*100:.2f}%  flowers fine labels {flowers[2]}  ks={a.ks}  git={git_hash()}", flush=True)
    if a.save_dir: os.makedirs(a.save_dir, exist_ok=True)
    saved = []
    for k in a.ks:
        a.k = k
        charts = {}
        if "flower_pca" in a.charts: charts["flower_pca"] = PCAChart(Ftr_t, k, dev)          # public flowers (train split)
        if "cifar_pca" in a.charts: charts["cifar_pca"] = PCAChart(Xtr_t, k, dev)            # generic public images
        for bname in a.batches:
            (X_real, y, fine), bb = batches[bname]
            line = bb.m + a.r - a.N
            A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
            for cname, chart in charts.items():
                W = chart.coords_of(X_real); X_on = chart.psi(W)
                repr_err = torch.linalg.norm(X_on - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)
                for setting, X_train in [("raw", X_real), ("on", X_on)]:
                    mar, res0 = margins_of(bb, X_train, y)
                    H = bb.phi(X_train); A_T, B_T = train_release(H, A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
                    _, _, _, _, C = traced_release(H, A0, bb.W0, y, bb.m, a.T, a.lr)
                    assert float(torch.linalg.norm(C.sum(0) - B_T) / torch.linalg.norm(B_T)) < 1e-10
                    imp = torch.linalg.norm(C.reshape(a.N, -1), dim=1); sB = torch.linalg.svdvals(B_T)
                    with torch.no_grad(): pred = bb.logits(X_train).argmax(0)
                    row = dict(part="new_class", batch=bname, chart=cname, setting=setting, k=k, m=bb.m, n=bb.n, N=a.N, r=a.r, T=a.T, lr=a.lr,
                               seed=a.seed, capacity_line=line, below_line=bool(k < line), backbone_test_acc=acc, y=y.tolist(), fine_labels=fine,
                               pred_at_W0=pred.tolist(), acc_at_W0=float((pred == y).double().mean()),
                               margins=[float(v) for v in mar], margin_median=float(mar.median()), residual_W0=[float(v) for v in res0],
                               imprint_norms=[float(v) for v in imp], imprint_rel=[float(v / imp.max()) for v in imp],
                               rank_B_T=int((sB > 1e-12 * sB[0]).sum()), rank_B_T_1e8=int((sB > 1e-8 * sB[0]).sum()),
                               B_T_spectrum_rel=[float(v / sB[0]) for v in sB], B_T_sigma_ratio=float(sB[a.N - 1] / sB[0]),
                               chart_repr_err_median=float(repr_err.median()), chart_repr_err_max=float(repr_err.max()),
                               git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                    print(json.dumps(row), flush=True)
                    if a.out:
                        with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
                for cell in a.cells:
                    r, X_hat, X_on_c = invert_cell(chart, bb, X_real, y, a, cell, dev, g, lambda s: None)
                    r.update(part="new_class", batch=bname, chart=cname, k=k, m=bb.m, n=bb.n, N=a.N, r=a.r, T=a.T, lr=a.lr, seed=a.seed,
                             capacity_line=line, below_line=bool(k < line), backbone_test_acc=acc, y=y.tolist(), fine_labels=fine,
                             oracle=["near_init", "labels"], init_noise=a.init_noise, identifiability_test=True,
                             git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                    print(json.dumps(r), flush=True)
                    if a.out:
                        with open(a.out, "a") as f: f.write(json.dumps(r) + "\n")
                    if a.save_dir:
                        p = os.path.join(a.save_dir, f"{bname}_{cname}_k{k}_{cell}.pth")
                        torch.save(dict(x_real=X_real.cpu(), x_chart=X_on_c.cpu(), x_hat=X_hat.cpu(), meta=r), p); saved.append(p)
    if a.fig and saved: grid(saved, a.fig)


if __name__ == "__main__":
    main()
