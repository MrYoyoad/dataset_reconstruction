#!/usr/bin/env python3
"""The truncation tolerance is the attacker's free knob: rebuild C from the same B_T at a sweep of SVD tolerances
and report, per dtype, the number of images the certificate still annihilates against the tolerance, beside B_T's
own singular spectrum.  The wall -- where the curve stops moving -- is the release's information content; a count
at any single tolerance is partly a statement about the tolerance.  Also decides what the twentieth image of the
wide-head cell is: returns at a tighter tolerance (our truncation), or sigma_20/sigma_1 far below its imprint
(collinearity, not magnitude), or never (the release's precision).

  python -m experiments.exact_inversion.tolerance_sweep
"""
import argparse, json, math, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch, release_and_imprints, optdigits
from experiments.exact_inversion.precision_check import quantise, DTYPES

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["models/exact_inversion/mnist_mlp_m26_strong.pth:optdigits:20:64:8:on",
                                                   "models/exact_inversion/mnist_mlp_strong.pth:optdigits:20:64:8:on",
                                                   "models/exact_inversion/mnist_mlp_strong.pth:repeated:8:16:16:on",
                                                   "models/exact_inversion/mnist_mlp_strong.pth:confident:8:16:16:on"],
                    help="model:set:N:r:k:setting")
    ap.add_argument("--tols", nargs="*", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16])
    ap.add_argument("--dtypes", nargs="*", default=["fp64", "fp32", "tf32", "fp16", "bf16"])
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-fit", type=int, default=50000); ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--optdigits-path", default="data/ood_digits/optdigits.tes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    print(f"# tolerance_sweep  git={git_hash()}", flush=True)
    for cell in a.cells:
        model, sname, N, r, k, setting = cell.split(":"); N, r, k = int(N), int(r), int(k)
        bb = TrainedBackbone(model, dev, "gelu"); chart = PCAChart(Xtr_t, k, dev)
        if sname == "optdigits":
            labels = [i % 10 for i in range(N)]; X_real = optdigits(labels, a.optdigits_path, a.seed).to(dev); y = torch.tensor(labels, device=dev)
        else:
            idx = torch.tensor(pick_batch(sname, bb, Xte_t, yte_t, N, perm), device=dev); X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        X = chart.psi(chart.coords_of(X_real)) if setting == "on" else X_real
        class A: pass
        aa = A(); aa.k = k; aa.N = N; aa.r = r; aa.T = a.T; aa.lr = a.lr; aa.sigma0 = 1.0 / math.sqrt(bb.n)
        g = torch.Generator().manual_seed(a.seed + 7); A0 = (aa.sigma0 * torch.randn(r, bb.n, generator=g)).to(dev)
        A_T, B_T, imp, sB, C = release_and_imprints(bb, X, y, A0, aa)
        H = bb.phi(X); AH = A_T @ H; imp_rel = imp / imp.max()
        present = [i for i in range(N) if float(imp_rel[i]) > 1e-12]; invisible = [i for i in range(N) if i not in present]
        for dname in a.dtypes:
            Aq, Bq = quantise(A_T, dname), quantise(B_T, dname)
            U, S, Vh = torch.linalg.svd(Bq, full_matrices=False); srel = (S / S[0])
            for tol in a.tols:
                Np = int((srel > tol).sum()); Q = Vh[:Np].T; Cq = Aq - Q @ (Q.T @ Aq)
                cr = (torch.linalg.norm(Cq @ H, dim=0) / torch.linalg.norm(AH, dim=0))
                inv_min = float(min(cr[i] for i in invisible)) if invisible else float("nan")
                sep = sum(1 for i in present if (float(cr[i]) < (inv_min / 100 if invisible else 1e-3)))
                row = dict(part="tolerance_sweep", model=model.split("/")[-1], set=sname, N=N, r=r, k=k, m=bb.m, setting=setting, dtype=dname,
                           quantisation_noise_rel=float(torch.linalg.norm(Bq - B_T) / torch.linalg.norm(B_T)), tol=tol, rank_B_T=Np, rank_C=r - Np,
                           n_present=len(present), n_annihilated_1e3=sum(1 for i in present if float(cr[i]) < 1e-3), n_separable=sep,
                           B_T_spectrum_rel=[float(v) for v in srel[:N + 2]], imprint_rel=[float(v) for v in imp_rel],
                           cert_residual_per_image=[float(v) for v in cr], git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                print(json.dumps(row), flush=True)
                if a.out:
                    with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
