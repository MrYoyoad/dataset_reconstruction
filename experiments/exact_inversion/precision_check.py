#!/usr/bin/env python3
"""Is the released adapter's numerical precision a privacy parameter?  (yoado-ed)

"Recorded" is not binary: every example with a nonzero imprint is in row(B_T) mathematically, but only those above
the numerical resolution are in it usably.  An imprint 1e-20 below the largest is absent from an FP64 release and
far more absent from a bfloat16 one.  So: take releases already measured, round (A_T, B_T) to FP32 / TF32 / FP16 /
bfloat16 and back, and re-read rank B_T, rank C and the per-image certificate residuals -- with the SVD tolerance
following the dtype's epsilon (a fixed 1e-12 tolerance would read rounding noise as EXTRA rank).  Per image the
imprint is compared with the quantisation noise it must clear.  Prediction: the number of certificate-recoverable
examples falls monotonically with precision; examples move from recoverable to merely-present as the mantissa
shortens.  No training, no solving.

  python -m experiments.exact_inversion.precision_check
"""
import argparse, json, math, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch, release_and_imprints, optdigits
from experiments.exact_inversion.certificate import certificate

torch.set_default_dtype(torch.float64)
DTYPES = {"fp64": (torch.float64, 2.2e-16), "fp32": (torch.float32, 1.2e-7), "tf32": (None, 9.8e-4), "fp16": (torch.float16, 9.8e-4), "bf16": (torch.bfloat16, 7.8e-3)}


def quantise(M, name):
    if name == "fp64": return M.clone()
    if name == "tf32":                                               # 10-bit mantissa at fp32 range
        m32 = M.float(); mant, exp = torch.frexp(m32)
        return (torch.round(mant * 2 ** 11) / 2 ** 11 * 2.0 ** exp).double()
    return M.to(DTYPES[name][0]).double()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--cells", nargs="*", default=["repeated:8:16:on", "repeated:8:16:raw", "confident:8:16:on", "hard1_diff:8:16:on", "optdigits:20:64:on"],
                    help="set:N:r:setting")
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data"); ap.add_argument("--optdigits-path", default="data/ood_digits/optdigits.tes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    bb = TrainedBackbone(a.model, dev, "gelu"); chart = PCAChart(Xtr_t, a.k, dev)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    print(f"# precision_check  git={git_hash()}", flush=True)
    for cell in a.cells:
        sname, N, r, setting = cell.split(":"); N, r = int(N), int(r)
        if sname == "optdigits":
            labels = [i % 10 for i in range(N)]; X_real = optdigits(labels, a.optdigits_path, a.seed).to(dev); y = torch.tensor(labels, device=dev)
        else:
            idx = torch.tensor(pick_batch(sname, bb, Xte_t, yte_t, N, perm), device=dev); X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        X = chart.psi(chart.coords_of(X_real)) if setting == "on" else X_real
        class A: pass
        aa = A(); aa.k = a.k; aa.N = N; aa.r = r; aa.T = a.T; aa.lr = a.lr; aa.sigma0 = 1.0 / math.sqrt(bb.n)
        g = torch.Generator().manual_seed(a.seed + 7); A0 = (aa.sigma0 * torch.randn(r, bb.n, generator=g)).to(dev)
        A_T, B_T, imp, sB, C = release_and_imprints(bb, X, y, A0, aa)
        H = bb.phi(X); AH = A_T @ H
        for dname in ["fp64", "fp32", "tf32", "fp16", "bf16"]:
            Aq, Bq = quantise(A_T, dname), quantise(B_T, dname); eps = DTYPES[dname][1]
            noise_rel = float(torch.linalg.norm(Bq - B_T) / torch.linalg.norm(B_T))
            Cq, Np, S = certificate(Aq, Bq, tol=10 * eps)                          # dtype-aware tolerance
            cert_res = (torch.linalg.norm(Cq @ H, dim=0) / torch.linalg.norm(AH, dim=0))
            rankC = int(torch.linalg.matrix_rank(Cq, rtol=10 * eps))
            imp_rel = imp / imp.max()
            recoverable = [i for i in range(N) if float(cert_res[i]) < 1e-3]        # the certificate still holds for it
            present = [i for i in range(N) if float(imp_rel[i]) > 1e-12]
            row = dict(part="precision", set=sname, setting=setting, N=N, r=r, k=a.k, dtype=dname, eps=eps, quantisation_noise_rel=noise_rel,
                       rank_B_T=Np, rank_C=rankC, B_T_spectrum_rel=[float(v / S[0]) for v in S[:N + 2]],
                       imprint_rel=[float(v) for v in imp_rel], imprint_over_noise=[float(v / max(noise_rel, 1e-300)) for v in imp_rel],
                       cert_residual_per_image=[float(v) for v in cert_res], n_present=len(present), n_recoverable=len(recoverable),
                       present_not_recoverable=[i for i in present if i not in recoverable],
                       git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
