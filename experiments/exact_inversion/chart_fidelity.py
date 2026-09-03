#!/usr/bin/env python3
"""The fidelity axis of the certificate ladder, computed standalone: does a PCA chart at dimension k preserve
class identity?  A property of the chart family and k alone -- no adapter, release, rank or search enters -- so
it is one forward-pass job over a LARGE held-out set (the private batch of 7-8 would move in steps of 1/7).
Per k: base-model accuracy on the k-projections of n_eval test digits (strong / mid / weak models), the median
representation error, and the accuracy on the raw digits for reference.  The ladder's admissible k at each r
(k < r - N') are then vertical marks on this curve.

  python -m experiments.exact_inversion.chart_fidelity --ks 2 4 6 8 10 12 16 20 24 32 40 48 56 64
"""
import argparse, json, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="*", type=int, default=[2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64])
    ap.add_argument("--models", nargs="*", default=["strong=models/exact_inversion/mnist_mlp_strong.pth", "mid=models/exact_inversion/mnist_mlp_mid.pth",
                                                    "weak=dataset_reconstruction/models/weights-mnist10_gelu.pth"])
    ap.add_argument("--n-eval", type=int, default=2000); ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xev = torch.tensor(Xte[:a.n_eval], device=dev).T.contiguous(); yev = torch.tensor(yte[:a.n_eval], device=dev)
    models = {spec.split("=")[0]: TrainedBackbone(spec.split("=", 1)[1], dev, "gelu") for spec in a.models}
    print(f"# chart_fidelity  n_eval={a.n_eval}  ks={a.ks}  git={git_hash()}", flush=True)
    with torch.no_grad():
        raw_acc = {n: float((m.logits(Xev).argmax(0) == yev).double().mean()) for n, m in models.items()}
        for k in a.ks:
            chart = PCAChart(Xtr_t, k, dev); Xon = chart.psi(chart.coords_of(Xev))
            err = torch.linalg.norm(Xon - Xev, dim=0) / torch.linalg.norm(Xev, dim=0)
            row = dict(part="chart_fidelity", k=k, n_eval=a.n_eval, chart="mnist_pca_train50k",
                       chart_repr_err_median=float(err.median()), chart_repr_err_p90=float(err.kthvalue(int(0.9 * a.n_eval)).values),
                       raw_acc=raw_acc, proj_acc={n: float((m.logits(Xon).argmax(0) == yev).double().mean()) for n, m in models.items()},
                       git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
