#!/usr/bin/env python3
"""Falsifier for (R5): the trajectory sees only lr/N, so a batch of N with N - N' invisible members gives the SAME
release as its N' recorded members alone at step lr*N'/N -- exactly up to the omitted imprints.  Consequence: N is
not identifiable from the release, only N' (rank B_T).  Thirty seconds.  Controls: a random N'-subset at the same
scaled step (should differ at O(1)); the recorded subset at the UNscaled step (the recipe error of job 634238).

  python -m experiments.exact_inversion.batch_scale_check
"""
import argparse, json, math, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import train_release, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch, release_and_imprints

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--sets", nargs="*", default=["repeated", "hard1_diff", "confident"])
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-fit", type=int, default=50000); ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    bb = TrainedBackbone(a.model, dev, "gelu"); a.sigma0 = a.sigma0 or 1.0 / math.sqrt(bb.n)
    chart = PCAChart(Xtr_t, a.k, dev)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    print(f"# batch_scale_check  git={git_hash()}", flush=True)
    for sname in a.sets:
        idx = torch.tensor(pick_batch(sname, bb, Xte_t, yte_t, a.N, perm), device=dev); X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        for setting, X in [("raw", X_real), ("on", chart.psi(chart.coords_of(X_real)))]:
            g = torch.Generator().manual_seed(a.seed + 7); A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
            A_T, B_T, imp, sB, C = release_and_imprints(bb, X, y, A0, a)
            rec = [i for i in range(a.N) if imp[i] / imp.max() > 1e-12]; Np = len(rec)
            if Np == a.N:
                print(json.dumps(dict(set=sname, setting=setting, n_prime=Np, note="all recorded; nothing to drop"))); continue
            H = bb.phi(X); omitted_rel = float(torch.linalg.norm(C[[i for i in range(a.N) if i not in rec]].sum(0)) / torch.linalg.norm(B_T))
            def rel(A2, B2): return float(torch.linalg.norm(B2 - B_T) / torch.linalg.norm(B_T)), float(torch.linalg.norm(A2 - A_T) / torch.linalg.norm(A_T))
            sub = torch.tensor(rec, device=dev)
            dB_scaled, dA_scaled = rel(*train_release(H[:, sub], A0, bb.W0, y[sub], bb.m, a.T, a.lr * Np / a.N, "sgd"))
            dB_unscaled, dA_unscaled = rel(*train_release(H[:, sub], A0, bb.W0, y[sub], bb.m, a.T, a.lr, "sgd"))
            # controls whose membership DIFFERS by construction (a random N'-subset shares most members at large N'):
            # one_swapped = recorded set with its weakest member replaced by the strongest invisible one;
            # complement_heavy = all invisible members, filled up to N' with the weakest recorded ones
            inv = [i for i in range(a.N) if i not in rec]
            by_imp = sorted(rec, key=lambda i: float(imp[i])); inv_by_imp = sorted(inv, key=lambda i: -float(imp[i]))
            swapped = sorted(by_imp[1:] + inv_by_imp[:1])
            comp = sorted((inv_by_imp + by_imp)[:Np])
            def ctrl(sub):
                t = torch.tensor(sub, device=dev)
                return rel(*train_release(H[:, t], A0, bb.W0, y[t], bb.m, a.T, a.lr * Np / a.N, "sgd"))[0]
            row = dict(set=sname, setting=setting, N=a.N, n_prime=Np, recorded=rec, omitted_imprint_rel=omitted_rel,
                       recorded_at_scaled_step_dB=dB_scaled, recorded_at_scaled_step_dA=dA_scaled,
                       recorded_at_UNscaled_step_dB=dB_unscaled,
                       one_swapped_subset=swapped, one_swapped_overlap=len(set(swapped) & set(rec)), one_swapped_at_scaled_step_dB=ctrl(swapped),
                       complement_heavy_subset=comp, complement_heavy_overlap=len(set(comp) & set(rec)), complement_heavy_at_scaled_step_dB=ctrl(comp),
                       git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
