#!/usr/bin/env python3
"""What orders the certificate channel's basin sizes?  Post-hoc over Part-B rows (certificate.py), per cell.

For every on-chart Part-B row: rebuild the batch and release at that (set, r, k, N, seed) -- no solve -- and read
per recorded image four candidate predictors: the imprint ||C_i||, the margin at W0, ||A_T phi(x_on)||, ||x_on||.
Landings per image come from the row's per-start records.  Per cell: Kendall concordance of landings against each
predictor (count/pairs, tau, and the null sd sqrt(2(2n+5)/(9n(n-1)))), the pairwise taus among predictors, and the
per-image counts with Poisson errors.  Across cells the per-cell taus are pooled (weighted by pairs) -- never the
raw (image, landings) pairs, which would mix between-cell variation into a within-cell question.  Reading rule:
at n = 7 only |tau| = 1 discriminates; a coarse three-way banding of images survives where a full ranking does not.

  python -m experiments.exact_inversion.basin_predictors results/exact_inversion/step67_cert_below_*.jsonl results/exact_inversion/step69_cert_rank_*.jsonl
"""
import argparse, itertools, json, math, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch, release_and_imprints, margins_of, optdigits

torch.set_default_dtype(torch.float64)


def kendall(a, b):
    n = len(a); pairs = n * (n - 1) // 2
    c = sum((a[i] - a[j]) * (b[i] - b[j]) > 0 for i, j in itertools.combinations(range(n), 2))
    return c, pairs, (2 * c - pairs) / pairs, math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--n-fit", type=int, default=50000); ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--optdigits-path", default="data/ood_digits/optdigits.tes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    bb = TrainedBackbone(a.model, dev, "gelu")
    cells = []
    for f in a.files:
        for l in open(f):
            d = json.loads(l)
            if d.get("part") != "B" or d.get("setting") != "on" or "runs" not in d: continue
            cells.append((f, d))
    print(f"# basin_predictors over {len(cells)} on-chart Part-B cells  git={git_hash()}", flush=True)
    per_cell = []
    for f, d in cells:
        N, r, k, seed, sname = d["N"], d["r"], d["k"], d["seed"], d["set"]
        perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(seed + 7))
        if sname == "optdigits":
            labels = [i % 10 for i in range(N)]; X_real = optdigits(labels, a.optdigits_path, seed).to(dev); y = torch.tensor(labels, device=dev)
        elif sname == "mnist_control":
            idx, seen = [], set()
            for i in perm.tolist():
                if int(yte[i]) not in seen: idx.append(i); seen.add(int(yte[i]))
                if len(idx) == N: break
            idx = torch.tensor(idx, device=dev); X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        else:
            idx = torch.tensor(pick_batch(sname, bb, Xte_t, yte_t, N, perm), device=dev); X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        chart = PCAChart(Xtr_t, k, dev); X_on = chart.psi(chart.coords_of(X_real))
        class A: pass
        aa = A(); aa.k = k; aa.N = N; aa.r = r; aa.T = d["T"]; aa.lr = d["lr"]; aa.sigma0 = 1.0 / math.sqrt(bb.n)
        g = torch.Generator().manual_seed(seed + 7); A0 = (aa.sigma0 * torch.randn(r, bb.n, generator=g)).to(dev)
        A_T, B_T, imp, sB, C = release_and_imprints(bb, X_on, y, A0, aa)
        mar, _ = margins_of(bb, X_on, y)
        rec = d["recorded_idx"]
        land = {i: 0 for i in rec}
        for run in d["runs"]:
            if run.get("landed_on_recorded"): land[run["nearest_recorded"]] += 1
        L = [land[i] for i in rec]
        P = {"imprint": [float(imp[i]) for i in rec], "neg_margin": [-float(mar[i]) for i in rec],
             "A_T_phi_norm": [float(torch.linalg.norm(A_T @ bb.phi(X_on[:, i:i + 1]))) for i in rec],
             "x_on_norm": [float(torch.linalg.norm(X_on[:, i])) for i in rec]}
        with torch.no_grad():                                          # does the k-projection still read as its own digit?
            pred_on = bb.logits(X_on).argmax(0); pred_raw = bb.logits(X_real).argmax(0)
        proj_acc = float((pred_on == y).double().mean()); proj_acc_rec = float(sum(int(pred_on[i]) == int(y[i]) for i in rec) / len(rec))
        row = dict(file=f, set=sname, r=r, k=k, N=N, n_prime=len(rec), starts=len(d["runs"]), landings={str(i): land[i] for i in rec},
                   chart_proj_class_acc=proj_acc, chart_proj_class_acc_recorded=proj_acc_rec, raw_class_acc=float((pred_raw == y).double().mean()),
                   proj_pred_labels={str(i): int(pred_on[i]) for i in range(N)}, true_labels={str(i): int(y[i]) for i in range(N)},
                   chart_repr_err_median=float((torch.linalg.norm(X_on - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)).median()),
                   landings_min=min(L), landings_max=max(L), poisson_rel_err_at_min=(1 / math.sqrt(max(1, min(L)))),
                   predictors={n: {str(i): v for i, v in zip(rec, vals)} for n, vals in P.items()})
        for n, vals in P.items():
            c, pairs, tau, sd = kendall(L, vals); row[f"tau_{n}"] = tau; row[f"conc_{n}"] = f"{c}/{pairs}"; row["tau_null_sd"] = sd
        for (n1, v1), (n2, v2) in itertools.combinations(P.items(), 2):
            row[f"tau_{n1}_vs_{n2}"] = kendall(v1, v2)[2]
        # coarse banding: top third vs bottom third by landings, do the predictors separate them?
        order = sorted(rec, key=lambda i: land[i]); third = max(1, len(rec) // 3)
        lo, hi = order[:third], order[-third:]
        for n, vals in P.items():
            v = dict(zip(rec, vals)); row[f"band_{n}_hi_minus_lo_median"] = (sorted(v[i] for i in hi)[len(hi)//2] - sorted(v[i] for i in lo)[len(lo)//2])
        print(json.dumps(row), flush=True); per_cell.append(row)
        if a.out:
            with open(a.out, "a") as fo: fo.write(json.dumps(row) + "\n")
    if per_cell:
        print("# pooled per-cell taus (weighted by pairs):")
        for n in ["imprint", "neg_margin", "A_T_phi_norm", "x_on_norm"]:
            w = [c["n_prime"] * (c["n_prime"] - 1) / 2 for c in per_cell]; t = [c[f"tau_{n}"] for c in per_cell]
            print(f"   {n:>13}: tau = {sum(wi * ti for wi, ti in zip(w, t)) / sum(w):+.3f} over {len(per_cell)} cells, {int(sum(w))} pairs")


if __name__ == "__main__":
    main()
