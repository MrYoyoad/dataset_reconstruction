#!/usr/bin/env python3
"""The certificate C = P_{row(B_T)^perp} A_T, its condition C h = 0, and what it buys once imprints are understood.

Derivation.  With B0 = 0 and SGD every update to A lies in row(B_T), so C = P^perp A0 and C h_i = P^perp A0 h_i.
In the imprint form B_T ~ sum_i q_i (A0 h_i)^T the row space is spanned by A0 h_i of the RECORDED images only, so
  C h_i = 0 for recorded images, C h_i != 0 for invisible ones,  rank C = r - N' (N' = number recorded).
Hence (A) ||C phi(x)|| is a RECIPE-FREE test of whether x was recorded (no eta, T or labels in it), and
(B) for the dominant image (N' = 1) the r-1 linear equations C phi(psi(w)) = 0 determine w when k < r - 1,
    with NO unrolled dynamics -- a recipe-free inversion whose landscape is far simpler than the full one.
The certificate line is k < r - N'.  Adam releases have rank B_T = r, C = 0, and none of this applies.

Part A: per image ||C h_i|| / ||A_T h_i|| beside the imprint on the strong-model batches (prediction: ~1e-15 for
        recorded, O(1) for invisible).
Part B: certificate-only inversion of the dominant image from R random public-scale starts, no labels, no recipe,
        at k in {12, 14, 15, 16} (isolated solutions expected for k <= 14, a curve at 16).  Evaluated vs the truth.

  python -m experiments.exact_inversion.certificate --part A B
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch, release_and_imprints, margins_of

torch.set_default_dtype(torch.float64)


def certificate(A_T, B_T, tol=1e-12):
    """C = (I - Q Q^T) A_T with Q an orthonormal basis of row(B_T) (right singular vectors above tol)."""
    U, S, Vh = torch.linalg.svd(B_T, full_matrices=False)
    Np = int((S > tol * S[0]).sum()); Q = Vh[:Np].T                       # r x N'
    C = A_T - Q @ (Q.T @ A_T)
    return C, Np, S


def lm_cert(fun, w0, iters=300, lam=1e-2):
    """Levenberg-Marquardt on the cheap map w -> C phi(psi(w)) (no unroll); returns w, final objective, iters used."""
    w = w0.clone(); f = fun(w); obj = float(f @ f)
    for it in range(iters):
        J = tf.jacfwd(fun)(w)
        for _ in range(12):
            step = torch.linalg.solve(J.T @ J + lam * torch.eye(J.shape[1], device=w.device), -(J.T @ f))
            w_new = w + step; f_new = fun(w_new); obj_new = float(f_new @ f_new)
            if obj_new < obj: w, f, obj = w_new, f_new, obj_new; lam = max(lam / 3, 1e-15); break
            lam *= 4
        else:
            return w, obj, it + 1
        if obj < 1e-30: return w, obj, it + 1
    return w, obj, iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="*", default=["A", "B"])
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--sets", nargs="*", default=["mnist_control", "hard1_diff", "confident"])
    ap.add_argument("--ks", nargs="*", type=int, default=[12, 14, 15, 16]); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--random-starts", type=int, default=16); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    bb = TrainedBackbone(a.model, dev, "gelu")
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(bb.n)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    if a.save_dir: os.makedirs(a.save_dir, exist_ok=True)
    print(f"# certificate  model={a.model}  sets={a.sets}  ks={a.ks}  git={git_hash()}", flush=True)

    def batch(sname):
        if sname == "mnist_control":
            idx, seen = [], set()
            for i in perm.tolist():
                if int(yte[i]) not in seen: idx.append(i); seen.add(int(yte[i]))
                if len(idx) == a.N: break
            return idx
        return pick_batch(sname, bb, Xte_t, yte_t, a.N, perm)

    for sname in a.sets:
        idx = torch.tensor(batch(sname), device=dev); X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        for k in a.ks:
            a.k = k; chart = PCAChart(Xtr_t, k, dev)
            coord_std = chart.coords_of(Xtr_t[:10000].T).std(dim=1, keepdim=True)
            W_all = chart.coords_of(X_real); X_on = chart.psi(W_all)
            for setting, X_train in [("raw", X_real), ("on", X_on)]:
                g = torch.Generator().manual_seed(a.seed + 7)
                A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
                A_T, B_T, imp, sB, Cimp = release_and_imprints(bb, X_train, y, A0, a)
                C, Np, S = certificate(A_T, B_T)
                H = bb.phi(X_train); AH = A_T @ H
                cert_res = (torch.linalg.norm(C @ H, dim=0) / torch.linalg.norm(AH, dim=0))
                top = int(torch.argmax(imp))
                rowA = dict(part="A", set=sname, setting=setting, k=k, N=a.N, r=a.r, m=bb.m, T=a.T, lr=a.lr, seed=a.seed, y=y.tolist(),
                            imprint_rel=[float(v / imp.max()) for v in imp], rank_B_T=Np, rank_C=int(torch.linalg.matrix_rank(C, rtol=1e-10)),
                            cert_line=a.r - Np, cert_residual_per_image=[float(v) for v in cert_res],
                            cert_residual_recorded_max=float(max(cert_res[i] for i in range(a.N) if imp[i] / imp.max() > 1e-12)),
                            cert_residual_invisible_min=float(min([cert_res[i] for i in range(a.N) if imp[i] / imp.max() <= 1e-12] or [float("nan")])),
                            git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                if "A" in a.part:
                    print(json.dumps(rowA), flush=True)
                    if a.out:
                        with open(a.out, "a") as f: f.write(json.dumps(rowA) + "\n")
                if "B" in a.part and Np == 1:                                  # certificate-only inversion of the dominant image
                    nA = torch.linalg.norm(A_T)
                    fun = lambda w: (C @ bb.phi(chart.psi(w.reshape(k, 1)))).reshape(-1) / nA
                    gs = torch.Generator().manual_seed(a.seed + 31); t0 = time.time(); runs = []
                    for s in range(a.random_starts):
                        w0 = (torch.randn(k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
                        w, obj, it = lm_cert(fun, w0, a.iters)
                        x_hat = chart.psi(w.reshape(k, 1))
                        e_on = float(torch.linalg.norm(x_hat[:, 0] - X_on[:, top]) / torch.linalg.norm(X_on[:, top]))
                        e_raw = float(torch.linalg.norm(x_hat[:, 0] - X_real[:, top]) / torch.linalg.norm(X_real[:, top]))
                        e_all = [float(torch.linalg.norm(x_hat[:, 0] - X_on[:, j]) / torch.linalg.norm(X_on[:, j])) for j in range(a.N)]
                        runs.append(dict(start=s, objective=obj, iters=it, err_vs_top_chart=e_on, err_vs_top_raw=e_raw,
                                         nearest=int(min(range(a.N), key=lambda j: e_all[j])), w=w.detach().cpu()))
                    # the truth's own objective (gate) and the attacker's pick = smallest objective
                    w_true = W_all[:, top].reshape(-1); obj_true = float(fun(w_true) @ fun(w_true))
                    best = min(runs, key=lambda d: d["objective"])
                    at_floor = [d for d in runs if d["objective"] <= 1e3 * max(obj_true, 1e-32)]
                    rowB = dict(part="B", set=sname, setting=setting, k=k, cert_line=a.r - Np, below_cert_line=bool(k < a.r - Np),
                                N=a.N, r=a.r, m=bb.m, T=a.T, lr=a.lr, seed=a.seed, top_image_eval=top, top_label_eval=int(y[top]),
                                oracle=[], recipe_used=False, labels_used=False, random_starts=a.random_starts, iters=a.iters,
                                objective_at_truth=obj_true, argmin_objective=best["objective"], argmin_err_vs_top_chart=best["err_vs_top_chart"],
                                argmin_err_vs_top_raw=best["err_vs_top_raw"], argmin_nearest_is_top=bool(best["nearest"] == top),
                                frac_starts_at_floor=len(at_floor) / len(runs),
                                frac_starts_recovered=sum(d["err_vs_top_chart"] < 1e-2 for d in runs) / len(runs),
                                runs=[{kk: v for kk, v in d.items() if kk != "w"} for d in runs], seconds=time.time() - t0,
                                git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                    print(json.dumps(rowB), flush=True)
                    if a.out:
                        with open(a.out, "a") as f: f.write(json.dumps(rowB) + "\n")
                    if a.save_dir:
                        torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat=chart.psi(best["w"].to(dev).reshape(k, 1)).cpu(), top=top, meta=rowB),
                                   os.path.join(a.save_dir, f"cert_{sname}_{setting}_k{k}.pth"))


if __name__ == "__main__":
    main()
