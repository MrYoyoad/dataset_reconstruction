#!/usr/bin/env python3
"""The release Jacobian AT THE TRUTH over several A0 draws -- no solve.

An accidental replicate in Step 14 (the same global-PCA chart through two scripts, different A0) put the
single-draw noise at ~6% in sigma_min and ~1.6x in cond, while every chart number on file was one draw.  The
at-truth quantities need only a Jacobian, so several seeds cost several Jacobians, not several inversions.
Per (encoder, label draw, chart, k, A0 seed) this records sigma_min, sigma_max, cond, rank J, the residual at
the truth, and the release's own witness: rank B_T, sigma_N/sigma_1 and sigma_{N+1}/sigma_1 of B_T, rank X.

Charts: global / local PCA are built here; vae / betavae / cvae decoders are loaded from the artefacts the
cell scripts save (--vae label:act:path, --cvae label:path).  Encoders: --encoder label=path, or label=random
(RandomBackbone: the weak checkpoint's shapes and layer norms with Gaussian weights).

  python -m experiments.exact_inversion.truth_spectrum --encoder weak=dataset_reconstruction/models/weights-mnist10_gelu.pth \
      --encoder random=random --labels repeated distinct --charts global local --a0-seeds 1 2 3
"""
import argparse, json, math, socket, sys
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import train_release, simulate_sgd_reduced, qr_canon, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.conditional_charts import LocalPCAChart, CondVAE, CondVAEChart
from experiments.exact_inversion.vae_chart import VAE, VAEChart
from experiments.exact_inversion.random_encoder_control import RandomBackbone

torch.set_default_dtype(torch.float64)


def pick_digits(perm, yte, N, mode):
    """Same as random_encoder_control.py (duplicated: that module is under a running job)."""
    if mode == "repeated": return perm[:N].tolist()
    idx, seen = [], set()
    for i in perm.tolist():
        if int(yte[i]) not in seen: idx.append(i); seen.add(int(yte[i]))
        if len(idx) == N: break
    return idx


def coords(chart, X, inner_iters):
    """chart.coords_of with the inner projection budget exposed where the chart has one (learned charts)."""
    return chart.coords_of(X, iters=inner_iters) if "iters" in chart.coords_of.__code__.co_varnames else chart.coords_of(X)


def inner_diag(chart, X_real, W):
    """The inner projection's own convergence at W (a learned chart's coords_of is a fixed-step Adam solve and
       W IS the truth of the cell, so slack here moves the truth and sigma_min with it), plus the chart
       Jacobian dpsi/dw per column (block-diagonal across images; orthonormal for PCA, so those read 1)."""
    w = W.clone().requires_grad_(True)
    loss = ((chart.psi(w) - X_real) ** 2).sum()
    g, = torch.autograd.grad(loss, w)
    smin, smax = [], []
    for i in range(W.shape[1]):
        def col(wi, i=i):
            Wf = torch.cat([W[:, :i], wi[:, None], W[:, i + 1:]], dim=1)
            return chart.psi(Wf)[:, i]
        sv = torch.linalg.svdvals(tf.jacfwd(col)(W[:, i].detach()))
        smin.append(float(sv[-1])); smax.append(float(sv[0]))
    return dict(inner_loss=float(loss), inner_grad_norm=float(g.norm()),
                dec_jac_sigma_min=min(smin), dec_jac_sigma_max=max(smax), dec_jac_cond_worst=max(smax) / min(smin))


def spectrum_at_truth(chart, bb, X_real, y, a, A0, W_true=None):
    """The cell-(a) construction of vae_chart.invert_cell, up to the Jacobian at the truth."""
    m, n = bb.m, bb.n; k = a.k; N = a.N; nW = k * N
    if W_true is None: W_true = chart.coords_of(X_real)
    X_on = chart.psi(W_true); H = bb.phi(X_on)
    A_T, B_T = train_release(H, A0, bb.W0, y, m, a.T, a.lr, "sgd")
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T)

    def res_vec(v):
        Wc = v[:nW].reshape(k, N); aux = v[nW:].reshape(a.r, N)
        Bs, Xis, Uc = simulate_sgd_reduced(bb.phi(chart.psi(Wc)), aux, bb.W0, y, m, a.T, a.lr, 0.0)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])

    U_true, _ = qr_canon(H); v0 = torch.cat([W_true.reshape(-1), (A0 @ U_true).reshape(-1)]).detach()
    sv = torch.linalg.svdvals(tf.jacfwd(res_vec)(v0).detach())
    sB = torch.linalg.svdvals(B_T); sX = torch.linalg.svdvals(A0 @ U_true)
    repr_err = float((torch.linalg.norm(X_on - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)).median())
    return dict(jac_sigma_min_truth=float(sv[-1]), jac_sigma_max_truth=float(sv[0]), jac_cond_truth=float(sv[0] / sv[-1]),
                jac_rank_truth=int((sv > 1e-12 * sv[0]).sum()), jac_cols=int(sv.numel()),
                res_at_truth=float(torch.linalg.norm(res_vec(v0))), chart_repr_err=repr_err,
                rank_B_T=int((sB > 1e-12 * sB[0]).sum()), B_T_sigma_ratio=float(sB[N - 1] / sB[0]),
                B_T_sigma_Np1_over_1=float(sB[N] / sB[0]) if sB.numel() > N else 0.0,
                B_T_spectrum_rel=[float(v / sB[0]) for v in sB[:N + 2]],
                rank_X=int((sX > 1e-12 * sX[0]).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", default=[], help="label=path | label=random (repeatable)")
    ap.add_argument("--norm-model", default="dataset_reconstruction/models/weights-mnist10_gelu.pth",
                    help="checkpoint whose shapes/layer norms the random encoder copies")
    ap.add_argument("--encoder-seed", type=int, default=101)
    ap.add_argument("--labels", nargs="*", default=["repeated", "distinct"])
    ap.add_argument("--charts", nargs="*", default=["global", "local"])
    ap.add_argument("--vae", action="append", default=[], help="label:act:path of a saved VAE decoder (repeatable)")
    ap.add_argument("--cvae", action="append", default=[], help="label:path of a saved conditional VAE (repeatable)")
    ap.add_argument("--ks", nargs="*", type=int, default=[16]); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--a0-seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--inner-iters", nargs="*", type=int, default=[300],
                    help="inner projection budgets for learned charts; all A0 seeds run at the LAST level, "
                         "earlier levels use the first seed only (they exist to show the truth moving)")
    ap.add_argument("--perturb", type=int, default=0,
                    help="extra at-truth spectra at this many nearby on-chart points (W + scale*std*noise), "
                         "first seed, last inner level: a piecewise-linear (ReLU) chart Jacobian is only stable "
                         "if these agree")
    ap.add_argument("--perturb-scale", type=float, default=1e-3)
    ap.add_argument("--perturb-charts", nargs="*", default=None, help="restrict --perturb to these chart labels")
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); ytr_t = torch.tensor(ytr[:a.n_fit], device=dev)
    Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    norm_bb = TrainedBackbone(a.norm_model, dev, "gelu")
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))   # the cells' draw
    print(f"# truth spectrum  encoders={a.encoder}  labels={a.labels}  charts={a.charts}  vae={a.vae}  cvae={a.cvae}"
          f"  ks={a.ks}  a0_seeds={a.a0_seeds}  git={git_hash()}", flush=True)

    for spec in a.encoder:
        label, path = spec.split("=", 1)
        bb = RandomBackbone(norm_bb, dev, a.encoder_seed) if path == "random" else TrainedBackbone(path, dev, "gelu")
        sigma0 = a.sigma0 if a.sigma0 is not None else 1.0 / math.sqrt(bb.n)
        with torch.no_grad():
            acc = float((bb.logits(Xte_t[:2000].T).argmax(0) == yte_t[:2000]).double().mean())
        for lmode in a.labels:
            idx = torch.tensor(pick_digits(perm, yte, a.N, lmode), device=dev)
            X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
            for k in a.ks:
                a.k = k
                charts = {}
                if "global" in a.charts: charts["global"] = PCAChart(Xtr_t, k, dev)
                if "local" in a.charts: charts["local"] = LocalPCAChart(Xtr_t, ytr_t, k, y, dev)
                for v in a.vae:
                    vl, act, vp = v.split(":", 2)
                    net = VAE(k, act=act); net.load_state_dict(torch.load(vp, map_location="cpu")); charts[vl] = VAEChart(net.to(dev).double().eval(), k)
                for c in a.cvae:
                    cl, cp = c.split(":", 1)
                    net = CondVAE(k); net.load_state_dict(torch.load(cp, map_location="cpu")); charts[cl] = CondVAEChart(net.to(dev).double().eval(), k, y, dev)
                def emit(row, **extra):
                    row.update(extra); row.update(encoder=label, encoder_path=path, backbone_test_acc=acc, labels=lmode,
                                                  y=y.tolist(), k=k, N=a.N, r=a.r, m=bb.m, n=bb.n, T=a.T, lr=a.lr, seed=a.seed,
                                                  capacity_line=bb.m + a.r - a.N, git=git_hash(), host=socket.gethostname(),
                                                  cmd=" ".join(sys.argv))
                    print(json.dumps(row), flush=True)
                    if a.out:
                        with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

                def A0_of(s):
                    return (sigma0 * torch.randn(a.r, bb.n, generator=torch.Generator().manual_seed(1000 + s))).to(dev)

                for cname, chart in charts.items():
                    learned = "iters" in chart.coords_of.__code__.co_varnames
                    levels = a.inner_iters if learned else [None]
                    for li, L in enumerate(levels):
                        W_true = coords(chart, X_real, L) if learned else chart.coords_of(X_real)
                        diag = inner_diag(chart, X_real, W_true)
                        seeds = a.a0_seeds if li == len(levels) - 1 else a.a0_seeds[:1]
                        for s in seeds:
                            row = spectrum_at_truth(chart, bb, X_real, y, a, A0_of(s), W_true)
                            emit(row, chart=cname, a0_seed=s, inner_iters=L, perturb=0, **diag)
                        if a.perturb and li == len(levels) - 1 and (a.perturb_charts is None or cname in a.perturb_charts):
                            gp = torch.Generator().manual_seed(a.seed + 99)
                            for j in range(1, a.perturb + 1):
                                Wp = W_true + a.perturb_scale * W_true.std() * torch.randn(W_true.shape, generator=gp).to(dev)
                                row = spectrum_at_truth(chart, bb, X_real, y, a, A0_of(a.a0_seeds[0]), Wp)
                                emit(row, chart=cname, a0_seed=a.a0_seeds[0], inner_iters=L, perturb=j,
                                     perturb_scale=a.perturb_scale, **inner_diag(chart, X_real, Wp))


if __name__ == "__main__":
    main()
