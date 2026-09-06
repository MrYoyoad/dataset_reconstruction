#!/usr/bin/env python3
"""HEAD-TO-HEAD on the SAME release: the NTK/linearised reconstruction against the certificate.

Both attacks see exactly the same thing -- the same fine-tuned adapter on the same private images, the same public
base model, the same chart, the same random starts, the same budget, the same landing criterion. The only difference
is the equation each one solves. Swept over CHART TYPE, CHART DIMENSION k, and the number of fine-tuning STEPS T.

Setting: N private images of a class the base model does not have (CIFAR: keyboard / apple / both; MNIST: EMNIST
letter a / t / both), each added class getting its own new output row, LoRA of rank r on the HEAD, B_0 = 0, vanilla
full-batch SGD, float64. The private training inputs are their own chart projections, so both methods have a
reachable target (the on-chart setting).

THE TWO EQUATIONS
  NTK / linearised, FREE COEFFICIENTS (Experiment B's realistic mode -- no oracle arm anywhere in this script).
  The attacker assumes the fine-tune is one linearised step around the public base; for a head layer the per-sample
  gradient is (p_i - e_{y_i}) phi(x_i)^T, so
        Delta W  ~=  sum_i  r_i  phi(x_i)^T                              (r_i FREE in R^m, optimised)
  and the attack minimises ||Delta W - sum_i r_i phi(G(z_i))^T||_F / ||Delta W||_F jointly over the N latents z_i and
  the N coefficient vectors r_i. Delta W = B_T A_T is what a merged release exposes.

  Certificate.  C = P_{row(B_T)^perp} A_T, computed from the released factors alone, and the attack minimises, per
  image and independently, ||C phi(G(z))|| / ||A_T phi(G(z))||.

Reported per cell alongside both attacks: the LINEARISATION ERROR AT THE TRUTH,
      || Delta W - sum_i r_i^true phi(x_i^true)^T ||_F / || Delta W ||_F   with r_i^true the true base-model residuals,
which is the floor the NTK model itself leaves even when handed the right images -- the honest diagnostic for why
the route does or does not work at a given T.

WHY THE COMPARISON IS INFORMATIVE. The NTK loss couples all N images through one sum, so a start returns a whole
set and any recombination reproducing that sum is a minimiser (the superposition problem). The certificate is
separable: a start returns ONE image and the other N-1 never enter.

Both are optimised by Adam, batched over restarts, with identical starts and iteration counts, so nothing is a
solver artefact; the certificate's native Levenberg-Marquardt solver runs as a third arm for reference.

  python -u -m experiments.cifar.ntk_vs_certificate --dataset cifar --newclass cifar100:keyboard
  python -u -m experiments.cifar.ntk_vs_certificate --dataset cifar --newclass cifar100:keyboard --newclass2 cifar100:apple
  python -u -m experiments.cifar.ntk_vs_certificate --dataset mnist --letter a --letter2 t
"""
import argparse, json, math, os, socket, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def log(s): print(s, flush=True)


# ------------------------------------------------------------------------------------------------ backbones
def _pool_cifar(a, spec):
    from experiments.cifar.cifar_newclass import load_cifar100_class, load_flowers102
    if spec.startswith("cifar100:"): return load_cifar100_class(a.data_root, spec.split(":", 1)[1])
    return load_flowers102(a.data_root, seed=a.seed)


def build_cifar(a, dev):
    """The project's own MLP shape (3072-1000-1000-10, GELU) trained on CIFAR-10, head extended by one row per class."""
    from experiments.cifar.cifar_newclass import train_backbone
    net, te, tr = train_backbone(a.ckpt or "models/exact_inversion/cifar10_mlp_newclass.pth", a.data_root, dev, 60, 0.50, 0.90, "mlp")
    net = net.double()
    for p_ in net.parameters(): p_.requires_grad_(False)
    specs = [a.newclass] + ([a.newclass2] if a.newclass2 else [])
    pools = [_pool_cifar(a, sp) for sp in specs]
    Pub = [torch.tensor(p["train"], dtype=torch.float64, device=dev) for p, _ in pools]
    Pri = [torch.tensor(p["test"], dtype=torch.float64, device=dev) for p, _ in pools]
    W0 = torch.cat([net.head.weight.double(), torch.zeros(len(specs), net.head.weight.shape[1], dtype=torch.float64, device=dev)], 0)
    return (lambda X: net.phi(X.T).T), W0, Pub, Pri, "+".join(nm for _, nm in pools), dict(test_acc=te, train_acc=tr), (3, 32, 32)


def build_mnist(a, dev):
    """The repo's 98% MNIST MLP with EMNIST letters as new classes (the cell of RESULTS Step 25)."""
    from experiments.exact_inversion.trained_backbone import TrainedBackbone, read_idx
    from experiments.exact_inversion.new_class import load_emnist_letters, ExtendedHead
    base = TrainedBackbone(a.model_mnist, dev, "gelu")
    Xte, yte = read_idx(a.mnist_root, "test")
    with torch.no_grad():
        acc = float((base.logits(torch.tensor(Xte[:2000], device=dev).T).argmax(0) == torch.tensor(yte[:2000], device=dev)).double().mean())
    letters = [a.letter] + ([a.letter2] if a.letter2 else [])
    fls = [load_emnist_letters(a.mnist_root, L) for L in letters]
    Pub = [torch.tensor(f["train"][0], device=dev) for f in fls]; Pri = [torch.tensor(f["test"][0], device=dev) for f in fls]
    bb = ExtendedHead(base, "zero", a.seed)
    W0 = torch.cat([base.W0, torch.zeros(len(letters), base.W0.shape[1], device=dev)], 0)
    bb.W0 = W0; bb.m = W0.shape[0]
    return (lambda X: bb.phi(X)), W0, Pub, Pri, "+".join(f"letter_{L}" for L in letters), dict(test_acc=acc, train_acc=float("nan")), (1, 28, 28)


# ------------------------------------------------------------------------------------------------ charts
class PCAChartLocal:
    """Linear chart: the top-k principal components of the PUBLIC images of the added class(es)."""
    kind = "pca"
    def __init__(s, Pub, k, dev, shape, epochs=0):
        s.mean = Pub.mean(0); U, S, Vh = torch.linalg.svd(Pub - s.mean, full_matrices=False)
        s.V = Vh[:k].T.contiguous(); s.k = k; s.explained = float((S[:k] ** 2).sum() / (S ** 2).sum())
    def psi(s, Z): return s.mean[:, None] + s.V @ Z                                    # (k, P) -> (D, P)
    def psi_batch(s, Z): return s.mean[None, :, None] + torch.einsum("dk,pkn->pdn", s.V, Z)
    def coords(s, X): return s.V.T @ (X - s.mean[:, None])
    def std(s, Pub): return s.coords(Pub[:5000].T).std(dim=1, keepdim=True)
    def describe(s): return f"public PCA k={s.k} (explains {s.explained:.2f} of variance)"


class AEChartLocal(nn.Module):
    """Nonlinear chart: the decoder of an autoencoder trained on the PUBLIC images of the added class(es)."""
    kind = "ae"
    def __init__(s, Pub, k, dev, shape, epochs=150):
        super().__init__()
        D = int(np.prod(shape)); s.k = k; s.shape = shape
        h = 1024
        s.enc = nn.Sequential(nn.Linear(D, h), nn.GELU(), nn.Linear(h, 256), nn.GELU(), nn.Linear(256, k)).to(dev).double()
        s.dec = nn.Sequential(nn.Linear(k, 256), nn.GELU(), nn.Linear(256, h), nn.GELU(), nn.Linear(h, D), nn.Sigmoid()).to(dev).double()
        opt = torch.optim.Adam(list(s.enc.parameters()) + list(s.dec.parameters()), 1e-3)
        for ep in range(epochs):
            for i in torch.randperm(len(Pub), device=dev).split(256):
                xb = Pub[i]; opt.zero_grad(); F.mse_loss(s.dec(s.enc(xb)), xb).backward(); opt.step()
        with torch.no_grad():
            s.recon = float((torch.linalg.norm(s.dec(s.enc(Pub)) - Pub, dim=1) / torch.linalg.norm(Pub, dim=1)).median())
        for p_ in s.parameters(): p_.requires_grad_(False)
    def psi(s, Z): return s.dec(Z.T).T
    def psi_batch(s, Z):
        P, k, N = Z.shape
        return s.dec(Z.permute(0, 2, 1).reshape(-1, k)).reshape(P, N, -1).permute(0, 2, 1)
    def coords(s, X): return s.enc(X.T).T
    def std(s, Pub): return s.coords(Pub[:5000].T).std(dim=1, keepdim=True)
    def describe(s): return f"public autoencoder decoder k={s.k} (median public reconstruction error {s.recon:.3f})"


CHARTS = {"pca": PCAChartLocal, "ae": AEChartLocal}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar", "mnist"], default="cifar")
    ap.add_argument("--newclass", default="cifar100:keyboard"); ap.add_argument("--newclass2", default=None)
    ap.add_argument("--letter", default="a"); ap.add_argument("--letter2", default=None)
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--charts", nargs="*", default=["pca", "ae"]); ap.add_argument("--ks", nargs="*", type=int, default=[16, 32, 48])
    ap.add_argument("--Ts", nargs="*", type=int, default=[1, 400], help="fine-tuning steps: T=1 is where the linearisation is exact by construction, T=400 is the normal fine-tune")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--starts", type=int, default=200); ap.add_argument("--adam_iters", type=int, default=4000); ap.add_argument("--adam_lr", type=float, default=5e-2)
    ap.add_argument("--lm_iters", type=int, default=300); ap.add_argument("--ae_epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--model-mnist", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--data-root", default="data"); ap.add_argument("--mnist-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default="results/ntk_vs_cert"); ap.add_argument("--fig-dir", default="figures/ntk_vs_cert")
    a = ap.parse_args(); dev = torch.device(a.device)
    os.makedirs(a.save_dir, exist_ok=True); os.makedirs(a.fig_dir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    from experiments.cifar.cifar_newclass import lm_cert

    phi, W0, Pub_list, Pri_list, cname, accs, shape = (build_cifar if a.dataset == "cifar" else build_mnist)(a, dev)
    m, n = W0.shape; D = int(np.prod(shape))
    g = torch.Generator().manual_seed(a.seed + 7)
    ncls = len(Pri_list); per = a.N // ncls
    X_raw = torch.cat([Pri_list[c][torch.randperm(Pri_list[c].shape[0], generator=g)[: per]].T for c in range(ncls)], 1).contiguous()
    y = torch.cat([torch.full((per,), m - ncls + c, device=dev) for c in range(ncls)])
    Pub = torch.cat(Pub_list, 0)
    log(f"# ntk_vs_certificate  dataset={a.dataset}  class='{cname}' ({ncls} added class(es), labels {y.tolist()})  "
        f"backbone test {accs['test_acc']*100:.1f}%  m={m} n={n} N={a.N} r={a.r}  charts={a.charts} ks={a.ks} Ts={a.Ts}  host={socket.gethostname()}")
    Y = torch.eye(m, device=dev, dtype=torch.float64)[y].T
    A0 = (1.0 / math.sqrt(n) * torch.randn(a.r, n, generator=torch.Generator().manual_seed(a.seed + 7), dtype=torch.float64)).to(dev)

    cells = []; panels = {}
    for chart_name in a.charts:
        for k in a.ks:
            chart = CHARTS[chart_name](Pub, k, dev, shape, a.ae_epochs)
            coord_std = chart.std(Pub)
            X_on = chart.psi(chart.coords(X_raw))
            repr_err = torch.linalg.norm(X_on - X_raw, dim=0) / torch.linalg.norm(X_raw, dim=0)
            H = phi(X_on)
            Z0 = (torch.randn(a.starts, k, a.N, generator=torch.Generator().manual_seed(a.seed + 31)).to(dev) * coord_std[None])
            log(f"\n### chart={chart.describe()};  chart representation error of the privates: median {float(repr_err.median()):.3f}")
            for T in a.Ts:
                # ---- the release, at this T
                A, B = A0.clone(), torch.zeros(m, a.r, dtype=torch.float64, device=dev)
                for t in range(T):
                    z = W0 @ H + B @ (A @ H); Dm = (torch.softmax(z, 0) - Y) / a.N
                    B, A = B - a.lr * (Dm @ (A @ H).T), A - a.lr * (B.T @ Dm @ H.T)
                A_T, B_T = A, B; dW = B_T @ A_T; ndW = torch.linalg.norm(dW)
                sB = torch.linalg.svdvals(B_T); Np = int((sB > 1e-12 * sB[0]).sum())
                _, _, VhB = torch.linalg.svd(B_T, full_matrices=False); Q = VhB[:Np].T
                C = A_T - Q @ (Q.T @ A_T)
                with torch.no_grad():
                    cert_truth = torch.linalg.norm(C @ H, dim=0) / torch.linalg.norm(A_T @ H, dim=0)
                    R_true = torch.softmax(W0 @ H, 0) - Y
                    lin_err = float(torch.linalg.norm(dW - R_true @ H.T) / ndW)
                log(f"  -- T={T}: rank B_T={Np}, rank C={int(torch.linalg.matrix_rank(C, rtol=1e-10))}, ||dW||={ndW:.2e}; "
                    f"certificate residual at the truths {float(cert_truth.max()):.1e}; NTK linearisation error at the truth {lin_err:.3f}")

                def err_matrix(Xc):
                    return torch.stack([torch.linalg.norm(Xc - X_on[:, i:i + 1], dim=0) / torch.linalg.norm(X_on[:, i]) for i in range(a.N)], 1)

                # ---- NTK, free coefficients, batched Adam
                t0 = time.time(); Z = Z0.clone().requires_grad_(True); Rc = torch.zeros(a.starts, m, a.N, device=dev, requires_grad=True)
                opt = torch.optim.Adam([Z, Rc], a.adam_lr)
                for it in range(a.adam_iters):
                    Xc = chart.psi_batch(Z)
                    Hc = phi(Xc.permute(1, 0, 2).reshape(D, -1)).reshape(n, a.starts, a.N).permute(1, 0, 2)
                    loss_p = torch.linalg.norm((torch.einsum("pmn,pkn->pmk", Rc, Hc) - dW[None]).reshape(a.starts, -1), dim=1) / ndW
                    opt.zero_grad(); loss_p.sum().backward(); opt.step()
                with torch.no_grad():
                    Xc = chart.psi_batch(Z).detach()
                    Hc = phi(Xc.permute(1, 0, 2).reshape(D, -1)).reshape(n, a.starts, a.N).permute(1, 0, 2)
                    ntk_res = (torch.linalg.norm((torch.einsum("pmn,pkn->pmk", Rc, Hc) - dW[None]).reshape(a.starts, -1), dim=1) / ndW).detach()
                    cand = Xc.permute(1, 0, 2).reshape(D, -1)
                    E = err_matrix(cand); ntk_best = E.min(0).values; ntk_idx = E.argmin(0)
                    ntk_found = int((ntk_best < 1e-2).sum())
                ntk_sec = time.time() - t0
                log(f"     NTK  (free coefficients): residual {float(ntk_res.min()):.3e} (median {float(ntk_res.median()):.3e}); "
                    f"images recovered {ntk_found}/{a.N}; closest per image {[f'{v:.3f}' for v in ntk_best.tolist()]}  [{ntk_sec:.0f}s]")

                # ---- certificate, same starts, native LM
                t0 = time.time(); Xs, objs = [], []
                fun = lambda w: (C @ phi(chart.psi(w.reshape(k, 1)))).reshape(-1) / torch.linalg.norm(A_T @ phi(chart.psi(w.reshape(k, 1))))
                for p in range(a.starts):
                    w, obj, _ = lm_cert(fun, Z0[p, :, 0].clone(), a.lm_iters); Xs.append(chart.psi(w.reshape(k, 1))[:, 0]); objs.append(obj)
                Xs = torch.stack(Xs, 1); objs = torch.tensor(objs, device=dev)
                with torch.no_grad():
                    E = err_matrix(Xs); cert_best = E.min(0).values; cert_idx = E.argmin(0)
                    cert_found = int((cert_best < 1e-2).sum()); landed = int((E.min(1).values < 1e-2).sum())
                    order = torch.argsort(objs); top20 = sum(int(E[j].min() < 1e-2) for j in order[:20].tolist())
                cert_sec = time.time() - t0
                log(f"     CERT (same starts, LM)  : residual {float(objs.min())**0.5:.3e}; images recovered {cert_found}/{a.N}; "
                    f"landed starts {landed}/{a.starts}; top-20 by residual landed {top20}/20; closest per image {[f'{v:.3f}' for v in cert_best.tolist()]}  [{cert_sec:.0f}s]")

                cell = dict(part="ntk_vs_cert", dataset=a.dataset, class_name=cname, n_classes=ncls, chart=chart_name, chart_desc=chart.describe(),
                            k=k, T=T, lr=a.lr, N=a.N, r=a.r, m=m, n=n, seed=a.seed, starts=a.starts, n_prime=Np,
                            chart_repr_err_median=float(repr_err.median()), chart_repr_err=repr_err.tolist(),
                            cert_residual_at_truth=cert_truth.tolist(), ntk_linearisation_error_at_truth=lin_err,
                            ntk_residual_min=float(ntk_res.min()), ntk_residual_median=float(ntk_res.median()),
                            ntk_images_found=ntk_found, ntk_closest_per_image=ntk_best.tolist(), ntk_seconds=ntk_sec,
                            cert_residual_min=float(objs.min()) ** 0.5, cert_images_found=cert_found, cert_landed_starts=landed,
                            cert_top20_landed=top20, cert_closest_per_image=cert_best.tolist(), cert_seconds=cert_sec,
                            backbone=accs, host=socket.gethostname(), cmd=" ".join(sys.argv))
                cells.append(cell)
                print(json.dumps(cell), flush=True)
                if a.out:
                    with open(a.out, "a") as f: f.write(json.dumps(cell) + "\n")
                panels[(chart_name, k, T)] = dict(x_raw=X_raw.cpu(), x_chart=X_on.cpu(), ntk=cand[:, ntk_idx].cpu(), cert=Xs[:, cert_idx].cpu(),
                                                  ntk_err=ntk_best.tolist(), cert_err=cert_best.tolist())

    # ---------------------------------------------------------------- summary table + figures
    tag = f"{a.dataset}_{cname.replace('+','_and_')}_N{a.N}_r{a.r}"
    torch.save(dict(cells=cells, panels=panels, y=y.cpu()), os.path.join(a.save_dir, f"{tag}.pth"))
    tab = ["| chart | k | T | chart repr. err | NTK linearisation err at truth | NTK images | NTK residual | certificate images | certificate landed | top-20 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        tab.append(f"| {c['chart']} | {c['k']} | {c['T']} | {c['chart_repr_err_median']:.3f} | {c['ntk_linearisation_error_at_truth']:.3f} | "
                   f"**{c['ntk_images_found']}/{c['N']}** | {c['ntk_residual_min']:.2e} | **{c['cert_images_found']}/{c['N']}** | "
                   f"{c['cert_landed_starts']}/{c['starts']} | {c['cert_top20_landed']}/20 |")
    open(os.path.join(a.fig_dir, f"{tag}_table.md"), "w").write("\n".join(tab) + "\n")
    log("\n=== HEAD TO HEAD (free coefficients only; no oracle anywhere) ===\n" + "\n".join(tab))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # (a) summary: images recovered vs k, per chart type and T
    fig, axes = plt.subplots(1, len(a.Ts), figsize=(5.2 * len(a.Ts), 4.2), squeeze=False)
    for ti, T in enumerate(a.Ts):
        ax = axes[0][ti]
        for ci, ch in enumerate(a.charts):
            sel = [c for c in cells if c["chart"] == ch and c["T"] == T]
            if not sel: continue
            ks = [c["k"] for c in sel]
            ax.plot(ks, [c["cert_images_found"] for c in sel], "o-", color=["#1f77b4", "#2ca02c"][ci], label=f"certificate, {ch}")
            ax.plot(ks, [c["ntk_images_found"] for c in sel], "s--", color=["#d62728", "#ff7f0e"][ci], label=f"NTK free-c, {ch}")
        ax.set_xlabel("chart dimension k"); ax.set_ylabel(f"private images recovered (of {a.N})"); ax.set_ylim(-0.4, a.N + 0.4)
        ax.set_title(f"T = {T} fine-tuning steps (lr={a.lr})", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Same release, two attacks — '{cname}' as {'a new class' if ncls == 1 else f'{ncls} new classes'} on a "
                 f"{'CIFAR-10' if a.dataset == 'cifar' else 'MNIST'} MLP, head LoRA r={a.r}, N={a.N}, {a.starts} shared random starts", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig_dir, f"{tag}_summary.png"), dpi=200); plt.close(fig)

    # (b) image panels, one figure per (chart, k, T)
    for (ch, k, T), pn in panels.items():
        rows_fig = [("private (raw)", pn["x_raw"]), ("chart projection = the target", pn["x_chart"]),
                    ("NTK linearised, free coefficients", pn["ntk"]), ("certificate", pn["cert"])]
        fig, ax = plt.subplots(4, a.N, figsize=(1.35 * a.N + 2.6, 6.6)); fig.subplots_adjust(left=0.21, top=0.86, bottom=0.02, hspace=0.35)
        for ri, (name, imgs) in enumerate(rows_fig):
            for j in range(a.N):
                im = imgs[:, j].reshape(*shape).permute(1, 2, 0).clamp(0, 1).float().numpy()
                ax[ri, j].imshow(im.squeeze(), cmap=None if shape[0] == 3 else "gray", vmin=None if shape[0] == 3 else 0, vmax=None if shape[0] == 3 else 1)
                ax[ri, j].axis("off")
                if ri >= 2:
                    e = (pn["ntk_err"] if ri == 2 else pn["cert_err"])[j]
                    ax[ri, j].set_title("landed" if e < 1e-2 else f"err {e:.2f}", fontsize=6.5)
            p_ = ax[ri, 0].get_position(); fig.text(0.012, (p_.y0 + p_.y1) / 2, name, fontsize=7.5, va="center")
        c = next(c for c in cells if c["chart"] == ch and c["k"] == k and c["T"] == T)
        fig.suptitle(f"'{cname}' as {'a new class' if ncls == 1 else f'{ncls} new classes'} — head LoRA r={a.r}, T={T} SGD steps (lr={a.lr}), {c['chart_desc']}\n"
                     f"NTK linearised (free coefficients) recovered {c['ntk_images_found']}/{a.N}; certificate recovered {c['cert_images_found']}/{a.N} "
                     f"({c['cert_landed_starts']}/{a.starts} starts). NTK linearisation error at the truth: {c['ntk_linearisation_error_at_truth']:.2f}", fontsize=9)
        fig.savefig(os.path.join(a.fig_dir, f"{tag}_{ch}_k{k}_T{T}.png"), dpi=150); plt.close(fig)
    log(f"saved {a.save_dir}/{tag}.pth and {a.fig_dir}/{tag}_*.png")


if __name__ == "__main__":
    main()
