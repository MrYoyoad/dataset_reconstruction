#!/usr/bin/env python3
"""A GRADED chart family at fixed k: a beta-VAE sweep on the trained backbone, cell (a) only.

Step 14's four charts (global PCA, class-local PCA, VAE-GELU, VAE-ReLU) came with better representation AND
worse conditioning, but they are two families and the sigma_min response is not uniform inside them (135x
within PCA for a 0.057 change in representation error; 1.1x within VAE for 0.013).  That supports "across the
charts tried" and NOT "conditioning degrades gradedly with richness", which is the mechanism claim.  Here ONE
family varies in richness with nothing else moving: same decoder architecture, same latent k, same training
data and seed; only the KL weight beta changes.  Larger beta = harder bottleneck = coarser chart.

  betas   {0.25, 1, 4, 16}.  Cell (a) on-chart, so a truth exists and sigma_min at the truth is defined.
  budget  --lm-iters 3000 --restarts 4 from the start, so iterations-to-floor is read WITHOUT the budget
          confound of jobs 611033/612643 (which stopped every cell at 200 iterations, off the floor).

Read: chart_repr_err (richness), jac_sigma_min_truth (conditioning, solver-independent), lm_iters_used and
residual (cost to the floor; residual is a SUM OF SQUARES, floor 1e-28), err_vs_REAL_max.  A large beta can
collapse latents; that shows up as a chart-side rank loss (sigma_min ~ 1e-19 at the truth) and must be read
as the chart degenerating, not as the release losing information.

  python -m experiments.exact_inversion.beta_vae_sweep --betas 0.25 1 4 16 --lm-iters 3000 --restarts 4
"""
import argparse, json, math, os, socket, sys
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, read_idx
from experiments.exact_inversion.vae_chart import VAE, VAEChart, invert_cell

torch.set_default_dtype(torch.float64)


def train_beta_vae(Xtr, k, beta, dev, epochs, bs, lr, seed, act="gelu", log=print):
    """vae_chart.train_vae with the KL weight exposed.  Duplicated rather than adding a `beta=1.0` kwarg there
       because vae_chart.py is under a running multi-cell job (611033); fold back into train_vae afterwards."""
    torch.manual_seed(seed)
    v = VAE(k, act=act).to(dev).float()
    opt = torch.optim.Adam(v.parameters(), lr=lr)
    X = Xtr.float()
    for ep in range(epochs):
        perm = torch.randperm(X.shape[0], device=dev)
        tot = 0.0
        for i in range(0, X.shape[0], bs):
            xb = X[perm[i:i + bs]]
            xh, mu, lv = v(xb)
            rec = ((xh - xb) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * xb.shape[0]
        log(f"    beta={beta:g} epoch {ep+1}/{epochs}  loss {tot/X.shape[0]:.4f}")
    return v.double().eval()


def decoder_jacobian_at_truth(chart, X_real):
    """Control A: the decoder's own geometry at the true coordinates.  d(phi.psi)/dw = dphi/dx . dpsi/dw, so a
       beta that moves dpsi/dw moves the release Jacobian directly; these three numbers separate that from the
       encoder's treatment of the chart's directions.  PCA charts have dpsi/dw orthonormal (all ones) and need
       no such control.  Per column (the map is block-diagonal across images); worst column reported."""
    W_true = chart.coords_of(X_real)
    smin, smax = [], []
    for i in range(W_true.shape[1]):
        J = tf.jacfwd(lambda w: chart.v.dec(w[None])[0])(W_true[:, i].detach())      # (784, k)
        sv = torch.linalg.svdvals(J); smin.append(float(sv[-1])); smax.append(float(sv[0]))
    return dict(dec_jac_sigma_min=min(smin), dec_jac_sigma_max=max(smax),
                dec_jac_cond_worst=max(smax) / min(smin), dec_jac_sigma_min_per_image=smin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=4); ap.add_argument("--lm-iters", type=int, default=3000)
    ap.add_argument("--betas", nargs="*", type=float, default=[0.25, 1.0, 4.0, 16.0])
    ap.add_argument("--cells", nargs="*", default=["a"])
    ap.add_argument("--vae-epochs", type=int, default=12); ap.add_argument("--vae-bs", type=int, default=256)
    ap.add_argument("--vae-lr", type=float, default=1e-3)
    ap.add_argument("--vae-act", choices=["gelu", "relu"], default="gelu")
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    bb = TrainedBackbone(a.model, dev, "gelu")
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(bb.n)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev)
    yte_t = torch.tensor(yte, device=dev)
    with torch.no_grad():
        acc = float((bb.logits(Xte_t[:2000].T).argmax(0) == yte_t[:2000]).double().mean())
    line = bb.m + a.r - a.N
    print(f"# beta-VAE chart family  backbone acc {acc*100:.2f}%  k={a.k}  line k < m+r-N = {line}  "
          f"({'below' if a.k < line else 'AT/ABOVE'})  betas={a.betas}  vae_act={a.vae_act}  git={git_hash()}", flush=True)
    g = torch.Generator().manual_seed(a.seed + 7)               # same private digits as vae_chart.py
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]

    for beta in a.betas:
        print(f"# training beta={beta:g} VAE chart on the TRAIN split only (public)", flush=True)
        vae = train_beta_vae(Xtr_t, a.k, beta, dev, a.vae_epochs, a.vae_bs, a.vae_lr, a.seed, a.vae_act)
        ch = VAEChart(vae, a.k)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            torch.save(vae.state_dict(),
                       os.path.join(a.save_dir, f"betavae_{a.vae_act}_b{beta:g}_k{a.k}_s{a.seed}.pth"))
        for cell in a.cells:
            print(f"##### beta={beta:g} cell={cell}", flush=True)
            row, X_hat, X_on = invert_cell(ch, bb, X_real, y, a, cell, dev, g, lambda s: None)
            row.update(**decoder_jacobian_at_truth(ch, X_real))
            row.update(chart=f"betavae_{a.vae_act}_b{beta:g}", beta=beta, chart_analytic=bool(a.vae_act == "gelu"),
                       k=a.k, N=a.N, r=a.r, m=bb.m, n=bb.n, T=a.T, lr=a.lr, seed=a.seed,
                       capacity_line=line, below_line=bool(a.k < line), backbone_test_acc=acc,
                       git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
            if a.save_dir:
                torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat=X_hat.cpu(), meta=row),
                           os.path.join(a.save_dir, f"betavae_b{beta:g}_{cell}_k{a.k}_s{a.seed}.pth"))


if __name__ == "__main__":
    main()
