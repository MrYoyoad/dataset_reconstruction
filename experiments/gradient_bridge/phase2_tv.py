"""GB-Phase 2 + the TV lever: convert the bridge's decoded gradient into an image with a prior.

Phase-0 (shown to Gal, slides 21-23) established that gradient-match cosine SATURATES (~0.95) while the
image quality is set by the TV PRIOR: at fixed cos, SSIM ran 0.10 -> 0.55 as TV weight rose (tv=1e-1
sweet spot on 224x224 ViT). Our bridge two-sided decode reaches img_cos 0.90 but the raw-SVD image is
SSIM 0.15 -- exactly the no-prior corner. This applies the same lever to the bridge and SWEEPS the TV
weight to find the MNIST-scale sweet spot (learning from Phase 0, not guessing).

Pipeline: two-sided decoder on the INPUT layer (softplus) -> decoded gradient G ~= u (x) x^T. Instead of
x_hat = top right singular vector (no prior), optimize x to maximize cos(u (x) x^T, G) - lambda*TV(x)
(+ soft box), vectorized over the eval batch, for each lambda. Report SSIM/cos vs lambda + a grid.
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))
from common_utils.image import get_ssim_all                                    # noqa: E402
from experiments.configs import FIGURES_DIR                                    # noqa: E402
from experiments.gradient_bridge.generate_pairs import generate_pair_bank       # noqa: E402
from experiments.gradient_bridge.train_decoder import train, _adapter_input     # noqa: E402


def _norm01(t):
    t = t.reshape(-1, 1, 28, 28)
    mn = t.amin((2, 3), keepdim=True); mx = t.amax((2, 3), keepdim=True)
    return ((t - mn) / (mx - mn + 1e-8)).float()


def _tv(x):                                    # x [n,1,28,28] -> [n]
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum((1, 2, 3))
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum((1, 2, 3))
    return dh + dw


def _ssim_diag(a, b):
    S = get_ssim_all(_norm01(a).cpu(), _norm01(b).cpu())
    return (S.diag().mean().item() if S.dim() == 2 else float(S.mean()))


def tv_invert(G, u, xinit, lam, iters=800, box=0.05, lr=0.05):
    """Optimize x [n,784] to max cos(u (x) x^T, G) - lam*TV(x) - box*out-of-[0,1]. Vectorized."""
    n, out_f, in_f = G.shape
    x = xinit.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    Gf = G.reshape(n, -1)
    for _ in range(iters):
        opt.zero_grad()
        pred = (u.unsqueeze(2) * x.unsqueeze(1)).reshape(n, -1)                 # outer(u,x) flattened
        cos = torch.nn.functional.cosine_similarity(pred, Gf, dim=1)           # [n]
        tv = _tv(x.reshape(n, 1, 28, 28))
        boxp = (x - 1).clamp(min=0).pow(2).sum(1) + (-x).clamp(min=0).pow(2).sum(1)
        loss = (-cos + lam * tv + box * boxp).mean()
        loss.backward(); opt.step()
    return x.detach()


def run(activation, device, n_train, n_eval, rank, epochs, lams):
    tb = generate_pair_bank(n_train, 0, rank, activation=activation, seed=0, device=device,
                            verbose=False, two_sided=True, a_init_scale=0.1)
    eb = generate_pair_bank(n_eval, 0, rank, activation=activation, seed=1, device=device,
                            verbose=False, two_sided=True, a_init_scale=0.1)
    dec, _, summ = train(tb, epochs=epochs, out_mode='lowrank', out_rank=16, batch=128, device=device)
    dec.eval()
    A = eb['A'].float().to(device); B0 = eb['B0'].float().to(device)
    xtrue = eb['g_inp'].float().to(device)
    inp = _adapter_input(A, B0, two_sided=True, grad_A=eb['grad_A'].float().to(device),
                         grad_B=eb['grad_B'].float().to(device), A0=eb['A0'].float().to(device))
    with torch.no_grad():
        G = dec(inp.to(device)).reshape(-1, dec.out_features, dec.in_features).double()
        U, S, V = torch.svd_lowrank(G, q=6)
        u = U[:, :, 0]                                                          # [n, out] left factor
        x_svd = V[:, :, 0]                                                      # [n, in] SVD baseline
    # sign-align SVD init to the true image direction
    x_svd = x_svd * torch.sign((x_svd * xtrue.double()).sum(1, keepdim=True)).clamp(min=-1)
    print(f"# {activation}  decoder_cos={summ['best_full_cos']:.4f}  (two-sided, layer 0)")
    print(f"{'lambda_TV':>10s} {'img_cos':>8s} {'img_SSIM':>9s}")
    best = (None, -1, None)
    for lam in lams:
        if lam == 0:
            xh = x_svd
        else:
            xh = tv_invert(G, u, x_svd, lam)
            xh = xh * torch.sign((xh * xtrue.double()).sum(1, keepdim=True)).clamp(min=-1)
        cos = torch.nn.functional.cosine_similarity(xh, xtrue.double(), dim=1).mean().item()
        ss = _ssim_diag(xh, xtrue)
        print(f"{lam:10.4g} {cos:8.4f} {ss:9.4f}")
        if ss > best[1]:
            best = (lam, ss, xh)
    return best, x_svd, xtrue


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--activation', default='softplus')
    p.add_argument('--n_train', type=int, default=12000)
    p.add_argument('--n_eval', type=int, default=128)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lams', type=float, nargs='+', default=[0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_default_dtype(torch.float32)

    (best_lam, best_ss, best_x), x_svd, xtrue = run(
        args.activation, args.device, args.n_train, args.n_eval, args.rank, args.epochs, args.lams)
    print(f"\nBEST: lambda_TV={best_lam}  SSIM={best_ss:.4f}  (SVD baseline was lambda=0)")

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        os.makedirs(os.path.join(FIGURES_DIR, 'gradient_bridge'), exist_ok=True)
        xt, xs, xb = _norm01(xtrue).cpu(), _norm01(x_svd).cpu(), _norm01(best_x).cpu()
        fig, axs = plt.subplots(3, 8, figsize=(12, 4.6))
        for j in range(8):
            for r, (img, lab) in enumerate([(xt, 'true'), (xs, 'SVD (no prior)'), (xb, f'TV λ={best_lam:g}')]):
                axs[r, j].imshow(img[j, 0], cmap='gray'); axs[r, j].axis('off')
                if j == 0:
                    axs[r, j].set_title(lab, loc='left', fontsize=9)
        fig.suptitle(f'GB-Phase 2 + TV lever ({args.activation}): true / raw-SVD / TV-prior')
        fig.tight_layout()
        out = os.path.join(FIGURES_DIR, 'gradient_bridge', f'phase2_tv_{args.activation}.png')
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"grid -> {out}")
    except Exception as e:
        print(f"  (grid skipped: {type(e).__name__}: {e})")


if __name__ == '__main__':
    main()
