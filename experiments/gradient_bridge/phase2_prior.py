"""GB-Phase 2 + the RIGHT prior (fix after TV failed).

Learning from phase2_tv.py: TV *hurt* monotonically (best λ=0). Two reasons: (a) the bridge inversion is
a DETERMINED rank-1 factorization (x is the row factor of G=g_err·xᵀ) — no null-space for a prior to
resolve, unlike Phase-0's model-based inversion; (b) TV is the WRONG prior for MNIST — digits are sharp
strokes (high TV is correct), so smoothing erases them. The decisive question: is the SSIM 0.15 a
PRIOR problem or a DECODER-COSINE problem? This compares priors MATCHED to digit statistics
(L1-sparsity, non-negativity) against TV and the raw-SVD baseline, sweeping the weight. If a matched
prior beats SVD, it is prior-fixable; if all flatline/hurt, the decoder cosine (0.90 = low-frequency
recovery) is the ceiling and the fix is a better decoder, not a hand-crafted prior.

Efficient: decoder plateaus by ~ep60; run the requested activation FIRST.
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


def _tv(x):
    x = x.reshape(-1, 1, 28, 28)
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum((1, 2, 3)) + \
           (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum((1, 2, 3))


def _ssim_diag(a, b):
    S = get_ssim_all(_norm01(a).cpu(), _norm01(b).cpu())
    return (S.diag().mean().item() if S.dim() == 2 else float(S.mean()))


def invert(G, u, xinit, prior, w, iters=600, box=0.1, lr=0.05):
    """max cos(u⊗x, G) - w·prior(x) - box·out-of-[0,1]. prior in {tv,l1,nonneg}."""
    n = G.shape[0]
    x = xinit.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    Gf = G.reshape(n, -1)
    for _ in range(iters):
        opt.zero_grad()
        pred = (u.unsqueeze(2) * x.unsqueeze(1)).reshape(n, -1)
        cos = torch.nn.functional.cosine_similarity(pred, Gf, dim=1)
        if prior == 'tv':
            pen = _tv(x)
        elif prior == 'l1':
            pen = x.abs().sum(1)
        else:                                          # 'nonneg' -> prior handled only by box
            pen = torch.zeros(n, device=x.device, dtype=x.dtype)
        boxp = (x - 1).clamp(min=0).pow(2).sum(1) + (-x).clamp(min=0).pow(2).sum(1)
        (-cos + w * pen + box * boxp).mean().backward()
        opt.step()
    return x.detach()


def run(activation, device, n_train, n_eval, rank, epochs):
    tb = generate_pair_bank(n_train, 0, rank, activation=activation, seed=0, device=device,
                            verbose=False, two_sided=True, a_init_scale=0.1)
    eb = generate_pair_bank(n_eval, 0, rank, activation=activation, seed=1, device=device,
                            verbose=False, two_sided=True, a_init_scale=0.1)
    dec, _, summ = train(tb, epochs=epochs, out_mode='lowrank', out_rank=16, batch=128,
                         device=device, verbose=False)
    dec.eval()
    A = eb['A'].float().to(device); B0 = eb['B0'].float().to(device)
    xtrue = eb['g_inp'].float().to(device)
    inp = _adapter_input(A, B0, two_sided=True, grad_A=eb['grad_A'].float().to(device),
                         grad_B=eb['grad_B'].float().to(device), A0=eb['A0'].float().to(device))
    with torch.no_grad():
        G = dec(inp.to(device)).reshape(-1, dec.out_features, dec.in_features).double()
        U, S, V = torch.svd_lowrank(G, q=6)
        u, x_svd = U[:, :, 0], V[:, :, 0]
    sgn = lambda xh: xh * torch.sign((xh * xtrue.double()).sum(1, keepdim=True)).clamp(min=-1)
    x_svd = sgn(x_svd)

    print(f"# {activation}  decoder_cos={summ['best_full_cos']:.4f}  (two-sided, layer 0)")
    print(f"{'prior':8s} {'weight':>8s} {'img_cos':>8s} {'img_SSIM':>9s}")
    base_ss = _ssim_diag(x_svd, xtrue)
    print(f"{'svd':8s} {0:8.4g} {torch.nn.functional.cosine_similarity(x_svd, xtrue.double(), dim=1).mean().item():8.4f} {base_ss:9.4f}")
    best = ('svd', 0.0, base_ss, x_svd)
    grid = {'l1': [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], 'tv': [3e-3, 1e-2, 3e-2], 'nonneg': [0.0]}
    for prior, ws in grid.items():
        for w in ws:
            xh = sgn(invert(G, u, x_svd, prior, w))
            cos = torch.nn.functional.cosine_similarity(xh, xtrue.double(), dim=1).mean().item()
            ss = _ssim_diag(xh, xtrue)
            print(f"{prior:8s} {w:8.4g} {cos:8.4f} {ss:9.4f}")
            if ss > best[2]:
                best = (prior, w, ss, xh)
    print(f"BEST: {best[0]} w={best[1]:g}  SSIM={best[2]:.4f}  (svd baseline {base_ss:.4f})")
    return best, x_svd, xtrue


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--activations', nargs='+', default=['gelu', 'softplus'])
    p.add_argument('--n_train', type=int, default=12000)
    p.add_argument('--n_eval', type=int, default=96)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--epochs', type=int, default=70)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_default_dtype(torch.float32)
    for act in args.activations:
        print(f"\n########## {act} ##########")
        try:
            (bp, bw, bss, bx), x_svd, xtrue = run(act, args.device, args.n_train, args.n_eval,
                                                  args.rank, args.epochs)
        except Exception as e:
            print(f"  SKIP {act}: {type(e).__name__}: {e}"); continue
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            os.makedirs(os.path.join(FIGURES_DIR, 'gradient_bridge'), exist_ok=True)
            xt, xs, xb = _norm01(xtrue).cpu(), _norm01(x_svd).cpu(), _norm01(bx).cpu()
            fig, axs = plt.subplots(3, 8, figsize=(12, 4.6))
            for j in range(8):
                for r, (img, lab) in enumerate([(xt, 'true'), (xs, 'SVD'), (xb, f'{bp} w={bw:g}')]):
                    axs[r, j].imshow(img[j, 0], cmap='gray'); axs[r, j].axis('off')
                    if j == 0:
                        axs[r, j].set_title(lab, loc='left', fontsize=9)
            fig.suptitle(f'GB-Phase 2 priors ({act}): true / raw-SVD / best-prior')
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR, 'gradient_bridge', f'phase2_prior_{act}.png'), dpi=130)
            plt.close(fig)
        except Exception as e:
            print(f"  (grid skipped: {e})")


if __name__ == '__main__':
    main()
