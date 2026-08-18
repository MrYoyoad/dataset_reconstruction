"""GB-Phase 2: end-to-end bridge attack -- decoded gradient -> actual IMAGE.

For the INPUT layer (layer 0), the single-sample gradient is rank-1: dL/dW0 = g_err (x) x^T, where
x is the input image (784-dim) and is stored as the bank's g_inp factor. So once the R2F decoder
recovers dL/dW0 from the LoRA adapter (A,B0), the image is the top RIGHT singular vector of the decoded
gradient (up to sign/scale). This tests whether the bridge's high cosine (softplus 0.997 hidden-layer;
here on layer 0) actually yields a recoverable image -- the "necessary but not sufficient" milestone.

Per activation: generate a train bank + a held-out eval bank (layer 0), train the decoder, decode the
eval adapters, x_hat = top right singular vector, and score SSIM/cosine vs the true image. Saves a grid.
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))
from common_utils.image import get_ssim_all                                   # noqa: E402
from experiments.configs import FIGURES_DIR, RESULTS_DIR                       # noqa: E402
from experiments.gradient_bridge.generate_pairs import generate_pair_bank      # noqa: E402
from experiments.gradient_bridge.train_decoder import train, _adapter_input    # noqa: E402


def _norm01(t):
    t = t.reshape(-1, 1, 28, 28)
    mn = t.amin((2, 3), keepdim=True); mx = t.amax((2, 3), keepdim=True)
    return ((t - mn) / (mx - mn + 1e-8)).float()


def phase2(activation, device, n_train, n_eval, rank, epochs, two_sided=False):
    tb = generate_pair_bank(n_train, 0, rank, activation=activation, seed=0, device=device,
                            verbose=False, two_sided=two_sided, a_init_scale=0.1)
    eb = generate_pair_bank(n_eval, 0, rank, activation=activation, seed=1, device=device,
                            verbose=False, two_sided=two_sided, a_init_scale=0.1)
    dec, _, summ = train(tb, epochs=epochs, out_mode='lowrank', out_rank=16, batch=128, device=device)
    dec.eval()

    ts = eb['meta'].get('two_sided', False)
    A = eb['A'].float().to(device); B0 = eb['B0'].float().to(device)
    xtrue = eb['g_inp'].float().to(device)                                     # [n, 784] = true images
    if ts:
        inp = _adapter_input(A, B0, two_sided=True, grad_A=eb['grad_A'].float().to(device),
                             grad_B=eb['grad_B'].float().to(device), A0=eb['A0'].float().to(device))
    else:
        inp = _adapter_input(A, B0)
    with torch.no_grad():
        pred = dec(inp.to(device))                                             # [n, out*in]
        G = pred.reshape(-1, dec.out_features, dec.in_features).double()        # [n, 1000, 784]
        U, S, V = torch.svd_lowrank(G, q=6)                                     # top components
        xhat = V[:, :, 0]                                                       # [n, 784] input direction
    # SVD sign ambiguity: align to the true image
    sign = torch.sign((xhat * xtrue.double()).sum(1, keepdim=True))
    sign[sign == 0] = 1.0
    xhat = xhat * sign
    cos = torch.nn.functional.cosine_similarity(xhat, xtrue.double(), dim=1).mean().item()
    xh01, xt01 = _norm01(xhat).cpu(), _norm01(xtrue).cpu()
    Smat = get_ssim_all(xh01, xt01)                                            # [n, n]
    ssim = (Smat.diag().mean().item() if Smat.dim() == 2 else float(Smat.mean()))
    return summ['best_full_cos'], cos, ssim, xh01, xt01


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--activations', nargs='+', default=['softplus', 'gelu', 'relu'])
    p.add_argument('--n_train', type=int, default=12000)
    p.add_argument('--n_eval', type=int, default=128)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--two_sided', action='store_true',
                   help='nonzero-A measurement (observes col(B0)+row(A0)) — stronger, single-sample')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.set_default_dtype(torch.float32)

    print(f"# rank={args.rank} two_sided={args.two_sided}")
    print(f"{'activation':13s} {'decoder_cos':>11s} {'img_cos':>8s} {'img_SSIM':>9s}")
    print('-' * 45)
    grids = {}
    for act in args.activations:
        try:
            dcos, icos, iss, xh, xt = phase2(act, args.device, args.n_train, args.n_eval,
                                             args.rank, args.epochs, two_sided=args.two_sided)
        except Exception as e:
            print(f"  SKIP {act}: {type(e).__name__}: {e}"); continue
        print(f"{act:13s} {dcos:11.4f} {icos:8.4f} {iss:9.4f}")
        grids[act] = (xh[:8], xt[:8])

    # visual grid: true (top) vs recovered (bottom) for the first activation that produced output
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        os.makedirs(os.path.join(FIGURES_DIR, 'gradient_bridge'), exist_ok=True)
        for act, (xh, xt) in grids.items():
            fig, axs = plt.subplots(2, 8, figsize=(12, 3.2))
            for j in range(8):
                axs[0, j].imshow(xt[j, 0], cmap='gray'); axs[0, j].axis('off')
                axs[1, j].imshow(xh[j, 0], cmap='gray'); axs[1, j].axis('off')
            axs[0, 0].set_ylabel('true', rotation=0, ha='right'); axs[1, 0].set_ylabel('recon', rotation=0, ha='right')
            fig.suptitle(f'GB-Phase 2 ({act}): decoded-gradient -> image (top=true, bottom=recovered)')
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES_DIR, 'gradient_bridge', f'phase2_{act}.png'), dpi=120)
            plt.close(fig)
        print(f"\nGrids -> {FIGURES_DIR}/gradient_bridge/phase2_*.png")
    except Exception as e:
        print(f"  (grid skipped: {type(e).__name__}: {e})")


if __name__ == '__main__':
    main()
