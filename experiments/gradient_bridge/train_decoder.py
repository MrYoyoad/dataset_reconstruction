"""GB-Phase 1 decoder training + honest evaluation.

Trains f_φ:(A,B)->∇_W L on a proxy pair bank with a cosine-similarity loss, then
reports BOTH metrics on a held-out split:
  - full cosine   : cos(decoder(A,B), ∇_W L)               <- the honest milestone (>0.9)
  - projected cos : cos(P_col(B0) ∇_W L, ∇_W L)            <- the measurement-only ceiling

If full-cosine ≈ projected-cosine, the decoder only recovers what the adapter
already measured (weak — expected for the near-analytic output layer). If
full-cosine clearly exceeds the projected ceiling, the decoder is genuinely
hallucinating the out-of-subspace component from the proxy prior (real signal —
the open question for the hidden layer).

Usage:
    python -m experiments.gradient_bridge.train_decoder \
        --bank results/gb_pairs_L2_r8.pth --epochs 200 --device cuda
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

import torch

from experiments.configs import RESULTS_DIR, FIGURES_DIR
from experiments.gradient_bridge.decoder import (
    GradientDecoder, cosine_loss, mean_cosine, project_onto_colspace,
    project_onto_two_sided,
)


def _grad3d(g_err, g_inp):
    """Per-sample gradient ∇_W L = Σ_i g_err_i ⊗ g_inp_i, shape [b, out, in].

    Handles both the rank-1 case (g_err [b,out], g_inp [b,in]) and the
    multi-sample case (g_err [b,m,out], g_inp [b,m,in], summed over m)."""
    if g_err.dim() == 3:
        return torch.einsum('bmo,bmi->boi', g_err, g_inp)
    return torch.einsum('bo,bi->boi', g_err, g_inp)


def _target_grad(g_err, g_inp):
    """∇_W L flattened to [b, out*in] (sum of outer products for m>1)."""
    return _grad3d(g_err, g_inp).reshape(g_err.shape[0], -1)


def _adapter_input(A, B0, two_sided=False, grad_A=None, grad_B=None, A0=None):
    """Flatten the decoder input.

    Single-sided: [A, B0]                       -> [b, r*in + out*r].
    Two-sided:    [∇_A L, ∇_B L, B0, A0]        -> [b, 2*r*in + 2*out*r].
    The two-sided layout exposes both measurement channels AND both bases."""
    b = A.shape[0]
    if two_sided:
        return torch.cat([grad_A.reshape(b, -1), grad_B.reshape(b, -1),
                          B0.reshape(b, -1), A0.reshape(b, -1)], dim=1)
    return torch.cat([A.reshape(b, -1), B0.reshape(b, -1)], dim=1)


def train(bank, epochs=200, batch=256, lr=1e-3, hidden=1024, depth=2,
          out_mode='auto', out_rank=16, val_frac=0.1, device='cuda',
          seed=0, verbose=True):
    torch.manual_seed(seed)
    meta = bank['meta']
    out_f, in_f = meta['out'], meta['in']
    lr_ft, scaling = meta['lr'], meta['scaling']
    two_sided = meta.get('two_sided', False)

    # float32 for decoder training (cosine is scale-free; halves memory).
    A = bank['A'].float(); B0 = bank['B0'].float()
    g_err = bank['g_err'].float(); g_inp = bank['g_inp'].float()
    n = A.shape[0]
    if two_sided:
        A0 = bank['A0'].float(); grad_A = bank['grad_A'].float(); grad_B = bank['grad_B'].float()
    else:
        A0 = grad_A = grad_B = None

    if out_mode == 'auto':
        out_mode = 'dense' if out_f * in_f <= 5000 else 'lowrank'
    if two_sided:
        in_dim = grad_A[0].numel() + grad_B[0].numel() + B0[0].numel() + A0[0].numel()
    else:
        in_dim = A[0].numel() + B0[0].numel()
    dec = GradientDecoder(in_dim, out_f, in_f, hidden=hidden, depth=depth,
                          out_mode=out_mode, out_rank=out_rank).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    print(f"Decoder: in_dim={in_dim} out={out_f}x{in_f} mode={out_mode} "
          f"two_sided={two_sided} m={meta.get('samples_per_pair', 1)} "
          f"params={sum(p.numel() for p in dec.parameters()):,}")

    # split
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(batch, int(n * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    def batches(idx, bs, shuffle):
        order = idx[torch.randperm(len(idx))] if shuffle else idx
        for i in range(0, len(order), bs):
            yield order[i:i + bs]

    def _inp(bi):
        """Assemble the decoder input for index batch bi (device tensors)."""
        a, b0 = A[bi].to(device), B0[bi].to(device)
        if two_sided:
            return _adapter_input(a, b0, two_sided=True, grad_A=grad_A[bi].to(device),
                                  grad_B=grad_B[bi].to(device), A0=A0[bi].to(device))
        return _adapter_input(a, b0)

    @torch.no_grad()
    def evaluate():
        dec.eval()
        full, proj, seen = 0.0, 0.0, 0
        for bi in batches(val_idx, batch, shuffle=False):
            b0 = B0[bi].to(device)
            ge, gi = g_err[bi].to(device), g_inp[bi].to(device)
            grad3d = _grad3d(ge, gi)                               # [b, out, in]
            tgt = grad3d.reshape(len(bi), -1)
            pred = dec(_inp(bi))
            if two_sided:
                proj_est = project_onto_two_sided(grad3d, b0, A0[bi].to(device))
            else:
                proj_est = project_onto_colspace(grad3d, b0)
            proj_est = proj_est.reshape(len(bi), -1)
            full += mean_cosine(pred, tgt) * len(bi)
            proj += mean_cosine(proj_est, tgt) * len(bi)
            seen += len(bi)
        return full / seen, proj / seen

    history = []
    best_full = -1.0
    for ep in range(epochs):
        dec.train()
        for bi in batches(tr_idx, batch, shuffle=True):
            tgt = _target_grad(g_err[bi].to(device), g_inp[bi].to(device))
            opt.zero_grad()
            loss = cosine_loss(dec(_inp(bi)), tgt)
            loss.backward()
            opt.step()
        full_c, proj_c = evaluate()
        history.append({'epoch': ep, 'val_full_cos': full_c, 'val_proj_cos': proj_c})
        best_full = max(best_full, full_c)
        if verbose and (ep % max(1, epochs // 20) == 0 or ep == epochs - 1):
            print(f"  ep {ep:>3d}: val full-cos={full_c:.4f}  "
                  f"proj-cos(ceiling)={proj_c:.4f}  best={best_full:.4f}")

    return dec, history, {'best_full_cos': best_full,
                          'final_full_cos': history[-1]['val_full_cos'],
                          'final_proj_cos': history[-1]['val_proj_cos'],
                          'out_mode': out_mode,
                          'two_sided': two_sided,
                          'samples_per_pair': meta.get('samples_per_pair', 1),
                          'rank': meta.get('rank'),
                          'hidden': hidden, 'depth': depth, 'out_rank': out_rank,
                          'meta': meta}


def _plot(history, save_path, layer_idx, rank):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eps = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(eps, [h['val_full_cos'] for h in history], marker='.', label='decoder full cosine')
    ax.plot(eps, [h['val_proj_cos'] for h in history], '--', color='gray',
            label='col(B₀) projection ceiling')
    ax.axhline(0.9, color='green', alpha=0.4, linestyle=':', label='milestone 0.9')
    ax.set_xlabel('epoch'); ax.set_ylabel('held-out cosine similarity')
    ax.set_title(f'Gradient bridge decoder — layer {layer_idx}, r={rank}')
    ax.legend(); ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"Saved: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--out_mode', type=str, default='auto',
                        choices=['auto', 'dense', 'lowrank'])
    parser.add_argument('--out_rank', type=int, default=16)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--tag', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    bank = torch.load(args.bank, weights_only=False)
    li, rk = bank['meta']['layer_idx'], bank['meta']['rank']
    tag = args.tag or f"L{li}_r{rk}"

    dec, history, summary = train(
        bank, epochs=args.epochs, batch=args.batch, lr=args.lr, hidden=args.hidden,
        depth=args.depth, out_mode=args.out_mode, out_rank=args.out_rank, device=device)

    print("\n=== Decoder result ===")
    print(f"  best held-out full cosine     = {summary['best_full_cos']:.4f}")
    print(f"  final full cosine             = {summary['final_full_cos']:.4f}")
    print(f"  final col(B0) projection cos  = {summary['final_proj_cos']:.4f}  (measurement ceiling)")
    gain = summary['final_full_cos'] - summary['final_proj_cos']
    print(f"  decoder gain over ceiling     = {gain:+.4f}  ->",
          "hallucinating out-of-subspace (real signal)" if gain > 0.05
          else "near ceiling (weak — mostly the measured projection)")
    print(f"  milestone >0.9 full cosine    :",
          "MET" if summary['best_full_cos'] > 0.9 else "not met")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save({'state_dict': dec.state_dict(), 'summary': summary, 'history': history},
               os.path.join(RESULTS_DIR, f"gb_decoder_{tag}.pth"))
    with open(os.path.join(RESULTS_DIR, f"gb_decoder_{tag}.json"), 'w') as f:
        json.dump({'summary': {k: v for k, v in summary.items() if k != 'meta'},
                   'history': history}, f, indent=2)
    _plot(history, os.path.join(FIGURES_DIR, 'gradient_bridge', f"gb_cosine_{tag}.png"), li, rk)
    print("\n=== Done ===")
