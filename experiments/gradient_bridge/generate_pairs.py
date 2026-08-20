"""GB-Phase 1 pair generation: LoRA adapter (A,B) -> full-rank gradient ∇_W L.

Each pair is a single-step LoRA measurement on ONE public-proxy sample:
  - fresh random B0 (Kaiming), A=0 init, one SGD step  ->  A1 = -lr·scaling·B0ᵀ∇_W L.
  - target ∇_W L is the per-sample gradient of the base weight; it is rank-1
    (∇_W L = g_err ⊗ g_inp), so we store the factors (g_err, g_inp), NOT the dense
    [out,in] matrix. This keeps ANY layer storable without the 150-200 GB blowup
    (experiment_plan.md GB-Phase 1 risk).

Fresh B0 per pair means the bank spans many measurement subspaces col(B0) — the
R2F setting where the decoder must learn to hallucinate the out-of-subspace
component from the proxy prior (the open question the milestone tests).

The per-sample factors come from ONE forward+backward over a proxy mini-batch,
using module hooks: the layer input gives g_inp per sample, the layer's
grad_output gives g_err per sample (undoing the 1/B mean-reduction scale).

Usage:
    python -m experiments.gradient_bridge.generate_pairs --layer 2 --rank 8 \
        --n_pairs 20000 --device cuda --save results/gb_pairs_L2_r8.pth
"""
import sys
import os
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

import torch

torch.set_default_dtype(torch.float64)

from experiments.configs import DATASETS_DIR, RESULTS_DIR, TRAIN_LR, LABELS_DICT, DATASET_SPECS
from experiments.run_experiment_b import create_model, load_pretrained
from experiments.data_utils import _load_dataset, _get_binary_label

def _layer_shapes(dataset):
    """(out, in) per Linear layer for the input_dim -> hidden... -> 1 MLP of `dataset`.

    Generalizes to any depth (2-hidden MNIST or the 4-hidden monster): hidden layer i has shape
    (h[i], dims[i]) where dims = [input_dim, h0, h1, ...]; the output layer is (1, h[-1])."""
    spec = DATASET_SPECS[dataset]; h = spec['hidden']; d = spec['input_dim']
    dims = [d] + list(h)
    shapes = {i: (h[i], dims[i]) for i in range(len(h))}
    shapes[len(h)] = (1, h[-1])
    return shapes


def _load_proxy(dataset, n, seed, device):
    """Load up to n public-proxy images of `dataset`: [<=n,C,H,W] + binary parity labels.

    Flowers splits are tiny (train=1020), so for flowers* we pool train+test (all public flowers;
    the victim's 2 test images are a negligible fraction). MNIST/fashion use train only (byte-compat)."""
    dss = [_load_dataset(dataset, train=True, root=DATASETS_DIR)]
    if dataset in ('flowers', 'flowers32', 'flowers64'):
        dss.append(_load_dataset(dataset, train=False, root=DATASETS_DIR))
    items = [(d, i) for d in dss for i in range(len(d))]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(items), generator=g)[:n]
    sel = [items[j][0][items[j][1]] for j in order]                        # one __getitem__ each
    xs = torch.stack([im for im, _ in sel]).to(device).double()           # [<=n,C,H,W]
    ys = torch.tensor([_get_binary_label(lab) for _, lab in sel],
                      device=device, dtype=torch.float64)
    return xs, ys


def _kaiming_B(out, rank, device, gen):
    """Standard LoRA B init: Kaiming-uniform, matching lora_wrapper.LoRALinear."""
    B = torch.empty(out, rank, dtype=torch.float64, device=device)
    # kaiming_uniform_(a=sqrt(5)) on a [out,rank] (fan_in=rank) matrix reduces to
    # bound = 1/sqrt(rank); matches lora_wrapper.LoRALinear._init_lora_weights.
    bound = 1.0 / math.sqrt(rank) if rank > 0 else 0.0
    return B.uniform_(-bound, bound, generator=gen)


def generate_pair_bank(n_pairs, layer_idx, rank, lr=TRAIN_LR, scaling=1.0,
                       activation='gelu', batch=256, seed=0, device='cuda',
                       grad_tol=1e-2, samples_per_pair=1, two_sided=False,
                       a_init_scale=0.1, dataset='mnist', verbose=True):
    """Generate n_pairs single-step LoRA measurements for one layer.

    Each pair aggregates ``samples_per_pair`` (=m) informative proxy samples, so
    the target gradient is ∇_W L = Σ_{i=1..m} g_err_i ⊗ g_inp_i (rank ≤ m — the
    realistic batch-gradient setting; m=1 is the original rank-1 case).

    Returns dict of tensors (n = number of pairs actually produced):
        A      [n, rank, in]        post-step LoRA A = -lr·scaling·B0ᵀ∇_W L
        B0     [n, out, rank]       the LoRA init B (column measurement basis)
        g_err  [n, out]  (m=1)  or  [n, m, out]  (m>1)   error factors
        g_inp  [n, in]   (m=1)  or  [n, m, in]   (m>1)   input factors
    with ∇_W L recoverable as Σ_i g_err_i ⊗ g_inp_i.

    Two-sided mode (--two_sided) additionally draws a small A0 = s·randn [rank,in]
    and stores BOTH analytic measurement channels + both bases:
        A0      [n, rank, in]       the row measurement basis
        grad_A  [n, rank, in]       ∇_A L = scaling·B0ᵀ∇_W L
        grad_B  [n, out, rank]      ∇_B L = scaling·∇_W L·A0ᵀ
    """
    spec = DATASET_SPECS[dataset]
    out_f, in_f = _layer_shapes(dataset)[layer_idx]
    m = int(samples_per_pair)
    # Use the smooth activation (matches the DI / anchor tracks). θ₀ is frozen.
    base = create_model(device=device, activation_name=activation,
                        input_dim=spec['input_dim'], hidden=spec['hidden'])
    base.load_state_dict(load_pretrained(device=device, pretrained_path=spec['pretrained'],
                                         input_dim=spec['input_dim'], hidden=spec['hidden']).state_dict())
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    layer = base.layers[layer_idx]
    layer.weight.requires_grad_(True)  # need grad on the target layer only

    # Hooks to capture per-sample layer input and grad_output.
    cap = {}
    h1 = layer.register_forward_hook(lambda mod, i, o: cap.__setitem__('inp', i[0].detach()))
    h2 = layer.register_full_backward_hook(
        lambda mod, gi, go: cap.__setitem__('gout', go[0].detach()))

    # We need n_pairs·m informative single-sample factors. Oversample the proxy
    # pool so gradient-norm filtering still yields enough. Samples the base model
    # already classifies confidently have ≈0 gradient — no LoRA signal, and an
    # ill-defined cosine — so they are dropped (else they poison the decoder loss
    # and the ceiling metric).
    need = n_pairs * m
    pool = min(60000, max(need + batch, 3 * need))
    xs, ys = _load_proxy(dataset, pool, seed, device)
    gen = torch.Generator(device=device).manual_seed(seed + 1)

    err_all, inp_all = [], []
    kept, seen = 0, 0
    pos = 0
    while kept < need and pos < xs.shape[0]:
        xb = xs[pos:pos + batch]
        yb = ys[pos:pos + batch]
        pos += xb.shape[0]
        b = xb.shape[0]
        base.zero_grad(set_to_none=True)
        logits = base(xb).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        # per-sample factors: g_inp = layer input; g_err = grad_output * b (undo mean/b)
        g_inp = cap['inp']                     # [b, in]
        g_err = cap['gout'] * b                # [b, out]  (mean-reduction had 1/b)
        seen += b
        # keep only informative samples (non-trivial per-sample gradient)
        mask = g_err.norm(dim=1) > grad_tol
        if mask.sum() == 0:
            continue
        err_all.append(g_err[mask].cpu()); inp_all.append(g_inp[mask].cpu())
        kept += int(mask.sum())
        if verbose and (kept % (batch * 10) < b or kept >= need):
            print(f"  kept {min(kept, need)}/{need} samples (from {seen} proxy samples)")

    h1.remove(); h2.remove()

    err_flat = torch.cat(err_all)                  # [kept, out]
    inp_flat = torch.cat(inp_all)                  # [kept, in]
    n_full = err_flat.shape[0] // m
    if n_full < n_pairs:
        print(f"  WARNING: only {n_full}/{n_pairs} pairs (m={m}) from a pool of "
              f"{pool} (grad_tol={grad_tol}); returning {n_full}.")
    n_pairs = min(n_pairs, n_full)
    survival = kept / max(seen, 1)
    if verbose:
        print(f"  informative-sample survival rate: {survival:.1%} "
              f"({kept} kept / {seen} seen, grad_tol={grad_tol}) -> {n_pairs} pairs (m={m})")

    # Group into [n_pairs, m, ...] and run the LoRA-step algebra on device.
    err_g = err_flat[:n_pairs * m].view(n_pairs, m, out_f).to(device)   # [p,m,out]
    inp_g = inp_flat[:n_pairs * m].view(n_pairs, m, in_f).to(device)    # [p,m,in]

    B0 = _kaiming_B(out_f * n_pairs, rank, device, gen).view(n_pairs, out_f, rank)  # [p,out,r]
    BT_err = torch.einsum('por,pmo->pmr', B0, err_g)                    # [p,m,r] = B0ᵀ g_err_i
    grad_A = scaling * torch.einsum('pmr,pmi->pri', BT_err, inp_g)      # [p,r,in] = scaling·B0ᵀ∇_W L
    A1 = -lr * grad_A                                                   # [p,r,in]

    # Squeeze the sample axis in the m=1 case for backward compatibility.
    g_err_store = err_g if m > 1 else err_g.squeeze(1)
    g_inp_store = inp_g if m > 1 else inp_g.squeeze(1)

    bank = {
        'A': A1.cpu(), 'B0': B0.cpu(),
        'g_err': g_err_store.cpu(), 'g_inp': g_inp_store.cpu(),
        'meta': {'layer_idx': layer_idx, 'rank': rank, 'lr': lr, 'scaling': scaling,
                 'activation': activation, 'out': out_f, 'in': in_f,
                 'n_pairs': n_pairs, 'grad_tol': grad_tol, 'survival_rate': survival,
                 'samples_per_pair': m, 'two_sided': two_sided, 'dataset': dataset,
                 'a_init_scale': a_init_scale if two_sided else 0.0},
    }

    if two_sided:
        A0 = a_init_scale * torch.randn(n_pairs, rank, in_f, dtype=torch.float64,
                                        device=device, generator=gen)          # [p,r,in]
        A0g = torch.einsum('pri,pmi->pmr', A0, inp_g)                          # [p,m,r] = A0 g_inp_i
        grad_B = scaling * torch.einsum('pmo,pmr->por', err_g, A0g)            # [p,out,r] = scaling·∇_W L·A0ᵀ
        bank['A0'] = A0.cpu(); bank['grad_A'] = grad_A.cpu(); bank['grad_B'] = grad_B.cpu()

    return bank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--n_pairs', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=TRAIN_LR)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grad_tol', type=float, default=1e-2)
    parser.add_argument('--samples_per_pair', type=int, default=1,
                        help='m: proxy samples aggregated per pair (rank-m batch gradient).')
    parser.add_argument('--two_sided', action='store_true',
                        help='Also measure via a small A0 (∇_A L and ∇_B L channels).')
    parser.add_argument('--a_init_scale', type=float, default=0.1,
                        help='Std scale s for A0 = s·randn (two-sided mode only).')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    bank = generate_pair_bank(args.n_pairs, args.layer, args.rank, lr=args.lr,
                              activation=args.activation, batch=args.batch,
                              seed=args.seed, device=device, grad_tol=args.grad_tol,
                              samples_per_pair=args.samples_per_pair,
                              two_sided=args.two_sided, a_init_scale=args.a_init_scale)
    out = args.save or os.path.join(RESULTS_DIR, f"gb_pairs_L{args.layer}_r{args.rank}.pth")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(bank, out)
    print(f"Saved {args.n_pairs} pairs to {out}  "
          f"(A {tuple(bank['A'].shape)}, B0 {tuple(bank['B0'].shape)}, "
          f"g_err {tuple(bank['g_err'].shape)}, g_inp {tuple(bank['g_inp'].shape)})")
