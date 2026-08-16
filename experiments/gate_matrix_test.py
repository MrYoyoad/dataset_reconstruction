"""Gate-matrix diagnostic (test from notes/linearization_leakage_theory.tex, Section 8).

For each activation, at the anchor theta_0, form:
  - the gate matrix M[k,i] = sigma'(<w_k, x_i>)  (first hidden layer), shape [width, N];
  - the full per-sample gradient features phi(x_i) = grad_theta f(x_i), matrix Phi [P, N];
and report, per activation:
  - eff_rank(M), eff_rank(Phi)      (entropy effective rank of the singular values)
  - mean per-neuron gate variance   Var_i sigma'(z_k^i)         (Lemma A: the info channel)
  - mean per-neuron gate range      max_i sigma'(z_k^i) - min_i ...
  - mean |pairwise cosine| of the columns of M and of Phi       (collinearity <-> superposition)
  - mean |sigma''| over the pre-activations                     (curvature that also sets lin-error)

Theory predictions (Prop. 1 + gate-range): kinked (relu) -> high eff_rank / low cosine / large gate
range (separable, leaks); smooth with clustered pre-activations (softplus) -> eff_rank ~1 / high cosine
/ tiny gate range (entangled). eff_rank(M) should track the measured LoRA leakage margins; |sigma''|
should track the linearization error. Forward + one backward pass only; no extraction.
"""
import argparse
import csv
import os

import torch

from experiments.configs import RESULTS_DIR, ACTIVATION_CHOICES
from experiments.run_experiment_b import make_activation, load_pretrained
from experiments.data_utils import get_finetuning_data


def effective_rank(A):
    """Entropy effective rank (Roy & Vetterli): exp(-sum p_i log p_i), p_i = s_i / sum s."""
    s = torch.linalg.svdvals(A.double())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def mean_abs_offdiag_cosine(cols):
    """cols: [D, N] — mean |cosine| between distinct columns."""
    C = cols / (cols.norm(dim=0, keepdim=True) + 1e-30)
    G = (C.t() @ C).abs()
    n = G.shape[0]
    if n < 2:
        return float('nan')
    off = (G.sum() - G.diag().sum()) / (n * (n - 1))
    return float(off)


@torch.no_grad()
def _preacts(model, X):
    return model.layers[0](X)  # first linear layer output [N, width]


def diagnose(act, n_per_class, seed, device):
    torch.set_default_dtype(torch.float64)
    x_ft, y_ft, digits, _ = get_finetuning_data(n_per_class, seed=seed, device=device)
    N = x_ft.shape[0]
    Xf = x_ft.reshape(N, -1).double()

    model = load_pretrained(device=device)         # theta_0, correct architecture + weights
    model.activation = make_activation(act).to(device).double()
    model.eval()

    # --- gate matrix M[k,i] = sigma'(z_k^i), z from the first hidden layer ---
    Z = _preacts(model, Xf).detach().clone().requires_grad_(True)        # [N, width]
    sp = torch.autograd.grad(model.activation(Z).sum(), Z)[0]            # sigma'(Z)  [N, width]
    # sigma''(Z) via double backward. For kinked units (relu/leaky_relu/hardswish) the curvature is a
    # Dirac autograd cannot see -> it returns 0/None; that is expected (their info lives in gate_range).
    try:
        Z2 = Z.detach().clone().requires_grad_(True)
        spp = torch.autograd.grad(model.activation(Z2).sum(), Z2, create_graph=True)[0]
        sdd = torch.autograd.grad(spp.sum(), Z2, allow_unused=True)[0]
        sigma_dd = sdd if sdd is not None else torch.zeros_like(Z)
    except RuntimeError:
        sigma_dd = torch.zeros_like(Z)
    M = sp.t()                                                           # [width, N]
    gate_var = float(sp.var(dim=0, unbiased=False).mean())              # Var_i sigma'(z_k^i), mean over k
    gate_range = float((sp.max(dim=0).values - sp.min(dim=0).values).mean())
    mean_abs_spp = float(sigma_dd.abs().mean())

    # --- full per-sample gradient features phi(x_i) = grad_theta f(x_i) ---
    feats = []
    for i in range(N):
        model.zero_grad(set_to_none=True)
        out = model(x_ft[i:i + 1]).sum()
        g = torch.autograd.grad(out, list(model.parameters()), retain_graph=False)
        feats.append(torch.cat([gg.reshape(-1) for gg in g]).detach())
    Phi = torch.stack(feats, dim=1)                                      # [P, N]

    return {
        'activation': act, 'N': N, 'digits': str(digits),
        'eff_rank_M': round(effective_rank(M), 4),
        'eff_rank_phi': round(effective_rank(Phi), 4),
        'gate_var': round(gate_var, 6),
        'gate_range': round(gate_range, 6),
        'mean_abs_sigma_dd': round(mean_abs_spp, 6),
        'cos_M': round(mean_abs_offdiag_cosine(M), 4),
        'cos_phi': round(mean_abs_offdiag_cosine(Phi), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--activations', nargs='+',
                   default=['softplus', 'silu', 'gelu', 'gelu_tanh', 'mish', 'tanh', 'sigmoid',
                            'elu', 'celu', 'selu', 'hardswish', 'leaky_relu', 'relu',
                            'softplus_b0.5', 'softplus_b2', 'softplus_b5', 'softplus_b10', 'softplus_b50'])
    p.add_argument('--n_per_class', type=int, nargs='+', default=[1, 5])   # N = 2, 10
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default=os.path.join(RESULTS_DIR, 'gate_matrix_test.csv'))
    args = p.parse_args()

    rows = []
    for npc in args.n_per_class:
        print(f"\n===== N = {2*npc} (seed {args.seed}) — smooth -> kinked =====")
        hdr = (f"{'act':13s} {'effrk_M':>8s} {'effrk_phi':>9s} {'gate_var':>9s} {'gate_rng':>9s} "
               f"{'mean|sdd|':>9s} {'cos_M':>7s} {'cos_phi':>7s}")
        print(hdr); print('-' * len(hdr))
        for act in args.activations:
            try:
                r = diagnose(act, npc, args.seed, args.device)
            except Exception as e:
                print(f"  SKIP {act}: {type(e).__name__}: {e}"); continue
            rows.append(r)
            print(f"{r['activation']:13s} {r['eff_rank_M']:8.3f} {r['eff_rank_phi']:9.3f} "
                  f"{r['gate_var']:9.5f} {r['gate_range']:9.5f} {r['mean_abs_sigma_dd']:8.4f} "
                  f"{r['cos_M']:7.3f} {r['cos_phi']:7.3f}")

    if rows:
        cols = list(rows[0].keys())
        with open(args.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == '__main__':
    main()
