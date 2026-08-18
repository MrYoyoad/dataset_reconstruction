"""GB-Phase 2, full-pipeline test: decoded per-layer gradients -> full ΔW -> Experiment-B inverter.

Unlike the input-layer SVD (phase2_image.py), the Experiment-B extraction matches the FULL ΔW across
all layers and uses the known θ_0 architecture. The hidden-layer gradient decodes at 0.997 and contains
h0 = σ(W_0 x); the extraction can in principle invert through the known W_0/σ. So the full pipeline may
recover x even though the input-layer decode (0.637) alone does not.

Sensitivity form (decisive, reuses Experiment B): compute the TRUE single-step ΔW for a victim, then
corrupt each layer to its MEASURED decoder cosine (a random rotation preserving the norm), and run the
extraction. Arms isolate which layer's fidelity matters. If x is recovered when only the input layer is
degraded (hidden/output perfect), the bridge could work with good hidden decode; if not, the input layer
is essential and the bridge fails end-to-end. (The real per-layer decoder pipeline is the confirmation
step if this passes.)
"""
import argparse
import torch

from experiments.configs import TRAIN_LR
from experiments.run_experiment_b import create_model, load_pretrained
from experiments.data_utils import get_finetuning_data
from experiments.ntk_steps import compute_multi_step_update, compute_known_coefficients
from experiments.ntk_extraction import run_ntk_extraction
from experiments.metrics import compute_all_metrics


def corrupt(d, c, gen):
    """Rotate d to cosine c with itself, preserving ‖d‖ (c=1 -> unchanged)."""
    if c >= 0.9999:
        return d.clone()
    f = d.reshape(-1); nf = f.norm()
    if nf < 1e-12:
        return d.clone()
    fh = f / nf
    n = torch.randn(f.shape, generator=gen, device=f.device, dtype=f.dtype)
    n = n - (n @ fh) * fh
    n = n / (n.norm() + 1e-12)
    out = c * fh + (1.0 - c * c) ** 0.5 * n
    return (out * nf).reshape(d.shape)


def _layer_idx(key):
    for tok in key.split('.'):
        if tok.isdigit():
            return int(tok)
    return -1


def run_arm(activation, cos_by_layer, device, npc, seed, epochs):
    torch.set_default_dtype(torch.float64)
    x_ft, y_ft, digits, _ = get_finetuning_data(npc, seed=seed, device=device)
    model = load_pretrained(device=device)
    upd = compute_multi_step_update(model, x_ft.clone(), y_ft.clone(), lr=TRAIN_LR, n_steps=1,
                                    activation_name=activation)
    ds_mean, delta_w = upd['ds_mean'], upd['delta_w']
    x_cen = x_ft - ds_mean

    gen = torch.Generator(device=device).manual_seed(0)
    dw = {k: corrupt(v, cos_by_layer.get(_layer_idx(k), 1.0), gen) for k, v in delta_w.items()}

    m0 = create_model(device=device, activation_name=activation)
    m0.load_state_dict(upd['theta_0']); m0.eval()
    coeffs = compute_known_coefficients(m0, x_cen, y_ft)
    x_recon, _ = run_ntk_extraction(m0, dw, coeffs, TRAIN_LR, 1, npc,
                                    extraction_epochs=epochs, optimizer_type='lbfgs',
                                    device=device, verbose=False)
    met = compute_all_metrics(x_recon, x_cen, ds_mean)
    return met['ssim']['mean'], met['ssim_norm']['mean'], met['ssim_mean_baseline']['mean']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--activation', default='softplus')
    p.add_argument('--npc', type=int, default=1)          # N=2
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=50000)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    # measured per-layer decoder cosines (layer 0 input / 1 hidden / 2 output)
    ARMS = {
        'true (baseline)':          {0: 1.00, 1: 1.00, 2: 1.00},
        'decoded (0.64/0.997/0.94)':{0: 0.64, 1: 0.997, 2: 0.94},
        'input-only bad (0.64/1/1)':{0: 0.64, 1: 1.00, 2: 1.00},
        'hidden-only bad (1/0.64/1)':{0: 1.00, 1: 0.64, 2: 1.00},
        'uniform 0.64':             {0: 0.64, 1: 0.64, 2: 0.64},
    }
    print(f"# activation={args.activation} N={2*args.npc} T=1 (corrupt ΔW per layer -> extract)")
    print(f"{'arm':28s} {'ssim':>7s} {'ssim_norm':>9s} {'baseline':>9s}")
    print('-' * 58)
    for name, cbl in ARMS.items():
        try:
            s, sn, base = run_arm(args.activation, cbl, args.device, args.npc, args.seed, args.epochs)
            print(f"{name:28s} {s:7.3f} {sn:9.3f} {base:9.3f}")
        except Exception as e:
            print(f"{name:28s} SKIP: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
