"""GB-Phase 2 END-TO-END: real decoded per-layer ΔW -> Experiment-B extraction (NOT SVD).

The culmination of the bridge track. The learning chain so far:
  - the input-layer SVD inverter gives only a COARSE image (SSIM 0.17 even at decode 0.945) because it
    is a prior-free DETERMINED rank-1 factorization -- priors (TV/L1/non-neg) and higher rank do not help;
  - the corruption proxy (phase2_full.py) showed the MODEL-BASED Experiment-B extraction recovers
    SSIM ~0.5 from a ΔW whose input layer is at cosine 0.64 (vs the SVD's 0.10) -- so the SVD, not the
    decoder, was the weak link. The extraction uses the known θ0 + all-layer structure as an implicit
    prior, closing the cosine->pixel gap.

This wires the REAL trained per-layer decoders into that extraction (no corruption stand-in):
  1. train a two-sided decoder for each layer {0,1,2} on public MNIST proxy pairs;
  2. for the victim's N fine-tuning samples, measure each sample's per-layer single-step LoRA update,
     decode it, and SUM the per-sample decoded gradients -> the aggregate decoded ΔW per layer
     (scaled to the true per-layer norm, preserving the decoded DIRECTION);
  3. run run_ntk_extraction on the decoded ΔW.

Arms: TRUE ΔW (upper bound) / DECODED all-layers (the real end-to-end attack) / DECODED input-only
(does the model-based inverter beat the SVD's 0.10 on the same decoded input layer?). If the decoded
all-layers arm recovers x near the TRUE-ΔW ceiling, the bridge attack closes end-to-end.
"""
import argparse
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dataset_reconstruction'))

import torch
import torch.nn.functional as F

from experiments.configs import TRAIN_LR, FIGURES_DIR, RESULTS_DIR
from experiments.run_experiment_b import create_model, load_pretrained
from experiments.data_utils import get_finetuning_data
from experiments.ntk_steps import compute_multi_step_update, compute_known_coefficients
from experiments.ntk_extraction import run_ntk_extraction
from experiments.metrics import compute_all_metrics
from experiments.gradient_bridge.generate_pairs import generate_pair_bank, _kaiming_B
from experiments.gradient_bridge.train_decoder import train, _adapter_input


def _layer_key(delta_w, li):
    for k in delta_w:
        for tok in k.split('.'):
            if tok.isdigit() and int(tok) == li:
                return k
    raise KeyError(li)


def measure_victim(model, x, y, layers, rank, a_init_scale, lr, gen, device):
    """Per-sample two-sided single-step LoRA measurement for the victim, every layer at once.

    Mirrors generate_pairs' algebra (g_err = grad_output*B undoes the 1/B mean-reduction) but on the
    given victim (x, y) instead of proxy MNIST. Returns per-layer dicts of per-sample tensors."""
    caps, handles = {}, []
    for li in layers:
        layer = model.layers[li]
        handles.append(layer.register_forward_hook(
            lambda m, i, o, li=li: caps.__setitem__((li, 'inp'), i[0].detach())))
        handles.append(layer.register_full_backward_hook(
            lambda m, gi, go, li=li: caps.__setitem__((li, 'gout'), go[0].detach())))
    model.zero_grad(set_to_none=True)
    logits = model(x).view(-1)
    F.binary_cross_entropy_with_logits(logits, y).backward()
    for h in handles:
        h.remove()

    B = x.shape[0]
    meas = {}
    for li in layers:
        g_inp = caps[(li, 'inp')]                       # [B, in]
        g_err = caps[(li, 'gout')] * B                  # [B, out]  undo mean/B
        out_f, in_f = g_err.shape[1], g_inp.shape[1]
        B0 = _kaiming_B(out_f * B, rank, device, gen).view(B, out_f, rank)      # [B,out,r]
        A0 = a_init_scale * torch.randn(B, rank, in_f, dtype=x.dtype, device=device, generator=gen)
        BT_err = torch.einsum('bor,bo->br', B0, g_err)                          # [B,r]
        grad_A = torch.einsum('br,bi->bri', BT_err, g_inp)                      # [B,r,in] = B0ᵀ∇W
        A1 = -lr * grad_A
        A0g = torch.einsum('bri,bi->br', A0, g_inp)                             # [B,r]
        grad_B = torch.einsum('bo,br->bor', g_err, A0g)                         # [B,out,r] = ∇W·A0ᵀ
        meas[li] = dict(A=A1, B0=B0, A0=A0, grad_A=grad_A, grad_B=grad_B,
                        g_err=g_err, g_inp=g_inp, out=out_f, in_=in_f)
    return meas


def decode_aggregate(dec, m, true_dw_L, device):
    """Decode each victim sample's per-layer gradient, SUM, scale to ‖true ΔW_L‖ (keep decoded dir).

    Returns (decoded_dw_L [out,in], aggregate decode cosine vs the true ΔW direction)."""
    inp = _adapter_input(m['A'], m['B0'], two_sided=True,
                         grad_A=m['grad_A'], grad_B=m['grad_B'], A0=m['A0'])
    with torch.no_grad():
        Ghat = dec(inp.float().to(device)).reshape(-1, dec.out_features, dec.in_features).double()
    Gagg = Ghat.sum(0)                                                         # [out,in]
    tv, gv = true_dw_L.reshape(-1), Gagg.reshape(-1)
    cos = (gv @ tv) / (gv.norm() * tv.norm() + 1e-12)
    dw = Gagg / (Gagg.norm() + 1e-12) * true_dw_L.norm()                        # scale to true norm
    return dw, cos.item()


def extract(m0, dw, coeffs, npc, epochs, device):
    x_recon, _ = run_ntk_extraction(m0, dw, coeffs, TRAIN_LR, 1, npc,
                                    extraction_epochs=epochs, optimizer_type='lbfgs',
                                    device=device, verbose=False)
    return x_recon


def run(activation, device, npc, seed, n_train, dec_epochs, ext_epochs, rank, a_init_scale):
    layers = [0, 1, 2]

    # --- 1. train a two-sided decoder per layer on public proxy pairs ---
    # Decoders train in float32 (matches phase2_image; the loaded model stays float64 internally);
    # the victim measurement + extraction below switch to float64 (matches phase2_full).
    torch.set_default_dtype(torch.float32)
    decs, dcos = {}, {}
    for li in layers:
        tb = generate_pair_bank(n_train, li, rank, activation=activation, seed=0, device=device,
                                verbose=False, two_sided=True, a_init_scale=a_init_scale)
        dec, _, summ = train(tb, epochs=dec_epochs, out_mode='auto', out_rank=16, batch=128,
                             device=device, verbose=False)
        dec.eval()
        decs[li] = dec
        dcos[li] = summ['best_full_cos']
        print(f"  layer {li}: decoder proxy full-cos = {summ['best_full_cos']:.4f}")

    # --- 2. victim: true single-step ΔW + oracle coefficients (matches phase2_full) ---
    torch.set_default_dtype(torch.float64)
    x_ft, y_ft, digits, _ = get_finetuning_data(npc, seed=seed, device=device)
    m_meas = create_model(device=device, activation_name=activation)   # frozen θ0 model for measurement
    m_meas.load_state_dict(load_pretrained(device=device).state_dict())
    m_meas.eval()
    for p in m_meas.parameters():
        p.requires_grad_(False)
    for li in layers:
        m_meas.layers[li].weight.requires_grad_(True)

    upd = compute_multi_step_update(m_meas, x_ft.clone(), y_ft.clone(), lr=TRAIN_LR, n_steps=1,
                                    activation_name=activation)
    ds_mean, true_dw = upd['ds_mean'], upd['delta_w']
    x_cen = x_ft - ds_mean

    # --- 3. measure victim per-sample, decode, aggregate ---
    gen = torch.Generator(device=device).manual_seed(123)
    meas = measure_victim(m_meas, x_ft.clone(), y_ft.clone(), layers, rank, a_init_scale,
                          TRAIN_LR, gen, device)
    decoded_dw, agg_cos = {}, {}
    for li in layers:
        key = _layer_key(true_dw, li)
        dw_L, cos_L = decode_aggregate(decs[li], meas[li], true_dw[key], device)
        decoded_dw[key] = dw_L
        agg_cos[li] = cos_L

    # --- 4. extraction arms ---
    m0 = create_model(device=device, activation_name=activation)
    m0.load_state_dict(upd['theta_0']); m0.eval()
    coeffs = compute_known_coefficients(m0, x_cen, y_ft)

    def metrics_of(dw):
        met = compute_all_metrics(extract(m0, dw, coeffs, npc, ext_epochs, device), x_cen, ds_mean)
        return met['ssim']['mean'], met['ssim_norm']['mean'], met['ssim_mean_baseline']['mean']

    input_key = _layer_key(true_dw, 0)
    arms = {
        'TRUE ΔW (ceiling)':      dict(true_dw),
        'DECODED all-layers':     dict(decoded_dw),
        'DECODED input-only':     {input_key: decoded_dw[input_key]},
        'TRUE input-only':        {input_key: true_dw[input_key]},
    }
    print(f"\n# {activation}  N={2*npc}  aggregate decode cos: "
          f"L0={agg_cos[0]:.3f} L1={agg_cos[1]:.3f} L2={agg_cos[2]:.3f}")
    print(f"{'arm':22s} {'ssim':>7s} {'ssim_norm':>9s} {'baseline':>9s}")
    print('-' * 52)
    results = {}
    for name, dw in arms.items():
        try:
            s, sn, base = metrics_of(dw)
            print(f"{name:22s} {s:7.3f} {sn:9.3f} {base:9.3f}")
            results[name] = (s, sn, base)
        except Exception as e:
            print(f"{name:22s} SKIP: {type(e).__name__}: {e}")
    return results, agg_cos, dcos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--activations', nargs='+', default=['softplus', 'gelu'])
    p.add_argument('--npc', type=int, default=1)              # N = 2*npc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_train', type=int, default=15000)
    p.add_argument('--dec_epochs', type=int, default=120)
    p.add_argument('--ext_epochs', type=int, default=50000)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--a_init_scale', type=float, default=0.1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    for act in args.activations:
        print(f"\n########## {act} ##########")
        try:
            run(act, args.device, args.npc, args.seed, args.n_train, args.dec_epochs,
                args.ext_epochs, args.rank, args.a_init_scale)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  SKIP {act}: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
