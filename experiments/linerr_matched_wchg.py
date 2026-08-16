"""Lemma B at MATCHED weight_change (closes the confound in notes/linearization_leakage_theory.tex).

Raw linearization error is confounded: L_lin ~ sigma'' * ||delta||^2, and ||delta|| (weight_change)
differs across activations, so softplus (which trains ~2x more) shows a high lin-error despite LOW
sigma''. This diagnostic calibrates the LR per activation (geometric bisection) to hit a COMMON target
weight_change, then computes the function-space Taylor residual. Prediction: at matched ||delta||,
lin-error follows sigma'' -- gelu / high-beta-softplus (high sigma'') ABOVE softplus / tanh (low
sigma''), and the kinked units (relu/leaky, Dirac curvature) highest. Full FT, no extraction -> fast.
"""
import argparse, csv, os
import torch

from experiments.configs import RESULTS_DIR
from experiments.run_experiment_b import create_model, load_pretrained
from experiments.data_utils import get_finetuning_data
from experiments.ntk_steps import compute_multi_step_update
from experiments.ntk_verification import compute_relative_weight_change, compute_function_space_lin_error


def _fit_wc(act, lr, x_ft, y_ft, T, device):
    m = load_pretrained(device=device)
    res = compute_multi_step_update(m, x_ft.clone(), y_ft.clone(), lr=lr, n_steps=T, activation_name=act)
    wc = compute_relative_weight_change(res['theta_0'], res['theta_T'])['overall']
    return wc, res


def linerr_at_matched(act, target_wc, x_ft, y_ft, T, device, tol=0.04, iters=22):
    lo, hi, res, wc = 1e-3, 2.0, None, None
    for _ in range(iters):
        mid = (lo * hi) ** 0.5                      # geometric bisection (wc grows with lr)
        wc, res = _fit_wc(act, mid, x_ft, y_ft, T, device)
        if abs(wc - target_wc) / target_wc < tol:
            break
        if wc < target_wc: lo = mid
        else: hi = mid
    theta_0, theta_T = res['theta_0'], res['theta_T']
    delta = {k: theta_T[k] - theta_0[k] for k in theta_0 if k in theta_T}
    x_cen = x_ft - res['ds_mean'] if res.get('ds_mean') is not None else x_ft
    m0 = create_model(device=device, activation_name=act); m0.load_state_dict(theta_0); m0.eval()
    mT = create_model(device=device, activation_name=act); mT.load_state_dict(theta_T); mT.eval()
    lin = compute_function_space_lin_error(m0, mT, x_cen, delta)['relative_error']
    return mid, wc, float(lin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--activations', nargs='+',
                   default=['softplus', 'tanh', 'silu', 'gelu', 'mish', 'sigmoid', 'elu', 'selu',
                            'hardswish', 'leaky_relu', 'relu',
                            'softplus_b0.5', 'softplus_b2', 'softplus_b5', 'softplus_b10', 'softplus_b50'])
    p.add_argument('--target_wc', type=float, default=0.05)
    p.add_argument('--n_steps', type=int, default=10)
    p.add_argument('--n_per_class', type=int, default=1)     # N=2 (lin-error is fn-space, N-agnostic)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default=os.path.join(RESULTS_DIR, 'linerr_matched_wchg.csv'))
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    x_ft, y_ft, digits, _ = get_finetuning_data(args.n_per_class, seed=args.seed, device=args.device)
    print(f"digits={digits}  target weight_change={args.target_wc}  T={args.n_steps}\n")
    hdr = f"{'act':13s} {'matched_lr':>11s} {'wchg':>8s} {'lin_err_fs':>11s}"
    print(hdr); print('-' * len(hdr))
    rows = []
    for act in args.activations:
        try:
            lr, wc, lin = linerr_at_matched(act, args.target_wc, x_ft, y_ft, args.n_steps, args.device)
        except Exception as e:
            print(f"  SKIP {act}: {type(e).__name__}: {e}"); continue
        rows.append({'activation': act, 'matched_lr': round(lr, 6), 'weight_change': round(wc, 5),
                     'lin_err_fs': round(lin, 6)})
        print(f"{act:13s} {lr:11.5f} {wc:8.4f} {lin:11.5f}")
    if rows:
        with open(args.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == '__main__':
    main()
