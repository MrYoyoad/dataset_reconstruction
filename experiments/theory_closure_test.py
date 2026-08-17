"""Theory closure (notes/linearization_leakage_theory.tex rev.3).

(1) Quantitative check of the corrected Lemma B: the RELATIVE anchor lin-error should track
    sigma'' / ||grad Phi|| (curvature relative to gradient magnitude), NOT sigma'' alone. We measure,
    per activation at theta_0: mean|sigma''| and the mean per-sample gradient norm ||grad Phi||, join
    with the matched-weight_change lin-error (results/linerr_matched_wchg.csv), and compare the
    Pearson correlation of lin-error with sigma'' vs with sigma''/||grad Phi||.
(2) High-k closure: eff_rank(X) of the private data for MNIST vs flowers32 vs flowers64 at N=64 --
    is flowers64 (d=12288) actually higher intrinsic-dim than MNIST, or (like flowers32) not?
Forward + one backward pass; no extraction.
"""
import csv, os
import torch
from experiments.configs import RESULTS_DIR
from experiments.run_experiment_b import make_activation, load_pretrained
from experiments.data_utils import get_finetuning_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ACTS = ['softplus', 'tanh', 'silu', 'gelu', 'mish', 'sigmoid', 'elu', 'selu', 'hardswish',
        'leaky_relu', 'relu', 'softplus_b0.5', 'softplus_b2', 'softplus_b5', 'softplus_b10', 'softplus_b50']


def eff_rank(A):
    s = torch.linalg.svdvals(A.double()); s = s[s > 1e-9]; p = s / s.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5; sy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (sx * sy + 1e-30)


def grad_stats(act, x_ft):
    torch.set_default_dtype(torch.float64)
    model = load_pretrained(device=DEVICE)
    model.activation = make_activation(act).to(DEVICE).double(); model.eval()
    N = x_ft.shape[0]
    norms = []
    for i in range(N):
        model.zero_grad(set_to_none=True)
        out = model(x_ft[i:i + 1]).sum()
        g = torch.autograd.grad(out, list(model.parameters()))
        norms.append(torch.cat([gg.reshape(-1) for gg in g]).norm().item())
    gnorm = sum(norms) / len(norms)
    Z = model.layers[0](x_ft.reshape(N, -1)).detach().clone().requires_grad_(True)
    try:
        sp = torch.autograd.grad(model.activation(Z).sum(), Z, create_graph=True)[0]
        sdd = torch.autograd.grad(sp.sum(), Z, allow_unused=True)[0]
        sdd = sdd if sdd is not None else torch.zeros_like(Z)
    except RuntimeError:
        sdd = torch.zeros_like(Z)
    return gnorm, float(sdd.abs().mean())


def main():
    torch.set_default_dtype(torch.float64)
    x_ft, _, _, _ = get_finetuning_data(1, seed=42, device=DEVICE)   # N=2, matches lin-error test

    lin = {}
    p = os.path.join(RESULTS_DIR, 'linerr_matched_wchg.csv')
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            lin[r['activation']] = float(r['lin_err_fs'])

    print("=== (1) corrected Lemma B: lin-error vs sigma'' vs sigma''/||grad Phi|| ===")
    hdr = f"{'act':13s} {'sigdd':>8s} {'||gradPhi||':>11s} {'sigdd/gn':>9s} {'lin_err':>8s}"
    print(hdr); print('-' * len(hdr))
    rows = []
    for act in ACTS:
        try:
            gn, sdd = grad_stats(act, x_ft)
        except Exception as e:
            print(f"  SKIP {act}: {type(e).__name__}: {e}"); continue
        le = lin.get(act)
        ratio = sdd / (gn + 1e-30)
        rows.append({'activation': act, 'sigma_dd': sdd, 'grad_norm': gn, 'sigdd_over_gn': ratio,
                     'lin_err_fs': le})
        print(f"{act:13s} {sdd:8.4f} {gn:11.4f} {ratio:9.5f} {(le if le is not None else float('nan')):8.4f}")

    paired = [(r['sigma_dd'], r['sigdd_over_gn'], r['lin_err_fs']) for r in rows if r['lin_err_fs'] is not None]
    if len(paired) >= 3:
        sdds, ratios, les = zip(*paired)
        print(f"\n  Pearson(lin_err, sigma'')          = {pearson(list(sdds), list(les)):+.3f}")
        print(f"  Pearson(lin_err, sigma''/||gradPhi||) = {pearson(list(ratios), list(les)):+.3f}")
        print("  (the correction wins if the 2nd is markedly higher)")

    print("\n=== (2) high-k closure: eff_rank(X) at N=64 ===")
    for ds in ('mnist', 'flowers32', 'flowers64'):
        try:
            x, _, _, _ = get_finetuning_data(32, seed=42, dataset=ds, device=DEVICE)
            X = x.reshape(x.shape[0], -1)
            print(f"  {ds:10s} N={X.shape[0]} d={X.shape[1]:5d}  eff_rank(X)={eff_rank(X):.2f}")
        except Exception as e:
            print(f"  {ds:10s} SKIP: {type(e).__name__}: {e}")

    if rows:
        out = os.path.join(RESULTS_DIR, 'theory_closure_test.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == '__main__':
    main()
