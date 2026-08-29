"""Adapter-space atlas — FACTORIAL ZOO builder (plan §7; user-authorized GPU build 2026-08-29).

Builds a BALANCED, MULTI-SEED-PER-CELL population of LoRA adapters over the factorial
    {activation} × {composition = data-seed → distinct digit-pair} × {lr} × {init-seed},
and saves the raw factors (B, A) per cell — so BOTH clustering methods the user asked for run off the SAME
population: ΔW = B·A (gauge-clean product) AND the raw (B,A) pair. Saving (B,A) also lets us reconstruct ΔW
on load, so the bank stays small (A: r×in, B: out×r; not the dense 1000×784 product).

WHY this design (from the audited plan): the composition (digit-pair) is what we want to attribute; init/lr
are the nuisances; multiple INIT seeds per (composition×lr×activation) cell give the POWER to separate the
composition signal from seed/init noise (which the seed-mean arm data cannot). The frozen base is a fixed
pretrained checkpoint per activation (weights-mnist_<act>.pth), so composition lives only in the adapter.

Downstream (separate analysis, atlas_analyze.py): variance-decomposition of the ΔW-clustering across
{init,lr,activation,composition} (headline = "% different from init/lr"), the raw-(B,A) init-contrast
(UNcanonicalized — the init frame IS the signal, association-vs-null not bare divergence), and the
composition-recovery Facet-C with the §12 DML-IF cluster-robust CI. All observe-framed, weakest-attacker.

bsub-only. float64. Records max_bce per cell (drop/flag non-converged).

Run:  python -u -m experiments.dataset_sensitivity.atlas_zoo --save
"""
import argparse
import os
import torch

import experiments.jacobian_spectrum as J
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.data_utils import get_finetuning_data

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/atlas_zoo"

# ---- factorial (balanced, multi-seed-per-cell) ----
ACTS = ["gelu", "relu", "softplus_b1"]        # smooth C∞ / kinked / smooth C∞ (softplus_b1≈softplus; has a checkpoint)
COMPOSITIONS = [0, 1, 2, 3, 4]               # data seeds → distinct digit-pairs (recorded)
LRS = [0.1, 0.5]
INITS = [100, 101, 102, 103, 104, 105]       # B0 init seeds (the nuisance we must have power to separate)
N_PER_CLASS = 2                              # N=4, fixed across cells (composition = which digit-pair)
T = 200
RANK = 8
CONV_BCE = 1e-2                              # convergence flag threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--same_digits", action="store_true",
                    help="fixed digit-pair [0,1] across compositions → composition = different IMAGE samples "
                         "(the STRINGENT instance-level test, vs the default where each composition is a "
                         "different digit-set).")
    ap.add_argument("--out", default="zoo_bank.pth")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"[atlas-zoo] device={dev} | {len(ACTS)}×{len(COMPOSITIONS)}×{len(LRS)}×{len(INITS)} = "
          f"{len(ACTS)*len(COMPOSITIONS)*len(LRS)*len(INITS)} cells | N={2*N_PER_CLASS} T={T} rank={RANK}")

    bank, n_conv, n_tot = [], 0, 0
    for act_name in ACTS:
        act = make_activation(act_name)
        # fix the frozen base ONCE per activation (composition lives only in the adapter); reference seed 42
        xr, yr, _ = build_set(N_PER_CLASS, seed=42, device=dev, dataset=args.dataset)
        _, frozen, b0, _B0all, ds_mean = _honest_target(xr, yr, T, RANK, act_name, LRS[0], dev,
                                                        args.dataset, num_classes=2)
        out_f = frozen[0].shape[0]
        print(f"\n## activation={act_name}  base=weights-{args.dataset}_{act_name}.pth  out_f={out_f}")
        for comp in COMPOSITIONS:
            if args.same_digits:   # fixed classes {0,1}, comp-seed varies only the IMAGE sample (instance-level)
                x_ft, y_ft, digits, _ = get_finetuning_data(N_PER_CLASS, seed=comp, device=dev,
                                                            dataset=args.dataset, num_classes=2,
                                                            classes_present=[0, 1])
                x_ft, y_ft = x_ft.to(torch.float64), y_ft.to(torch.float64)
            else:
                x_ft, y_ft, digits = build_set(N_PER_CLASS, seed=comp, device=dev, dataset=args.dataset)
            x0 = (x_ft - ds_mean)
            dig_sig = tuple(sorted(set(int(d) for d in digits)))
            for lr in LRS:
                for init in INITS:
                    B0 = draw_B0(init, out_f, RANK, dev)
                    A, B, max_bce, _dW = train_adapter(frozen, b0, B0, x0, y_ft, lr, T, act, RANK)
                    conv = max_bce < CONV_BCE
                    n_tot += 1
                    n_conv += int(conv)
                    bank.append(dict(
                        A=A[0].detach().cpu(), B=B[0].detach().cpu(),   # raw factors (compact; ΔW=B@A on load)
                        activation=act_name, composition=comp, digits=dig_sig,
                        lr=lr, init_seed=init, max_bce=max_bce, converged=conv,
                    ))
            print(f"  comp={comp} digits={dig_sig}: done ({len(LRS)*len(INITS)} cells)")

    print(f"\n[atlas-zoo] converged {n_conv}/{n_tot} cells (max_bce<{CONV_BCE})")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        out = os.path.join(RESULTS, args.out)
        torch.save(dict(bank=bank, meta=dict(acts=ACTS, comps=COMPOSITIONS, lrs=LRS, inits=INITS,
                                             N=2 * N_PER_CLASS, T=T, rank=RANK, dataset=args.dataset,
                                             n_converged=n_conv, n_total=n_tot)), out)
        print(f"[saved] {out}  ({len(bank)} cells)")


if __name__ == "__main__":
    main()
