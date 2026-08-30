"""Ecosystem PARTIAL-OVERLAP regime — the regime BETWEEN the two nulls.

atlas null = shared-is-everything (subtract too much); eco null = shared⊥private (subtract nothing). The
ecosystem effect, if real, lives where the θ0 common-mode PARTIALLY overlaps the private signal. This zoo
builds ANCHOR-DIGIT tasks {0,1},{0,2},{0,3},{0,4},{0,5} — all SHARE digit 0, differ in the second digit. So
the LOO-shared subspace partially captures the target's digit-0 signal; subtracting it should ISOLATE the
target's UNIQUE second-digit signal. Projection should now be MID-RANGE (not 0.001); distractors share digit
0 → harder retrieval → headroom. eco_analyze-compatible (task, seed, priv_imgs, ds_mean). bsub, float64, gelu.
"""
import argparse, os, torch
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.dataset_sensitivity.eco_zoo import digit_pair_data
from experiments.data_utils import _load_dataset

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/partial_zoo"
TASKS = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]    # anchor digit 0 → PARTIAL content overlap
SEEDS = list(range(400, 408))
N_PER_CLASS, T, RANK, LR, ACT = 2, 200, 8, 0.5, "gelu"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(N_PER_CLASS, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    print(f"[partial-zoo] anchor-digit tasks {TASKS} × {len(SEEDS)} seeds | N={2*N_PER_CLASS} (all share digit 0)")
    bank, nc, nt = [], 0, 0
    for task in TASKS:
        x_ft, y_ft = digit_pair_data(task, N_PER_CLASS, data_seed=42, ds=ds, device=dev)  # fixed private set per task
        x0 = x_ft - ds_mean
        for seed in SEEDS:
            A, B, mbce, _dW = train_adapter(frozen, b0, draw_B0(seed, out_f, RANK, dev), x0, y_ft, LR, T, act, RANK)
            conv = mbce < 1e-2; nt += 1; nc += int(conv)
            bank.append(dict(A=A[0].detach().cpu(), B=B[0].detach().cpu(), task=task, seed=seed,
                             max_bce=mbce, converged=conv,
                             priv_imgs=x_ft.reshape(x_ft.shape[0], -1).detach().cpu()))
        print(f"  task={task}: done ({len(SEEDS)} seeds)")
    print(f"[partial-zoo] converged {nc}/{nt}")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(bank=bank, ds_mean=ds_mean.reshape(-1).detach().cpu(),
                        meta=dict(tasks=TASKS, seeds=SEEDS, N=2 * N_PER_CLASS, T=T, rank=RANK, anchor=0)),
                   os.path.join(RESULTS, "partial_bank.pth"))
        print(f"[saved] {RESULTS}/partial_bank.pth ({len(bank)} adapters)")


if __name__ == "__main__":
    main()
