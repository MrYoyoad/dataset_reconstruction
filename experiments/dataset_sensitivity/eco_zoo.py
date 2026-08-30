"""Ecosystem attack — WEAK-SIGNAL MULTI-TASK zoo (plan §2, user-authorized build).

Many adapters on DIFFERENT DISJOINT-content tasks sharing one base theta0: 5 disjoint digit-pairs
{0,1},{2,3},{4,5},{6,7},{8,9} (each even+odd → a binary parity task on DISJOINT digits), K init seeds each,
small N (few images) for HEADROOM. Saves raw (B,A) + the private images per adapter so the analysis can do
LOO common-mode subtraction + NN-retrieval of the target's private images. bsub-only, float64, gelu base.
"""
import argparse, os, torch, math
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.data_utils import _load_dataset, _get_binary_label

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/eco_zoo"
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]   # disjoint digit-pairs
SEEDS = list(range(200, 208))                       # 8 init seeds per task
N_PER_CLASS = 2                                      # N=4 total — WEAK signal (headroom)
T, RANK, ACT = 200, 8, "gelu"


def digit_pair_data(digits, n_per_class, data_seed, ds, device):
    tgt = ds.targets if torch.is_tensor(ds.targets) else torch.tensor(ds.targets)
    dat = ds.data
    g = torch.Generator().manual_seed(int(data_seed))
    xs, ys = [], []
    for d in digits:
        idx = (tgt == d).nonzero(as_tuple=True)[0]
        pick = idx[torch.randperm(len(idx), generator=g)[:n_per_class]]
        for i in pick:
            xs.append(dat[int(i)].to(torch.float64).view(-1) / 255.0)
            ys.append(float(_get_binary_label(int(d))))
    return torch.stack(xs).to(device), torch.tensor(ys, dtype=torch.float64, device=device)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True)
    act = make_activation(ACT)
    # fix the base once (gelu checkpoint); reference set only for frozen/b0/ds_mean
    xr, yr, _ = build_set(N_PER_CLASS, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, 0.5, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    print(f"[eco-zoo] {len(TASKS)} disjoint tasks × {len(SEEDS)} seeds, N={2*N_PER_CLASS}, base=gelu, out_f={out_f}")
    bank, nc, nt = [], 0, 0
    for task in TASKS:
        x_ft, y_ft = digit_pair_data(task, N_PER_CLASS, data_seed=42, ds=ds, device=dev)  # fixed private set per task
        x0 = x_ft - ds_mean
        for seed in SEEDS:
            A, B, mbce, _dW = train_adapter(frozen, b0, draw_B0(seed, out_f, RANK, dev), x0, y_ft, 0.5, T, act, RANK)
            conv = mbce < 1e-2; nt += 1; nc += int(conv)
            bank.append(dict(A=A[0].detach().cpu(), B=B[0].detach().cpu(), task=task, seed=seed,
                             max_bce=mbce, converged=conv, priv_imgs=x_ft.detach().cpu()))  # private images for retrieval
        print(f"  task={task}: done ({len(SEEDS)} seeds)")
    print(f"[eco-zoo] converged {nc}/{nt}")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(bank=bank, ds_mean=ds_mean.detach().cpu(),
                        meta=dict(tasks=TASKS, seeds=SEEDS, N=2*N_PER_CLASS, T=T, rank=RANK)),
                   os.path.join(RESULTS, "eco_bank.pth"))
        print(f"[saved] {RESULTS}/eco_bank.pth ({len(bank)} adapters)")


if __name__ == "__main__":
    main()
