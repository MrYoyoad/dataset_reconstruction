"""A (build) — RESOLUTION-LIMIT zoo (swap-k). Removes crutch ii (max-separable candidate sets).

At each swap level k_pc (images swapped PER CLASS), build M candidate sets that SHARE a common core of
(Npc−k_pc) images/class and differ only in k_pc set-UNIQUE images/class. k_pc small ⇒ sets nearly identical
(hard); k_pc=Npc ⇒ disjoint (the current easy case). All images DISJOINT (core + every set's unique). Trains
M sets × K inits per k-level on the shared gelu base; downstream a_resolution.py does leave-one-init-out
matching accuracy vs k → the resolution CURVE. Npc=4 (N=8) so the finest cut (k_pc=1) = "sets differ by ONE
image per class". bsub, float64, gelu.
"""
import argparse, os, torch
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.data_utils import _load_dataset, _get_binary_label

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/a_resolution_zoo"
DIGITS = (0, 1)
M_SETS = 10                       # chance = 1/10 = 0.10
KPC = [1, 2, 3, 4]                # images swapped per class; 4 = disjoint, 1 = differ by 1/class
INITS = [700, 701, 702, 703]
N_PC = 4                          # N=8
T, RANK, LR, ACT = 200, 8, 0.5, "gelu"


def build_sets(ds, k_pc, seed):
    """M sets sharing a (N_PC−k_pc)/class core + k_pc unique/class, all images disjoint."""
    tgt = ds.targets if torch.is_tensor(ds.targets) else torch.tensor(ds.targets)
    dat = ds.data; g = torch.Generator().manual_seed(seed)
    img = lambda i: dat[int(i)].to(torch.float64).unsqueeze(0) / 255.0
    pool = {d: (tgt == d).nonzero(as_tuple=True)[0][torch.randperm(int((tgt == d).sum()), generator=g)] for d in DIGITS}
    ptr = {d: 0 for d in DIGITS}
    core = {d: [pool[d][ptr[d] + j] for j in range(N_PC - k_pc)] for d in DIGITS}
    for d in DIGITS:
        ptr[d] += N_PC - k_pc
    sets = []
    for s in range(M_SETS):
        xs, ys = [], []
        for d in DIGITS:
            idxs = list(core[d]) + [pool[d][ptr[d] + j] for j in range(k_pc)]
            ptr[d] += k_pc
            for i in idxs:
                xs.append(img(i)); ys.append(float(_get_binary_label(int(d))))
        sets.append((torch.stack(xs), torch.tensor(ys, dtype=torch.float64)))
    return sets


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(N_PC, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    print(f"[A-zoo] M={M_SETS} sets × {len(INITS)} inits × k_pc={KPC} | N={2*N_PC} | chance={1/M_SETS:.2f}")
    bank, nc, nt = [], 0, 0
    for k_pc in KPC:
        sets = build_sets(ds, k_pc, seed=800 + k_pc)   # distinct images per k-level
        for sid, (x_ft, y_ft) in enumerate(sets):
            x_ft = x_ft.to(dev); y_ft = y_ft.to(dev); x0 = x_ft - ds_mean
            for init in INITS:
                A, B, mbce, _dW = train_adapter(frozen, b0, draw_B0(init, out_f, RANK, dev), x0, y_ft, LR, T, act, RANK)
                conv = mbce < 1e-2; nt += 1; nc += int(conv)
                bank.append(dict(A=A[0].detach().cpu(), B=B[0].detach().cpu(), k_pc=k_pc, set_id=sid,
                                 init=init, max_bce=mbce, converged=conv))
        print(f"  k_pc={k_pc}: done ({M_SETS} sets × {len(INITS)} inits)")
    print(f"[A-zoo] converged {nc}/{nt}")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(bank=bank, meta=dict(M=M_SETS, kpc=KPC, inits=INITS, N=2 * N_PC, digits=DIGITS)),
                   os.path.join(RESULTS, "a_bank.pth"))
        print(f"[saved] {RESULTS}/a_bank.pth ({len(bank)} adapters)")


if __name__ == "__main__":
    main()
