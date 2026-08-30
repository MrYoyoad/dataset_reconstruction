"""C (build) — MIA zoo with a GLOBALLY-HELD-OUT negative pool (auditor's fix).

Partitions the MNIST {0,1} images into DISJOINT slices: a negative pool used in NO adapter's private set, then
one private slice per adapter. Trains K distinct private sets × few inits on the shared gelu base, saving raw
(B,A) + each adapter's private images + the global negative pool + the same-distribution mean μ (mean of {0,1}).
This lets C score membership with negatives that are non-members of EVERY adapter (not just the target) — the
hole that would otherwise inflate the pooled AUC via shared {0,1} structure. bsub, float64, gelu.
"""
import argparse, os, torch
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.data_utils import _load_dataset, _get_binary_label

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/mia_zoo"
DIGITS = (0, 1)
N_SETS = 10                 # distinct private sets
INITS = [600, 601]          # 2 inits each → 20 adapters, cluster over the 10 sets
N_PER_CLASS = 2             # N=4 private images
NEG_PER_CLASS = 40          # 80 globally-held-out negatives
T, RANK, LR, ACT = 200, 8, 0.5, "gelu"


def partition(ds, seed=7):
    tgt = ds.targets if torch.is_tensor(ds.targets) else torch.tensor(ds.targets)
    dat = ds.data; g = torch.Generator().manual_seed(seed)
    pools = {d: (tgt == d).nonzero(as_tuple=True)[0][torch.randperm(int((tgt == d).sum()), generator=g)] for d in DIGITS}
    flat = lambda i: dat[int(i)].to(torch.float64).view(-1) / 255.0
    img = lambda i: dat[int(i)].to(torch.float64).unsqueeze(0) / 255.0
    neg, ptr = [], {}
    for d in DIGITS:
        for i in pools[d][:NEG_PER_CLASS]:
            neg.append(flat(i))
        ptr[d] = NEG_PER_CLASS
    sets = []
    for s in range(N_SETS):
        xs, ys = [], []
        for d in DIGITS:
            for _ in range(N_PER_CLASS):
                i = pools[d][ptr[d]]; ptr[d] += 1
                xs.append(img(i)); ys.append(float(_get_binary_label(int(d))))
        sets.append((torch.stack(xs), torch.tensor(ys, dtype=torch.float64)))
    # mu = same-distribution mean over a large {0,1} sample (all reserved indices)
    allidx = torch.cat([pools[d][:600] for d in DIGITS])
    mu = torch.stack([flat(i) for i in allidx]).mean(0)
    return sets, torch.stack(neg), mu


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(N_PER_CLASS, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    sets, neg, mu = partition(ds)
    print(f"[mia-zoo] {N_SETS} private sets × {len(INITS)} inits | {neg.shape[0]} globally-held-out negatives "
          f"(disjoint from ALL sets) | N={2*N_PER_CLASS}")
    bank, nc, nt = [], 0, 0
    for sid, (x_ft, y_ft) in enumerate(sets):
        x_ft = x_ft.to(dev); y_ft = y_ft.to(dev); x0 = x_ft - ds_mean
        for init in INITS:
            A, B, mbce, _dW = train_adapter(frozen, b0, draw_B0(init, out_f, RANK, dev), x0, y_ft, LR, T, act, RANK)
            conv = mbce < 1e-2; nt += 1; nc += int(conv)
            bank.append(dict(A=A[0].detach().cpu(), B=B[0].detach().cpu(), set_id=sid, init=init,
                             max_bce=mbce, converged=conv,
                             priv_imgs=x_ft.reshape(x_ft.shape[0], -1).detach().cpu()))
        print(f"  set {sid}: done")
    print(f"[mia-zoo] converged {nc}/{nt}")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(bank=bank, neg_pool=neg, mu=mu, ds_mean=ds_mean.reshape(-1).detach().cpu(),
                        meta=dict(n_sets=N_SETS, inits=INITS, N=2 * N_PER_CLASS, neg=neg.shape[0], digits=DIGITS)),
                   os.path.join(RESULTS, "mia_bank.pth"))
        print(f"[saved] {RESULTS}/mia_bank.pth ({len(bank)} adapters, {neg.shape[0]} global negatives)")


if __name__ == "__main__":
    main()
