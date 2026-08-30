"""Ecosystem graded-overlap — the 100%-CONTENT-OVERLAP end point (all tasks share BOTH digits {0,1}).

Completes the graded curve (0% disjoint / 50% anchor-digit / 100% same-digits). Here all 5 "tasks" are the
SAME digit pair {0,1} but DIFFERENT image samples → the shared subspace = the whole {0,1} structure; the
target's UNIQUE part is only its specific images. Prediction (the bracket): GAIN → 0 at this end (shared =
everything relevant, atlas-degenerate). LOO groups by gid (distinct per sample); digits=(0,1) so the overlap
metric reads 100%. eco_analyze/graded_overlap-compatible. bsub, float64, gelu.
"""
import argparse, os, torch
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.dataset_sensitivity.eco_zoo import digit_pair_data
from experiments.data_utils import _load_dataset

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/full_zoo"
DIGITS = (0, 1)
SAMPLES = list(range(5))            # 5 DIFFERENT image-samples of the SAME {0,1}
SEEDS = list(range(500, 508))
N_PER_CLASS, T, RANK, LR, ACT = 2, 200, 8, 0.5, "gelu"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(N_PER_CLASS, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    print(f"[full-zoo] {len(SAMPLES)} image-samples of {DIGITS} × {len(SEEDS)} seeds (100% content overlap)")
    bank, nc, nt = [], 0, 0
    for s in SAMPLES:
        x_ft, y_ft = digit_pair_data(DIGITS, N_PER_CLASS, data_seed=2000 + s, ds=ds, device=dev)
        x0 = x_ft - ds_mean
        for seed in SEEDS:
            A, B, mbce, _dW = train_adapter(frozen, b0, draw_B0(seed, out_f, RANK, dev), x0, y_ft, LR, T, act, RANK)
            conv = mbce < 1e-2; nt += 1; nc += int(conv)
            bank.append(dict(A=A[0].detach().cpu(), B=B[0].detach().cpu(), task=DIGITS, gid=s, digits=DIGITS,
                             seed=seed, max_bce=mbce, converged=conv,
                             priv_imgs=x_ft.reshape(x_ft.shape[0], -1).detach().cpu()))
        print(f"  sample {s}: done ({len(SEEDS)} seeds)")
    print(f"[full-zoo] converged {nc}/{nt}")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(bank=bank, ds_mean=ds_mean.reshape(-1).detach().cpu(),
                        meta=dict(samples=SAMPLES, seeds=SEEDS, digits=DIGITS, N=2 * N_PER_CLASS, T=T, rank=RANK)),
                   os.path.join(RESULTS, "full_bank.pth"))
        print(f"[saved] {RESULTS}/full_bank.pth ({len(bank)} adapters)")


if __name__ == "__main__":
    main()
