"""B2 (build) — INSTANCE-level recipe-invariance zoo. The harder half of B.

Fixed {0,1}, M image-samples × {gelu, relu, softplus_b1} activations × inits. Different activations use
DIFFERENT frozen bases (weights-mnist_<act>.pth) → different ΔW geometry. Downstream: can we match a target to
its exact IMAGE-SAMPLE using reference adapters of a DIFFERENT activation? (does the INSTANCE fingerprint —
not just content — survive a recipe change?). Saves atlas-compatible cells (composition=sample). bsub, float64.
"""
import argparse, os, torch
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.dataset_sensitivity.eco_zoo import digit_pair_data
from experiments.data_utils import _load_dataset

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/b2_instance_recipe_zoo"
DIGITS = (0, 1)
SAMPLES = list(range(8))                       # 8 image-samples of {0,1}
ACTS = ["gelu", "relu", "softplus_b1"]
INITS = [300, 301]
N_PER_CLASS, T, RANK, LR = 2, 200, 8, 0.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True)
    print(f"[B2-zoo] {len(SAMPLES)} samples × {len(ACTS)} acts × {len(INITS)} inits | {DIGITS} N={2*N_PER_CLASS}")
    bank, nc, nt = [], 0, 0
    for act_name in ACTS:
        act = make_activation(act_name)
        xr, yr, _ = build_set(N_PER_CLASS, seed=42, device=dev, dataset="mnist")
        _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, act_name, LR, dev, "mnist", num_classes=2)
        out_f = frozen[0].shape[0]
        for s in SAMPLES:
            x_ft, y_ft = digit_pair_data(DIGITS, N_PER_CLASS, data_seed=1000 + s, ds=ds, device=dev)
            x0 = x_ft - ds_mean
            for init in INITS:
                A, B, mbce, _dW = train_adapter(frozen, b0, draw_B0(init, out_f, RANK, dev), x0, y_ft, LR, T, act, RANK)
                conv = mbce < 1e-2; nt += 1; nc += int(conv)
                bank.append(dict(A=A[0].detach().cpu(), B=B[0].detach().cpu(), activation=act_name,
                                 composition=s, digits=DIGITS, lr=LR, init_seed=init, max_bce=mbce, converged=conv))
        print(f"  activation={act_name}: done ({len(SAMPLES)}×{len(INITS)})")
    print(f"[B2-zoo] converged {nc}/{nt}")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(bank=bank, meta=dict(acts=ACTS, comps=SAMPLES, lrs=[LR], inits=INITS,
                                             N=2 * N_PER_CLASS, T=T, rank=RANK, digits=DIGITS)),
                   os.path.join(RESULTS, "b2_bank.pth"))
        print(f"[saved] {RESULTS}/b2_bank.pth ({len(bank)} cells)")


if __name__ == "__main__":
    main()
