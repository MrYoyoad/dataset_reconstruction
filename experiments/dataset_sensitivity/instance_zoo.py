"""Instance-level atlas — the STRINGENT same-digits test (the OPEN question the atlas left).

Atlas showed ΔW recovers composition at ≥CONTENT-level (compositions differed in DIGIT content). This zoo
fixes the digit pair to {0,1} and makes each COMPOSITION a DIFFERENT IMAGE SAMPLE of those SAME digits
(distinct data seed), × K init seeds. Question: can the ΔW clustering recover WHICH image-instance beyond
the recipe (INSTANCE-level leakage), or only content? Saves atlas_analyze-compatible cells → Facet-C on
composition=image-sample-id gives the answer with a cluster-robust CI. bsub-only, float64, gelu.
"""
import argparse, os, torch
from experiments.jacobian_spectrum import _honest_target, make_activation
from experiments.dataset_sensitivity.arm_b_dilution import train_adapter, draw_B0, build_set
from experiments.dataset_sensitivity.eco_zoo import digit_pair_data
from experiments.data_utils import _load_dataset

torch.set_default_dtype(torch.float64)
RESULTS = "/home/projects/galvardi/yoado/results/instance_zoo"
DIGITS = (0, 1)
COMPOSITIONS = list(range(8))       # 8 DISTINCT image-samples (data seeds) of the SAME pair {0,1}
INITS = list(range(300, 308))       # 8 init seeds (the recipe nuisance)
N_PER_CLASS, T, RANK, LR, ACT = 2, 200, 8, 0.5, "gelu"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ds = _load_dataset("mnist", train=True); act = make_activation(ACT)
    xr, yr, _ = build_set(N_PER_CLASS, seed=42, device=dev, dataset="mnist")
    _, frozen, b0, _b, ds_mean = _honest_target(xr, yr, T, RANK, ACT, LR, dev, "mnist", num_classes=2)
    out_f = frozen[0].shape[0]
    print(f"[instance-zoo] digits={DIGITS} | {len(COMPOSITIONS)} image-samples × {len(INITS)} inits | N={2*N_PER_CLASS}")
    bank, nc, nt = [], 0, 0
    for comp in COMPOSITIONS:
        x_ft, y_ft = digit_pair_data(DIGITS, N_PER_CLASS, data_seed=1000 + comp, ds=ds, device=dev)  # distinct images
        x0 = x_ft - ds_mean
        for init in INITS:
            A, B, mbce, _dW = train_adapter(frozen, b0, draw_B0(init, out_f, RANK, dev), x0, y_ft, LR, T, act, RANK)
            conv = mbce < 1e-2; nt += 1; nc += int(conv)
            bank.append(dict(A=A[0].detach().cpu(), B=B[0].detach().cpu(), activation=ACT, composition=comp,
                             digits=DIGITS, lr=LR, init_seed=init, max_bce=mbce, converged=conv))
        print(f"  image-sample {comp}: done ({len(INITS)} inits)")
    print(f"[instance-zoo] converged {nc}/{nt}")
    if args.save:
        os.makedirs(RESULTS, exist_ok=True)
        torch.save(dict(bank=bank, meta=dict(acts=[ACT], comps=COMPOSITIONS, lrs=[LR], inits=INITS,
                                             N=2 * N_PER_CLASS, T=T, rank=RANK, dataset="mnist",
                                             digits=DIGITS, n_converged=nc, n_total=nt)),
                   os.path.join(RESULTS, "instance_bank.pth"))
        print(f"[saved] {RESULTS}/instance_bank.pth ({len(bank)} cells)")


if __name__ == "__main__":
    main()
