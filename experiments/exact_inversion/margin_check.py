#!/usr/bin/env python3
"""Does the release record an image in proportion to the model's RESIDUAL on it?  (mechanism check, no solve)

Step 18's spectra: at k=16, below the line, the strong (98%) backbone's release has rank B_T = 5-6 of N=8 with
repeated labels and sigma_N/sigma_1(B_T) = 4e-9 with distinct ones; the truth Jacobian loses rank.  Hypothesis:
B_T = P_T X^T and column i of P_T is the accumulated softmax residual p_t(x_i) - e_{y_i} along the trajectory;
a strong model already classifies an easy private digit with margin M, so its residual is ~e^{-M} from step 0
and the digit is recorded at that scale -- below the FP64 floor when M ~ 30.  A model fine-tuned on examples it
already fits leaves no fingerprint of them.  If true, per image: ||P_T[:, i]|| tracks ||p_0(x_i) - e_{y_i}||
(residual at W0) across encoders and digits, and the rank loss sits on the largest-margin digits.

Records per (encoder, label draw, image): margin at W0, residual norm at W0, ||P_T[:, i]|| (P_T recovered as
B_T X (X^T X)^{-1}, exact when rank X = N), and the ratio.  Same digits, recipe and A0 draw as the cells.

  python -m experiments.exact_inversion.margin_check --encoder weak=... --encoder strong=... --labels repeated distinct
"""
import argparse, json, math, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import train_release, qr_canon, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.random_encoder_control import RandomBackbone
from experiments.exact_inversion.truth_spectrum import pick_digits

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", action="append", default=[])
    ap.add_argument("--norm-model", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--encoder-seed", type=int, default=101)
    ap.add_argument("--labels", nargs="*", default=["repeated", "distinct"])
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--a0-seed", type=int, default=1)
    ap.add_argument("--on-chart", action="store_true", help="evaluate on the PCA projection of the digits (cell a) instead of the raw digits")
    ap.add_argument("--pick", choices=["draw", "confident", "hard1_same", "hard1_diff"], default="draw",
                    help="batch composition, margins taken under --pick-model over the test split: "
                         "draw = the cells' random draw (--labels); confident = the N largest-margin digits of N "
                         "distinct classes (prediction: every column < 1e-6, rank loss outright); "
                         "hard1_same = the most-misclassified digit + the N-1 largest-margin digits of ITS class; "
                         "hard1_diff = the same hard digit + the largest-margin digit of N-1 other classes "
                         "(prediction: the Gram coupling lifts the same-class batch, not the different-class one)")
    ap.add_argument("--pick-model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, ytr = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    norm_bb = TrainedBackbone(a.norm_model, dev, "gelu")
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    chart = PCAChart(Xtr_t, a.k, dev) if a.on_chart else None
    picked = None
    if a.pick != "draw":
        ref = TrainedBackbone(a.pick_model, dev, "gelu")
        with torch.no_grad():
            z = ref.logits(Xte_t.T); zy = z[yte_t, torch.arange(z.shape[1], device=dev)]
            zo = z.clone(); zo[yte_t, torch.arange(z.shape[1], device=dev)] = -float("inf")
            mar = (zy - zo.max(0).values).cpu()
        def top_of(c, n, exclude=()):
            cand = [(float(mar[i]), i) for i in range(len(mar)) if int(yte[i]) == c and i not in exclude]
            return [i for _, i in sorted(cand, reverse=True)[:n]]
        if a.pick == "confident":
            picked = [top_of(c, 1)[0] for c in sorted(range(10), key=lambda c: -float(top_of(c, 1) and mar[top_of(c, 1)[0]]))[:a.N]]
        else:
            hard = int(torch.argmin(mar)); c = int(yte[hard])
            if a.pick == "hard1_same":
                picked = [hard] + top_of(c, a.N - 1, exclude={hard})
            else:
                others = [k for k in range(10) if k != c]
                others = sorted(others, key=lambda k: -float(mar[top_of(k, 1)[0]]))[:a.N - 1]
                picked = [hard] + [top_of(k, 1)[0] for k in others]
        print(f"# pick={a.pick} under {a.pick_model}: idx={picked} y={[int(yte[i]) for i in picked]} "
              f"margins={[round(float(mar[i]), 2) for i in picked]}", flush=True)
        a.labels = [a.pick]
    print(f"# margin check  encoders={a.encoder}  labels={a.labels}  on_chart={a.on_chart}  git={git_hash()}", flush=True)
    for spec in a.encoder:
        label, path = spec.split("=", 1)
        bb = RandomBackbone(norm_bb, dev, a.encoder_seed) if path == "random" else TrainedBackbone(path, dev, "gelu")
        sigma0 = a.sigma0 if a.sigma0 is not None else 1.0 / math.sqrt(bb.n)
        for lmode in a.labels:
            idx = torch.tensor(picked if picked is not None else pick_digits(perm, yte, a.N, lmode), device=dev)
            X = Xte_t[idx].T.contiguous(); y = yte_t[idx]
            if chart is not None: X = chart.psi(chart.coords_of(X))
            H = bb.phi(X); z = bb.W0 @ H                                   # logits at W0 (B0 = 0)
            p = torch.softmax(z, dim=0); E = torch.eye(bb.m, device=dev)[:, y]
            res0 = torch.linalg.norm(p - E, dim=0)
            zy = z[y, torch.arange(a.N)]; zo = z.clone(); zo[y, torch.arange(a.N)] = -float("inf")
            margin = zy - zo.max(0).values
            A0 = (sigma0 * torch.randn(a.r, bb.n, generator=torch.Generator().manual_seed(1000 + a.a0_seed))).to(dev)
            A_T, B_T = train_release(H, A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
            U, _ = qr_canon(H); Xs = A0 @ U                                   # r x N
            P_T = B_T @ Xs @ torch.linalg.inv(Xs.T @ Xs)
            colP = torch.linalg.norm(P_T, dim=0); sB = torch.linalg.svdvals(B_T)
            for i in range(a.N):
                row = dict(encoder=label, labels=lmode, pick=a.pick, image=i, y=int(y[i]), margin_W0=float(margin[i]),
                           residual_W0=float(res0[i]), P_T_col_norm=float(colP[i]),
                           P_T_over_res0=float(colP[i] / res0[i]) if float(res0[i]) > 0 else None,
                           rank_B_T=int((sB > 1e-12 * sB[0]).sum()), B_T_sigma_ratio=float(sB[a.N - 1] / sB[0]),
                           on_chart=a.on_chart, k=a.k, N=a.N, r=a.r, m=bb.m, n=bb.n, T=a.T, lr=a.lr, seed=a.seed, a0_seed=a.a0_seed,
                           git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                print(json.dumps(row), flush=True)
                if a.out:
                    with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
