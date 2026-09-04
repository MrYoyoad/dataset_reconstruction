"""One certificate PER ADAPTED LAYER (Yoad, §9 of the assumption-relaxation programme).

Every adapted layer has its own released pair, hence its own certificate `C_l = P_{row(B_T^l)^perp} A_T^l`, testing
that layer's own inputs `h^l`. The derivation is exact only at the FIRST adapted layer, whose inputs are frozen:
deeper, `h^l` moves during training because the adapters below it move, so `C_l h^l = 0` cannot hold exactly. The
conjecture is that it holds APPROXIMATELY, to the order of that drift, and that in the ordinary LoRA regime the
drift is small (the headline cell's `A_T` moves 9.3% from `A_0`).

Cheap by construction: projections only, no unrolled solve.

Pre-registered (yoado-cd, before any row):
  * layer 1 (frozen inputs): `||C_1 h_i||/||A_T^1 h_i||` at machine precision for recorded images -- this is the
    localisation result of section 3 and the run cannot lose it;
  * layers >= 2: the residual **scales with the lower adapters' drift** and stays orders below the non-member level
    (0.1 ... 1) while the drift is small. Sweeping `lr` (and `T`) moves the drift, and the residual must track it.
    If the deep-layer residual is already at non-member level at ordinary `lr`, the idea is dead and only the first
    adapted layer is usable.
  * per layer, whether the cap `N' <= m-1` binds: at a hidden layer the softmax zero-sum is gone, so it should not.
    Requires `N >= m` for the test to be non-vacuous (yoado-b9), so the batch is sized accordingly.
Reported per layer: drift of every LOWER adapter, `rank B_T^l`, the recorded/non-member separation, and the imprint
spectrum beside the count (cells are comparable by spectrum, not by `N'`).
"""
import argparse, json, math, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import PCAChart, read_idx
from experiments.exact_inversion.certificate import certificate
from experiments.exact_inversion.multilayer_lora import forward_adapted, run_training, GELU

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--N", type=int, default=12, help=">= m so the output cap can bind (else the deep-layer test is vacuous)")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--lrs", nargs="*", type=float, default=[0.002, 0.01, 0.05, 0.2], help="sweeps the DRIFT")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    keys = sorted(k for k in sd if k.endswith("weight"))          # 784-1000-1000-10: three weight matrices
    Ws = [sd[k].to(dev).double() for k in keys]
    b1 = sd[[k for k in sd if k.endswith("bias")][0]].to(dev).double()
    m = Ws[-1].shape[0]
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(Ws[0].shape[1])
    g = torch.Generator().manual_seed(a.seed + 7)
    perm_all = torch.randperm(Xte_t.shape[0], generator=g)
    idx = perm_all[:a.N].to(dev); nonmem = perm_all[a.N:a.N + 12].to(dev)      # held-out non-members
    chart = PCAChart(Xtr_t, a.k, dev)
    X_real = Xte_t[idx].T.contiguous(); X_on = chart.psi(chart.coords_of(X_real)); y = yte_t[idx]
    print(f"# per_layer_certificate  m={m} N={a.N} r={a.r} k={a.k} lrs={a.lrs} git={git_hash()} dev={dev}", flush=True)

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    for lr in a.lrs:
        gA = torch.Generator().manual_seed(a.seed + 11)
        A0s = [(a.sigma0 * torch.randn(a.r, W.shape[1], generator=gA)).to(dev) for W in Ws]
        As, Bs = run_training(X_on, Ws, b1, A0s, y, m, a.T, lr)
        # the inputs each adapted layer actually saw at the END of training, and at the START (frozen model)
        def inputs_at(As_, Bs_):
            h0 = X_on
            h1 = GELU(Ws[0] @ h0 + b1[:, None] + Bs_[0] @ (As_[0] @ h0))
            h2 = GELU(Ws[1] @ h1 + Bs_[1] @ (As_[1] @ h1))
            return [h0, h1, h2]
        H_end = inputs_at(As, Bs)
        H_start = inputs_at(A0s, [torch.zeros(W.shape[0], a.r, device=dev) for W in Ws])
        for l in range(3):
            drift_below = [float(torch.linalg.norm(As[j] - A0s[j]) / torch.linalg.norm(A0s[j])) for j in range(l)]
            input_drift = float(torch.linalg.norm(H_end[l] - H_start[l]) / torch.linalg.norm(H_start[l]))
            C, Np, sB = certificate(As[l], Bs[l])
            res = torch.linalg.norm(C @ H_end[l], dim=0) / torch.linalg.norm(As[l] @ H_end[l], dim=0)
            # NON-MEMBER control, in the same row: held-out images pushed to this layer's inputs. A certificate that
            # annihilates non-members too is VACUOUS -- which happens exactly when rank(B_T) = r, since row(B_T) is
            # then everything and C = 0.
            Hn0 = chart.psi(chart.coords_of(Xte_t[nonmem].T.contiguous()))
            hn = Hn0
            for j in range(l):
                hn = GELU(Ws[j] @ hn + (b1[:, None] if j == 0 else 0) + Bs[j] @ (As[j] @ hn))
            res_non = torch.linalg.norm(C @ hn, dim=0) / torch.linalg.norm(As[l] @ hn, dim=0)
            C_norm = float(torch.linalg.norm(C)); A_norm = float(torch.linalg.norm(As[l]))
            vacuous = bool(Np >= a.r or C_norm <= 1e-12 * A_norm or float(res_non.median()) < 1e-6)
            width = Ws[l].shape[0] if l < 2 else m
            emit(dict(part="LAYER", layer=l + 1, exact=(l == 0), lr=lr, T=a.T, N=a.N, r=a.r, k=a.k, m=m,
                      layer_width=width, cap_is=("m-1" if l == 2 else "layer width"),
                      cap_value=(m - 1 if l == 2 else width), rank_B_T=Np, cap_binds=bool(Np >= (m - 1 if l == 2 else width)),
                      lower_adapter_drift=drift_below, input_drift=input_drift,
                      cert_residual_median=float(res.median()), cert_residual_max=float(res.max()),
                      nonmember_residual_median=float(res_non.median()), nonmember_residual_min=float(res_non.min()),
                      separation_orders=float(torch.log10(res_non.median() / res.median())) if float(res.median()) > 0 else float("inf"),
                      C_norm_over_A_norm=C_norm / A_norm, rank_C=int(torch.linalg.matrix_rank(C, rtol=1e-10)),
                      VACUOUS=vacuous, vacuity_reason=("rank(B_T) = r: row(B_T) is everything, C = 0" if Np >= a.r else
                                                       ("C is numerically zero" if C_norm <= 1e-12 * A_norm else
                                                        ("non-members annihilated too" if vacuous else ""))),
                      cert_residual_per_image=[float(v) for v in res],
                      B_T_sigma_rel=[float(v / sB[0]) for v in sB[:min(12, len(sB))]] if float(sB[0]) > 0 else None,
                      start_model="n/a (projection only, no solve)", claim_class="algebraic check",
                      git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
            print(f"  lr={lr} layer {l+1}{' (exact)' if l==0 else ''}: input drift {input_drift:.2e}, "
                  f"cert residual median {float(res.median()):.2e} max {float(res.max()):.2e}, "
                  f"rank {Np} against cap {(m-1) if l==2 else width}", flush=True)


if __name__ == "__main__":
    main()
