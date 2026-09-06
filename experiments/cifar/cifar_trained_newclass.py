#!/usr/bin/env python3
"""Certificate-only random-start search on the repo's TRAINED CIFAR-10 backbone with a NEW class added by head LoRA.

The user's replica (cifar_certificate.py) uses a 15-epoch MLP it trains itself. This runs the same attack on the
repo's trained CIFAR-10 backbone (models/exact_inversion/cifar10_mlp.pth, 3072-1000-1000-10 GELU, the model of
RESULTS Step 21 "flowers as an 11th class") with the recipe of the MNIST letters headline (Step 25): head LoRA on the
1000-dim penultimate features, ExtendedHead with a zero 11th row, r = 64, T = 400, lr = 0.01, PCA chart fitted on the
PUBLIC train split of the new class, private images from the TEST split, on-chart privates, certificate tol 1e-12,
LM search from public-coordinate-scale random starts (record_strength.run_search == certificate.py Part B).

  --new flowers   CIFAR-100 flower superclass (Step 21's class; 2,500 public train images, privates from test)
  --new apple     CIFAR-100 fine class 0 (the replica's class; 500 public train images, privates from test)

  python -u -m experiments.cifar.cifar_trained_newclass --new flowers --ks 32 48 --out results/record_strength/cifar_trained.jsonl
"""
import argparse, json, math, os, pickle, socket, sys, time
import numpy as np, torch

from experiments.exact_inversion.lora_exact_inversion import train_release, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart
from experiments.exact_inversion.train_cifar_backbone import load_cifar10
from experiments.exact_inversion.new_class import load_cifar100_flowers, ExtendedHead
from experiments.exact_inversion.certificate import certificate
from experiments.record_strength.record_strength import run_search, rel

torch.set_default_dtype(torch.float64)


def load_cifar100_fine(root, fine):
    out = {}
    for split in ("train", "test"):
        b = pickle.load(open(os.path.join(root, "cifar-100-python", split), "rb"), encoding="bytes")
        lab = np.array(b[b"fine_labels"]); X = b[b"data"].astype(np.float64) / 255.0
        out[split] = (X[lab == fine], lab[lab == fine])
    return out


def ssim32(a, b):
    import kornia.metrics as km
    a = a.reshape(1, 3, 32, 32).clamp(0, 1).float(); b = b.reshape(1, 3, 32, 32).clamp(0, 1).float()
    return float(km.ssim(a, b, window_size=3).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", choices=["flowers", "apple"], default="flowers")
    ap.add_argument("--model", default="models/exact_inversion/cifar10_mlp.pth")
    ap.add_argument("--ks", nargs="*", type=int, default=[32, 48]); ap.add_argument("--settings", nargs="*", default=["on", "raw"])
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--starts", type=int, default=500); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--data-root", default="dataset_reconstruction/data"); ap.add_argument("--cifar100-root", default="data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default="results/record_strength"); ap.add_argument("--fig-dir", default="figures/record_strength")
    a = ap.parse_args(); dev = torch.device(a.device); os.makedirs(a.save_dir, exist_ok=True); os.makedirs(a.fig_dir, exist_ok=True)
    log = lambda s: print(s, flush=True)
    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    Xtr, ytr, Xte, yte = load_cifar10(a.data_root)
    fl = load_cifar100_flowers(a.cifar100_root) if a.new == "flowers" else load_cifar100_fine(a.cifar100_root, 0)
    Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    Ftr_t = torch.tensor(fl["train"][0], device=dev); Fte_t = torch.tensor(fl["test"][0], device=dev)
    base = TrainedBackbone(a.model, dev, "gelu")
    with torch.no_grad(): acc = float((base.logits(Xte_t[:2000].T).argmax(0) == yte_t[:2000]).double().mean())
    sigma0 = 1.0 / math.sqrt(base.n)
    g = torch.Generator().manual_seed(a.seed + 7); pf = torch.randperm(Fte_t.shape[0], generator=g)[:a.N]
    X_real = Fte_t[pf].T.contiguous(); y = torch.full((a.N,), 10, device=dev); bb = ExtendedHead(base, "zero", a.seed)
    log(f"# cifar_trained_newclass  backbone {a.model} acc {acc*100:.1f}%  new={a.new} (public train {Ftr_t.shape[0]}, privates from test)  m={bb.m} n={bb.n} r={a.r}  git={git_hash()} host={socket.gethostname()}")
    with torch.no_grad(): pred0 = bb.logits(X_real).argmax(0).tolist()
    log(f"# base model's reading of the privates (class 10 = new): {pred0}")
    Xp_all = Ftr_t                                                                           # public same-class pool (train split; privates are test)
    for k in a.ks:
        chart = PCAChart(Ftr_t, k, dev); coord_std = chart.coords_of(Ftr_t[:10000].T).std(dim=1, keepdim=True)
        W_all = chart.coords_of(X_real); X_on = chart.psi(W_all)
        for setting in a.settings:
            X_train = X_on if setting == "on" else X_real
            g = torch.Generator().manual_seed(a.seed + 7); A0 = (sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
            H = bb.phi(X_train); A_T, B_T = train_release(H, A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
            sB = torch.linalg.svdvals(B_T); C, Np, _ = certificate(A_T, B_T, a.tol)
            with torch.no_grad():
                res_truth = torch.linalg.norm(C @ H, dim=0) / torch.linalg.norm(A_T @ H, dim=0)
                H_on = bb.phi(X_on); res_on = torch.linalg.norm(C @ H_on, dim=0) / torch.linalg.norm(A_T @ H_on, dim=0)
                Hp = bb.phi(Xp_all[:200].T); res_pub = torch.linalg.norm(C @ Hp, dim=0) / torch.linalg.norm(A_T @ Hp, dim=0)
                Q, _ = torch.linalg.qr(H)
            recorded = list(range(a.N)) if Np >= a.N else [i for i in range(a.N) if float(res_truth[i]) < 1e-3]
            gap = float(sB[Np - 1] / sB[Np]) if Np < len(sB) else float("inf")
            log(f"# k={k} {setting}: rank B_T (tol {a.tol:g}) = {Np}, rank C = {int(torch.linalg.matrix_rank(C, rtol=1e-10))}, gap {gap:.1e}, ||CH||/||A_T H|| at truth {[f'{v:.1e}' for v in res_truth.tolist()]}, at chart projections {[f'{v:.1e}' for v in res_on.tolist()]}, public median {res_pub.median():.2e} min {res_pub.min():.2e}")
            # landings are scored against the chart projections X_on (the target the chart can express; == the truth on-chart)
            runs, W, sec = run_search(bb, chart, k, C, A_T, X_on, X_real, recorded, coord_std, Ftr_t, a.starts, a.iters, a.seed, dev, log)
            X_found = chart.psi(W.T.to(dev))                                                    # (3072, starts)
            with torch.no_grad(): blend = ((Q.T @ bb.phi(X_found)).norm(dim=0) ** 2 / bb.phi(X_found).norm(dim=0) ** 2)
            per = []; grid_found = []
            nearest_pub = Xp_all[torch.cdist(X_real.T, Xp_all).argmin(1)].T                      # nearest public image to each private
            for i in range(a.N):
                mine = [(s, r_) for s, r_ in enumerate(runs) if r_["nearest_all"] == i]
                land = sum(1 for r_ in runs if r_["landed_rec"] and r_["nearest_rec"] == i) if i in recorded else 0
                if mine:
                    s_best, r_best = min(mine, key=lambda sr: sr[1]["err_all"]); xb = X_found[:, s_best]
                    per.append(dict(i=i, landings=land, best_err_chart=r_best["err_all"], best_err_real=rel(xb, X_real[:, i]), best_objective=r_best["objective"],
                                    ssim_vs_chart=ssim32(xb, X_on[:, i]), ssim_vs_raw=ssim32(xb, X_real[:, i]), blend_best=float(blend[s_best]),
                                    control_ssim_max=max(ssim32(X_found[:, s], nearest_pub[:, i]) for s in range(0, len(runs), max(1, len(runs) // 100)))))
                else:
                    per.append(dict(i=i, landings=land, best_err_chart=None, best_err_real=None, best_objective=None, ssim_vs_chart=None, ssim_vs_raw=None, blend_best=None, control_ssim_max=None)); s_best = None
                per[-1].update(chart_floor_err=rel(X_on[:, i], X_real[:, i]), chart_floor_ssim=ssim32(X_on[:, i], X_real[:, i]), res_truth=float(res_truth[i]), res_projection=float(res_on[i]))
                grid_found.append(X_found[:, s_best].cpu() if s_best is not None else torch.zeros(3072))
            best = min((r_ for r_ in runs if not r_["degenerate"]), key=lambda r_: r_["objective"], default=None)
            row = dict(part="cifar_trained", new=a.new, backbone=a.model, backbone_acc=acc, k=k, setting=setting, N=a.N, r=a.r, m=bb.m, n=bb.n, T=a.T, lr=a.lr, seed=a.seed, tol=a.tol,
                       n_prime=Np, cert_line=a.r - Np, gap=gap, pred_at_W0=pred0, recorded=recorded, starts=len(runs),
                       landed=sum(r_["landed_rec"] for r_ in runs), landings_per_image=[p_["landings"] for p_ in per], images_found=sum(1 for p_ in per if p_["landings"] > 0),
                       argmin_objective=(best["objective"] if best else None), argmin_landed=(best["landed_rec"] if best else None),
                       objective_median=float(torch.tensor([r_["objective"] for r_ in runs]).median()), res_public_median=float(res_pub.median()), res_public_min=float(res_pub.min()),
                       blend_found_median=float(blend.median()), per_image=per, n_degenerate=sum(r_["degenerate"] for r_ in runs), sec=sec, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            emit(row)
            log(f"# RESULT k={k} {setting}: landed {row['landed']}/{len(runs)}, per image {row['landings_per_image']}, found {row['images_found']}/{a.N}; SSIM vs raw {[None if p_['ssim_vs_raw'] is None else round(p_['ssim_vs_raw'], 2) for p_ in per]} floor {[round(p_['chart_floor_ssim'], 2) for p_ in per]} control {[None if p_['control_ssim_max'] is None else round(p_['control_ssim_max'], 2) for p_ in per]}")
            torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_found_best=torch.stack(grid_found, 1), A_T=A_T.cpu(), B_T=B_T.cpu(), A0=A0.cpu(), C=C.cpu(), W=W, runs=runs, meta=row),
                       os.path.join(a.save_dir, f"cifar_trained_{a.new}_k{k}_{setting}.pth"))
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            rows = [("private (raw, test split)", X_real.cpu()), (f"chart projection (PCA k={k}, public train split)", X_on.cpu()), ("closest random start (certificate only)", torch.stack(grid_found, 1))]
            fig, ax = plt.subplots(3, a.N, figsize=(1.3 * a.N, 4.6))
            for ri, (name, imgs) in enumerate(rows):
                for j in range(a.N):
                    ax[ri, j].imshow(imgs[:, j].reshape(3, 32, 32).permute(1, 2, 0).clamp(0, 1).float().numpy()); ax[ri, j].axis("off")
                    if ri == 2: ax[ri, j].set_title(f"{'landed x%d' % per[j]['landings'] if per[j]['landings'] else 'err %.2f' % (per[j]['best_err_chart'] or float('nan'))}", fontsize=6)
                ax[ri, 0].text(-0.1, 1.15, name, fontsize=7, transform=ax[ri, 0].transAxes)
            fig.suptitle(f"{a.new} as class 11 on the trained CIFAR-10 MLP ({acc*100:.0f}%), head LoRA r={a.r}, k={k}, {setting}: {row['landed']}/{len(runs)} landed, {row['images_found']}/{a.N} found", fontsize=8)
            plt.tight_layout(); fig.savefig(os.path.join(a.fig_dir, f"cifar_trained_{a.new}_k{k}_{setting}.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
