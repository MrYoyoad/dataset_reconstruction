#!/usr/bin/env python3
"""Can the attacker recover the MOST-LEAKING example from the release alone?

On a strong model fine-tuned on ordinary data, ONE example carries the release (Step 19: relative imprints 1 vs
<= 2e-5 on the 98% model).  So the attacker's problem is one image, not N: k + r unknowns.  Procedure, using
nothing from the truth:
  (1) spectrum of B_T: sigma_2/sigma_1 below --tau -> treat the release as one image (N' = 1);
  (2) for each label c in 0..9 and each of R RANDOM chart-coordinate starts (scale = the public PCA coordinate
      std over the train split), solve the one-image inversion; the seed block is initialised from the
      candidate's own features (the attacker-available 'span' start);
  (3) take the (label, start) with the smallest residual; polish the best few with a larger budget.
Evaluation only (never in the objective): which of the N private images the answer is nearest to, its error vs
that image's chart projection and vs the raw digit, the predicted floor ||sum_{j != top} C_j||^2/||B_T||^2 that
the other images impose, sigma_min of the one-image residual at the truth, and a NEAR-TRUTH start as control.

  python -m experiments.exact_inversion.most_leaking --encoder strong --sets mnist_control hard1_diff
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import simulate_sgd_reduced, qr_canon, invert_lm, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch, release_and_imprints, margins_of, floor_pred
from experiments.exact_inversion.random_encoder_control import RandomBackbone

torch.set_default_dtype(torch.float64)


def solve_one(chart, bb, A_T, B_T, y1, W_init, a, budget, restarts):
    """One-image inversion of the full release from an explicit start.  Returns the solution and the residual.
       NB invert_lm's `restarts` is the number of ATTEMPTS (the first from W_init, later ones re-seeded): 1 = one
       attempt from the given start, which is what an attacker-fair search and polish both need."""
    with torch.no_grad():
        Uc, _ = qr_canon(bb.phi(chart.psi(W_init))); Xinit = A_T @ Uc          # 'span' seed start, attacker-available

    class Adapter:
        psi = staticmethod(chart.psi)
        features_from_latents = staticmethod(lambda Wc: bb.phi(chart.psi(Wc)))
    args = argparse.Namespace(m=bb.m, T=a.T, lr=a.lr, wd=0.0, release="sgd", seed=a.seed, restarts=restarts,
                              restart_noise=0.5, lm_iters=budget, lm_lambda=1e-2, lm_scale="identity",
                              stage_x=0, jac="fwd", solver="lm", outer=30, lbfgs_iter=20)
    W_hat, aux, resid, sec, nrs, diag = invert_lm(Adapter, A_T, B_T, bb.W0, y1, args, W_init, Xinit, lambda s: None)
    return W_hat, aux, float(resid), diag.get("lm_iters_used")


def one_image_truth_spectrum(chart, bb, A_T, B_T, x_true, y1, A0, a):
    """sigma_min of the one-image residual at (W_true, A0 U_true) and the residual there (= the floor realised)."""
    W_true = chart.coords_of(x_true); X_on = chart.psi(W_true); H1 = bb.phi(X_on); U1, _ = qr_canon(H1)
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T); k = a.k

    def res_vec(v):
        Wc = v[:k].reshape(k, 1); aux = v[k:].reshape(a.r, 1)
        Bs, Xis, Uc = simulate_sgd_reduced(bb.phi(chart.psi(Wc)), aux, bb.W0, y1, bb.m, a.T, a.lr, 0.0)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])
    v0 = torch.cat([W_true.reshape(-1), (A0 @ U1).reshape(-1)]).detach()
    sv = torch.linalg.svdvals(tf.jacfwd(res_vec)(v0).detach()); rv = res_vec(v0)
    return dict(jac_sigma_min_truth=float(sv[-1]), jac_sigma_max_truth=float(sv[0]), jac_cond_truth=float(sv[0] / sv[-1]),
                jac_rank_truth=int((sv > 1e-12 * sv[0]).sum()), jac_cols=int(sv.numel()),
                res_at_truth=float(rv.norm() ** 2), res_at_truth_B=float(rv[:bb.m * a.r].norm() ** 2)), W_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["weak", "mid", "strong", "random"], default="strong")
    ap.add_argument("--encoder-seed", type=int, default=101)
    ap.add_argument("--strong", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--mid", default="models/exact_inversion/mnist_mlp_mid.pth")
    ap.add_argument("--weak", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--sets", nargs="*", default=["mnist_control", "hard1_diff"])
    ap.add_argument("--settings", nargs="*", default=["raw", "on"])
    ap.add_argument("--tau", type=float, default=1e-3, help="sigma_2/sigma_1 below this -> the attacker treats the release as one image")
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--random-starts", type=int, default=8)
    ap.add_argument("--search-iters", type=int, default=200); ap.add_argument("--polish-iters", type=int, default=600)
    ap.add_argument("--polish-top", type=int, default=3)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    ref = TrainedBackbone(a.strong, dev, "gelu")                                        # batches picked under the strong model
    if a.encoder == "random":                                                             # negative control: a trigger that fires for the wrong reason
        bb = RandomBackbone(TrainedBackbone(a.weak, dev, "gelu"), dev, a.encoder_seed)
    else:
        bb = TrainedBackbone(dict(strong=a.strong, mid=a.mid, weak=a.weak)[a.encoder], dev, "gelu")
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(bb.n)
    chart = PCAChart(Xtr_t, a.k, dev)
    coord_std = chart.coords_of(Xtr_t[:10000].T).std(dim=1, keepdim=True)              # PUBLIC scale of the chart's coordinates
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    if a.save_dir: os.makedirs(a.save_dir, exist_ok=True)
    print(f"# most_leaking  encoder={a.encoder}  sets={a.sets}  settings={a.settings}  tau={a.tau}  R={a.random_starts}  git={git_hash()}", flush=True)

    for sname in a.sets:
        if sname == "mnist_control":
            idx, seen = [], set()
            for i in perm.tolist():
                if int(yte[i]) not in seen: idx.append(i); seen.add(int(yte[i]))
                if len(idx) == a.N: break
        elif sname == "hard1_same":
            mar_all, _ = margins_of(ref, Xte_t.T, yte_t); mar_all = mar_all.cpu()
            hard = int(torch.argmin(mar_all)); c = int(yte[hard])
            cand = sorted([(float(mar_all[i]), i) for i in range(len(mar_all)) if int(yte[i]) == c and i != hard], reverse=True)
            idx = [hard] + [i for _, i in cand[:a.N - 1]]
        else:
            idx = pick_batch(sname, ref, Xte_t, yte_t, a.N, perm)
        idx = torch.tensor(idx, device=dev); X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        W_all = chart.coords_of(X_real); X_on = chart.psi(W_all)
        for setting in a.settings:
            g = torch.Generator().manual_seed(a.seed + 7)
            X_train = X_on if setting == "on" else X_real
            A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
            A_T, B_T, imp, sB, C = release_and_imprints(bb, X_train, y, A0, a)
            mar, _ = margins_of(bb, X_train, y)
            spectrum = [float(v / sB[0]) for v in sB]
            one_image = bool(spectrum[1] < a.tau)                                  # the attacker's read
            with torch.no_grad():                                                   # is the gap from one IMPRINT or from collinear FEATURES?
                Hn = bb.phi(X_train); Hn = Hn / torch.linalg.norm(Hn, dim=0, keepdim=True)
                sG = torch.linalg.svdvals(Hn.T @ Hn)
            gram = dict(feat_gram_sigma2_over_1=float(sG[1] / sG[0]), feat_gram_cond=float(sG[0] / sG[-1]),
                        feat_gram_spectrum_rel=[float(v / sG[0]) for v in sG])
            top = int(torch.argmax(imp))                                            # evaluation only
            fl = floor_pred(C, B_T, [top])
            t0 = time.time()
            print(f"##### {sname}/{setting}: y={y.tolist()} margins={[round(float(v),1) for v in mar]} imprints_rel={[f'{float(v/imp.max()):.0e}' for v in imp]} "
                  f"sigma2/sigma1={spectrum[1]:.1e} -> one_image={one_image}; top(eval)={top} (y={int(y[top])}) floor_pred={fl:.2e}", flush=True)
            # ---- the attacker's search: labels x random starts, nothing from the truth
            gs = torch.Generator().manual_seed(a.seed + 23)
            search = []
            for c in range(10):
                for s in range(a.random_starts):
                    W0_ = (torch.randn(a.k, 1, generator=gs).to(dev) * coord_std)
                    W_hat, aux, resid, iters = solve_one(chart, bb, A_T, B_T, torch.tensor([c], device=dev), W0_, a, a.search_iters, 1)
                    search.append(dict(label=c, start=s, residual=resid, iters=iters, W=W_hat.detach()))
                best_c = min((d for d in search if d["label"] == c), key=lambda d: d["residual"])
                print(f"      label {c}: best residual over {a.random_starts} random starts {best_c['residual']:.3e}", flush=True)
            search.sort(key=lambda d: d["residual"])
            polished = []
            for d in search[:a.polish_top]:
                W_hat, aux, resid, iters = solve_one(chart, bb, A_T, B_T, torch.tensor([d["label"]], device=dev), d["W"], a, a.polish_iters, 1)
                polished.append(dict(label=d["label"], start=d["start"], residual=resid, iters=iters, W=W_hat.detach()))
            polished.sort(key=lambda d: d["residual"]); best = polished[0]
            x_hat = chart.psi(best["W"])
            # ---- evaluation against the truth
            e_chart = (torch.linalg.norm(x_hat - X_on, dim=0) / torch.linalg.norm(X_on, dim=0))
            e_real = (torch.linalg.norm(x_hat - X_real, dim=0) / torch.linalg.norm(X_real, dim=0))
            nearest = int(torch.argmin(e_chart))
            truth = {}
            try:
                truth, W_top = one_image_truth_spectrum(chart, bb, A_T, B_T, X_real[:, top:top + 1], y[top:top + 1], A0, a)
            except Exception as ex:
                print(f"      (truth spectrum failed: {ex})", flush=True); W_top = W_all[:, top:top + 1]
            # ---- near-truth control (same procedure from the truth's neighbourhood, true label)
            W_near = W_top + a.init_noise * torch.randn(a.k, 1, generator=gs).to(dev) * W_top.std()
            _, _, res_near, it_near = solve_one(chart, bb, A_T, B_T, y[top:top + 1], W_near, a, a.polish_iters, 1)
            row = dict(part="most_leaking", encoder=a.encoder, backbone_test_acc=None, set=sname, setting=setting, y=y.tolist(),
                       margins=[float(v) for v in mar], imprint_rel=[float(v / imp.max()) for v in imp], B_T_spectrum_rel=spectrum,
                       tau=a.tau, attacker_reads_one_image=one_image, **gram, top_image_eval=top, top_label_eval=int(y[top]),
                       residual_floor_pred=fl, oracle=[], start="random (public coordinate scale)", random_starts=a.random_starts,
                       search_iters=a.search_iters, polish_iters=a.polish_iters,
                       search_best_per_label=[min(d["residual"] for d in search if d["label"] == c) for c in range(10)],
                       search_all=[dict(label=d["label"], start=d["start"], residual=d["residual"], iters=d["iters"]) for d in search],
                       # basin measure on the FIRST stage alone (no polish): fraction of (label, start) pairs at the floor
                       first_stage_frac_at_floor=(float(sum(d["residual"] <= 10 * fl for d in search)) / len(search)) if fl > 0 else None,
                       first_stage_frac_at_floor_true_label=(float(sum(d["residual"] <= 10 * fl for d in search if d["label"] == int(y[top]))) / a.random_starts) if fl > 0 else None,
                       first_stage_argmin_label=search[0]["label"], first_stage_argmin_label_correct=bool(search[0]["label"] == int(y[top])),
                       found_label=best["label"], label_correct=bool(best["label"] == int(y[top])),
                       argmin_residual_label=best["label"], argmin_residual_label_correct=bool(best["label"] == int(y[top])),
                       residual=best["residual"], residual_over_floor=(best["residual"] / fl if fl > 0 else None),
                       nearest_image_eval=nearest, nearest_is_top=bool(nearest == top),
                       argmin_residual_nearest_is_dominant=bool(nearest == top),
                       err_vs_top_chart=float(e_chart[top]), err_vs_top_real=float(e_real[top]),
                       err_vs_nearest_chart=float(e_chart[nearest]), chart_best_vs_real_top=float(torch.linalg.norm(X_on[:, top] - X_real[:, top]) / torch.linalg.norm(X_real[:, top])),
                       near_truth_control=dict(residual=res_near, iters=it_near, init_noise=a.init_noise),
                       **truth, k=a.k, N=a.N, r=a.r, m=bb.m, n=bb.n, T=a.T, lr=a.lr, seed=a.seed, seconds=time.time() - t0,
                       git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            print(json.dumps(row), flush=True)
            if a.out:
                with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
            if a.save_dir:
                torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat=x_hat.cpu(), top=top, meta=row),
                           os.path.join(a.save_dir, f"{a.encoder}_{sname}_{setting}.pth"))


if __name__ == "__main__":
    main()
