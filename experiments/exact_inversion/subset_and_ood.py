#!/usr/bin/env python3
"""Two attacker-side questions on the trained backbones (default: the strong 98% MLP).

A. FIND SOME OF THE SAMPLES.  The release carries only the examples the model had to learn (Step 18): B_T has
   numerical rank N' < N.  The attacker can read N' off the release and invert for N' images instead of N -- the
   budget per image grows (k < m + r - N'), and the invisible images set a residual FLOOR instead of zero.
   WHAT IS ATTACKER-REALIZABLE HERE IS ONLY THE COUNT N'.  WHICH images are the recorded ones is taken from the
   ground-truth imprints (subset_identity = "oracle (imprint)" in every row), the start is near-truth, and for
   N' > 2 the labels are given: Part A asks "given which images are recorded, are they identifiable from the
   release?", not "does an attacker discover which images are recorded".  Every row carries its oracle list,
   start_err_median, init_noise and identifiability_test=True.
   Labels: for N' = 1 searched over the 10 classes (the attacker does not know them); for N' > 1 GIVEN for the
   recorded images (oracle, flagged).  Start: near-truth of the recorded images (identifiability test, as in
   every cell here).  The N' = N inversion of the same release is the control ("find all" against "find some").

B. A DIFFERENT DISTRIBUTION.  Fine-tune on digits unlike the training set: rendered from system fonts (rotated)
   and the UCI optdigits scans (8x8, other writers, upscaled).  The mechanism predicts LOW margins -> strongly
   recorded; the attacker's MNIST chart predicts they are poorly drawable.  Per image: margin, residual, imprint
   ||C_i|| (B_T = sum_i C_i, basis-free); rank B_T; then cells (a) on-chart and (b) off-chart with the MNIST PCA chart, against the chart's
   own best.  A RANDOM MNIST draw runs through the same pipeline as the in-distribution control (the Part-A
   batches are margin-picked and must not serve as that).  Margins are reported FIRST: an OOD set whose margins
   are not below the MNIST control's is a failed manipulation, not a result.  The single number is rank B_T at
   fixed N, with the chart representation error beside it as the separable fidelity axis.  Grids saved.

  python -m experiments.exact_inversion.subset_and_ood --part A B
"""
import argparse, glob, itertools, json, math, os, socket, sys, time
import numpy as np, torch, torch.func as tf
from PIL import Image, ImageDraw, ImageFont

from experiments.exact_inversion.lora_exact_inversion import train_release, simulate_sgd_reduced, qr_canon, invert_lm, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.vae_chart import invert_cell
from experiments.exact_inversion.margin_check import traced_release

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------------------------- batches (part A)
def margins_of(bb, X, y):
    with torch.no_grad():
        z = bb.logits(X)
    ar = torch.arange(z.shape[1], device=z.device)
    zy = z[y, ar]; zo = z.clone(); zo[y, ar] = -float("inf")
    return zy - zo.max(0).values, torch.linalg.norm(torch.softmax(z, 0) - torch.eye(bb.m, device=z.device)[:, y], dim=0)


def pick_batch(mode, ref, Xte_t, yte_t, N, perm):
    """Same definitions as margin_check.py --pick (copied: that module is under a running job)."""
    yte = yte_t.cpu()
    if mode == "repeated": return perm[:N].tolist()
    mar, _ = margins_of(ref, Xte_t.T, yte_t); mar = mar.cpu()
    def top_of(c, n, exclude=()):
        cand = [(float(mar[i]), i) for i in range(len(mar)) if int(yte[i]) == c and i not in exclude]
        return [i for _, i in sorted(cand, reverse=True)[:n]]
    hard = int(torch.argmin(mar)); c = int(yte[hard])
    if mode == "hard1_diff":
        others = sorted([k for k in range(10) if k != c], key=lambda k: -float(mar[top_of(k, 1)[0]]))[:N - 1]
        return [hard] + [top_of(k, 1)[0] for k in others]
    if mode == "confident":
        return [top_of(c, 1)[0] for c in sorted(range(10), key=lambda c: -float(mar[top_of(c, 1)[0]]))[:N]]
    raise ValueError(mode)


def release_and_imprints(bb, X_train, y, A0, a):
    """Returns the release and the per-image IMPRINT norms ||C_i|| (B_T = sum_i C_i; basis-independent)."""
    H = bb.phi(X_train)
    A_T, B_T = train_release(H, A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
    _, B_tr, _, _, C = traced_release(H, A0, bb.W0, y, bb.m, a.T, a.lr)
    assert float(torch.linalg.norm(C.sum(0) - B_T) / torch.linalg.norm(B_T)) < 1e-10
    sB = torch.linalg.svdvals(B_T)
    return A_T, B_T, torch.linalg.norm(C.reshape(C.shape[0], -1), dim=1), sB, C


def invert_subset(chart, bb, A_T, B_T, X_sub_true, y_sub, a, dev, g, log=lambda s: None, A0=None):
    """Invert the FULL release for a SUBSET of images (N' columns).  Unknowns: k x N' latents + r x N' seed block.
       With A0 given, also the Jacobian of the SUBSET residual at the subset truth (W_true, A0 U_sub): sigma_min
       there separates "identifiable subset" from "the near-truth start did not move"; the residual there is the
       floor the omitted images impose (not zero), reported per block."""
    k = a.k; Np = len(y_sub); nW = k * Np
    W_true = chart.coords_of(X_sub_true)
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T)

    def res_vec(v):
        Wc = v[:nW].reshape(k, Np); aux = v[nW:].reshape(a.r, Np)
        Bs, Xis, Uc = simulate_sgd_reduced(bb.phi(chart.psi(Wc)), aux, bb.W0, y_sub, bb.m, a.T, a.lr, 0.0)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])

    truth = {}
    if A0 is not None:
        U_sub, _ = qr_canon(bb.phi(X_sub_true)); v0 = torch.cat([W_true.reshape(-1), (A0 @ U_sub).reshape(-1)]).detach()
        sv = torch.linalg.svdvals(tf.jacfwd(res_vec)(v0).detach()); rv = res_vec(v0)
        truth = dict(jac_sigma_min_truth=float(sv[-1]), jac_sigma_max_truth=float(sv[0]), jac_cond_truth=float(sv[0] / sv[-1]),
                     jac_rank_truth=int((sv > 1e-12 * sv[0]).sum()), jac_cols=int(sv.numel()),
                     res_at_truth=float(rv.norm() ** 2), res_at_truth_B=float(rv[:bb.m * a.r].norm() ** 2),
                     res_at_truth_A=float(rv[bb.m * a.r:].norm() ** 2))
    W_init = W_true + a.init_noise * torch.randn(k, Np, generator=g).to(dev) * W_true.std()
    with torch.no_grad():
        Uc, _ = qr_canon(bb.phi(chart.psi(W_init))); Xinit = A_T @ Uc
        e0 = torch.linalg.norm(chart.psi(W_init) - X_sub_true, dim=0) / torch.linalg.norm(X_sub_true, dim=0)

    class Adapter:
        psi = staticmethod(chart.psi)
        features_from_latents = staticmethod(lambda Wc: bb.phi(chart.psi(Wc)))
    args = argparse.Namespace(m=bb.m, T=a.T, lr=a.lr, wd=0.0, release="sgd", seed=a.seed, restarts=a.restarts,
                              restart_noise=0.1, lm_iters=a.lm_iters, lm_lambda=1e-2, lm_scale="identity",
                              stage_x=0, jac="fwd", solver="lm", outer=30, lbfgs_iter=20)
    t0 = time.time()
    W_hat, aux, resid, sec, nrs, diag = invert_lm(Adapter, A_T, B_T, bb.W0, y_sub, args, W_init, Xinit, log)
    X_hat = chart.psi(W_hat)
    with torch.no_grad():                                          # the residual per block (different floors)
        Bs, Xis, Uc = simulate_sgd_reduced(bb.phi(X_hat), aux, bb.W0, y_sub, bb.m, a.T, a.lr, 0.0)
        res_B = float(torch.linalg.norm(Bs - B_T) ** 2 / torch.linalg.norm(B_T) ** 2)
        res_A = float(torch.linalg.norm(Xis - A_T @ Uc) ** 2 / torch.linalg.norm(A_T) ** 2)
    e = torch.linalg.norm(X_hat - X_sub_true, dim=0) / torch.linalg.norm(X_sub_true, dim=0)
    return dict(residual=float(resid), residual_B=res_B, residual_A=res_A, err_max=float(e.max()), err_median=float(e.median()),
                err_per_image=[float(v) for v in e], start_err_median=float(e0.median()), start_err_max=float(e0.max()),
                init_noise=a.init_noise, identifiability_test=True, **truth,
                seconds=time.time() - t0, lm_iters_used=diag.get("lm_iters_used")), X_hat


def floor_pred(C, B_T, subset):
    """What an N'-image model can at best leave: the omitted images' summed imprint, relative (B-block)."""
    omitted = [j for j in range(C.shape[0]) if j not in subset]
    if not omitted: return 0.0
    return float(torch.linalg.norm(C[omitted].sum(0)) ** 2 / torch.linalg.norm(B_T) ** 2)


def label_search(chart, bb, A_T, B_T, X_sub_true, a, dev, budget):
    """All 10^N' labelings, reduced budget; returns the ranking by residual (the attacker's procedure)."""
    Np = X_sub_true.shape[1]; out = []
    b = argparse.Namespace(**vars(a)); b.lm_iters = budget; b.restarts = 1
    for combo in itertools.product(range(10), repeat=Np):
        r, _ = invert_subset(chart, bb, A_T, B_T, X_sub_true, torch.tensor(combo, device=dev), b, dev,
                             torch.Generator().manual_seed(a.seed + 11))
        out.append(dict(labels=list(combo), residual=r["residual"], residual_B=r["residual_B"], err_max=r["err_max"]))
    return sorted(out, key=lambda d: d["residual"])


def part_A(a, bb, ref, chart, Xte_t, yte_t, perm, dev, out, save_dir):
    g = torch.Generator().manual_seed(a.seed + 7)
    rows = []
    for mode, setting in [("repeated", "on"), ("hard1_diff", "on"), ("hard1_diff", "raw"), ("confident", "on")]:
        idx = torch.tensor(pick_batch(mode, ref, Xte_t, yte_t, a.N, perm), device=dev)
        X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        W_all = chart.coords_of(X_real); X_on = chart.psi(W_all)
        X_train = X_on if setting == "on" else X_real
        A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
        A_T, B_T, imp, sB, C = release_and_imprints(bb, X_train, y, A0, a)
        mar, res0 = margins_of(bb, X_train, y)
        spectrum = [float(v / sB[0]) for v in sB]
        order = torch.argsort(imp, descending=True).tolist()         # evaluation only: who IS recorded (by imprint)
        n12 = int((sB > 1e-12 * sB[0]).sum()); n8 = int((sB > 1e-8 * sB[0]).sum())
        print(f"##### A: batch={mode} setting={setting} y={y.tolist()} margins={[round(float(v),1) for v in mar]} "
              f"|imprint|={[f'{float(v):.1e}' for v in imp]}  rank(B_T) 1e-12/1e-8 = {n12}/{n8}  spectrum={[f'{v:.1e}' for v in spectrum]}", flush=True)
        base = dict(part="A", batch=mode, setting=setting, y=y.tolist(), margins=[float(v) for v in mar],
                    imprint_norms=[float(v) for v in imp], imprint_order=order, B_T_spectrum_rel=spectrum,
                    rank_B_T_1e12=n12, rank_B_T_1e8=n8, k=a.k, N=a.N, r=a.r, m=bb.m, T=a.T, lr=a.lr, seed=a.seed,
                    budget_line_full=bb.m + a.r - a.N)
        for Np in sorted(set([n12, n8])):
            rec = order[:Np]
            swapped = sorted(rec[:-1] + [order[Np]]) if Np < a.N else None           # weakest recorded -> strongest invisible
            confident_only = order[-Np:] if Np < a.N else None
            for sname, sub in [("recorded", rec), ("one_swapped", swapped), ("confident_only", confident_only)]:
                if sub is None: continue
                sub = sorted(sub); sub_t = torch.tensor(sub, device=dev)
                X_sub_true = X_on[:, sub_t]; y_sub = y[sub_t]
                fl = floor_pred(C, B_T, sub)
                row = dict(base, subset=sname, subset_idx=sub, n_prime=Np, residual_floor_pred=fl,
                           budget_line_subset=bb.m + a.r - Np, oracle=["subset_identity", "near_init"],
                           subset_identity="oracle (imprint)", n_prime_source="rank(B_T) read off the release")
                if sname == "recorded" and Np <= 2:                    # the attacker's label procedure
                    ranked = label_search(chart, bb, A_T, B_T, X_sub_true, a, dev, budget=300)
                    found = ranked[0]["labels"]; y_run = torch.tensor(found, device=dev)
                    row.update(label_search_top5=ranked[:5], label_found=found, label_true=y_sub.tolist(),
                               label_correct=bool(found == y_sub.tolist()), labels_oracle=False)
                else:
                    y_run = y_sub; row.update(labels_oracle=True); row["oracle"] = row["oracle"] + ["labels"]
                r_sub, X_hat = invert_subset(chart, bb, A_T, B_T, X_sub_true, y_run, a, dev, torch.Generator().manual_seed(a.seed + 11), A0=A0)
                row.update(**r_sub, err_vs_chart_max=r_sub["err_max"],
                           residual_over_floor=(r_sub["residual_B"] / fl if fl > 0 else None))
                print(json.dumps(row), flush=True); rows.append(row)
                if save_dir:
                    torch.save(dict(x_real=X_real[:, sub_t].cpu(), x_chart=X_sub_true.cpu(), x_hat=X_hat.cpu(), meta=row),
                               os.path.join(save_dir, f"A_{mode}_{setting}_N{Np}_{sname}.pth"))
        # control: "find all" on the same release (floor 0)
        r_all, X_hat_all = invert_subset(chart, bb, A_T, B_T, X_on, y, a, dev, torch.Generator().manual_seed(a.seed + 11), A0=A0)
        rec = order[:n12]
        row_all = dict(base, subset="all (control)", subset_idx=list(range(a.N)), n_prime=a.N, residual_floor_pred=0.0,
                       oracle=["near_init", "labels"], labels_oracle=True, subset_identity="n/a (all N)", **r_all)
        row_all.update(err_vs_chart_max=r_all["err_max"],
                       err_recorded_max=float(max(r_all["err_per_image"][i] for i in rec)),
                       err_invisible_max=float(max([r_all["err_per_image"][i] for i in range(a.N) if i not in rec] or [float("nan")])))
        print(json.dumps(row_all), flush=True); rows.append(row_all)
        if save_dir:
            torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat=X_hat_all.cpu(), meta=row_all),
                       os.path.join(save_dir, f"A_{mode}_{setting}_all.pth"))
    return rows


# ----------------------------------------------------------------------------------------------- OOD digits (part B)
def mnist_like(img28):
    """Centre a 28x28 float image by its centre of mass, as MNIST is."""
    x = torch.tensor(img28, dtype=torch.float64)
    if x.sum() <= 0: return x.reshape(-1)
    ys, xs = torch.meshgrid(torch.arange(28.), torch.arange(28.), indexing="ij")
    cy = float((x * ys).sum() / x.sum()); cx = float((x * xs).sum() / x.sum())
    sh = (int(round(13.5 - cy)), int(round(13.5 - cx)))
    x = torch.roll(x, shifts=sh, dims=(0, 1))
    return x.reshape(-1)


def font_digits(labels, seed):
    """White-on-black digits rendered from DejaVu fonts into a 20-px box, rotated, centred like MNIST."""
    fonts = sorted(glob.glob("/usr/share/fonts/dejavu-*/DejaVu*.ttf"))
    prefer = [f for f in fonts if os.path.basename(f) in ("DejaVuSans-Bold.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSansMono-Bold.ttf",
                                                           "DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf")] or fonts
    rng = np.random.RandomState(seed); X = []
    for i, d in enumerate(labels):
        fp = prefer[i % len(prefer)]
        big = Image.new("L", (112, 112), 0); dr = ImageDraw.Draw(big)
        font = ImageFont.truetype(fp, 84)
        bbox = dr.textbbox((0, 0), str(d), font=font)
        dr.text((56 - (bbox[0] + bbox[2]) / 2, 56 - (bbox[1] + bbox[3]) / 2), str(d), fill=255, font=font)
        big = big.rotate(float(rng.uniform(-12, 12)), resample=Image.BILINEAR)
        arr = np.asarray(big, dtype=np.float64); ys, xs = np.nonzero(arr > 32)
        crop = Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8))
        w, h = crop.size; s = 20.0 / max(w, h)
        crop = crop.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)
        canvas = Image.new("L", (28, 28), 0); canvas.paste(crop, ((28 - crop.size[0]) // 2, (28 - crop.size[1]) // 2))
        X.append(mnist_like(np.asarray(canvas, dtype=np.float64) / 255.0))
    return torch.stack(X, 1), [os.path.basename(prefer[i % len(prefer)]) for i in range(len(labels))]


def optdigits(labels, path, seed):
    """UCI optdigits (8x8, 0..16), upscaled to a 20-px box and centred; one digit per requested label."""
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64); Xd, yd = raw[:, :64] / 16.0, raw[:, 64].astype(int)
    rng = np.random.RandomState(seed); X = []
    for d in labels:
        i = int(rng.choice(np.nonzero(yd == d)[0]))
        t = torch.tensor(Xd[i].reshape(1, 1, 8, 8))
        up = torch.nn.functional.interpolate(t, size=(20, 20), mode="bilinear", align_corners=False)[0, 0]
        canvas = torch.zeros(28, 28); canvas[4:24, 4:24] = up
        X.append(mnist_like(canvas.numpy()))
    return torch.stack(X, 1)


def part_B(a, backbones, chart, dev, out, save_dir, perm):
    labels = [0, 3, 5, 1, 9, 6, 7, 4]                                   # the distinct draw's label set
    # in-distribution CONTROL: the distinct-label random draw from the test split (NOT a margin-picked batch)
    idx, seen = [], set()
    for i in perm.tolist():
        if int(a.yte_t[i]) not in seen: idx.append(i); seen.add(int(a.yte_t[i]))
        if len(idx) == a.N: break
    X_mn = a.Xte_t[torch.tensor(idx, device=dev)].T.contiguous(); y_mn = a.yte_t[torch.tensor(idx, device=dev)]
    sets = {"mnist_control": (X_mn, y_mn),
            "font": (font_digits(labels, a.seed)[0].to(dev), torch.tensor(labels, device=dev)),
            "optdigits": (optdigits(labels, a.optdigits, a.seed).to(dev), torch.tensor(labels, device=dev))}
    rows = []
    for sname, (X_ood, y) in sets.items():
        labels = y.tolist()
        W = chart.coords_of(X_ood); X_on = chart.psi(W)
        repr_err = (torch.linalg.norm(X_on - X_ood, dim=0) / torch.linalg.norm(X_ood, dim=0))
        for ename, bb in backbones.items():
            g = torch.Generator().manual_seed(a.seed + 7)
            with torch.no_grad():
                acc = float((bb.logits(a.Xte_t[:2000].T).argmax(0) == a.yte_t[:2000]).double().mean())
                pred = bb.logits(X_ood).argmax(0)
            A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
            for setting, X_train in [("raw", X_ood), ("on", X_on)]:
                mar, res0 = margins_of(bb, X_train, y)
                A_T, B_T, imp, sB, _ = release_and_imprints(bb, X_train, y, A0, a)
                row = dict(part="B", ood_set=sname, encoder=ename, backbone_test_acc=acc, setting=setting, y=labels,
                           ood_pred_at_W0=pred.tolist(), ood_acc_at_W0=float((pred == y).double().mean()),
                           margins=[float(v) for v in mar], margin_median=float(mar.median()),
                           residual_W0=[float(v) for v in res0],
                           imprint_norms=[float(v) for v in imp], rank_B_T=int((sB > 1e-12 * sB[0]).sum()),
                           rank_B_T_1e8=int((sB > 1e-8 * sB[0]).sum()), B_T_spectrum_rel=[float(v / sB[0]) for v in sB],
                           B_T_sigma_ratio=float(sB[a.N - 1] / sB[0]),
                           chart_repr_err_median=float(repr_err.median()), chart_repr_err_max=float(repr_err.max()),
                           k=a.k, N=a.N, r=a.r, m=bb.m, T=a.T, lr=a.lr, seed=a.seed)
                print(json.dumps(row), flush=True); rows.append(row)
            # the attack with the MNIST chart, on-chart (a) and off-chart (b)
            for cell in ("a", "b"):
                r, X_hat, X_on_c = invert_cell(chart, bb, X_ood, y, a, cell, dev, g, lambda s: None)
                r.update(part="B", ood_set=sname, encoder=ename, backbone_test_acc=acc, chart="mnist_pca", y=labels,
                         oracle=["near_init", "labels"], init_noise=a.init_noise, identifiability_test=True,
                         start_convention="invert_cell: W_true + init_noise*std*noise in latent space (start_err not recomputed here)",
                         k=a.k, N=a.N, r=a.r, m=bb.m, T=a.T, lr=a.lr, seed=a.seed)
                print(json.dumps(r), flush=True); rows.append(r)
                if save_dir:
                    torch.save(dict(x_real=X_ood.cpu(), x_chart=X_on_c.cpu(), x_hat=X_hat.cpu(), meta=r),
                               os.path.join(save_dir, f"B_{sname}_{ename}_{cell}.pth"))
    return rows


def grid(save_dir, fig_path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(save_dir, "*.pth")))
    if not files: return
    fig, axes = plt.subplots(len(files) * 3, 8, figsize=(8, 3 * len(files)))
    for fi, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for r, key in enumerate(["x_real", "x_chart", "x_hat"]):
            X = d[key]
            for c in range(8):
                ax = axes[fi * 3 + r, c]; ax.axis("off")
                if c < X.shape[1]: ax.imshow(X[:, c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[fi * 3 + r, 0].set_title(f"{os.path.basename(f)[:-4]} / {key}", fontsize=6, loc="left")
    plt.tight_layout(); plt.savefig(fig_path, dpi=110); print(f"# figure {fig_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="*", default=["A", "B"])
    ap.add_argument("--strong", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--weak", default="dataset_reconstruction/models/weights-mnist10_gelu.pth")
    ap.add_argument("--mid", default="models/exact_inversion/mnist_mlp_mid.pth")
    ap.add_argument("--optdigits", default="data/ood_digits/optdigits.tes")
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--restarts", type=int, default=2); ap.add_argument("--lm-iters", type=int, default=1000)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None); ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); a.Xte_t = torch.tensor(Xte, device=dev); a.yte_t = torch.tensor(yte, device=dev)
    strong = TrainedBackbone(a.strong, dev, "gelu"); weak = TrainedBackbone(a.weak, dev, "gelu"); mid = TrainedBackbone(a.mid, dev, "gelu")
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(strong.n)
    chart = PCAChart(Xtr_t, a.k, dev)
    perm = torch.randperm(a.Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    if a.save_dir: os.makedirs(a.save_dir, exist_ok=True)
    print(f"# subset_and_ood  parts={a.part}  k={a.k} N={a.N} r={a.r} T={a.T} lr={a.lr}  git={git_hash()}", flush=True)
    rows = []
    if "A" in a.part: rows += part_A(a, strong, strong, chart, a.Xte_t, a.yte_t, perm, dev, a.out, a.save_dir)
    if "B" in a.part: rows += part_B(a, {"weak": weak, "mid": mid, "strong": strong}, chart, dev, a.out, a.save_dir, perm)
    for row in rows:
        row.update(git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    if a.save_dir and a.fig: grid(a.save_dir, a.fig)


if __name__ == "__main__":
    main()
