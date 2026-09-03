#!/usr/bin/env python3
"""Two attacker-side questions on the trained backbones (default: the strong 98% MLP).

A. FIND SOME OF THE SAMPLES.  The release carries only the examples the model had to learn (Step 18): B_T has
   numerical rank N' < N.  The attacker can read N' off the release and invert for N' images instead of N -- the
   budget per image grows (k < m + r - N'), and the invisible images set a residual FLOOR instead of zero.
   Labels: for N' = 1 searched over the 10 classes (the attacker does not know them); for N' > 1 GIVEN for the
   recorded images (oracle, flagged).  Start: near-truth of the recorded images (identifiability test, as in
   every cell here).  The N' = N inversion of the same release is the control ("find all" against "find some").

B. A DIFFERENT DISTRIBUTION.  Fine-tune on digits unlike the training set: rendered from system fonts (rotated)
   and the UCI optdigits scans (8x8, other writers, upscaled).  The mechanism predicts LOW margins -> strongly
   recorded; the attacker's MNIST chart predicts they are poorly drawable.  Per image: margin, residual, P_T
   column; rank B_T; then cells (a) on-chart and (b) off-chart with the MNIST PCA chart, against the chart's
   own best.  Image grids saved under figures/exact_inversion/.

  python -m experiments.exact_inversion.subset_and_ood --part A B
"""
import argparse, glob, json, math, os, socket, sys, time
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

from experiments.exact_inversion.lora_exact_inversion import train_release, qr_canon, invert_lm, git_hash
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


def release_and_columns(bb, X_train, y, A0, a):
    """Returns the release and the per-image IMPRINT norms ||C_i|| (B_T = sum_i C_i; basis-independent)."""
    H = bb.phi(X_train)
    A_T, B_T = train_release(H, A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
    _, B_tr, _, _, C = traced_release(H, A0, bb.W0, y, bb.m, a.T, a.lr)
    assert float(torch.linalg.norm(C.sum(0) - B_T) / torch.linalg.norm(B_T)) < 1e-10
    sB = torch.linalg.svdvals(B_T)
    return A_T, B_T, torch.linalg.norm(C.reshape(C.shape[0], -1), dim=1), sB


def invert_subset(chart, bb, A_T, B_T, X_sub_true, y_sub, a, dev, g, log=lambda s: None):
    """Invert the FULL release for a SUBSET of images (N' columns).  Unknowns: k x N' latents + r x N' seed block."""
    k = a.k; Np = len(y_sub)
    W_true = chart.coords_of(X_sub_true)
    W_init = W_true + a.init_noise * torch.randn(k, Np, generator=g).to(dev) * W_true.std()
    with torch.no_grad():
        Uc, _ = qr_canon(bb.phi(chart.psi(W_init))); Xinit = A_T @ Uc

    class Adapter:
        psi = staticmethod(chart.psi)
        features_from_latents = staticmethod(lambda Wc: bb.phi(chart.psi(Wc)))
    args = argparse.Namespace(m=bb.m, T=a.T, lr=a.lr, wd=0.0, release="sgd", seed=a.seed, restarts=a.restarts,
                              restart_noise=0.1, lm_iters=a.lm_iters, lm_lambda=1e-2, lm_scale="identity",
                              stage_x=0, jac="fwd", solver="lm", outer=30, lbfgs_iter=20)
    t0 = time.time()
    W_hat, aux, resid, sec, nrs, diag = invert_lm(Adapter, A_T, B_T, bb.W0, y_sub, args, W_init, Xinit, log)
    X_hat = chart.psi(W_hat)
    e = torch.linalg.norm(X_hat - X_sub_true, dim=0) / torch.linalg.norm(X_sub_true, dim=0)
    return dict(residual=float(resid), err_max=float(e.max()), err_median=float(e.median()),
                err_per_image=[float(v) for v in e], seconds=time.time() - t0, lm_iters_used=diag.get("lm_iters_used")), X_hat


def part_A(a, bb, ref, chart, Xte_t, yte_t, perm, dev, out, save_dir):
    g = torch.Generator().manual_seed(a.seed + 7)
    rows = []
    for mode, setting in [("repeated", "on"), ("hard1_diff", "on"), ("hard1_diff", "raw"), ("confident", "on")]:
        idx = torch.tensor(pick_batch(mode, ref, Xte_t, yte_t, a.N, perm), device=dev)
        X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]
        W_all = chart.coords_of(X_real); X_on = chart.psi(W_all)
        X_train = X_on if setting == "on" else X_real
        A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
        A_T, B_T, colP, sB = release_and_columns(bb, X_train, y, A0, a)
        mar, res0 = margins_of(bb, X_train, y)
        Np = int((sB > 1e-12 * sB[0]).sum())                       # what the attacker reads off the release
        order = torch.argsort(colP, descending=True)                 # evaluation only: which images ARE recorded (by imprint)
        rec = order[:Np]
        print(f"##### A: batch={mode} setting={setting} y={y.tolist()} margins={[round(float(v),1) for v in mar]} "
              f"|imprint|={[f'{float(v):.1e}' for v in colP]}  rank(B_T)@1e-12 = {Np}  recorded idx={rec.tolist()}", flush=True)
        base = dict(part="A", batch=mode, setting=setting, y=y.tolist(), margins=[float(v) for v in mar],
                    imprint_norms=[float(v) for v in colP], rank_B_T=Np, rank_B_T_1e8=int((sB > 1e-8 * sB[0]).sum()),
                    B_T_sigma_ratio=float(sB[a.N - 1] / sB[0]), recorded_idx=rec.tolist(), k=a.k, N=a.N, r=a.r, m=bb.m,
                    T=a.T, lr=a.lr, seed=a.seed, budget_line_full=bb.m + a.r - a.N, budget_line_subset=bb.m + a.r - Np)
        # the chart's truth for the recorded subset (on-chart: exact; raw: the chart's projection of the raw digit)
        X_sub_true = X_on[:, rec]; y_sub = y[rec]
        if Np == 1:                                                   # label SEARCH: the attacker does not know y
            search = []
            for c in range(10):
                r_c, _ = invert_subset(chart, bb, A_T, B_T, X_sub_true, torch.tensor([c], device=dev), a, dev,
                                       torch.Generator().manual_seed(a.seed + 11))
                search.append(dict(label=c, residual=r_c["residual"], err_max=r_c["err_max"]))
                print(f"      label {c}: residual {r_c['residual']:.3e}  err {r_c['err_max']:.3e}", flush=True)
            best = min(search, key=lambda d: d["residual"])
            row = dict(base, subset="N'=1, label searched", n_prime=1, labels_oracle=False, label_search=search,
                       label_found=best["label"], label_true=int(y_sub[0]), label_correct=bool(best["label"] == int(y_sub[0])),
                       residual=best["residual"], err_vs_chart_max=best["err_max"])
            e_real = float(torch.linalg.norm(chart.psi(chart.coords_of(X_sub_true)) - X_real[:, rec], dim=0).max() /
                           torch.linalg.norm(X_real[:, rec], dim=0).max())
            row.update(chart_best_vs_REAL=e_real)
        else:
            r_sub, X_hat = invert_subset(chart, bb, A_T, B_T, X_sub_true, y_sub, a, dev, torch.Generator().manual_seed(a.seed + 11))
            row = dict(base, subset=f"N'={Np}, labels given", n_prime=Np, labels_oracle=True, **r_sub)
            row.update(err_vs_chart_max=r_sub["err_max"])
            if save_dir:
                torch.save(dict(x_real=X_real[:, rec].cpu(), x_chart=X_sub_true.cpu(), x_hat=X_hat.cpu(), meta=row),
                           os.path.join(save_dir, f"A_{mode}_{setting}_subset.pth"))
        print(json.dumps(row), flush=True); rows.append(row)
        # control: "find all" on the same release
        r_all, X_hat_all = invert_subset(chart, bb, A_T, B_T, X_on, y, a, dev, torch.Generator().manual_seed(a.seed + 11))
        row_all = dict(base, subset="N'=N (find all, control)", n_prime=a.N, labels_oracle=True, **r_all)
        row_all.update(err_vs_chart_max=r_all["err_max"],
                       err_recorded_max=float(max(r_all["err_per_image"][i] for i in rec.tolist())),
                       err_invisible_max=float(max([r_all["err_per_image"][i] for i in range(a.N) if i not in rec.tolist()] or [float("nan")])))
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


def part_B(a, backbones, chart, dev, out, save_dir):
    labels = [0, 3, 5, 1, 9, 6, 7, 4]                                   # the distinct draw's label set
    sets = {"font": font_digits(labels, a.seed)[0].to(dev), "optdigits": optdigits(labels, a.optdigits, a.seed).to(dev)}
    y = torch.tensor(labels, device=dev)
    rows = []
    for sname, X_ood in sets.items():
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
                A_T, B_T, colP, sB = release_and_columns(bb, X_train, y, A0, a)
                row = dict(part="B", ood_set=sname, encoder=ename, backbone_test_acc=acc, setting=setting, y=labels,
                           ood_pred_at_W0=pred.tolist(), ood_acc_at_W0=float((pred == y).double().mean()),
                           margins=[float(v) for v in mar], residual_W0=[float(v) for v in res0],
                           imprint_norms=[float(v) for v in colP], rank_B_T=int((sB > 1e-12 * sB[0]).sum()),
                           B_T_sigma_ratio=float(sB[a.N - 1] / sB[0]),
                           chart_repr_err_median=float(repr_err.median()), chart_repr_err_max=float(repr_err.max()),
                           k=a.k, N=a.N, r=a.r, m=bb.m, T=a.T, lr=a.lr, seed=a.seed)
                print(json.dumps(row), flush=True); rows.append(row)
            # the attack with the MNIST chart, on-chart (a) and off-chart (b)
            for cell in ("a", "b"):
                r, X_hat, X_on_c = invert_cell(chart, bb, X_ood, y, a, cell, dev, g, lambda s: None)
                r.update(part="B", ood_set=sname, encoder=ename, backbone_test_acc=acc, chart="mnist_pca", y=labels,
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
    if "B" in a.part: rows += part_B(a, {"weak": weak, "mid": mid, "strong": strong}, chart, dev, a.out, a.save_dir)
    for row in rows:
        row.update(git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    if a.save_dir and a.fig: grid(a.save_dir, a.fig)


if __name__ == "__main__":
    main()
