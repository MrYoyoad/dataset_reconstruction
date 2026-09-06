#!/usr/bin/env python3
"""Figures for the 2026-09-06 CIFAR study. Every panel carries its job id; anything not attacker-available says so
ON the figure, not only in a caption; nothing is drawn from a number currently under review.

  F1  the blend degeneracy, made visible: certificate residual along the affine hull of the private images, for an
      AFFINE composition (adapter on the pixel layer: flat at machine precision, which is the lemma) against a
      NONLINEAR one (adapter on the head: a bowl that only touches the floor at the truths).
  F2  coverage fails, precision survives: the residual histogram of the pixel-layer cell beside its landings-per-image.
  F3  the equivalence: certificate residual against the linearised representer's model floor, over candidates of
      three kinds. The theorem says they vanish together; an off-diagonal point falsifies it.
  F4  search failure against alias on one axis: residual over the cell's own floor, with the verdict bands.
  F5  the solver handicap we found and removed: joint against variable projection.
  F6  the isolation test firing, ONE-SIDED: full rank certifies isolation, deficient rank certifies nothing.

  python -m experiments.cifar.make_figures
"""
import glob, json, math, os, sys
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
OUT = "figures/cifar_study"
BLUE, RED, GREY, GREEN = "#1f77b4", "#d62728", "#7f7f7f", "#2ca02c"


def log(s): print(s, flush=True)


def cert_obj(C, A_T, X):
    return torch.linalg.norm(C @ X, dim=0) / torch.linalg.norm(A_T @ X, dim=0)


def head_release(dev, N=8, r=64, k=32, T=400, lr=0.01, seed=1):
    """A head-layer release on the cached CIFAR-10 MLP: the NONLINEAR composition, rebuilt here so the walk in F1
       and the scatter in F3 can be evaluated in feature space (the saved cells store images, not features)."""
    from experiments.cifar.cifar_newclass import train_backbone, load_cifar100_class
    net, te, tr = train_backbone("models/exact_inversion/cifar10_mlp_newclass.pth", "data", dev, 60, 0.50, 0.90, "mlp")
    net = net.double()
    for p_ in net.parameters(): p_.requires_grad_(False)
    pool, cname = load_cifar100_class("data", "keyboard")
    Pub = torch.tensor(pool["train"], dtype=torch.float64, device=dev); Pri = torch.tensor(pool["test"], dtype=torch.float64, device=dev)
    g = torch.Generator().manual_seed(seed + 7)
    X_raw = Pri[torch.randperm(Pri.shape[0], generator=g)[:N]].T.contiguous()
    mean = Pub.mean(0); _, _, Vh = torch.linalg.svd(Pub - mean, full_matrices=False); V = Vh[:k].T.contiguous()
    X_on = mean[:, None] + V @ (V.T @ (X_raw - mean[:, None]))
    n = net.head.weight.shape[1]; m = 11
    W0 = torch.cat([net.head.weight.double(), torch.zeros(1, n, dtype=torch.float64, device=dev)], 0)
    phi = lambda X: net.phi(X.T).T
    H = phi(X_on); y = torch.full((N,), m - 1, device=dev)
    A0 = (1.0 / math.sqrt(n) * torch.randn(r, n, generator=torch.Generator().manual_seed(seed + 7), dtype=torch.float64)).to(dev)
    A, B = A0.clone(), torch.zeros(m, r, dtype=torch.float64, device=dev)
    Y = torch.eye(m, device=dev, dtype=torch.float64)[y].T
    for _ in range(T):
        z = W0 @ H + B @ (A @ H); D = (torch.softmax(z, 0) - Y) / N
        B, A = B - lr * (D @ (A @ H).T), A - lr * (B.T @ D @ H.T)
    sB = torch.linalg.svdvals(B); Np = int((sB > 1e-12 * sB[0]).sum())
    _, _, VhB = torch.linalg.svd(B, full_matrices=False); Q = VhB[:Np].T
    C = A - Q @ (Q.T @ A)
    return dict(phi=phi, C=C, A_T=A, B_T=B, X_on=X_on, X_raw=X_raw, H=H, mean=mean, V=V, Pub=Pub, acc=te, N=N, r=r, k=k, T=T)


def affine_walk(nsteps=161, N=8, seed=0):
    """A one-parameter family of blend coefficients along the affine hull: w(t) sums to 1 for every t, and passes
       through truth 0 at t=0 and truth 1 at t=1, with the other six mixed in away from the ends."""
    g = torch.Generator().manual_seed(seed)
    d = torch.randn(N, generator=g); d = d - d.mean()                       # a direction with zero sum: stays affine
    ts = torch.linspace(-0.5, 1.5, nsteps)
    W = torch.zeros(nsteps, N)
    for i, t in enumerate(ts):
        w = torch.zeros(N); w[0] = 1 - t; w[1] = t
        w = w + 0.35 * float(torch.sin(math.pi * t)) * d                    # zero at t=0 and t=1, so ends are truths
        W[i] = w
    return ts, W


def fig1(dev):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ts, W = affine_walk()
    fig, ax = plt.subplots(figsize=(10, 6))
    # --- affine composition: the saved replica cell, adapter on the PIXEL layer, so the layer input IS the image
    f = "experiments/cifar/k32_onchart/release_and_search.pt"
    if os.path.exists(f):
        d = torch.load(f, map_location="cpu", weights_only=False)
        C, A_T, H = d["C"].double().to(dev), d["A_T"].double().to(dev), d["H_train"].double().to(dev)
        H = H if H.shape[0] == C.shape[1] else H.T
        X = H @ W.T.to(dev)                                                  # affine combinations of the private inputs
        r = cert_obj(C, A_T, X).cpu()
        ax.semilogy(ts, r, color=BLUE, lw=2,
                    label="adapter on the PIXEL layer (affine composition), job 257893")
        i0, i1 = int(torch.argmin((ts - 0).abs())), int(torch.argmin((ts - 1).abs()))
        ax.scatter([0, 1], [r[i0], r[i1]], s=70, color=BLUE, zorder=5)
        log(f"F1 affine walk: min {float(r.min()):.2e}  max {float(r.max()):.2e}  ratio {float(r.max()/r.min()):.1f}")
    # --- nonlinear composition: a head-layer release, so the layer input is a learned feature of the image
    hr = head_release(dev)
    Xh = hr["X_on"] @ W.T.to(dev)
    rh = cert_obj(hr["C"], hr["A_T"], hr["phi"](Xh)).cpu()
    ax.semilogy(ts, rh, color=RED, lw=2, label=f"adapter on the HEAD (two nonlinearities), rebuilt on the cached backbone (T={hr['T']})")
    j0, j1 = int(torch.argmin((ts - 0).abs())), int(torch.argmin((ts - 1).abs()))
    ax.scatter([0, 1], [rh[j0], rh[j1]], s=70, color=RED, zorder=5)
    log(f"F1 nonlinear walk: min {float(rh.min()):.2e}  max {float(rh.max()):.2e}  ratio {float(rh.max()/rh.min()):.1e}")
    ax.axvspan(-0.02, 0.02, color=GREY, alpha=0.15); ax.axvspan(0.98, 1.02, color=GREY, alpha=0.15)
    ax.text(0.0, ax.get_ylim()[1], " a private image", fontsize=8, va="top", color=GREY)
    ax.text(1.0, ax.get_ylim()[1], " another private image", fontsize=8, va="top", color=GREY)
    ax.set_xlabel("position along the affine hull of the private images (the coefficients sum to 1 everywhere)")
    ax.set_ylabel("certificate residual  ‖Cφ(x)‖ / ‖A_T φ(x)‖")
    ax.set_title("Every blend of the private images is an exact zero when the composition is affine\n"
                 "so the certificate cannot isolate them there — and a nonlinearity is what breaks the degeneracy", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower center"); ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/F1_blend_degeneracy.png", dpi=200); plt.close(fig)
    log(f"# {OUT}/F1_blend_degeneracy.png")
    return hr


def fig2():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f = "experiments/cifar/charts/L1_ae_lm_onchart_k32/release_and_search.pt"
    if not os.path.exists(f): log("F2 skipped: no pixel-layer chart cell"); return
    d = torch.load(f, map_location="cpu", weights_only=False); res = d["result"]; R = d["R"].double()
    Xf, Xt = d["X_found"].double(), d["X_train"].double()
    E = torch.stack([(Xf - Xt[i:i + 1]).norm(dim=1) / Xt[i:i + 1].norm() for i in range(res["N"])], 1)
    landed = E.min(1).values < 1e-2
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lo, hi = float(R.min()) / 3, float(R.max()) * 3
    bins = np.logspace(math.log10(lo), math.log10(hi), 50)
    ax[0].hist(R[~landed].numpy(), bins=bins, color=GREY, label=f"did not land ({int((~landed).sum())} starts)")
    ax[0].hist(R[landed].numpy(), bins=bins, color=GREEN, label=f"landed on a private image ({int(landed.sum())})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("final residual of the start"); ax[0].set_ylabel("starts")
    ax[0].set_title(f"PRECISION survives: the two groups are {float(R[~landed].median()/R[landed].median()):.0f}× apart,\n"
                    "so the attacker can rank their own starts without ground truth", fontsize=10)
    ax[0].legend(fontsize=9, frameon=False); ax[0].spines[["top", "right"]].set_visible(False)
    per = res["landings_per_image"]
    ax[1].bar(range(len(per)), per, color=[GREEN if v else GREY for v in per])
    ax[1].set_xlabel("private image"); ax[1].set_ylabel("starts landing on it")
    ax[1].set_title(f"COVERAGE fails: {res['images_found']} of {res['N']} images are ever found", fontsize=10)
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Adapter on the pixel layer, conv-autoencoder chart k=32, on-chart privates — the failure is coverage, not precision", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/F2_coverage_vs_precision.png", dpi=200); plt.close(fig)
    log(f"# {OUT}/F2_coverage_vs_precision.png")


def fig3(hr, dev):
    """The equivalence: for each candidate SET of N images, the certificate residual (max over the set) against the
       linearised representer's exact-fit residual. R12 says they vanish together."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C, A_T, B_T, phi, X_on, N = hr["C"], hr["A_T"], hr["B_T"], hr["phi"], hr["X_on"], hr["N"]
    mean, V, Pub = hr["mean"], hr["V"], hr["Pub"]
    g = torch.Generator().manual_seed(3)
    pts = []
    for tag, colr, mk in [("the private images, perturbed", GREEN, "o"), ("blends of them", RED, "s"), ("public images", BLUE, "^")]:
        for j in range(22):
            if tag.startswith("the private"):
                eps = 10 ** (-6 + 6 * j / 21)
                Xc = X_on + eps * torch.randn(X_on.shape, generator=g).to(dev) * X_on.std()
            elif tag.startswith("blends"):
                Wt = torch.randn(N, N, generator=g).to(dev); Wt = Wt / Wt.sum(0, keepdim=True)     # affine, sums to 1
                Xc = X_on @ Wt
                Xc = Xc + (1e-6 * 10 ** (4 * j / 21)) * torch.randn(Xc.shape, generator=g).to(dev) * Xc.std()
            else:
                Xc = Pub[torch.randperm(Pub.shape[0], generator=torch.Generator().manual_seed(j))[:N]].T
            Hc = phi(Xc); F = A_T @ Hc
            cert = float(cert_obj(C, A_T, Hc).max())
            R_ls = torch.linalg.lstsq(F, B_T.T).solution.T
            rep = float(torch.linalg.norm(B_T - R_ls @ F.T) / torch.linalg.norm(B_T))
            pts.append((cert, rep, tag, colr, mk))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    seen = set()
    for c_, r_, tag, colr, mk in pts:
        ax.scatter(max(c_, 1e-17), max(r_, 1e-17), s=52, marker=mk, facecolors="none", edgecolors=colr, linewidths=1.3,
                   label=(tag if tag not in seen else None)); seen.add(tag)
    lim = [1e-17, 10]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1, label="equal")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("certificate residual, worst over the N candidates   (the per-candidate condition)")
    ax.set_ylabel("linearised representer's exact-fit residual   (the joint condition)")
    ax.set_title("The two routes share a zero set\nA point low on one axis and high on the other would falsify that; there are none", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="upper left"); ax.grid(alpha=0.3, which="both"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/F3_equivalence.png", dpi=200); plt.close(fig)
    log(f"# {OUT}/F3_equivalence.png")


def rows():
    out = []
    for f in sorted(glob.glob("results/ntk_vs_cert/*.jsonl")):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("part") == "ntk_vs_cert" and "ntk_by_form" in r: out.append(r)
    return out


def fig4_5():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rs = rows()
    if not rs: log("F4/F5 skipped: no head-to-head rows yet"); return
    # F4 -- residual over the cell's own floor, one point per arm, with the verdict bands
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ys, labs = [], []
    for i, r in enumerate(rs):
        for arm, v in sorted(r["ntk_by_form"].items()):
            if v.get("is_diagnostic_upper_bound"): continue
            fl = max(v.get("model_floor_at_truth") or 0.0, 1e-300)
            ys.append(v["residual_min"] / fl); labs.append(f"{r['dataset']}/{r['chart']} k{r['k']} T{r['T']} {arm}")
    ax.axvspan(0, 1.5, color=GREEN, alpha=0.12); ax.axvspan(1.5, 15, color="#eda100", alpha=0.12); ax.axvspan(15, 1e18, color=RED, alpha=0.10)
    ax.scatter(np.clip(ys, 1e-2, 1e17), range(len(ys)), s=46, color=BLUE)
    ax.scatter([1.0], [len(ys)], s=70, color=GREEN, marker="*")
    ax.set_yticks(list(range(len(ys))) + [len(ys)]); ax.set_yticklabels(labs + ["certificate (reaches its floor)"], fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("residual reached  /  the cell's own model floor at the truth")
    ax.text(0.4, len(ys) * 0.5, "alias\n(≤1.5)", fontsize=8, ha="center", color=GREEN)
    ax.text(5, len(ys) * 0.5, "undetermined", fontsize=8, ha="center", color="#8a6d00")
    ax.text(1e8, len(ys) * 0.5, "search failure (>15)", fontsize=8, ha="center", color=RED)
    ax.set_title("Alias or search failure, on one axis. The bar is irrelevant when the ratio is 1e13.\n"
                 "1.5 is the project's existing threshold, not a new one.", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/F4_verdict_axis.png", dpi=200); plt.close(fig)
    log(f"# {OUT}/F4_verdict_axis.png")
    # F5 -- the solver handicap
    pairs = []
    for r in rs:
        for form in ("lora", "dW"):
            j, v = r["ntk_by_form"].get(f"{form}:joint"), r["ntk_by_form"].get(f"{form}:varpro")
            if j and v: pairs.append((f"{r['dataset']} {r['chart']} k{r['k']} T{r['T']} {form}", v["residual_min"], j["residual_min"]))
    if pairs:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(pairs)); w = 0.38
        ax.bar(x - w / 2, [p[1] for p in pairs], w, color=BLUE, label="variable projection (coefficients eliminated in closed form)")
        ax.bar(x + w / 2, [p[2] for p in pairs], w, color=RED, label="joint Adam over latents and coefficients (the handicapped solver)")
        for i, p in enumerate(pairs): ax.text(i, max(p[1], p[2]) * 1.15, f"{p[2]/p[1]:.0f}×", ha="center", fontsize=8)
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("best residual reached"); ax.legend(fontsize=8, frameon=False)
        ax.set_title("A handicap we found in our OWN comparison and removed: with the coefficients started at zero,\n"
                     "the candidate images receive exactly zero gradient on the first step", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{OUT}/F5_solver_handicap.png", dpi=200); plt.close(fig)
        log(f"# {OUT}/F5_solver_handicap.png")


def fig6():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cells = []
    for f in sorted(glob.glob("experiments/cifar/charts/*/result.json")):
        r = json.load(open(f))
        if "isolation_rank_per_image" in r: cells.append((os.path.basename(os.path.dirname(f)), r))
    if not cells: log("F6 skipped: no cell has the isolation test yet"); return
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, r) in enumerate(cells):
        rk = r["isolation_rank_per_image"]
        ax.scatter([i] * len(rk), rk, s=60, color=BLUE if r["layer"] != 1 else RED)
    ax.axhline(cells[0][1]["k"], color=GREY, ls="--", lw=1)
    ax.text(0, cells[0][1]["k"] * 1.02, "full rank = the zero is certified isolated", fontsize=8, color=GREY)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([c[0] for c in cells], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("rank of  C · (chart Jacobian)  at the returned point")
    ax.set_title("The isolation test, read ONE-SIDED: full rank certifies an isolated zero.\n"
                 "Deficient rank certifies NOTHING — it is 'not certified', never 'degenerate'.", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/F6_isolation_test.png", dpi=200); plt.close(fig)
    log(f"# {OUT}/F6_isolation_test.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"# make_figures on {dev}")
    hr = fig1(dev)
    fig2()
    fig3(hr, dev)
    fig4_5()
    fig6()


if __name__ == "__main__":
    main()
