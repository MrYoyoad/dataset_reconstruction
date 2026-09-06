#!/usr/bin/env python3
"""Record strength vs recovery, inside a fixed chart (measured, not modelled).

Question. Does an example's recovery from the certificate search depend on how strongly it was recorded in the
adapter, or is recording a threshold? Exact arithmetic says threshold: if example i's direction is in row(B_T) the
certificate C = P_{row(B_T)^perp} A_T annihilates it exactly, whatever the gradient size; if not, no constraint.

Releases (both rebuilt from their seeds, gated against the saved job rows; NO canonical script is modified):
  letters    EMNIST 'a' as an 11th class on the 98% MNIST MLP, r = 64, k = 32, N = 8, T = 400, lr = 0.01, seed 1,
             trained in fp64 / fp32 / bf16 / fp16 (train_precision.release_in) -- job 760909 (Figure 2) and the
             tight-tolerance arm 764976.
  confident  the eight most confidently classified test digits, r = 16, k in {6, 8, 10}, on-chart, fp64 --
             job 706721 (backup figure A: one digit the base model already handles is invisible).

Per example i (experimenter quantities need A0 and H; attacker quantities need only the release + public model):
  a_i = A0 h_i;  a~_i = a_i projected off span{a_j : j != i}, normalised;   sigma_i = ||B_T a~_i||       (record strength)
  ||C_i||  the traced imprint (B_T = sum_i C_i);   u_i = sum_t lr * ||softmax_t,i - y_i||                (accumulated update)
  certificate residual at the chart projection of the truth, two forms: ||C h|| / (||C||_F ||h||) (as asked) and the
  search objective ||C h|| / ||A_T h||.
  recovery: the certificate search re-run with the ORIGINAL start generator (seed + 31), every start saved; per example
  the landings (basin size), the closest approach, its error vs the chart truth and vs the raw image, and SSIM
  (kornia.metrics.ssim, window 3, on [0,1]).

  python -u -m experiments.record_strength.record_strength --cells letters --out results/record_strength/rs_letters.jsonl
  python -u -m experiments.record_strength.record_strength --cells confident --out results/record_strength/rs_confident.jsonl
  python -u -m experiments.record_strength.record_strength --plot results/record_strength/rs_*.jsonl
"""
import argparse, glob, json, math, os, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import train_release, git_hash
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch
from experiments.exact_inversion.new_class import load_emnist_letters, ExtendedHead
from experiments.exact_inversion.certificate import certificate, lm_cert
from experiments.exact_inversion.train_precision import release_in, FORMATS
from experiments.exact_inversion.margin_check import traced_release

torch.set_default_dtype(torch.float64)

# Saved landings per recorded image from the job rows, for the reproduction gate (printed beside the new counts).
SAVED = {
    ("letters", "fp64", 1e-12): dict(job=760909, landings={0: 108, 1: 10, 2: 11, 3: 8, 4: 6, 5: 7, 6: 7, 7: 35}),
    ("letters", "fp32", 1.2e-06): dict(job=760909, landings={0: 0, 1: 0, 2: 14, 3: 9, 4: 7, 5: 6, 6: 6, 7: 39}),
    ("letters", "fp32", 1e-12): dict(job=764976, landings={0: 94, 1: 4, 2: 8, 3: 6, 4: 6, 5: 6, 6: 4, 7: 36}),
    ("letters", "fp16", 0.0098): dict(job=760909, landings={i: 0 for i in range(8)}),
    ("letters", "bf16", 1e-12): dict(job=764976, landings={i: 0 for i in range(8)}),
    ("confident", 6, 1e-12): dict(job=706721, landings={0: 37, 1: 99, 2: 11, 4: 63, 5: 414, 6: 383, 7: 13}),
    ("confident", 8, 1e-12): dict(job=706721, landings={0: 10, 1: 21, 2: 9, 4: 106, 5: 4, 6: 211, 7: 3}),
    ("confident", 10, 1e-12): dict(job=706721, landings={1: 17, 2: 6, 5: 7, 6: 233, 7: 5}),
}


def trace_in(H, A0, W0, y, m, T, lr, dtype):
    """train_precision.release_in with the per-step residual norms recorded -- the SAME operations in the SAME order,
       so in `dtype` it must reproduce release_in's B_T bit for bit (gated by the caller)."""
    dev = H.device; N = H.shape[1]
    H = H.to(dtype); A = A0.to(dtype).clone(); W0 = W0.to(dtype); lr_ = torch.tensor(lr, dtype=dtype, device=dev)
    Y = torch.eye(m, device=dev, dtype=dtype)[y].T
    B = torch.zeros(m, A.shape[0], device=dev, dtype=dtype)
    C = torch.zeros(N, m, A.shape[0], device=dev, dtype=dtype)
    res = []
    for t in range(1, T + 1):
        AH = A @ H
        BAH = B @ AH
        z = W0 @ H + BAH
        R = torch.softmax(z, dim=0) - Y
        res.append(torch.linalg.norm(R.double(), dim=0))
        D = R / N
        gB = D @ AH.T
        gA = B.T @ D @ H.T
        C -= lr_ * torch.einsum("mi,ri->imr", D, AH)
        B, A = B - lr_ * gB, A - lr_ * gA
    return A.double(), B.double(), C.double(), torch.stack(res)


def record_strength(A0, H, B_T):
    """sigma_i = ||B_T a~_i|| with a~_i = a_i projected off span{a_j, j != i}, normalised; a_i = A0 h_i."""
    a = A0 @ H; N = a.shape[1]; out = []
    for i in range(N):
        others = torch.cat([a[:, :i], a[:, i + 1:]], 1)
        Q, _ = torch.linalg.qr(others)
        at = a[:, i] - Q @ (Q.T @ a[:, i]); at = at / torch.linalg.norm(at)
        out.append(float(torch.linalg.norm(B_T @ at)))
    return out


def ssim28(a, b):
    import kornia.metrics as km
    a = a.reshape(1, 1, 28, 28).clamp(0, 1).float(); b = b.reshape(1, 1, 28, 28).clamp(0, 1).float()
    return float(km.ssim(a, b, window_size=3).mean())


def rel(x, y): return float(torch.linalg.norm(x - y) / torch.linalg.norm(y))


def run_search(bb, chart, k, C, A_T, X_on, X_real, recorded, coord_std, public, n_starts, iters, seed, dev, log):
    """certificate.py Part B / train_precision Part B, start for start: objective ||C phi|| / ||A_T phi||, starts from
       N(0, coord_std) with generator seed + 31, LM 300 iterations, landing = image error < 1e-2 vs the on-chart truth."""
    N = X_on.shape[1]
    def fun(w):
        f = bb.phi(chart.psi(w.reshape(k, 1)))
        return (C @ f).reshape(-1) / torch.linalg.norm(A_T @ f)
    with torch.no_grad():
        feat_ref = float(torch.linalg.norm(A_T @ bb.phi(public[:256].T), dim=0).median())
    gs = torch.Generator().manual_seed(seed + 31); t0 = time.time(); runs = []; W = []
    for s in range(n_starts):
        w0 = (torch.randn(k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
        w, obj, it = lm_cert(fun, w0, iters)
        x_hat = chart.psi(w.reshape(k, 1))[:, 0]
        e_all = [rel(x_hat, X_on[:, i]) for i in range(N)]
        j_all = min(range(N), key=lambda i: e_all[i])
        j_rec = min(recorded, key=lambda i: e_all[i]) if recorded else j_all
        with torch.no_grad(): fr = float(torch.linalg.norm(A_T @ bb.phi(x_hat.reshape(-1, 1))) / feat_ref)
        runs.append(dict(objective=obj, iters=it, nearest_all=j_all, err_all=e_all[j_all], nearest_rec=j_rec, err_rec=e_all[j_rec],
                         landed_rec=bool(e_all[j_rec] < 1e-2), landed_any=bool(e_all[j_all] < 1e-2), feat_ratio=fr, degenerate=bool(fr < 0.05)))
        W.append(w.detach().cpu())
        if (s + 1) % 100 == 0: log(f"      {s + 1}/{n_starts} starts, {time.time() - t0:.0f}s, landed so far {sum(r['landed_rec'] for r in runs)}")
    return runs, torch.stack(W), time.time() - t0


def per_example_rows(tag, meta, runs, W, chart, k, X_on, X_real, y, H, A0, A_T, B_T, C, Np, imp, u, mar1, marT, recorded, dev):
    """One row per example: record strength, accumulated update, certificate residuals, recovery."""
    N = X_on.shape[1]; sig = record_strength(A0, H, B_T)
    with torch.no_grad():
        nC = float(torch.linalg.norm(C)); Ch = C @ H
        res_att = [float(torch.linalg.norm(Ch[:, i]) / (nC * torch.linalg.norm(H[:, i]))) for i in range(N)]
        res_obj = [float(torch.linalg.norm(Ch[:, i]) / torch.linalg.norm(A_T @ H[:, i])) for i in range(N)]
    rows = []
    for i in range(N):
        mine = [(s, r) for s, r in enumerate(runs) if r["nearest_all"] == i]
        land_rec = sum(1 for r in runs if r["landed_rec"] and r["nearest_rec"] == i) if i in recorded else 0
        land_any = sum(1 for r in runs if r["landed_any"] and r["nearest_all"] == i)
        row = dict(part="example", **meta, i=i, label=int(y[i]), recorded=bool(i in recorded), in_rowspace_at_tol=bool(res_obj[i] < 1e-3),
                   sigma_i=sig[i], imprint_i=float(imp[i]), imprint_rel_i=float(imp[i] / imp.max()), u_i=float(u[i]),
                   margin_t1=float(mar1[i]), margin_T=float(marT[i]), cert_res_attacker=res_att[i], cert_res_objective=res_obj[i],
                   landings=land_rec, landings_any=land_any, basin=land_rec / len(runs),
                   chart_floor_err=rel(X_on[:, i], X_real[:, i]), chart_floor_ssim=ssim28(X_on[:, i], X_real[:, i]), n_starts_nearest=len(mine))
        if mine:
            s_best, r_best = min(mine, key=lambda sr: sr[1]["err_all"])
            x_best = chart.psi(W[s_best].to(dev).reshape(k, 1))[:, 0]
            row.update(best_start=s_best, best_err_chart=r_best["err_all"], best_err_real=rel(x_best, X_real[:, i]), best_objective=r_best["objective"],
                       best_ssim_real=ssim28(x_best, X_real[:, i]), best_ssim_chart=ssim28(x_best, X_on[:, i]), best_landed=bool(r_best["err_all"] < 1e-2))
        else:
            row.update(best_start=None, best_err_chart=None, best_err_real=None, best_objective=None, best_ssim_real=None, best_ssim_chart=None, best_landed=False)
        rows.append(row)
    return rows, sig


def cell_letters(a, base, dev, emit, log):
    """train_precision.main's letters_a:32 cell, line for line."""
    fl = load_emnist_letters(a.data_root, "a")
    Ftr_t = torch.tensor(fl["train"][0], device=dev); Fte_t = torch.tensor(fl["test"][0], device=dev)
    g = torch.Generator().manual_seed(a.seed + 7); pf = torch.randperm(Fte_t.shape[0], generator=g)[:a.N]
    X_real = Fte_t[pf].T.contiguous(); y = torch.full((a.N,), 10, device=dev); bb = ExtendedHead(base, "zero", a.seed)
    k, r = 32, 64; chart = PCAChart(Ftr_t, k, dev); public = Ftr_t
    sigma0 = 1.0 / math.sqrt(base.n)
    g = torch.Generator().manual_seed(a.seed + 7); A0 = (sigma0 * torch.randn(r, bb.n, generator=g)).to(dev)
    W_all = chart.coords_of(X_real); X_on = chart.psi(W_all); H = bb.phi(X_on)
    coord_std = chart.coords_of(public[:10000].T).std(dim=1, keepdim=True)
    A_ref, B_ref = train_release(H, A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
    for dname in a.dtypes:
        dtype, eps = FORMATS[dname]
        A_T, B_T, Cimp, zero_frac, mar1, marT, feedback = release_in(H, A0, bb.W0, y, bb.m, a.T, a.lr, dtype)
        A_tr, B_tr, C_tr, res = trace_in(H, A0, bb.W0, y, bb.m, a.T, a.lr, dtype)
        trace_gate = float(torch.linalg.norm(B_tr - B_T)); nB = float(torch.linalg.norm(B_T))
        fp64_gate = float(torch.linalg.norm(B_T - B_ref) / torch.linalg.norm(B_ref))
        imp = torch.linalg.norm(Cimp.reshape(a.N, -1), dim=1); u = a.lr * res.sum(0)
        sB = torch.linalg.svdvals(B_T)
        recorded = [i for i in range(a.N) if imp[i] / imp.max() > 1e-12]
        tols = [1e-12] if dname == "fp64" else sorted({float(f"{10 * eps:.3g}"), 1e-12})
        log(f"# letters {dname}: ||B_T|| {nB:.4e}  trace gate {trace_gate:.1e}  vs fp64 {fp64_gate:.1e}  sigma_N/sigma_N+1 {float(sB[a.N-1]/sB[a.N]):.2e}")
        for tol in tols:
            C, Np, _ = certificate(A_T, B_T, tol)
            meta = dict(cell="letters", k=k, r=r, N=a.N, m=bb.m, T=a.T, lr=a.lr, seed=a.seed, train_dtype=dname, tol=tol, n_prime=Np, cert_line=r - Np,
                        rank_C=int(torch.linalg.matrix_rank(C, rtol=1e-10)))
            runs, W, sec = run_search(bb, chart, k, C, A_T, X_on, X_real, recorded, coord_std, public, a.starts_letters, a.iters, a.seed, dev, log)
            rows, sig = per_example_rows("letters", meta, runs, W, chart, k, X_on, X_real, y, H, A0, A_T, B_T, C, Np, imp, u, mar1, marT, recorded, dev)
            counts = {i: rows[i]["landings"] for i in range(a.N)}
            saved = SAVED.get(("letters", dname, tol))
            emit(dict(part="cell", **meta, B_T_norm=nB, B_T_sigma=[float(v) for v in sB[: a.N + 3]], sigma_min_topN=float(sB[a.N - 1]),
                      gap_sigmaN_over_sigmaN1=float(sB[a.N - 1] / sB[a.N]), trace_gate_abs=trace_gate, vs_fp64_release_rel=fp64_gate,
                      residual_entries_exactly_zero_frac=zero_frac, feedback=feedback, recorded=recorded, sigma_i=sig, imprint=[float(v) for v in imp],
                      u_i=[float(v) for v in u], starts=len(runs), landings=counts, landings_saved=(saved["landings"] if saved else None),
                      saved_job=(saved["job"] if saved else None), landings_match=(saved["landings"] == counts if saved else None),
                      frac_landed=sum(r["landed_rec"] for r in runs) / len(runs), n_degenerate=sum(r["degenerate"] for r in runs), sec=sec,
                      git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
            for row in rows: emit(row)
            if a.save_dir:
                torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), y=y.cpu(), A_T=A_T.cpu(), B_T=B_T.cpu(), A0=A0.cpu(), H=H.cpu(), C=C.cpu(),
                                imprint_C=Cimp.cpu(), res_trace=res.cpu(), W=W, runs=runs, rows=rows, meta=meta),
                           os.path.join(a.save_dir, f"letters_k32_{dname}_tol{tol:g}.pth"))


def cell_confident(a, base, dev, emit, log):
    """certificate.py's confident / on-chart cells at k in {6, 8, 10}, r = 16 (job 706721), line for line."""
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:50000], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    idx = torch.tensor(pick_batch("confident", base, Xte_t, yte_t, a.N, perm), device=dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]; bb = base; r = 16; sigma0 = 1.0 / math.sqrt(bb.n)
    for k in a.ks:
        chart = PCAChart(Xtr_t, k, dev); coord_std = chart.coords_of(Xtr_t[:10000].T).std(dim=1, keepdim=True)
        W_all = chart.coords_of(X_real); X_on = chart.psi(W_all); H = bb.phi(X_on)
        g = torch.Generator().manual_seed(a.seed + 7); A0 = (sigma0 * torch.randn(r, bb.n, generator=g)).to(dev)
        A_T, B_T = train_release(H, A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
        A_tr, B_tr, res, mar, Cimp = traced_release(H, A0, bb.W0, y, bb.m, a.T, a.lr)
        trace_gate = float(torch.linalg.norm(B_tr - B_T)); nB = float(torch.linalg.norm(B_T))
        imp = torch.linalg.norm(Cimp.reshape(a.N, -1), dim=1); u = a.lr * res.sum(0); mar1, marT = mar[0], mar[-1]
        sB = torch.linalg.svdvals(B_T); recorded = [i for i in range(a.N) if imp[i] / imp.max() > 1e-12]
        tol = 1e-12; C, Np, _ = certificate(A_T, B_T, tol)
        with torch.no_grad():
            obj_at = [float(torch.linalg.norm(C @ H[:, i]) ** 2 / torch.linalg.norm(A_T @ H[:, i]) ** 2) for i in range(a.N)]
        saved_pth = f"results/exact_inversion/step67_706721/cert_confident_on_k{k}.pth"
        saved_obj = torch.load(saved_pth, weights_only=False, map_location="cpu")["meta"]["objective_at_each_truth"] if os.path.exists(saved_pth) else None
        log(f"# confident k={k}: ||B_T|| {nB:.4e}  trace gate {trace_gate:.1e}  N'={Np}  recorded {recorded}  gap {float(sB[Np-1]/sB[Np]):.2e}")
        if saved_obj: log(f"#   objective at truth now  {['%.1e' % v for v in obj_at]}\n#   saved (706721)          {['%.1e' % v for v in saved_obj]}")
        meta = dict(cell="confident", k=k, r=r, N=a.N, m=bb.m, T=a.T, lr=a.lr, seed=a.seed, train_dtype="fp64", tol=tol, n_prime=Np, cert_line=r - Np,
                    rank_C=int(torch.linalg.matrix_rank(C, rtol=1e-10)))
        runs, W, sec = run_search(bb, chart, k, C, A_T, X_on, X_real, recorded, coord_std, Xtr_t, a.starts_confident, a.iters, a.seed, dev, log)
        rows, sig = per_example_rows("confident", meta, runs, W, chart, k, X_on, X_real, y, H, A0, A_T, B_T, C, Np, imp, u, mar1, marT, recorded, dev)
        counts = {i: rows[i]["landings"] for i in recorded}; saved = SAVED.get(("confident", k, tol))
        emit(dict(part="cell", **meta, B_T_norm=nB, B_T_sigma=[float(v) for v in sB[: a.N + 3]], sigma_min_topN=float(sB[Np - 1]),
                  gap_sigmaN_over_sigmaN1=float(sB[Np - 1] / sB[Np]), trace_gate_abs=trace_gate, recorded=recorded, sigma_i=sig,
                  imprint=[float(v) for v in imp], u_i=[float(v) for v in u], objective_at_truth=obj_at, objective_at_truth_saved=saved_obj,
                  starts=len(runs), landings=counts, landings_saved=(saved["landings"] if saved else None), saved_job=(saved["job"] if saved else None),
                  landings_match=(saved["landings"] == counts if saved else None), frac_landed=sum(r["landed_rec"] for r in runs) / len(runs),
                  n_degenerate=sum(r["degenerate"] for r in runs), sec=sec, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
        for row in rows: emit(row)
        if a.save_dir:
            torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), y=y.cpu(), A_T=A_T.cpu(), B_T=B_T.cpu(), A0=A0.cpu(), H=H.cpu(), C=C.cpu(),
                            imprint_C=Cimp.cpu(), res_trace=res.cpu(), W=W, runs=runs, rows=rows, meta=meta),
                       os.path.join(a.save_dir, f"confident_k{k}_fp64_tol{tol:g}.pth"))


# ----------------------------------------------------------------------------------------------------------- plots
def series_label(row):
    if row["cell"] == "confident": return f"confident digits, r=16, k={row['k']}"
    return f"letters, r=64, k=32, {row['train_dtype']}-trained, tol {row['tol']:g}"


def plot(files, fig_dir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    ex = [r for r in rows if r["part"] == "example"]
    key = lambda r: (r["cell"], r["k"], r["train_dtype"], r["tol"])
    groups = {}
    for r in ex: groups.setdefault(key(r), []).append(r)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    def scatter(ax, keys, xf, yf, floor=None, annotate=True, legend_loc="best"):
        n_inv = 0
        for j, kk in enumerate(keys):
            g = groups[kk]; c = colors[j % 8]; m = markers[j % 8]; lab = series_label(g[0])
            for r in g:
                x = xf(r); yv = yf(r)
                if x is None or yv is None: continue
                if floor is not None: x = max(x, floor)
                ax.scatter(x, yv, s=46, marker=m, facecolors=(c if r.get("best_landed", r["landings"] > 0) else "none"), edgecolors=c, linewidths=1.2, label=lab, zorder=3)
                lab = None
                if annotate and not r["recorded"]:                      # hollow marker = not landed; label only the invisible examples, staggered
                    ax.annotate("invisible", (x, yv), textcoords="offset points", xytext=(6, 4 + 9 * (n_inv % 4)), fontsize=8, color=c); n_inv += 1
        h, l = ax.get_legend_handles_labels(); seen = {}
        for hh, ll in zip(h, l): seen.setdefault(ll, hh)
        ax.legend(seen.values(), seen.keys(), fontsize=8, frameon=False, loc=legend_loc)
        ax.grid(True, which="major", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

    fp64_keys = [kk for kk in sorted(groups) if kk[2] == "fp64"]
    err = lambda r: r["best_err_chart"]; sig = lambda r: r["sigma_i"]
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter(ax, fp64_keys, sig, err, floor=1e-30); ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1e-2, color="gray", lw=0.8, ls="--"); ax.text(ax.get_xlim()[0], 1.3e-2, "landing threshold 1e-2", fontsize=8, color="gray")
    ax.set_xlabel(r"record strength $\sigma_i=\|B_T\tilde a_i\|$ (fp64-trained releases)"); ax.set_ylabel("closest approach: image error vs chart truth (filled = landed)")
    ax.set_title("Recovery error vs record strength, one point per private example", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "recovery_error_vs_sigma.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter(ax, fp64_keys, sig, lambda r: r["basin"], floor=1e-30); ax.set_xscale("log")
    ax.set_xlabel(r"record strength $\sigma_i=\|B_T\tilde a_i\|$ (fp64-trained releases)"); ax.set_ylabel("basin size: fraction of random starts landing on example i")
    ax.set_title("Basin size vs record strength", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "basin_vs_sigma.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter(ax, fp64_keys, lambda r: r["u_i"], sig, floor=1e-30); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"accumulated update $u_i=\sum_t \eta\,\|p_{t,i}-y_i\|_2$"); ax.set_ylabel(r"record strength $\sigma_i$")
    ax.set_title("Does the accumulated softmax residual predict the record strength?", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "sigma_vs_u.png"), dpi=200); plt.close(fig)

    let_keys = [kk for kk in sorted(groups, key=lambda kk: (list(FORMATS).index(kk[2]), -kk[3])) if kk[0] == "letters"]
    if let_keys:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        scatter(axes[0], let_keys, sig, err, floor=1e-30, annotate=False, legend_loc="center left"); axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].axhline(1e-2, color="gray", lw=0.8, ls="--")
        axes[0].text(axes[0].get_xlim()[0] * 1.05, 1.3e-2, "landing threshold 1e-2", fontsize=8, color="gray")
        axes[0].set_xlabel(r"$\sigma_i$ (release trained in the stated format)"); axes[0].set_ylabel("closest approach: image error vs chart truth (filled = landed)")
        scatter(axes[1], let_keys, sig, lambda r: r["basin"], floor=1e-30, annotate=False, legend_loc="upper left"); axes[1].set_xscale("log")
        axes[1].set_xlabel(r"$\sigma_i$ (release trained in the stated format)"); axes[1].set_ylabel("basin size")
        fig.suptitle("Same eight private letters, same chart (k=32): releases trained in fp64 / fp32 / bf16 / fp16 (hollow = not landed; bf16/fp16 truths are not in row(B_T) at either tolerance)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "letters_precision_overlay.png"), dpi=200); plt.close(fig)

    # the table for RESULT.md
    lines = ["| release | i | label | recorded | sigma_i | imprint | u_i | cert res (attacker) | cert res (objective) | landings | basin | best err (chart) | best err (raw) | SSIM vs raw | SSIM vs chart | chart-floor SSIM |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v, s="{:.2e}": ("--" if v is None else s.format(v))
    for kk in sorted(groups, key=lambda kk: (kk[0], kk[1], list(FORMATS).index(kk[2]), -kk[3])):
        for r in groups[kk]:
            lines.append(f"| {series_label(r)} | {r['i']} | {r['label']} | {'yes' if r['recorded'] else 'NO'} | {f(r['sigma_i'])} | {f(r['imprint_i'])} | {f(r['u_i'])} | "
                         f"{f(r['cert_res_attacker'])} | {f(r['cert_res_objective'])} | {r['landings']} | {r['basin']:.3f} | {f(r['best_err_chart'])} | {f(r['best_err_real'])} | "
                         f"{f(r['best_ssim_real'], '{:.3f}')} | {f(r['best_ssim_chart'], '{:.3f}')} | {r['chart_floor_ssim']:.3f} |")
    cells = [r for r in rows if r["part"] == "cell"]
    lines += ["", "| release | N' | rank C | sigma_N (top-N min) | gap sigma_N/sigma_N+1 | starts | landed | landings now | landings saved (job) | match |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(cells, key=lambda r: (r["cell"], r["k"], list(FORMATS).index(r["train_dtype"]), -r["tol"])):
        lines.append(f"| {series_label(r)} | {r['n_prime']} | {r['rank_C']} | {r['sigma_min_topN']:.2e} | {r['gap_sigmaN_over_sigmaN1']:.2e} | {r['starts']} | {r['frac_landed']:.3f} | "
                     f"{r['landings']} | {r['landings_saved']} ({r['saved_job']}) | {r['landings_match']} |")
    open(os.path.join(fig_dir, "table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["letters", "confident"])
    ap.add_argument("--dtypes", nargs="*", default=["fp64", "fp32", "bf16", "fp16"])
    ap.add_argument("--ks", nargs="*", type=int, default=[6, 8, 10])
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--starts-letters", type=int, default=500); ap.add_argument("--starts-confident", type=int, default=2000)
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default="results/record_strength")
    ap.add_argument("--plot", nargs="*", default=None, help="jsonl files -> figures/record_strength/*.png + table.md (no compute)")
    ap.add_argument("--fig-dir", default="figures/record_strength")
    a = ap.parse_args()
    if a.plot is not None:
        files = sorted(set(sum([glob.glob(p) for p in a.plot], []))); print("# plotting from", files); plot(files, a.fig_dir); return
    dev = torch.device(a.device); os.makedirs(a.save_dir, exist_ok=True)
    log = lambda s: print(s, flush=True)
    def emit(row):
        print(json.dumps({k: v for k, v in row.items()}), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    base = TrainedBackbone(a.model, dev, "gelu")
    log(f"# record_strength cells={a.cells} git={git_hash()} host={socket.gethostname()} device={dev}")
    if "letters" in a.cells: cell_letters(a, base, dev, emit, log)
    if "confident" in a.cells: cell_confident(a, base, dev, emit, log)


if __name__ == "__main__":
    main()
