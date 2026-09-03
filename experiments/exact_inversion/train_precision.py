"""Training-precision cell (RESULTS Step 24 continued): the release loop itself run in fp64 / fp32 / bf16 / fp16.

Question. The headline certificate cell (confident batch, r = 64, k = 32) was recovered from an FP64 release of norm
7.6e-18: the strong model classifies the k = 32 projections with margins 42-60 and records only their exp(-margin)
softmax residuals. Storage precision was tested by quantising the FP64 release (job 753371). This script asks what
the TRAINING arithmetic does to the release: the same recipe (T steps of full-batch SGD on the LoRA pair, same seed,
same A0, same projected batch) with every tensor cast to the format and the loop run in it.

Pre-registered (RESULTS "Step 24 continued", corrected form): the residual's off-class entries p_j = exp(z_j - z_y)/Z
survive down to the format's subnormal floor (fp32/bf16: margins to ~100; fp16: to ~16.6); unit roundoff removes
only the own-class entry R_y = p_y - 1 once sum_j p_j < eps (one row of the imprint, not its direction A0 h_i). The
release never feeds back into the logits when it is 1e-18 against logits of ~40 (below their ulp even in FP64), so
A_T = A0 to working precision and B_T is the closed-form one-step gradient x T. Predictions: confident k=32 -> ~7.6e-18
(own-class rows lost, ~sqrt2 lower) under fp32 and bf16, EXACTLY 0 under fp16 (a nonzero sqrt2-down fp32 release is
confirmation, an exactly-zero one the falsifier); control k=32 (margins 13.6-25) -> ~7.9e-6 under fp32/bf16, rank falls
under fp16; confident k=8 -> ~0.47 in all three. The decisive arm (yoado-ed): a NEW CLASS (EMNIST letter 'a' on the
strong MNIST model, zero-initialised eleventh head row) has negative margins and O(1) residuals at every k, so its
release should survive every format; if the certificate search recovers it from an fp32-trained release at k = 32,
the exposure is not an artefact of exact arithmetic. Objection pre-empted (yoado-ed): at 1e-18 the adapter never
moves the logits, so the headline is "one gradient step in disguise" -- every row therefore carries the FEEDBACK
||B_T A_T H|| / ||z|| and the batch's margin change from t = 1 to T; the control and letter cells are the companions
where the adapter does move.

Per (cell, dtype) a Part-A row: ||B_T||, spectrum, rank at 1e-10 / 1e-6, absolute imprints (traced in the format),
||sum_i C_i - B_T|| (the imprint-sum mismatch, which must scale with the signal, not with O(1) intermediates -- yoado-6e),
margins of the projected batch at t = 1 and t = T in that format, the fraction of residual entries that are EXACTLY
zero, feedback, A_T's distance from A0, and the certificate residual at each truth (FP64, from the upcast release) at
tol 1e-12 and at the noise-matched tolerance. Optionally a Part-B row: the random-start certificate search with
certificate.py's objective ||C phi|| / ||A_T phi||, start distribution (public coordinate std) and landing threshold
(1e-2 on-chart image error vs the nearest recorded image).

The fp64 arm is a gate: the copied loop must reproduce lora_exact_inversion.train_release to roundoff (assertion
floored at 1e-10*||B_T|| + 1e-13, LESSONS 2026-09-03). No shared module is edited (jobs 753371/753886 are
multi-invocation and hold certificate.py / lora_exact_inversion.py).
"""
import argparse, json, math, os, socket, sys, time
import torch
from experiments.exact_inversion.lora_exact_inversion import train_release, git_hash, simulate_sgd_reduced, qr_canon, invert_lm
import torch.func as tf
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch
from experiments.exact_inversion.new_class import load_emnist_letters, ExtendedHead
from experiments.exact_inversion.certificate import certificate, lm_cert

torch.set_default_dtype(torch.float64)
FORMATS = {"fp64": (torch.float64, 2.2e-16), "fp32": (torch.float32, 1.2e-7), "bf16": (torch.bfloat16, 7.8e-3), "fp16": (torch.float16, 9.8e-4)}


def release_in(H, A0, W0, y, m, T, lr, dtype):
    """lora_exact_inversion.train_release's SGD branch (as copied in margin_check.traced_release), run in `dtype`.
       Returns the FP64 upcast of A_T, B_T, the per-image imprints C_i (traced in the format), the fraction of residual
       entries that were exactly zero, the margins at t = 1 and t = T, and the feedback ||B A H|| / ||z|| at t = T."""
    dev = H.device; N = H.shape[1]; ar = torch.arange(N, device=dev)
    H = H.to(dtype); A = A0.to(dtype).clone(); W0 = W0.to(dtype); lr_ = torch.tensor(lr, dtype=dtype, device=dev)
    Y = torch.eye(m, device=dev, dtype=dtype)[y].T
    B = torch.zeros(m, A.shape[0], device=dev, dtype=dtype)
    C = torch.zeros(N, m, A.shape[0], device=dev, dtype=dtype)
    zeros = 0; total = 0; mar = []; feedback = float("nan")
    for t in range(1, T + 1):
        AH = A @ H
        BAH = B @ AH
        z = W0 @ H + BAH
        R = torch.softmax(z, dim=0) - Y
        zeros += int((R == 0).sum()); total += R.numel()
        if t in (1, T):
            zy = z[y, ar]; zo = z.clone(); zo[y, ar] = -float("inf"); mar.append((zy - zo.max(0).values).double())
        if t == T:
            feedback = float(torch.linalg.norm(BAH.double()) / torch.linalg.norm(z.double()))
        D = R / N
        gB = D @ AH.T
        gA = B.T @ D @ H.T
        C -= lr_ * torch.einsum("mi,ri->imr", D, AH)
        B, A = B - lr_ * gB, A - lr_ * gA
    return A.double(), B.double(), C.double(), zeros / total, mar[0], mar[1], feedback


def recipe_route(chart, bb, X_real, X_on, y, A0, A_T, B_T, a, dev, g):
    """vae_chart.invert_cell's recipe route ("cell a": the truth is on the chart), copied so the RELEASE can be the one
       trained in a low-precision format while the SIMULATOR stays FP64 -- the attacker who knows the recipe but not
       the arithmetic it was run in. Near start (init_noise) + restarts, LM with the autograd Jacobian; the residual at
       the truth is the arithmetic mismatch floor, sigma_min at the truth the local identifiability."""
    m, n = bb.m, bb.n; k = a.k; N = a.N; nW = k * N
    H = bb.phi(X_on); W_true = chart.coords_of(X_real)
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T)

    def res_vec(v):
        Wc = v[:nW].reshape(k, N); aux = v[nW:].reshape(a.r, N)
        Bs, Xis, Uc = simulate_sgd_reduced(bb.phi(chart.psi(Wc)), aux, bb.W0, y, m, a.T, a.lr, 0.0)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])

    U_true, _ = qr_canon(H); v0 = torch.cat([W_true.reshape(-1), (A0 @ U_true).reshape(-1)]).detach()
    sv = torch.linalg.svdvals(tf.jacfwd(res_vec)(v0).detach())
    smin, smax = float(sv[-1]), float(sv[0]); res_truth = float(torch.linalg.norm(res_vec(v0)))
    W_init = W_true + a.init_noise * torch.randn(k, N, generator=g).to(dev) * W_true.std()
    with torch.no_grad():
        Uc, _ = qr_canon(bb.phi(chart.psi(W_init))); Xinit = A_T @ Uc

    class Adapter:
        psi = staticmethod(chart.psi)
        features_from_latents = staticmethod(lambda Wc: bb.phi(chart.psi(Wc)))
    args = argparse.Namespace(m=m, T=a.T, lr=a.lr, wd=0.0, release="sgd", seed=a.seed,
                              restarts=a.restarts, restart_noise=0.1, lm_iters=a.lm_iters, lm_lambda=1e-2,
                              lm_scale="identity", stage_x=0, jac="fwd", solver="lm", outer=30, lbfgs_iter=20)
    t0 = time.time()
    W_hat, aux, resid, sec, nrs, diag = invert_lm(Adapter, A_T, B_T, bb.W0, y, args, W_init, Xinit, lambda s: None)
    X_hat = chart.psi(W_hat)
    e_chart = torch.linalg.norm(X_hat - X_on, dim=0) / torch.linalg.norm(X_on, dim=0)
    e_real = torch.linalg.norm(X_hat - X_real, dim=0) / torch.linalg.norm(X_real, dim=0)
    return dict(jac_sigma_min_truth=smin, jac_sigma_max_truth=smax, res_at_truth=res_truth, residual=resid,
                err_vs_chart_per_image=[float(v) for v in e_chart], err_vs_chart_max=float(e_chart.max()), err_vs_chart_median=float(e_chart.median()),
                err_vs_REAL_max=float(e_real.max()), err_vs_REAL_median=float(e_real.median()), start_noise=a.init_noise,
                seconds=time.time() - t0, **diag), X_hat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--cells", nargs="*", default=["letters_a:32", "confident:32", "mnist_control:32", "confident:8"], help="set:k; sets: confident, "
                    "mnist_control (first digit of each class in the seed permutation), letters_<x> (EMNIST letter x as an 11th class, zero head row), "
                    "letters_<x>_random (the same with a Gaussian head row at the RMS norm of the digit rows -- the robustness arm)")
    ap.add_argument("--dtypes", nargs="*", default=["fp64", "fp32", "bf16", "fp16"])
    ap.add_argument("--partb-dtypes", nargs="*", default=["fp64", "fp32", "fp16"], help="run the random-start certificate search from these releases")
    ap.add_argument("--partb-cells", nargs="*", default=["letters_a:32", "confident:32", "mnist_control:32"])
    ap.add_argument("--r", type=int, default=64); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--random-starts", type=int, default=500); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--max-np", type=int, default=11)
    ap.add_argument("--partb-tol", type=float, default=None, help="certificate tolerance for the Part-B search from a non-fp64 release; default = "
                    "noise-matched 10*eps (job 760909/760912); pass 1e-12 for the tight tolerance (the noise rank), which storage job 753371 "
                    "showed recovers MORE images")
    ap.add_argument("--recipe-cells", nargs="*", default=[], help="run the RECIPE route (FP64 simulator, near start) against the release trained in each format")
    ap.add_argument("--init-noise", type=float, default=0.10); ap.add_argument("--restarts", type=int, default=1); ap.add_argument("--lm-iters", type=int, default=600)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    base = TrainedBackbone(a.model, dev, "gelu")
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(base.n)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    if a.save_dir: os.makedirs(a.save_dir, exist_ok=True)
    print(f"# train_precision  model={a.model}  cells={a.cells}  dtypes={a.dtypes}  r={a.r}  git={git_hash()}  device={dev}", flush=True)

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    for cell in a.cells:
        sname, k = cell.split(":"); k = int(k)
        if sname.startswith("letters_"):
            init = "random" if sname.endswith("_random") else "zero"; letter = sname.split("_")[1][0]
            fl = load_emnist_letters(a.data_root, letter)
            Ftr_t = torch.tensor(fl["train"][0], device=dev); Fte_t = torch.tensor(fl["test"][0], device=dev)
            g = torch.Generator().manual_seed(a.seed + 7); pf = torch.randperm(Fte_t.shape[0], generator=g)[:a.N]
            X_real = Fte_t[pf].T.contiguous(); y = torch.full((a.N,), 10, device=dev); bb = ExtendedHead(base, init, a.seed)
            chart = PCAChart(Ftr_t, k, dev); chart_name = "new_pca"; public = Ftr_t
        else:
            if sname == "mnist_control":
                idx, seen = [], set()
                for i in perm.tolist():
                    if int(yte[i]) not in seen: idx.append(i); seen.add(int(yte[i]))
                    if len(idx) == a.N: break
            else:
                idx = pick_batch(sname, base, Xte_t, yte_t, a.N, perm)
            idx = torch.tensor(idx, device=dev); X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]; bb = base
            chart = PCAChart(Xtr_t, k, dev); chart_name = "old_pca"; public = Xtr_t
        g = torch.Generator().manual_seed(a.seed + 7); A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g)).to(dev)
        W_all = chart.coords_of(X_real); X_on = chart.psi(W_all); H = bb.phi(X_on)
        coord_std = chart.coords_of(public[:10000].T).std(dim=1, keepdim=True)
        repr_err = (torch.linalg.norm(X_on - X_real, dim=0) / torch.linalg.norm(X_real, dim=0))
        A_ref, B_ref = train_release(H, A0, bb.W0, y, bb.m, a.T, a.lr, "sgd")
        for dname in a.dtypes:
            dtype, eps = FORMATS[dname]; t0 = time.time()
            try:
                A_T, B_T, C, zero_frac, mar1, marT, feedback = release_in(H, A0, bb.W0, y, bb.m, a.T, a.lr, dtype)
            except Exception as e:                                       # a format the device cannot run (recorded, not hidden)
                emit(dict(part="A", set=sname, k=k, train_dtype=dname, error=repr(e), git=git_hash(), host=socket.gethostname())); continue
            nB = float(torch.linalg.norm(B_T)); nBref = float(torch.linalg.norm(B_ref))
            if dname == "fp64":                                          # gate: the copied loop reproduces train_release
                mis = float(torch.linalg.norm(B_T - B_ref))
                assert mis <= 1e-10 * nBref + 1e-13, f"fp64 loop vs train_release: abs {mis:.3e}, ||B_T|| {nBref:.3e}"
            imp = torch.linalg.norm(C.reshape(a.N, -1), dim=1)
            sB = torch.linalg.svdvals(B_T) if nB > 0 else torch.zeros(min(B_T.shape), device=dev)
            rank10 = int((sB > 1e-10 * sB[0]).sum()) if nB > 0 else 0; rank6 = int((sB > 1e-6 * sB[0]).sum()) if nB > 0 else 0
            rel_to_fp64 = float(torch.linalg.norm(B_T - B_ref) / nBref) if nBref > 0 else float("nan")
            rowA = dict(part="A", set=sname, chart=chart_name, head_init=(bb.new_row_init if hasattr(bb, "new_row_init") else "n/a"), k=k, N=a.N, r=a.r, m=bb.m, T=a.T, lr=a.lr, seed=a.seed, y=y.tolist(), train_dtype=dname,
                        unit_roundoff=eps, B_T_norm=nB, B_T_norm_fp64=nBref, B_T_rel_dev_from_fp64=rel_to_fp64, B_T_sigma=[float(v) for v in sB],
                        rank_B_T_1e10=rank10, rank_B_T_1e6=rank6, imprint_abs=[float(v) for v in imp],
                        imprint_rel=[float(v / imp.max()) if float(imp.max()) > 0 else 0.0 for v in imp],
                        imprint_sum_mismatch_abs=float(torch.linalg.norm(C.sum(0) - B_T)),
                        residual_entries_exactly_zero_frac=zero_frac, margin_t1=[float(v) for v in mar1], margin_T=[float(v) for v in marT],
                        feedback_BAH_over_z_at_T=feedback, A_T_minus_A0_rel=float(torch.linalg.norm(A_T - A0) / torch.linalg.norm(A0)),
                        chart_repr_err_median=float(repr_err.median()), sec=time.time() - t0, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
            tol_nm = 10 * eps
            for tname, tol in [("1e-12", 1e-12), ("noise_matched", tol_nm)]:
                if nB == 0:
                    rowA[f"Np_{tname}"] = 0; rowA[f"cert_residual_{tname}"] = [float("nan")] * a.N; continue
                Cc, Np, _ = certificate(A_T, B_T, tol)
                res = torch.linalg.norm(Cc @ H, dim=0) / torch.linalg.norm(A_T @ H, dim=0)
                rowA[f"Np_{tname}"] = Np; rowA[f"cert_line_{tname}"] = a.r - Np; rowA[f"cert_residual_{tname}"] = [float(v) for v in res]
            emit(rowA)
            if cell in a.recipe_cells and nB > 0:                         # ---- recipe route against THIS release (yoado-6e's probe)
                a.k = k
                rr, X_hat = recipe_route(chart, bb, X_real, X_on, y, A0, A_T, B_T, a, dev, torch.Generator().manual_seed(a.seed + 11))
                rowR = dict(part="R", set=sname, chart=chart_name, k=k, N=a.N, r=a.r, m=bb.m, seed=a.seed, train_dtype=dname, simulator="fp64",
                            B_T_rel_dev_from_fp64=rel_to_fp64, **rr, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                emit(rowR)
                if a.save_dir:
                    torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat=X_hat.detach().cpu(), meta=rowR),
                               os.path.join(a.save_dir, f"recipe_{sname}_k{k}_{dname}.pth"))
            # ---- Part B: random-start certificate search from THIS release (certificate.py's objective, starts, threshold)
            if cell in a.partb_cells and dname in a.partb_dtypes and nB > 0:
                tol = (a.partb_tol if a.partb_tol is not None else tol_nm) if dname != "fp64" else 1e-12
                Cc, Np, _ = certificate(A_T, B_T, tol)
                if Np > a.max_np: emit(dict(part="B", set=sname, k=k, train_dtype=dname, skipped=f"N'={Np} > max-np")); continue
                def fun(w):
                    f = bb.phi(chart.psi(w.reshape(k, 1)))
                    return (Cc @ f).reshape(-1) / torch.linalg.norm(A_T @ f)
                with torch.no_grad():
                    feat_ref = float(torch.linalg.norm(A_T @ bb.phi(public[:256].T), dim=0).median())
                    cert_res_truth = torch.linalg.norm(Cc @ H, dim=0) / torch.linalg.norm(A_T @ H, dim=0)
                recorded = [i for i in range(a.N) if imp[i] / imp.max() > 1e-12]
                gs = torch.Generator().manual_seed(a.seed + 31); t1 = time.time(); runs = []; first = {}
                for s in range(a.random_starts):
                    w0 = (torch.randn(k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
                    w, obj, it = lm_cert(fun, w0, a.iters)
                    x_hat = chart.psi(w.reshape(k, 1))[:, 0]
                    e_all = {i: float(torch.linalg.norm(x_hat - X_on[:, i]) / torch.linalg.norm(X_on[:, i])) for i in recorded}
                    j, e = min(e_all.items(), key=lambda kv: kv[1])
                    with torch.no_grad(): fr = float(torch.linalg.norm(A_T @ bb.phi(x_hat.reshape(-1, 1))) / feat_ref)
                    landed = bool(e < 1e-2)
                    runs.append(dict(objective=obj, iters=it, nearest=j, err=e, landed=landed, degenerate=bool(fr < 0.05)))
                    if landed and j not in first: first[j] = x_hat.detach().cpu()
                valid = [d for d in runs if not d["degenerate"]]
                best = min(valid, key=lambda d: d["objective"]) if valid else None
                counts = {str(i): sum(1 for d in runs if d["landed"] and d["nearest"] == i) for i in recorded}
                # closest approach per recorded image over ALL starts: a "not found" at a residual near the 1e-2 bar is a degraded
                # recovery, not a miss (yoado-ed) -- report the image error, not only the pass/fail
                min_err = {str(i): min([d["err"] for d in runs if d["nearest"] == i] or [float("nan")]) for i in recorded}
                rowB = dict(part="B", set=sname, chart=chart_name, k=k, N=a.N, r=a.r, m=bb.m, seed=a.seed, train_dtype=dname, cert_tol=tol, n_prime=Np,
                            cert_line=a.r - Np, below_cert_line=bool(k < a.r - Np), random_starts=a.random_starts, iters=a.iters,
                            cert_residual_at_truth=[float(v) for v in cert_res_truth], recorded=recorded,
                            frac_starts_on_a_recorded_image=sum(d["landed"] for d in runs) / len(runs),
                            frac_starts_at_floor=sum(d["objective"] <= 1e-20 for d in runs) / len(runs),
                            recorded_images_found=sorted(int(i) for i, c in counts.items() if c > 0), landings_per_recorded_image=counts,
                            min_err_per_recorded_image=min_err,
                            argmin_objective=(best["objective"] if best else None), argmin_landed_on_recorded=(best["landed"] if best else None),
                            argmin_err_vs_nearest_recorded=(best["err"] if best else None), n_degenerate_starts=len(runs) - len(valid),
                            objective_median=float(torch.tensor([d["objective"] for d in runs]).median()), sec=time.time() - t1,
                            git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
                emit(rowB)
                if a.save_dir:
                    torch.save(dict(x_real=X_real.cpu(), x_chart=X_on.cpu(), x_hat_per_recorded=first, meta=rowB),
                               os.path.join(a.save_dir, f"trainprec_{sname}_k{k}_{dname}.pth"))


if __name__ == "__main__":
    main()
