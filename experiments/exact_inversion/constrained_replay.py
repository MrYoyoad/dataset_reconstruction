"""Certificate-CONSTRAINED replay: does the certificate's zero set meet the replay route's basin? (Yoad, via yoado-cd.)

The two routes have opposite failure modes. The certificate is recipe-free and needs no start — random starts land
on recorded images — but it pins only r - N' equations per image, so below its line it gives class-level fidelity.
Replay pins (m-1) + r - N per image, enough for recognisable images, but no cell in this study has ever reached its
floor from an attacker-buildable start: its binding constraint is the initialiser.

This joins them. At low rank the band between the two lines is where the certificate alone is coarse and replay
would be sharp. The attacker takes ~500 random certificate landings as starts, and runs the replay LM with every
step CONFINED to the certificate's zero manifold {w : C phi(psi(w)) = 0} -- the set the certificate already knows
the truth lies in. Concretely, at each iterate the LM step in the latent block is projected onto the null space of
Jg = d/dw [C phi(psi(w))], and a Gauss-Newton correction -Jg^+ g pulls the iterate back onto the manifold, so the
search never leaves the set the certificate certifies while the replay residual drives it toward the truth.

Pre-registered (yoado-cd, before any row is read):
  (a) constrained replay reaches the floor AT THE TRUTH from those starts -> the chain closes the low-rank attack:
      the certificate is a start generator for replay, and no prior is needed;
  (b) it stalls (residual not zero) or lands at the floor on a wrong image (alias) -> the certificate's zero set
      does not meet replay's basin, and the start must come from a prior.
The two outcomes in (b) are reported separately, never merged (ground rule 4). `fwd_check` is read first: until the
simulator reproduces the release at the truth to machine precision, no later number means anything.

Controls in the same job: the SAME replay LM from the SAME landings but UNCONSTRAINED (does the constraint help, or
would the landings have sufficed?), and from random public-scale starts (the attacker's baseline, expected to fail).
"""
import argparse, json, math, os, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import (train_release, simulate_sgd_reduced, qr_canon, git_hash)
from experiments.exact_inversion.trained_backbone import TrainedBackbone, PCAChart, read_idx
from experiments.exact_inversion.subset_and_ood import pick_batch, release_and_imprints, margins_of
from experiments.exact_inversion.certificate import certificate, lm_cert

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--set", default="confident", help="batch composition; the recorded member is the low-margin one")
    ap.add_argument("--ks", nargs="*", type=int, default=[8, 10, 12, 14])
    ap.add_argument("--r", type=int, default=8); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cert-starts", type=int, default=500); ap.add_argument("--cert-iters", type=int, default=300)
    ap.add_argument("--n-landings", type=int, default=8, help="distinct certificate landings carried into replay")
    ap.add_argument("--lm-iters", type=int, default=300); ap.add_argument("--lm-lambda", type=float, default=1e-2)
    ap.add_argument("--arms", nargs="*", default=["d0", "constrained", "unconstrained", "random", "null"])
    ap.add_argument("--d0-steps", nargs="*", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
                    help="D0: distances along Z_C from the truth (relative to the latent std) at which replay is retried")
    ap.add_argument("--d0-min-radius", type=float, default=0.05, help="D0 gate: if replay's in-manifold basin is below "
                    "this, D1/D2 do not launch -- no handoff from certificate landings can work (plan section 3)")
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None); ap.add_argument("--save-dir", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev); yte_t = torch.tensor(yte, device=dev)
    bb = TrainedBackbone(a.model, dev, "gelu")
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(bb.n)
    perm = torch.randperm(Xte_t.shape[0], generator=torch.Generator().manual_seed(a.seed + 7))
    if a.save_dir: os.makedirs(a.save_dir, exist_ok=True)
    print(f"# constrained_replay model={a.model} set={a.set} r={a.r} N={a.N} ks={a.ks} git={git_hash()} dev={dev}", flush=True)

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    idx = torch.tensor(pick_batch(a.set, bb, Xte_t, yte_t, a.N, perm), device=dev)
    X_real = Xte_t[idx].T.contiguous(); y = yte_t[idx]

    for k in a.ks:
        a.k = k
        chart = PCAChart(Xtr_t, k, dev)
        W_all = chart.coords_of(X_real); X_on = chart.psi(W_all); H = bb.phi(X_on)
        coord_std = chart.coords_of(Xtr_t[:10000].T).std(dim=1, keepdim=True)
        g0 = torch.Generator().manual_seed(a.seed + 7); A0 = (a.sigma0 * torch.randn(a.r, bb.n, generator=g0)).to(dev)
        A_T, B_T, imp, sB, _ = release_and_imprints(bb, X_on, y, A0, a)
        C, Np, S = certificate(A_T, B_T)
        mar, _ = margins_of(bb, X_on, y)
        rec = [i for i in range(a.N) if float(imp[i] / imp.max()) > 1e-12]
        cert_res = (torch.linalg.norm(C @ H, dim=0) / torch.linalg.norm(A_T @ H, dim=0))
        emit(dict(part="A", set=a.set, k=k, r=a.r, N=a.N, m=bb.m, T=a.T, lr=a.lr, seed=a.seed, n_prime=Np,
                  cert_line=a.r - Np, replay_line=bb.m + a.r - len(rec), below_cert_line=bool(k < a.r - Np),
                  in_band=bool(k >= a.r - Np and k < bb.m + a.r - len(rec)), recorded=rec, y=y.tolist(),
                  margins=[float(v) for v in mar], imprint_rel=[float(v / imp.max()) for v in imp],
                  B_T_norm=float(torch.linalg.norm(B_T)), B_T_sigma=[float(v) for v in sB],
                  cert_residual_per_image=[float(v) for v in cert_res], git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
        # CERTIFICATE GATE (yoado-b9): the band premise {rho=0} subset {Ch=0} is numerical, not formal. If the
        # certificate does not vanish at the truth in this cell, Z_C does not contain the truth and every branch of
        # the pre-registration is void -- the analogue of fwd_check, for the constraint rather than the simulator.
        cert_at_truth = float(cert_res[top])
        print(f"  [k={k}] certificate gate at the truth: ||Ch||/||A_T h|| = {cert_at_truth:.3e}", flush=True)
        if cert_at_truth > 1e-6:
            emit(dict(part="GATE", set=a.set, k=k, r=a.r, n_prime=Np, cert_at_truth=cert_at_truth, passed=False,
                      note="Z_C does not contain the truth at this k; chain branches void", git=git_hash()))
            print(f"  [k={k}] GATE FAILED -- skipping (not scored as a replay outcome)", flush=True); continue
        emit(dict(part="GATE", set=a.set, k=k, r=a.r, n_prime=Np, cert_at_truth=cert_at_truth, passed=True,
                  fwd_check_pending=True, git=git_hash()))
        if Np != 1:
            print(f"  [k={k}] N'={Np} != 1: the cell is not the one-image band; running anyway, rows carry N'", flush=True)

        top = int(torch.argmax(imp))                                    # the recorded (low-margin) image
        nW = k

        # ---- the two maps: g (certificate, r x 1 per image) and the replay residual
        def g_of(w):                                                    # certificate map at one image
            f = bb.phi(chart.psi(w.reshape(k, 1)))
            return (C @ f).reshape(-1) / torch.linalg.norm(A_T @ f)

        def replay_res(v):                                              # v = [w (k), X (r*Np)]
            w = v[:nW].reshape(k, 1); Xc = v[nW:].reshape(a.r, Np)
            Hc = bb.phi(chart.psi(w))
            Bs, Xis, Uc = simulate_sgd_reduced(Hc, Xc, bb.W0, y[top:top + 1], bb.m, a.T, a.lr * Np / a.N, 0.0)
            return torch.cat([((Bs - B_T) / torch.linalg.norm(B_T)).reshape(-1),
                              ((Xis - A_T @ Uc) / torch.linalg.norm(A_T)).reshape(-1)])

        # fwd_check: the simulator at the TRUE latents and the true span-block (read first, ground rule 5)
        w_true = W_all[:, top].reshape(k, 1)
        H_true1 = bb.phi(chart.psi(w_true)); U_true, _ = qr_canon(H_true1); X_true = (A0 @ U_true)
        v_true = torch.cat([w_true.reshape(-1), X_true.reshape(-1)])
        with torch.no_grad(): fwd = float(torch.linalg.norm(replay_res(v_true)))
        sv_truth = torch.linalg.svdvals(tf.jacfwd(replay_res)(v_true).detach())
        smin_truth, smax_truth = float(sv_truth[-1]), float(sv_truth[0])
        print(f"  [k={k}] fwd_check at the truth (subset recipe, lr*N'/N): {fwd:.3e}", flush=True)

        # ---- certificate landings from random starts
        def run(v0, constrained, tag, start_err):
            v = v0.clone(); jac = tf.jacfwd(replay_res)
            with torch.no_grad(): F = replay_res(v)
            fval = float(F @ F); lam = a.lm_lambda; t1 = time.time(); used = 0; trace = [fval]
            for it in range(a.lm_iters):
                J = jac(v).detach(); JtJ = J.T @ J; JtF = J.T @ F
                accepted = False
                for _ in range(12):
                    step = torch.linalg.solve(JtJ + lam * torch.eye(JtJ.shape[0], device=dev), JtF)
                    if constrained:                                     # project the latent block onto null(Jg)
                        Jg = tf.jacfwd(g_of)(v[:nW]).detach()
                        _, _, Vh = torch.linalg.svd(Jg, full_matrices=True)
                        nullsp = Vh[Jg.shape[0]:].T if Jg.shape[0] < k else None
                        if nullsp is not None and nullsp.shape[1] > 0:
                            step[:nW] = nullsp @ (nullsp.T @ step[:nW])
                        else:
                            step[:nW] = 0.0
                    vn = v - step
                    if constrained:                                     # pull back onto the manifold (one GN step)
                        with torch.no_grad(): gv = g_of(vn[:nW])
                        Jg2 = tf.jacfwd(g_of)(vn[:nW]).detach()
                        vn = vn.clone(); vn[:nW] = vn[:nW] - torch.linalg.lstsq(Jg2, gv.unsqueeze(1)).solution.reshape(-1)
                    with torch.no_grad(): Fn = replay_res(vn)
                    if float(Fn @ Fn) < fval:
                        v, F, fval = vn, Fn, float(Fn @ Fn); lam = max(lam / 3, 1e-15); accepted = True; break
                    lam *= 5
                used = it + 1; trace.append(fval)
                if fval < 1e-30 or not accepted or lam > 1e12: break
            xh = chart.psi(v[:nW].reshape(k, 1))[:, 0]
            e_on = float(torch.linalg.norm(xh - X_on[:, top]) / torch.linalg.norm(X_on[:, top]))
            e_all = [float(torch.linalg.norm(xh - X_on[:, i]) / torch.linalg.norm(X_on[:, i])) for i in range(a.N)]
            e_raw = float(torch.linalg.norm(xh - X_real[:, top]) / torch.linalg.norm(X_real[:, top]))
            with torch.no_grad(): gnorm = float(torch.linalg.norm(g_of(v[:nW])))
            # Pre-registered thresholds (yoado-b9): the cell is FP64 throughout, so the absolute form applies --
            # branch 1 is OBJECTIVE <= 1e-28 (not `residual`, which is its square root and floors at 1e-15 even in
            # FP64) AND image error <= 1e-10. The relative form `100 x fwd^2` is carried alongside so the scoring
            # cannot be tightened after the fact if this cell's own floor turns out worse than the letters cell's.
            at_floor_abs = fval <= 1e-28; at_floor_rel = fval <= 100 * fwd ** 2
            at_floor = at_floor_abs or at_floor_rel
            verdict = ("recovered" if at_floor and e_on <= 1e-10 else
                       "recovered (loose: image error < 1e-2)" if at_floor and e_on < 1e-2 else
                       "alias (residual zero, wrong image)" if at_floor else
                       "optimisation failure (residual not zero)")
            return dict(part="B", arm=tag, set=a.set, k=k, r=a.r, N=a.N, n_prime=Np, seed=a.seed, constrained=constrained,
                        fwd_check=fwd, res_at_truth=fwd, jac_sigma_min_truth=smin_truth, jac_sigma_max_truth=smax_truth,
                        n_landings_total=n_land, n_landings_replayed=len(landings),
                        start_err_vs_truth=start_err, residual=fval ** 0.5, objective=fval,
                        lm_iters_used=used, verdict=verdict, err_vs_chart_truth=e_on, err_vs_chart_all=e_all,
                        nearest_image=int(min(range(a.N), key=lambda i: e_all[i])), err_vs_raw=e_raw,
                        cert_norm_at_end=gnorm, at_floor_abs=at_floor_abs, at_floor_rel=at_floor_rel,
                        cert_at_truth=cert_at_truth, objective_trace_full=[float(x) for x in trace],
                        seconds=time.time() - t1, git=git_hash(), host=socket.gethostname())

        # ---------- D0 (gating): replay's basin radius ALONG Z_C, measured before any landing is spent ----------
        def walk(w0, dist, gen):
            """one random tangent step of size `dist` (relative to the latent std), pulled back onto Z_C."""
            w = w0.clone().reshape(-1)
            Jg = tf.jacfwd(g_of)(w).detach()
            _, _, Vh = torch.linalg.svd(Jg, full_matrices=True)
            null = Vh[Jg.shape[0]:].T if Jg.shape[0] < k else None
            if null is None or null.shape[1] == 0: return None
            d = null @ torch.randn(null.shape[1], generator=gen).to(dev)
            w = w + dist * float(W_all.std()) * d / torch.linalg.norm(d)
            for _ in range(8):                                        # Gauss-Newton back onto the manifold
                gv = g_of(w); Jg2 = tf.jacfwd(g_of)(w).detach()
                w = w - torch.linalg.lstsq(Jg2, gv.unsqueeze(1)).solution.reshape(-1)
                if float(torch.linalg.norm(g_of(w))) < 1e-12: break
            return w
        d0_radius = None
        if "d0" in a.arms:
            gd = torch.Generator().manual_seed(a.seed + 991); ok = []
            for dist in a.d0_steps:
                w_st = walk(w_true.reshape(-1), dist, gd)
                if w_st is None:
                    emit(dict(part="D0", set=a.set, k=k, r=a.r, n_prime=Np, dist=dist, note="Z_C has no tangent directions at this k")); continue
                with torch.no_grad(): cert_st = float(torch.linalg.norm(g_of(w_st)))
                e_st = float(torch.linalg.norm(chart.psi(w_st.reshape(k, 1))[:, 0] - X_on[:, top]) / torch.linalg.norm(X_on[:, top]))
                r0 = run(torch.cat([w_st, torch.zeros(a.r * Np, device=dev)]), False, "d0", e_st)
                emit(dict(part="D0", dist=dist, station_cert_norm=cert_st, station_err=e_st, **r0))
                if r0["verdict"].startswith("recovered"): ok.append(dist)
            d0_radius = max(ok) if ok else 0.0
            emit(dict(part="D0SUM", set=a.set, k=k, r=a.r, n_prime=Np, in_manifold_basin_radius=d0_radius,
                      gate_threshold=a.d0_min_radius, passed=bool(d0_radius >= a.d0_min_radius), git=git_hash()))
            print(f"  [k={k}] D0: replay recovers along Z_C out to {d0_radius} (gate {a.d0_min_radius})", flush=True)
            if d0_radius < a.d0_min_radius:
                print(f"  [k={k}] D0 GATE FAILED -- D1/D2 not launched at this k (plan section 3)", flush=True); continue

        gs = torch.Generator().manual_seed(a.seed + 31); landings = []; n_land = 0; t0 = time.time()
        cert_objs = []; land_errs = []; start_errs = []
        for s_i in range(a.cert_starts):
            w0 = (torch.randn(k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
            w, obj, _ = lm_cert(g_of, w0, a.cert_iters)
            xh = chart.psi(w.reshape(k, 1))[:, 0]
            e = min((float(torch.linalg.norm(xh - X_on[:, i]) / torch.linalg.norm(X_on[:, i])), i) for i in rec)
            cert_objs.append(obj); land_errs.append(e[0])
            e0 = min(float(torch.linalg.norm(chart.psi(w0.reshape(k, 1))[:, 0] - X_on[:, i]) / torch.linalg.norm(X_on[:, i])) for i in rec)
            start_errs.append(e0)                                   # the SAME start before the certificate solve
            if e[0] < 1e-2:
                n_land += 1
                if len(landings) < a.n_landings: landings.append((w.detach(), obj, e[0], e[1]))
        edges = [1e-28, 1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 1e-2, 1.0]      # the landing SPECTRUM (yoado-cd): tells
        hist = {f"<={e:.0e}": sum(1 for o in cert_objs if o ** 0.5 <= e) for e in edges}   # "the manifold misses the
        qs = sorted(o ** 0.5 for o in cert_objs)                          # basin" from "the landings were never on it"
        # THE CHEAPEST FALSIFIER (yoado-81), reported first after fwd_check: does the certificate move a start
        # CLOSER to a private image at all? If the landing distribution is no better than the random starts', the
        # chain is a smaller search of an equally bad space and D2 need not run (cf. 753886: exact certificate
        # zeros 0.84 away, above the line).
        le = sorted(land_errs); se = sorted(start_errs)
        emit(dict(part="HANDOFF", set=a.set, k=k, r=a.r, n_prime=Np,
                  landing_err=dict(min=le[0], p10=le[len(le)//10], median=le[len(le)//2]),
                  random_start_err=dict(min=se[0], p10=se[len(se)//10], median=se[len(se)//2]),
                  median_ratio=float(se[len(se)//2] / le[len(le)//2]) if le[len(le)//2] > 0 else float("inf"),
                  frac_landings_closer_than_best_random=float(sum(1 for x in land_errs if x < se[0]) / len(land_errs)),
                  git=git_hash()))
        print(f"  [k={k}] HANDOFF: landing err median {le[len(le)//2]:.3f} vs random-start median {se[len(se)//2]:.3f}", flush=True)
        emit(dict(part="L", set=a.set, k=k, r=a.r, n_prime=Np, cert_starts=a.cert_starts, n_landings_total=n_land,
                  n_landings_replayed=len(landings), cert_residual_cumulative=hist,
                  cert_residual_quantiles=dict(min=qs[0], p1=qs[len(qs)//100], p10=qs[len(qs)//10], median=qs[len(qs)//2]),
                  landing_err_min=float(min(land_errs)), landing_err_median=float(sorted(land_errs)[len(land_errs)//2]),
                  seconds=time.time() - t0, git=git_hash(), host=socket.gethostname()))
        print(f"  [k={k}] certificate: {n_land}/{a.cert_starts} landings ({time.time()-t0:.0f}s), carrying {len(landings)}", flush=True)
        if n_land == 0:
            print(f"  [k={k}] NO LANDINGS -- outcome (4): the chain test did not run here; not a replay failure", flush=True)
            continue

        # ---- replay LM, optionally confined to {g = 0}
        # null manifold (yoado-cd): the same construction on a Z_C built from a RESAMPLED B_T -- same dimension and
        # conditioning, wrong subspace. If constrained replay works there too, the constraint is not doing the work.
        gn = torch.Generator().manual_seed(a.seed + 555)
        perm_rows = torch.randperm(B_T.shape[0], generator=gn)
        B_null = B_T[perm_rows][:, torch.randperm(B_T.shape[1], generator=gn)]
        C_null, Np_null, _ = certificate(A_T, B_null)
        def g_null(w):
            f = bb.phi(chart.psi(w.reshape(k, 1)))
            return (C_null @ f).reshape(-1) / torch.linalg.norm(A_T @ f)

        gx = torch.Generator().manual_seed(a.seed + 77)
        for j, (w_l, obj_l, err_l, near_l) in enumerate(landings):
            v0 = torch.cat([w_l.reshape(-1), torch.zeros(a.r * Np, device=dev)])   # X unknown: start at zero
            if "constrained" in a.arms: emit(dict(landing=j, cert_objective=obj_l, landing_err=err_l, landing_nearest=near_l, **run(v0, True, "constrained", err_l)))
            if "unconstrained" in a.arms: emit(dict(landing=j, cert_objective=obj_l, landing_err=err_l, landing_nearest=near_l, **run(v0, False, "unconstrained", err_l)))
            if "null" in a.arms:
                g_true, g_of = g_of, g_null                            # swap the constraint for the null one
                emit(dict(landing=j, arm_note="constrained onto a RESAMPLED B_T's zero set (same dim, wrong subspace)",
                          n_prime_null=Np_null, **run(v0, True, "null_manifold", err_l)))
                g_of = g_true
        if "random" in a.arms:
            for j in range(min(4, a.n_landings)):
                w0 = (torch.randn(k, 1, generator=gx).to(dev) * coord_std).reshape(-1)
                e0 = float(torch.linalg.norm(chart.psi(w0.reshape(k, 1))[:, 0] - X_on[:, top]) / torch.linalg.norm(X_on[:, top]))
                emit(dict(landing=-1, **run(torch.cat([w0, torch.zeros(a.r * Np, device=dev)]), False, "random", e0)))


if __name__ == "__main__":
    main()
