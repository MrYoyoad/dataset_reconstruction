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


def dist_to_recorded(x, X_on, rec):
    """ONE candidate against the recorded set: the nearest RECORDED image (never an unrecorded one, never a fixed
       target). A fixed target deflates when several images are recorded; a nearest-over-ALL match would credit a
       drift toward an unrecorded image. Returns (error, index)."""
    return min(((float(torch.linalg.norm(x - X_on[:, i]) / torch.linalg.norm(X_on[:, i])), i) for i in rec))


def assign_to_recorded(Xh, X_on, rec):
    """A candidate SET against the recorded set: optimal one-to-one assignment (greedy nearest-match inflates by
       letting several returns claim the same truth). Returns (per-image errors, assignment, cost matrix)."""
    from scipy.optimize import linear_sum_assignment
    Cst = torch.zeros(Xh.shape[1], len(rec))
    for ii in range(Xh.shape[1]):
        for jj, tj in enumerate(rec):
            Cst[ii, jj] = torch.linalg.norm(Xh[:, ii] - X_on[:, tj]) / torch.linalg.norm(X_on[:, tj])
    ri, ci = linear_sum_assignment(Cst.cpu().numpy())
    return [float(Cst[i_, j_]) for i_, j_ in zip(ri, ci)], (ri, ci), Cst


def pick_constructed(ref, Xte_t, yte_t, N, perm):
    """The CONSTRUCTED N' = 1 cell (c9's option (b)): the single lowest-margin test image, plus N-1 fillers taken as
       the highest-margin image of each other class. The plan's words describe exactly this batch and no existing
       picker builds it -- the natural batches record 3 to 7 images on-chart at r = 8 (probe, 18 cells), so the
       one-image band the pre-registration is written for does not occur naturally and must be constructed. This is
       honest only because the specification was written first: the cell realises the pre-registration rather than
       being chosen after seeing results, and it is labelled `constructed` on every row."""
    yte = yte_t.cpu(); mar, _ = margins_of(ref, Xte_t.T, yte_t); mar = mar.cpu()
    hard = int(torch.argmin(mar)); c = int(yte[hard])
    fillers = []
    for cls in sorted((k for k in range(10) if k != c), key=lambda k: -max(float(mar[i]) for i in range(len(mar)) if int(yte[i]) == k)):
        best = max((float(mar[i]), i) for i in range(len(mar)) if int(yte[i]) == cls)[1]
        fillers.append(best)
        if len(fillers) == N - 1: break
    return [hard] + fillers
from experiments.exact_inversion.certificate import certificate, lm_cert
from experiments.exact_inversion.margin_check import traced_release

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_strong.pth")
    ap.add_argument("--set", default="confident", help="batch composition; the recorded member is the low-margin one")
    ap.add_argument("--sets", nargs="*", default=None, help="probe several batches (Part A + gate only) to find the N'=1 cell")
    ap.add_argument("--ks", nargs="*", type=int, default=[8, 10, 12, 14])
    ap.add_argument("--r", type=int, default=8); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cert-starts", type=int, default=500); ap.add_argument("--cert-iters", type=int, default=300)
    ap.add_argument("--n-landings", type=int, default=8, help="distinct certificate landings carried into replay")
    ap.add_argument("--lm-iters", type=int, default=300); ap.add_argument("--lm-lambda", type=float, default=1e-2)
    ap.add_argument("--arms", nargs="*", default=["d0", "constrained", "unconstrained", "random", "null"])
    ap.add_argument("--positive-control", action="store_true", help="run the certificate search alone at this cell and "
                    "require it to recover a recorded image from random starts -- a null from this harness is not "
                    "reportable unless the same pipeline reproduces a known positive (yoado-cd, after three artefact nulls)")
    ap.add_argument("--d0-steps", nargs="*", type=float, default=[0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
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

    if a.sets:                                                          # probe mode: which batch gives N' = 1 on-chart?
        for sname in a.sets:
            picker = pick_constructed if sname == "constructed" else pick_batch
            idxp = torch.tensor((picker(bb, Xte_t, yte_t, a.N, perm) if sname == "constructed"
                                 else picker(sname, bb, Xte_t, yte_t, a.N, perm)), device=dev)
            Xp = Xte_t[idxp].T.contiguous(); yp = yte_t[idxp]
            for k in a.ks:
                chartp = PCAChart(Xtr_t, k, dev); Xonp = chartp.psi(chartp.coords_of(Xp)); Hp = bb.phi(Xonp)
                gp = torch.Generator().manual_seed(a.seed + 7); A0p = (a.sigma0 * torch.randn(a.r, bb.n, generator=gp)).to(dev)
                A_Tp, B_Tp, impp, sBp, _ = release_and_imprints(bb, Xonp, yp, A0p, a)
                Cp, Npp, _ = certificate(A_Tp, B_Tp)
                resp = torch.linalg.norm(Cp @ Hp, dim=0) / torch.linalg.norm(A_Tp @ Hp, dim=0)
                topp = int(torch.argmax(impp))
                emit(dict(part="PROBE", set=sname, k=k, r=a.r, N=a.N, m=bb.m, n_prime=Npp, cert_line=a.r - Npp,
                          replay_line=bb.m + a.r - Npp, band_lo=a.r - Npp, band_hi=(bb.m - 1) + a.r - Npp,
                          sigma2_over_sigma1=float(sBp[1] / sBp[0]), imprint_rel=[float(v / impp.max()) for v in impp],
                          cert_at_truth_top=float(resp[topp]), git=git_hash()))
        return
    idx = torch.tensor((pick_constructed(bb, Xte_t, yte_t, a.N, perm) if a.set == "constructed"
                        else pick_batch(a.set, bb, Xte_t, yte_t, a.N, perm)), device=dev)
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
        top = int(torch.argmax(imp))                                     # the recorded (low-margin) image
        # CERTIFICATE GATE (yoado-b9): the band premise {rho=0} subset {Ch=0} is numerical, not formal. If the
        # certificate does not vanish at the truth in this cell, Z_C does not contain the truth and every branch of
        # the pre-registration is void -- the analogue of fwd_check, for the constraint rather than the simulator.
        # (yoado-b9) per image, against that image's own imprint -- a worst-case over the set is dominated by a
        # marginal member the certificate cannot certify, and reads as if the containment had failed for the cell.
        cert_at_truth = float(max(cert_res[i] for i in rec))             # worst over the RECORDED set (reported)
        cert_by_image = sorted(((float(imp[i] / imp.max()), float(cert_res[i]), i) for i in rec), reverse=True)
        imp_sorted = [x[0] for x in cert_by_image]
        gaps = [(imp_sorted[q] / imp_sorted[q + 1], q + 1) for q in range(len(imp_sorted) - 1) if imp_sorted[q + 1] > 0]
        gap_ratio, n_strong = max(gaps) if gaps else (1.0, len(rec))     # the natural cut in the imprint spectrum
        strong = [i for _, _, i in cert_by_image[:n_strong]]
        cert_at_truth_strong = float(max(cert_res[i] for i in strong))
        # (c9) the imprint GRAM per cell: aligned imprints lower the joint sigma_min AND defeat the subset solve, so
        # every null must be read against it -- "too aligned to separate" is a different verdict from "the chain
        # fails". Cimp holds the per-image contributions C_i with B_T = sum_i C_i.
        _, _, _, _, Cimp = traced_release(bb.phi(X_on), A0, bb.W0, y, bb.m, a.T, a.lr)
        Ci = torch.stack([Cimp[i].reshape(-1) for i in rec]); Ci = Ci / torch.linalg.norm(Ci, dim=1, keepdim=True)
        cosG = (Ci @ Ci.T); off = cosG[~torch.eye(len(rec), dtype=torch.bool, device=dev)]
        sG = torch.linalg.svdvals(cosG)

        print(f"  [k={k}] certificate gate at the truth: ||Ch||/||A_T h|| = {cert_at_truth:.3e}", flush=True)
        cert_line_certified = a.r - n_strong; replay_line_certified = (bb.m - 1) + a.r - n_strong
        in_band_certified = bool(cert_line_certified <= k <= replay_line_certified)
        gate_row = dict(part="GATE", set=a.set, k=k, r=a.r, n_prime_raw=Np, n_prime_certified=n_strong,
                        cert_line_raw=a.r - Np, replay_line_raw=(bb.m - 1) + a.r - Np,
                        cert_line=cert_line_certified, replay_line=replay_line_certified, in_band=in_band_certified,
                        n_prime=Np, cert_at_truth=cert_at_truth,
                        cert_at_truth_strong_only=cert_at_truth_strong, n_strong=n_strong, imprint_gap_ratio=gap_ratio,
                        imprint_spectrum=[x[0] for x in cert_by_image], cert_by_image=[x[1] for x in cert_by_image],
                        passed_all=bool(cert_at_truth <= 1e-6), passed_strong=bool(cert_at_truth_strong <= 1e-6))
        if cert_at_truth > 1e-6 and cert_at_truth_strong > 1e-6:
            emit({**gate_row, "passed": False, "note": "Z_C does not contain even the strongly-recorded truths here"})
            print(f"  [k={k}] GATE FAILED for the strong set too -- skipping", flush=True); continue
        if cert_at_truth > 1e-6:
            print(f"  [k={k}] gate: worst over the recorded set {cert_at_truth:.2e} is a MARGINAL image "
                  f"(imprint {imp_sorted[-1]:.1e}); the {n_strong} strongly-recorded images are certified to "
                  f"{cert_at_truth_strong:.2e} -- proceeding on the strong set, N'={n_strong}", flush=True)
        emit({**gate_row, "passed": True,
                  "imprint_cos_offdiag": dict(median=float(off.abs().median()), max=float(off.abs().max())),
                  "imprint_gram_sigma_ratio": float(sG[-1] / sG[0]), "fwd_check_pending": True, "git": git_hash()})
        if Np != 1:
            print(f"  [k={k}] N'={Np} != 1: the cell is not the one-image band; running anyway, rows carry N'", flush=True)


        # ---- the two maps: g (certificate, r x 1 per image) and the replay residual
        def g_one(w):                                                   # certificate map at ONE image (the search)
            f = bb.phi(chart.psi(w.reshape(k, 1)))
            return (C @ f).reshape(-1) / torch.linalg.norm(A_T @ f)

        def g_of(wv):                                                   # JOINT certificate map over the recorded set:
            W = wv.reshape(k, -1); F = bb.phi(chart.psi(W))             # stacked, so Z_C is the joint zero set that
            return ((C @ F) / torch.linalg.norm(A_T @ F, dim=0)).reshape(-1)   # constrained replay is confined to

        rec_t = torch.tensor(rec, device=dev); nrec = len(rec)           # JOINT subset solve over the recorded set
        kept, nkept = rec, nrec                                          # narrowed to the COVERED subset below if the
        nW = k * nrec                                                    # certificate did not cover the whole set

        def replay_res(v):                                              # v = [W (k x nrec), X (r x nrec)]
            Wc = v[:nW].reshape(k, nrec); Xc = v[nW:].reshape(a.r, nrec)
            Hc = bb.phi(chart.psi(Wc))
            Bs, Xis, Uc = simulate_sgd_reduced(Hc, Xc, bb.W0, y[rec_t], bb.m, a.T, a.lr * nrec / a.N, 0.0)
            return torch.cat([((Bs - B_T) / torch.linalg.norm(B_T)).reshape(-1),
                              ((Xis - A_T @ Uc) / torch.linalg.norm(A_T)).reshape(-1)])

        # fwd_check: the simulator at the TRUE latents and the true span-block (read first, ground rule 5)
        w_true = W_all[:, rec_t]                                        # (k, nrec) -- the recorded set's coordinates
        H_true1 = bb.phi(chart.psi(w_true)); U_true, _ = qr_canon(H_true1); X_true = (A0 @ U_true)
        v_true = torch.cat([w_true.reshape(-1), X_true.reshape(-1)])
        with torch.no_grad(): fwd = float(torch.linalg.norm(replay_res(v_true)))
        sv_truth = torch.linalg.svdvals(tf.jacfwd(replay_res)(v_true).detach())
        smin_truth, smax_truth = float(sv_truth[-1]), float(sv_truth[0])
        # (c9) the count is NECESSARY, sigma_min is SUFFICIENT: alignment and the r(N-N') free seed directions can
        # collapse it where the counting guard holds. A cell with sigma_min ~ 0 is unidentifiable and its chain rows
        # are uninterpretable -- gate it like fwd_check rather than reading them.
        emit(dict(part="SGATE", set=a.set, k=k, r=a.r, n_prime=Np, n_recorded=nrec, fwd_check=fwd,
                  jac_sigma_min_truth=smin_truth, jac_sigma_max_truth=smax_truth,
                  free_seed_directions=a.r * (a.N - nrec), passed=bool(smin_truth > 1e-12), git=git_hash()))
        print(f"  [k={k}] sigma_min(Drho) at the truth: {smin_truth:.3e}  (free seed dirs {a.r * (a.N - nrec)})", flush=True)
        if smin_truth <= 1e-12:
            print(f"  [k={k}] SIGMA GATE FAILED -- unidentifiable at the truth; chain rows would be uninterpretable", flush=True); continue
        print(f"  [k={k}] fwd_check at the truth (subset recipe, lr*N'/N): {fwd:.3e}", flush=True)

        # ---- certificate landings from random starts
        n_land = 0; landings = []                                      # defined before run(): D0 calls it first

        def seed_start(Wv):                                            # the project's standard seed initialiser
            with torch.no_grad():
                Uc, _ = qr_canon(bb.phi(chart.psi(Wv.reshape(k, -1))))
                return (A_T @ Uc).reshape(-1)
        def run(v0, constrained, tag, start_err, gfun=None):
            gfun = gfun if gfun is not None else g_of
            v = v0.clone(); jac = tf.jacfwd(replay_res)
            with torch.no_grad(): F = replay_res(v)
            fval = float(F @ F); lam = a.lm_lambda; t1 = time.time(); used = 0; trace = [fval]
            for it in range(a.lm_iters):
                J = jac(v).detach(); JtJ = J.T @ J; JtF = J.T @ F
                accepted = False
                for _ in range(12):
                    # the joint normal equations are singular at a rank-deficient point (the r(N-N') free seed
                    # directions and any alignment), so damp with diag(JtJ) as Marquardt and fall back to lstsq
                    Dmp = torch.diag(torch.diagonal(JtJ).clamp_min(1e-30))
                    try:
                        step = torch.linalg.solve(JtJ + lam * Dmp + 1e-30 * torch.eye(JtJ.shape[0], device=dev), JtF)
                    except Exception:
                        step = torch.linalg.lstsq(JtJ + lam * Dmp, JtF.unsqueeze(1)).solution.reshape(-1)
                    if constrained:                                     # project the latent block onto null(Jg)
                        Jg = tf.jacfwd(gfun)(v[:nW]).detach()
                        U_, S_, Vh = torch.linalg.svd(Jg, full_matrices=True)
                        rk = int((S_ > 1e-10 * S_[0]).sum()) if S_.numel() and float(S_[0]) > 0 else 0
                        nullsp = Vh[rk:].T                               # RANK, not shape
                        if nullsp.shape[1] > 0:
                            step[:nW] = nullsp @ (nullsp.T @ step[:nW])
                        else:
                            step[:nW] = 0.0
                    vn = v - step
                    if constrained:                                     # pull back onto the manifold (one GN step)
                        with torch.no_grad(): gv = gfun(vn[:nW])
                        Jg2 = tf.jacfwd(gfun)(vn[:nW]).detach()
                        vn = vn.clone(); vn[:nW] = vn[:nW] - torch.linalg.pinv(Jg2, rtol=1e-10) @ gv
                    with torch.no_grad(): Fn = replay_res(vn)
                    if float(Fn @ Fn) < fval:
                        v, F, fval = vn, Fn, float(Fn @ Fn); lam = max(lam / 3, 1e-15); accepted = True; break
                    lam *= 5
                used = it + 1; trace.append(fval)
                if fval < 1e-30 or not accepted or lam > 1e12: break
            Xh = chart.psi(v[:nW].reshape(k, nrec))                     # (784, nrec) the returned set
            # (yoado-b9) OPTIMAL ONE-TO-ONE ASSIGNMENT, never greedy nearest-match: a joint solve with a mixing
            # symmetry can return several near-copies of the easiest image, and greedy matching would score that
            # well. The number of DISTINCT truths matched is reported separately as mode collapse.
            per_image, (ri, ci), Cst = assign_to_recorded(Xh, X_on, rec)   # shared helper: one-to-one assignment
            # (c9) Hungarian matching can INFLATE the count: it minimises total error, so a blend gets assigned to a
            # truth even when its match is poor. An image counts as recovered only if its match is UNAMBIGUOUS --
            # the assigned error at least 10x below that return's second-best truth.
            second = []
            for i_, j_ in zip(ri, ci):
                others = [float(Cst[i_, t_]) for t_ in range(nrec) if t_ != j_]
                second.append(min(others) if others else float("inf"))
            unambiguous = [bool(per_image[q] * 10 <= second[q]) for q in range(len(per_image))]
            e_on = max(per_image)
            n_under = sum(1 for q, e in enumerate(per_image) if e <= 1e-10 and unambiguous[q])
            n_under_ambiguous_ok = sum(1 for e in per_image if e <= 1e-10)
            # label-blind SET error: the same assignment ignoring which truth carries which label (a permutation-only
            # failure is a leakage SUCCESS -- the attacker has the images)
            n_distinct = len(set(int(j_) for j_ in ci))
            e_raw = max(float(torch.linalg.norm(Xh[:, i_] - X_real[:, rec[j_]]) / torch.linalg.norm(X_real[:, rec[j_]]))
                        for i_, j_ in zip(ri, ci))
            with torch.no_grad(): gnorm = float(torch.linalg.norm(gfun(v[:nW])))
            # Pre-registered thresholds (yoado-b9): the cell is FP64 throughout, so the absolute form applies --
            # branch 1 is OBJECTIVE <= 1e-28 (not `residual`, which is its square root and floors at 1e-15 even in
            # FP64) AND image error <= 1e-10. The relative form `100 x fwd^2` is carried alongside so the scoring
            # cannot be tightened after the fact if this cell's own floor turns out worse than the letters cell's.
            at_floor_abs = fval <= 1e-28; at_floor_rel = fval <= 100 * fwd ** 2
            at_floor = at_floor_abs or at_floor_rel
            verdict = ("recovered (all N')" if at_floor and n_under == nrec and n_distinct == nrec else
                       "mode collapse (returned set covers fewer truths)" if at_floor and n_distinct < nrec else
                       f"partial ({n_under} of {nrec} under 1e-10)" if at_floor and n_under > 0 else
                       "alias (residual zero, wrong images)" if at_floor else
                       "optimisation failure (residual not zero)")
            in_band_row = in_band_certified                            # lines from the CERTIFIED subset
            if not in_band_row:
                verdict = "NOT IN BAND -- no chain verdict is emitted from this cell (" + verdict + " below/above the band)"
            return dict(part="B", arm=tag, set=a.set, k=k, r=a.r, N=a.N, n_prime=Np, seed=a.seed, constrained=constrained,
                        in_band=in_band_row, cert_line=cert_line_certified, replay_line=replay_line_certified,
                        n_prime_certified=n_strong, cert_at_truth_strong_only=cert_at_truth_strong,
                        fwd_check=fwd, res_at_truth=fwd, jac_sigma_min_truth=smin_truth, jac_sigma_max_truth=smax_truth,
                        n_landings_total=n_land, n_landings_replayed=len(landings),
                        start_err_vs_truth=start_err, residual=fval ** 0.5, objective=fval,
                        lm_iters_used=used, verdict=verdict, err_vs_chart_truth_MAX=e_on,
                        err_per_image_assigned=per_image, n_under_1e10=n_under, n_distinct_truths_matched=n_distinct,
                        match_unambiguous=unambiguous, second_best_err=second,
                        n_under_1e10_ambiguous_ok=n_under_ambiguous_ok, j_recovered=n_under, j_over_Nprime=float(n_under / nrec),
                        n_recorded=nrec, assignment=[int(x) for x in ci], err_vs_raw=e_raw,
                        recovered_imprints=[float(imp[rec[j_]] / imp.max()) for i_, j_ in zip(ri, ci) if float(Cst[i_, j_]) <= 1e-10],
                        cert_norm_at_end=gnorm, at_floor_abs=at_floor_abs, at_floor_rel=at_floor_rel,
                        cert_at_truth=cert_at_truth, objective_trace_full=[float(x) for x in trace],
                        seconds=time.time() - t1, git=git_hash(), host=socket.gethostname())

        if a.positive_control:
            gp = torch.Generator().manual_seed(a.seed + 4242); hits = 0; best = 1e9
            for _ in range(200):
                w0 = (torch.randn(k, 1, generator=gp).to(dev) * coord_std).reshape(-1)
                w, obj, _ = lm_cert(g_one, w0, a.cert_iters)
                xh = chart.psi(w.reshape(k, 1))[:, 0]
                e = min(float(torch.linalg.norm(xh - X_on[:, i]) / torch.linalg.norm(X_on[:, i])) for i in rec)
                best = min(best, e); hits += int(e < 1e-2)
            emit(dict(part="PCTRL", set=a.set, k=k, r=a.r, n_prime=Np, cert_line=a.r - Np,
                      below_cert_line=bool(k < a.r - Np), starts=200, hits=hits, best_err=best,
                      passed=bool(hits > 0), note="the certificate alone must recover a recorded image here",
                      git=git_hash()))
            print(f"  [k={k}] POSITIVE CONTROL: {hits}/200 random starts recovered a recorded image (best err {best:.2e})", flush=True)

        # ---------- D0 (gating): replay's basin radius ALONG Z_C, measured before any landing is spent ----------
        def walk(w0, dist, gen):
            """one random tangent step of size `dist` (relative to the latent std), pulled back onto Z_C."""
            w = w0.clone().reshape(-1)
            Jg = tf.jacfwd(g_of)(w).detach()
            U_, S_, Vh = torch.linalg.svd(Jg, full_matrices=True)
            rk = int((S_ > 1e-10 * S_[0]).sum()) if S_.numel() and float(S_[0]) > 0 else 0
            null = Vh[rk:].T                                            # RANK, not shape: C has rank r - N'
            if null.shape[1] == 0: return None
            d = null @ torch.randn(null.shape[1], generator=gen).to(dev)
            w = w + dist * float(W_all.std()) * d / torch.linalg.norm(d)
            for _ in range(12):                                       # Gauss-Newton back onto the manifold
                gv = g_of(w); Jg2 = tf.jacfwd(g_of)(w).detach()
                w = w - torch.linalg.pinv(Jg2, rtol=1e-10) @ gv         # pinv: Jg is rank-deficient by construction
                if float(torch.linalg.norm(g_of(w))) < 1e-12: break
            return w
        d0_radius = None
        if "d0" in a.arms:
            gd = torch.Generator().manual_seed(a.seed + 991); ok = []; stations = []
            for dist in a.d0_steps:
                w_st = w_true.reshape(-1).clone() if dist == 0.0 else walk(w_true.reshape(-1), dist, gd)
                if w_st is None:
                    emit(dict(part="D0", set=a.set, k=k, r=a.r, n_prime=Np, dist=dist, note="Z_C has no tangent directions at this k")); continue
                with torch.no_grad(): cert_st = float(torch.linalg.norm(g_of(w_st)))
                if cert_st > 1e-8:                                       # the pull-back did not converge: the station
                    emit(dict(part="D0", set=a.set, k=k, dist=dist, station_cert_norm=cert_st,   # is not on Z_C, so its
                              valid=False, note="station left Z_C; replay outcome not scored"))  # replay says nothing
                    print(f"    [k={k}] station at {dist}: OFF the manifold (cert {cert_st:.2e}) -- not scored", flush=True)
                    continue
                Xst = chart.psi(w_st.reshape(k, nrec))                  # a station is the whole recorded SET
                e_st = max(float(torch.linalg.norm(Xst[:, q] - X_on[:, rec[q]]) / torch.linalg.norm(X_on[:, rec[q]]))
                           for q in range(nrec))                        # worst image at the station
                r0 = run(torch.cat([w_st, seed_start(w_st)]), False, "d0", e_st)
                emit({**r0, "part": "D0", "dist": dist, "station_cert_norm": cert_st, "station_err": e_st})
                stations.append((dist, e_st))
                if r0["verdict"].startswith("recovered"): ok.append((dist, e_st))
            # (yoado-c9) the basin radius and the landing distribution must be in ONE metric or their intersection is
            # an eyeball across units. IMAGE ERROR is the primary here -- it is what the handoff row reports and the
            # attacker-meaningful quantity -- with the latent step size kept alongside for the walk's own bookkeeping.
            d0_radius_img = max((e for _, e in ok), default=0.0)
            d0_radius = max((d for d, _ in ok), default=0.0)
            # (yoado-c9) the basin is anisotropic -- wide along truth-directions, narrow in attacker-reachable ones --
            # so a single max radius is an optimistic CEILING, not a threshold. The honest object is the
            # recover-fraction-vs-image-error CURVE over stations, with the landing distribution overlaid on it; the
            # gap between the ceiling's overlap and D1's actual fraction from the real landings IS the anisotropy.
            emit(dict(part="D0SUM", set=a.set, k=k, r=a.r, n_prime=Np,
                      in_manifold_basin_radius_IMAGE_ERR=d0_radius_img, in_manifold_basin_radius_latent=d0_radius,
                      metric="image error vs the on-chart truth (same metric as the HANDOFF row's landing_err)",
                      stations_recovered=[dict(latent=d, image_err=e) for d, e in ok],
                      recover_curve=[dict(latent=d, image_err=e, recovered=bool((d, e) in ok)) for d, e in stations],
                      n_stations=len(stations),
                      gate_threshold=a.d0_min_radius, passed=bool(d0_radius >= a.d0_min_radius), git=git_hash()))
            print(f"  [k={k}] D0: replay recovers out to image error {d0_radius_img:.4f} along Z_C "
                  f"(latent step {d0_radius}; gate {a.d0_min_radius})", flush=True)
            if d0_radius < a.d0_min_radius:
                print(f"  [k={k}] D0 GATE FAILED -- D1/D2 not launched at this k (plan section 3)", flush=True); continue

        gs = torch.Generator().manual_seed(a.seed + 31); by_image = {}; t0 = time.time()
        cert_objs = []; land_errs = []; start_errs = []; normmatch_errs = []; norm_ratios = []
        for s_i in range(a.cert_starts):
            w0 = (torch.randn(k, 1, generator=gs).to(dev) * coord_std).reshape(-1)
            w, obj, _ = lm_cert(g_one, w0, a.cert_iters)
            xh = chart.psi(w.reshape(k, 1))[:, 0]
            e = dist_to_recorded(xh, X_on, rec)                        # shared helper
            cert_objs.append(obj)
            # (yoado-b9) the handoff must be measured against the RECORDED image specifically, never the post-hoc
            # nearest one -- the certificate has no information about images it did not record, and a start drifting
            # toward an unrecorded one would score as a pass. `top` is the recorded image.
            def err_to_rec(wv):                                       # shared helper: nearest RECORDED image
                return dist_to_recorded(chart.psi(wv.reshape(k, 1))[:, 0], X_on, rec)[0]
            land_errs.append(err_to_rec(w)); start_errs.append(err_to_rec(w0))
            # (yoado-b9) SHRINKAGE control: if the solve mostly shrinks ||w|| toward the chart mean, every start moves
            # closer to every image without acquiring information. Compare against a random point of the SAME norm.
            nw = float(torch.linalg.norm(w)); n0 = float(torch.linalg.norm(w0))
            wr = torch.randn(k, generator=gs).to(dev); wr = wr * (nw / float(torch.linalg.norm(wr)))
            normmatch_errs.append(err_to_rec(wr)); norm_ratios.append(nw / n0 if n0 > 0 else float("nan"))
            if e[0] < 1e-2:
                n_land += 1
                by_image.setdefault(e[1], []).append((w.detach(), obj, e[0]))
        edges = [1e-28, 1e-20, 1e-16, 1e-12, 1e-8, 1e-4, 1e-2, 1.0]      # the landing SPECTRUM (yoado-cd): tells
        hist = {f"<={e:.0e}": sum(1 for o in cert_objs if o ** 0.5 <= e) for e in edges}   # "the manifold misses the
        qs = sorted(o ** 0.5 for o in cert_objs)                          # basin" from "the landings were never on it"
        # THE CHEAPEST FALSIFIER (yoado-81), reported first after fwd_check: does the certificate move a start
        # CLOSER to a private image at all? If the landing distribution is no better than the random starts', the
        # chain is a smaller search of an equally bad space and D2 need not run (cf. 753886: exact certificate
        # zeros 0.84 away, above the line).
        le = sorted(land_errs); se = sorted(start_errs); ne = sorted(normmatch_errs); nr = sorted(norm_ratios)
        # floor-reachers: in band this set INCLUDES spurious zeros, which is the point of the test, not contamination
        floor_q = [q for q in range(len(cert_objs)) if cert_objs[q] ** 0.5 <= 1e-8]
        nbins = 10 if len(floor_q) >= 100 else (4 if len(floor_q) >= 40 else 0)
        order = sorted(floor_q, key=lambda q: cert_objs[q]) if nbins else []
        dec = []
        for d in range(nbins):
            idx = order[d * len(order) // nbins:(d + 1) * len(order) // nbins]
            if not idx: continue
            dec.append(dict(decile=d, bin_of=nbins,
                            cert_residual_lo=float(min(cert_objs[q] ** 0.5 for q in idx)),
                            cert_residual_hi=float(max(cert_objs[q] ** 0.5 for q in idx)),
                            cert_residual_median=float(sorted(cert_objs[q] ** 0.5 for q in idx)[len(idx) // 2]),
                            landing_err_median=float(sorted(land_errs[q] for q in idx)[len(idx) // 2]),
                            norm_matched_err_median=float(sorted(normmatch_errs[q] for q in idx)[len(idx) // 2]),
                            frac_closer_than_norm_matched=float(sum(1 for q in idx if land_errs[q] < normmatch_errs[q]) / len(idx)),
                            frac_landed_1e2=float(sum(1 for q in idx if land_errs[q] < 1e-2) / len(idx))))
        # SURVIVES iff some ADJACENT bin boundary has >= 3 orders of separation in median certificate residual,
        # landing rate >= 0.8 below it and <= 0.3 above it. The boundary's POSITION is free and reported, not fixed.
        best_cut = None
        for b in range(len(dec) - 1):
            lo = dec[:b + 1]; hi = dec[b + 1:]
            mlo = sorted(x["cert_residual_median"] for x in lo)[len(lo) // 2]
            mhi = sorted(x["cert_residual_median"] for x in hi)[len(hi) // 2]
            rlo = sum(x["frac_landed_1e2"] * 1 for x in lo) / len(lo)
            rhi = sum(x["frac_landed_1e2"] * 1 for x in hi) / len(hi)
            sep = (mhi / mlo) if mlo > 0 else float("inf")
            ok = bool(sep >= 1e3 and rlo >= 0.8 and rhi <= 0.3)
            if ok and best_cut is None:
                best_cut = dict(boundary_after_bin=b, separation_orders=float(torch.log10(torch.tensor(sep))),
                                landing_rate_low=rlo, landing_rate_high=rhi)
        emit(dict(part="HANDOFF", set=a.set, k=k, r=a.r, n_prime_raw=Np, n_prime_certified=n_strong,
                  in_band=in_band_certified, cert_line=cert_line_certified, replay_line=replay_line_certified,
                  cert_at_truth_strong_only=cert_at_truth_strong,
                  landing_err=dict(min=le[0], p10=le[len(le)//10], median=le[len(le)//2]),
                  random_start_err=dict(min=se[0], p10=se[len(se)//10], median=se[len(se)//2]),
                  norm_matched_random_err=dict(min=ne[0], p10=ne[len(ne)//10], median=ne[len(ne)//2]),
                  latent_norm_ratio=dict(min=nr[0], median=nr[len(nr)//2], max=nr[-1]),
                  median_ratio=float(se[len(se)//2] / le[len(le)//2]) if le[len(le)//2] > 0 else float("inf"),
                  median_ratio_vs_norm_matched=float(ne[len(ne)//2] / le[len(le)//2]) if le[len(le)//2] > 0 else float("inf"),
                  frac_landings_closer_than_best_random=float(sum(1 for x in land_errs if x < se[0]) / len(land_errs)),
                  frac_landings_closer_than_norm_matched=float(sum(1 for i in range(len(land_errs)) if land_errs[i] < normmatch_errs[i]) / len(land_errs)),
                  measured_against="the nearest RECORDED image (all are legitimate targets; unrecorded images excluded)",
                  by_certificate_residual_decile=dec, n_floor_reachers=len(floor_q), n_bins=nbins,
                  scoring=("VOID (<40 floor-reachers)" if nbins == 0 else ("deciles" if nbins == 10 else "quartiles")),
                  survives=bool(best_cut is not None), survival_cut=best_cut,
                  git=git_hash()))
        print(f"  [k={k}] HANDOFF vs the recorded image: landing {le[len(le)//2]:.3f} | random start {se[len(se)//2]:.3f} | "
              f"norm-matched random {ne[len(ne)//2]:.3f} | latent norm ratio {nr[len(nr)//2]:.2f}", flush=True)
        emit(dict(part="L", set=a.set, k=k, r=a.r, n_prime=Np, cert_starts=a.cert_starts, n_landings_total=n_land,
                  n_landings_replayed=len(landings), cert_residual_cumulative=hist,
                  cert_residual_quantiles=dict(min=qs[0], p1=qs[len(qs)//100], p10=qs[len(qs)//10], median=qs[len(qs)//2]),
                  landing_err_min=float(min(land_errs)), landing_err_median=float(sorted(land_errs)[len(land_errs)//2]),
                  seconds=time.time() - t0, git=git_hash(), host=socket.gethostname()))
        # A JOINT start needs one landing per recorded image. The certificate finds ONE image per start, so the
        # handoff is only complete if the landings COVER the recorded set -- coverage is itself a result, and an
        # incomplete cover means the chain cannot be assembled at all at this cell (reported, not patched).
        covered = sorted(by_image); n_cov = len(covered)
        emit(dict(part="COVER", set=a.set, k=k, r=a.r, n_prime=Np, n_recorded=nrec, images_covered=covered,
                  n_images_covered=n_cov, complete_cover=bool(n_cov == nrec),
                  landings_per_image={str(i): len(v) for i, v in by_image.items()},
                  n_joint_starts=min((len(v) for v in by_image.values()), default=0) if n_cov == nrec else 0,
                  git=git_hash()))
        print(f"  [k={k}] certificate covers {n_cov}/{nrec} recorded images; joint starts assemblable: "
              f"{min((len(v) for v in by_image.values()), default=0) if n_cov == nrec else 0}", flush=True)
        # (GM/Yoad) coverage is a SAMPLING COST, not a structural blocker. N' is read off the release as rank(B_T)
        # -- a MILD assumption, not an oracle (unlike a near-truth start) -- so knowing it gives the stopping rule
        # "draw starts until N' distinct landings", and landings are deduplicated by comparing recovered images,
        # which the attacker can do unaided. An incomplete cover is reported as a cost curve and the chain still runs
        # on the images actually covered, at N'_kept = the covered count with the step rescaled accordingly.
        if n_cov < nrec:
            print(f"  [k={k}] cover not achieved in {a.cert_starts} starts: {n_cov} of {nrec} distinct images "
                  f"-- running the chain on the covered subset (N'_kept = {n_cov})", flush=True)
        kept = covered if n_cov < nrec else rec
        nkept = len(kept)
        if nkept != nrec:
            # narrow the solve to the covered subset: the residual, the constraint and BOTH gates must be those of
            # the subset actually solved, not of the full recorded set (the step rescales as lr * N'_kept / N).
            rec, nrec, rec_t, nW = kept, nkept, torch.tensor(kept, device=dev), k * nkept
            w_true = W_all[:, rec_t]; H_t = bb.phi(chart.psi(w_true)); U_t, _ = qr_canon(H_t)
            v_true = torch.cat([w_true.reshape(-1), (A0 @ U_t).reshape(-1)])
            with torch.no_grad(): fwd = float(torch.linalg.norm(replay_res(v_true)))
            sv2 = torch.linalg.svdvals(tf.jacfwd(replay_res)(v_true).detach())
            smin_truth, smax_truth = float(sv2[-1]), float(sv2[0])
            emit(dict(part="SGATE", set=a.set, k=k, r=a.r, n_prime=Np, n_recorded=nkept, subset_of=len(covered),
                      note="re-gated on the COVERED subset", fwd_check=fwd, jac_sigma_min_truth=smin_truth,
                      jac_sigma_max_truth=smax_truth, passed=bool(smin_truth > 1e-12), git=git_hash()))
            print(f"  [k={k}] re-gated on the covered subset: fwd {fwd:.2e}, sigma_min {smin_truth:.3e}", flush=True)
            if smin_truth <= 1e-12:
                print(f"  [k={k}] covered subset unidentifiable -- skipping", flush=True); continue
        for j in range(min(a.n_landings, min(len(by_image[i]) for i in kept))):
            W0j = torch.stack([by_image[i][j][0].reshape(-1) for i in kept], dim=1)
            landings.append((W0j, max(by_image[i][j][1] for i in kept), max(by_image[i][j][2] for i in kept), None))
        print(f"  [k={k}] certificate: {n_land}/{a.cert_starts} landings ({time.time()-t0:.0f}s), assembled {len(landings)} joint starts", flush=True)
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
        def g_null(wv):                                                 # joint, same shape as g_of
            W = wv.reshape(k, -1); F = bb.phi(chart.psi(W))
            return ((C_null @ F) / torch.linalg.norm(A_T @ F, dim=0)).reshape(-1)

        gx = torch.Generator().manual_seed(a.seed + 77)
        for j, (w_l, obj_l, err_l, near_l) in enumerate(landings):
            v0 = torch.cat([w_l.reshape(-1), seed_start(w_l)])          # X from the span estimate, as everywhere else
            if "constrained" in a.arms: emit({**run(v0, True, "constrained", err_l), "landing": j, "cert_objective": obj_l, "landing_err": err_l})
            if "unconstrained" in a.arms: emit({**run(v0, False, "unconstrained", err_l), "landing": j, "cert_objective": obj_l, "landing_err": err_l})
            if "null" in a.arms:
                emit({**run(v0, True, "scrambled_manifold", err_l, gfun=g_null), "landing": j,
                      "arm_note": "constrained onto a RESAMPLED B_T's zero set (same dim, wrong subspace)", "n_prime_null": Np_null})
        if "random" in a.arms:
            for j in range(min(4, a.n_landings)):
                w0 = (torch.randn(k, 1, generator=gx).to(dev) * coord_std).reshape(-1)
                e0 = float(torch.linalg.norm(chart.psi(w0.reshape(k, 1))[:, 0] - X_on[:, top]) / torch.linalg.norm(X_on[:, top]))
                Wr = torch.stack([(torch.randn(k, generator=gx).to(dev) * coord_std.reshape(-1)) for _ in range(nrec)], dim=1)
                emit({**run(torch.cat([Wr.reshape(-1), seed_start(Wr)]), False, "random", e0), "landing": -1})


if __name__ == "__main__":
    main()
