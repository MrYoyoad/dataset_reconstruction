#!/usr/bin/env python3
"""Numerical sanity checks for theory/T1..T6 (multilayer certificate track).

Every check prints PASS/FAIL against a PRE-STATED tolerance and appends one JSON line per check to the results
file.  A PASS is a sanity check, NOT a proof: theory/*.md records the status of the proof, and this script can
only ever falsify.  FP64 throughout.

    python -u -m experiments.multilayer_cert.theory_checks --check all --out results/multilayer_cert/checks.jsonl
"""
import argparse, json, math, os, socket, subprocess, sys, time
import torch

from experiments.multilayer_cert.common import (MultiLoRANet, certificate, numrank, rel_annihilation, span_of,
                                                gelu, _gelu_prime)

torch.set_default_dtype(torch.float64)
RES = []


def log(s): print(s, flush=True)


def record(name, ok, **kw):
    RES.append(dict(check=name, passed=bool(ok), **kw))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: " + "  ".join(f"{k}={v}" for k, v in kw.items()))
    return ok


# ------------------------------------------------------------------ T1
def check_T1(dev, seed=0):
    """T1.1 exact unrolling for a LINEAR net; T1.2 bound + first-order slope for the GELU net."""
    N, r, dims = 4, 8, [12, 10, 10, 10, 6]
    net = MultiLoRANet(dims, r, N, seed=seed, dev=dev)
    x = torch.randn(dims[0], N, generator=torch.Generator().manual_seed(seed + 1)).to(dev)
    y = torch.arange(N) % dims[-1]

    # --- T1.1: linear network, exact unrolled formula
    lin = MultiLoRANet(dims, r, N, seed=seed, dev=dev)
    lin.forward_reps = lambda xx, A, B, _s=lin: _linear_reps(_s, xx, A, B)
    A, B, reps = lin.train(x, y, T=6, lr=0.5)
    H0 = _linear_reps(lin, x, lin.A0, lin.B0)
    Ht = _linear_reps(lin, x, A, B)
    worst = 0.0
    for l in range(1, lin.L + 1):                       # Delta_{l} = sum_{j<l} (prod T_i) s B_j A_j H_j^0
        acc = torch.zeros_like(Ht[l])
        for j in range(l):
            term = lin.s * B[j] @ (A[j] @ H0[j])
            for i in range(j + 1, l):
                term = (lin.W0[i] + lin.s * B[i] @ A[i]) @ term
            acc = acc + term
        worst = max(worst, float((Ht[l] - H0[l] - acc).norm() / (Ht[l] - H0[l]).norm().clamp_min(1e-300)))
    ok1 = record("T1.1_linear_unrolling_exact", worst < 1e-10, rel_err=f"{worst:.2e}", tol="1e-10")

    # --- T1.2: relative drift vs sum of adapter norms, log-log slope should be 1
    slopes = []
    for l in range(1, net.L):
        xs, ys = [], []
        for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
            A, B, reps = net.train(x, y, T=4, lr=lr)
            Hb = net.base_reps(x)
            d = max(float((rt[l] - Hb[l]).norm() / Hb[l].norm()) for rt in reps)
            beta = sum(float(net.s * torch.linalg.matrix_norm(B[j] @ A[j], 2)) for j in range(l))
            if d > 0 and beta > 0:
                xs.append(math.log(beta)); ys.append(math.log(d))
        if len(xs) >= 3:
            n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
            slopes.append(sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs))
    ok2 = record("T1.2_drift_first_order_in_beta", all(abs(s - 1) < 0.15 for s in slopes),
                 slopes=[f"{s:.3f}" for s in slopes], tol="|slope-1|<0.15")
    return ok1 and ok2


def _largest_sane_lr(net, x, y, T, band=(0.05, 5.0)):
    """Largest lr whose worst-layer relative drift lands in `band` and whose trajectory stays finite.

    The point of these checks is LARGE drift, but an exploding net (drift ~1e5) makes every singular subspace
    numerical noise -- it does not test the theorem, it tests float64.  See LESSONS_LEARNED 2026-09-07.
    """
    best = (None, None, None, None, None)
    for lr in [3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.02]:
        A, B, reps = net.train(x, y, T=T, lr=lr)
        if not (all(torch.isfinite(h).all() for rt in reps for h in rt)
                and all(torch.isfinite(t).all() for t in A + B)):
            continue
        Hb = net.base_reps(x)
        d = max(float((rt[l] - Hb[l]).norm() / Hb[l].norm())
                for rt in reps for l in range(net.L) if float(Hb[l].norm()) > 0)
        if band[0] <= d <= band[1]:
            return A, B, reps, lr, d
        if d < band[0] and best[0] is None:
            best = (A, B, reps, lr, d)                       # fall back to the largest sub-band drift
    return best


def _linear_reps(net, x, A, B):
    H, out = x, [x]
    for i in range(net.L):
        H = (net.W0[i] + net.s * B[i] @ A[i]) @ H
        out.append(H)
    return out


# ------------------------------------------------------------------ T2
def check_T2(dev, seed=0):
    """Prop. A: the FULL certificate annihilates every H_{l,t} -- including H_l^0 -- exactly, at LARGE drift.

    Also: rank C_full == r - N' with N' the training-span dimension, and the exact identity T2.1.
    """
    N, r, dims, T = 3, 24, [16, 20, 20, 20, 5], 5      # width >= N*T so B3 CAN hold at a deep layer
    net = MultiLoRANet(dims, r, N, seed=seed, dev=dev)
    x = torch.randn(dims[0], N, generator=torch.Generator().manual_seed(seed + 1)).to(dev)
    y = torch.arange(N) % dims[-1]
    # LARGE but CONVERGENT drift: the largest lr on a geometric ladder that keeps the trajectory finite.
    # (lr=2.0 diverges for this net; a diverged run has H_t = inf, which makes every downstream number NaN
    #  and -- see LESSONS_LEARNED 2026-09-07 -- silently *passes* a max()-based tolerance test.)
    A, B, reps, used_lr, used_drift = _largest_sane_lr(net, x, y, T)
    if A is None:
        return record("T2_PropA_full_cert_exact_at_large_drift", False, error="no lr gives a sane drift band")
    log(f"    (lr={used_lr}: worst-layer drift {used_drift:.3f} -- large, but NOT an exploded net: at "
        f"drift ~1e5 every singular subspace below is numerical noise and the test measures nothing)")
    Hb = net.base_reps(x)

    rows, worst_all, worst_base, rank_ok = [], 0.0, 0.0, True
    for l in range(net.L):
        drift = max(float((rt[l] - Hb[l]).norm() / Hb[l].norm()) for rt in reps)
        Nprime, _ = span_of([rt[l] for rt in reps])
        C, q, sB = certificate(A[l], B[l])                          # FULL certificate
        rho_t = [rel_annihilation(C, rt[l]) for rt in reps]
        rho0 = rel_annihilation(C, Hb[l])
        rC, _ = numrank(C, ref=float(A[l].norm()))
        exact_hyp = (q == Nprime)                                   # B3 of theory/T2
        exp_rC = max(0, min(r - Nprime, dims[l] - Nprime))
        if exact_hyp:                       # explicit, so a NaN can never be swallowed by max()
            for v in rho_t + [rho0]:
                if not math.isfinite(v):
                    worst_all = worst_base = float("inf")
            if math.isfinite(worst_all):
                worst_all = max(worst_all, max(rho_t)); worst_base = max(worst_base, rho0)
        rank_ok &= (not exact_hyp) or (rC == exp_rC)
        rows.append(dict(layer=l, drift=drift, N_prime=Nprime, rank_B=q, rank_C=rC, expect_rank_C=exp_rC,
                         expect_rank_C_as_written=r - Nprime, as_written_law_agrees=bool(rC == r - Nprime),
                         rho_base=rho0, rho_worst_t=max(rho_t), exact_hypothesis=exact_hyp))
        log(f"    layer {l}: drift={drift:.3f}  N'={Nprime}  rank B_T={q}  rank C={rC} "
            f"(corrected min={exp_rC}; as-written r-N'={r - Nprime}"
            f"{'  <-- AS-WRITTEN OVERCOUNTS' if (exact_hyp and rC != r - Nprime) else ''})"
            f"  rho(H^0)={rho0:.2e}  max_t rho={max(rho_t):.2e}  B3={'yes' if exact_hyp else 'NO'}"
            f"  sigma(B_T)[:6]={[f'{v:.1e}' for v in sB[:6].tolist()]}")
    exercised = [rw for rw in rows if rw["exact_hypothesis"] and rw["layer"] >= 1 and rw["drift"] > 1e-3]
    ok1 = record("T2_PropA_full_cert_exact_at_large_drift",
                 bool(exercised) and worst_base < 1e-11 and worst_all < 1e-11,
                 max_rho_base=f"{worst_base:.2e}", max_rho_any_t=f"{worst_all:.2e}", tol="1e-11",
                 exercised_deep_layers=[rw["layer"] for rw in exercised],
                 max_drift_exercised=f"{max([rw['drift'] for rw in exercised], default=0):.3f}",
                 note="FAILS as vacuous if no DEEP layer with real drift satisfies B3", layers=rows)
    ok2 = record("T2_rank_C_matches_corrected_min_law", rank_ok, law="max(0, min(r-Nprime, n_l-Nprime))")
    # The statement AS ORIGINALLY WRITTEN in theory/T2 was rank C = r - N'. Record a check against THAT law too, so
    # a row exists that can disagree. It FAILS at any narrow layer (n_l < r); that failure is the point (audit A18)
    # and is deliberately NOT gated into the return value.
    as_written_ok = all((not rw["exact_hypothesis"]) or rw["rank_C"] == rw["expect_rank_C_as_written"] for rw in rows)
    record("T2_rank_C_matches_AS_WRITTEN_r_minus_Nprime", as_written_ok, law="r - Nprime",
           note="the statement as originally written; expected to FAIL at any narrow layer n_l<r")

    # --- T2.1 exact identity for the truncated certificate, at a layer with drift
    ident, gaps = 0.0, []
    for l in range(1, net.L):
        Nprime, _ = span_of([rt[l] for rt in reps])
        Cf, q, sB = certificate(A[l], B[l])
        if q != Nprime or q <= N:
            continue
        Ct, _, _ = certificate(A[l], B[l], keep=N)                  # truncated back to rank r - N
        _, _, Vh = torch.linalg.svd(B[l], full_matrices=False)
        Rt, Rh = Vh[:q].T, Vh[:N].T                                 # R and Rhat
        P = Rt @ Rt.T - Rh @ Rh.T                                   # P_{R (-) Rhat}
        lhs, rhs = Ct @ Hb[l], P @ (A[l] @ Hb[l])
        ident = max(ident, float((lhs - rhs).norm() / lhs.norm().clamp_min(1e-300)))
        gaps.append(float(sB[N - 1] / sB[N]))
    ok3 = record("T2.1_exact_error_identity", ident < 1e-10, rel_err=f"{ident:.2e}", tol="1e-10", gaps=gaps)
    return ok1 and ok2 and ok3


def check_T2_death(dev, seed=0):
    """Cor. A.1 / T4-C1: rank C_full = r - N' collapses to 0 as the training span reaches r."""
    N, r, dims = 2, 6, [10, 9, 9, 4]
    net = MultiLoRANet(dims, r, N, seed=seed, dev=dev)
    x = torch.randn(dims[0], N, generator=torch.Generator().manual_seed(seed + 1)).to(dev)
    y = torch.arange(N) % dims[-1]
    curve = []
    for T in range(1, 9):
        A, B, reps, lr_used, _ = _largest_sane_lr(net, x, y, T)
        if A is None:
            return record("CorA1_rank_death_by_span_inflation", False, error=f"no sane lr at T={T}")
        l = 1
        Nprime, _ = span_of([rt[l] for rt in reps])
        C, q, sB = certificate(A[l], B[l])
        rC, _ = numrank(C, ref=float(A[l].norm()))
        exp_rC = max(0, min(r - Nprime, dims[l] - Nprime))
        curve.append(dict(T=T, N_prime=Nprime, rank_B=q, rank_C=rC, expect=exp_rC, lr=lr_used))
        log(f"    T={T}: N'={Nprime}  rank B_T={q}  rank C_full={rC} (expect {exp_rC})  (r={r})  "
            f"sigma(B_T)[:4]={[f'{v:.1e}' for v in sB[:4].tolist()]}")
    mono = all(curve[i]["rank_C"] >= curve[i + 1]["rank_C"] for i in range(len(curve) - 1))
    matches = all(rw["rank_C"] == rw["expect"] for rw in curve)
    died = curve[-1]["rank_C"] == 0 or curve[-1]["N_prime"] >= min(r, dims[1])
    return record("CorA1_rank_death_by_span_inflation", mono and died and matches, curve=curve,
                  rank_formula_holds=matches,
                  note="rank C must decrease monotonically and reach 0 as N' -> r")


# ------------------------------------------------------------------ T3
def check_T3(dev, seed=0):
    """The closed-form first-order coefficient of theory/T3.1, the slope-1 law, and the in-span cancellation."""
    n, r, m, eta = 30, 9, 7, 0.7
    g = torch.Generator().manual_seed(seed + 5)
    A0 = (torch.randn(r, n, generator=g) / math.sqrt(n)).to(dev)
    h = torch.randn(n, 1, generator=g).to(dev)
    d0 = torch.randn(m, 1, generator=g).to(dev)
    d1 = torch.randn(m, 1, generator=g).to(dev)
    dperp = torch.randn(n, 1, generator=g).to(dev)
    dperp = dperp - h * (h.T @ dperp) / (h.T @ h)                       # a drift direction OUT of span{h}

    def run(delta):
        x = A0 @ h
        xi = A0 @ delta
        B2 = -eta * ((d0 + d1) @ x.T + d1 @ xi.T)
        A2 = A0 + eta ** 2 * float(d0.T @ d1) * (x @ (h + delta).T)
        Cf, q, sB = certificate(A2, B2)
        Ct, _, _ = certificate(A2, B2, keep=1)
        c = 1.0 + eta ** 2 * float(d0.T @ d1) * float(h.T @ h + delta.T @ h)     # A_2 h = c * x
        return float((Cf @ h).norm()), float((Ct @ h).norm()), x, c

    # (a) full certificate exact; (b) truncated matches the closed form; (c) slope 1
    xs, ys, worst_full, worst_coef = [], [], 0.0, 0.0
    for e in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        delta = e * dperp * float(h.norm()) / float(dperp.norm())
        nf, nt, x, c = run(delta)
        u = x / x.norm()
        pred = abs(c) * float((A0 @ delta - u * (u.T @ (A0 @ delta))).norm()
                              * abs(float((d0 + d1).T @ d1)) / float(((d0 + d1).T @ (d0 + d1))))
        worst_full = max(worst_full, nf / float(h.norm()))
        worst_coef = max(worst_coef, abs(nt - pred) / pred)
        xs.append(math.log(e)); ys.append(math.log(nt))
    nn = len(xs); mx = sum(xs) / nn; my = sum(ys) / nn
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
    ok1 = record("T3_full_certificate_exact", worst_full < 1e-12, max_resid=f"{worst_full:.2e}", tol="1e-12")
    ok2 = record("T3.1_closed_form_coefficient", worst_coef < 2e-2, max_rel_err=f"{worst_coef:.2e}", tol="2e-2")
    ok3 = record("T3.1_first_order_slope_is_one", abs(slope - 1) < 0.02, slope=f"{slope:.4f}",
                 tol="|slope-1|<0.02", note="an O(eps^2) rate would give slope 2 -- the conjecture is FALSE")

    # (d) T3.2: drift INSIDE span{h} costs nothing, at 100% drift
    nf, nt, _, _ = run(1.0 * h)
    ok4 = record("T3.2_in_span_drift_is_free", nt / float(h.norm()) < 1e-12,
                 resid=f"{nt / float(h.norm()):.2e}", tol="1e-12", note="drift = 100% of ||h||, still exact")
    return ok1 and ok2 and ok3 and ok4


# ------------------------------------------------------------------ T5
def check_T5(dev, seed=0):
    """T5.1 nesting ceiling, T5.2 additivity min(k_1, sum q_l), and the shared-seed collapse (R2)."""
    N, k, r, dims = 3, 20, 12, [40, 30, 30, 30, 8]
    net = MultiLoRANet(dims, r, N, seed=seed, dev=dev)
    gg = torch.Generator().manual_seed(seed + 3)
    V = torch.linalg.qr(torch.randn(dims[0], k, generator=gg))[0].to(dev)
    mean = torch.randn(dims[0], 1, generator=gg).to(dev)
    zstar = torch.randn(k, 1, generator=gg).to(dev)
    G = lambda z: mean + V @ z.reshape(k, -1)
    Xpriv = G(torch.randn(k, N, generator=gg).to(dev))
    Xpriv[:, :1] = G(zstar)
    Hb = net.base_reps(Xpriv)

    shared_base = (torch.randn(r, max(dims), generator=torch.Generator().manual_seed(seed + 77))).to(dev)
    out = {}
    for tag in ("independent", "shared_seed"):
        Cs = []
        for l in range(net.L):
            A0l = net.A0[l] if tag == "independent" else shared_base[:, :dims[l]] / math.sqrt(dims[l])
            U = torch.linalg.qr(Hb[l])[0][:, :N]                       # basis of col(H_l^0)
            Qx = torch.linalg.qr(A0l @ U)[0]                           # col(X_l), the zero-drift row space
            Cs.append(A0l - Qx @ (Qx.T @ A0l))

        def Fl(z, l):
            return (Cs[l] @ net.base_reps(G(z))[l][:, :1]).reshape(-1)

        blocks = [torch.autograd.functional.jacobian(lambda z, l=l: Fl(z, l), zstar.reshape(-1))
                  for l in range(net.L)]
        M1 = torch.autograd.functional.jacobian(
            lambda z: net.base_reps(G(z))[0][:, :1].reshape(-1), zstar.reshape(-1))
        k1, _ = numrank(M1, rtol=1e-9)
        qs = [numrank(b, rtol=1e-9)[0] for b in blocks]
        stacked = [numrank(torch.cat(blocks[:L], 0), rtol=1e-9)[0] for L in range(1, net.L + 1)]
        preds = [min(k1, sum(qs[:L])) for L in range(1, net.L + 1)]
        out[tag] = dict(k1=k1, q_l=qs, rank_stacked=stacked, predicted=preds)
        log(f"    {tag}: k1={k1}  q_l={qs}  rank J_F(1..L)={stacked}  predicted={preds}")

    ind = out["independent"]
    ok1 = record("T5.1_nesting_ceiling", all(rk <= ind["k1"] for rk in ind["rank_stacked"]),
                 k1=ind["k1"], ranks=ind["rank_stacked"])
    ok2 = record("T5.2_additivity_min_k1_sum_q", ind["rank_stacked"] == ind["predicted"],
                 observed=ind["rank_stacked"], predicted=ind["predicted"])
    sh = out["shared_seed"]
    collapsed = sh["rank_stacked"][-1] < ind["rank_stacked"][-1]
    ok3 = record("T5_R2_shared_seed_observation", True,                       # OBSERVATION, not a pass/fail claim
                 shared=sh["rank_stacked"], independent=ind["rank_stacked"], collapsed=bool(collapsed),
                 note=("tying A_0 across layers DID reduce the stacked rank" if collapsed else
                       "tying A_0 across layers did NOT reduce the stacked rank -- the certificates still "
                       "differ through the layer-specific feature span U_l and Jacobian M_l, so R2 is NOT a "
                       "defence on its own. T5's defence claim is downgraded accordingly."))
    return ok1 and ok2 and ok3


# ------------------------------------------------------------------ T6
def check_T6(dev, seed=0):
    """T6.2: ||zhat - z*|| <= 4 rho / sigma_min, with rho varied by the truncation level."""
    N, k, r, dims, T = 3, 6, 20, [30, 24, 24, 5], 4
    net = MultiLoRANet(dims, r, N, seed=seed, dev=dev)
    gg = torch.Generator().manual_seed(seed + 11)
    V = torch.linalg.qr(torch.randn(dims[0], k, generator=gg))[0].to(dev)
    mean = torch.randn(dims[0], 1, generator=gg).to(dev)
    Z = torch.randn(k, N, generator=gg).to(dev)
    G = lambda z: mean + V @ z.reshape(k, -1)
    X = G(Z)
    y = torch.arange(N) % dims[-1]
    A, B, reps = net.train(X, y, T=T, lr=0.4)
    l = 1
    rows = []
    for keep in [N, N + 1, None]:
        C, q, _ = certificate(A[l], B[l], keep=keep)
        F = lambda z: (C @ net.base_reps(G(z))[l][:, :1]).reshape(-1)
        zs = Z[:, :1].reshape(-1).clone()
        rho = float(F(zs).norm())
        J = torch.autograd.functional.jacobian(F, zs)
        s = torch.linalg.svdvals(J)
        sigma = float(s[min(k, len(s)) - 1])
        z = (zs + 0.02 * torch.randn(k, generator=gg).to(dev)).clone().requires_grad_(True)
        for _ in range(6):
            opt = torch.optim.LBFGS([z], max_iter=500, tolerance_grad=1e-18, tolerance_change=1e-20,
                                    line_search_fn="strong_wolfe")
            opt.step(lambda: _closure(opt, z, F))
        err = float((z.detach() - zs).norm())
        rho_hat = float(F(z.detach()).norm())
        bound = 4 * rho / sigma if sigma > 0 else float("inf")
        # S3 of theory/T6: the statement is about a MINIMISER. If the optimiser stopped above the residual at
        # the truth, its hypothesis is not met and the row tests the solver, not the theorem -- say so.
        s3 = rho_hat <= rho * (1 + 1e-6) + 1e-300
        rows.append(dict(keep=q, rho=rho, rho_at_zhat=rho_hat, sigma_min=sigma, err=err, bound=bound,
                         S3_minimiser_hypothesis_met=bool(s3), holds=bool((not s3) or err <= bound + 1e-12)))
        log(f"    keep={q}: rho={rho:.3e}  rho(zhat)={rho_hat:.3e}  sigma_min={sigma:.3e}  "
            f"||zhat-z*||={err:.3e}  bound 4rho/sigma={bound:.3e}  S3={'met' if s3 else 'NOT met (solver-limited)'}")
    tested = [rw for rw in rows if rw["S3_minimiser_hypothesis_met"]]
    return record("T6.2_stability_bound_holds", bool(tested) and all(rw["holds"] for rw in tested),
                  rows=rows, n_rows_testing_the_theorem=len(tested),
                  note="rows where S3 is not met are solver-limited and test nothing")


def _closure(opt, z, F):
    opt.zero_grad()
    loss = (F(z) ** 2).sum()
    loss.backward()
    return loss


# ------------------------------------------------------------------ main
CHECKS = dict(T1=check_T1, T2=check_T2, T2death=check_T2_death, T3=check_T3, T5=check_T5, T6=check_T6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="all", help="all or a comma list of " + ",".join(CHECKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/multilayer_cert/theory_checks.jsonl")
    a = ap.parse_args()
    names = list(CHECKS) if a.check == "all" else a.check.split(",")
    dev = a.device
    log(f"# multilayer certificate theory checks | device={dev} | seed={a.seed} | host={socket.gethostname()}")
    try:
        githash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        githash = "?"
    t0, allok = time.time(), True
    for nm in names:
        log(f"\n== {nm} ==")
        try:
            allok &= bool(CHECKS[nm](dev, a.seed))
        except Exception as e:
            allok = False
            record(f"{nm}_CRASHED", False, error=repr(e)[:400])
            import traceback; traceback.print_exc()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        for rrow in RES:
            f.write(json.dumps(dict(rrow, seed=a.seed, git=githash, host=socket.gethostname(),
                                    cmd=" ".join(sys.argv), seconds=time.time() - t0)) + "\n")
    log(f"\n# {'ALL CHECKS PASSED' if allok else 'SOME CHECKS FAILED'} "
        f"({sum(r['passed'] for r in RES)}/{len(RES)}) in {time.time()-t0:.1f}s -> {a.out}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
