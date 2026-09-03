#!/usr/bin/env python3
"""The recipe is a SEQUENCE, not a scalar.  What survives?

The user's objection (2026-09-03): the learning rate need not be constant.  Schedules, warmup, decay,
weight decay, adaptivity — the "recipe" is a function of t, not a number.

Three things are tested here.

A. WEIGHT DECAY IS NOT AN OBSTACLE, IT IS A SECOND REGRESSOR.  One probe step under SGD with decay gives
   dB = -eta*gB - eta*wd*B_T exactly.  The attacker knows BOTH gB and B_T, so this is a 2-parameter least
   squares in span{-gB, -B_T}, not a 1-parameter one.  The earlier 1-parameter fit left cos = 0.9999987
   and a 3.1e-4 error; the 2-parameter fit should be exact.

B. A NON-CONSTANT SCHEDULE HELPS THE ATTACKER RATHER THAN HURTING.  Probing reveals eta at the CONTINUATION
   steps, i.e. eta_T, eta_{T+1}, ..., not the history eta_0..eta_{T-1}.  With a CONSTANT rate that is all
   you get, and T stays hidden.  But if the schedule is a known parametric family, its local value and
   curvature pin down WHERE ON THE CURVE the release sits — which recovers T.  Cosine:
       eta(t) = eta_0/2 * (1 + cos(pi * t / T_max))
   Three consecutive probe values give eta, its first and second difference, and hence (eta_0, T_max, t).
   A decaying schedule therefore leaks the step count that a constant one hides.

C. A FREE PER-STEP SEQUENCE IS NOT IDENTIFIABLE, BY COUNTING.  Recipe unknowns add to demand:
   Nk + rN + p <= N((m-1)+r-N) + rN, so p <= N(m-1+r-N) - Nk.  At the standard cell that is 120, so a free
   schedule is identifiable only for T <= 120 and is hopeless at T = 400.  Parametric schedules (p = 2-4)
   are comfortably affordable.  This is a statement about identifiability, not about any solver.

  python -m experiments.exact_inversion.schedule_probe --schedule cosine --T 400 --lr 0.01
"""
import argparse, json, math, socket, sys
import torch

from experiments.exact_inversion.lora_exact_inversion import World, softmax_cols, git_hash

torch.set_default_dtype(torch.float64)


def eta_at(t, base, T_max, kind):
    if kind == "const":  return base
    if kind == "cosine": return 0.5 * base * (1.0 + math.cos(math.pi * min(t, T_max) / T_max))
    if kind == "linear": return base * max(0.0, 1.0 - t / T_max)
    raise ValueError(kind)


@torch.no_grad()
def train_scheduled(H, A0, W0, y, m, T, base, T_max, kind, wd=0.0):
    """Release generator with a per-step learning rate.  Same recipe family as the main testbed
       otherwise (plain SGD, softmax CE, B_0 = 0)."""
    N = H.shape[1]; r = A0.shape[0]
    Y = torch.eye(m, device=H.device)[y].T
    A = A0.clone(); B = torch.zeros(m, r, device=H.device)
    for t in range(T):
        lr = eta_at(t, base, T_max, kind)
        D = (softmax_cols(W0 @ H + B @ (A @ H)) - Y) / N
        gB = D @ (A @ H).T; gA = B.T @ D @ H.T
        B, A = (1 - lr * wd) * B - lr * gB, (1 - lr * wd) * A - lr * gA
    return A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=["const", "cosine", "linear"], default="cosine")
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--T-max", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--Nprobe", type=int, default=6); ap.add_argument("--probe-steps", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(a.n)
    dev = torch.device(a.device)
    world = World(a.k, a.P, a.n, a.seed, dev)
    g = torch.Generator().manual_seed(a.seed + 7)
    N, r, m, n, k = a.N, a.r, a.m, a.n, a.k

    W_true = torch.randn(k, N, generator=g).to(dev)
    H = world.phi(world.psi(W_true))
    y = (torch.arange(N) % m).to(dev)
    W0 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(dev)
    A0 = (a.sigma0 * torch.randn(r, n, generator=g)).to(dev)
    A_T, B_T = train_scheduled(H, A0, W0, y, m, a.T, a.lr, a.T_max, a.schedule, a.wd)

    gp = torch.Generator().manual_seed(a.seed + 991)
    Wp = torch.randn(k, a.Nprobe, generator=gp).to(dev)
    Hp = world.phi(world.psi(Wp))
    Yp = torch.eye(m, device=dev)[(torch.arange(a.Nprobe) % m).to(dev)].T

    # ---- probe: continue training the RELEASED adapter with the attacker's own data ----
    A, B = A_T.clone(), B_T.clone(); etas_true = []; etas_hat = []; wds_hat = []
    for j in range(a.probe_steps):
        t = a.T + j
        lr_true = eta_at(t, a.lr, a.T_max, a.schedule); etas_true.append(lr_true)
        with torch.no_grad():
            AH = A @ Hp
            D = (softmax_cols(W0 @ Hp + B @ AH) - Yp) / a.Nprobe
            gB = D @ AH.T; gA = B.T @ D @ Hp.T
            Bn = (1 - lr_true * a.wd) * B - lr_true * gB
            An = (1 - lr_true * a.wd) * A - lr_true * gA
            dB = Bn - B
            # 2-parameter least squares: dB ~ c1*(-gB) + c2*(-B).  c1 = eta, c2 = eta*wd.
            M = torch.stack([(-gB).reshape(-1), (-B).reshape(-1)], dim=1)
            coef = torch.linalg.lstsq(M, dB.reshape(-1, 1)).solution.reshape(-1)
            fit_resid = float(torch.linalg.norm(M @ coef.reshape(-1, 1) - dB.reshape(-1, 1)) /
                              torch.linalg.norm(dB))
            etas_hat.append(float(coef[0])); wds_hat.append(float(coef[1] / coef[0]) if abs(float(coef[0])) > 0 else float("nan"))
            A, B = An, Bn
    # ---- from the probed eta sequence, can we recover (base, T_max, T)?  cosine only ----
    rec = {}
    if a.schedule == "cosine" and a.probe_steps >= 3:
        e = etas_hat
        # eta(t) = base/2 (1 + cos(pi t / T_max)).  Solve for (base, T_max, t) from three samples by
        # a small 3-parameter least squares on the probed values.
        # Fit in LOG/SCALED coordinates: base ~ 1e-2, T_max ~ 1e3, t ~ 1e2 differ by five orders, and an
        # unscaled optimiser simply does not move (an earlier version returned its own initialisation).
        # Also report the conditioning of the fit, because over a short probe window the cosine is nearly
        # linear and (base, T_max, t) are genuinely close to degenerate -- that would be a real
        # obstruction, not an optimiser artefact, and the two must be told apart.
        tgt = torch.tensor(e, device=dev)
        idx = torch.arange(len(e), device=dev, dtype=torch.float64)

        def model(q):                       # q = (log base, log T_max, log t)
            base, Tm, t0 = torch.exp(q[0]), torch.exp(q[1]), torch.exp(q[2])
            return 0.5 * base * (1 + torch.cos(math.pi * (t0 + idx) / Tm))

        q = torch.tensor([math.log(a.lr * 1.3), math.log(a.T_max * 1.3), math.log(a.T * 1.3)],
                         device=dev, requires_grad=True)
        opt = torch.optim.LBFGS([q], lr=1.0, max_iter=2000, history_size=100,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-18, tolerance_change=1e-20)
        for _ in range(12):
            def closure():
                opt.zero_grad(); loss = ((model(q) - tgt) ** 2).sum(); loss.backward(); return loss
            opt.step(closure)
        with torch.no_grad():
            Jq = torch.autograd.functional.jacobian(model, q.detach())
            sv = torch.linalg.svdvals(Jq)
            fit_cond = float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf")
            fit_res = float(torch.linalg.norm(model(q.detach()) - tgt) / torch.linalg.norm(tgt))
        rec = dict(fit_cond=fit_cond, fit_residual=fit_res, probe_window=len(e),
                   base_hat=float(torch.exp(q[0])), T_max_hat=float(torch.exp(q[1])), T_hat=float(torch.exp(q[2])),
                   base_rel_err=abs(float(torch.exp(q[0])) - a.lr) / a.lr,
                   T_max_rel_err=abs(float(torch.exp(q[1])) - a.T_max) / a.T_max,
                   T_rel_err=abs(float(torch.exp(q[2])) - a.T) / a.T)

    out = dict(schedule=a.schedule, T=a.T, T_max=a.T_max, lr_base=a.lr, wd=a.wd,
               etas_true=etas_true, etas_hat=etas_hat, wd_hat=wds_hat,
               eta_max_rel_err=max(abs(h - t) / t for h, t in zip(etas_hat, etas_true)),
               two_param_fit_residual=fit_resid, **rec,
               N=N, k=k, r=r, m=m, n=n, seed=a.seed, git=git_hash(), host=socket.gethostname(),
               cmd=" ".join(sys.argv))
    print(f"schedule={a.schedule} T={a.T} T_max={a.T_max} base={a.lr} wd={a.wd}", flush=True)
    print(f"  probed eta true : {[f'{x:.6f}' for x in etas_true]}", flush=True)
    print(f"  probed eta est  : {[f'{x:.6f}' for x in etas_hat]}   max rel err {out['eta_max_rel_err']:.2e}", flush=True)
    print(f"  wd estimate     : {[f'{x:.6f}' for x in wds_hat]}   (true {a.wd});  2-param fit residual {fit_resid:.2e}", flush=True)
    if rec:
        print(f"  SCHEDULE FIT: base {rec['base_hat']:.6f} (rel err {rec['base_rel_err']:.2e}), "
              f"T_max {rec['T_max_hat']:.1f} (rel err {rec['T_max_rel_err']:.2e}), "
              f"T {rec['T_hat']:.1f} (TRUE {a.T}, rel err {rec['T_rel_err']:.2e})", flush=True)
        print(f"    fit residual {rec['fit_residual']:.2e}, fit conditioning {rec['fit_cond']:.2e} over a "
              f"{rec['probe_window']}-step probe window", flush=True)
    if a.out:
        with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
