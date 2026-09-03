#!/usr/bin/env python3
"""Is the recipe a MUST?  Three arms.

The exact inversion uses the training recipe AS the forward model, so "the attacker knows the recipe" is
an assumption by construction.  This module measures how binding it actually is.

  R1  misspecification is self-detecting.  Invert under a deliberately WRONG recipe (eta off by 2x, T off
      by +-25%, wrong optimizer).  Prediction: the residual cannot reach the floor under a wrong map,
      because the release is not a function of the data under that map.  Among a menu of hypotheses the
      correct recipe should be the unique floor-reacher -> the residual is itself a recipe test.
  R2  joint recipe fitting.  Put log(eta) in with (w, X) as a free unknown.  Recipe scalars ADD TO DEMAND,
      so this is affordable only with slack below the capacity line: k < m+r-N - p/N.
  R3  the eta*T degeneracy.  In the small-step limit the endpoint depends on effective time eta*T, so
      (eta, T) should be jointly unidentifiable along that curve.  Is it exact or only asymptotic --
      does discrete SGD at larger eta break it?  Also: are the LABELS identifiable the same way?

Imports the main testbed so the recipe cannot drift, and so a running multi-cell job is undisturbed.
"""
import argparse, itertools, json, math, socket, sys, time
import torch

from experiments.exact_inversion.lora_exact_inversion import (
    World, train_release, simulate_sgd_reduced, simulate_adam_full, qr_canon, git_hash)

torch.set_default_dtype(torch.float64)


def build(a, dev):
    """The private world and the TRUE release."""
    world = World(a.k, a.P, a.n, a.seed, dev)
    g = torch.Generator().manual_seed(a.seed + 7)
    W_true = torch.randn(a.k, a.N, generator=g).to(dev)
    X_img = world.psi(W_true); H = world.phi(X_img)
    y = (torch.arange(a.N) % a.m).to(dev)
    W0 = (torch.randn(a.m, a.n, generator=g) / math.sqrt(a.n)).to(dev)
    A0 = (a.sigma0 * torch.randn(a.r, a.n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, a.m, a.T, a.lr, a.release, a.wd)
    return world, g, W_true, X_img, H, y, W0, A0, A_T, B_T


def make_resvec(world, A_T, B_T, W0, y, a, sim_release, sim_T, sim_lr, fit_eta=False):
    """Residual VECTOR under an ASSUMED recipe (sim_*), which may differ from the true one.
       If fit_eta, the last coordinate of the parameter vector is log(eta) and is solved for."""
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T)
    k, N, r = a.k, a.N, a.r
    aux_shape = (r, N) if sim_release == "sgd" else (r, a.n)
    nW = k * N; nAux = aux_shape[0] * aux_shape[1]

    def res(v):
        W = v[:nW].reshape(k, N); aux = v[nW:nW + nAux].reshape(aux_shape)
        lr = torch.exp(v[nW + nAux]) if fit_eta else sim_lr
        Hc = world.features_from_latents(W)
        if sim_release == "sgd":
            Bs, Xis, Uc = simulate_sgd_reduced(Hc, aux, W0, y, a.m, sim_T, lr, a.wd)
            return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - A_T @ Uc) / nA).reshape(-1)])
        As_, Bs = simulate_adam_full(Hc, aux, W0, y, a.m, sim_T, lr, a.wd)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((As_ - A_T) / nA).reshape(-1)])
    return res, nW, nAux


def lm_solve(res, v, iters=80, lam=1e-2):
    """Levenberg-Marquardt with an autograd Jacobian -- the SAME solver the rest of the study uses.
       Returns (final sum-of-squares, final parameter vector)."""
    import torch.func as tf
    F = res(v); f = float(F @ F); stall = 0
    for _ in range(iters):
        J = tf.jacfwd(res)(v).detach()
        JtJ = J.T @ J; JtF = J.T @ F
        accepted = False
        for _ in range(12):
            step = torch.linalg.solve(JtJ + lam * torch.eye(JtJ.shape[0], device=v.device), JtF)
            vn = v - step; Fn = res(vn)
            if float(Fn @ Fn) < f:
                v, F, f = vn, Fn, float(Fn @ Fn); lam = max(lam / 3, 1e-15); accepted = True; break
            lam *= 5
        stall = 0 if accepted else stall + 1
        if f < 1e-30 or stall >= 2 or lam > 1e12: break
    return f, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["R1", "R2", "R3"], required=True)
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--release", default="sgd"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--lm-iters", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(a.n)
    dev = torch.device(a.device)
    world, g, W_true, X_img, H, y, W0, A0, A_T, B_T = build(a, dev)
    U_true, _ = qr_canon(H)
    aux_true = (A0 @ U_true) if a.release == "sgd" else A0
    prov = dict(arm=a.arm, k=a.k, N=a.N, r=a.r, m=a.m, n=a.n, T=a.T, lr=a.lr, seed=a.seed,
                true_release=a.release, git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
    print(f"# {a.arm}  true recipe: {a.release} T={a.T} lr={a.lr}  cell k={a.k} N={a.N} "
          f"(capacity line k < m+r-N = {a.m + a.r - a.N})\n# git={git_hash()}", flush=True)

    def start():
        W = (W_true + a.init_noise * torch.randn(a.k, a.N, generator=g).to(dev)).clone()
        with torch.no_grad():
            Uc, _ = qr_canon(world.features_from_latents(W))
            aux = (A_T @ Uc).clone() if a.release == "sgd" else A_T.clone()
        return W, aux

    def emit(d):
        d.update(prov); print(json.dumps(d), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(d) + "\n")

    if a.arm == "R1":
        # A menu of recipe hypotheses. Only the true one should reach the reproduction floor.
        menu = [("correct", a.release, a.T, a.lr),
                ("eta x2", a.release, a.T, a.lr * 2), ("eta /2", a.release, a.T, a.lr / 2),
                ("eta x1.01", a.release, a.T, a.lr * 1.01),
                ("T +25%", a.release, int(a.T * 1.25), a.lr), ("T -25%", a.release, int(a.T * 0.75), a.lr),
                ("T +1 step", a.release, a.T + 1, a.lr),
                ("wrong optimizer", "adam" if a.release == "sgd" else "sgd", a.T, a.lr)]
        for name, rel, T_, lr_ in menu:
            W, aux = start()
            if rel == "adam" and a.release == "sgd": aux = A_T.clone()          # Adam nuisance is all of A0
            if rel == "sgd" and a.release == "adam":
                with torch.no_grad():
                    Uc, _ = qr_canon(world.features_from_latents(W)); aux = (A_T @ Uc).clone()
            res, nW, nAux = make_resvec(world, A_T, B_T, W0, y, a, rel, T_, lr_)
            v = torch.cat([W.reshape(-1), aux.reshape(-1)]).detach()
            t0 = time.time(); f, v = lm_solve(res, v, a.lm_iters)
            with torch.no_grad():
                W = v[:nW].reshape(a.k, a.N)
                err = (torch.linalg.norm(world.psi(W) - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
            emit(dict(hypothesis=name, sim_release=rel, sim_T=T_, sim_lr=lr_, residual=f,
                      final_err_max=float(err.max()), final_err_median=float(err.median()),
                      reached_floor=bool(f < 1e-24), seconds=time.time() - t0))

    elif a.arm == "R2":
        for start_scale in (1.0, 2.0, 0.5):
            W, aux = start()
            res, nW, nAux = make_resvec(world, A_T, B_T, W0, y, a, a.release, a.T, a.lr, fit_eta=True)
            v = torch.cat([W.reshape(-1), aux.reshape(-1),
                           torch.tensor([math.log(a.lr * start_scale)], device=dev)]).detach()
            t0 = time.time(); f, v = lm_solve(res, v, a.lm_iters)
            with torch.no_grad():
                W = v[:nW].reshape(a.k, a.N)
                err = (torch.linalg.norm(world.psi(W) - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
                eta_hat = float(torch.exp(v[nW + nAux]))
            emit(dict(eta_start=a.lr * start_scale, eta_fitted=eta_hat, eta_true=a.lr,
                      eta_rel_err=abs(eta_hat - a.lr) / a.lr, residual=f, final_err_max=float(err.max()),
                      reached_floor=bool(f < 1e-24), seconds=time.time() - t0,
                      demand=a.N * a.k + a.r * a.N + 1, supply=a.N * (a.m + a.r - a.N) + a.r * a.N))

    else:  # R3
        # Is the endpoint a function of eta*T only?  Hold eta*T fixed, vary the split.
        prod = a.lr * a.T
        for T_ in (a.T // 4, a.T // 2, a.T, a.T * 2):
            lr_ = prod / T_
            W, aux = start()
            res, nW, nAux = make_resvec(world, A_T, B_T, W0, y, a, a.release, T_, lr_)
            v = torch.cat([W.reshape(-1), aux.reshape(-1)]).detach()
            f, v = lm_solve(res, v, a.lm_iters)
            with torch.no_grad():
                W = v[:nW].reshape(a.k, a.N)
                err = (torch.linalg.norm(world.psi(W) - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
            emit(dict(test="eta_T_product", sim_T=T_, sim_lr=lr_, eta_T=lr_ * T_, residual=f,
                      final_err_max=float(err.max()), reached_floor=bool(f < 1e-24)))
        # Are the LABELS identifiable?  Try a few wrong assignments (swap two, cyclic shift).
        for name, perm in [("true", list(range(a.N))),
                           ("swap 0,1", [1, 0] + list(range(2, a.N))),
                           ("cyclic shift", [(i + 1) % a.N for i in range(a.N)])]:
            y_alt = y[torch.tensor(perm, device=dev)]
            W, aux = start()
            res, nW, nAux = make_resvec(world, A_T, B_T, W0, y_alt, a, a.release, a.T, a.lr)
            v = torch.cat([W.reshape(-1), aux.reshape(-1)]).detach()
            f, v = lm_solve(res, v, a.lm_iters)
            emit(dict(test="labels", labels=name, residual=f, reached_floor=bool(f < 1e-24)))


if __name__ == "__main__":
    main()
