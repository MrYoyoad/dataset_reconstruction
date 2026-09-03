#!/usr/bin/env python3
"""
lora_exact_inversion.py  --  Exact inversion of the LoRA training map (framework Rev 10, Primitive 3, exact form)

Idea (framework_rev10.pdf, Section 5):
    The released LoRA factors (A_T, B_T) are a deterministic function of the private data and of the random
    initialisation.  Under an SGD-class recipe the initialisation enters ONLY through X = A_0 U, where U is an
    orthonormal basis of the private feature span (Theorem 1, normal form).  So instead of assuming a
    representer / gradient-mixture equation with guessed coefficients, we SIMULATE the public recipe on a
    candidate data set and a candidate X, and solve

        find  {w_i}, X   such that   Recipe_T( phi(psi(w_i)), X )  ==  ( B_T , A_T U )

    by backpropagating through the unrolled training loop.  Under Adam the same idea holds with the full A_0
    as unknown (the normal form does not apply), which this script also implements.

Modes
    --release sgd    plain SGD recipe (default).  Unknowns: latents W (k x N) and X (r x N).
    --release adam   Adam recipe.                 Unknowns: latents W (k x N) and A0 (r x n).

Initialisers (--init)
    near      truth + init_noise * N(0,1) in latent space (basin study; NOT attacker-available)
    random    N(0,1) latents (global start; attacker-available)
    span      random start -> project features onto the released span estimate
              row(P_{row(B_T)} A_T) (refined by projecting onto ker C, Prop. 2) -> least-squares pull-back to latents
    cert      random start -> minimise |C phi(psi(w_i))|^2 per image (Primitive-1 anchor) -> start there
    spananchor random start -> minimise the out-of-estimated-span energy |P_{Hhat^perp} phi(psi(w_i))|^2 per image

Everything is FP64.  The toy world (tanh generator -> tanh encoder -> softmax head) is the one used in the
audit experiments (results_rev9.pdf), so numbers should match the finite-difference results there.

Outputs one JSON line per (cell, seed) to --out (append), with provenance (seed, git hash, command line, host),
certificate checks, start/final image errors, residual, a verdict, and timing.  Tensors (true / start / final
images) are saved next to it as .pth.  All numbers are provisional (dagger) until reproduced from the committed
script with a recorded seed.
"""
import argparse, json, math, os, socket, subprocess, sys, time
import torch

torch.set_default_dtype(torch.float64)

SQRT_FLOOR = 1e-300       # see F1 above: keeps d/dv sqrt(v) finite at v = 0 without changing the FP64 value
RECOVER_TOL = 1e-2        # relative image error below which an image counts as recovered
RESID_ZERO = 1e-28        # residual below which the release is reproduced. The residual is a sum of SQUARED
                          # relative Frobenius errors; genuine recovery sits at 9e-31..2e-30, so 1e-16 (a 1e-8
                          # relative reproduction error) would let a merely-stalled run be labelled an alias.


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------------------------------------
# Toy world
# ----------------------------------------------------------------------------------------------------------
class World:
    """k-dim image manifold x = psi(w) = tanh(W2 tanh(W1 w) + b) in R^P;  encoder phi(x) = tanh(E2 tanh(E1 x)) in R^n."""
    def __init__(self, k, P, n, seed, device, gen_hidden=32):
        """gen_hidden caps the manifold dimension: psi factors through it, so the effective dimension is
           min(k, gen_hidden) whatever k is asked for.  It was hardcoded at 32, which silently confounded
           every cell with k > 32 (the generator, not the release, was the binding constraint there).
           Default 32 keeps all earlier runs reproducible; raise it to probe k > 32 honestly."""
        g = torch.Generator().manual_seed(seed)
        self.k, self.P, self.n, self.gen_hidden = k, P, n, gen_hidden
        self.W1 = (torch.randn(gen_hidden, k, generator=g) / math.sqrt(k)).to(device)
        self.W2 = (torch.randn(P, gen_hidden, generator=g) / math.sqrt(gen_hidden)).to(device)
        self.b = (0.7 * torch.randn(P, generator=g)).to(device)           # breaks the x -> -x symmetry
        self.E1 = (torch.randn(128, P, generator=g) / math.sqrt(P)).to(device)
        self.E2 = (torch.randn(n, 128, generator=g) / math.sqrt(128)).to(device)

    def psi(self, W):            # W: (k, N) -> images (P, N)
        return torch.tanh(self.W2 @ torch.tanh(self.W1 @ W) + self.b[:, None])

    def phi(self, Ximg):         # (P, N) -> features (n, N)
        return torch.tanh(self.E2 @ torch.tanh(self.E1 @ Ximg))

    def features_from_latents(self, W):
        return self.phi(self.psi(W))


# ----------------------------------------------------------------------------------------------------------
# Recipes (release generation).  These define the forward model; the attacker is assumed to know them.
# ----------------------------------------------------------------------------------------------------------
def softmax_cols(z):
    z = z - z.max(dim=0, keepdim=True).values
    p = torch.exp(z)
    return p / p.sum(dim=0, keepdim=True)


@torch.no_grad()
def train_release(H, A0, W0, y, m, T, lr, release, wd=0.0, betas=(0.9, 0.999), eps=1e-8):
    """Train LoRA head  z = W0 h + B A h  with softmax cross-entropy on fixed features H (A1)."""
    N = H.shape[1]; r = A0.shape[0]
    Y = torch.eye(m, device=H.device)[y].T
    A = A0.clone(); B = torch.zeros(m, r, device=H.device)
    if release == "adam":
        mA = torch.zeros_like(A); vA = torch.zeros_like(A); mB = torch.zeros_like(B); vB = torch.zeros_like(B)
    for t in range(1, T + 1):
        z = W0 @ H + B @ (A @ H)
        D = (softmax_cols(z) - Y) / N
        gB = D @ (A @ H).T
        gA = B.T @ D @ H.T
        if release == "sgd":
            B, A = (1 - lr * wd) * B - lr * gB, (1 - lr * wd) * A - lr * gA
        else:
            b1, b2 = betas
            mA = b1 * mA + (1 - b1) * gA; vA = b2 * vA + (1 - b2) * gA * gA
            mB = b1 * mB + (1 - b1) * gB; vB = b2 * vB + (1 - b2) * gB * gB
            c1, c2 = 1 - b1 ** t, 1 - b2 ** t
            A = (1 - lr * wd) * A - lr * (mA / c1) / (torch.sqrt(vA / c2 + SQRT_FLOOR) + eps)
            B = (1 - lr * wd) * B - lr * (mB / c1) / (torch.sqrt(vB / c2 + SQRT_FLOOR) + eps)
    return A, B


# ----------------------------------------------------------------------------------------------------------
# Differentiable simulators (attacker side)
# ----------------------------------------------------------------------------------------------------------
def qr_canon(M):
    """QR with diag(R) > 0.  LAPACK picks each pivot sign from the pivot entry, so a bare qr() jumps
       (a whole column flips) when a feature crosses zero; the simulated release would then be
       discontinuous in the candidate data and LM would reject steps that cross the seam (F4)."""
    Q, R = torch.linalg.qr(M)
    sgn = torch.sign(torch.diagonal(R)); sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    return Q * sgn, R * sgn[:, None]


def simulate_sgd_reduced(Hc, X, W0, y, m, T, lr, wd=0.0):
    """SGD in span-adapted coordinates.  Hc: candidate features (n,N); X: candidate A_0 U (r,N).
       Returns simulated (B_T, Xi_T = A_T U, U).  Exact under the normal form; cost is independent of n."""
    N = Hc.shape[1]; r = X.shape[0]
    U, _ = qr_canon(Hc)                              # (n, N) orthonormal basis of the candidate span
    RH = U.T @ Hc                                    # (N, N)  so that Hc = U RH
    Y = torch.eye(m, device=Hc.device)[y].T
    WH = W0 @ Hc
    Xi = X; B = torch.zeros(m, r, device=Hc.device)
    for t in range(T):
        AH = Xi @ RH                                 # = A_t H   (r, N)
        D = (softmax_cols(WH + B @ AH) - Y) / N
        B_new = (1 - lr * wd) * B - lr * D @ AH.T
        Xi = (1 - lr * wd) * Xi - lr * B.T @ D @ RH.T
        B = B_new
    return B, Xi, U


def simulate_adam_full(Hc, A0c, W0, y, m, T, lr, wd=0.0, betas=(0.9, 0.999), eps=1e-8):
    """Adam in full coordinates.  A0c: candidate initialisation (r, n).  Returns simulated (A_T, B_T)."""
    N = Hc.shape[1]; r = A0c.shape[0]
    Y = torch.eye(m, device=Hc.device)[y].T
    A = A0c; B = torch.zeros(m, r, device=Hc.device)
    mA = torch.zeros_like(A); vA = torch.zeros_like(A); mB = torch.zeros_like(B); vB = torch.zeros_like(B)
    b1, b2 = betas
    for t in range(1, T + 1):
        AH = A @ Hc
        D = (softmax_cols(W0 @ Hc + B @ AH) - Y) / N
        gB = D @ AH.T; gA = B.T @ D @ Hc.T
        mA = b1 * mA + (1 - b1) * gA; vA = b2 * vA + (1 - b2) * gA * gA
        mB = b1 * mB + (1 - b1) * gB; vB = b2 * vB + (1 - b2) * gB * gB
        c1, c2 = 1 - b1 ** t, 1 - b2 ** t
        A = (1 - lr * wd) * A - lr * (mA / c1) / (torch.sqrt(vA / c2 + SQRT_FLOOR) + eps)
        B = (1 - lr * wd) * B - lr * (mB / c1) / (torch.sqrt(vB / c2 + SQRT_FLOOR) + eps)
    return A, B


# ----------------------------------------------------------------------------------------------------------
# Certificate (Primitive 1) and span estimator (Prop. 2) -- attacker-computable from the release
# ----------------------------------------------------------------------------------------------------------
def orth(M, tol=1e-9):
    Ub, S, _ = torch.linalg.svd(M, full_matrices=False)
    keep = int((S > tol * S[0]).sum()) if S.numel() and S[0] > 0 else 0
    return Ub[:, :keep]


@torch.no_grad()
def certificate(A_T, B_T, tol=1e-9):
    Us = orth(B_T.T, tol)                                          # basis of row(B_T) in R^r
    C = (torch.eye(A_T.shape[0], device=A_T.device) - Us @ Us.T) @ A_T
    s = torch.linalg.svdvals(C)
    rankC = int((s > tol * s[0]).sum()) if s[0] > 0 else 0
    return C, Us.shape[1], rankC, Us


@torch.no_grad()
def span_estimate(A_T, C, Us):
    """Hhat = row(P_{row(B_T)} A_T)  (n x N basis), refined by projecting onto ker C (Prop. 2)."""
    Hhat = orth((Us @ Us.T @ A_T).T)
    if C.shape[0] > 0 and torch.linalg.norm(C) > 0:
        kerC = orth(torch.eye(A_T.shape[1], device=A_T.device) - torch.linalg.pinv(C) @ C)
        Hhat = orth(kerC @ (kerC.T @ Hhat))
    return Hhat


def principal_angles_deg(U1, U2):
    s = torch.linalg.svdvals(U1.T @ U2).clamp(0, 1)
    return torch.rad2deg(torch.arccos(s))


# ----------------------------------------------------------------------------------------------------------
# Attacker-available initialisers
# ----------------------------------------------------------------------------------------------------------
def lbfgs_fit(loss_fn, W, iters, log=None, tag=""):
    W = W.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([W], lr=1.0, max_iter=iters, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-14, tolerance_change=1e-16)
    def closure():
        opt.zero_grad(); f = loss_fn(W); f.backward(); return f
    f = opt.step(closure)
    if log: log(f"    init[{tag}] loss {float(f):.3e}")
    return W.detach()


def make_init(args, world, A_T, B_T, C, W_true, g, dev, log):
    k, N = args.k, args.N
    if args.init == "near":
        return W_true + args.init_noise * torch.randn(k, N, generator=g).to(dev), {}
    W_rand = torch.randn(k, N, generator=g).to(dev)
    if args.init == "random":
        return W_rand, {}
    info = {}
    if args.init == "cert":
        nC = torch.linalg.norm(C, 2)
        W = lbfgs_fit(lambda W: (torch.linalg.norm(C @ world.features_from_latents(W)) / nC) ** 2,
                      W_rand, args.init_iters, log, "cert")
        with torch.no_grad():
            info["cert_anchor_resid"] = float(torch.linalg.norm(C @ world.features_from_latents(W)) / nC)
        return W, info
    _, _, _, Us = certificate(A_T, B_T)
    Hhat = span_estimate(A_T, C, Us)                                  # (n, N)
    Pperp = torch.eye(A_T.shape[1], device=dev) - Hhat @ Hhat.T
    if args.init == "span":
        with torch.no_grad():
            target = Hhat @ (Hhat.T @ world.features_from_latents(W_rand))
        W = lbfgs_fit(lambda W: ((torch.linalg.norm(world.features_from_latents(W) - target)) / torch.linalg.norm(target)) ** 2,
                      W_rand, args.init_iters, log, "span")
    elif args.init == "spananchor":
        W = lbfgs_fit(lambda W: (torch.linalg.norm(Pperp @ world.features_from_latents(W)) / math.sqrt(N)) ** 2,
                      W_rand, args.init_iters, log, "spananchor")
    else:
        raise ValueError(args.init)
    with torch.no_grad():
        Hf = world.features_from_latents(W)
        info["span_out_frac"] = float(torch.linalg.norm(Pperp @ Hf) / torch.linalg.norm(Hf))
    return W, info


# ----------------------------------------------------------------------------------------------------------
# Inversion
# ----------------------------------------------------------------------------------------------------------
def restart_point(world, A_T, W_init, Xinit, rs, args, g):
    """Re-seed the WHOLE unknown vector on a restart.  Jittering only the latents leaves the r x N nuisance
       block X frozen at a value built from the UN-jittered start -- at the r=16, N=8 work point that is 57%
       of the unknowns never restarted, and the stale X no longer matches the moved candidate span (F5)."""
    if rs == 0:
        return W_init.clone(), Xinit.clone()
    W = W_init + args.restart_noise * torch.randn(W_init.shape, generator=g).to(W_init.device)
    with torch.no_grad():
        if args.release == "sgd":
            U, _ = qr_canon(world.features_from_latents(W))
            aux = A_T @ U                            # rebuild the crude X estimate for the MOVED span
        else:
            aux = A_T + args.restart_noise * torch.randn(A_T.shape, generator=g).to(A_T.device) * A_T.std()
    return W, aux


def invert(world, A_T, B_T, W0, y, args, W_init, Xinit, log=print):
    """LBFGS on the residual of the simulated release.  Returns best (W, aux, residual, timing)."""
    m = args.m; T = args.T; lr = args.lr
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T)

    def residual(W, aux):
        Hc = world.features_from_latents(W)
        if args.release == "sgd":
            Bs, Xis, U = simulate_sgd_reduced(Hc, aux, W0, y, m, T, lr, args.wd)
            target = A_T @ U                         # normalise by |A_T| (fixed), NOT |A_T U| which U (hence W) moves
            return (torch.linalg.norm(Bs - B_T) / nB) ** 2 + (torch.linalg.norm(Xis - target) / nA) ** 2
        else:
            As, Bs = simulate_adam_full(Hc, aux, W0, y, m, T, lr, args.wd)
            return (torch.linalg.norm(Bs - B_T) / nB) ** 2 + (torch.linalg.norm(As - A_T) / nA) ** 2

    best = (None, None, float("inf"))
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
    outer_times = []; n_restarts_used = 0
    for rs in range(args.restarts):
        n_restarts_used += 1
        W, aux = restart_point(world, A_T, W_init, Xinit, rs, args, g)
        W.requires_grad_(True); aux.requires_grad_(True)
        opt = torch.optim.LBFGS([W, aux], lr=1.0, max_iter=args.lbfgs_iter, history_size=50,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-14, tolerance_change=1e-16)
        t0 = time.time(); fval = float("inf")
        for outer in range(args.outer):
            def closure():
                opt.zero_grad()
                f = residual(W, aux)
                f.backward()
                return f
            t1 = time.time()
            f = opt.step(closure)
            outer_times.append(time.time() - t1)
            fprev, fval = fval, float(f)
            log(f"    restart {rs} outer {outer:3d}  residual {fval:.3e}  ({time.time()-t0:.0f}s)")
            if fval < 1e-26 or (outer > 3 and abs(fprev - fval) <= 1e-3 * fval and fval < 1e-24): break
        with torch.no_grad():
            fval = float(residual(W, aux))
        if fval < best[2]:
            best = (W.detach().clone(), aux.detach().clone(), fval)
        if fval < 1e-20: break
    return best + (sum(outer_times) / max(1, len(outer_times)), n_restarts_used, {})


def invert_lm(world, A_T, B_T, W0, y, args, W_init, Xinit, log=print):
    """Levenberg-Marquardt on the residual VECTOR of the simulated release, Jacobian by autograd
       (torch.func.jacfwd, vmapped forward-mode through the unrolled recipe; --jac rev uses jacrev).
       Same semantics as the finite-difference LM of results_rev9 S3b.  Returns (W, aux, residual, sec/iter, restarts)."""
    import torch.func as tf
    m = args.m; T = args.T; lr = args.lr
    nB = torch.linalg.norm(B_T); nA = torch.linalg.norm(A_T)
    k, N = W_init.shape; aux_shape = Xinit.shape; nW = k * N

    def res_vec(v):
        W = v[:nW].reshape(k, N); aux = v[nW:].reshape(aux_shape)
        Hc = world.features_from_latents(W)
        if args.release == "sgd":
            Bs, Xis, U = simulate_sgd_reduced(Hc, aux, W0, y, m, T, lr, args.wd)
            target = A_T @ U
            return torch.cat([((Bs - B_T) / nB).reshape(-1), ((Xis - target) / nA).reshape(-1)])
        As, Bs = simulate_adam_full(Hc, aux, W0, y, m, T, lr, args.wd)
        return torch.cat([((Bs - B_T) / nB).reshape(-1), ((As - A_T) / nA).reshape(-1)])

    jac = tf.jacfwd(res_vec) if args.jac == "fwd" else tf.jacrev(res_vec)
    diag = {}
    best = (None, None, float("inf")); g = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
    it_times = []; n_rs = 0
    for rs in range(args.restarts):
        n_rs += 1
        W, aux0 = restart_point(world, A_T, W_init, Xinit, rs, args, g)
        v = torch.cat([W.reshape(-1), aux0.reshape(-1)]).detach()
        with torch.no_grad(): F = res_vec(v)
        lam = args.lm_lambda; t0 = time.time(); fval = float(F @ F); stall = 0; diag = {}
        for it in range(args.lm_iters):
            t1 = time.time()
            J = jac(v).detach()                                        # (R, P)
            JtJ = J.T @ J; JtF = J.T @ F
            # Staged schedule: solve for the nuisance block X alone first, with the data held fixed.
            # X enters the recipe close to linearly, so this is the cheap half of the problem; the
            # bundle prototype did exactly this (iters_x=10) and reported a larger basin (NOTES.md 2).
            free = slice(nW, JtJ.shape[0]) if it < args.stage_x else slice(0, JtJ.shape[0])
            JtJb = JtJ[free, free]; JtFb = JtF[free]
            # Marquardt scaling: damp with diag(JtJ) instead of I.  The latent block and the A_0 / X
            # block differ in natural scale, so unscaled lam*I preferentially freezes one of them --
            # which is what an ill-conditioned Adam Jacobian (cond ~ 1e8) needs fixed.
            Dmp = torch.diag(torch.diagonal(JtJb).clamp_min(1e-30)) if args.lm_scale == "marquardt" \
                  else torch.eye(JtJb.shape[0], device=v.device)
            accepted = False
            for _ in range(12):
                step_b = torch.linalg.solve(JtJb + lam * Dmp, JtFb)
                step = torch.zeros_like(v); step[free] = step_b
                vn = v - step
                with torch.no_grad(): Fn = res_vec(vn)
                if float(Fn @ Fn) < fval:
                    v, F, fval = vn, Fn, float(Fn @ Fn); lam = max(lam / 3, 1e-15); accepted = True; break
                lam *= 5
            it_times.append(time.time() - t1)
            log(f"    restart {rs} lm-iter {it:3d}  residual {fval:.3e}  lambda {lam:.1e}  ({time.time()-t0:.0f}s)")
            stall = 0 if accepted else stall + 1
            if fval < 1e-30 or stall >= 2 or lam > 1e12:
                if it < args.stage_x: continue                          # never stop during the staged phase
                sv = torch.linalg.svdvals(J)
                diag = dict(lm_iters_used=it + 1, lm_lambda_final=float(lam),
                            jac_cond=float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf"),
                            jac_sigma_min=float(sv[-1]), stop=("converged" if fval < 1e-30 else ("stall" if stall >= 2 else "lambda")))
                break
        if not diag:                                  # loop ran to the iteration cap without breaking
            sv = torch.linalg.svdvals(J)
            diag = dict(lm_iters_used=args.lm_iters, lm_lambda_final=float(lam),
                        jac_cond=float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf"),
                        jac_sigma_min=float(sv[-1]), stop="iteration cap (still descending)")
        if fval < best[2]:
            best = (v[:nW].reshape(k, N).clone(), v[nW:].reshape(aux_shape).clone(), fval); best_diag = dict(diag)
        if fval < 1e-24: break
    return best + (sum(it_times) / max(1, len(it_times)), n_rs, locals().get("best_diag", diag))


def assign_err(D):
    """Greedy one-to-one assignment of reconstructions to ground-truth columns (scipy is absent in the
       default env).  Without it, N reconstructions that all collapsed onto ONE training image would each
       report a tiny 'nearest training image' error and score 1.0 (F6)."""
    D = D.clone(); n = D.shape[0]; out = torch.empty(n, dtype=D.dtype, device=D.device)
    for _ in range(n):
        idx = int(torch.argmin(D)); i, j = idx // D.shape[1], idx % D.shape[1]
        out[i] = D[i, j]; D[i, :] = float("inf"); D[:, j] = float("inf")
    return out


def verdict(err_max, res, frac):
    """Keyed off the WORST image, not the median: torch.median returns the lower median, so a cell that
       recovers exactly half of an even N would otherwise be labelled 'recovered' (F2)."""
    if err_max < RECOVER_TOL:
        return "recovered"
    if frac > 0:
        return f"partial ({frac:.2f} of images recovered)" + ("; residual at the reproduction floor" if res < RESID_ZERO else "")
    if res < RESID_ZERO:
        return "alias (residual at the reproduction floor, wrong image -> non-identifiability)"
    return "optimisation failure (residual not zero)"


def run_cell(args, log=print, save_prefix=None):
    dev = torch.device(args.device)
    world = World(args.k, args.P, args.n, args.seed, dev, args.gen_hidden)
    g = torch.Generator().manual_seed(args.seed + 7)
    N, r, m, n, k = args.N, args.r, args.m, args.n, args.k
    # ---- private data and release ----
    W_true = torch.randn(k, N, generator=g).to(dev)
    X_img = world.psi(W_true); H = world.phi(X_img)
    y = (torch.arange(N) % m).to(dev)
    W0 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(dev)
    A0 = (args.sigma0 * torch.randn(r, n, generator=g)).to(dev)
    A_T, B_T = train_release(H, A0, W0, y, m, args.T, args.lr, args.release, args.wd)
    # ---- certificate report ----
    C, rankB, rankC, Us = certificate(A_T, B_T)
    eps_inv = float(torch.linalg.norm(C @ H) / (torch.linalg.norm(A_T, 2) * torch.linalg.norm(H)))
    cert_norm = float(torch.linalg.norm(C) / torch.linalg.norm(A_T))     # C == 0 (rank B_T = r) makes eps_inv vacuous
    cert_vacuous = bool(cert_norm < 1e-12)
    deform = float(torch.linalg.norm(A_T - A0) / torch.linalg.norm(A0))
    with torch.no_grad():
        Hhat = span_estimate(A_T, C, Us)
        U_true, _ = qr_canon(H)
        span_angles = principal_angles_deg(U_true, Hhat) if Hhat.shape[1] == N else torch.full((N,), float("nan"))
    # ---- forward-model sanity at the truth ----
    with torch.no_grad():
        if args.release == "sgd":
            Bs, Xis, _ = simulate_sgd_reduced(H, A0 @ U_true, W0, y, m, args.T, args.lr, args.wd)
            fwd_check = float(torch.linalg.norm(Bs - B_T) / torch.linalg.norm(B_T))
            fwd_check_A = float(torch.linalg.norm(Xis - A_T @ U_true) / torch.linalg.norm(A_T @ U_true))
        else:
            As, Bs = simulate_adam_full(H, A0, W0, y, m, args.T, args.lr, args.wd)
            fwd_check = float(torch.linalg.norm(Bs - B_T) / torch.linalg.norm(B_T))
            fwd_check_A = float(torch.linalg.norm(As - A_T) / torch.linalg.norm(A_T))
    log(f"  cell k={k} N={N} T={args.T} lr={args.lr} seed={args.seed}: rankB={rankB} rankC={rankC} (r-N={r-N}) eps_inv={eps_inv:.1e} "
        f"cert_norm={cert_norm:.1e}{' VACUOUS (C=0)' if cert_vacuous else ''} deformation={deform:.3f} fwd_check={fwd_check:.1e} fwd_check_A={fwd_check_A:.1e} span_angle_mean={float(span_angles.mean()):.1f}deg")
    # ---- local identifiability at the TRUTH (Prop. 6): rank/conditioning of the simulator Jacobian
    #      evaluated at the ground-truth parameters, not at wherever the solver stopped ----
    if args.jac_at_truth:
        import torch.func as tf
        nB_ = torch.linalg.norm(B_T); nA_ = torch.linalg.norm(A_T)
        aux_true = (A0 @ U_true) if args.release == "sgd" else A0
        nW_ = k * N

        def res_truth(v):
            Wc = v[:nW_].reshape(k, N); aux = v[nW_:].reshape(aux_true.shape)
            Hc = world.features_from_latents(Wc)
            if args.release == "sgd":
                Bs, Xis, Uc = simulate_sgd_reduced(Hc, aux, W0, y, m, args.T, args.lr, args.wd)
                return torch.cat([((Bs - B_T) / nB_).reshape(-1), ((Xis - A_T @ Uc) / nA_).reshape(-1)])
            As_, Bs = simulate_adam_full(Hc, aux, W0, y, m, args.T, args.lr, args.wd)
            return torch.cat([((Bs - B_T) / nB_).reshape(-1), ((As_ - A_T) / nA_).reshape(-1)])

        v_true = torch.cat([W_true.reshape(-1), aux_true.reshape(-1)]).detach()
        Jt = tf.jacfwd(res_truth)(v_true).detach()
        svt = torch.linalg.svdvals(Jt)
        truth_diag = dict(jac_sigma_min_truth=float(svt[-1]), jac_sigma_max_truth=float(svt[0]),
                          jac_cond_truth=float(svt[0] / svt[-1]) if float(svt[-1]) > 0 else float("inf"),
                          jac_rows=int(Jt.shape[0]), jac_cols=int(Jt.shape[1]),
                          jac_full_rank_truth=bool(float(svt[-1]) > 1e-12 * float(svt[0])),
                          res_at_truth=float(torch.linalg.norm(res_truth(v_true))))
        log(f"  Jacobian AT THE TRUTH: {Jt.shape[0]}x{Jt.shape[1]} sigma_min={float(svt[-1]):.3e} "
            f"sigma_max={float(svt[0]):.3e} cond={truth_diag['jac_cond_truth']:.3e} "
            f"full_rank={truth_diag['jac_full_rank_truth']} |res(truth)|={truth_diag['res_at_truth']:.1e}")
    else:
        truth_diag = {}
    # ---- initialiser ----
    W_init, init_info = make_init(args, world, A_T, B_T, C, W_true, g, dev, log)
    with torch.no_grad():
        H_init = world.features_from_latents(W_init)
        if args.release == "sgd":
            U_init, _ = qr_canon(H_init)
            Xinit = A_T @ U_init                    # crude: A_T U ~ X (ignores the deformation)
        else:
            Xinit = A_T.clone()                     # crude: A_T ~ A_0
    start_err = (torch.linalg.norm(world.psi(W_init) - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
    # ---- invert ----
    t0 = time.time()
    solver = invert_lm if args.solver == "lm" else invert
    W_hat, aux_hat, res, sec_outer, n_rs, solver_diag = solver(world, A_T, B_T, W0, y, args, W_init, Xinit, log)
    X_hat = world.psi(W_hat)
    err = (torch.linalg.norm(X_hat - X_img, dim=0) / torch.linalg.norm(X_img, dim=0))
    # nearest training image (any column) -> detects basin hopping / permutation
    dall = torch.cdist(X_hat.T, X_img.T) / torch.linalg.norm(X_img, dim=0)[None, :]
    err_any = dall.min(dim=1).values                       # per-reconstruction nearest ground-truth column
    err_match = assign_err(dall)                           # one-to-one matching: a collapse cannot score well (F6)
    out = dict(release=args.release, k=k, N=N, r=r, r_minus_N=r - N, m=m, n=n, P=args.P, T=args.T, lr=args.lr, wd=args.wd,
               sigma0=args.sigma0, seed=args.seed, git=git_hash(), host=socket.gethostname(), device=args.device,
               cmd=" ".join(sys.argv),
               **truth_diag, rankB=rankB, rankC=(0 if cert_vacuous else rankC), cert_norm=cert_norm, cert_vacuous=cert_vacuous, eps_inv=eps_inv, deformation=deform, fwd_check=fwd_check, fwd_check_A=fwd_check_A,
               span_angle_mean_deg=float(span_angles.mean()), span_angle_max_deg=float(span_angles.max()),
               init=args.init, init_noise=args.init_noise, restarts=args.restarts, restarts_used=n_rs,
               restart_noise=args.restart_noise, solver=args.solver, lm_scale=args.lm_scale, stage_x=args.stage_x, **solver_diag, outer=args.outer, lbfgs_iter=args.lbfgs_iter, lm_iters=args.lm_iters, **init_info,
               start_err_median=float(start_err.median()), start_err_max=float(start_err.max()),
               final_err_median=float(err.median()), final_err_max=float(err.max()),
               final_err_any_median=float(err_any.median()), final_err_matched_median=float(err_match.median()),
               final_err_matched_max=float(err_match.max()),
               frac_recovered=float((err < RECOVER_TOL).double().mean()),
               frac_recovered_any=float((err_any < RECOVER_TOL).double().mean()),
               frac_recovered_matched=float((err_match < RECOVER_TOL).double().mean()),
               residual=res, verdict=verdict(float(err.max()), res, float((err < RECOVER_TOL).double().mean())),
               sec_per_iter=sec_outer, seconds=time.time() - t0)
    log(json.dumps(out))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(out) + "\n")
    if save_prefix:
        torch.save(dict(x_true=X_img.cpu(), x_init=world.psi(W_init).cpu(), x_hat=X_hat.cpu(), W_true=W_true.cpu(),
                        W_hat=W_hat.cpu(), aux_hat=aux_hat.cpu(), A_T=A_T.cpu(), B_T=B_T.cpu(), meta=out),
                   f"{save_prefix}_{args.release}_k{k}_N{N}_T{args.T}_{args.init}{args.init_noise}_s{args.seed}.pth")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--k", type=int, default=12); ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=16); ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--gen-hidden", type=int, default=32,
                    help="generator hidden width; CAPS the manifold dimension at min(k, this). Must exceed k.")
    ap.add_argument("--T", type=int, default=1500); ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=0.0); ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="run the cell for each seed (overrides --seed)")
    ap.add_argument("--init", choices=["near", "random", "span", "cert", "spananchor"], default="near")
    ap.add_argument("--init-noise", type=float, default=0.10)
    ap.add_argument("--init-iters", type=int, default=300, help="LBFGS iterations for the span/cert pre-solve")
    ap.add_argument("--restarts", type=int, default=1); ap.add_argument("--restart-noise", type=float, default=0.1)
    ap.add_argument("--solver", choices=["lm", "lbfgs"], default="lm", help="lm = Levenberg-Marquardt with autograd Jacobian (default)")
    ap.add_argument("--jac", choices=["fwd", "rev"], default="fwd", help="autograd mode for the LM Jacobian")
    ap.add_argument("--lm-iters", type=int, default=60); ap.add_argument("--lm-lambda", type=float, default=1e-2)
    ap.add_argument("--lm-scale", choices=["identity", "marquardt"], default="identity",
                    help="damping matrix: lam*I (as run so far) or lam*diag(JtJ) (Marquardt scaling)")
    ap.add_argument("--stage-x", type=int, default=0,
                    help="solve for the nuisance block (X or A0) alone for this many LM iterations first")
    ap.add_argument("--outer", type=int, default=30); ap.add_argument("--lbfgs-iter", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="JSONL results file (append)")
    ap.add_argument("--save-prefix", default=None, help="save tensors to <prefix>_<cell>.pth")
    ap.add_argument("--jac-at-truth", action="store_true",
                    help="also report rank/conditioning of the simulator Jacobian AT THE GROUND TRUTH (Prop. 6)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="(N,k) grid in the current release mode -> --sweep-out")
    ap.add_argument("--sweep-out", default="sweep.json")
    ap.add_argument("--sweep-Ns", type=int, nargs="*", default=[2, 4, 6, 8, 10, 12, 14])
    ap.add_argument("--sweep-ks", type=int, nargs="*", default=[2, 4, 6, 8, 10, 12, 14])
    args = ap.parse_args()
    if args.sigma0 is None: args.sigma0 = 1.0 / math.sqrt(args.n)
    ks_probed = args.sweep_ks if args.sweep else [args.k]
    if max(ks_probed) >= args.gen_hidden:
        print(f"# WARNING: k up to {max(ks_probed)} >= gen_hidden {args.gen_hidden}: the generator caps the "
              f"manifold dimension, so those cells test the GENERATOR, not the release. Raise --gen-hidden.",
              flush=True)
    if args.P < max(ks_probed):
        print(f"# WARNING: P={args.P} < k={max(ks_probed)}: the image space itself is smaller than the chart.",
              flush=True)
    print(f"# device={args.device} release={args.release} r={args.r} m={args.m} n={args.n} git={git_hash()} "
          f"host={socket.gethostname()}\n# cmd: {' '.join(sys.argv)}", flush=True)
    log = (lambda s: None) if args.quiet else (lambda s: print(s, flush=True))
    seeds = args.seeds if args.seeds else [args.seed]
    if not args.sweep:
        for s in seeds:
            args.seed = s
            run_cell(args, log=log, save_prefix=args.save_prefix)
        return
    results = []
    for N in args.sweep_Ns:
        for k in args.sweep_ks:
            for s in seeds:
                a = argparse.Namespace(**vars(args)); a.N = N; a.k = k; a.seed = s
                results.append(run_cell(a, log=lambda s: None, save_prefix=args.save_prefix))
                print(json.dumps(results[-1]), flush=True)
            json.dump(results, open(args.sweep_out, "w"), indent=1)


if __name__ == "__main__":
    main()
