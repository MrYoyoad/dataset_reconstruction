"""Does the AFFINE-chart degeneracy break the replay route, or only the certificate route?

The lemma (0a, R1) says: if the composition Psi = phi . psi from chart coordinates to the adapted
layer's input is AFFINE, then every affine combination of the privates is an exact zero of C, so the
truths sit on a flat of dimension min(N-1, k) and are never isolated -- at any k, however far below
the capacity line. Job 311215 confirmed that for the CERTIFICATE route: exact zeros at the blends,
coefficients summing to 1.000, isolation rank deficit exactly N-1, nothing recovered.

It says NOTHING about the replay route, and the nesting runs the wrong way to inherit it:

        {truth}  subset  {rho = 0}  subset  {C h = 0}

so rho = 0 implies C h = 0 and never the converse. A blend is an exact zero of C; it is not
generally a zero of rho, because reproducing B_T means reproducing what the recipe actually did and
the recipe run on blends does not give B_T back. Three lanes spent an evening arguing this. It is a
one-cell question and this is the cell.

Design: k is chosen BELOW BOTH capacity lines and ABOVE N-1, so that
  - the certificate count k < r - N' is satisfied  (nothing is explained by running out of equations)
  - the replay count k < m + r - N' is satisfied
  - the affine hull of the N truths has dimension N-1 < k, so the chart is not trivially inside it
and the ONLY hypothesis that fails is A8. Both arms then run on the SAME release from the SAME
random starts, and the only difference between them is which residual is minimised.

PRE-REGISTERED (written before the rows):
  certificate arm -- reaches the truths' own residual at points that are exact affine combinations
    (coefficient sums 1.000), lands nothing, isolation rank = k - (N-1).
  replay arm      -- OPEN. If it recovers, the chart's role is route-specific and identifiability is
    a property of the (release, chart) PAIR in the sharpest sense. If it fails too, the impossibility
    covers both routes and can be stated without hedging. Either way the argument ends.
"""
import argparse, json, math, os, socket, time
import torch
from experiments.exact_inversion.lora_exact_inversion import (
    train_release, simulate_sgd_reduced, qr_canon, invert_lm, git_hash)
from experiments.exact_inversion.certificate import certificate, lm_cert

torch.set_default_dtype(torch.float64)


class AffineWorld:
    """psi affine and phi the identity: the adapted layer IS the input layer and the chart is linear,
       so Psi = phi . psi is affine. This is exactly A8's failure case, by construction rather than
       by approximation -- unlike a conv-autoencoder chart, where the blends lie only NEAR the image."""
    def __init__(self, k, P, seed, device):
        g = torch.Generator().manual_seed(seed)
        self.k, self.P, self.n = k, P, P
        self.L = (torch.randn(P, k, generator=g) / math.sqrt(k)).to(device)
        self.b = (0.7 * torch.randn(P, generator=g)).to(device)

    def psi(self, W):  return self.L @ W + self.b[:, None]
    def phi(self, X):  return X
    def features_from_latents(self, W): return self.phi(self.psi(W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12);  ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", type=int, default=24);  ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--starts", type=int, default=60)
    ap.add_argument("--cert-iters", type=int, default=300)
    ap.add_argument("--lm-iters", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device
    k, N, r, m, P = a.k, a.N, a.r, a.m, a.P
    assert k > N - 1,      f"k={k} must exceed N-1={N-1} or the chart lies inside the affine hull entirely"
    assert k < r - N,      f"k={k} must be below the certificate line r-N={r-N}"
    assert k < m + r - N,  f"k={k} must be below the replay line m+r-N={m+r-N}"

    g = torch.Generator().manual_seed(a.seed)
    world = AffineWorld(k, P, a.seed, dev)
    W_true = torch.randn(k, N, generator=g).to(dev)
    H = world.features_from_latents(W_true)                       # (P, N) = the private layer inputs
    sigma0 = a.sigma0 if a.sigma0 else 1.0 / math.sqrt(P)
    A0 = (sigma0 * torch.randn(r, P, generator=g)).to(dev)
    W0 = (torch.randn(m, P, generator=g) / math.sqrt(P)).to(dev)
    y = torch.arange(N, device=dev) % m

    A_T, B_T = train_release(H, A0, W0, y, m, T=a.T, lr=a.lr, release="sgd")
    C, Np, sing = certificate(A_T, B_T, tol=1e-12)
    CH = float(torch.linalg.norm(C @ H) / (torch.linalg.norm(C) * torch.linalg.norm(H)))
    rank_C = int(torch.linalg.matrix_rank(C, rtol=1e-10))

    # THE LEMMA'S OWN PREDICTION, checked before either arm runs: DPsi = L is constant for an affine
    # chart, so the isolation test is a single number for the whole chart rather than one per point.
    J = C @ world.L
    iso_rank = int(torch.linalg.matrix_rank(J, rtol=1e-10))

    feat = lambda W: world.features_from_latents(W)
    obj_cert = lambda w: (torch.linalg.norm(C @ feat(w.reshape(k, 1))) /
                          torch.linalg.norm(A_T @ feat(w.reshape(k, 1))))
    res_truth = [float(obj_cert(W_true[:, i])) for i in range(N)]

    rows = []
    gs = torch.Generator().manual_seed(a.seed + 31)
    for s in range(a.starts):
        w0 = torch.randn(k, 1, generator=gs).to(dev).reshape(-1)

        # ---- ARM A: the certificate route (no recipe, no start needed)
        fun = lambda w: (C @ feat(w.reshape(k, 1)) /
                         torch.linalg.norm(A_T @ feat(w.reshape(k, 1)))).reshape(-1)
        wc, oc, itc = lm_cert(fun, w0.clone(), a.cert_iters)
        xc = world.psi(wc.reshape(k, 1))[:, 0]
        ec = min(float(torch.linalg.norm(xc - H[:, i]) / torch.linalg.norm(H[:, i])) for i in range(N))
        cc, *_ = torch.linalg.lstsq(H, xc.unsqueeze(1))            # blend coefficients against the privates
        cc = cc.squeeze(1)
        blend_res = float(torch.linalg.norm(xc - H @ cc) / torch.linalg.norm(xc))

        # ---- ARM B: the replay route, SAME start, same release
        args = argparse.Namespace(release="sgd", m=m, T=a.T, lr=a.lr, wd=0.0, jac="fwd", seed=a.seed + s,
                                  lm_iters=a.lm_iters, lm_lambda=1e-2, lm_scale="identity", stage_x=0,
                                  restarts=1, restart_noise=0.1, device=dev, quiet=True)
        W_init = w0.reshape(k, 1).repeat(1, N) + 0.5 * torch.randn(k, N, generator=gs).to(dev)
        with torch.no_grad():
            U0, _ = qr_canon(world.features_from_latents(W_init)); X_init = A_T @ U0
        t0 = time.time()
        try:
            W_hat, aux, resid, *_ = invert_lm(world, A_T, B_T, W0, y, args, W_init, X_init, log=lambda *x: None)
            Xr = world.psi(W_hat)
            er = [min(float(torch.linalg.norm(Xr[:, j] - H[:, i]) / torch.linalg.norm(H[:, i]))
                      for j in range(N)) for i in range(N)]
            replay = dict(residual=float(resid), worst_image_err=max(er), best_image_err=min(er),
                          images_landed=sum(1 for e in er if e < 1e-2), sec=time.time() - t0)
        except Exception as exc:                                   # a failure here is data, not a crash
            replay = dict(error=f"{type(exc).__name__}: {exc}")

        rows.append(dict(start=s, cert_objective=float(oc), cert_iters=int(itc),
                         cert_err_to_nearest=ec, cert_landed=bool(ec < 1e-2),
                         cert_blend_residual=blend_res, cert_coeff_sum=float(cc.sum()), replay=replay))

    landed_c = [d for d in rows if d["cert_landed"]]
    ok = [d for d in rows if "error" not in d["replay"]]
    landed_r = [d for d in ok if d["replay"]["images_landed"] > 0]
    med = lambda v: float(torch.tensor(v).median()) if v else None
    out = dict(cell="affine_chart_two_routes", k=k, N=N, r=r, m=m, P=P, T=a.T, lr=a.lr, seed=a.seed,
               starts=a.starts, cert_line=r - N, replay_line=m + r - N,
               below_both_lines=True, affine_hull_dim=min(N - 1, k),
               CH_rel=CH, rank_C=rank_C, n_prime=Np, cert_residual_at_truths=res_truth,
               isolation_rank=iso_rank, isolation_rank_predicted=k - min(N - 1, k),
               isolation_note="one-sided: full column rank k certifies isolation; a deficit certifies nothing. "
                              "DPsi is constant for an affine chart, so this is one number for the whole chart.",
               cert_starts_landing=len(landed_c),
               cert_objective_median=med([d["cert_objective"] for d in rows]),
               cert_objective_min=min(d["cert_objective"] for d in rows),
               cert_blend_residual_median=med([d["cert_blend_residual"] for d in rows]),
               cert_coeff_sum_median=med([d["cert_coeff_sum"] for d in rows]),
               cert_err_to_nearest_median=med([d["cert_err_to_nearest"] for d in rows]),
               replay_ok=len(ok), replay_starts_recovering=len(landed_r),
               replay_residual_median=med([d["replay"]["residual"] for d in ok]),
               replay_residual_min=min([d["replay"]["residual"] for d in ok]) if ok else None,
               replay_worst_err_median=med([d["replay"]["worst_image_err"] for d in ok]),
               replay_best_err_min=min([d["replay"]["best_image_err"] for d in ok]) if ok else None,
               rows=rows, git=git_hash(), host=socket.gethostname())
    print(json.dumps({kk: vv for kk, vv in out.items() if kk != "rows"}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
