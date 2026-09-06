#!/usr/bin/env python3
"""Where the linearised solve actually ends, not where its answer would be.

Two mechanisms for the one failing cell of four are already dead: batch class composition (refuted by a
single-class cell that recovers) and conditioning of the design AT THE TRUTH (refuted by measurement -- the failing
cell is the BEST conditioned of the four on every such measure). This measures the two things that could not have
been seen from the truth, both cheap, both on all four cells so a third proposal is tested rather than fitted.

  (1) CONDITIONING AT THE ENDPOINT. Nothing stops the search from stalling in a configuration where the candidate
      readings are near-duplicates of each other: there the projector is ill-conditioned and the gradient is weak,
      however well conditioned the true answer is. Static and dynamic conditioning are independent, and only the
      second could explain a stall. Disposition: if the endpoint condition number is comparable to the truth's,
      conditioning is dead in every form and is not to be proposed again.

  (2) HOW MANY RECORDED DIRECTIONS THE ENDPOINT ACTUALLY CAPTURED. The variable-projection residual is exactly the
      part of the release left outside the span of the candidate readings, so
          residual^2 / ||B_T||^2  =  1 - (captured share of B_T's squared spectrum),
      and comparing the observed residual against the partial sums of B_T's singular values reads off, as a COUNT,
      how many of the N recorded directions the endpoint design spans. A count is diagnosable where an unexplained
      residual is not. Its prediction, checkable in the same run: a design that has collapsed to an effectively
      low-rank set should have its N candidates close to EACH OTHER, not merely far from the truths.

Neither is a working hypothesis. Both are tests, and the outcome is reported whichever way it falls.

  python -m experiments.cifar.endpoint_diagnosis
"""
import json, math, socket
import numpy as np
import torch

from experiments.cifar.ntk_vs_certificate import build_cifar, build_mnist
from experiments.cifar.cell_conditioning import CELLS, cond

torch.set_default_dtype(torch.float64)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only cells whose tag contains this substring (the GPU queue is\n                    saturated; the failing cell alone is the one with an open question and it fits on a CPU slot)")
    ap.add_argument("--device", default=None)
    a_ = ap.parse_args()
    dev = torch.device(a_.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    N, r, k, T, lr, seed, starts, iters, alr = 8, 64, 16, 1, 0.01, 1, 200, 4000, 5e-2
    print(f"# endpoint_diagnosis  N={N} r={r} k={k} T={T} starts={starts} iters={iters}  host={socket.gethostname()}", flush=True)
    out = []
    for cell in CELLS:
        if a_.only and a_.only not in cell["tag"]: continue
        a = type("A", (), dict(newclass=cell["newclass"], newclass2=cell["newclass2"], letter=cell["letter"], letter2=cell["letter2"],
                               seed=seed, ckpt=None, model_mnist="models/exact_inversion/mnist_mlp_strong.pth",
                               data_root="data", mnist_root="dataset_reconstruction/data"))()
        phi, W0, Pub_l, Pri_l, cname, accs, shape = (build_cifar if cell["dataset"] == "cifar" else build_mnist)(a, dev)
        m, n = W0.shape; D = int(np.prod(shape))
        g = torch.Generator().manual_seed(seed + 7); ncls = len(Pri_l); per = N // ncls
        X_raw = torch.cat([Pri_l[c][torch.randperm(Pri_l[c].shape[0], generator=g)[:per]].T for c in range(ncls)], 1).contiguous()
        y = torch.cat([torch.full((per,), m - ncls + c, device=dev) for c in range(ncls)])
        Pub = torch.cat(Pub_l, 0)
        mean = Pub.mean(0); _, _, Vh = torch.linalg.svd(Pub - mean, full_matrices=False); V = Vh[:k].T.contiguous()
        coords = lambda X: V.T @ (X - mean[:, None])
        psi_b = lambda Z: mean[None, :, None] + torch.einsum("dk,pkn->pdn", V, Z)
        X_on = mean[:, None] + V @ coords(X_raw)
        coord_std = coords(Pub[:5000].T).std(dim=1, keepdim=True)
        H = phi(X_on)
        A0 = (1.0 / math.sqrt(n) * torch.randn(r, n, generator=torch.Generator().manual_seed(seed + 7), dtype=torch.float64)).to(dev)
        A, B = A0.clone(), torch.zeros(m, r, dtype=torch.float64, device=dev)
        Y = torch.eye(m, device=dev, dtype=torch.float64)[y].T
        for _ in range(T):
            z = W0 @ H + B @ (A @ H); Dm = (torch.softmax(z, 0) - Y) / N
            B, A = B - lr * (Dm @ (A @ H).T), A - lr * (B.T @ Dm @ H.T)
        A_T, B_T = A, B; nB = torch.linalg.norm(B_T); B_T_T = B_T.T.contiguous()
        sB = torch.linalg.svdvals(B_T)
        # the residual the design WOULD leave if it spanned exactly the top j recorded directions
        tail = torch.sqrt(torch.flip(torch.cumsum(torch.flip(sB ** 2, [0]), 0), [0])) / nB      # tail[j] = residual if top j captured
        # ---- the variable-projection solve, same starts and budget as the sweep
        Z0 = (torch.randn(starts, k, N, generator=torch.Generator().manual_seed(seed + 31)).to(dev) * coord_std[None])
        Z = Z0.clone().requires_grad_(True); opt = torch.optim.Adam([Z], alr); eyeN = torch.eye(N, device=dev)

        def design(Z_):
            Xc = psi_b(Z_)
            Hc = phi(Xc.permute(1, 0, 2).reshape(D, -1)).reshape(n, starts, N).permute(1, 0, 2)
            return Xc, torch.einsum("rn,pnN->prN", A_T, Hc)

        def resid(Fc):
            Ft = Fc.transpose(1, 2); G = Ft @ Fc
            ridge = (1e-12 * torch.diagonal(G, dim1=1, dim2=2).sum(1) / N).clamp(min=1e-300)
            Rsol = torch.linalg.solve(G + ridge[:, None, None] * eyeN, Ft @ B_T_T)
            return torch.linalg.norm((Fc @ Rsol - B_T_T[None]).reshape(starts, -1), dim=1) / nB

        for _ in range(iters):
            _, Fc = design(Z); rr = resid(Fc)
            opt.zero_grad(); rr.sum().backward(); opt.step()
        with torch.no_grad():
            Xc, Fc = design(Z); rr = resid(Fc); best = int(rr.argmin())
            err = torch.stack([torch.linalg.norm(Xc.permute(1, 0, 2).reshape(D, -1) - X_on[:, i:i + 1], dim=0) / torch.linalg.norm(X_on[:, i])
                               for i in range(N)], 1)
            found = int((err.min(0).values < 1e-2).sum())
            cs = [cond(Fc[p])[0] for p in range(starts)]; smin = [cond(Fc[p])[1] for p in range(starts)]
            Xb = Xc[best]                                                    # (D, N) the best start's N candidate images
            pd = torch.cdist(Xb.T, Xb.T) / torch.linalg.norm(Xb, dim=0).mean()
            offd = pd[~torch.eye(N, dtype=torch.bool, device=dev)]
            pdt = torch.cdist(X_on.T, X_on.T) / torch.linalg.norm(X_on, dim=0).mean()
            offt = pdt[~torch.eye(N, dtype=torch.bool, device=dev)]
            # CAPTURED COUNT, attacker-side: the smallest j whose tail energy beyond the top j recorded directions,
            # relative to the release norm, is at or below the achieved residual. Needs the release (which is
            # released) and the attacker's own objective value. No ground truth enters, so it is a progress meter
            # an attacker can read: "I have explained as much of this release as capturing j of its N directions."
            captured = int((tail > float(rr.min())).sum())
            # PER-IMAGE ANGLE TEST: are three IMAGES missing, or is the whole SUBSPACE rotated? Principal angles
            # between the subspaces cannot separate those -- five small and three large are consistent with both --
            # so compare each private image's reading to the endpoint's span individually.
            Fb = Fc[best]                                                    # (r, N) the best start's design
            Qb, _ = torch.linalg.qr(Fb)
            AH = A_T @ H                                                     # each private image's reading
            frac = torch.linalg.norm(Qb.T @ AH, dim=0) / torch.linalg.norm(AH, dim=0)
            angles = torch.rad2deg(torch.arccos(frac.clamp(-1, 1)))
            err_per_image = err.min(0).values
        row = dict(cell=cell["tag"], recovers=cell["recovers"], images_found=found,
                   residual_best=float(rr.min()), residual_median=float(rr.median()),
                   cond_design_at_truth=cond(A_T @ H)[0], sigma_min_at_truth=cond(A_T @ H)[1],
                   cond_design_at_endpoint_best=cond(Fc[best])[0], cond_design_at_endpoint_median=float(np.median(cs)),
                   sigma_min_at_endpoint_best=cond(Fc[best])[1], sigma_min_at_endpoint_median=float(np.median(smin)),
                   captured_directions_of_N=captured, N=N,
                   residual_if_top_j_captured=[float(v) for v in tail[:N + 1]],
                   endpoint_pairwise_dist_min=float(offd.min()), endpoint_pairwise_dist_median=float(offd.median()),
                   truth_pairwise_dist_min=float(offt.min()), truth_pairwise_dist_median=float(offt.median()),
                   B_T_sigma_ratio_N_over_1=float(sB[N - 1] / sB[0]),
                   angle_to_endpoint_span_deg=[float(v) for v in angles],
                   n_images_near_span_lt10deg=int((angles < 10).sum()), n_images_far_gt60deg=int((angles > 60).sum()),
                   closest_err_per_image=[float(v) for v in err_per_image],
                   angle_vs_error_agree=[int(i) for i in torch.argsort(angles, descending=True)[:3]] ==
                                        [int(i) for i in torch.argsort(err_per_image, descending=True)[:3]])
        out.append(row); print(json.dumps(row), flush=True)
        print(f"   angles of each private reading to the endpoint span (deg): {[f'{v:.1f}' for v in angles.tolist()]}", flush=True)
        print(f"   closest error per image:                                  {[f'{v:.3f}' for v in err_per_image.tolist()]}", flush=True)
        torch.save(dict(Z=Z.detach().cpu(), F=Fc.detach().cpu(), A_T=A_T.cpu(), B_T=B_T.cpu(), C=C.cpu(), H=H.cpu(),
                        X_on=X_on.cpu(), row=row), f"results/cifar_newclass/endpoint_{cell['tag'].replace(' ', '_').replace('+', '_and_')}.pth")
    print("\n| cell | recovers | images | residual | cond @ truth | cond @ ENDPOINT (median) | σ_min @ endpoint | directions captured | images >60° from the span | endpoint spread | truth spread |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r_ in out:
        print(f"| {r_['cell']} | {'yes' if r_['recovers'] else '**NO**'} | {r_['images_found']}/{r_['N']} | {r_['residual_best']:.2e} | "
              f"{r_['cond_design_at_truth']:.1f} | {r_['cond_design_at_endpoint_median']:.2e} | {r_['sigma_min_at_endpoint_median']:.2e} | "
              f"{r_['captured_directions_of_N']} of {r_['N']} | {r_['n_images_far_gt60deg']} of {r_['N']} | {r_['endpoint_pairwise_dist_median']:.3f} | {r_['truth_pairwise_dist_median']:.3f} |")
    print("\n# Dispositions, recorded before the numbers existed: endpoint conditioning comparable to the truth's kills")
    print("# conditioning in every form. A captured count far below N with a collapsed endpoint spread is the low-rank")
    print("# stall. A captured count near N means the search is simply nowhere near a solution and neither applies.")
    print("# Angle test, recorded before the numbers: three near 90 and five near 0 means three IMAGES are missing and")
    print("# they can be named; all eight intermediate with none near 0 means the SUBSPACE is rotated and no single")
    print("# image is absent. Cross-check: if images are missing, they are the three worst in closest-error-per-image.")


if __name__ == "__main__":
    main()
