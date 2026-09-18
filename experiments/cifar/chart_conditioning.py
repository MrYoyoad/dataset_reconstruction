#!/usr/bin/env python3
"""P7 -- charts on ONE release (plan 2026-09-18, audit item 9). Analysis on the saved head release; no new attack runs.

Release: the CIFAR CNN head release of the bootstrap / ladder / WP2 line (`cifar10_cnn_newclass.pth`, CIFAR-100
`motorcycle` as an 11th class, N = 8, r = 64, T = 400 SGD, lr 0.01, A0 = randn(r, n, seed+7)/sqrt(n), B0 = 0), rebuilt
with the bootstrap's own `Release` on the RAW privates ("raw": the ONE release every chart is measured on). rank C = 56,
so k in {16, 32, 48} only (k = 128 is alias-prone by construction, item 9).

Charts (each fitted on the motorcycle PUBLIC train pool, 500 images, exactly as `ntk_vs_certificate.py` fits them):
  pca            pooled PCA-k (on a single-class cell the pooled and per-class pools coincide, so `pca_perclass` is the
                 same chart -- recorded as identical, not recomputed; WP2 skips it for the same reason)
  ae             the WP2 autoencoder decoder (AEChartLocal, 150 epochs; RE-TRAINED here -- WP2 did not save its instance)
  local          local PCA-k of the K = 200 nearest public neighbours of the TRUTH (bootstrap variant B, oracle-anchor
                 construction; not attacker-available), one chart per image
  sd-vae         SKIPPED: no saved latent chart at k <= 48 exists (results/decoder_chart holds fidelity rows only,
                 `release_read: False`; the audit lists it at k in {16, 32, 66}, ceiling-bound)

Per chart x k x image, at x*_chart (the LM optimum from the oracle start = coords of the truth's projection, 300 iters,
the bootstrap's reference; NOT the projection itself):
  rank / gap / cond of J = C . J_phi . J_psi  at the 1e-10 rung with the absolute floor sigma_1(A_T J_phi J_psi)
  (multilayer_cert.common.numrank); for the nonlinear chart also J_orth = C J_phi Q with Q = orth(J_psi), so the chart's
  own conditioning cond(J_psi) = sigma_1 / sigma_k (= 1 for a linear chart) is separated from the certificate's;
  the raw projection error of the truth; err(x*_chart, truth); the objective at x*_chart.
"onchart" rows: the WP2 construction (privates = their chart projections, release retrained on them; x*_chart = the
truth by construction, objective ~1e-13) so the join to the WP2 landing rate is on the SAME release for `pca`; for `ae`
the retrained decoder is a different instance, so that join is approximate and flagged.

Join: `results/ntk_vs_cert/sweep_*.jsonl` rows with dataset cifar / backbone cnn / class motorcycle / T = 400 / same
chart name / same k -> landing rate = cert_landed_starts / starts, images found, closest-per-image.

PRE-REGISTERED: the landing rate orders the charts by cond_at_rank, not by projection error. Both orderings are printed;
a tie or a reversal is the finding. (The learned-chart conditioning 4.28 -> 4.72 -> 5.90 is on record as a correlate.)

  python -u -m experiments.cifar.chart_conditioning --charts pca ae local --ks 16 32 48
"""
import argparse, glob, json, os, socket, sys, time
from types import SimpleNamespace
import numpy as np
import torch

from experiments.bootstrap_chart.bootstrap import Chart, Release, local_chart, rel_err, FLOOR, LAND
from experiments.bootstrap_chart.handover_jacobian import DEFAULTS, load_release, fwd_check, jac_phi, spectrum_stats, orth, RTOL
from experiments.cifar.ntk_vs_certificate import AEChartLocal
from experiments.multilayer_cert.common import provenance

torch.set_default_dtype(torch.float64)


def log(s): print(s, flush=True)


class AEWrap:
    """The WP2 decoder with the bootstrap Chart interface (psi / coords / project / err / k)."""
    def __init__(s, ae, name): s.ae, s.k, s.name, s.mean, s.V = ae, ae.k, name, None, None
    def psi(s, W): return s.ae.psi(W)
    def coords(s, X): return s.ae.coords(X)
    def project(s, X): return s.psi(s.coords(X))
    def err(s, X): return torch.linalg.norm(s.project(X) - X, dim=0) / torch.linalg.norm(X, dim=0)
    def jpsi(s, w): return torch.func.jacfwd(lambda v: s.ae.dec(v[None, :])[0])(w).detach()      # (D, k)


def jpsi_of(ch, w):
    return ch.jpsi(w) if isinstance(ch, AEWrap) else ch.V


def measure(rel, phi, ch, x_star, w_star):
    """Spectra of J = C J_phi J_psi at x_star (and of the orthonormalised chart for a nonlinear one)."""
    Jphi = jac_phi(phi, x_star); Jpsi = jpsi_of(ch, w_star)
    sp = torch.linalg.svdvals(Jpsi); cond_psi = float(sp[0] / sp[-1])
    def stats(V):
        JV = Jphi @ V; ref = float(torch.linalg.svdvals(rel.A_T @ JV)[0]); return spectrum_stats(rel.C @ JV, ref)
    st = stats(Jpsi); out = dict(J=st, cond_Jpsi=cond_psi, sigma_Jpsi=[float(v) for v in sp.cpu()])
    if isinstance(ch, AEWrap): out["J_orth"] = stats(orth(Jpsi))
    else: out["J_orth"] = st
    return out


def wp2_rows(pattern="results/ntk_vs_cert/sweep_*.jsonl"):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("part") != "ntk_vs_cert" or r.get("dataset") != "cifar" or r.get("same_row"): continue
            bb = r.get("backbone"); bb = bb.get("backbone") if isinstance(bb, dict) else bb       # pre-WP0 rows carry no backbone: skipped
            if bb == "cnn" and r.get("class_name") == "motorcycle" and r.get("T") == 400:
                rows.append(dict(file=os.path.basename(f), chart=r["chart"], k=r["k"], starts=r["starts"], landed=r["cert_landed_starts"],
                                 landing_rate=r["cert_landed_starts"] / r["starts"], images_found=r["cert_images_found"], closest=r["cert_closest_per_image"],
                                 repr_err_median=r["chart_repr_err_median"], n_prime=r["n_prime"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", nargs="*", default=["pca", "ae", "local", "pca_perclass"]); ap.add_argument("--ks", nargs="*", type=int, default=[16, 32, 48])
    ap.add_argument("--releases", nargs="*", default=["raw", "onchart"]); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--ae-epochs", type=int, default=150); ap.add_argument("--K", type=int, default=200); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="results/cifar"); ap.add_argument("--out", default=None)
    args = ap.parse_args(); dev = torch.device(args.device)
    a = SimpleNamespace(**DEFAULTS); a.seed = args.seed; a.iters = args.iters
    assert all(k < 56 for k in args.ks), "k must be below the certificate line r - N' = 56 (audit item 9)"
    os.makedirs(args.save_dir, exist_ok=True)
    job = os.environ.get("LSB_JOBID", "local"); out = args.out or os.path.join(args.save_dir, f"chart_conditioning_{job}.jsonl")
    prov = provenance(__file__); t_start = time.time()
    log(f"# chart_conditioning charts={args.charts} ks={args.ks} releases={args.releases} dev={dev} git={prov['git']} sha={prov['script_sha']} host={socket.gethostname()} job={job}")

    S = load_release("cifar", a, dev); rel = S["rel"]; phi = S["phi"]; X_raw = S["X_raw"]; pool = S["class_pool"](S["true_cls"])
    r0 = sorted(glob.glob("results/bootstrap_chart/cifar_round0_round0_generic_*.pth"))
    chk = fwd_check(rel, torch.load(r0[-1], map_location="cpu", weights_only=False)) if r0 else {"pass": None, "note": "no saved round-0 tensors"}
    log(f"# release: rank B_T {rel.Np}, rank C {rel.rank_C}, line k < {rel.r - rel.Np}; objective at the raw truths max {float(rel.res_truth.max()):.1e}; fwd_check vs {r0[-1] if r0 else '-'}: {chk}")
    if chk.get("pass") is False: log("# fwd_check FAILED -- rows below are NOT meaningful (ground rule 5); continuing so the discrepancy is on record")
    wp2 = wp2_rows(); log(f"# WP2 rows joinable (cifar/cnn/motorcycle/T=400): {[(w['chart'], w['k'], w['file']) for w in wp2]}")
    common = dict(part="chart_conditioning", release_base="cifar10_cnn_newclass.pth", class_name="motorcycle", N=a.N, r=a.r, T=a.T, lr=a.lr, n_prime=rel.Np,
                  rank_C=rel.rank_C, cert_line=rel.r - rel.Np, fwd_check=chk, rtol=RTOL, rank_floor="sigma_1(A_T J_phi J_psi) (numrank ref)", iters=a.iters, K=args.K,
                  ae_epochs=args.ae_epochs, seed=a.seed, pool_size=int(pool.shape[0]), git=prov["git"], script_sha=prov["script_sha"], host=socket.gethostname(), cmd=" ".join(sys.argv),
                  prereg="landing rate orders charts by cond_at_rank, not by projection error; both orderings recorded, tie or reversal is the finding",
                  sd_vae="skipped: no saved latent chart at k <= 48 (results/decoder_chart has fidelity rows only, release_read=False)")
    cells = []

    def emit(row):
        row = dict(**common, **row); cells.append(row)
        with open(out, "a") as f: f.write(json.dumps(row) + "\n")

    for name in args.charts:
        if name == "pca_perclass":
            emit(dict(chart=name, k=None, release=None, identical_to="pca", note="single-class cell: the per-class pool IS the pooled pool (WP2 skips it too)"))
            log(f"# {name}: identical to pca on a single-class cell -- marker row only"); continue
        for k in args.ks:
            t0 = time.time()
            if name == "pca": charts = [Chart.pca(pool, k, "pca")] * a.N
            elif name == "ae":
                torch.manual_seed(a.seed); ae = AEChartLocal(pool, k, dev, S["shape"], args.ae_epochs); charts = [AEWrap(ae, "ae")] * a.N
            elif name == "local": charts = [local_chart(X_raw[:, i], pool, args.K, k, "local")[0] for i in range(a.N)]
            else: raise ValueError(name)
            desc = (f"AE decoder, public recon err {ae.recon:.3f}" if name == "ae" else f"explained {charts[0].explained:.3f}" if name == "pca" else f"K={args.K} neighbours of the truth")
            X_proj = torch.stack([charts[i].project(X_raw[:, i:i + 1])[:, 0] for i in range(a.N)], 1)
            proj_err = [float(v) for v in torch.linalg.norm(X_proj - X_raw, dim=0) / torch.linalg.norm(X_raw, dim=0)]
            log(f"\n### chart={name} k={k} ({desc}; {time.time() - t0:.0f}s to fit): raw projection error of the truths median {np.median(proj_err):.3f}, per image {[round(v, 3) for v in proj_err]}")
            join = next((w for w in wp2 if w["chart"] == name and w["k"] == k), None)
            for release in args.releases:
                per = []; t1 = time.time()
                if release == "raw":
                    rl = rel
                    for i in range(a.N):
                        ch = charts[i]; res = rl.search(ch, [ch.coords(X_raw[:, i:i + 1]).reshape(-1)], a.iters)[0]
                        m = measure(rl, phi, ch, res["x"], res["w"])
                        per.append(dict(i=i, proj_err=proj_err[i], obj_xstar=res["objective"], obj_projection=float(rl.obj_of(X_proj[:, i:i + 1])[0]), lm_iters=res["iters"],
                                        err_xstar_truth=float(rel_err(res["x"], X_raw[:, i:i + 1])[0]), err_xstar_projection=float(rel_err(res["x"], X_proj[:, i:i + 1])[0]), **m))
                else:
                    y = torch.full((a.N,), rel.m - 1, device=dev)
                    A0 = (1.0 / (rel.n ** 0.5) * torch.randn(a.r, rel.n, generator=torch.Generator().manual_seed(a.seed + 7), dtype=torch.float64)).to(dev)
                    rl = Release("cifar_onchart", phi, S["W0"], X_proj, y, A0, a.T, a.lr, a.tol, pool)
                    for i in range(a.N):
                        ch = charts[i]; x = X_proj[:, i]; w = ch.coords(x[:, None]).reshape(-1)
                        m = measure(rl, phi, ch, x, w)
                        per.append(dict(i=i, proj_err=proj_err[i], obj_xstar=float(rl.obj_of(x[:, None])[0]), obj_projection=float(rl.obj_of(x[:, None])[0]), lm_iters=0,
                                        err_xstar_truth=0.0, err_xstar_projection=0.0, **m))
                med = lambda key: float(np.median([p[key] for p in per])); medJ = lambda key, J="J": float(np.median([p[J][key] for p in per]))
                sm = dict(proj_err_median=med("proj_err"), obj_xstar_median=med("obj_xstar"), err_xstar_truth_median=med("err_xstar_truth"),
                          rank=[p["J"]["rank"] for p in per], gap_median=medJ("gap_at_rank"), cond_median=medJ("cond_at_rank"), cond_max=max(p["J"]["cond_at_rank"] for p in per),
                          cond_orth_median=medJ("cond_at_rank", "J_orth"), cond_Jpsi_median=med("cond_Jpsi"), cond_Jpsi_max=max(p["cond_Jpsi"] for p in per),
                          landing_rate=(join["landing_rate"] if join else None), images_found=(join["images_found"] if join else None), wp2_file=(join["file"] if join else None),
                          join_release_matches=(release == "onchart" and name == "pca") if join else None, n_prime_release=rl.Np, rank_C_release=rl.rank_C)
                emit(dict(chart=name, k=k, release=release, chart_desc=desc, per_image=per, seconds=time.time() - t1, **sm))
                log(f"  [{release}] rank {sm['rank']} gap med {sm['gap_median']:.1e} | cond med {sm['cond_median']:.2e} (max {sm['cond_max']:.2e}; orth {sm['cond_orth_median']:.2e}; "
                    f"cond J_psi {sm['cond_Jpsi_median']:.2f}) | obj x* med {sm['obj_xstar_median']:.1e} err(x*,truth) med {sm['err_xstar_truth_median']:.3f} | "
                    f"landing {sm['landing_rate']} ({sm['wp2_file']})  [{time.time() - t1:.0f}s]")

    # ---------------- orderings (pre-registered)
    for release in args.releases:
        cs = [c for c in cells if c.get("release") == release]
        if not cs: continue
        lab = lambda c: f"{c['chart']}-k{c['k']}"
        by_cond = sorted(cs, key=lambda c: c["cond_median"]); by_proj = sorted(cs, key=lambda c: c["proj_err_median"])
        withL = [c for c in cs if c["landing_rate"] is not None]; by_land = sorted(withL, key=lambda c: -c["landing_rate"])
        fm = lambda cs_, key, f_: [(lab(c), f_ % c[key]) for c in cs_]
        log("\n# [%s] ordering by cond_at_rank (best first): %s" % (release, fm(by_cond, "cond_median", "%.2e")))
        log("# [%s] ordering by projection error (best first): %s" % (release, fm(by_proj, "proj_err_median", "%.3f")))
        log("# [%s] ordering by WP2 landing rate (best first; joinable cells only): %s" % (release, fm(by_land, "landing_rate", "%.3f")))
        if len(withL) >= 2:
            rc = [lab(c) for c in sorted(withL, key=lambda c: c["cond_median"])]; rp = [lab(c) for c in sorted(withL, key=lambda c: c["proj_err_median"])]; rl_ = [lab(c) for c in by_land]
            log(f"# [{release}] on the joinable cells: cond order {rc} | proj-err order {rp} | landing order {rl_} -> "
                f"cond {'MATCHES' if rc == rl_ else 'does NOT match'} landing; proj-err {'MATCHES' if rp == rl_ else 'does NOT match'} landing")
        with open(out, "a") as f:
            f.write(json.dumps(dict(part="chart_conditioning_ordering", release=release, job=job, by_cond=[lab(c) for c in by_cond], by_proj_err=[lab(c) for c in by_proj],
                                    by_landing=[lab(c) for c in by_land], n_joinable=len(withL))) + "\n")
    log(f"\n| chart | k | release | rank | gap (med) | cond@rank (med / max) | cond J_psi | proj err (med) | err(x*,truth) med | obj x* med | WP2 landing rate | images found |")
    log("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        if c.get("release") is None: continue
        rk = c["rank"]; rks = str(rk[0]) if len(set(rk)) == 1 else f"{min(rk)}-{max(rk)}"; lr_s = "-" if c["landing_rate"] is None else f"{c['landing_rate']:.3f}"
        log(f"| {c['chart']} | {c['k']} | {c['release']} | {rks} | {c['gap_median']:.1e} | {c['cond_median']:.2e} / {c['cond_max']:.2e} | {c['cond_Jpsi_median']:.2f} | {c['proj_err_median']:.3f} | "
            f"{c['err_xstar_truth_median']:.3f} | {c['obj_xstar_median']:.1e} | {lr_s} | {c['images_found']} |")
    log(f"# done in {time.time() - t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
