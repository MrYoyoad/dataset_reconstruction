#!/usr/bin/env python3
"""P8 -- chart hand-over (plan 2026-09-18, audit items 10a-c). Analysis on SAVED bootstrap releases; no new training.

For every (release, round pair t -> t+1) of the bootstrap run and every truth (global charts) or slot (local charts):
at the chart optimum x*_chart of round t and of round t+1 compute the Jacobian of the certificate map,
        J = C . J_phi(x) . V,          C = P_{row(B_T)^perp} A_T   (r x n),  J_phi = d phi / d x (n x D),  V (D x k),
under the round-t chart V_t, the round-(t+1) chart V_{t+1}, and the UNION chart U = orth([V_t V_{t+1}]); for each the
singular spectrum, the rank at the 1e-10 rung with the absolute floor of `multilayer_cert.common.numrank` (reference
scale sigma_1(A_T J_phi V), the matrix C was built from), the gap sigma_rank / sigma_{rank+1} and cond = sigma_1 / sigma_rank.
Also: the principal angles between col(V_t) and col(V_{t+1}); the certificate objective at x*_chart(t) evaluated in
chart t+1 (i.e. at its projection into chart t+1) and vice versa; whether the union width exceeds r - N' (= 56 here).

PRE-REGISTERED (audit 2026-09-18):
  10a  The reference point is x*_chart, the chart optimum reached by the LM solver from the ORACLE start (coords of the
       truth's projection; not attacker-available) -- never the truth's projection itself (10-80x apart in objective).
       Global charts: x*_chart is read from the saved tensors (`x_opt`). Local charts (variant B) were not saved with
       their bases, so the chart is REBUILT deterministically from the saved anchors (previous round's slot candidates,
       the same K nearest public neighbours, the same generator for the random-anchor control) and x*_chart is
       re-solved with the same solver and budget; the row's saved objective_opt / err_opt_truth are re-checked.
  10b  The measured slot collapse is read as COVERAGE (8 slots -> 2 truths), not as an alias event. Its falsifier is
       the alias flag: objective at the recovery at the floor (<= objective at x*_chart) AND recovery != x*_chart
       (relative error >= 1e-2). Alias flags are COUNTED per round from the saved recoveries; the count is the result.
  10c  The round-1 chart is built after seeing C (the recovery selected the class / the neighbours), so no k < r - q
       theorem applies to V_{t+1} or to the union; 2 x 32 > 56 regardless. The union-chart rank / cond is a
       MEASUREMENT, not a theorem test. Where the union width exceeds r - N' the cell is alias-prone by construction:
       reported, and identifiability is NOT tested there.

Ground rules: FP64; `fwd_check` first -- the release is rebuilt with the bootstrap's own `Release` and compared to the
saved A_T, B_T, C of the round-0 tensors to machine precision before any Jacobian is read.

  python -u -m experiments.bootstrap_chart.handover_jacobian --release mnist --job 355987
  python -u -m experiments.bootstrap_chart.handover_jacobian --release cifar --job 355988 --b-arms recovery
"""
import argparse, glob, json, math, os, socket, sys, time
from types import SimpleNamespace
import numpy as np
import torch

from experiments.bootstrap_chart.bootstrap import (Chart, Release, local_chart, oracle_optimum, rel_err, load_emnist_all,
                                                   load_cifar100_all, FLOOR, LAND)
from experiments.exact_inversion.trained_backbone import TrainedBackbone
from experiments.cifar.cifar_newclass import train_backbone, load_cifar100_class
from experiments.multilayer_cert.common import numrank, provenance

torch.set_default_dtype(torch.float64)
RTOL = 1e-10


def log(s): print(s, flush=True)


# ------------------------------------------------------------------------------------------------- linear algebra
def jac_phi(phi, x):
    """J_phi = d phi / d x at one image x (D,) -> (n, D). torch.func.jacrev; falls back to row-wise autograd."""
    f = lambda v: phi(v.reshape(-1, 1))[:, 0]
    try:
        return torch.func.jacrev(f)(x).detach()
    except Exception as e:                                                   # e.g. a module vmap cannot trace
        log(f"#   jacrev failed ({type(e).__name__}: {e}); row-wise autograd")
        xg = x.clone().requires_grad_(True); out = f(xg); rows = []
        for i in range(out.shape[0]):
            g, = torch.autograd.grad(out[i], xg, retain_graph=True); rows.append(g.detach())
        return torch.stack(rows)


def spectrum_stats(J, ref, rtol=RTOL, keep=72):
    """rank at the rtol rung with the absolute floor `ref` (numrank), gap at the rank, cond = s1/s_rank, spectrum."""
    rank, s = numrank(J, rtol, ref)
    s = s.detach().cpu()
    gap = float(s[rank - 1] / s[rank]) if 0 < rank < len(s) and float(s[rank]) > 0 else float("inf")
    cond = float(s[0] / s[rank - 1]) if rank > 0 else float("inf")
    return dict(rank=rank, gap_at_rank=gap, cond_at_rank=cond, sigma_1=float(s[0]), sigma_rank=float(s[rank - 1]) if rank else 0.0,
                sigma_next=float(s[rank]) if rank < len(s) else 0.0, ref_scale=float(ref), n_sv=len(s), spectrum=[float(v) for v in s[:keep]])


def cert_jacobians(rel, phi, x, bases):
    """At image x: J_phi once, then J = C J_phi V for every basis V in `bases` (dict name -> (D, k) matrix).
    Also the Jacobian of the NORMALISED residual the LM solver actually minimises, f = C phi / ||A_T phi||."""
    Jphi = jac_phi(phi, x)                                                   # (n, D)
    f = phi(x.reshape(-1, 1))[:, 0]; Af = rel.A_T @ f; nAf = torch.linalg.norm(Af); Cf = rel.C @ f
    out = {}
    for name, V in bases.items():
        JV = Jphi @ V; AJV = rel.A_T @ JV; J = rel.C @ JV                       # (r, k)
        ref = float(torch.linalg.svdvals(AJV)[0])                            # the scale J was built from: sigma_1(A_T J_phi V)
        st = spectrum_stats(J, ref)
        Jn = J / nAf - torch.outer(Cf, Af) @ AJV / nAf ** 3                  # d/dw of C phi / ||A_T phi||
        stn = spectrum_stats(Jn, ref / float(nAf))
        out[name] = dict(**st, width=int(V.shape[1]), normalised=dict(rank=stn["rank"], gap_at_rank=stn["gap_at_rank"], cond_at_rank=stn["cond_at_rank"]))
    return out, float((torch.linalg.norm(Cf) / nAf) ** 2)


def principal_angles(V1, V2):
    """Angles (degrees) between col(V1) and col(V2), both orthonormal."""
    s = torch.linalg.svdvals(V1.T @ V2).clamp(-1, 1)
    ang = torch.rad2deg(torch.acos(s)).cpu()
    return dict(min_deg=float(ang.min()), median_deg=float(ang.median()), max_deg=float(ang.max()),
                n_below_1deg=int((ang < 1.0).sum()), n_below_10deg=int((ang < 10.0).sum()), angles_deg=[float(v) for v in ang])


def union_basis(V1, V2, rtol=RTOL):
    M = torch.cat([V1, V2], 1); U, s, _ = torch.linalg.svd(M, full_matrices=False)
    d = int((s > rtol * s[0]).sum()); return U[:, :d].contiguous(), d


def orth(V):
    Q, _ = torch.linalg.qr(V); return Q


# ------------------------------------------------------------------------------------------------- release rebuild
DEFAULTS = dict(N=8, r=64, k=32, T=400, lr=0.01, tol=1e-12, K=200, iters=300, seed=1, letter="a", cifar_class="motorcycle",
                mnist_model="models/exact_inversion/mnist_mlp_strong.pth", cifar_ckpt="models/exact_inversion/cifar10_cnn_newclass.pth",
                data_root="data", mnist_root="dataset_reconstruction/data")


def load_release(domain, a, dev):
    """The bootstrap's release (same privates, A0, recipe) WITHOUT the in-job base gate (already stamped on the saved rows)."""
    if domain == "mnist":
        bb = TrainedBackbone(a.mnist_model, dev, "gelu"); em = load_emnist_all(a.mnist_root)
        Xpub = torch.tensor(em["train"][0], device=dev); ypub = torch.tensor(em["train"][1], device=dev)
        names = [chr(ord("a") + i) for i in range(26)]; true_cls = names.index(a.letter)
        Pri = torch.tensor(em["test"][0][em["test"][1] == true_cls], device=dev)
        phi = bb.phi; W0 = torch.cat([bb.W0, torch.zeros(1, bb.n, device=dev)], 0); n = bb.n; shape = (1, 28, 28)
    else:
        net, _, _ = train_backbone(a.cifar_ckpt, a.data_root, dev, 0, 0.0, 0.0, "cnn"); net = net.double()
        for p_ in net.parameters(): p_.requires_grad_(False)
        Xp, yp, names = load_cifar100_all(a.data_root)
        Xpub = torch.tensor(Xp, dtype=torch.float64, device=dev); ypub = torch.tensor(yp, device=dev)
        pool, cname = load_cifar100_class(a.data_root, a.cifar_class); true_cls = names.index(cname)
        Pri = torch.tensor(pool["test"], dtype=torch.float64, device=dev)
        n = net.head.weight.shape[1]; W0 = torch.cat([net.head.weight.double(), torch.zeros(1, n, dtype=torch.float64, device=dev)], 0)
        phi = lambda X: net.phi(X.T).T; shape = (3, 32, 32)
    g = torch.Generator().manual_seed(a.seed + 7); perm = torch.randperm(Pri.shape[0], generator=g)
    X_raw = Pri[perm[: a.N]].T.contiguous(); y = torch.full((a.N,), W0.shape[0] - 1, device=dev)
    A0 = (1.0 / math.sqrt(n) * torch.randn(a.r, n, generator=torch.Generator().manual_seed(a.seed + 7), dtype=torch.float64)).to(dev)
    rel = Release(domain, phi, W0, X_raw, y, A0, a.T, a.lr, a.tol, Xpub)
    return dict(domain=domain, phi=phi, rel=rel, W0=W0, X_raw=X_raw, Xpub=Xpub, ypub=ypub, names=names, true_cls=true_cls, shape=shape,
                private_idx=perm[: a.N].tolist(), class_pool=lambda c: Xpub[ypub == c])


def fwd_check(rel, blob):
    d = {k: float((getattr(rel, k) - blob[k].to(rel.C.device)).abs().max()) for k in ("A_T", "B_T", "C")}
    d["A_T_scale"] = float(rel.A_T.abs().max()); d["pass"] = bool(max(d["A_T"], d["B_T"], d["C"]) < 1e-9)
    return d


# ------------------------------------------------------------------------------------------------- one pair
def analyse_pair(S, Vt, mt, Vt1, mt1, x_t, x_t1, x_rec, x_rec_opt_ref, truth):
    """Everything P8 asks for one (pair, truth/slot). Vt/Vt1 orthonormal (D, k); mt/mt1 chart means; x_t = x*_chart(t),
    x_t1 = x*_chart(t+1); x_rec the round-(t+1) recovery (for the alias flag against x_rec_opt_ref = x*_chart(t+1))."""
    rel, phi = S["rel"], S["phi"]
    U, width = union_basis(Vt, Vt1)
    bases = dict(V_t=Vt, V_t1=Vt1, union=U)
    J_at_t, obj_t = cert_jacobians(rel, phi, x_t, bases)
    J_at_t1, obj_t1 = cert_jacobians(rel, phi, x_t1, bases)
    proj = lambda x, V, m: m + V @ (V.T @ (x - m))
    x_t_in_t1 = proj(x_t, Vt1, mt1); x_t1_in_t = proj(x_t1, Vt, mt)
    with torch.no_grad():
        obj_cross = dict(obj_xt=obj_t, obj_xt1=obj_t1,
                         obj_xt_in_chart_t1=float(rel.obj_of(x_t_in_t1[:, None])[0]), obj_xt1_in_chart_t=float(rel.obj_of(x_t1_in_t[:, None])[0]),
                         move_xt_to_chart_t1=float(rel_err(x_t, x_t_in_t1[:, None])[0]), move_xt1_to_chart_t=float(rel_err(x_t1, x_t1_in_t[:, None])[0]),
                         err_xt_xt1=float(rel_err(x_t, x_t1[:, None])[0]), err_xt_truth=float(rel_err(x_t, truth[:, None])[0]), err_xt1_truth=float(rel_err(x_t1, truth[:, None])[0]))
        obj_rec = float(rel.obj_of(x_rec[:, None])[0]); obj_ref = float(rel.obj_of(x_rec_opt_ref[:, None])[0])
        err_rec_opt = float(rel_err(x_rec, x_rec_opt_ref[:, None])[0])
        at_floor = bool(obj_rec <= obj_ref * (1 + 1e-6) + FLOOR)
        alias = dict(obj_recovery=obj_rec, obj_xstar=obj_ref, err_recovery_xstar=err_rec_opt, at_floor=at_floor,
                     alias_flag=bool(at_floor and err_rec_opt >= LAND), reached_xstar=bool(err_rec_opt < LAND), solver_short=bool(not at_floor and err_rec_opt >= LAND))
    return dict(angles=principal_angles(Vt, Vt1), union_width=width, union_exceeds_line=bool(width > rel.r - rel.Np), cert_line=rel.r - rel.Np,
                J_at_xstar_t=J_at_t, J_at_xstar_t1=J_at_t1, **obj_cross, **alias)


def summarise(rows):
    """Medians over the truths/slots of a pair, for the printed table."""
    med = lambda k: float(np.median([r[k] for r in rows]))
    out = dict(n=len(rows), alias_flags=sum(r["alias_flag"] for r in rows), reached_xstar=sum(r["reached_xstar"] for r in rows),
               solver_short=sum(r["solver_short"] for r in rows), union_width=[r["union_width"] for r in rows],
               angles_min=med("angles_min"), angles_median=med("angles_median"), n_below_1deg=[r["angles"]["n_below_1deg"] for r in rows],
               obj_xt=med("obj_xt"), obj_xt1=med("obj_xt1"), obj_xt_in_chart_t1=med("obj_xt_in_chart_t1"), obj_xt1_in_chart_t=med("obj_xt1_in_chart_t"))
    for at in ("J_at_xstar_t", "J_at_xstar_t1"):
        for b in ("V_t", "V_t1", "union"):
            out[f"{at}.{b}"] = dict(rank=[r[at][b]["rank"] for r in rows], gap=float(np.median([r[at][b]["gap_at_rank"] for r in rows])),
                                    cond=float(np.median([r[at][b]["cond_at_rank"] for r in rows])), cond_max=float(max(r[at][b]["cond_at_rank"] for r in rows)))
    return out


# ------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", choices=["mnist", "cifar"], required=True); ap.add_argument("--job", required=True, help="bootstrap job id whose tensors are read")
    ap.add_argument("--b-arms", nargs="*", default=["recovery", "random_anchor", "oracle_anchor"])
    ap.add_argument("--a-arms", nargs="*", default=["recognised", "wrong_class", "oracle_class"])
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default="results/bootstrap_chart"); ap.add_argument("--out", default=None)
    args = ap.parse_args(); dev = torch.device(args.device)
    a = SimpleNamespace(**DEFAULTS); a.seed = args.seed; a.iters = args.iters
    myjob = os.environ.get("LSB_JOBID", "local"); out = args.out or os.path.join(args.save_dir, f"handover_{myjob}.jsonl")
    prov = provenance(__file__); t_start = time.time()
    log(f"# handover_jacobian  release={args.release} source_job={args.job} b_arms={args.b_arms} dev={dev} git={prov['git']} sha={prov['script_sha']} host={socket.gethostname()} job={myjob}")
    dom = args.release
    pth = lambda v, t, arm: os.path.join(args.save_dir, f"{dom}_{v}_round{t}_{arm}_{args.job}.pth")
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    rows_src = {}
    for l in open(os.path.join(args.save_dir, f"rounds_{args.job}.jsonl")):
        r = json.loads(l)
        if r["release"] == dom: rows_src[(r["variant"], r["round"], r["arm"])] = r
    log(f"# source rows found: {sorted(rows_src)}")

    S = load_release(dom, a, dev); rel = S["rel"]; X_raw = S["X_raw"]; phi = S["phi"]
    r0 = load(pth("round0", 0, "generic"))
    chk = fwd_check(rel, r0); log(f"# fwd_check (rebuilt release vs saved round-0 tensors): {chk}")
    if not chk["pass"]: log("# fwd_check FAILED -- rows below are NOT meaningful (ground rule 5); continuing so the discrepancy is on record")
    assert float((r0["x_raw"].to(dev) - X_raw).abs().max()) < 1e-12, "private images differ from the saved ones"
    common = dict(part="handover_jacobian", release=dom, source_job=args.job, job=myjob, N=a.N, r=a.r, k=a.k, T=a.T, K=a.K, n_prime=rel.Np, rank_C=rel.rank_C,
                  cert_line=rel.r - rel.Np, fwd_check=chk, rtol=RTOL, rank_floor="sigma_1(A_T J_phi V) (numrank ref)", iters=a.iters, seed=a.seed,
                  git=prov["git"], script_sha=prov["script_sha"], host=socket.gethostname(), cmd=" ".join(sys.argv),
                  prereg="10a x*_chart reference (oracle-start LM) | 10b coverage reading, alias flag is the falsifier (counted) | 10c union rank is a measurement, not a theorem test")
    dev_ = lambda t: t.to(dev)
    summaries = []

    def emit(pair, per):
        sm = summarise(per); summaries.append((pair, sm))
        row = dict(**common, **pair, per_item=per, summary=sm)
        with open(out, "a") as f: f.write(json.dumps(row) + "\n")
        log(f"#   >> {pair['pair']}: width {sm['union_width']} | angles min/med {sm['angles_min']:.1f}/{sm['angles_median']:.1f} deg, <1deg {sm['n_below_1deg']} | "
            f"rank@x*t V_t/V_t1/U {sm['J_at_xstar_t.V_t']['rank']}/{sm['J_at_xstar_t.V_t1']['rank']}/{sm['J_at_xstar_t.union']['rank']} | "
            f"cond@x*t1 V_t/V_t1/U {sm['J_at_xstar_t1.V_t']['cond']:.2e}/{sm['J_at_xstar_t1.V_t1']['cond']:.2e}/{sm['J_at_xstar_t1.union']['cond']:.2e} | "
            f"alias {sm['alias_flags']}/{sm['n']} reached {sm['reached_xstar']} short {sm['solver_short']}")

    # ---------------- global charts: round 0 -> A round t, A round t-1 -> A round t
    V0, m0 = dev_(r0["chart_V"]), dev_(r0["chart_mean"]); x_opt0 = dev_(r0["x_opt"])
    rec0 = torch.stack([dev_(r0["x_found"][:, p["matched_start"]]) for p in rows_src[("round0", 0, "generic")]["per_image"]], 1)
    # alias count for round 0 itself (its own chart; falsifier bookkeeping only)
    with torch.no_grad():
        o_rec = rel.obj_of(rec0); o_opt = rel.obj_of(x_opt0); e = torch.stack([rel_err(rec0[:, i], x_opt0[:, i:i + 1])[0] for i in range(a.N)])
        af = ((o_rec <= o_opt * (1 + 1e-6) + FLOOR) & (e >= LAND)).sum().item()
    log(f"# round 0 (own chart): alias flags {af}/{a.N} (row says {rows_src[('round0', 0, 'generic')]['alias_in_chart']})")
    for arm in args.a_arms:
        prev = (V0, m0, x_opt0, "round0"); t = 1
        while os.path.exists(pth("A", t, arm)):
            cur = load(pth("A", t, arm)); row = rows_src[("A", t, arm)]
            if row.get("identical_to"): log(f"# A round {t} {arm}: identical chart to '{row['identical_to']}' (same numbers) -- analysed anyway, marked")
            V1, m1, x1 = dev_(cur["chart_V"]), dev_(cur["chart_mean"]), dev_(cur["x_opt"]); xm = dev_(cur["x_matched"])
            per = []
            for i in range(a.N):
                d = analyse_pair(S, prev[0], prev[1], V1, m1, prev[2][:, i], x1[:, i], xm[:, i], x1[:, i], X_raw[:, i])
                d.update(truth=i, angles_min=d["angles"]["min_deg"], angles_median=d["angles"]["median_deg"]); per.append(d)
            emit(dict(pair=f"{prev[3]} -> A{t}:{arm}", variant="A", arm=arm, round_from=t - 1, round_to=t, chart_from=prev[3], chart_to=row["chart"],
                      identical_to=row.get("identical_to"), row_alias_in_chart=row["alias_in_chart"], per_index="truth"), per)
            prev = (V1, m1, x1, f"A{t}:{arm}"); t += 1

    # ---------------- local charts (variant B): rebuild the slot charts from the saved anchors, re-solve x*_chart
    rec_cls = rows_src[("round0", 0, "generic")]["recognition"]["rank1"]; pool_B = S["class_pool"](rec_cls); K = min(a.K, pool_B.shape[0])
    gr = torch.Generator().manual_seed(a.seed + 77)
    for arm in args.b_arms:
        if not os.path.exists(pth("B", 1, arm)): log(f"# B {arm}: no saved rounds for job {args.job} -- skipped"); continue
        slots_prev = [dev_(r0["x_cand"][:, j]) for j in range(r0["x_cand"].shape[1])]
        prev_charts = None                                                    # per slot: (V, mean, x*, truth index, label)
        t = 1
        while os.path.exists(pth("B", t, arm)):
            cur = load(pth("B", t, arm)); row = rows_src[("B", t, arm)]; Xs = dev_(cur["x_slots"]); per = []; t0 = time.time()
            new_charts = [None] * len(slots_prev)
            for j, xj in enumerate(slots_prev):
                nearest = int(rel_err(xj, X_raw).argmin())
                if arm == "recovery": anchor = xj
                elif arm == "random_anchor": anchor = pool_B[int(torch.randint(pool_B.shape[0], (1,), generator=gr))]
                else: anchor = X_raw[:, nearest]
                ch, nb = local_chart(anchor, pool_B, K, a.k, f"local:{arm}")
                i = int(rel_err(Xs[:, j], ch.project(X_raw)).argmin())       # the slot's nearest truth (bootstrap's rule: nearest in projection)
                ps = row["per_slot"][j]
                opt = oracle_optimum(rel, ch, X_raw[:, i:i + 1], a.iters); xs = opt["X"][:, 0]
                rebuild = dict(nearest_truth_saved=ps["nearest_truth"], nearest_truth_rebuilt=i, anchor_err_saved=ps["anchor_err_vs_truth"],
                               anchor_err_rebuilt=float(rel_err(anchor, X_raw)[i]), chart_err_saved=ps["chart_err_truth"], chart_err_rebuilt=float(ch.err(X_raw)[i]),
                               obj_opt_saved=ps["objective_opt"], obj_opt_rebuilt=opt["objective"][0], err_opt_truth_saved=ps["err_opt_truth"], err_opt_truth_rebuilt=opt["err_truth"][0])
                if t == 1: Vp, mp, xp, ip = V0, m0, x_opt0[:, i], i                      # round 0: the global chart, x*_chart of THIS slot's truth
                else: Vp, mp, xp, ip, _lab = prev_charts[j]                               # round t-1: this slot's own local chart and x*_chart
                d = analyse_pair(S, Vp, mp, ch.V, ch.mean, xp, xs, Xs[:, j], xs, X_raw[:, i])
                d.update(slot=j, truth=i, truth_prev=ip, rebuild_check=rebuild, angles_min=d["angles"]["min_deg"], angles_median=d["angles"]["median_deg"]); per.append(d)
                new_charts[j] = (ch.V, ch.mean, xs, i, f"B{t}:{arm}")
            drift = max(abs(p["rebuild_check"]["obj_opt_saved"] - p["rebuild_check"]["obj_opt_rebuilt"]) / max(p["rebuild_check"]["obj_opt_saved"], FLOOR) for p in per)
            log(f"#   B round {t} {arm}: charts rebuilt ({time.time() - t0:.0f}s); nearest-truth match {sum(p['rebuild_check']['nearest_truth_saved'] == p['rebuild_check']['nearest_truth_rebuilt'] for p in per)}/{len(per)}, "
                f"chart-err max |saved-rebuilt| {max(abs(p['rebuild_check']['chart_err_saved'] - p['rebuild_check']['chart_err_rebuilt']) for p in per):.1e}, "
                f"x*_chart objective max rel drift vs saved {drift:.1e}")
            emit(dict(pair=f"{'round0' if t == 1 else f'B{t-1}:{arm}'} -> B{t}:{arm}", variant="B", arm=arm, round_from=t - 1, round_to=t,
                      chart_from=("generic_pca" if t == 1 else f"local_pca:{arm}"), chart_to=f"local_pca:{arm}", identical_to=None,
                      row_alias_in_chart=row["alias_in_chart"], per_index="slot", rebuilt_local_charts=True), per)
            slots_prev = [Xs[:, j] for j in range(Xs.shape[1])]; prev_charts = new_charts; t += 1

    # ---------------- printed table
    log("\n| pair | union width | angles min/med (deg), #<1deg | rank@x*(t): V_t / V_t+1 / U | rank@x*(t+1): V_t / V_t+1 / U | gap@x*(t+1) V_t+1 | cond@x*(t+1): V_t / V_t+1 / U | obj x*(t) in t+1 / x*(t+1) in t | alias / reached / short |")
    log("|---|---|---|---|---|---|---|---|---|")
    rk = lambda v: (str(v[0]) if len(set(v)) == 1 else f"{min(v)}-{max(v)}")
    for pair, sm in summaries:
        log(f"| {pair['pair']} | {rk(sm['union_width'])}{' (>line)' if max(sm['union_width']) > common['cert_line'] else ''} | {sm['angles_min']:.1f}/{sm['angles_median']:.1f}, {rk(sm['n_below_1deg'])} | "
            f"{rk(sm['J_at_xstar_t.V_t']['rank'])} / {rk(sm['J_at_xstar_t.V_t1']['rank'])} / {rk(sm['J_at_xstar_t.union']['rank'])} | "
            f"{rk(sm['J_at_xstar_t1.V_t']['rank'])} / {rk(sm['J_at_xstar_t1.V_t1']['rank'])} / {rk(sm['J_at_xstar_t1.union']['rank'])} | {sm['J_at_xstar_t1.V_t1']['gap']:.1e} | "
            f"{sm['J_at_xstar_t1.V_t']['cond']:.2e} / {sm['J_at_xstar_t1.V_t1']['cond']:.2e} / {sm['J_at_xstar_t1.union']['cond']:.2e} | "
            f"{sm['obj_xt_in_chart_t1']:.1e} / {sm['obj_xt1_in_chart_t']:.1e} | {sm['alias_flags']} / {sm['reached_xstar']} / {sm['solver_short']} |")
    log(f"# done in {time.time() - t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
