#!/usr/bin/env python3
"""How many independent constraints does the release place on the RAW IMAGE, as a function of adapted DEPTH?

Three points (58 -> 102 -> 158 conditions on 784 pixels at 1, 2, 3 adapted layers, every supplied condition
independent) are three points.  Read as a trend they extrapolate to a chart-free determination of the image,
and a three-point extrapolation must not reach a document.  This runs the same measurement on a deep backbone
so the curve exists.

What is measured, per adapted-depth L, at each recorded image's own truth (no solve, no start, no chart):
  * rank of the stacked per-layer certificate Jacobian d/dx of [C_l h^l(x) / ||A_l h^l(x)||]_{l<=L}   -- `rank K`
  * rank of dh^1/dx                                                                                  -- `rank DF_1`
    Every adapted layer's pixel map factors through the first layer, so rank K <= rank DF_1 ALWAYS; DF_1 is the
    architecture's own ceiling and is reported beside K so the cap is visible rather than assumed.
  * the exact feature-space codimension sum_l rank(C_l) (C_l h = 0 is LINEAR in h: exact, global, no
    linearisation) and the ENCODER COST = codimension - pixel rank, i.e. what the network's own map destroys.
  * sigma_min at the rank and the condition number.  Formal independence is cheap; near-parallel conditions are
    full rank and practically empty, so the additivity reading survives only if sigma_min stays usable.

PRE-REGISTERED, before any row (see RESULTS.md "the extended-layer curve"):
  ADDITIVE   per-layer rank increments stay within a factor of 2 of the first three layers' mean, until the
             curve meets rank DF_1.  Then the 3-point line was a line.
  FLATTENING increments decay geometrically and the curve saturates well below rank DF_1 -- the 3 points were
             the early part of a curve, and the chart-free reading dies here.
  Independently of which: sigma_min at the rank fell ~4x per layer over the first three (0.28 -> 0.069 ->
  0.0115).  If that continues, formal rank keeps growing while the USABLE rank (conditions above the release's
  own noise floor) flattens -- reported as its own column, and it is the honest number.
This is an algebraic check at the truth, not an attack: it counts what the release pins around the true image,
never that an attacker can find it.

  python -u -m experiments.exact_inversion.layer_curve --model models/exact_inversion/mnist_mlp_d15w1000.pth
"""
import argparse, json, math, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.lora_exact_inversion import git_hash
from experiments.exact_inversion.trained_backbone import PCAChart, read_idx
from experiments.exact_inversion.certificate import certificate
from experiments.exact_inversion.deep_stack import inputs_of, run_training_deep, load_deep

torch.set_default_dtype(torch.float64)


def med(vals):
    s = sorted(vals); return s[len(s) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_d15w1000.pth")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--T", type=int, default=400); ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--adapt", nargs="*", type=int, default=None,
                    help="1-indexed layers to adapt; everything else is FROZEN. Adapting a layer with a frozen "
                         "NONLINEAR encoder below it is the only cell that can charge the encoder: the condition "
                         "is C phi(x), the pixel Jacobian is C.Dphi(x), and rank(C.Dphi) can be strictly less "
                         "than r - N'. The gap is the measurement. Default: every layer.")
    ap.add_argument("--layers", nargs="*", type=int, default=None,
                    help="adapted-depths L to measure (default 1,2,3,4,6,8,10,12,14,depth)")
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)

    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Ws, b1, ck = load_deep(a.model, dev)
    D = len(Ws); m = Ws[-1].shape[0]
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(Ws[0].shape[1])
    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    chart = PCAChart(Xtr_t, a.k, dev)
    X_real = Xte_t[idx].T.contiguous(); X_on = chart.psi(chart.coords_of(X_real)); y = yte_t[idx]
    npix = X_on.shape[0]
    Ls = a.layers or sorted({l for l in (1, 2, 3, 4, 6, 8, 10, 12, 14, D) if l <= D})

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    print(f"# layer_curve depth={D} width={Ws[0].shape[0]} backbone_acc={ck.get('test_acc')} "
          f"m={m} N={a.N} r={a.r} k={a.k} T={a.T} lr={a.lr} pixels={npix} git={git_hash()} dev={dev}", flush=True)

    adapted = set((x - 1 for x in a.adapt)) if a.adapt else set(range(D))
    gA = torch.Generator().manual_seed(a.seed + 11)
    A0s = [((a.sigma0 * torch.randn(a.r, W.shape[1], generator=gA)).to(dev) if l in adapted else None)
           for l, W in enumerate(Ws)]
    t0 = time.time()
    As, Bs = run_training_deep(X_on, Ws, b1, A0s, y, m, a.T, a.lr)
    print(f"# release trained on all {D} layers in {time.time()-t0:.1f}s", flush=True)

    H_true = inputs_of(X_on, Ws, b1, As, Bs)
    Cs, margins, Nps = [], [], []
    for l in range(D):
        if As[l] is None:
            Cs.append(None); Nps.append(-1); margins.append(0); continue
        C, Np, _ = certificate(As[l], Bs[l]); Cs.append(C); Nps.append(Np); margins.append(a.r - Np)
        res = torch.linalg.norm(C @ H_true[l], dim=0) / torch.linalg.norm(As[l] @ H_true[l], dim=0)
        emit(dict(part="LAYER", layer=l + 1, depth=D, r=a.r, n_prime=Np, certificate_margin=a.r - Np,
                  usable=bool(a.r - Np > 0), cert_residual_median=float(res.median()),
                  cert_residual_max=float(res.max()), git=git_hash()))
        print(f"  layer {l+1:2d}: N'={Np} margin={a.r-Np} cert residual median {float(res.median()):.2e}", flush=True)
    usable = [l for l in range(D) if Cs[l] is not None and margins[l] > 0]
    print(f"  usable layers {[l+1 for l in usable]} of {D}", flush=True)

    # rank DF_1 = the architecture's own ceiling on every deeper layer's pixel map
    def f1(x): return inputs_of(x.reshape(npix, 1), Ws, b1, As, Bs)[1].reshape(-1)
    df1 = []
    for i in range(a.N):
        sv = torch.linalg.svdvals(tf.jacfwd(f1)(X_on[:, i].contiguous()).detach())
        df1.append(int((sv > 1e-10 * sv[0]).sum()) if float(sv[0]) > 0 else 0)
    rank_df1 = med(df1)
    emit(dict(part="DF1", rank_DF1_median=rank_df1, per_image=df1, pixel_count=npix, depth=D,
              note="every adapted layer's pixel map factors through the first, so rank K <= rank DF_1",
              git=git_hash()))
    print(f"  rank DF_1 = {rank_df1} of {npix} pixels -- the ceiling on rank K at every depth", flush=True)

    for l in usable:                                                   # what the ENCODER below layer l can carry
        if l == 0: continue
        def enc(x, _l=l): return inputs_of(x.reshape(npix, 1), Ws, b1, As, Bs)[_l].reshape(-1)
        rks = []
        for i in range(a.N):
            sv = torch.linalg.svdvals(tf.jacfwd(enc)(X_on[:, i].contiguous()).detach())
            rks.append(int((sv > 1e-10 * sv[0]).sum()) if float(sv[0]) > 0 else 0)
        emit(dict(part="ENCODER", layer=l + 1, rank_Dphi_median=med(rks), per_image=rks, pixel_count=npix,
                  margin=margins[l], n_prime=Nps[l],
                  note="rank(C.Dphi) <= min(r - N', rank Dphi); the GAP against r - N' is what the encoder costs",
                  git=git_hash()))
        print(f"  encoder below layer {l+1}: rank Dphi = {med(rks)} of {npix}   (margin r-N' = {margins[l]})",
              flush=True)

    for L in Ls:
        layers = [l for l in usable if l < L]
        if not layers: continue
        def g_pix(x):
            hs = inputs_of(x.reshape(npix, 1), Ws, b1, As, Bs)
            return torch.cat([(Cs[l] @ hs[l]).reshape(-1) / torch.linalg.norm(As[l] @ hs[l]) for l in layers])
        out = []; t1 = time.time()
        for i in range(a.N):
            J = tf.jacfwd(g_pix)(X_on[:, i].contiguous()).detach()
            sv = torch.linalg.svdvals(J)
            rk = {lab: (int((sv > tol * sv[0]).sum()) if float(sv[0]) > 0 else 0)
                  for tol, lab in ((1e-10, "1e-10"), (1e-8, "1e-8"), (1e-6, "1e-6"))}
            r0 = rk["1e-10"]
            out.append(dict(image=i, rank_by_tol=rk, n_rows=int(J.shape[0]), sigma_max=float(sv[0]),
                            sigma_min_of_rank=float(sv[r0 - 1]) if r0 else 0.0,
                            cond_of_rank=float(sv[0] / sv[r0 - 1]) if r0 and float(sv[r0 - 1]) > 0 else float("inf"),
                            usable_above_1e8=int(sum(1 for v in sv if float(v / sv[0]) > 1e-8)) if float(sv[0]) > 0 else 0,
                            sigma_rel=[float(v / sv[0]) for v in sv[:min(32, len(sv))]] if float(sv[0]) > 0 else []))
        ranks = {lab: med([o["rank_by_tol"][lab] for o in out]) for lab in ("1e-10", "1e-8", "1e-6")}
        supplied = sum(margins[l] for l in layers)
        feat_codim = int(sum(int(torch.linalg.matrix_rank(Cs[l], rtol=1e-10)) for l in layers))
        emit(dict(part="PIXELRANK", n_layers=len(layers), layers_in_objective=[l + 1 for l in layers], depth=D,
                  independent_conditions_on_pixels_median=ranks, n_conditions_supplied=supplied,
                  encoder_gap=supplied - ranks["1e-10"], adapted_layers=[l + 1 for l in sorted(adapted)],
                  raw_pixel_cell=bool(0 in layers),
                  feature_space_codimension_exact=feat_codim, encoder_cost=feat_codim - ranks["1e-10"],
                  all_supplied_independent=bool(ranks["1e-10"] >= min(supplied, rank_df1)),
                  rank_DF1_median=rank_df1, headroom_to_DF1=rank_df1 - ranks["1e-10"],
                  usable_rank_above_1e8_median=med([o["usable_above_1e8"] for o in out]),
                  sigma_max_median=med([o["sigma_max"] for o in out]),
                  sigma_min_of_rank_median=med([o["sigma_min_of_rank"] for o in out]),
                  cond_median=med([o["cond_of_rank"] for o in out]),
                  pixel_count=npix, fraction_of_pixels=ranks["1e-10"] / npix, per_image=out,
                  r=a.r, N=a.N, k=a.k, seconds=time.time() - t1,
                  start_model="n/a (Jacobian at the truth, no solve)", claim_class="algebraic check",
                  git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv)))
        print(f"  L={len(layers):2d}: pixel rank {ranks['1e-10']:4d}/{npix} ({100*ranks['1e-10']/npix:.1f}%)  "
              f"supplied {supplied}  feature codim {feat_codim}  encoder cost {feat_codim-ranks['1e-10']}  "
              f"usable(>1e-8) {med([o['usable_above_1e8'] for o in out])}  "
              f"sigma_min {med([o['sigma_min_of_rank'] for o in out]):.3e}  "
              f"cond {med([o['cond_of_rank'] for o in out]):.3e}  DF_1 headroom {rank_df1-ranks['1e-10']}", flush=True)


if __name__ == "__main__":
    main()
