#!/usr/bin/env python3
"""Real-encoder test of the depth rank law: does the stacked certificate Jacobian saturate at k_1 (T5.2) or
STRICTLY BELOW k_1 at min_j(d_j + sum_{l<j} q_l) (F11's corrected law)?  ONE row per (config, depth), every
quantity on that row, a config key on every row, across a pre-registered tolerance ladder.

FIRST SENTENCE, because it is the thing most likely to be misread (GM, 2026-09-17): **this is a THEORY test of
the rank law, NOT an attack configuration.**  The discriminating regime needs a chart wide enough that the real
encoder can contract it below k_1 -- here the pixel space itself (k_1 up to ~692), which is ~10x past the
identifiability cap measured on real releases (k <= 66).  "A deep adapted stack is trapped at ~319, not the naive
692" is a statement about the rank LAW; it is not an attack number.  The attack-side re-costing needs d_j composed
with an attacker-buildable chart at k <= 66 (the `chart66` arm below), where every number changes.  Two separate
claims; the stronger-sounding one must not carry the other.

WHY A RUN AND NOT A ROW READ.  Section 19a's saved rows could not be turned into a two-law comparison: the ENCODER
rows carried no config key, d_j and the stacked rank lived on different rows, and file-order alignment produced a
measured 717 against a proved upper bound of 468 (a bound violation that refutes the alignment, not the theorem).
This run fixes exactly those defects.

WHY ZERO DRIFT / NO TRAINING.  T5.2 and the corrected law are both ZERO-DRIFT statements.  The clean test is
check_T5's construction -- the zero-drift certificate C_l = P_{col(A_{l,0} U_l)^perp} A_{l,0} with U_l a basis of
the base features col(H_l^0) -- evaluated on a REAL frozen deep encoder.  No LoRA training happens, so there is no
divergence gate to inherit from survival.py (its abs-vs-relative bug cannot arise where nothing is trained), and
the drift regime (T5.4) is deliberately out of scope.

THE CONFIG THAT DISCRIMINATES, verified by profile arithmetic before submission (GM requirement):
  The 3-layer 784-1000-1000-10 MLP provably CANNOT discriminate -- its pixel-rank profile is 784/784/690, a drop of
  ~94 arriving too late, so q_1 ~ k_1 and both laws saturate at k_1 immediately.  The deep 15-layer backbone drops
  784->692->219->187->138->104->85: putting the first adapted layer at the pre-drop (d_1~692) and adapting through
  the 692->219 cliff with margins large enough that sum q_l > k_1 gives corrected ~319 vs T5.2 ~692 -- a factor
  better than two, clear of any tolerance question.

PRE-REGISTERED, before any row:
  DISCRIMINATION (corrected law tight, T5.2 refuted):  in a config with corrected_pred < t52_pred, the measured
    stacked rank sits at corrected_pred and STRICTLY BELOW k_1.  (corrected_pred is a PROVED upper bound from
    nesting, so measured <= corrected_pred always; the test is whether it is TIGHT and far below t52_pred.)
  FALSIFIED:  the measured rank sits at t52_pred ~ k_1, i.e. saturation AT k_1 in a config where the laws differ.
  CONTROL (must exist in the SAME job/row-format):  a config with corrected_pred == t52_pred (rank-preserving,
    e.g. first adapted layer shallow, few layers) MUST saturate at that common value -- this verifies the harness
    can produce the non-discriminating outcome and is not rigged toward "below k_1".
  VACUOUS / real-scale null:  if no swept config has corrected_pred < t52_pred, the encoder does not contract
    enough against the achievable margins, and the two laws coincide at real scale.  A legitimate result.
  Every row records ||H0|| per layer with an UNDERFLOW test (not merely smallness): the field that decides two
    still-unexplained survival.py rows, and here it is measured rather than inferred from the architecture.

THE k-SWEEP (added 2026-09-17, the question M6 opened). M6 showed the two laws differ 1.8x at k=784 (pixel) and
coincide at k=66 (depth free, one layer saturates). This sweeps `k` to find `k*` — where depth STOPS being free,
i.e. the smallest chart width at which the full stack has `discriminates: True` and the measured rank sits strictly
below `k_1`. Same axis as the chart-window question, so each row also carries the chart error (filled) and slots
for the identifiability nullity + condition number (present, EMPTY — folded in from `experiments/utils/
identifiability.py`, 6e's tool, which owns the conventions). PRE-REGISTERED before submission:
  * The tolerance ladder is 1e-6 … 1e-14 (finer than M6's 1e-12, which reproduced an unconverged elbow). The
    RESULT is the elbow's STABILITY across the ladder (`ladder_converged`), never the value at the finest rung.
  * `k* = min k` with the full-stack `discriminates: True`; below it depth is free (both laws agree), above it the
    nesting binds. Report `k*` against the k<=66 identifiability cap: if `k* > cap`, the depth re-costing is a
    theory statement the attacker never reaches.
  * 6e's caveat, which is MINE to test: their `N·max(0, k-(m+r-N-1))` nullity law is validated on a SYNTHETIC
    AFFINE release; this harness has a real nonlinear 15-layer phi. A nullity that violates their law is a FINDING
    about depth and nonlinearity, to be reported as one, not reconciled. Sweep k at FIXED T (nullity is T-free but
    the condition number is not).

  python -u -m experiments.multilayer_cert.real_encoder_ranklaw --model models/exact_inversion/mnist_mlp_d15w1000.pth
"""
import argparse, json, math, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.trained_backbone import PCAChart, read_idx
from experiments.exact_inversion.deep_stack import inputs_of, load_deep
from experiments.multilayer_cert.common import provenance

torch.set_default_dtype(torch.float64)
LADDER = ("1e-6", "1e-8", "1e-10", "1e-12", "1e-13", "1e-14")   # finer rungs: the 1e-12 stop reproduced ambiguity
TOL = {"1e-6": 1e-6, "1e-8": 1e-8, "1e-10": 1e-10, "1e-12": 1e-12, "1e-13": 1e-13, "1e-14": 1e-14}


def med(v):
    s = sorted(v); return s[len(s) // 2]


def numrank(sv, tol):
    sv = sv.tolist()
    return int(sum(1 for v in sv if v > tol * sv[0])) if sv and sv[0] > 0 else 0


def zero_drift_cert(A0, U):
    """C = P_{col(A0 U)^perp} A0 -- the zero-drift certificate (check_T5), rank r - N'."""
    Q, _ = torch.linalg.qr(A0 @ U)
    return A0 - Q @ (Q.T @ A0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_d15w1000.pth")
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=108)
    ap.add_argument("--ks", nargs="*", type=int,
                    default=[16, 32, 66, 96, 128, 192, 256, 384, 512, 692, 784],
                    help="chart dimensions to sweep (the k-sweep: at what k does depth stop being free); "
                         "k >= pixel count uses the identity/pixel chart. FIXED T is implicit (no training here).")
    ap.add_argument("--first", nargs="*", type=int, default=[1, 3], help="1-indexed first adapted layer per config")
    ap.add_argument("--maxL", type=int, default=8, help="max adapted layers stacked from the first")
    ap.add_argument("--sigma0", type=float, default=None); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-fit", type=int, default=50000)
    ap.add_argument("--data-root", default="dataset_reconstruction/data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(); dev = torch.device(a.device)
    PROV = provenance(__file__)                          # attested provenance: commit (+ -dirty) AND a script hash

    Xtr, _ = read_idx(a.data_root, "train"); Xte, yte = read_idx(a.data_root, "test")
    Xtr_t = torch.tensor(Xtr[:a.n_fit], device=dev); Xte_t = torch.tensor(Xte, device=dev)
    Ws, b1, ck = load_deep(a.model, dev)
    D = len(Ws)
    if a.sigma0 is None: a.sigma0 = 1.0 / math.sqrt(Ws[0].shape[1])
    g = torch.Generator().manual_seed(a.seed + 7)
    idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
    X_real = Xte_t[idx].T.contiguous()                       # (784, N) raw pixels -- the truth points
    npix = X_real.shape[0]

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    print(f"# real_encoder_ranklaw depth={D} width={Ws[0].shape[0]} acc={ck.get('test_acc')} "
          f"N={a.N} r={a.r} maxL={a.maxL} pixels={npix} git={PROV['git']} script_sha={PROV['script_sha']} "
          f"dev={dev}", flush=True)
    print(f"# THEORY TEST of the rank law, NOT an attack config (pixel arm k_1~692 is ~10x the k<=66 cap)", flush=True)

    for k in a.ks:
        if k >= npix:                                        # identity/pixel chart: differentiate w.r.t. pixels
            k_eff, chart_label, in_dim = npix, "pixel", npix
            X_domain = X_real
            to_pixels = (lambda v: v.reshape(npix, -1))
            chart_err = 0.0                                  # pixels contain the image exactly
        else:
            chart = PCAChart(Xtr_t, k, dev)
            k_eff, chart_label, in_dim = k, f"pca{k}", k
            X_domain = chart.coords_of(X_real)               # (k, N) chart coords at the truth
            to_pixels = (lambda w, _k=k: chart.psi(w.reshape(_k, -1)))
            X_rec = chart.psi(X_domain)                      # chart error: what the k-chart fails to contain
            chart_err = float((torch.linalg.norm(X_rec - X_real, dim=0)
                               / torch.linalg.norm(X_real, dim=0)).median())

        # frozen base features as a function of the domain point (pixels, or chart coords via psi)
        def feats(v):                                        # -> list [h^0 .. h^{D-1}] for a single column v
            return inputs_of(to_pixels(v), Ws, b1)

        H0_all = feats(X_domain)                             # base features at all truths, per layer
        H0_norm = [float(torch.linalg.norm(H0_all[l], dim=0).median()) for l in range(D)]
        underflow = [bool(H0_norm[l] < 1e-290) for l in range(D)]

        for first in a.first:
            f0 = first - 1                                   # 0-indexed first adapted layer
            adapted = [l for l in range(f0, min(D, f0 + a.maxL))]
            if not adapted:
                continue
            # zero-drift certificate on each adapted layer's BASE features
            gA = torch.Generator().manual_seed(a.seed + 11 + first)
            Cs, dof = {}, {}
            for l in adapted:
                A0 = (a.sigma0 * torch.randn(a.r, Ws[l].shape[1], generator=gA)).to(dev)
                U = torch.linalg.qr(H0_all[l])[0][:, :a.N]
                Cs[l] = zero_drift_cert(A0, U)
                dof[l] = int(torch.linalg.matrix_rank(Cs[l], rtol=1e-10))   # r - N' (certificate rank)

            # per-layer d_l = rank M_l (encoder Jacobian to layer l) and q_l = rank(C_l M_l), per image
            d_layer, q_layer = {}, {}
            for l in adapted:
                dl, ql = [], []
                for i in range(a.N):
                    Ml = tf.jacfwd(lambda v, _l=l: feats(v)[_l].reshape(-1))(X_domain[:, i].contiguous()).detach()
                    sv_d = torch.linalg.svdvals(Ml)
                    dl.append(numrank(sv_d, 1e-10))
                    sv_q = torch.linalg.svdvals(Cs[l] @ Ml)
                    ql.append(numrank(sv_q, 1e-10))
                d_layer[l] = med(dl); q_layer[l] = med(ql)

            k1 = d_layer[adapted[0]]                          # nesting ceiling = first adapted layer's M-rank

            # stacked rank, per depth L, across the tolerance ladder, plus both law predictions
            for L in range(1, len(adapted) + 1):
                layers = adapted[:L]

                def g_stack(v, _ls=tuple(layers)):
                    hs = feats(v)
                    return torch.cat([(Cs[l] @ hs[l]).reshape(-1) for l in _ls])

                rows = []
                for i in range(a.N):
                    J = tf.jacfwd(g_stack)(X_domain[:, i].contiguous()).detach()
                    sv = torch.linalg.svdvals(J)
                    row = {lab: numrank(sv, TOL[lab]) for lab in LADDER}
                    row["sigma_max"] = float(sv[0]); row["n_rows"] = int(J.shape[0])
                    rows.append(row)
                meas = {lab: med([o[lab] for o in rows]) for lab in LADDER}

                dl = [d_layer[l] for l in layers]
                ql = [q_layer[l] for l in layers]
                Sq = sum(ql)
                cum = [sum(ql[:j]) for j in range(L)]        # sum_{l<j} q_l for j=1..L (0-indexed)
                t52 = min(k1, Sq)
                corrected = min([dl[j] + cum[j] for j in range(L)] + [Sq])
                # elbow stability across the ladder: spread of the measured rank over the tolerance cuts
                ladder_vals = [meas[lab] for lab in LADDER]
                fine = LADDER[-1]                            # finest rung, for the explicit-rung match flags
                emit(dict(part="RANKLAW", config=dict(model=a.model.split("/")[-1], chart=chart_label, k=k_eff,
                          in_dim=in_dim, first_adapted=first, seed=a.seed, r=a.r, N=a.N, depth=D),
                          n_layers=L, layers_in_objective=[l + 1 for l in layers],
                          d_j=dl, q_l=ql, cert_rank=[dof[l] for l in layers], k1=k1, sum_q=Sq,
                          t52_pred=int(t52), corrected_pred=int(corrected),
                          measured_rank_by_tol=meas, measured_at_1e10=meas["1e-10"], measured_at_finest=meas[fine],
                          ladder_spread=int(max(ladder_vals) - min(ladder_vals)),
                          ladder_converged=bool(meas[LADDER[-1]] - meas[LADDER[-2]] == 0),
                          discriminates=bool(corrected < t52),
                          saturated_below_k1=bool(meas[fine] < k1),
                          saturated_at_k1=bool(meas[fine] >= k1 - 1),
                          matches_corrected_at_finest=bool(abs(meas[fine] - corrected) <= 1),
                          matches_t52_at_finest=bool(abs(meas[fine] - t52) <= 1),
                          chart_error=chart_err,
                          cert_route_nullity_finest=in_dim - meas[fine],   # CERT route: chart dirs the stacked C leaves free (tol-dep)
                          release_route_nullity=None,        # 6e's SEPARATE job joins this on the k axis (release/recipe route)
                          condition_number=None,             # from 6e's identifiability(), joined on the same k
                          nullity_note=("cert_route and release_route nullities measure DIFFERENT zero sets "
                                        "({C h = 0} superset {replay residual = 0} superset {truth}); the cert one "
                                        "is a superset so it can be large while the release one is 0 — that is the "
                                        "nesting, NOT a disagreement. Never difference them. cert_route is "
                                        "zero-drift; release_route (6e) carries drift, a different regime."),
                          H0_norm_per_layer=[H0_norm[l] for l in layers], H0_underflow=[underflow[l] for l in layers],
                          sigma_max_median=med([o["sigma_max"] for o in rows]),
                          claim_class="theory rank-law test at the truth (no solve, no attack)",
                          note=("k-sweep for where depth stops being free; the pixel/full-k end is a THEORY test "
                                "(k_1 >> the k<=66 identifiability cap), the small-k end is attacker-buildable scale"),
                          git=PROV["git"], script_sha=PROV["script_sha"],
                          host=socket.gethostname(), cmd=" ".join(sys.argv)))
                print(f"  [{chart_label} k={k_eff} first={first} L={L}] k1={k1} Sq={Sq} "
                      f"| T5.2={t52} corr={corrected} discr={corrected < t52} "
                      f"| MEASURED {'/'.join(str(meas[t]) for t in LADDER)} "
                      f"{'BELOW k1' if meas[fine] < k1 else 'AT k1'} "
                      f"conv={meas[LADDER[-1]] - meas[LADDER[-2]] == 0} chart_err={chart_err:.2e}", flush=True)


if __name__ == "__main__":
    main()
