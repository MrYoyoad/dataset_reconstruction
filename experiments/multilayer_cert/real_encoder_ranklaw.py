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

PROGRAM FLAGS (plan 2026-09-18, packages P1 and P4; added 2026-09-18).  Default behaviour is byte-identical to the
rows of job 365681 (same rows, same fields, plus the new ones below).
  * `--r` is a LIST (P1): one config per rank, `r` on the config key; sigma0 stays 1/sqrt(n_in) per r.  r is the
    INNERMOST loop (k -> seed -> first -> r -> pattern): the encoder Jacobians M_l and d_j do not depend on r and are
    computed once per (k, seed, first) and reused across r.
  * `--seed` is a LIST (audit item 5): the seed selects the truths (randperm(seed+7)[:N]), A_0 (seed+11+first) and the
    random layer draws (seed+23+first); `seed` on the config key as today.
  * `rank_test_outcome` (audit item 6, the THIRD outcome) on every row: "compare" (gap_at_corrected > 1e3, so the
    +-1 flags matches_*_at_finest are meaningful and `rank_verdict` is one of corrected/t52/both/neither),
    "no_gap_vacuous" (no gap at the cut with a live Jacobian: the ladder and spectrum are the result, `rank_verdict`
    is None -- never a +-1 verdict without a gap), "dead" (dead_jacobian: collapsed phi).
  * `--layers` is a LIST of layer-selection PATTERNS (P4), each producing one or more layer lists; every RANKLAW row
    carries `layer_pattern` (the pattern string, "prefix" for today's behaviour) and `pattern_draw` (random draw
    index, else None).  Patterns, all relative to the ADAPTABLE layers = first..D in network order:
      prefix | alternate (nested: one row per depth L = 1..maxL) | suffix:<L> | middle:<L> | random:<L>:<n_draws>
      (seeded by --seed and first) | single:<l> | explicit:<l1,l2,...> (absolute 1-indexed network layers).
    Both law predictions are computed over the CHOSEN layers in network order: k_1 = d of the SHALLOWEST chosen
    layer, T5.2 = min(k_1, sum q_l), corrected = min_j (d_j + sum_{l<j} q_l) with j over the chosen layers and l<j
    the chosen layers earlier in the network.  FOR NON-PREFIX PATTERNS THE CORRECTED LAW'S NESTING READING IS WHAT
    IS BEING TESTED (plan P4): the nesting proof (F11) is stated for a contiguous prefix; applying it to a subset
    that skips layers is the hypothesis "the shallowest chosen layer caps everything and deeper layers add only
    what fits inside it", and a measured rank ABOVE corrected_pred on such a row is a finding against that
    reading, not a harness bug.  A0 is drawn for EVERY adaptable layer in network order from one generator per
    (seed, first), so A0[l] is the same matrix under every pattern (and identical to the prefix-only harness).
  * Condition-number fields (P1/P4 metric): `cond_at_1e10`, `cond_at_corrected`, `cond_at_fp16` = median over
    images of sigma_1/sigma_c at that cut, None where c = 0 or sigma_c is not a real singular value (the gap's
    dead-Jacobian guard); `sigma_at_rank_1e10` + `rank_1e10_per_image` let the number be recomputed.

  python -u -m experiments.multilayer_cert.real_encoder_ranklaw --model models/exact_inversion/mnist_mlp_d15w1000.pth
  python -u -m experiments.multilayer_cert.real_encoder_ranklaw --model models/exact_inversion/mnist_mlp_d15w1000_full.pth \
      --r 8 16 32 64 108 256 --layers prefix suffix:2 middle:4 alternate random:2:3 single:5
"""
import argparse, json, math, socket, sys, time
import torch, torch.func as tf

from experiments.exact_inversion.trained_backbone import PCAChart, read_idx
from experiments.exact_inversion.deep_stack import inputs_of, load_deep
from experiments.multilayer_cert.common import provenance

torch.set_default_dtype(torch.float64)
LADDER = ("bf16", "fp16", "1e-4", "1e-5", "1e-6", "1e-8", "1e-10", "1e-12",   # UPWARD to the precision adapters
          "1e-13", "1e-14", "1e-15", "1e-16")   # SHIP in (bf16/fp16 are the attacker-relevant cuts) AND down to
TOL = {"bf16": 3.9e-3, "fp16": 4.9e-4, "1e-4": 1e-4, "1e-5": 1e-5,            # machine precision (6e: a deep phi
       "1e-6": 1e-6, "1e-8": 1e-8, "1e-10": 1e-10, "1e-12": 1e-12,           # has no gap; the rank is a choice of
       "1e-13": 1e-13, "1e-14": 1e-14, "1e-15": 1e-15, "1e-16": 1e-16}       # threshold, and the SHIP cut is coarse)


def med(v):
    s = sorted(v); return s[len(s) // 2]


def numrank(sv, tol):
    sv = sv.tolist()
    return int(sum(1 for v in sv if v > tol * sv[0])) if sv and sv[0] > 0 else 0


def zero_drift_cert(A0, U):
    """C = P_{col(A0 U)^perp} A0 -- the zero-drift certificate (check_T5), rank r - N'."""
    Q, _ = torch.linalg.qr(A0 @ U)
    return A0 - Q @ (Q.T @ A0)


def layer_selections(patterns, adaptable, maxL, seed, first):
    """Expand `--layers` patterns into (pattern, draw, layers) triples; layers are 0-indexed, in NETWORK ORDER.
    `adaptable` = every layer from the first adapted one to the end of the network (0-indexed, ascending).
      prefix              adaptable[:L] for L = 1..maxL      (today's behaviour; one triple per depth)
      alternate           every other adaptable layer from `first`, nested the same way, up to maxL layers
      suffix:<L>          the last L adaptable layers         (one triple)
      middle:<L>          a centred block of L adaptable layers (one triple)
      random:<L>:<n>      n subsets of size L without replacement; generator seeded by (seed, first); index in `draw`
      single:<l>          one ABSOLUTE 1-indexed network layer
      explicit:<l1,..>    ABSOLUTE 1-indexed network layers, sorted into network order
    single/explicit entries with a layer outside `adaptable` (before `first`, or past the network) are skipped with
    a printed note, as is any size-L pattern with L > len(adaptable) (never silently truncated to a different test)."""
    out, n = [], len(adaptable)

    def skip(pat, why):
        print(f"# --layers {pat}: skipped for first={first} ({why})", flush=True)

    for pat in patterns:
        name, *args = pat.split(":")
        if name == "prefix" or name == "alternate":
            ls = (adaptable if name == "prefix" else adaptable[::2])[:maxL]
            out += [(pat, None, ls[:L]) for L in range(1, len(ls) + 1)]
        elif name in ("suffix", "middle", "random"):
            L = int(args[0])
            if L < 1 or L > n:
                skip(pat, f"L={L} outside 1..{n} adaptable layers"); continue
            if name == "suffix":
                out.append((pat, None, adaptable[n - L:]))
            elif name == "middle":
                s0 = (n - L) // 2; out.append((pat, None, adaptable[s0:s0 + L]))
            else:
                g = torch.Generator().manual_seed(seed + 23 + first)
                for i in range(int(args[1])):
                    pick = sorted(torch.randperm(n, generator=g)[:L].tolist())
                    out.append((pat, i, [adaptable[j] for j in pick]))
        elif name in ("single", "explicit"):
            ls = sorted({int(x) - 1 for x in args[0].split(",")})
            bad = [l + 1 for l in ls if l not in adaptable]
            if bad:
                skip(pat, f"layers {bad} not adaptable (first={first}, network has {adaptable[-1] + 1} layers)"); continue
            out.append((pat, None, ls))
        else:
            raise SystemExit(f"unknown --layers pattern {pat!r}")
    return out


def rank_outcome(dead, gap_at_corr, match_corrected, match_t52):
    """The THIRD outcome (audit item 6).  "compare": a real gap (> 1e3) at corrected_pred, so the +-1 flags are
    meaningful and the verdict names which law the finest rung lands on; "no_gap_vacuous": live Jacobian, no gap at
    the cut -- the ladder and spectrum are the result, verdict None (never a +-1 verdict without a gap); "dead": a
    collapsed Jacobian."""
    if dead: return "dead", None
    if gap_at_corr is None or not (gap_at_corr > 1e3): return "no_gap_vacuous", None
    verdict = ("both" if (match_corrected and match_t52) else "corrected" if match_corrected
               else "t52" if match_t52 else "neither")
    return "compare", verdict


def cond_at(sv, c):
    """sigma_1 / sigma_c at rank c; None unless sigma_c is a REAL singular value (the gap's dead-Jacobian guard:
    sv[0] above an absolute floor AND sv[c-1]/sv[0] above fp64 resolution) -- a collapsed Jacobian has no condition
    number, and c = 0 has none either."""
    if sv is None or c <= 0 or c > len(sv): return None
    s0, sc = float(sv[0]), float(sv[c - 1])
    if not (s0 > 1e-25 and sc / s0 > 1e-14): return None
    return s0 / sc


def cond_fields(svs, ranks_1e10, ranks_fp16, corrected):
    """The condition-number row fields (plan 2026-09-18): median over images of sigma_1/sigma_c at the 1e-10 rung
    (c = that image's rank there), at c = corrected_pred, and at the fp16 rung; None when no image has a defined
    value.  `sigma_at_rank_1e10` (per image, sigma_c at the 1e-10 rank) + `rank_1e10_per_image` recompute cond_at_1e10."""
    def medn(vals):
        v = [x for x in vals if x is not None]; return med(v) if v else None
    return dict(cond_at_1e10=medn([cond_at(sv, c) for sv, c in zip(svs, ranks_1e10)]),
                cond_at_corrected=medn([cond_at(sv, corrected) for sv in svs]),
                cond_at_fp16=medn([cond_at(sv, c) for sv, c in zip(svs, ranks_fp16)]),
                rank_1e10_per_image=[int(c) for c in ranks_1e10],
                sigma_at_rank_1e10=[(float(sv[c - 1]) if (sv is not None and 0 < c <= len(sv)) else None)
                                    for sv, c in zip(svs, ranks_1e10)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exact_inversion/mnist_mlp_d15w1000.pth")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--r", nargs="*", type=int, default=[108], help="LoRA ranks, one config per rank (P1)")
    ap.add_argument("--ks", nargs="*", type=int,
                    default=[16, 32, 66, 96, 128, 192, 256, 384, 512, 692, 784],
                    help="chart dimensions to sweep (the k-sweep: at what k does depth stop being free); "
                         "k >= pixel count uses the identity/pixel chart. FIXED T is implicit (no training here).")
    ap.add_argument("--first", nargs="*", type=int, default=[1, 3], help="1-indexed first adapted layer per config")
    ap.add_argument("--maxL", type=int, default=8, help="max adapted layers stacked from the first")
    ap.add_argument("--layers", nargs="*", default=["prefix"],
                    help="layer-selection patterns (P4): prefix | alternate | suffix:<L> | middle:<L> | random:<L>:<n> | "
                         "single:<l> | explicit:<l1,l2,...>; see the module docstring")
    ap.add_argument("--sigma0", type=float, default=None)
    ap.add_argument("--seed", nargs="*", type=int, default=[1], help="seeds (truths, A_0, random layer draws); one config per seed")
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
    truths = {}
    for seed in a.seed:
        g = torch.Generator().manual_seed(seed + 7)
        idx = torch.randperm(Xte_t.shape[0], generator=g)[:a.N].to(dev)
        truths[seed] = Xte_t[idx].T.contiguous()                 # (784, N) raw pixels -- the truth points
        print(f"# private join-key indices: {idx.tolist()} (seed {seed})", flush=True)   # 6e joins on these exact indices
    npix = Xte_t.shape[1]

    def emit(row):
        print(json.dumps(row), flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")

    print(f"# real_encoder_ranklaw depth={D} width={Ws[0].shape[0]} acc={ck.get('test_acc')} "
          f"N={a.N} r={a.r} seeds={a.seed} maxL={a.maxL} layers={a.layers} pixels={npix} git={PROV['git']} script_sha={PROV['script_sha']} "
          f"dev={dev}", flush=True)
    print(f"# THEORY TEST of the rank law, NOT an attack config (pixel arm k_1~692 is ~10x the k<=66 cap)", flush=True)

    for k in a.ks:
        chart = None if k >= npix else PCAChart(Xtr_t, k, dev)   # the chart does not depend on the seed
        if chart is None:                                    # identity/pixel chart: differentiate w.r.t. pixels
            k_eff, chart_label, in_dim = npix, "pixel", npix
        else:
            k_eff, chart_label, in_dim = k, f"pca{k}", k
        for seed in a.seed:
            X_real = truths[seed]
            if chart is None:
                X_domain = X_real
                to_pixels = (lambda v: v.reshape(npix, -1))
                chart_err = 0.0                              # pixels contain the image exactly
            else:
                X_domain = chart.coords_of(X_real)           # (k, N) chart coords at the truth
                to_pixels = (lambda w, _k=k, _c=chart: _c.psi(w.reshape(_k, -1)))
                X_rec = chart.psi(X_domain)                  # chart error: what the k-chart fails to contain
                chart_err = float((torch.linalg.norm(X_rec - X_real, dim=0)
                                   / torch.linalg.norm(X_real, dim=0)).median())

            # frozen base features as a function of the domain point (pixels, or chart coords via psi)
            def feats(v):                                    # -> list [h^0 .. h^{D-1}] for a single column v
                return inputs_of(to_pixels(v), Ws, b1)

            H0_all = feats(X_domain)                         # base features at all truths, per layer
            H0_norm = [float(torch.linalg.norm(H0_all[l], dim=0).median()) for l in range(D)]
            underflow = [bool(H0_norm[l] < 1e-290) for l in range(D)]

            for first in a.first:
                f0 = first - 1                               # 0-indexed first adapted layer
                adaptable = list(range(f0, D))               # every layer from the first adapted one, network order
                sels = layer_selections(a.layers, adaptable, a.maxL, seed, first)
                if not sels:
                    continue
                needed = sorted({l for _, _, ls in sels for l in ls})   # layers any pattern puts in the objective

                # encoder Jacobians M_l (to each needed layer, per image) and d_l = rank M_l: r-INDEPENDENT, computed
                # once per (k, seed, first) and reused by every r below
                Ms, d_layer = {}, {}
                for l in needed:
                    Ms[l] = [tf.jacfwd(lambda v, _l=l: feats(v)[_l].reshape(-1))(X_domain[:, i].contiguous()).detach()
                             for i in range(a.N)]
                    d_layer[l] = med([numrank(torch.linalg.svdvals(Ml), 1e-10) for Ml in Ms[l]])

                for r in a.r:
                    # zero-drift certificate on each needed layer's BASE features.  A0 is drawn for EVERY adaptable
                    # layer in network order from one generator per (seed, first): A0[l] is the same matrix under
                    # every pattern, and byte-identical to the prefix-only harness for the prefix layers.
                    gA = torch.Generator().manual_seed(seed + 11 + first)
                    Cs, dof = {}, {}
                    for l in adaptable:
                        A0 = a.sigma0 * torch.randn(r, Ws[l].shape[1], generator=gA)
                        if l not in needed:
                            continue
                        A0 = A0.to(dev)
                        U = torch.linalg.qr(H0_all[l])[0][:, :a.N]
                        Cs[l] = zero_drift_cert(A0, U)
                        dof[l] = int(torch.linalg.matrix_rank(Cs[l], rtol=1e-10))   # r - N' (certificate rank)
                    # q_l = rank(C_l M_l), per image then median
                    q_layer = {l: med([numrank(torch.linalg.svdvals(Cs[l] @ Ml), 1e-10) for Ml in Ms[l]]) for l in needed}

                    # stacked rank per layer selection, across the tolerance ladder, plus both law predictions
                    # computed over the CHOSEN layers in network order (k_1 = the SHALLOWEST chosen layer's M-rank)
                    for pat, draw, layers in sels:
                        L = len(layers)
                        k1 = d_layer[layers[0]]

                        def g_stack(v, _ls=tuple(layers)):
                            hs = feats(v)
                            return torch.cat([(Cs[l] @ hs[l]).reshape(-1) for l in _ls])

                        dl = [d_layer[l] for l in layers]
                        ql = [q_layer[l] for l in layers]
                        Sq = sum(ql)
                        cum = [sum(ql[:j]) for j in range(L)]    # sum_{l<j} q_l for j=1..L (0-indexed)
                        t52 = min(k1, Sq)
                        corrected = min([dl[j] + cum[j] for j in range(L)] + [Sq])

                        rows, gaps, spec0, spec0_lo, svs = [], [], None, None, []
                        for i in range(a.N):
                            J = tf.jacfwd(g_stack)(X_domain[:, i].contiguous()).detach()
                            sv = torch.linalg.svdvals(J); svs.append(sv)
                            row = {lab: numrank(sv, TOL[lab]) for lab in LADDER}
                            row["sigma_max"] = float(sv[0]); row["n_rows"] = int(J.shape[0])
                            rows.append(row)
                            # 6e's decisive diagnostic: a REAL rank has a GAP (sv[c-1]/sv[c] >> 1) at the effective
                            # rank; smooth decay (no gap, ratio ~ 1) means "rank" is a choice of threshold, not a
                            # property.  DEAD-JACOBIAN GUARD (convention 4, 6e 2026-09-18): a collapsed phi gives a
                            # ~0 Jacobian, where sv[c-1]/sv[c] is 0/0 or inf and reads as a strong gap for a net
                            # computing nothing. The gap is meaningful only if sv[c-1] is a REAL singular value:
                            # sv[0] above an absolute floor AND sv[c-1]/sv[0] above fp64 resolution.
                            if 0 < corrected < len(sv) and float(sv[0]) > 1e-25 and float(sv[corrected - 1] / sv[0]) > 1e-14:
                                gaps.append(float(sv[corrected - 1] / sv[corrected]))
                            if i == 0 and float(sv[0]) > 0:  # FULL spectral profile: if there is no gap, the
                                spec0_lo = 0                  # profile IS the honest object, not a rank integer
                                spec0 = [float(sv[j] / sv[0]) for j in range(len(sv))]
                        meas = {lab: med([o[lab] for o in rows]) for lab in LADDER}
                        ladder_vals = [meas[lab] for lab in LADDER]
                        fine = LADDER[-1]                    # finest rung, for the explicit-rung match flags
                        gap_at_corr = med(gaps) if gaps else None    # >>1: real effective rank at `corrected`; ~1: no gap
                        cond = cond_fields(svs, [o["1e-10"] for o in rows], [o["fp16"] for o in rows], corrected)
                        c10 = None if cond["cond_at_1e10"] is None else f"{cond['cond_at_1e10']:.2e}"
                        dead = bool(med([o["sigma_max"] for o in rows]) < 1e-25)   # collapsed phi guard
                        match_c = bool(abs(meas[fine] - corrected) <= 1); match_t = bool(abs(meas[fine] - t52) <= 1)
                        # verdict at the 1e-10 rung (the record's rung; the finest rung sits at ambient wherever
                        # there is no gap, so a finest-rung verdict reads "neither" while 1e-10 lands exactly)
                        match_c10 = bool(abs(meas["1e-10"] - corrected) <= 1); match_t10 = bool(abs(meas["1e-10"] - t52) <= 1)
                        outcome, verdict = rank_outcome(dead, gap_at_corr, match_c10, match_t10)
                        emit(dict(part="RANKLAW", config=dict(model=a.model.split("/")[-1], chart=chart_label, k=k_eff,
                                  in_dim=in_dim, first_adapted=first, seed=seed, r=r, N=a.N, depth=D),
                                  n_layers=L, layers_in_objective=[l + 1 for l in layers],
                                  layer_pattern=pat, pattern_draw=draw, nesting_ceiling_layer=layers[0] + 1,
                                  d_j=dl, q_l=ql, cert_rank=[dof[l] for l in layers], k1=k1, sum_q=Sq,
                                  q_l_formula=[min(dof[l], d_layer[l]) for l in layers],   # T5.2's min(r-N', rank M_l)
                                  q_l_measured_matches_formula=bool(ql == [min(dof[l], d_layer[l]) for l in layers]),
                                  t52_pred=int(t52), corrected_pred=int(corrected),
                                  measured_rank_by_tol=meas, measured_at_1e10=meas["1e-10"], measured_at_finest=meas[fine],
                                  usable_rank_fp16=meas["fp16"], usable_rank_bf16=meas["bf16"],   # DEPLOYMENT precision:
                                  # the attacker-relevant usable rank is set by the precision the adapter SHIPS in, not ours
                                  ambient_maxrank=min(rows[0]["n_rows"], in_dim),   # tau->0 count saturates HERE; a value
                                  # the count merely CROSSES on the way to ambient is not a real rank (6e's tau->0 test)
                                  ladder_spread=int(max(ladder_vals) - min(ladder_vals)),
                                  ladder_converged=bool(meas[LADDER[-1]] - meas[LADDER[-2]] == 0),
                                  gap_at_corrected=gap_at_corr,
                                  dead_jacobian=dead,
                                  real_rank_at_corrected=bool(gap_at_corr is not None and gap_at_corr > 10),
                                  spectrum_window_img0=spec0, spectrum_window_start=spec0_lo,
                                  discriminates=bool(corrected < t52),
                                  saturated_below_k1=bool(meas[fine] < k1),
                                  saturated_at_k1=bool(meas[fine] >= k1 - 1),
                                  matches_corrected_at_finest=match_c,
                                  matches_t52_at_finest=match_t,
                                  rank_test_outcome=outcome, rank_verdict=verdict, rank_verdict_rung="1e-10",   # the THIRD outcome; no +-1 verdict without a gap
                                  chart_error=chart_err,
                                  cert_route_nullity_finest=in_dim - meas[fine],   # CERT route: chart dirs the stacked C leaves free (tol-dep)
                                  release_route_nullity=None,        # 6e's SEPARATE job joins this on the k axis (release/recipe route)
                                  # condition number: the former `condition_number=None` placeholder is REPLACED by the
                                  # measured fields cond_at_1e10 / cond_at_corrected / cond_at_fp16 (median over images
                                  # of sigma_1/sigma_c) + sigma_at_rank_1e10 / rank_1e10_per_image to recompute them
                                  **cond,
                                  nullity_note=("cert_route and release_route nullities measure DIFFERENT zero sets "
                                                "({C h = 0} superset {replay residual = 0} superset {truth}); the cert one "
                                                "is a superset so it can be large while the release one is 0 — that is the "
                                                "nesting, NOT a disagreement. Never difference them. cert_route is "
                                                "zero-drift; release_route (6e) carries drift, a different regime."),
                                  H0_norm_per_layer=[H0_norm[l] for l in layers], H0_underflow=[underflow[l] for l in layers],
                                  sigma_max_median=med([o["sigma_max"] for o in rows]),
                                  claim_class="theory rank-law test at the truth (no solve, no attack)",
                                  note=("k-sweep for where depth stops being free; the pixel/full-k end is a THEORY test "
                                        "(k_1 >> the k<=66 identifiability cap), the small-k end is attacker-buildable scale"
                                        + ("" if pat == "prefix" else "; NON-PREFIX layer pattern: the corrected law's "
                                           "nesting reading over the chosen layers is what is being tested (plan P4)")),
                                  git=PROV["git"], script_sha=PROV["script_sha"],
                                  host=socket.gethostname(), cmd=" ".join(sys.argv)))
                        print(f"  [{chart_label} k={k_eff} seed={seed} r={r} first={first} {pat}{'' if draw is None else '#' + str(draw)} "
                              f"L={L} layers={[l + 1 for l in layers]}] k1={k1} Sq={Sq} "
                              f"| T5.2={t52} corr={corrected} discr={corrected < t52} "
                              f"| MEASURED {'/'.join(str(meas[t]) for t in LADDER)} "
                              f"{'BELOW k1' if meas[fine] < k1 else 'AT k1'} "
                              f"conv={meas[LADDER[-1]] - meas[LADDER[-2]] == 0} chart_err={chart_err:.2e} "
                              f"cond1e10={c10} outcome={outcome}/{verdict}", flush=True)
                del Ms


if __name__ == "__main__":
    main()
