"""Part 3 — How to measure it: the secret-swap test (4 slides).

M1  The test: hide one image, swap it, ask the adapter   (native schematic D / D' -> K seeds -> two clouds)
M2  Why this function: the best any attacker can do      (native raw-vs-whitened cartoon + equivalences)
M3  Making the estimator honest                          (estimator_honest.png + 3-way / sign-flip null)
M4  The plan: what each arm was built to test            (native 9-row table)

Sources: experiments/dataset_sensitivity/whitened_metric.py:1-58, notes/whitened_sensitivity_metric.md,
notes/dataset_sensitivity_program_plan.md (I.1, II, III).
"""
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_LINE
from . import config as C
from . import helpers as H
from .eq_render import render_math, render_lines

Y0 = C.CT + Inches(0.55)          # first content row (below the lead line)


# ---- local drawing helpers (native shapes only) -----------------------------------------
def _oval(slide, cx, cy, w, h, *, fill=None, line=C.GRAY, width=1.0, rot=0.0, dash=False):
    shp = H.add_rect(slide, cx - w // 2, cy - h // 2, w, h, fill_color=fill, line_color=line,
                     line_width=width, shape=MSO_SHAPE.OVAL)
    shp.rotation = rot
    if dash:
        shp.line.dash_style = MSO_LINE.DASH
    return shp


def _dot(slide, cx, cy, d, color):
    return H.add_rect(slide, cx - d // 2, cy - d // 2, d, d, fill_color=color, line_color=None,
                      shape=MSO_SHAPE.OVAL)


def _label(slide, text, x, y, w, h=Inches(0.3), *, size=C.SZ_SMALL, color=C.GRAY, bold=False,
           italic=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP):
    return H.add_text(slide, text, x, y, w, h, size=size, color=color, bold=bold, italic=italic,
                      align=align, anchor=anchor)


# deterministic "random" offsets for the seed clouds (unit disc, rotated/stretched per cloud)
_CLOUD = [(-0.62, 0.10), (-0.40, -0.35), (-0.25, 0.42), (-0.10, -0.05), (0.05, 0.30), (0.12, -0.45),
          (0.30, 0.05), (0.42, 0.38), (0.55, -0.20), (0.68, 0.15), (-0.50, -0.05), (0.20, -0.18),
          (-0.05, 0.55), (0.48, -0.48), (-0.30, 0.15), (0.35, 0.25)]


def _cloud(slide, cx, cy, ax, ay, color, *, dot=Inches(0.09)):
    """Scatter of seeds inside an ellipse with semi-axes ax (along x) and ay (along y)."""
    for ux, uy in _CLOUD:
        _dot(slide, cx + int(ux * ax), cy + int(uy * ay), dot, color)


def _thumb_grid(slide, x, y, n_cols, n_rows, sq, gap, *, swap_idx, swap_style):
    """Dataset as a grid of little image squares; one square is the hidden/swapped image."""
    for i in range(n_cols * n_rows):
        r, c = divmod(i, n_cols)
        sx, sy = x + c * (sq + gap), y + r * (sq + gap)
        if i == swap_idx:
            if swap_style == "hidden":
                H.add_rect(slide, sx, sy, sq, sq, fill_color=C.LIGHTRED_FILL, line_color=C.RED, line_width=2.25)
            else:
                H.add_rect(slide, sx, sy, sq, sq, fill_color=C.LIGHTGREEN_FILL, line_color=C.GREEN_OK, line_width=2.25)
        else:
            H.add_rect(slide, sx, sy, sq, sq, fill_color=C.LIGHTGRAY_FILL, line_color=C.LGRAY, line_width=0.75)


# --------------------------------------------------------------------------------------
def slide_m1(prs):
    s = H.new_slide(prs)
    H.add_title(s, "The test: hide one image, swap it, ask the adapter")
    H.add_lead(s, "one number: how far the swap moves the adapter, in units of training noise")

    # ---- left: the two datasets ----
    sq, gap = Inches(0.36), Inches(0.08)
    n_cols, n_rows = 4, 2
    gw = n_cols * sq + (n_cols - 1) * gap
    gh = n_rows * sq + (n_rows - 1) * gap
    card_w, card_h = Inches(2.75), gh + Inches(0.62)
    cx0 = C.MX
    y_d = Y0 + Inches(0.05)
    y_dp = y_d + card_h + Inches(0.35)
    for yy, name, style, col in ((y_d, "D", "hidden", C.RED), (y_dp, "D′", "swapped", C.GREEN_OK)):
        H.add_rect(s, cx0, yy, card_w, card_h, fill_color=C.WHITE, line_color=C.GRAY, line_width=1.0,
                   shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        H.add_text(s, name, cx0 + Inches(0.12), yy + Inches(0.05), Inches(0.6), Inches(0.42), size=20, bold=True,
                   font=C.HEAD, color=col, anchor=MSO_ANCHOR.MIDDLE)
        _thumb_grid(s, cx0 + Inches(0.75), yy + Inches(0.45), n_cols, n_rows, sq, gap, swap_idx=5, swap_style=style)
    _label(s, "same N images, one hidden", cx0 + Inches(0.75), y_d + card_h - Inches(0.02), Inches(2.0),
           size=11, italic=True, align=PP_ALIGN.LEFT)
    _label(s, "that one replaced", cx0 + Inches(0.75), y_dp + card_h - Inches(0.02), Inches(2.0),
           size=11, italic=True, align=PP_ALIGN.LEFT)

    # ---- middle: fine-tune K seeds each ----
    mx = cx0 + card_w + Inches(0.45)
    mw, mh = Inches(1.75), Inches(1.15)
    my = (y_d + y_dp + card_h) // 2 - mh // 2
    H.add_rect(s, mx, my, mw, mh, fill_color=C.LIGHTBLUE_FILL, line_color=C.BLUE, line_width=1.25,
               shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    H.add_text(s, "LoRA fine-tune\nK seeds per side", mx, my, mw, mh, size=14, bold=True, font=C.HEAD,
               color=C.BLUE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    H.add_arrow(s, cx0 + card_w + Inches(0.05), y_d + card_h // 2, mx - Inches(0.05), my + mh // 3, color=C.GRAY)
    H.add_arrow(s, cx0 + card_w + Inches(0.05), y_dp + card_h // 2, mx - Inches(0.05), my + 2 * mh // 3, color=C.GRAY)

    # ---- right: two clouds of adapters, their centres, the shift, the noise ellipse ----
    rx = mx + mw + Inches(0.45)
    rw = C.SL_W - C.MX - rx
    ax, ay = Inches(1.35), Inches(0.72)      # semi-axes of the seed ellipse (before rotation)
    cy = my + mh // 2 - Inches(0.15)
    c1x = rx + Inches(1.7)
    c2x = rx + rw - Inches(1.85)
    H.add_arrow(s, mx + mw + Inches(0.05), my + mh // 2, rx + Inches(0.25), cy, color=C.GRAY)
    for cxx, col, fill in ((c1x, C.RED, C.LIGHTRED_FILL), (c2x, C.GREEN_OK, C.LIGHTGREEN_FILL)):
        _oval(s, cxx, cy, 2 * ax, 2 * ay, fill=fill, line=col, width=1.0, rot=-18.0, dash=True)
        _cloud(s, cxx, cy, int(ax * 0.85), int(ay * 0.85), col)
        _dot(s, cxx, cy, Inches(0.2), C.BLACK)
    _label(s, "μ(D)", c1x - Inches(0.6), cy + Inches(0.12), Inches(1.2), size=14, color=C.BLACK, bold=True)
    _label(s, "μ(D′)", c2x - Inches(0.6), cy + Inches(0.12), Inches(1.2), size=14, color=C.BLACK, bold=True)
    H.add_arrow(s, c1x + Inches(0.12), cy, c2x - Inches(0.12), cy, color=C.BLUE, width=2.5)
    _label(s, "Δμ", (c1x + c2x) // 2 - Inches(0.5), cy - Inches(0.42), Inches(1.0), size=16,
           color=C.BLUE, bold=True)
    _label(s, "Σ_seed : the spread over seeds (training noise)", c1x - ax, cy - ay - Inches(0.62),
           c2x - c1x + 2 * ax, size=C.SZ_SMALL, italic=True)
    _label(s, "each dot = one trained adapter  ΔW = BA", c1x - ax, cy + ay + Inches(0.3),
           c2x - c1x + 2 * ax, size=C.SZ_SMALL, italic=True)

    # ---- equation, bottom, centred under the clouds ----
    p = render_math(r"d^2=(\mu(D')-\mu(D))^{\top}\,\Sigma^{-1}\,(\mu(D')-\mu(D))", "eq_m1_d2")
    ew = Inches(6.4)
    H.add_eq(s, p, C.MX + (C.CW - ew) // 2 + Inches(0.6), Inches(5.55), w=ew)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: The secret-swap test. Take a dataset D of N images; build D' by replacing ONE image (the target) with another. Fine-tune a LoRA on each K=50 times with different training seeds (headline K; K=20 for scouting). Each trained adapter is one point in flattened dW = BA space. D gives a cloud with mean mu(D) and seed-covariance Sigma; D' gives another. The whitened sensitivity d^2 = (mu(D') - mu(D))^T Sigma^-1 (mu(D') - mu(D)) is the swap's mean effect measured in units of the training noise.
WHY THIS FUNCTION: This is the finite-difference version of the Jacobian question: J_SNR is the derivative of the same map, d^2 is the same object for a discrete data change (q_eff is literally the thresholded spectrum of this object, so the two programs are one). The noise covariance Sigma_seed IS the natural (Fisher) metric of the adapter space. Everything is computed in the dW = BA gauge, never on the raw A, B factors (BA = (BR)(R^-1 A): the factors are gauge-dependent, the product is not).
WHY REPRESENTATIVE: The same estimator runs on every arm of the battery (B dilution, C imbalance, D context, E duplication, S1 similarity, ViT LoRA, H gate); arms differ only in which image is swapped and what the surrounding dataset is.
GAL-ASK: This is the centrepiece: a leakage CEILING that does not depend on any attack, so we can test dataset-composition hypotheses without first building a decoder.
CAVEATS: The plug-in d-hat^2 is biased upward (Hotelling-T^2, ~dim/(K-dim)), so the reported statistic is NEVER the raw value: report against a sign-flip permutation null on the K paired diffs (v_j -> s_j v_j, s_j in {+1,-1}), which carries the identical estimator bias (same K, p, folds, shrinkage; only the labels flipped). sensitivity = d-hat^2_obs - mean(d-hat^2_null); p = fraction of null draws >= observed (floor 1/501 at 500 permutations). p is the PRIMARY readout; the magnitude is a lower bound at a stated K (next two slides).
PROVENANCE: experiments/dataset_sensitivity/whitened_metric.py:1-58 (protocol docstring); notes/whitened_sensitivity_metric.md:17-25 (definition), 31-42 (permutation null); notes/dataset_sensitivity_program_plan.md:101-137 (reporting rules).""")
    return s


# --------------------------------------------------------------------------------------
def slide_m2(prs):
    s = H.new_slide(prs)
    H.add_title(s, "Why this function: the best any attacker can do")
    H.add_lead(s, "equal-covariance Gaussian: Neyman–Pearson optimal, Fisher information — bounds adapter space, not pixels")

    # ---- left: two-panel cartoon, raw vs whitened ----
    pw, ph = Inches(3.1), Inches(3.9)
    py = Y0 + Inches(0.05)
    px1 = C.MX
    px2 = px1 + pw + Inches(0.25)
    for px, title in ((px1, "raw shift  |Δμ|"), (px2, "whitened  d")):
        H.add_rect(s, px, py, pw, ph, fill_color=C.WHITE, line_color=C.LGRAY, line_width=0.75)
        _label(s, title, px, py + Inches(0.06), pw, size=14, bold=True, color=C.BLACK)
    # raw panel: elongated seed ellipses along a noisy axis; the swap shifts along the quiet axis
    ccy = py + Inches(2.05)
    a_long, a_short = Inches(1.25), Inches(0.38)
    shift = Inches(0.42)
    c1 = (px1 + pw // 2, ccy + shift // 2)
    c2 = (px1 + pw // 2, ccy - shift // 2)
    for (cx, cy), col, fill in ((c1, C.RED, C.LIGHTRED_FILL), (c2, C.GREEN_OK, C.LIGHTGREEN_FILL)):
        _oval(s, cx, cy, 2 * a_long, 2 * a_short, fill=fill, line=col, width=1.0, rot=-20.0, dash=True)
    for cx, cy in (c1, c2):
        _dot(s, cx, cy, Inches(0.16), C.BLACK)
    H.add_arrow(s, c1[0], c1[1] - Inches(0.05), c2[0], c2[1] + Inches(0.05), color=C.BLUE, width=2.5)
    _label(s, "a small shift, swamped by\nthe long (noisy) axis", px1 + Inches(0.1), py + ph - Inches(0.72),
           pw - Inches(0.2), Inches(0.6), size=11, italic=True)
    # whitened panel: circles, same shift now large in noise units
    r = Inches(0.55)
    shift_w = Inches(1.35)
    c1w = (px2 + pw // 2, ccy + shift_w // 2)
    c2w = (px2 + pw // 2, ccy - shift_w // 2)
    for (cx, cy), col, fill in ((c1w, C.RED, C.LIGHTRED_FILL), (c2w, C.GREEN_OK, C.LIGHTGREEN_FILL)):
        _oval(s, cx, cy, 2 * r, 2 * r, fill=fill, line=col, width=1.0, dash=True)
    for cx, cy in (c1w, c2w):
        _dot(s, cx, cy, Inches(0.16), C.BLACK)
    H.add_arrow(s, c1w[0], c1w[1] - Inches(0.1), c2w[0], c2w[1] + Inches(0.1), color=C.BLUE, width=2.5)
    _label(s, "every direction measured\nin its own noise units", px2 + Inches(0.1), py + ph - Inches(0.72),
           pw - Inches(0.2), Inches(0.6), size=11, italic=True)

    # ---- right: the equivalences ----
    rx = px2 + pw + Inches(0.45)
    rw = C.SL_W - C.MX - rx
    p = render_math(r"d^2=\mathrm{SNR}^2_{\mathrm{NP}}=2\,\mathrm{KL}(P_{D'}\,\|\,P_D)", "eq_m2_equivalences")
    H.add_eq(s, p, rx, py + Inches(0.1), w=rw - Inches(0.2))
    H.add_text(s, "⇒  a ceiling on detecting the change — for every attack, reconstruction included", rx, py + Inches(1.2), rw, Inches(0.45),
               size=16, bold=True, font=C.HEAD, color=C.BLUE, anchor=MSO_ANCHOR.MIDDLE)
    H.add_text(s, "one object, four readings:", rx, py + Inches(1.85), rw, Inches(0.3), size=C.SZ_SMALL,
               italic=True, color=C.GRAY)
    H.add_text(s, "• optimal-detector (Neyman–Pearson) SNR²\n"
                  "• KL divergence between the with/without-change adapter laws\n"
                  "• Fisher information → Cramér–Rao on the adapter-space change\n"
                  "• the f-INE / DP hypothesis-test statistic",
               rx, py + Inches(2.15), rw, Inches(1.5), size=13, color=C.BLACK, line_spacing=1.15)
    # the caveat chip
    cy_ = Inches(5.75)
    H.add_rect(s, rx, cy_, rw, Inches(0.72), fill_color=C.LIGHTGRAY_FILL, line_color=C.AMBER, line_width=1.0,
               shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    H.add_text(s, "bounds the adapter-space change Δμ, not pixel error:\nimages need Δμ pushed through the data→adapter Jacobian",
               rx + Inches(0.12), cy_, rw - Inches(0.24), Inches(0.72), size=C.SZ_SMALL, color=C.AMBER,
               anchor=MSO_ANCHOR.MIDDLE)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: Chose the whitened (Mahalanobis) sensitivity d^2 as the ONE metric of the program and justified it by what it is equal to. Under the Gaussian, equal-covariance approximation the same number is simultaneously (1) the optimal-detector (Neyman-Pearson) SNR^2, the best ANY attacker can do at telling D from D'; (2) 2*KL(P_D' || P_D) between the with/without-change adapter distributions; (3) the Fisher information the adapter carries about the change, so by Cramer-Rao it lower-bounds the estimation error of delta-mu (the ADAPTER-space change) for unbiased estimators; (4) the f-INE / DP hypothesis-test statistic (is the change separable from the training-randomness null). And it subsumes q_eff: q_eff is the thresholded spectrum of this object, so the discrete-dataset case is continuous with all the Jacobian work.
WHY THIS FUNCTION: The cartoon is why the raw norm ||delta-mu|| is the wrong ruler: the seed cloud is very anisotropic, so a swap that moves the adapter a small Euclidean distance along a QUIET direction is highly detectable, while a large move along the noisy long axis is invisible. Whitening measures every direction in its own noise units. Corollary: the whitening subspace must be SIGNAL-defined (top-p directions of the paired diffs), never Sigma's own top directions (those are exactly the noisy ones we should discount).
WHY REPRESENTATIVE: d^2 is an UPPER BOUND / NECESSARY CONDITION on every downstream attack, reconstruction included: if d^2 ~ 0 (the change is indistinguishable from reseeding) no attack can recover the change; if d^2 is large, recovery is PERMITTED, not achieved. That is exactly the posture we want for composition hypotheses: test the ceiling first, build decoders second.
GAL-ASK: This is the "ceiling, not the attack" framing you asked me to keep honest in May. The all-attacker ceiling is the mutual-information / Fano rate-distortion bound; a prior-equipped (biased) attacker, e.g. the Direction-3 diffusion/SDS prior, can beat the Cramer-Rao floor but not Fano.
CAVEATS: (1) The four equivalences hold ONLY under equal-Sigma Gaussian. If Sigma(D) != Sigma(D') the optimal detector is quadratic (QDA) and pure Mahalanobis drops the covariance term: test Sigma-invariance first (eff_rank/spectrum of Sigma(D) vs Sigma(D')); if they differ, use symmetrized (Jeffreys) KL with the pooled Sigma. (2) The seed noise here is non-Gaussian (heavy tails, skew -2.3 / excess kurtosis 36 at N=4; NaN draws) so the skew/kurtosis diagnostic is a REQUIRED gate; if it fails, d^2 is a heuristic SNR, not "the optimal detector". The permutation null is robust to BOTH failures, which is why it is the primary readout. (3) d^2 bounds adapter-space error, NOT pixel MSE directly (the ssim-overclaim trap): recovering the image needs delta-mu pushed through the nonlinear data->adapter Jacobian; the composed Fisher bridge F_img = J^T Sigma^-1 J is scheduled, not built.
PROVENANCE: notes/whitened_sensitivity_metric.md:49-70 (four equivalences + caveat S1/S2); notes/dataset_sensitivity_program_plan.md:19-36 (why d^2 is the ceiling; CRB vs Fano); Gaussianity numbers results/arm_b_dilution/arm_b_summary.json (gaussianity_skew, gaussianity_exkurt).""")
    return s


# --------------------------------------------------------------------------------------
def slide_m3(prs):
    s = H.new_slide(prs)
    H.add_title(s, "Making the estimator honest")
    H.add_lead(s, "the two-way estimator cheated itself (winner’s curse) — the retracted “sharpens with N” was that artifact")

    fw = Inches(7.95)
    fx, fy, fw, fh = H.fit_image(s, C.fig("estimator_honest.png"), C.MX, Y0 - Inches(0.05), fw, Inches(3.75))
    # captions under the two panels
    cap_y = fy + fh + Inches(0.12)
    half = fw // 2
    _label(s, "left: drift of E[d²] when K doubles — the old split inflates, the new one converges",
           fx, cap_y, half - Inches(0.1), Inches(0.5), size=11, italic=True, align=PP_ALIGN.LEFT)
    _label(s, "right: nothing swapped reads zero; one swapped image reads loud at every N",
           fx + half + Inches(0.1), cap_y, half - Inches(0.1), Inches(0.5), size=11, italic=True, align=PP_ALIGN.LEFT)

    # right column: the two fixes
    rx = C.MX + fw + Inches(0.4)
    rw = C.SL_W - C.MX - rx
    p = render_lines([r"\hat d^{\,2}_{3\mathrm{-way}}:\ \ \mathcal{A}\to U,\ \ \mathcal{B}\to\Delta\mu\cdot U,\ \ \mathcal{C}\to\lambda",
                      r"\mathrm{sens}=\hat d^{\,2}_{\mathrm{obs}}-\overline{\hat d^{\,2}_{\mathrm{null}}}"],
                     "eq_m3_threeway_null")
    H.add_eq(s, p, rx, Y0, w=rw)
    y = Y0 + Inches(1.25)
    fixes = [
        ("three disjoint seed folds, rotated", "subspace U, numerator Δμ·U and noise floor λ never share a seed", C.GREEN_OK, C.LIGHTGREEN_FILL),
        ("sign-flip null on the K paired diffs", "carries the identical bias; p = fraction of null ≥ observed", C.BLUE, C.LIGHTBLUE_FILL),
        ("K-convergence gate", "the statistic must hold from K to 2K, or it is an artifact", C.AMBER, C.LIGHTGRAY_FILL),
    ]
    rh, rg = Inches(0.98), Inches(0.14)
    for name, body, col, fill in fixes:
        H.add_rect(s, rx, y, rw, rh, fill_color=fill, line_color=col, line_width=1.0,
                   shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        H.add_text(s, name, rx + Inches(0.14), y + Inches(0.08), rw - Inches(0.28), Inches(0.3), size=14,
                   bold=True, font=C.HEAD, color=col)
        H.add_text(s, body, rx + Inches(0.14), y + Inches(0.4), rw - Inches(0.28), rh - Inches(0.44),
                   size=C.SZ_SMALL, color=C.BLACK)
        y += rh + rg
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: Found and fixed a self-deception in the estimator, then checked the fixed one reads zero on signal-free data at every K (consistent with unbiased). The first cross-fit was 2-WAY: the subspace U and the noise floor lambda came from one seed fold, only the numerator delta-mu from a disjoint fold. But lambda was then measured along a subspace its OWN samples helped define: selection-biased small (winner's curse), so d-hat^2 INFLATED with K instead of converging. Arm-B post-mortem: d^2(N=64) went 63 -> 161 as K went 50 -> 100; that K-growth was the whole "sharpens with N" story, now retracted. Fix: a 3-WAY disjoint split, rotated over all ordered fold-triples (double-ML style): fold A -> U (top-p right singular vectors of fold-A paired diffs), fold B -> numerator (delta-mu_B . u_i)^2, fold C -> denominator lambda_i (variance of fold-C reseed adapters along u_i).
WHY THIS FUNCTION: Left panel: over 60 synthetic datasets, the population-mean drift of E[d^2] from K to 2K is +44% for the 2-way estimator vs +6.3% for the 3-way (gate <= 15%); self-test 4/4 PASS (orthogonal -> detect, aligned -> mask, null -> flat, K-convergence). Right panel (null-diag, job 212413): the 3-way statistic on NO-SIGNAL data (reseed-vs-reseed, K=200) reads -0.001 at N=4 and +0.003 at N=16 with p = 0.58 / 0.42 and q_eff = 0, i.e. zero reads zero at every K; a real one-image swap at the same recipe reads 22 / 24 / 24 / 13 at N = 4 / 8 / 16 / 32 with p = 0.002 (arm B reconfirm, job 130198, K=50).
WHY REPRESENTATIVE: A third calibration on real data: the S1 similarity ladder's d=0 identity rung (swap the target for itself) reads sens = 0, p = 1.000 exactly (job 268959). Shrinkage rho for the whitening is chosen by CV against the null (the rho maximizing d-hat^2_obs - mean d-hat^2_null); small-eigenvalue floor mandatory.
GAL-ASK: This is the rigor you asked for in May (G2, "not best runs"): every reported number now comes with a null, a p-value and a K-convergence check, and the retractions are listed openly (appendix A4).
CAVEATS: The 3-way fix lowered the absolute scale but real-data d^2 still grows with K (arm B: 8 -> 22 as K 50 -> 100, denominator flat). Because the null stays at ~0 and flat, that growth is benign signal-direction RESOLUTION, so d^2 is a consistent LOWER BOUND that tightens with K: quote magnitudes only at a fixed K, compare only at fixed K and fixed p; the detection p-value is the headline. Single-draw K-flatness is a knife-edge, hence the gate is a population mean over datasets, not one run.
PROVENANCE: notes/whitened_sensitivity_metric.md:150-176 (correction + 3-way update, 2026-08-27); experiments/dataset_sensitivity/whitened_metric.py (self-test); null-diag job 212413 -> results/arm_b_dilution/null_diag.json; arm B reconfirm job 130198 -> results/arm_b_dilution/arm_b_summary.json (k2k); STATUS.md:2478-2509; identity rung job 268959 (STATUS.md 2026-08-28). Figure: figures/deck_2026_08_31/estimator_honest.png.""")
    return s


# --------------------------------------------------------------------------------------
_ROWS = [
    # arm,                 question,                              pre-registered prediction,   mark, outcome
    ("null-diag",          "is the estimator unbiased?",          "reads zero on no-signal data",   "✓", "reads zero at every K"),
    ("B — dilution",  "does N dilute one image?",            "yes, shrinks with N",               "✗", "flat in N"),
    ("E — duplication", "do copies add linearly?",           "linear in the copy count",          "✗", "sub-linear, β ≈ 0.2"),
    ("C — class imbalance", "does rarity matter?",           "the rare class leaks more",         "⚠", "class-dependent; identity asymmetry is intrinsic"),
    ("D — context rarity", "does the surrounding context matter?", "a rare context leaks more", "✗", "≈ no effect"),
    ("g₀ — who leaks", "is it predictable from the base model?", "base gradient predicts",  "✓", "strong, then saturates"),
    ("S1 — similarity", "is a near-duplicate swap visible?", "near the floor",                    "✓", "concept, not instance"),
    ("ViT LoRA",           "an MLP artifact?",                    "generalizes to a real ViT",         "✓", "single image detected"),
    ("H — gate",      "does detection track memorisation?",  "ρ > 0.4",                       "✓", "tracks it"),
]
_MARK_COLOR = {"✓": C.GREEN_OK, "✗": C.RED, "⚠": C.AMBER}


def slide_m4(prs):
    s = H.new_slide(prs)
    H.add_title(s, "The plan: what each arm was built to test")
    H.add_lead(s, "pre-registered predictions, then the data")

    cols = [("arm", Inches(2.3)), ("question", Inches(3.6)), ("pre-registered prediction", Inches(3.05)),
            ("outcome", C.CW - Inches(2.3) - Inches(3.6) - Inches(3.05))]
    hh, rh = Inches(0.38), Inches(0.455)
    x0, y = C.MX, Y0 - Inches(0.05)
    pad = Inches(0.1)
    # header
    x = x0
    for name, w in cols:
        H.add_rect(s, x, y, w, hh, fill_color=C.BLUE, line_color=C.WHITE, line_width=0.5)
        H.add_text(s, name, x + pad, y, w - 2 * pad, hh, size=C.SZ_SMALL, bold=True, color=C.WHITE,
                   anchor=MSO_ANCHOR.MIDDLE)
        x += w
    y += hh
    for i, (arm, q, pred, mark, out) in enumerate(_ROWS):
        fill = C.LIGHTGRAY_FILL if i % 2 else C.WHITE
        x = x0
        cells = [arm, q, pred, None]
        for (name, w), txt in zip(cols, cells):
            H.add_rect(s, x, y, w, rh, fill_color=fill, line_color=C.LGRAY, line_width=0.5)
            if txt is not None:
                H.add_text(s, txt, x + pad, y, w - 2 * pad, rh, size=C.SZ_SMALL, bold=(txt is arm),
                           color=C.BLACK, anchor=MSO_ANCHOR.MIDDLE)
            else:
                H.add_runs(s, [{"text": mark + "  ", "bold": True, "color": _MARK_COLOR[mark], "size": 14},
                               {"text": out, "size": C.SZ_SMALL}],
                           x + pad, y, w - 2 * pad, rh, anchor=MSO_ANCHOR.MIDDLE)
            x += w
        y += rh
    H.add_text(s, "reporting rules: detection p is primary · magnitude = lower bound at fixed K · "
                  "no word “leakage” before the H gate",
               x0, y + Inches(0.1), C.CW, Inches(0.3), size=11, italic=True, color=C.GRAY)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: Wrote the predictions down BEFORE running the battery (dataset_sensitivity_program_plan.md section III), one arm per composition hypothesis, all on the same estimator and the same recipe (MNIST 2-layer MLP + rank-8 LoRA, T=1000, lr=0.5, K=50 seeds per side, dW=BA gauge), then filled the outcome column from the landed jobs.
WHY THIS FUNCTION: Reading the rows. null-diag (job 212413): unbiased on no-signal data at K=50/100/200 (-0.001 / +0.003, p 0.58 / 0.42). B dilution (job 130198, K=50 and 100): predicted 1/N shrinkage; found detection FLAT in N (p=0.002 at every N, N-shape identical at K=50 and 100; the N=32 decline is unexplained). E duplication (jobs 162114, 217123 T-sweep, 246873 Fashion): predicted linear; found sub-linear beta ~ 0.234 (r=8) / 0.241 (r=32), R^2 = 0.85, and beta(T) decreasing 0.313 -> 0.256 -> 0.234 at T = 50 / 200 / 1000 (trends toward, does not reach, the max-margin duplication-invariant limit: the system is NOT at the KKT fixed point). C class imbalance (jobs 229722, 237301 role-swap): two separable effects, a ~3.3x INTRINSIC class-identity asymmetry that survives balance and inverts under role-swap (3.28 -> 0.34), and a rarity effect that is class-dependent (~3x for the loud class, absent for the quiet); all p=0.002. D context rarity (job 245964): fixed-image rarity WEAK (mean gain 1.11, ~noise): "some images leak more" is about the IMAGE, not the context. g0 (jobs 260171 n=12, 272504 n=24 margin-at-scale): rho(sens, g0) = +0.857 at n=12 (beats the max-margin dual lambda, +0.538); at n=24 +0.777 (p=1e-4), strong in the high-g0 range, saturating / indeterminate below. S1 similarity (job 268959): sensitivity rises monotonically with swap distance; near-duplicate swap near-null (0.03-0.07) vs cross-digit 8-24; d=0 rung reads 0 / p=1.000. ViT LoRA (jobs 247474 MVP, 256540 scaled): single image detectable in vit_tiny_patch16_224 rank-4 LoRA (blocks 0-2 qkv), N=16, K=50, all 3 targets p=0.002. H gate (job 272309): rho(LOO memorisation, sens) = +0.881, perm-p = 0.0001, n=12 (robust +0.850 excluding LOO-degenerate cells); bonus rho(memorisation, g0) = +0.798.
WHY REPRESENTATIVE: The table IS the plan; Part 4 shows the figures behind each row. Marks: check = prediction held, cross = prediction falsified, warning = mixed / needs the role-swap to read.
GAL-ASK: Which falsified rows do you find most surprising? My reading: composition knobs barely matter, the image itself (its base gradient) is what leaks, and the adapter records the concept rather than the instance.
CAVEATS: Reporting rules (section II, verbatim on every number): (1) PRIMARY = detection p-value (sign-flip permutation, floor-free, K-stable). (2) q_eff is a thresholded coarsening of the d^2 spectrum, not independent corroboration; the only genuine cross-metric check is sensitivity vs the H-gate memorisation score. (3) Magnitude = LOWER BOUND at a stated K; comparisons only at fixed K and fixed p; cross-rank magnitudes are dimension-confounded (r=32 ~ 15x r=8 absolute). (4) K >= 50 headline, {K, 2K} adequacy gate, shrinkage floor mandatory, NaN re-runs dropped with counts, Sigma-invariance + Gaussianity gates reported. (5) Everything in dW=BA gauge. (6) No "privacy leakage" framing for any magnitude until the H gate closes at scale; the n=12 spot-check de-risks it but the full gate at scale is still required. Fashion arm B (job 246872) and arm F cross-dataset are not in the table (running / reframed).
PROVENANCE: notes/dataset_sensitivity_program_plan.md:113-130 (rules), 172-186 (results table); STATUS.md 2026-08-27/28 entries (:2478-2630). Jobs per row: 212413 · 130198 · 162114 / 217123 / 246873 · 229722 / 237301 · 245964 · 260171 / 272504 · 268959 · 247474 / 256540 · 272309.""")
    return s


SLIDES = [slide_m1, slide_m2, slide_m3, slide_m4]
