"""Part 4 — Experiments, meaning, results (slides 15–21 of the 2026-08-31 deck).

R1 composition knobs · R2 class identity · R3 g₀ predictor · R4 concept-not-instance · R5 H gate ·
R6 beyond the MLP · R7 the atlas.  Story-slide rules (SLIDE_CONTRACT.md): title + one lead line + figure +
one rendered equation + ≤ 2 numeric tokens; everything else lives in the speaker notes.
"""
import os
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from . import config as C
from . import helpers as H
from .eq_render import render_math

_Y0 = C.CT + Inches(0.55)            # top of the content zone
_ZH = C.CB - _Y0                     # content-zone height (5.0")


def _fig_or_placeholder(s, path, x, y, w, h, label):
    """fit_image if the prebuilt figure exists, else a gray placeholder so the module still builds."""
    if os.path.exists(path):
        return H.fit_image(s, path, x, y, w, h)
    H.add_rect(s, x, y, w, h, fill_color=C.LIGHTGRAY_FILL, line_color=C.LGRAY, line_width=0.75)
    H.add_text(s, label, x, y, w, h, size=C.SZ_BODY, color=C.GRAY, align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE, italic=True)
    return x, y, w, h


def _side_label(s, text, x, y, w, *, color=C.GRAY, size=C.SZ_SMALL, h=Inches(0.6), bold=False):
    return H.add_text(s, text, x, y, w, h, size=size, color=color, bold=bold)


# --------------------------------------------------------------------------------------------------
def slide_r1_knobs(prs):
    s = H.new_slide(prs)
    H.add_title(s, "Composition knobs barely matter")
    H.add_lead(s, "the dataset around an image does not set how much it leaks")
    fig_w = Inches(11.4)
    px, py, pw, ph = H.fit_image(s, C.fig("battery_knobs.png"), C.MX + (C.CW - fig_w) // 2, _Y0, fig_w, Inches(4.15))
    eq = render_math(r"d^{2}\propto k^{\beta},\qquad \beta\approx 0.23\ll 1", "eq_r1_beta")
    H.add_eq(s, eq, C.MX + Inches(0.5), py + ph + Inches(0.12), h=Inches(0.55))
    _side_label(s, "duplication is sub-linear;  dilution is flat in N;  context rarity is ~nothing",
                C.MX + Inches(5.9), py + ph + Inches(0.25), Inches(6.3), size=C.SZ_SMALL)
    H.add_footer(s)
    H.add_text(s, "every leakage number here is a lower bound on the weakest attacker (prior-free, adapter-only, per-image)", C.MX, C.SL_H - Inches(0.62), C.CW, Inches(0.25), size=10, color=C.LGRAY, italic=True)
    H.set_notes(s, """WHAT WE DID: three composition knobs around ONE private image, each read with the 3-way whitened secret-swap ruler at fixed K=50 (T=1000, rank 8, N=16 unless swept). B — dilution: same swap inside N=4/8/16/32. E — duplication: k=1,2,4,8 copies of the image at fixed prevalence, Sigma frozen across k. D — context rarity: the SAME fixed image with m=1,2,4,8 same-class companions.
WHY THIS FUNCTION: d^2 is the whitened detectability of the swap (= 2 KL = Neyman-Pearson SNR^2 under equal-Sigma Gaussian). Reporting rule 1: detection p is primary; magnitudes are lower bounds at the stated K; comparisons only at fixed K.
WHY REPRESENTATIVE: all p = 0.002 (every swap detectable at every N, every k, every m). Panel B: flat in N, the N=32 decline is UNEXPLAINED (open sub-question, plan section III.1). Panel E: beta = 0.234 (r8, R^2 = 0.76) / 0.241 (r32) — rank-INVARIANT exponent, but the absolute d^2 across ranks is dimension-confounded (r32 ~15x r8 at k=1; rule 3: never quote a bare cross-rank magnitude). beta(T) DECREASES 0.313 -> 0.256 -> 0.234 at T=50/200/1000: the system trends toward, but does not reach, the beta=0 max-margin duplication-invariant limit, i.e. it is NOT at the KKT fixed point — beta(T) is a convergence diagnostic. Wording rule: 'sub-linear beta', never 'duplication-invariance'. Fashion replicates (beta 0.288 r8 / 0.359 r32, R^2 0.99; job 246873). Panel D: rarity gain sens(m=1)/sens(m=8) = 1.21, 0.96, 1.16 -> mean 1.11, non-monotone, ~noise.
GAL-ASK: G2 (more data / distributions, not best runs) — this is a battery, not a best run; the 'sharpens with N' headline was killed by the 3-way estimator (winner's-curse denominator artifact, retracted).
CAVEATS: MNIST MLP (Fashion for E only); K=50; magnitudes are lower bounds; the N=32 decline is open.
PROVENANCE: arm B reconfirm job 130198 (K=50/100); arm E job 162114 (r8/r32), T-sweep 217123, Fashion 246873; arm D job 245964; null-diag 212413. STATUS.md 2026-08-27/28; notes/dataset_sensitivity_program_plan.md section III.""")
    return s


# --------------------------------------------------------------------------------------------------
def slide_r2_class(prs):
    s = H.new_slide(prs)
    H.add_title(s, "What matters is the image itself: class identity")
    H.add_lead(s, "odd-digit images leak ~3× more whichever class is rare — intrinsic: it inverts under role-swap")
    box_h = Inches(4.9)
    px, py, pw, ph = H.fit_image(s, C.fig("arm_c.png"), C.MX, _Y0, Inches(7.5), box_h, align="left")
    rx = px + pw + Inches(0.35)
    rw = C.SL_W - C.MX - rx
    # native callout instead of an equation
    H.add_card(s, rx, _Y0 + Inches(0.3), rw, Inches(2.0), "role-swap: ratio inverts",
               "make the odd class rare, then the even class rare\n"
               "the odd-digit image is the loud one both times\n"
               "the asymmetry belongs to the image, not rarity",
               color=C.BLUE, fill=C.LIGHTBLUE_FILL, body_size=13)
    H.add_card(s, rx, _Y0 + Inches(2.6), rw, Inches(1.5), "rarity itself",
               "amplifies only the already-loud class\n(quiet class flat across counts)",
               color=C.GRAY, fill=C.LIGHTGRAY_FILL, body_size=13)
    H.add_footer(s)
    H.add_text(s, "every leakage number here is a lower bound on the weakest attacker (prior-free, adapter-only, per-image)", C.MX, C.SL_H - Inches(0.62), C.CW, Inches(0.25), size=10, color=C.LGRAY, italic=True)
    H.set_notes(s, """WHAT WE DID: arm C — class imbalance. N=16, K=50, T=1000, all memorised. Per-image whitened sensitivity of ONE swapped image from the rare class vs one from the common class, with m = 1, 2, 4, 8 rare-class images (m=8 = balanced control). Then the role-swap control: re-run with the OTHER class rare.
WHY THIS FUNCTION: the balanced m=8 cell is the built-in control that separates class identity from rarity; the role-swap separates class identity from 'which class is labelled minority'.
WHY REPRESENTATIVE: at m=8 (balanced) the class-1/class-0 ratio is 3.3, not ~1 — an INTRINSIC class-identity asymmetry (class 1 = odd digits, class 0 = even). Role-swap: balanced ratio 3.28 (class 1 rare) -> 0.34 (class 0 rare) = 1/3.28 — class-1 sensitivity ~10-11, class-0 ~3.5 regardless of which is labelled minority. Rarity is CLASS-DEPENDENT: class-1 sens vs its own count 1->18.9, 2->19.3, 4->10.3, 8->~11, 14->5.6 (~3x louder when rare); class-0 ~2.5-5 FLAT across counts 1..15 (no rarity effect). The naive 'minority leaks 7x' (m=1 raw ratio 7.1) was confounded by class identity. All p = 0.002.
GAL-ASK: none directly; feeds the mechanism question (why is class 1 loud?) answered on the next slide by g0: class 1 has a smaller base margin (2.95 vs 4.70) and a 2.4x larger base gradient norm (1.50 vs 0.61) — the class theta_0 is less confident about.
CAVEATS: rarity is entangled with 'the swapped image is a larger fraction of its class' — that is the mechanism, not a nuisance. Arm D (context rarity on a FIXED image) confirms the context effect is ~1.1x.
PROVENANCE: arm C job 229722; role-swap job 237301; STATUS.md 2026-08-27 / 2026-08-28. Figure figures/deck_2026_08_31/arm_c.png from results/arm_c_imbalance/arm_c_summary{,_minc0}.json.""")
    return s


# --------------------------------------------------------------------------------------------------
def slide_r3_g0(prs):
    s = H.new_slide(prs)
    H.add_title(s, "…and its base gradient: predictive at low g₀, saturating above")
    H.add_lead(s, "an attacker can rank which images will leak from θ₀ and the candidate image alone  (n=24, indeterminate by pre-registration)")
    px, py, pw, ph = H.fit_image(s, C.fig("g0_scatter.png"), C.MX, _Y0, Inches(7.4), Inches(4.9), align="left")
    rx = px + pw + Inches(0.35)
    rw = C.SL_W - C.MX - rx
    H.add_tag(s, "your ask: direct inversion → KKT?", rx, _Y0 + Inches(0.15), w=Inches(3.4))
    eq = render_math(r"g_{0}(x_i)=\|\nabla_{W_{0}}\mathrm{BCE}(\theta_{0};x_i)\|_{F}", "eq_r3_g0")
    H.add_eq(s, eq, rx, _Y0 + Inches(0.85), w=rw - Inches(0.2))
    _side_label(s, "the work the public model still has to do on the image", rx, _Y0 + Inches(1.75), rw,
                size=C.SZ_SMALL)
    H.add_card(s, rx, _Y0 + Inches(2.5), rw, Inches(1.75), "mechanism",
               "NTK / gradient-recording, not max-margin:\n"
               "g₀ ranks images better than the KKT dual λ;\n"
               "strong where g₀ is small, saturates where it is large",
               color=C.GREEN_OK, fill=C.LIGHTGREEN_FILL, body_size=13)
    H.add_footer(s)
    H.add_text(s, "every leakage number here is a lower bound on the weakest attacker (prior-free, adapter-only, per-image)", C.MX, C.SL_H - Inches(0.62), C.CW, Inches(0.25), size=10, color=C.LGRAY, italic=True)
    H.set_notes(s, """WHAT WE DID: margin / support-vector test. For each target image compute g0 = Frobenius norm of the BCE gradient w.r.t. the FULL layer-0 weight at the PUBLIC base model theta_0 (margin_vs_sensitivity.py:75-91, one backward per image), then correlate with the per-image whitened sensitivity. Compared against the max-margin dual proxy lambda = sigmoid(-margin_T) and the raw base margin.
WHY THIS FUNCTION: read Omega = sum_i g_i x_i^T — image i enters the update with coefficient g_i; its public-model norm is the attacker-side predictor that needs NO adapter access.
WHY REPRESENTATIVE: rho(sens, g0) = +0.857 at n=12 (job 260171, PASS vs the pre-registered +-0.15) and +0.777 at n=24 (job 272504; permutation p = 1e-4; 95% CI [0.53, 0.91]; graded INDETERMINATE by pre-registration). Tercile structure (sorted ascending in g0): +0.88 low-g0, +0.50 mid, -0.12 high-g0 — the predictor is strong where the base model does little work and SATURATES once g0 is large (the sign-flip that kept it below PASS; WHY it saturates is open — one hypothesis: g_i carries the loss residual, which decays as fast-fit images are fit). It beats the max-margin dual lambda: rho = 0.51 at n=24 (0.538 at n=12) => the operative mechanism is NTK / gradient-recording, not the KKT endpoint (present as the trajectory view refining the endpoint view, not defeating it). Typicality control: rho(sens, atypicality) = 0.05; partial rho(sens, g0 | atypicality) = 0.78 — g0 is not a typicality proxy. Explains arm C: class 1 has 2.4x larger mean g0 (1.50 vs 0.61) and smaller base margin (2.95 vs 4.70). Transfers to FULL fine-tuning: rho(full-FT LOO footprint, g0) = +0.83 (n=6, job 695782).
GAL-ASK: G1 (direct inversion) reframed — the adapter is a gradient recording; whether it sits at a KKT point is testable (beta(T) diagnostic, slide R1) and currently says no.
CAVEATS: not strictly lazy — per-module ||Delta W||/||W_0|| ~ 0.23 on the single LoRA target module (summary.json lazy_diagnostic; NOTES ONLY, never on a slide; spearman(g0, g_T) = 0.77 representative context). USPS OOD counterexample (n=2): higher g0 yet leaks less — a whitened predictor <g_i, Sigma^-1 g_i> is the proposed repair; open. n small; CI wide.
PROVENANCE: jobs 260171 (n=12), 272504 (n=24, results/margin_at_scale/summary.json: headline, mechanism_table, typicality_control, lazy_diagnostic), 695782 (full-FT transfer). STATUS.md 2026-08-28 + 2026-08-29 tercile correction.""")
    return s


# --------------------------------------------------------------------------------------------------
def slide_r4_ladder(prs):
    s = H.new_slide(prs)
    H.add_title(s, "The adapter records the concept, not the instance")
    H.add_lead(s, "near-duplicates are invisible to the adapter; a different digit is loud")
    px, py, pw, ph = H.fit_image(s, C.fig("ladder_strip.png"), C.MX, _Y0, C.CW, Inches(3.75))
    eq = render_math(r"s(d)=\frac{\mathrm{sens}(d)}{\mathrm{sens}(d_{\mathrm{cross}})}", "eq_r4_ladder")
    ey = py + ph + Inches(0.15)
    H.add_eq(s, eq, C.MX + Inches(0.6), ey, h=Inches(0.95))
    _side_label(s, "d = pixel distance of the swap;  s = sensitivity, normalised to a cross-digit swap\n"
                   "identity swap reads exactly zero (calibration); sensitivity rises with swap distance (mid-ladder wobble = cross-exemplar noise)",
                C.MX + Inches(4.3), ey + Inches(0.15), Inches(7.8), size=C.SZ_SMALL, h=Inches(0.8))
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: S1 similarity ladder (distance dial). Swap the private image for a graded sequence of alternatives — identity (d=0), tiny noise, small perturbations, blur/brightness, same-digit other exemplars, a different digit — 9 rungs x 2 targets, K seeds each, 3-way whitened sensitivity per rung.
WHY THIS FUNCTION: s(d) normalises each rung to the cross-digit swap so the two targets are comparable; the d=0 rung is a built-in calibration of the estimator (must read 0).
WHY REPRESENTATIVE: identity rung reads exactly 0 (p = 1.000). Near-duplicate swaps are near-null (sens 0.03-0.07) while cross-digit swaps are large (sens 8-24); sensitivity rises monotonically with swap distance. Predictor comparison (pooled n=18): sens ~ d_pixel rho = +0.807 (best) > sens ~ |Delta g0| +0.657 > sens ~ d_encoder(DINO) +0.399 (worst — DINO is out-of-domain on upscaled MNIST, builder-flagged). So the adapter tracks raw pixel distance best here.
GAL-ASK: none directly; it is the graded end of the atlas story (slide R7): content-level recovery +0.989 grades down to ~0 for single near-duplicate swaps.
CAVEATS: 2 targets, exploratory; n=18 small; the global-brightness rung is amplified by ds_mean centering — read with care. Privacy statement: recovery is concept-level — an attacker cannot distinguish an image from its close neighbours (weakest-attacker bound).
PROVENANCE: job 268959; results/similarity_ladder/similarity_ladder_summary.json; STATUS.md 2026-08-28 'DISTANCE DIAL'. Depth follow-up (S2): near-dup numerator L0 0.0223 > L1 0.0133 > L2 0.0033 — instance signal concentrates in the first layer.""")
    return s


# --------------------------------------------------------------------------------------------------
def slide_r5_hgate(prs):
    s = H.new_slide(prs)
    H.add_title(s, "Detection tracks behavioural memorisation — so we may say 'leakage'")
    H.add_lead(s, "the instrument and a behavioural leave-one-out score rank the same images  (spot-check n=12; full gate at scale pending)")
    px, py, pw, ph = H.fit_image(s, C.fig("h_gate.png"), C.MX, _Y0, Inches(7.0), Inches(4.9), align="left")
    rx = px + pw + Inches(0.35)
    rw = C.SL_W - C.MX - rx
    H.add_tag(s, "rigor gate", rx, _Y0 + Inches(0.15), w=Inches(1.6), color=C.GREEN_OK)
    eq = render_math(r"\mathrm{mem}_i=\mathbb{E}_{\mathrm{seeds}}[\mathrm{margin}(D)-\mathrm{margin}(D\setminus\{x_i\})]",
                     "eq_r5_mem")
    H.add_eq(s, eq, rx, _Y0 + Inches(0.8), w=rw)
    _side_label(s, "how much the model's margin on the image depends on having trained on it",
                rx, _Y0 + Inches(1.55), rw, size=C.SZ_SMALL)
    H.add_card(s, rx, _Y0 + Inches(2.4), rw, Inches(1.8), "why this is the real check",
               "q_eff is the same object as d² coarsened —\n"
               "the leave-one-out margin is the only quantity\n"
               "here that is independent of the adapter parameters",
               color=C.GREEN_OK, fill=C.LIGHTGREEN_FILL, body_size=13)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: H gate spot-check (plan section III.0, pre-registered). For 12 (image, context) cells (3 targets x m = 1,2,4,8; N=16; rank 8; T=1000; lr 0.5) train with the image and without it (DROP, size N-1) over K_loo = 10 seeds each; mem_i = mean margin gain on the image from having trained on it. Correlate with the whitened adapter sensitivity of the same cells.
WHY THIS FUNCTION: reporting rule 6 — no 'leakage' language before the H gate; rule 2 — q_eff is a thresholded coarsening of d^2, so it is NOT independent corroboration; the behavioural LOO score is the only genuine cross-metric check in the program.
WHY REPRESENTATIVE: rho(mem, sens) = +0.881, permutation p = 1.5e-4, n=12 (pre-registered PASS threshold +0.4); robust excluding the m=1 LOO-degenerate cells: +0.85, p = 0.003, n=9; per-m rho = 1.0, 1.0, 1.0, 0.5, all signs +. Bonus: rho(mem, g0) = +0.798, p = 0.002 — the base-model gradient predicts behavioural memorisation itself, so the chain base gradient -> adapter movement -> memorisation -> detection holds end-to-end at n=12.
GAL-ASK: 'detectability isn't reconstruction' — agreed; this gate licenses the word 'memorisation/leakage' for the detection claims, not a reconstruction claim.
CAVEATS: n=12 small; swap(N) vs drop(N-1) construction mismatch (rank correlation + per-m rho mitigate); the m=1 LOO set has no class-1 member; g0 has only 3 distinct values across these cells (coarse). The FULL H gate at scale (section III.3) is still required before any privacy claim — open.
PROVENANCE: job 272309; results/h_spotcheck/h_spotcheck.json (correlations, verdict PASS); STATUS.md 2026-08-28 'VALIDATION GATE PASSED'.""")
    return s


# --------------------------------------------------------------------------------------------------
def slide_r6_beyond(prs):
    s = H.new_slide(prs)
    H.add_title(s, "Not an MLP artifact: a real ViT LoRA, Fashion, full fine-tuning")
    H.add_lead(s, "full fine-tuning records ~5× more signal per image, at about the same resolution")
    px, py, pw, ph = H.fit_image(s, C.fig("beyond_mlp.png"), C.MX, _Y0, C.CW, Inches(3.6))
    eq = render_math(r"P_{\mathrm{LoRA}}(H)=BB^{\top}H+HA^{\top}A", "eq_r6_plora")
    ey = py + ph + Inches(0.18)
    H.add_eq(s, eq, C.MX + Inches(0.5), ey, h=Inches(0.5))
    _side_label(s, "what one LoRA step exposes of the full gradient H (a PSD operator, not a projection);\n"
                   "full fine-tuning sees H itself — more signal, target-dependent resolution",
                C.MX + Inches(6.4), ey - Inches(0.05), Inches(5.8), size=C.SZ_SMALL, h=Inches(0.8))
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: four generalisation checks of the secret-swap ruler beyond the MNIST MLP. (1) vit_tiny_patch16_224 with a rank-4 LoRA on blocks 0-2 qkv (9,216 params), N=16, K=50, 3 targets. (2) Fashion-MNIST arm B (one image swapped, N=8/16/32). (3) Full fine-tuning (all layers) vs LoRA r=8 on the SAME images: removal (leave-one-out) footprint, n=6 targets. (4) Valley width d*(0.1): how far a swap must go before the weights notice, full vs LoRA.
WHY THIS FUNCTION: P_LoRA(H) = B B^T H + H A^T A is the exact operator a single LoRA step applies to the full gradient (self-adjoint PSD, P^2 != P, not an orthogonal projection; at B=0 init only H A^T A survives — the seed enters through A). Full FT records H itself.
WHY REPRESENTATIVE: ViT: all 3 targets p = 0.002, sensitivities 1.13 / 1.24 / 1.52, fit 2.3e-4 (job 256540; MVP 247474 at N=6/K=10 gave p = 0.03). Fashion: p = 0.002 at N = 8/16/32, sensitivity 0.66-1.66 (job 246872). Removal footprint: cross-regime per-target rank correlation rho = +0.94 (n=6) — the SAME images have the biggest footprint under both; absolute footprint ~5x larger under full FT (target MEDIAN; per-target ~3-6x). g0 transfers to full FT: rho = +0.83 (n=6). Valley width d*(0.1) full/LoRA: geomean 1.02, median 0.86, full narrower on 4/6 targets, 2 flip — TARGET-DEPENDENT, approximately a wash; NEVER quote the arithmetic mean (Jensen-biased for a ratio, inverts the majority). Guards: B1 dimension-invariance PASS; B2 SGD-noise vs epsilon-noise divergent -> read qualitatively.
GAL-ASK: G2 (more data / distributions) and G1 (full vs adapter): the extra information is present under full FT — the open problem is extraction, not missing information (World B).
CAVEATS: ViT at modest scale; full-FT comparison n=6 exploratory (resist quoting the 4/6-vs-2/6 split until scale-up); K=50; magnitude (5x) and resolution (valley) claims kept separate. The prebuilt figure's four panel titles overlap at this width — flagged to the figure builder.
PROVENANCE: ViT jobs 247474 / 256540 (results/vit_lora_sensitivity/*.pth); Fashion job 246872; full-FT valley + removal job 695782 (results/fullft_valley/valley_headline_dstar.json); STATUS.md 2026-08-28/29.""")
    return s


# --------------------------------------------------------------------------------------------------
def slide_r7_atlas(prs):
    s = H.new_slide(prs)
    H.add_title(s, "And the adapter betrays what it was trained on")
    H.add_lead(s, "which digits were present is recoverable from ΔW above the recipe baseline; which exemplar is open")
    px, py, pw, ph = _fig_or_placeholder(s, C.fig("atlas_2panel.png"), C.MX, _Y0, Inches(8.5), Inches(4.9),
                                         "atlas figure (building)")
    rx = C.MX + Inches(8.85)
    rw = C.SL_W - C.MX - rx
    eq = render_math(r"BA=(BR)(R^{-1}A)", "eq_r7_gauge")
    H.add_eq(s, eq, rx, _Y0 + Inches(0.5), w=rw - Inches(0.3))
    _side_label(s, "gauge: the seed lives in (B, A); the data lives in ΔW", rx, _Y0 + Inches(1.35), rw,
                size=C.SZ_SMALL, h=Inches(0.5))
    H.add_card(s, rx, _Y0 + Inches(2.1), rw, Inches(2.5), "what the atlas says",
               "ΔW clusters by composition, blind to init / lr / activation\n"
               "raw (B, A) clusters by seed instead\n"
               "content-level: which digits — not which exemplar\n"
               "+0.989 above the fitted-recipe baseline (cross-fitted; CI excludes 0)",
               color=C.BLUE, fill=C.LIGHTBLUE_FILL, body_size=13)
    H.add_footer(s)
    H.add_text(s, "every leakage number here is a lower bound on the weakest attacker (prior-free, adapter-only, per-image)", C.MX, C.SL_H - Inches(0.62), C.CW, Inches(0.25), size=10, color=C.LGRAY, italic=True)
    H.set_notes(s, """WHAT WE DID: composition atlas — a factorial zoo of 169 converged adapters (of a 180-cell grid; 5 compositions x 3 activations x 2 learning rates x 6 init seeds; 11 non-converged dropped) on a shared base theta_0 (MNIST, N=4, rank 8). Cluster Delta W = BA and the raw factors (B, A) by each factor; test composition recovery from Delta W against a fitted-recipe baseline with a cross-fitted, cluster-robust accuracy difference.
WHY THIS FUNCTION: LoRA has the exact gauge symmetry BA = (BR)(R^-1 A) — only Delta W is gauge-invariant; the raw factors carry the init seed (P_LoRA at B=0 exposes H A^T A, the row space of a random A). The atlas is the empirical gauge-contrast: seed visible in (B, A), scrubbed in Delta W.
WHY REPRESENTATIVE: Delta W clusters by composition with adjusted Rand +1.00 (permutation p < 0.001) and is blind to init / lr / activation (ARI ~ 0); raw (B, A) clusters by seed (ARI +0.55, p < 0.001) while Delta W scrubs it (-0.03). Cross-fitted held-out accuracy difference vs the nuisance-only baseline: +0.989, 95% CI [+0.973, +1.005], G = 30 cluster-robust (CI upper clips > 1: near-ceiling normal-approximation artifact). The 5 compositions are distinct DIGIT SUBSETS (comp0 = {1,6,7,8}, comp3 = {0,1,4,9}, ...), so +0.989 recovers WHICH DIGITS were present = content / concept level, not the specific instance. Graded: +0.989 (content) down to ~0 for single near-duplicate swaps (arms 0.03-0.07, slide R4).
GAL-ASK: 'the composition result is just a t-SNE picture' — no longer: the cross-fitted cluster-robust test excludes 0 after a fold bug was fixed (first pass read +0.00 / CI [0, 0] because i%5 with 5 compositions isolated a whole composition into the test fold; fixed ab9eb99). Cite and differentiate Learning on LoRAs (Putterman / Lim et al., ICLR 2025, arXiv:2410.04207): they probe LoRA weights GL-equivariantly for data attributes; we ask whether composition is FORCED into Delta W above a recipe baseline.
CAVEATS: G = 30 small; population (stronger-than-weakest) attacker; MNIST, N=4, rank 8. The instance-level zoo (same digits, different exemplars) NEVER actually ran — the 'same-digits' bank was byte-identical to the composition bank — so instance-level recovery is OPEN. Ecosystem prototype (subtract population common-mode) was a degenerate honest null on this saturated zoo; parked.
PROVENANCE: build job 808715, analysis job 838868 (811847 fold-buggy); code experiments/dataset_sensitivity/atlas_{zoo,analyze}.py (dw_distance lines 46-64, facet_c 106-134); figure figures/atlas/atlas.png; STATUS.md 2026-08-30.""")
    return s


SLIDES = [slide_r1_knobs, slide_r2_class, slide_r3_g0, slide_r4_ladder, slide_r5_hgate, slide_r6_beyond,
          slide_r7_atlas]
