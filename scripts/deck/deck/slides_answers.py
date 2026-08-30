"""Part 1 — "Your questions from May, answered" (title + 7 story slides).

Each story slide: plain-statement title, ONE lead line, figure(s), ONE rendered equation, <= 2 numeric
tokens visible. Everything else lives in the speaker notes (WHAT WE DID / WHY THIS FUNCTION /
WHY REPRESENTATIVE / GAL-ASK / CAVEATS / PROVENANCE).

Derived raster assets that are not prebuilt (cropped DI grids, cropped faces, the two sigma-prime
sketches for the mechanism slide) are written to a scratch dir at build time (never into the repo).
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from . import config as C
from . import helpers as H
from .eq_render import render_math, render_lines

SCRATCH = os.environ.get(
    "DECK_PART1_SCRATCH",
    "/tmp/claude-50309/-home-projects-galvardi-yoado/80d56b79-39e3-4051-86e8-d83341e3460c/scratchpad/part1",
)
os.makedirs(SCRATCH, exist_ok=True)

TAG_W = Inches(3.1)
TAG_X = C.SL_W - C.MX - TAG_W       # tag sits at the right end of the title row
TAG_Y = C.MY + Inches(0.15)
CONTENT_Y = C.CT + Inches(0.55)     # first content row (below the lead line)
CONTENT_H = C.CB - CONTENT_Y


# ----------------------------------------------------------------------------------------------
# local helpers (module-private; config/helpers/eq_render are untouched)
# ----------------------------------------------------------------------------------------------
def _crop(src, name, box_frac):
    """Crop `src` by fractional box (l, t, r, b) into the scratch dir; cached by mtime."""
    out = os.path.join(SCRATCH, name)
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out
    with Image.open(src) as im:
        w, h = im.size
        l, t, r, b = box_frac
        im.crop((int(l * w), int(t * h), int(r * w), int(b * h))).save(out)
    return out


def _caption(slide, text, x, y, w, *, h=Inches(0.5), size=C.SZ_SMALL, color=C.GRAY, align=PP_ALIGN.LEFT,
             italic=False):
    return H.add_text(slide, text, x, y, w, h, size=size, color=color, align=align, italic=italic)


def _sigma_prime_sketch(kind):
    """Tiny sigma'(z) sketch: 'kinked' = step (ReLU), 'smooth' = sigmoid ramp (softplus)."""
    out = os.path.join(SCRATCH, f"sigma_prime_{kind}.png")
    if os.path.exists(out):
        return out
    z = np.linspace(-4, 4, 400)
    if kind == "kinked":
        y = (z > 0).astype(float)
        col = C.HEX_RED
    else:
        y = 1 / (1 + np.exp(-z))
        col = C.HEX_GREEN
    fig, ax = plt.subplots(figsize=(2.6, 1.5), dpi=220)
    ax.plot(z, y, color=col, lw=3.2, solid_capstyle="round")
    ax.axhline(0, color="#BBBBBB", lw=0.8)
    ax.axvline(0, color="#BBBBBB", lw=0.8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.12, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-3.9, 1.05, "σ′(z)", ha="left", va="top", fontsize=14, color="#333333")
    ax.text(3.9, 0.02, "z", ha="right", va="bottom", fontsize=13, color="#333333")
    fig.tight_layout(pad=0.15)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def _story(prs, title, lead, tag=None, *, tag_color=C.BLUE):
    s = H.new_slide(prs)
    H.add_title(s, title)
    H.add_lead(s, lead)
    if tag:
        H.add_tag(s, tag, TAG_X, TAG_Y, w=TAG_W, color=tag_color)
    return s


# ----------------------------------------------------------------------------------------------
# 1. title
# ----------------------------------------------------------------------------------------------
def slide_title(prs):
    s = H.new_slide(prs)
    H.add_text(s, "Where the weights remember", C.MX, Inches(2.3), C.CW, Inches(1.0), size=40, bold=True,
               font=C.HEAD, color=C.BLACK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    H.add_line(s, Inches(4.9), Inches(3.42), C.SL_W - Inches(4.9), Inches(3.42), color=C.BLUE, width=1.5)
    H.add_text(s, "LoRA adapters, private images, and what we can measure", C.MX, Inches(3.6), C.CW,
               Inches(0.6), size=22, font=C.HEAD, color=C.GRAY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    H.add_text(s, "Yoad Oxman  ·  Gal Vardi  ·  2026-08-31", C.MX, Inches(4.7), C.CW, Inches(0.5), size=16,
               color=C.GRAY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    H.add_text(s, "we observe, we do not conclude — every leakage number is a lower bound on the weakest attacker "
               "(prior-free, adapter-only, per-image)", C.MX, Inches(5.6), C.CW, Inches(0.5), size=13,
               color=C.LGRAY, italic=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    H.set_notes(s, """WHAT WE DID: sit-down deck, four parts. Part 1 answers the six asks from the 2026-05-14 meeting with
the runs that landed Jul 21-26 (direct inversion, anchor sweep, smooth activations) and Aug 23-29 (free-c ladder,
feature-stability-vs-T, dissociation). Parts 2-4: the theory that led to the secret-swap test, how we measure, results.
POSTURE: observe, don't conclude. Every leakage number bounds the WEAKEST attacker (prior-free, adapter-only,
per-image) from below; it never bounds what a stronger attacker could get.
GAL-ASK: all six (G1-G6) are indexed on the next slide.
PROVENANCE: notes/thesis_update_briefing.md §3-4 (the asks); notes/thesis_note_v2.md (the state).""")
    return s


# ----------------------------------------------------------------------------------------------
# 2. what changed since May — native-shape table
# ----------------------------------------------------------------------------------------------
_ROWS = [
    ("G1", "direct weight inversion", "the map inverts at small N; joint inversion superposes as N grows", "slide 7"),
    ("G2", "more data / distributions", "full-gradient ceiling recognizable on every dataset tried; adapter-only pixels open",
     "slide 8"),
    ("G3", "genuinely smooth activations  (your top ask)", "the opposite: kinked leaks most, at matched weight change",
     "slides 3–4"),
    ("G4", "anchor at the midpoint", "fixes linearity, not leakage", "slide 5"),
    ("G5", "one experiment + a mechanism", "the activation enters only through σ′: gate rank vs gate drift", "slide 6"),
    ("G6", "L-BFGS", "used inside the NTK least-squares; same plateau as SGD on LoRA — not the bottleneck", "notes"),
]


def slide_changed(prs):
    s = _story(prs, "What changed since May", "six asks, six answers — and one instrument you did not ask for")
    x0, y0 = C.MX, CONTENT_Y
    w1, w2, w3 = Inches(4.15), Inches(6.35), Inches(1.73)
    hdr_h, row_h = Inches(0.4), Inches(0.56)
    # header
    H.add_rect(s, x0, y0, w1 + w2 + w3, hdr_h, fill_color=C.LIGHTGRAY_FILL, line_color=None)
    for x, w, t in ((x0, w1, "you asked"), (x0 + w1, w2, "what we found"), (x0 + w1 + w2, w3, "where")):
        H.add_text(s, t, x + Inches(0.12), y0, w - Inches(0.2), hdr_h, size=13, bold=True, color=C.GRAY,
                   anchor=MSO_ANCHOR.MIDDLE)
    y = y0 + hdr_h
    for code, ask, found, where in _ROWS:
        H.add_line(s, x0, y, x0 + w1 + w2 + w3, y, color=C.LGRAY, width=0.5)
        H.add_runs(s, [{"text": code + "  ", "bold": True, "color": C.BLUE, "size": 14},
                       {"text": ask, "size": 14}],
                   x0 + Inches(0.12), y, w1 - Inches(0.2), row_h, anchor=MSO_ANCHOR.MIDDLE)
        H.add_text(s, found, x0 + w1 + Inches(0.12), y, w2 - Inches(0.2), row_h, size=14,
                   anchor=MSO_ANCHOR.MIDDLE)
        H.add_text(s, where, x0 + w1 + w2 + Inches(0.12), y, w3 - Inches(0.2), row_h, size=13, color=C.GRAY,
                   anchor=MSO_ANCHOR.MIDDLE)
        y += row_h
    # accent row
    H.add_rect(s, x0, y, w1 + w2 + w3, row_h + Inches(0.08), fill_color=C.LIGHTBLUE_FILL, line_color=C.BLUE,
               line_width=1.0)
    H.add_text(s, "+  an instrument you did not ask for", x0 + Inches(0.12), y, w1 - Inches(0.2),
               row_h + Inches(0.08), size=14, bold=True, color=C.BLUE, anchor=MSO_ANCHOR.MIDDLE)
    H.add_text(s, "a ruler for what the adapter records — and what it cannot", x0 + w1 + Inches(0.12), y,
               w2 - Inches(0.2), row_h + Inches(0.08), size=14, bold=True, color=C.BLUE, anchor=MSO_ANCHOR.MIDDLE)
    H.add_text(s, "Parts 2–4", x0 + w1 + w2 + Inches(0.12), y, w3 - Inches(0.2), row_h + Inches(0.08), size=13,
               bold=True, color=C.BLUE, anchor=MSO_ANCHOR.MIDDLE)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: index of the May asks (G1-G6, notes/thesis_update_briefing.md §3-4) against what landed.
G1 direct weight inversion: theta_T = F(theta_0, x_hat) with autograd through an unrolled SGD; recovers digits at
N=4 (ssim_norm ~0.57, job 500913) and superposes at N=10 (0.27) and N=20 (~0.15, job 887704). SimuDy (Tian et al.
ICLR 2025, sent by Gal 2026-06-29) publishes the same primitive for full FT -> we reframed onto LoRA-only leakage,
identifiability theory, linearized (memory-tractable) inversion (STATUS.md 'SimuDy collision', notes/simudy_decision_brief.md).
G2 more data: full-gradient ceiling recognizable on MNIST / Fashion / CIFAR-10 / Flowers (SSIM up to ~0.99) and
ViT-B/16 faces; the adapter-only pixel inversion is the open extraction milestone (E7 in thesis_note_v2.md).
G3 smooth activations (his top ask; his prediction: smoother -> NTK survives -> leaks more): the free-coefficient
weight-change ladder shows the opposite on MNIST - kinked cluster ~5x the smooth cluster (job 392821).
G4 midpoint anchor: two-curve plot done (job 532232, MNIST; flowers replicate). Linearization error falls with alpha;
leakage on the realistic control-margin metric does not track it (relu flat; softplus best at alpha=0).
G5 mechanism: the activation enters the whole theory only through sigma' (gate matrix M). Informativeness = rank of
the static M (kink wins); linearization fidelity = continuity of dM (smooth wins). Two objects, opposite directions.
G6 L-BFGS: implemented for the NTK least-squares (LESSONS_LEARNED.md 'L-BFGS vs SGD', 2026-02-22). Full model: L-BFGS
stalls in a shallow basin (SSIM ~0.82 vs SGD 0.996). LoRA r=8: both optimizers hit the same plateau (rank-mismatch
residual) -> the optimizer is not the bottleneck. The validated free-c recipe is SGD extraction + near-ReLU surrogate +
consistency + restarts (STATUS.md 'REALISTIC free-coefficient results'); L-BFGS is kept for quick rank sweeps.
THE INSTRUMENT: the whitened secret-swap sensitivity d^2 / q_eff (Parts 2-4) - an attack-independent read of how many
private directions the adapter records above its own seed noise.
CAVEATS: all Part-1 numbers are MNIST toy MLP unless stated; N=2, T=1 or T=10, rank 8; small n. Observe, don't conclude.
PROVENANCE: thesis_note_v2.md provenance table (lines 144-153); STATUS.md lines 49-105 (crux), 1667-1690 (DI, anchor).""")
    return s


# ----------------------------------------------------------------------------------------------
# 3. smooth activations: the opposite of the prediction
# ----------------------------------------------------------------------------------------------
def slide_crux(prs):
    s = _story(prs, "Smooth activations: the opposite of the prediction",
               "kinked ≈ 5× smooth, free coefficients, at matched weight change", "your ask: smooth activations")
    fig_w = Inches(8.3)
    H.fit_image(s, C.fig("crux_bars.png"), C.MX, CONTENT_Y, fig_w, CONTENT_H)
    rx = C.MX + fig_w + Inches(0.3)
    rw = C.SL_W - C.MX - rx
    H.add_text(s, "leakage metric  (control-margin)", rx, CONTENT_Y + Inches(0.55), rw, Inches(0.35), size=13,
               bold=True, color=C.BLUE)
    p = render_math(r"\mathrm{leak}=\mathrm{ssim}_{n}(\hat x,x)-\mathrm{ssim}_{n}(\hat x,x_{\mathrm{ctrl}})",
                    "eq_ctrl_margin")
    H.add_eq(s, p, rx, CONTENT_Y + Inches(0.95), w=rw)
    _caption(s, "similarity of the reconstruction to the private image, minus its similarity to a same-class "
                "control image — cancels shared background and clipping", rx, CONTENT_Y + Inches(1.35), rw,
             h=Inches(1.1))
    _caption(s, "red = kinked, green = smooth; diamonds = oracle coefficients (upper bound)", rx,
             CONTENT_Y + Inches(2.6), rw, h=Inches(0.8), italic=True)
    H.add_footer(s)
    H.add_text(s, "every leakage number here is a lower bound on the weakest attacker (prior-free, adapter-only, per-image)", C.MX, C.SL_H - Inches(0.70), C.CW, Inches(0.24), size=10, color=C.LGRAY, italic=True)
    H.set_notes(s, """WHAT WE DID: MNIST, N=2, T=1, LoRA rank 8, 13 activations, realistic FREE-coefficient attack (Haim-style;
oracle = diamonds, upper bound only). Four matched weight-change rungs {0.005, 0.03, 0.1, 0.3}; the bars show one rung
(0.1). Leakage = ctrl_margin_norm (experiments/recompute_metrics.py:87): ssim_norm(recon, private) minus
ssim_norm(recon, same-class control), mean/std-matched SSIM.
WHY THIS FUNCTION: the control margin is the instance-leakage bar - it cancels the shared background/clipping that makes
absolute SSIM untrustworthy at small N, and it is clip-robust (not raw ctrl_margin, not eff_rank).
WHY REPRESENTATIVE: the two-cluster gap (kinked mean ~0.47 vs smooth mean ~0.09) holds at EVERY rung; Spearman(smoothness,
leakage) is negative at every rung (-0.48 / -0.25 / -0.27 / -0.59) and strengthens with weight change. Oracle tracks free-c
closely, so the ordering is robust to both attack mode and weight change. Do NOT quote a single canonical Spearman: the
exact value depends on the smoothness ORDERING (no unique total order over 13 activations).
GAL-ASK: G3 (top ask). His prediction: smoother -> NTK survives -> leaks more. Observed: the opposite cluster order.
CAVEATS: two-cluster, NOT a monotone smoothness law (selu clusters with the sharp kinks; hardswish, also kinked, does not);
sign flips within the smooth-only set; MNIST only, N=2, n=13 activations - exploratory. Flowers32 free-c band pending
(a flowers oracle->free-c flip precedent exists, so dataset-dependence is OPEN). Does not flip the oracle.
PROVENANCE: job 392821 (52/52) -> results/rescored_freec_ladder_2026-08-29.csv; oracle
results/rescored_activations_857271_full_2026-08-28.csv; STATUS.md lines 49-71, 90-97; thesis_note_v2.md E3.""")
    return s


# ----------------------------------------------------------------------------------------------
# 4. smooth activations do stay linear longer
# ----------------------------------------------------------------------------------------------
def slide_fs(prs):
    s = _story(prs, "Smooth activations do stay linear longer",
               "so the premise held — smoothness buys linearization; it just does not buy leakage",
               "your ask: smooth activations")
    fig_w = Inches(8.2)
    H.fit_image(s, C.fig("fs_vs_T.png"), C.MX, CONTENT_Y, fig_w, CONTENT_H)
    rx = C.MX + fig_w + Inches(0.3)
    rw = C.SL_W - C.MX - rx
    H.add_text(s, "feature stability", rx, CONTENT_Y + Inches(0.55), rw, Inches(0.35), size=13, bold=True,
               color=C.BLUE)
    p = render_math(r"\mathrm{fs}(T)=\cos\left(\nabla_\theta f(\theta_0;x),\,\nabla_\theta f(\theta_T;x)\right)",
                    "eq_feature_stability")
    H.add_eq(s, p, rx, CONTENT_Y + Inches(0.95), w=rw)
    _caption(s, "cosine between a sample's feature gradient at the public model and after T fine-tuning steps "
                "— the NTK-laziness proxy", rx, CONTENT_Y + Inches(1.35), rw, h=Inches(1.0))
    _caption(s, "smooth (green) stays near the linear regime, gelu (blue) is the smooth outlier that drifts; kinked (red, dashed) "
                "leaves it at the first step — yet red is the cluster that leaks", rx, CONTENT_Y + Inches(2.55), rw, h=Inches(1.1), italic=True)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: feature-stability-vs-T, 13 activations, T in {1,2,5,10,20,50}, MNIST N=2 LoRA r=8 (job 390026, 65/65).
Shown: four representative lines (softplus, sigmoid, gelu; relu, leaky-relu).
WHY THIS FUNCTION: fs = cos(grad_theta f(theta_0;x), grad_theta f(theta_T;x)) (experiments/ntk_verification.py
compute_feature_stability) is the direct NTK-survival read: 1 means the feature map did not rotate, i.e. the
linearization at theta_0 still describes the fine-tuned network.
WHY REPRESENTATIVE: at every T the smoothest (sigmoid/softplus) sustain the highest fidelity and the kinked
(relu/leaky-relu) the lowest - robust at the extremes, two-cluster in the middle. fs > 0.99 (the 'linear regime' line)
only briefly, and only for sigmoid/softplus.
GAL-ASK: G3 - his PREMISE (smooth -> NTK survives) is right. His CONCLUSION (-> leaks more) is not: pooled
Spearman(feature stability, control-margin leakage) ~ 0 (-0.06; on ssim_norm +0.08, n=426). Kinked activations have the
WORST linearization fidelity and LEAK THE MOST - the dissociation. Fidelity does not drive leakage.
CAVEATS: weight-change confound at fixed lr (C1 activations take larger steps); the metric-scoped statement is on the
control margin (on ssim the correlation is mildly positive, +0.08 to +0.28 across subsets). Toy MLP, small n.
PROVENANCE: job 390026 -> results/rescored_tsweep_2026-08-29.csv; figures/crux/feature_stability_vs_T.png;
STATUS.md lines 97-102; notes/crux_activation_analysis.md lines 296-304; thesis_note_v2.md E3.""")
    return s


# ----------------------------------------------------------------------------------------------
# 5. moving the anchor fixes linearity, not leakage
# ----------------------------------------------------------------------------------------------
def slide_anchor(prs):
    s = _story(prs, "Moving the anchor fixes linearity, not leakage",
               "softplus: already linear at α=0;  relu: linearization error falls 25× while leakage stays flat",
               "your ask: midpoint anchor")
    fig_w = Inches(8.0)
    H.fit_image(s, C.fig("anchor_two_curve.png"), C.MX, CONTENT_Y, fig_w, CONTENT_H)
    rx = C.MX + fig_w + Inches(0.3)
    rw = C.SL_W - C.MX - rx
    H.add_text(s, "anchor family and function-space linearization error", rx, CONTENT_Y + Inches(0.4), rw,
               Inches(0.6), size=13, bold=True, color=C.BLUE)
    p = render_lines([r"\theta(\alpha)=(1-\alpha)\theta_0+\alpha\,\theta_T",
                      r"L_{\mathrm{lin}}=\frac{\|\Phi(\theta_T)-[\Phi(\theta_a)+\nabla\Phi(\theta_a)\,\delta]\|}"
                      r"{\|\Phi(\theta_T)-\Phi(\theta_a)\|}"], "eq_anchor_family", gap=0.75)
    H.add_eq(s, p, rx, CONTENT_Y + Inches(1.05), w=rw)
    _caption(s, "δ = θ_T − θ_a is the known displacement; the residual is pure approximation quality, no "
                "reconstruction involved", rx, CONTENT_Y + Inches(2.5), rw, h=Inches(0.9))
    _caption(s, "solid = leakage (control-margin), dashed = linearization error", rx, CONTENT_Y + Inches(3.5), rw,
             h=Inches(0.6), italic=True)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: anchor sweep theta(alpha) = (1-alpha) theta_0 + alpha theta_T, alpha in {0, 0.25, 0.5, 0.75, 0.9}
(experiments/configs.py:76-79; alpha capped below 1 because the anchor absorbs theta_T's training signal and
identifiability of x_i degrades). MNIST, T=10, r=8, N=2, seed 42; LoRA path; shown: relu vs softplus.
WHY THIS FUNCTION: L_lin is the function-space Taylor residual over the segment anchor -> theta_T
(experiments/ntk_verification.py compute_function_space_lin_error) - Gal's validation protocol verbatim: a pure
approximation-quality curve that never reconstructs x_i, drawn on the same axis as leakage(alpha).
WHY REPRESENTATIVE: relu: lin-err 0.398 -> 0.016 (25x) while control-margin sits flat/high 0.38 -> 0.62 (0.38, 0.39,
0.62, 0.54, 0.57) - margin decoupled from lin-error. softplus: lowest lin-err at every alpha (0.087 -> 0.010) and its
leakage is already maximal at alpha=0 (0.32) - the smoothest activation linearizes so well at init that the anchor buys
it nothing. gelu / silu (not shown): leakage climbs, peaks at alpha=0.75 (gelu 0.50, silu 0.29), then the alpha=0.9
identifiability collapse cuts it (full-space ssim_norm 0.96 -> 0.56) - the textbook two-curve shape; on the
first-pass SSIM read this was the 'interior optimum at alpha~0.75' (LoRA SSIM 0.06 -> 0.64).
GAL-ASK: G4. Midpoint (alpha=0.5) vs baseline: lin-err reduction is real for every activation; the leakage gain is
activation-dependent and, on the realistic metric, absent for the activations that linearize best.
CAVEATS: one seed; N=2; the attribution test (leakage should peak at or before the lin-err minimum) passes for gelu/silu/
softplus and FAILS for relu (flat) - relu's leakage is extraction/anchor geometry, not linearization fidelity.
Flowers32 free-c: smooth activations (gelu, softplus) do NOT leak at any alpha (control margin negative throughout);
relu leaks but climbs monotonically to alpha=0.9 -> attribution FAIL (anchor contamination). Full deck of curves in appendix.
PROVENANCE: job 532232 (MNIST two-curve); results/anchor_sweep_T10_r8_{relu,softplus}_s42.pth;
notes/crux_activation_analysis.md lines 155-195 (Task B tables); STATUS.md lines 1679-1690 and 1065-1071 (flowers).""")
    return s


# ----------------------------------------------------------------------------------------------
# 6. why: the activation enters only through sigma'
# ----------------------------------------------------------------------------------------------
def slide_mechanism(prs):
    s = _story(prs, "Why: the activation enters only through σ′",
               "two different objects, moving in opposite directions", "your ask: a mechanism")
    card_w = Inches(5.85)
    card_h = Inches(2.65)
    gap = C.CW - 2 * card_w
    y = CONTENT_Y
    sk_w, sk_h = Inches(2.3), Inches(1.3)
    for x, title, body, verdict, col, fill, kind in (
        (C.MX, "informativeness — rank of the static gate matrix M",
         "step-like σ′  →  each image gets a distinct, near-binary gate code\n"
         "→  distinct rows of M  →  high rank  →  the mixtures separate",
         "kinked wins", C.RED, C.LIGHTRED_FILL, "kinked"),
        (C.MX + card_w + gap, "linearization fidelity — continuity of the gate drift dM",
         "bounded σ″  →  small, continuous drift of the gates as θ moves\n"
         "→  features barely rotate;  a kink jumps at every crossing",
         "smooth wins", C.GREEN_OK, C.LIGHTGREEN_FILL, "smooth"),
    ):
        H.add_card(s, x, y, card_w, card_h, title, body, color=col, fill=fill, body_size=13)
        H.add_text(s, verdict, x + Inches(0.14), y + card_h - Inches(0.6), Inches(3.0), Inches(0.45), size=16,
                   bold=True, font=C.HEAD, color=col, anchor=MSO_ANCHOR.MIDDLE)
        H.fit_image(s, _sigma_prime_sketch(kind), x + card_w - sk_w - Inches(0.15), y + card_h - sk_h - Inches(0.12),
                    sk_w, sk_h, align="right")
    # equation strip
    ey = y + card_h + Inches(0.25)
    H.add_text(s, "the gate-weighted update — the activation appears only inside M", C.MX, ey, C.CW, Inches(0.3),
               size=13, bold=True, color=C.BLUE, align=PP_ALIGN.CENTER)
    p1 = render_math(r"\Omega=\sum_i g_i x_i^{\top}=G X^{\top},\quad G=D_v\,M\,D_c,\quad "
                     r"M_{ki}=\sigma'(\langle w_k,x_i\rangle)", "eq_gate_matrix")
    eq_w = Inches(7.4)
    pic = H.add_eq(s, p1, C.MX + (C.CW - eq_w) // 2, ey + Inches(0.4), w=eq_w)
    p2 = render_math(r"dM\propto \sigma''\,dz", "eq_gate_drift")
    eq2_h = Inches(0.42)
    iw, ih = H.image_size(p2)
    eq2_w = int(eq2_h * iw / ih)
    H.add_eq(s, p2, C.MX + (C.CW - eq2_w) // 2, pic.top + pic.height + Inches(0.15), h=eq2_h)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: wrote down the one object every experiment probes. For a one-hidden-layer net the first-layer weight
signal factors as Omega = G X^T with G = D_v M D_c: v_k is neuron-only, c_i is sample-only, and the gate
M_ki = sigma'(<w_k, x_i>) is the ONLY factor coupling neuron k and image i. Each neuron hands the attacker one weighted
mixture of the training images; the number of genuinely different mixtures is at most rank(M). The activation enters the
whole theory only through sigma' - through M.
WHY THIS FUNCTION: rank(G) = rank(M) needs D_v and D_c full rank (dead neurons drop D_v; at a KKT point non-support
vectors have c_i = 0, so the recoverable count is the rank over the SUPPORT-VECTOR columns). rank(M) = N is necessary
(else a blind subspace of dimension d(N - rank) is invisible to any attacker); the data-generated gate code supplies
sufficiency - if G were free, rank gives only the row space, not the images.
WHY REPRESENTATIVE (measured, job 668832, results/gate_matrix_test.csv, N=10 at theta_0): eff_rank(M) relu 6.4 ~ leaky
6.3 >> selu 3.4 > gelu 2.9 > mish 2.4 > silu 2.3 > softplus 1.7 > tanh 1.6 > sigmoid 1.2 - and this ordering matches the
independently measured LoRA leakage, including the selu surprise. Softplus-beta is a one-parameter dial: beta 0.5 -> 50
takes eff_rank 1.4 -> 5.3 and mean|sigma''| 0.13 -> 3.3, traversing smooth -> ReLU. Fidelity: dM ~ sigma'' dz, so bounded
sigma'' gives small continuous drift (smooth wins); ReLU is frozen within a region and jumps at every kink crossing.
GAL-ASK: G5 (one experiment + a mechanism). G3 and G4 are the two halves of this one picture.
CAVEATS: Lemma-B caveat - mean|sigma''| at theta_0 orders the linearization error only at MATCHED weight change (lin-err
~ sigma'' |delta|^2 and |delta| differs across activations). The Dirac point: relu/leaky have autograd mean|sigma''| = 0 yet
the maximal gate range - the information is in the range, not pointwise sigma''. Conditioning, not bare rank, is what we
report downstream (Parts 2-3).
PROVENANCE: notes/identifiability_rank_bound.tex lines 95-118; notes/linearization_leakage_theory.tex; STATUS.md lines
1397-1427 (gate-matrix test); thesis_note_v2.md §1.""")
    return s


# ----------------------------------------------------------------------------------------------
# 7. direct inversion: works at N=4, superposes at N=10
# ----------------------------------------------------------------------------------------------
def slide_direct_inversion(prs):
    s = _story(prs, "Direct inversion: works at N=4, superposes at N=10",
               "known-recipe upper bound — the map inverts; joint inversion is the bottleneck  (SimuDy's primitive, reframed)",
               "your ask: direct weight inversion")
    n4 = _crop(C.ASSET["di_N4"], "di_N4_crop.png", (0.27, 0.0, 1.0, 0.95))
    n10 = _crop(C.ASSET["di_N10"], "di_N10_crop.png", (0.277, 0.0, 1.0, 0.915))
    lab_w = Inches(1.45)
    img_x = C.MX + lab_w
    # row 1: N=4
    y1, h1 = CONTENT_Y, Inches(2.3)
    px, py, pw, ph = H.fit_image(s, n4, img_x, y1, Inches(5.0), h1, align="left")
    H.add_text(s, "four images", C.MX, py, lab_w, Inches(0.3), size=13, bold=True, color=C.BLUE)
    H.add_text(s, "private", C.MX, py + Inches(0.55), lab_w, Inches(0.3), size=12, color=C.GRAY)
    H.add_text(s, "recovered", C.MX, py + ph - Inches(0.75), lab_w, Inches(0.3), size=12, color=C.GRAY)
    # right column: equation
    rx = px + pw + Inches(0.4)
    rw = C.SL_W - C.MX - rx
    H.add_text(s, "endpoint matching", rx, y1 + Inches(0.15), rw, Inches(0.3), size=13, bold=True, color=C.BLUE,
               align=PP_ALIGN.RIGHT)
    p = render_math(r"\min_{\hat x}\ \|\theta_T-F(\theta_0,\hat x)\|^2", "eq_direct_inversion")
    eq_w = Inches(3.6)
    H.add_eq(s, p, rx + rw - eq_w, y1 + Inches(0.5), w=eq_w)
    _caption(s, "F = the whole fine-tuning run; autograd through the unrolled SGD reaches the candidate images",
             rx, y1 + Inches(1.3), rw, h=Inches(0.9), align=PP_ALIGN.RIGHT)
    # row 2: N=10
    y2 = y1 + h1 + Inches(0.2)
    h2 = C.CB - y2
    px2, py2, pw2, ph2 = H.fit_image(s, n10, img_x, y2, C.CW - lab_w, h2, align="left")
    H.add_text(s, "ten images", C.MX, py2, lab_w, Inches(0.3), size=13, bold=True, color=C.BLUE)
    H.add_text(s, "private", C.MX, py2 + Inches(0.5), lab_w, Inches(0.3), size=12, color=C.GRAY)
    H.add_text(s, "recovered", C.MX, py2 + ph2 - Inches(0.7), lab_w, Inches(0.3), size=12, color=C.GRAY)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: DI-Phase 0 (experiments/direct_inversion.py). Treat fine-tuning as a deterministic differentiable map
theta_T = F(theta_0, {x_i}) and minimise the endpoint loss ||theta_T - F(theta_0, x_hat)||^2 over the candidate images
(Regime A, endpoint only), Adam on x_hat, autograd through an UNROLLED full-batch SGD F. MNIST MLP, LoRA r=8, GELU, T=10.
WHY THIS FUNCTION: the endpoint loss is exactly Gal's formulation (briefing §3) - no linearization, no decoder, F as a
black box. GELU is REQUIRED: unrolling differentiates through the gradient (double backward); ModifiedReLU has no
double-backward. At T=1 the map is bit-exact (loss and input-gradient exactly 0 at the truth; DI-T1 0.57 vs the
Experiment-B oracle NTK T=1 ~0.50 - consistent, DI slightly above as it matches endpoints exactly).
WHY REPRESENTATIVE: ssim_norm 0.57 at N=4 (T=1..10 stable, degrading at T=20) -> 0.27 at N=10 -> ~0.15 at N=20. The N=4 ->
N=10 DROP is the signal; the absolute is ssim_norm (mean/std-matched, which inflates it; raw ssim lower). At N=10 the control
margin survives but shrinks (+0.05 / +0.06). A tighter pixel box (box=5) does NOT rescue N=10 (0.13 / 0.27, mixed) - the
collapse is joint-inversion difficulty, not clipping. Grids: figures/direct_inversion/di_grid_N{4,10}_r8_gelu_T10.png
(per-image SSIM captions cropped off for the slide).
GAL-ASK: G1. The map inverts at small N; the superposition wall (any recombination of per-sample gradients with the
same sum is indistinguishable) is what stops it. SimuDy (Tian et al., ICLR 2025; Gal's 2026-06-29 pointer) publishes the
same primitive for full FT (MLP/100 SSIM 0.34, ResNet/50 0.20, 22 GB / 15 h). We reframed: LoRA-only leakage,
identifiability theory (Parts 2-3), memory-tractable linearized inversion. Open ask for Gal: does the reframe hold?
CAVEATS: toy MLP; known recipe (lr, T, full batch) = best-case upper bound; single seed per N.
PROVENANCE: job 500913 (N=4 SSIM-vs-T, results/direct_inversion_N4_r8_gelu.pth); job 887704 (N=10/20 + pixel box);
STATUS.md lines 1632-1640, 1667-1677, 1750-1766; notes/simudy_decision_brief.md; thesis_note_v2.md E7.""")
    return s


# ----------------------------------------------------------------------------------------------
# 8. more data: the full-gradient ceiling is recognizable
# ----------------------------------------------------------------------------------------------
def slide_more_data(prs):
    s = _story(prs, "More data: the full-gradient ceiling is recognizable",
               "this is the CEILING (true weight change) — not the adapter-only attack", "your ask: more data")
    gal_w = Inches(6.9)
    px, py, pw, ph = H.fit_image(s, C.fig("gallery.png"), C.MX, CONTENT_Y, gal_w, Inches(2.7), align="left")
    faces = _crop(C.ASSET["faces"], "faces_crop.png", (0.062, 0.04, 1.0, 0.95))
    fx = C.MX + gal_w + Inches(0.35)
    fw = C.SL_W - C.MX - fx
    fpx, fpy, fpw, fph = H.fit_image(s, faces, fx, CONTENT_Y, fw, Inches(3.6), align="left")
    _caption(s, "ViT-B/16 on faces — top: private, bottom: three images recovered jointly from one captured "
                "gradient; colour is the ceiling's weakest channel", fx, fpy + fph + Inches(0.08), fpw, h=Inches(0.6), italic=True, align=PP_ALIGN.CENTER)
    # equation under the gallery
    ey = py + ph + Inches(0.3)
    H.add_text(s, "what the ceiling inverts", C.MX, ey, gal_w, Inches(0.3), size=13, bold=True, color=C.BLUE)
    p = render_math(r"\Delta W\approx\sum_i c_i\,\nabla_W f(\theta_a;x_i),\qquad \theta_a=\theta_0\ \mathrm{here}", "eq_delta_w")
    H.add_eq(s, p, C.MX, ey + Inches(0.4), h=Inches(0.7))
    _caption(s, "the weight change as a coefficient-weighted sum of per-image feature gradients at the anchor θ_a (α=0 → θ₀); recover the x_i (and "
                "the free c_i) that reproduce it.  A released adapter exposes only a low-rank image of ΔW — inverting "
                "that is the open milestone.", C.MX, ey + Inches(1.2), gal_w, h=Inches(0.85))
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: reconstruction from the TRUE full weight change (the strong-attacker / known-recipe ceiling), N=2,
GELU (softplus for Fashion), across MNIST / Fashion-MNIST / CIFAR-10 / Flowers32 (results/gb_e2e_*_N2_gelu.pth, key
'TRUE dW (ceiling)'), plus the ViT-B/16 Phase-0 run: three faces recovered jointly from one captured fine-tuning gradient
(figures/phase0/n3_three_faces.png; per-image SSIM 0.38 / 0.26 / 0.52, cropped off the slide).
WHY THIS FUNCTION: dW ~ sum_i c_i grad_W f(theta_0; x_i) is the NTK / gradient-recording view of the update; the
free-coefficient attack recovers x_i and c_i jointly (Haim-style; oracle c_i is an upper bound only). With the full-model
gradient the system is massively over-determined even in raw pixel space; the released LoRA adapter exposes only
P_LoRA(H) = B B^T H + H A^T A - a low-rank image of it.
WHY REPRESENTATIVE: SSIM up to ~0.99 on MNIST / CIFAR / Flowers; Fashion partial. Faces return identity and structure but
lose pose/clothing detail. These are the positive examples Gal asked for; the gallery is a sweep, not a best run.
GAL-ASK: G2 (more data / distributions, not best runs). Everything here is the CEILING - the information is in the
weights and is recoverable. The robust ADAPTER-ONLY pixel inversion is the open extraction milestone (E7, World B): the
ruler in Parts 2-3 says the directions are present; the decoder is what is missing. Do not read this slide as the
adapter attack succeeding.
CAVEATS: N=2 per dataset; ceiling only; faces are a Phase-0 result from the captured full gradient of a ViT-B/16 (not an
adapter); consent/provenance of the face images to be stated once (thesis_note_v2.md 'to resolve').
PROVENANCE: results/gb_e2e_{mnist,fashion,cifar10,flowers32}_N2_gelu.pth (full-gradient arm, job 956994 lineage);
figures/meeting/positive_reconstruction_gallery.png (experiments/plot_reconstruction_gallery.py);
notes/meeting_prep_2026-08-31.md F-0; thesis_note_v2.md E7 + provenance table.""")
    return s


SLIDES = [slide_title, slide_changed, slide_crux, slide_fs, slide_anchor, slide_mechanism,
          slide_direct_inversion, slide_more_data]
