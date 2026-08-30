"""Part 2 — Where that led me: the theory before the tests (3 slides).

T1  Fine-tuning is a measurement system            (native schematic z -> x -> LoRA -> F(z))
T2  Dimension count -> Jacobian rank -> spectrum    (native chain + rank_sweep.png)
T3  A direction counts only if it beats the noise   (spectrum_r8.png + J_SNR / q_eff + three worlds)

Source of record for the chain: notes/identifiability_feasibility_revision.tex:41-153.
"""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from . import config as C
from . import helpers as H
from .eq_render import render_math, render_lines

Y0 = C.CT + Inches(0.55)          # first content row (below the lead line)


def _box(slide, x, y, w, h, title, sub, *, color=C.BLUE, fill=None, title_size=15, sub_size=12):
    """Bordered rounded box with a bold one-line title and a small gray sub-line (centred)."""
    H.add_rect(slide, x, y, w, h, fill_color=fill or C.WHITE, line_color=color, line_width=1.25,
               shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    H.add_text(slide, title, x + Inches(0.08), y + Inches(0.06), w - Inches(0.16), Inches(0.36),
               size=title_size, bold=True, font=C.HEAD, color=color, align=PP_ALIGN.CENTER,
               anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        H.add_text(slide, sub, x + Inches(0.08), y + Inches(0.42), w - Inches(0.16), h - Inches(0.48),
                   size=sub_size, color=C.GRAY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP)


# --------------------------------------------------------------------------------------
def slide_t1(prs):
    s = H.new_slide(prs)
    H.add_title(s, "Fine-tuning is a measurement system")
    H.add_lead(s, "the whole privacy question in one map: private latents in, released adapter out")

    # four stages across, arrows between
    n, gap = 4, Inches(0.62)
    cw = int((C.CW - 3 * gap) / n)
    ch = Inches(1.1)
    y = Y0 + Inches(0.15)
    stages = [
        ("latent  z", "the private degrees of freedom", C.GRAY),
        ("images  x = g(z)", "on a manifold of dimension k, far below d", C.BLUE),
        ("LoRA training", "T steps, one random seed", C.BLUE),
        ("adapter  (A_T, B_T) = F(z)", "what is released", C.RED),
    ]
    xs = []
    for i, (t, sub, col) in enumerate(stages):
        x = C.MX + i * (cw + gap)
        xs.append(x)
        _box(s, x, y, cw, ch, t, sub, color=col)
        if i:
            H.add_arrow(s, x - gap + Inches(0.06), y + ch // 2, x - Inches(0.06), y + ch // 2,
                        color=C.GRAY, width=2.0)
    # the two lossy stages, both of which help the attacker
    ly = y + ch + Inches(0.12)
    H.add_text(s, "first lossy stage: images are not arbitrary pixels", xs[1] - gap // 2, ly, cw + gap,
               Inches(0.3), size=C.SZ_SMALL, italic=True, color=C.GRAY, align=PP_ALIGN.CENTER)
    H.add_text(s, "second lossy stage: gradients of real images are not arbitrary either", xs[2], ly,
               cw + gap + cw, Inches(0.3), size=C.SZ_SMALL, italic=True, color=C.GRAY, align=PP_ALIGN.CENTER)

    # the map as one equation, centred
    p = render_math(r"z\ \overset{g}{\longrightarrow}\ x\ \overset{\mathrm{LoRA}}{\longrightarrow}\ (A_T,B_T)=F(z)",
                    "eq_t1_measurement_map")
    ew = Inches(6.6)
    H.add_eq(s, p, C.MX + (C.CW - ew) // 2, ly + Inches(0.6), w=ew)

    # the question
    qy = Inches(5.45)
    qw = Inches(9.6)
    qx = C.MX + (C.CW - qw) // 2
    H.add_rect(s, qx, qy, qw, Inches(0.7), fill_color=C.LIGHTBLUE_FILL, line_color=C.BLUE, line_width=1.25,
               shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    H.add_text(s, "is F locally injective and well-conditioned on the image manifold?", qx, qy, qw, Inches(0.7),
               size=18, bold=True, font=C.HEAD, color=C.BLUE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: Wrote the whole experiment as one deterministic map. A latent z=(z_1..z_N) generates the private images x=g(z); LoRA training turns them into the released adapter (A_T,B_T)=:F(z). Privacy is then a single question about F: is it locally injective and well-conditioned on the image manifold M_X?
WHY THIS FUNCTION: F has TWO lossy stages, and both help the attacker. (1) Images are not arbitrary pixels: realistic images occupy a manifold of local dimension k << d (moving between them takes coordinated changes: pose, lighting, shape, texture). (2) Gradients of real images are not arbitrary either: M_G = {grad L(X): X realistic} is a thin subset of gradient space. So LoRA may destroy vast amounts of ARBITRARY gradient information while still separating points of M_G: the attacker needs a RESTRICTED inverse, not a full one.
WHY REPRESENTATIVE: This is the framing every later slide uses. The Jacobian program (next two slides) differentiates F; the secret-swap test (Part 3) is the finite-difference version of the same question (D vs D' = two points z, z').
GAL-ASK: This grew out of your May question about more data / distributions (G2): the honest way to ask "does the adapter carry the images" is to ask about the conditioning of F on the data manifold, not about a capacity number.
CAVEATS: The earlier rank theorem (gate matrix G = sigma'(<w_k,x_i>), Omega = G X^T; images survive in separable form iff rank(G) >= N) is a MECHANISM statement, not an attack: the attacker neither knows G nor can hold it fixed, because G depends on the very images being sought. Read rho >= N as a necessary floor on the first-order signal, not the verdict. (identifiability_feasibility_revision.tex:41-53.)
PROVENANCE: notes/identifiability_feasibility_revision.tex:41-72 ("Read the rank theorem correctly" + "Fine-tuning as a measurement system"), written 2026-08-20..23. Gate-matrix theorem: notes/identifiability_rank_bound.tex.""")
    return s


# --------------------------------------------------------------------------------------
def slide_t2(prs):
    s = H.new_slide(prs)
    H.add_title(s, "Only the spectrum measures usable leakage")
    H.add_lead(s, "dimension count → Jacobian rank → singular spectrum; a rank only sizes the null space")

    # left: the three-box chain, increasing honesty downwards
    lx, lw = C.MX, Inches(4.15)
    bh, gap = Inches(0.78), Inches(0.36)
    chain = [
        ("dimension count", "d\u00b7rank \u2265 private directions \u2014 necessary, nothing more", C.GRAY),
        ("Jacobian rank", "how many private directions are recorded at all", C.BLUE),
        ("singular spectrum", "how strongly each one is recorded", C.GREEN_OK),
    ]
    y = Y0 + Inches(0.05)
    for i, (t, sub, col) in enumerate(chain):
        _box(s, lx, y, lw, bh, t, sub, color=col, title_size=15, sub_size=11)
        if i < len(chain) - 1:
            H.add_arrow(s, lx + lw // 2, y + bh + Inches(0.03), lx + lw // 2, y + bh + gap - Inches(0.03),
                        color=C.GRAY, width=2.0)
        y += bh + gap
    p = render_math(r"J_{\mathrm{full}}=\frac{\partial\,\mathrm{vec}(A_T,B_T)}{\partial(z_1,\dots,z_N)}",
                    "eq_t2_j_full")
    H.add_eq(s, p, lx + Inches(0.55), y + Inches(0.1), w=Inches(3.1))
    H.add_text(s, "prediction: \u03c3_min(J) collapses where reconstruction collapses", lx,
               Inches(6.35), lw, Inches(0.4), size=C.SZ_SMALL, italic=True, color=C.GRAY, align=PP_ALIGN.CENTER)

    # right: the concrete counter-example to "rank-r LoRA => r images"
    fx = lx + lw + Inches(0.35)
    fw = C.SL_W - C.MX - fx
    H.fit_image(s, C.fig("rank_sweep.png"), fx, Y0 - Inches(0.05), fw, C.CB - Y0 - Inches(0.02))
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: Replaced the capacity-style argument with the object that actually decides identifiability. Chain of increasing honesty: (1) dimension count: N images with k_i local degrees of freedom give q = sum_i k_i private directions; a rank-rho measurement spans at most d*rho directions, so a regular locally invertible map needs d*rho >~ q. This is a dimensional plausibility check, necessary and nothing more. (2) Jacobian rank r_J of J_full = d vec(A_T,B_T) / d(z_1..z_N): how many private directions are recorded at all (noise-free). (3) The singular spectrum of J_full: how strongly each is recorded. Only (3) measures usable leakage.
WHY THIS FUNCTION: Counter-example that kills (1) and (2) as leakage measures: the map diag(1, 1e-10) preserves rank 2, yet the second coordinate is gone for any practical purpose. So "having enough dimensions" does not mean they point in useful directions; the d*rho >= N*k inequality sizes the NULL SPACE, not a capacity law. The rank-r LoRA manifold has dimension r(m+d-r), which is why a rank-count can be large while the spectrum is terrible.
WHY REPRESENTATIVE: The figure is the concrete example that "rank-r LoRA => r images" is FALSE. MNIST, N=10 private images x 8 latent directions each (Nk=80), gelu, T=1000, lr=0.5, S=320 seeds, both task heads, r in {8,16,32}. r_J = 80 (full) at EVERY rank, so rank says nothing. The recoverable fraction (q_eff at eps=1, over 80) stays ~3/4 for the binary task at every rank (59/60/58) while the 10-class task climbs 36 -> 47 -> 58 and catches up by r=32: gap 23 -> 13 -> 0. Rows r=2/4 are quarantined: the 10-class arm does not memorize there (max per-sample BCE > 1e-3), so their q_eff is confounded (job 635386 showed r=2/4 need T~4000-8000, past the FD-chaos wall).
GAL-ASK: Your G2 ("more data, not best runs"): this is a full rank sweep, both heads, all FD-clean (r=32, dimY=57088, FD rel err 3.2e-8).
CAVEATS: Jang et al. 2024 proves rank on the order of sqrt(N) kills spurious minima (NOT r >~ N); the K*N constraint count for multi-class is OUR extrapolation, not Jang's bound. The iso mechanism (seed-noise coupling into col(J)) explains the reversal at r=8 (10-class iso 0.683 > binary 0.491) but DECOUPLES at r>=N: at r=16 the iso gap flips (0.389 vs 0.808) yet q_eff still reverses (47 < 60). The prediction sigma_min(J_full) collapses where reconstruction collapses is the note's strongest empirical claim and is still to be tested on the k-known-by-construction generator setup (Worlds A/B/C, next slide). Fashion 10-class is bounded out (numerically unstable at the matched recipe); MNIST carries the headline.
PROVENANCE: notes/identifiability_feasibility_revision.tex:74-115. Rank sweep job 581629 (results/jacobian_j1_ranksweep_*.pth, log scripts/wexac_logs/mc_rank_sweep_581629.out), figure generator experiments/plot_rank_sweep.py (job 933413) -> figures/deck_2026_08_31/rank_sweep.png; STATUS.md:195-252; plan notes/lora_rank_sweep_plan.md.""")
    return s


# --------------------------------------------------------------------------------------
def slide_t3(prs):
    s = H.new_slide(prs)
    H.add_title(s, "A direction counts only if it beats the training noise")
    H.add_lead(s, "training randomness is the noise floor \u2014 this is what forced \u03a3_seed into the picture")

    # left: the per-direction SNR spectrum
    fw = Inches(6.35)
    H.fit_image(s, C.fig("spectrum_r8.png"), C.MX, Y0 - Inches(0.05), fw, C.CB - Y0 - Inches(0.3))

    # right: the two definitions, then the three worlds
    rx = C.MX + fw + Inches(0.35)
    rw = C.SL_W - C.MX - rx
    p = render_lines([r"J_{\mathrm{SNR}}=\Sigma_{\mathrm{seed}}^{-1/2}\,J",
                      r"q_{\mathrm{eff}}(\varepsilon)=\#\{\,i:\ \varepsilon\,\sigma_i(J_{\mathrm{SNR}})>1\,\}"],
                     "eq_t3_jsnr_qeff")
    H.add_eq(s, p, rx + Inches(0.1), Y0 - Inches(0.02), w=Inches(4.6))

    y = Y0 + Inches(1.55)
    H.add_text(s, "three worlds a negative reconstruction can live in", rx, y, rw, Inches(0.3),
               size=C.SZ_SMALL, italic=True, color=C.GRAY)
    y += Inches(0.35)
    worlds = [
        ("A", "identifiability wall", "J collapses  &  the attack fails", C.RED, C.LIGHTRED_FILL),
        ("B", "extraction-limited", "J fine, the attack still fails \u2014 decoder is the bottleneck", C.BLUE, C.LIGHTBLUE_FILL),
        ("C", "prior hallucination", "J collapses, yet the decoder still emits images", C.AMBER, C.LIGHTGRAY_FILL),
    ]
    rh, rg = Inches(0.86), Inches(0.12)
    for letter, name, body, col, fill in worlds:
        H.add_rect(s, rx, y, rw, rh, fill_color=fill, line_color=col, line_width=1.0,
                   shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        H.add_text(s, letter, rx + Inches(0.12), y, Inches(0.5), rh, size=26, bold=True, font=C.HEAD,
                   color=col, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
        H.add_text(s, name, rx + Inches(0.7), y + Inches(0.08), rw - Inches(0.8), Inches(0.32), size=14,
                   bold=True, font=C.HEAD, color=col)
        H.add_text(s, body, rx + Inches(0.7), y + Inches(0.42), rw - Inches(0.8), rh - Inches(0.46),
                   size=C.SZ_SMALL, color=C.BLACK)
        y += rh + rg
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: Put the training-randomness floor into the Jacobian. Rerun the same fine-tune S times with different seeds at a fixed secret, estimate Sigma_seed, and whiten: J_SNR = Sigma_seed^(-1/2) J. Its singular values are per-direction signal-to-noise ratios; q_eff(eps) = #{i : eps*sigma_i(J_SNR) > 1} counts the private directions that clear the floor at perturbation budget eps. The figure is the r=8 spectrum for both task heads (MNIST, N=10, k=8, Nk=80, S=320, eps=1 line = the noise floor).
WHY THIS FUNCTION: In detection language this is a Fisher-information / Neyman-Pearson object: q_eff is "how many directions the weakest attacker has non-trivial power on", which is why every number is a LOWER BOUND. Implementation: the full Sigma_seed (dimY ~ 14k) is unmeasurable at feasible S; we estimate the noise ONLY inside col(J) (Sigma_J = Cov(Q^T (Y - Ybar)), r_J x r_J) and whiten there. Observing only col(J) uses less of Y, so its Fisher <= the full Fisher (Schur complement) => q_eff|col(J) is a conservative lower bound on the true q_eff.
WHY REPRESENTATIVE: 10-class buries MORE directions under the floor than binary (36 vs 59 of 80 at r=8): CE injects more seed noise into col(J) (iso_ratio 0.683 vs 0.491). Both heads record ALL 80 directions noise-free (r_J = 80): the difference is entirely in the floor, which is the point of the slide.
GAL-ASK: This is the ruler the rest of the talk uses. Worlds: A is proven by the ruler (attack-independent); C is excluded by the disjoint-adapter control (prior-independent); only what survives both is a genuine World-B leak. Where do you want me to spend compute: proving A at scale (a scoped guarantee) or building the World-B decoder?
CAVEATS: (1) q_eff needs S >~ 4*Nk seeds (empirically eff_rank(Sigma_seed) >~ r_J) AND stability across {S, 2S}; the retracted "2x multi-class amplification" (q_eff 97) was S=64 undersampling of a 160-dim noise cloud stacked on an underfit binary arm. (2) Leakage is r_J and q_eff, NEVER eff_rank (entropy effective rank reads backwards: smooth activations concentrate the spectrum, eff_rank drops while true leakage hits max). (3) q_eff is coordinate-dependent under reparametrization a -> M a'; hard_rank and iso_ratio are invariant; report q_eff in a fixed natural metric. (4) Exact unrolled Jacobian has a meta-gradient-chaos wall (FD rel err 3.9e-8 at lr=0.6 -> 1.0 at lr=0.7); island is lr<=0.6, T<=1000; always FD-gate the actual J before quoting q_eff. (5) d^2=2KL and the recovery reading assume Gaussian seed noise + local linearity of J: q_eff certifies LOCAL identifiability, not global invertibility.
PROVENANCE: experiments/jacobian_spectrum.py:21,546-620 (snr_spectrum, q_eff, q_eff_colspace); LESSONS_LEARNED.md:79-216 (q_eff hygiene, 2026-08-23..26); spectrum figure from job 581629 (r=8 cell) via experiments/plot_rank_sweep.py -> figures/deck_2026_08_31/spectrum_r8.png; three worlds: notes/identifiability_feasibility_revision.tex:117-142 and notes/thesis_note_v2.md section 4.""")
    return s


SLIDES = [slide_t1, slide_t2, slide_t3]
