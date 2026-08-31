# Slide-module contract (for parallel builders)

Each `scripts/deck/deck/slides_<part>.py` exposes `SLIDES = [build_fn, ...]`, each `build_fn(prs)` adds ONE slide and
returns it. The orchestrator calls them in order and sets `helpers.set_total()`.

```python
from . import config as C
from . import helpers as H
from .eq_render import render_math, render_lines
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN

def slide_xx(prs):
    s = H.new_slide(prs)
    H.add_title(s, "Plain statement title (≤ 10 words)")
    H.add_lead(s, "one line, ≤ ~18 words, the point of the slide")        # y = C.CT
    H.add_tag(s, "your ask: smooth activations", C.SL_W - C.MX - Inches(3.0), C.MY + Inches(0.15), w=Inches(3.0))  # optional
    # content zone: x∈[C.MX, C.MX+C.CW], y∈[C.CT+Inches(0.5), C.CB]
    H.fit_image(s, C.fig("crux_bars.png"), C.MX, C.CT + Inches(0.55), Inches(8.2), C.CB - C.CT - Inches(0.6))
    p = render_math(r"\Omega=\sum_i g_i x_i^{\top}", "eq_omega")                     # name must be unique deck-wide
    H.add_eq(s, p, C.MX + Inches(8.5), C.CT + Inches(0.8), w=Inches(3.9))            # give w OR h
    H.add_footer(s)
    H.set_notes(s, """WHAT WE DID: ...
WHY THIS FUNCTION: ...
WHY REPRESENTATIVE: ...
GAL-ASK: ...
CAVEATS: ...
PROVENANCE: job ..., file ...""")
    return s
```

Rules (non-negotiable):
- Visible text per story slide: title + lead + ≤ 2 numeric tokens + short labels. Everything else → notes.
- Never on a slide: "0/40", "‖ΔW‖/‖W₀‖=0.23", "1.07", "0.61", "confirmed", "settled", "duplication-invariance", a single canonical crux Spearman.
- Equations: matplotlib mathtext only (no \text{}, no \operatorname; use \mathrm{}); keep each ≤ 1 line; unique `name`.
- Geometry: 16:9, EMU via Inches(); content must end above C.CB (6.85"); footer via H.add_footer(s) on every slide.
- Colours: C.BLUE (accent), C.RED (kinked / full-FT), C.GREEN_OK (smooth / positive), C.GRAY, C.AMBER (open).
- Native shapes (H.add_rect / H.add_card / H.add_arrow / H.add_text) for schematics and tables — no rasterised tables.
- Do not import anything from experiments/; figures are prebuilt in figures/deck_2026_08_31/ (names below).

Prebuilt figures (C.fig(name)): crux_bars.png · fs_vs_T.png · anchor_two_curve.png · rank_sweep.png · spectrum_r8.png ·
estimator_honest.png · battery_knobs.png · arm_c.png · g0_scatter.png · ladder_strip.png · h_gate.png · beyond_mlp.png ·
atlas_2panel.png · gallery.png.   As-is assets (C.ASSET[k]): di_N4, di_N10, faces, vit_face.

## Quality rules adopted from the hand-finished v20 (2026-08-31)

These came from comparing the generator's 29-slide build with the deck as actually finished for the meeting
(`figures/deck_v20_spec/`). They are requirements, not suggestions:

1. **Define the object before you show its number.** A result slide that introduces a new quantity gets a setup slide
   in front of it (what is perturbed, what is watched, what exactly is computed). Do not leave that in the notes.
2. **Every non-obvious figure carries a "reading the plot" block** — what one point/bar/colour is, in the audience's
   words. `H.add_reading_block(s, x, y, w, "each dot is one private image: ...")`.
3. **Uncertainty is a titled block in the body**, not a footnote: n, CI, what is confounded, what has NOT been tested.
   `H.add_caveat_block(s, x, y, w, "how sure are we? not yet", "n = 24; the predictors are correlated; ...")`.
4. **Titles state the claim and name the mechanism** ("The asymmetry is not rarity — it follows the base gradient"),
   never a bare object label ("class identity").
5. **Split rather than shrink.** Two ideas ⇒ two slides. A mechanism and its result are two slides.
6. **Answer the supervisor's own inputs explicitly** — a paper he sent, a question he asked, gets its own slide.
7. **The TOC is ask → answer → where** (slide number), not a list of section names.
8. **The appendix carries method honesty**: how the instrument was made honest, and why every knob is defensible
   (grid-searched / gate-checked / swept), not only formulas.
9. **A closing slide** that says what the appendix holds and invites the discussion.

Notes template gains one line for figure slides: `READING: <what one mark on the plot is>`.
