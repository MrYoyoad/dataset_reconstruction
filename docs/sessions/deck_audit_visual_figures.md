# Deck visual-legibility audit — figures & plots at meeting distance

Deck: 28-slide supervisor meeting deck (`figures/deck_2026_08_31/*`, render in `deck_render/`).
Scope: read-only judgement of every figure/plot as it reads projected, from across a room.
"Evaluation Warning" top-left watermark ignored (render artifact).

Bottom line: the plotting is strong — labels, ticks, legends, markers and colour usage are
clean and readable on almost every slide. Only a handful of genuine defects, none of them a
hard blocker. No invented nitpicks below.

---

## BLOCKER
(none)

---

## SHOULD-FIX

### S13 · right-column lead line · text truncated off the slide edge
The emphasised blue takeaway runs off the right margin: it reads
"⇒ a ceiling on detecting the change — for every atta" and then wraps to
"reconstruction included" — the word "attacker" is clipped between "atta" and the next
line, so the sentence is literally cut off the slide. This is the slide's headline claim.
Fix: shorten the phrase (e.g. "…for every attacker — reconstruction included"), narrow the
font, or widen/reflow the text box so the full line stays inside the slide.

### S14 · right panel ("zero reads zero; one image reads loud") · invisible gray control series
The legend lists "nothing swapped, seeds only" (gray), but that series has no visible bars —
the null is shown only as two tiny gray numeric labels ("-0.001" at N=4, "+0.003" at N=16),
with nothing at N=8 or N=32. At meeting distance the audience sees only the blue bars and a
gray legend entry with no matching mark, so the "zero reads zero" half of the panel title is
not visually supported. Fix: label all four N with the near-zero null value in a larger font,
or draw a visible near-zero gray bar/marker per N, or add "gray ≈ 0 (too small to see)" to the
caption.

---

## MINOR

### S3 vs S4 · gelu colour inconsistent across adjacent slides
On S3 (crux_bars) gelu is a green "smooth" bar; on S4 (fs_vs_T) gelu is drawn blue, separate
from the green smooth activations. Same activation, two colours two slides apart. Defensible
(S4 singles gelu out as the transition case), but it can read as a different activation. Also
the S4 side caption ("smooth (green) stays near the linear regime; kinked (red, dashed) leaves
it") omits the blue gelu line, which is a smooth activation that nonetheless drops to ~0.59 —
the green-vs-red dichotomy silently skips the blue middle case. Fix: either colour gelu green
like S3, or add one clause to the caption acknowledging gelu (blue) as the smooth outlier.

---

## Checked and clean (no action)
- S3 crux_bars, S5 anchor_two_curve, S10 rank_sweep, S11 spectrum_r8, S16 battery_knobs
  (3-panel), S17 arm_c, S18 g0_scatter, S20 h_gate, S21 beyond_mlp (4-panel), S22 atlas_2panel:
  axis/tick labels, legends, markers and lines all readable at distance; legends do not cover
  data.
- S7 DI grids (N=4, N=10): digits clearly visible; the noisy N=10 recovered row is the intended
  message, not a legibility fault.
- S8 MNIST/CIFAR/Flowers + ViT-faces panel: reconstructions and small row labels readable; the
  red tint on recovered faces is explained by the caption.
- S19 ladder_strip: per-tile s= and d= labels readable; message (identity→loud) visible.
- Colour semantics otherwise consistent: red = kinked/full-FT, green = smooth/good,
  blue = binary/accent, orange = 10-class/Fashion (e.g. S21 panel 4 red=full-FT vs blue=LoRA;
  S10/S11 blue=binary vs orange=10-class; S16 blue=MNIST vs orange=Fashion — all correct).

Note (not a deck defect): the as-is asset `figures/direct_inversion/di_grid_N10_r8_gelu_T10.png`
has an extreme ~5:1 aspect ratio with a large empty left margin, but S7 uses its own compact
re-layout, so the slide itself is fine.
