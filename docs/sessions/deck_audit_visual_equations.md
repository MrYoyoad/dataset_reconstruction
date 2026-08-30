# Deck visual audit — equation & notation rendering lens

**Scope:** all 28 rendered slides (`deck_render/slide_01–28.png`), plus full-res eq PNGs and
`scripts/deck/` sources for cross-referencing intended symbol forms.

**Summary:** 28 slides viewed, 0 UNVERIFIED. 7 defects — 1 high (content clipped off-slide),
2 medium (native-vs-rendered symbol mismatch / glitched re-typed equation), 4 low (cosmetic
notation inconsistencies + one under-sized appendix block). The equation rendering itself is clean:
no tofu/boxes, no literal unrendered LaTeX, all hero equations (`d²`, `leak`, `J`, KKT) are large and
legible at 2 m. Defects below are ranked most-severe first; clean slides are not listed.

---

## HIGH — content is lost / unreadable

**S13 · blue conclusion heading is clipped off the right slide edge.**
The line renders as "⇒ a ceiling on detecting the change — for every atta" — the word "attack" is
cut mid-word and "reconstruction included" is lost. Source: `slides_measure.py:187` places the full
string `"⇒  a ceiling on detecting the change — for every attack, reconstruction included"` in a box
whose right edge (rx+rw) runs past the slide, with no wrap. The equation above it
(`d² = SNR²_NP = 2 KL(P_D′ ‖ P_D)`) is fine — only this native heading overflows.
**Fix:** narrow the text box (reduce `rw`) and enable word-wrap so the line wraps to two lines within
the slide, or drop the font ~2 pt; verify "reconstruction included" is visible.

---

## MEDIUM — native-vs-rendered symbol mismatch on the same slide

**S18 · `∇_W` (plot axis, ASCII) vs `∇_{W₀}` (rendered equation) — same operator, two forms.**
Rendered eq (`slides_results.py:92`): `g₀(xᵢ) = ‖∇_{W₀} BCE(θ₀; xᵢ)‖_F`. The scatter x-axis
(`make_deck_figures.py:349`) reads `g₀ = ‖∇_W BCE‖ of the image at the PUBLIC base model θ₀` — it
drops the `₀` on the gradient subscript. **Fix:** unify the axis label to `∇_{W₀}` (match the equation).

**S5 · x-axis label re-types the slide's own equation and glitches the subscript.**
The properly typeset anchor equation `θ(α) = (1−α)θ₀ + α θ_T` is already rendered on the slide (right
column). The x-axis (`make_deck_figures.py:165`) repeats it as
`anchor α    θ(α) = (1−α)·θ₀ + α·θ_T`, where `θ_T` shows as a literal underscore-T (`θ_T`) — unicode
`θ₀` but ASCII `_T` in the same token, and redundant with the rendered eq. **Fix:** shorten the axis
label to just `anchor α` (the typeset eq carries the definition), or replace `θ_T` so it does not show
a bare `_T`.

---

## LOW — cosmetic notation inconsistencies

**S4 · y-axis `cos(∇f(θ₀), ∇f(θ_T))` shows literal `_T` and uses `∇f` where the rendered eq uses
`∇_θ f`.** Source `make_deck_figures.py:146`; rendered eq is
`fs(T) = cos(∇_θ f(θ₀;x), ∇_θ f(θ_T;x))`. **Fix:** drop the bare `_T` glitch; optionally align `∇f`→`∇_θ f`.

**S16 · legend `d² ∝ k^0.23` / `k^0.29` uses caret notation while the rendered eq below is
`d² ∝ k^β` with a true superscript.** Source `make_deck_figures.py:287`. Cosmetic caret-vs-superscript
mismatch on one slide. **Fix:** render exponents as superscripts in the legend, or accept as-is (minor).

**S27 · appendix sub-caption `MNIST, T=10, r=8, N=2 — θ(α)=(1−α)θ₀+αθ_T` shows literal `_T`.**
Source `slides_appendix.py:163`. Same underscore-T glitch as S5/S4. **Fix:** replace `θ_T` glyph.

**S24 · appendix rank-theorem multi-line block sits in the top half with a large empty lower half;
equations are legible but small relative to available space.** Not broken, just under-scaled for a
2 m read given the room on the slide. **Fix (optional):** bump the `render_lines` fontsize and/or push
the block down to use the empty lower third.
