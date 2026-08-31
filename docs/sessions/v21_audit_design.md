# v21 Deck — Design / Flow / Completeness Audit

- **Date:** 2026-08-31
- **Auditor:** yoado-40 (per yoado-6c request)
- **Audited:** `figures/supervisor_meeting_2026_08_31_v21.pptx` (37 physical slides), read via the imported spec `figures/deck_v20_spec/deck_spec.json`. Geometry checked numerically in EMU; no rendering. Read-only — no edits to deck, spec, or generator.
- **Method:** programmatic pass over all 37 slides / every shape — off-slide overflow, footer-zone intrusion, text-on-text overlap, text-density overflow, font sizes, page-number continuity, TOC-vs-actual, chip/label/colour consistency, and the 8 no-notes slides.

Slide numbers below are **physical** slide indices (1–37). The deck's own footer uses a **logical** numbering `N / 35` in which physical S7/S8/S9 all read "7 / 35" (S8, S9 are figure continuations of S7), so 37 physical = 35 logical pages. The TOC uses the logical numbers.

---

## BLOCKERS (would embarrass in the meeting)

**Slide 1 · Title/byline · Placeholder-looking title + supervisor's name dropped · Fix before presenting.**
The title (TextBox 1, 40 pt Cambria) reads **"More Work"**. The byline (TextBox 4, 16 pt) reads **"Yoad Oxman · 2026-08-31"** only. Gal Vardi appears nowhere on S1 (his name survives only inside speaker notes on S2 "sent by Gal" and S10 "Gal asked for"). For a supervisor meeting whose stated stakes are securing supervision, a 40 pt "More Work" reads as a leftover placeholder, and a dropped co-name reads as an editing accident, not a deliberate choice. Give it a real title and restore the intended byline (or confirm the solo byline is deliberate). Subtitle ("LoRA adapters, private images, and what we can measure") is fine.

---

## SHOULD-FIX

**Slides 3, 17, 20, 21, 22, 28, 33 · Geometry · "standard methods" footnote band sits below the 7.12 in safe line.**
Each `MethodNote` box spans top 7.076 in → bottom **7.283 in** (6,660,000 EMU), i.e. entirely inside the "nothing below 7.12 in except footer/page-number" no-go zone. Consistent across all 7 slides (deliberate footnote band), but at 7.28 in it is the element most exposed to projector overscan / bottom-edge clipping. Either raise the band above 7.12 in or accept the clipping risk knowingly. (`DstarNote` on S23 is fine — it sits mid-slide at 4.79 in.)

**Slides 11, 13, 21, 23, 26, 27 · Completeness · content-heavy slides with NO speaker notes.**
All six are dense, non-figure slides that carry argument, and each needs a note:
- **S11 "The paper you sent: SimuDy"** — the paper the supervisor personally sent (Tian et al., ICLR 2025). Highest-stakes of the six. Note must carry the "same primitive, different question — why this direction is not obsolete" framing: their result = our known-recipe ceiling; the released-adapter question is what we add.
- **S13 "What we perturb, and what we watch"** — the experiment-setup diagram behind all of part 3. Note must narrate N=10, k=8 → 80 private directions, the nudge-one-coordinate protocol, and that a rank is a yes/no recorded-count (not a magnitude).
- **S21 "Which picture does the adapter follow?"** — results + honesty. Note must state ρ=+0.78 (g₀) vs +0.51 (λ), n=24, that the two predictors are themselves correlated, and the "not yet sure" caveat so it isn't over-claimed live.
- **S23 "The valley (d\*)"** — defines a new object. Note must give the d* definition (swap distance at which the adapter notices), the identity-swap=0 sanity check, and that it has no standard LoRA-literature name.
- **S26 "One LoRA step keeps only part of the full gradient"** — theory + equation (H, η). Note must state the one-step tangent picture and "adapter = filtered copy of the full gradient."
- **S27 "Same images — adapter keeps far less"** — dual-panel result. Note must interpret ρ=+0.94 and "~5× less energy per image," tying the two panels together.

**Slides 29 & 37 · Completeness · no notes, but genuinely do not need them.** S29 is a "Thank you" closer (a one-line landing message would be nice-to-have, not required); S37 is a self-explanatory appendix knob-rationale table shown only if asked. Both presentable as-is.

---

## NITS

**Slide 3 · Geometry · legend box likely tight.** `TextBox 10` holds 221 chars in a 3.63 × 0.80 in box at 12 pt (est. capacity ~174 chars, ratio 1.27) — the "red = kinked, green = smooth; diamonds = oracle… T=1…" legend. Word-wrap is on, so it may auto-shrink, but it is the one box where character count exceeds estimated capacity; worth an eyeball at render.

**Slide 31 · Geometry · two boxes nearly touch at the bottom.** `TextBox 9` (4 bullets, 269 chars) spans top 5.30 → bottom 6.80 in; `ScalarScope` starts at top 6.26 in (bottom 6.80). If the 4 bullets wrap to fill their box they meet ScalarScope. Tight, not a certain collision.

**~15 slides · Geometry · "theme/validity link" caption band at 7.04 in.** The single-line `TextBox` captions ("theory link…", "validity link…") sit at bottom 7.04 in — inside the 6.85–7.12 in footer margin but above the 7.12 no-go line, and consistent across slides. Cosmetic.

**Deck-wide · Consistency · 9 pt footer captions.** The footnote/caption band runs at 9 pt (below a 10 pt caption floor) on nearly every slide. Consistent convention, legible enough, but below the guideline. Sub-9 pt runs (7–8.5 pt on S14/S30/S32/S34/S36) are math subscripts (`J_SNR`, `q_eff`) — legitimate, not body text.

**Consistency · part labels start at "part 2."** Part labels appear on S12 ("part 2"), S16 ("part 3"), S18 ("part 4"). Part 1 (the five answer slides 3–10) carries no "part 1 ·" label — it uses the blue "your ask:" chips instead. Minor grammar asymmetry; likely intentional (chips = part 1).

---

## Clean dimensions (checked, no problem found)

- **Off-slide overflow:** none. No shape's right edge exceeds 12,192,000 EMU or bottom exceeds 6,858,000 EMU. (S9's `Picture 1` reaches exactly 7.50 in — an intended full-bleed image, edge-exact, not overflow.)
- **Text-on-text overlap:** none real. Every substantial-overlap hit (S3/4/5/6/7/10 title↔chip, S6/S11 card↔caption, S12/16/18 title↔part-label) is the chip-in-title / caption-in-card design pattern — the small box nests inside the large box's bounding rectangle but does not visually collide (title text is short and left-aligned; captions sit in the card's lower band). Only S31 (above) is borderline.
- **TOC (S2) ↔ actual slides:** matches. Using logical page numbers: item 1 "slides 3–4" → S3–S4 ✓; item 2 "slide 5" → S5 ✓; item 3 "slide 6" → S6 ✓; item 4 "slide 7" → S7 ✓; item 5 "slide 8" → S10 (logical 8) ✓; "a new instrument, slides 10–26" → S12–S28 ✓; "the paper you sent (SimuDy), slide 9" → S11 (logical 9) ✓. No drift.
- **Page-number continuity:** clean 2/35 → 35/35 across physical S2–S37, with S7/8/9 sharing "7 / 35" by design. (S36's `392821 / 390026` and `229722 / 237301` are provenance data values on the appendix slide, not page numbers — false-positive on the regex.)
- **Chip / part-label grammar:** consistent. "your ask:" chips all blue `1F4E79`, fill none, top 0.50 in, right 12.70 in (S3,4,5,6,7,10). Part labels all gray `555555`, same position (S12,16,18).
- **Flow arc:** delivers answer → instrument → results → appendix. Part 1 answers (S3–S10) → the sent paper SimuDy (S11) → part 2 theory (S12–S15) → part 3 how-we-measure (S16–S17) → part 4 experiments/meaning (S18–S28) → close S29 → appendix S30–S37. No orphan slides; no slide doing two jobs.

---

## By-severity count

- **Blockers:** 1 (S1 title + byline)
- **Should-fix:** 2 groupings — footnote band below 7.12 in (7 slides); missing notes on 6 content slides
- **Nits:** 5 (S3 legend, S31 tight pair, caption band at 7.04 in, 9 pt captions, part-1 label asymmetry)
- **Clean:** off-slide overflow, real overlaps, TOC match, page-number continuity, chip/label grammar, flow arc

## Verdict

Structurally sound and internally consistent — flow, TOC, numbering, and visual grammar all hold up. One true blocker: the **"More Work" placeholder title with Gal's name dropped from the byline** on slide 1 must be fixed before the meeting. After that, add speaker notes to the six argument-carrying slides and decide whether the 7.28 in footnote band is a clipping risk on the room's projector.
