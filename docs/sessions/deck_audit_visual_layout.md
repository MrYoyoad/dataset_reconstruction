# Deck VISUAL / LAYOUT / TYPOGRAPHY audit — supervisor_meeting_2026_08_31.pptx (28 slides, post-fix render 16:25)
Auditor: yoado-23, for yoado-72. Lens: occlusion/clipping/wrapping/footer-safe/font-size/alignment (guardrails O1-O10, T1-T7, S1-S7).
Geometry: 13.333x7.5in; content bottom CB=6.85" (~657px @96dpi); footnote line 6.45" (~619px); page-num floor 7.12" (~683px);
footer 7.15"; H-margin 0.55" (L~53px / R~1227px). Watermark = Spire preview artifact (confirmed absent from pptx XML), ignored.

## Verdict
Layout is clean and consistent overall: uniform title size/weight, consistent right-column headers, consistent "your ask" chips
(S3-S8), consistent footer + page numbers. The 16:25 fixes all landed. ONE real clip (S13), one crowding (S7), and a
scope-line placement issue that is both low (past the footnote line) and inconsistent. Everything else is minor/borderline.

## RANKED DEFECTS
1. [HIGH — CLIP] S13 blue callout text OVERRUNS the right margin and is truncated. It reads
   "=> a ceiling on detecting the change — for every atta" [cut] / "reconstruction included" — the word "attacker" is clipped
   at the slide edge (text extends past x=1227 to ~1280). Introduced by lengthening the old (fitting) "=> a ceiling on every
   attack, reconstruction included". FIX: revert to "=> a ceiling on every attack — reconstruction included", or reduce the
   callout font ~2pt / set its text-box right edge to <=1227 so it wraps before the margin.
2. [MED — FOOTER/CROWDING] S7 the lead now wraps to 2 lines ("known-recipe upper bound: ... — we reframed)"), pushing the
   bottom "recovered" (ten-image) row down to ~660px — at/just past the CB=6.85" (657px) content-bottom guardrail (O4).
   FIX: shorten the lead to one line (e.g. "the map inverts; the joint inversion is the bottleneck (SimuDy's primitive,
   reframed)") to lift the bottom row back above 6.85".
3. [MED — O4 + CONSISTENCY] The italic scope line "every leakage number here is a lower bound on the weakest attacker
   (prior-free, adapter-only, per-image)" sits at y~6.97" (~669px) on S3/S16/S17/S18/S22 — BELOW the O4 6.45" footnote safe
   line (~0.5" into the footer approach; ~24px above the footer text, no collision but crowded). AND it is ABSENT on other
   content slides (S4/S9/S19/S20/S21...). FIX: (a) raise the line to <=6.45"; (b) decide one policy — put it on every
   leakage-number slide or drop it to S1 + a footer motif — currently its presence is arbitrary.
4. [LOW — O1] Right-column card boxes' right edge sits ~3px past the 0.55" margin (S11 A/B/C, S14 three boxes, ~1230 vs
   1227px). Cosmetic; verify n*col_w + gap <= CW=12.13" so nothing clips on a real projector.
5. [LOW — S6/whitespace] S23 "decisions" box (red) holds 4 bullets then ~40% empty lower half. Tighten box height (or add
   the 5th decision) so the card doesn't read half-empty next to the full three-worlds column.
6. [LOW — O4 edge] S26 invariance table bottom (~658px) and S7 bottom image row (~660px) land right at CB=6.85"; both are
   within ~1px of the guardrail. Give ~0.1" headroom so a slightly different renderer doesn't push them into the footer band.
7. [LOW — TYPOGRAPHY] Smallest text = S14 and S21 sub-plot captions (~9-10pt italic) and the S28 provenance table (~11pt,
   15 rows). Readable on-screen; verify >=10pt captions / >=11pt table for back-of-room projection (T1). Appendix-acceptable.
8. [LOW — WHITESPACE] S24 has a vertical gap between the last formula ("leakage <= min(...)", ~435px) and the bullet list
   (~520px); pull the bullets up ~0.5" or center the block. Appendix, minor.

## NON-DEFECTS / CONSISTENCY CHECKS THAT PASS
- Titles: uniform size/weight/colour; longest is S20 ("... so we may say 'leakage'") — fits one line, ~1157px, no clip.
- Leads: 1-line except S7/S18/S20 (2-line); only S7's causes downstream crowding (#2).
- S22 fix #2 (+0.989 line) fits inside the atlas box without overflow. S8 fix #7 (3-line face caption) fits above footer.
- S10 fixes #3 (r=10 = sqrt(K.N) 10-class threshold; binary ~sqrt(N)) and #10 (title) render correctly, annotation inside plot.
- S19 fix #4 ("rises with swap distance (mid-ladder wobble = cross-exemplar noise)") fits in 3-line caption.
- Footer + page number present and aligned on all content slides (S1 title correctly omits page number).
- No overlapping shapes, no shape below the 7.12" page-num floor, no clipped table columns (O2/O9) anywhere.

## PRIORITY FOR MONDAY: fix #1 (S13 clip) is the only must-fix (a visibly truncated word). #2 and #3 are polish that a
careful theorist would notice. #4-#8 are optional.
