# Presentation remarks log

Running log of every remark / request the user gives about slides (mandated by CLAUDE.md). Newest first.

## 2026-08-30 — supervisor deck for 2026-08-31 (`scripts/deck/`)

1. **Format:** pptx (not the caption-heavy PDF deck); white / Cambria theme, same look as the 2026-05-14 v18 deck
   (deviation from the dark `style_guide/pptx.md` template is deliberate — continuity across the two decks).
2. **Words:** barely any — figures, plots and crystal-clear examples; the critical formula ON the slide; derivations
   and caveats in speaker notes; major-but-non-critical math in a short appendix.
3. **Numbers:** only critical / essential ones on slides (≤ 2 per story slide); everything else in notes.
4. **Content:** drop the 0/40 adapter-only reconstruction result; show the positive reconstruction examples instead.
5. **Flow (rejection of first plan):** the secret-swap / whitened-sensitivity program ("hide an image, whiten the seed
   noise, detect it") is the biggest leap and must be the centerpiece, together with the pre-registered plan that drove it.
6. **Flow (rejection of second plan):** start with answers to Gal's May questions using the recent experiments → the
   theoretical thoughts that led to the sensitivity tests → the choice of how to measure → experiments + meaning +
   results. Plain, "real" titles — take Gal through the ideas, no showmanship. Tight, plot-heavy sit-down talk.
7. **Process:** split the build across sub-agents (one per deck part), parallel.
8. **Sibling audit round (2026-08-30, user: "ask siblings to audit"):** four parallel audits — numbers (yoado-3a),
   honesty/scoping (yoado-d1), math (yoado-ef), clarity/flow (yoado-23) — findings in `docs/sessions/deck_audit_*.md`.
   Applied: visible weakest-attacker scope line on the title + leakage slides; "ceiling on detecting the change";
   DI slide labelled known-recipe upper bound; g₀ title/lead carry the n=24 indeterminate grade; H-gate lead notes the
   n=12 spot-check; "proven unbiased" → "consistent with unbiased"; ΔW equation linearizes about θ_a; arm-E R² 0.76 (r8);
   +0.989 on the atlas slide; r=10 relabelled as the 10-class √(K·N) threshold; S10 title leads with the conclusion;
   S19 "monotonic" softened; faces caption notes colour as the weakest channel. Deliberately NOT applied: trimming
   S14/S21 density (user wants plot-heavy "meat"); the cut/merge list stays as the short-meeting fallback.
9. **Visual audit round (2026-08-30, user: "ask the siblings to audit visuals"):** four lenses — layout/typography (yoado-23),
   figure legibility (yoado-3a), consistency/first-glance (yoado-d1), equation rendering (yoado-ef) — findings in
   `docs/sessions/deck_audit_visual_*.md`. Applied: S13 callout wrapped (it had clipped after the honesty rewording);
   A/B/C world colours locked across S11/S23 (A red, B blue, C amber) and the decisions card sized to content; S14 single
   critical equation + legend states the null series ≈ 0; in-body chips on S18/S20 folded into card titles (top-right chip
   reserved for Part-1 "your ask" + new gray part eyebrows at the S9/S12/S16 seams); scope line on every leakage-number
   slide at one height; S7 lead back to one line; native-text notation glitches fixed (mathtext axis labels, no bare θ_T,
   superscript exponents); S4 caption names gelu as the smooth outlier; appendix rank block enlarged.
   Not applied: recolouring the kinked series away from red (legends disambiguate; red = kinked is used in every figure).
