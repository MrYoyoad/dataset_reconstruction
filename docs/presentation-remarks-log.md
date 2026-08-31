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
10. **Mac-side v2 reconciliation (2026-08-30):** the Mac session produced a v2 pptx + `deck_audit_2026-08-31.md`. Ported into the
    canonical generator: S5 axis label + "leakage does not follow it"; S15 g₀ row ⚠ "n=24 indeterminate"; S19 lead admits the
    s=1.39 same-digit swap; S20 title "'leakage', provisionally". Resolved from cluster data (told to the Mac): arm-B bars are
    the K=100 read (K=50 ≈ 8, K=200 ≈ 46; null ≈ 0 at every K) → panels now labelled K=100 and notes call the absolute a
    fixed-K relative statistic; R² is 0.76 (JSON), 0.85 is a stale STATUS line; q_eff 156/160 is S=1280 = S=640 (≥ 4·Nk),
    r_J=160; faces = data/faces/face1–3.jpg, one person, consent line to be stated by the author; the N=4 DI grid's "8"
    slot collapsed onto the 0 (corr 0.67 with the 0, −0.16 with the 8) → S7 lead says "3 of 4 recovered". NOT ported: the
    Mac's strapline-only-on-S3/S16 choice (this generator keeps one policy: every leakage-number slide).

11. **What the hand-finished v20 changed in QUALITY (2026-08-31, user + local Claude) — analysed from the imported spec.**
    Nine deltas against the generator's 29-slide build, all of which are now rules (see `scripts/deck/SLIDE_CONTRACT.md`):
    1. **Setup slide before result slide.** New slides define the object first: "What we perturb, and what we watch"
       (the N×k nudge experiment behind every part-3 number), "The base gradient: what exactly we compute" (θ₀, W₀,
       one number per image), "One LoRA step keeps only part of the full gradient" (H vs what the adapter can write).
       My build had these only in speaker notes — the audience saw the number before the object.
    2. **A "reading the plot" block on every non-obvious figure.** e.g. slide 15: "each point: the share of private
       directions that clear the noise floor at ε = 1; blue is the binary task, orange the 10-class"; slide 21: "each
       dot is one private image: its base gradient (across) against how much the adapter moved for it (up)".
    3. **Uncertainty as a titled block IN the body, not a footnote.** Slide 21 carries "how sure are we? not yet —
       n = 24; the two predictors are themselves correlated, so this is not a clean contest; no paired test on the
       difference." That is stronger than my notes-only caveats and it is what a theorist checks first.
    4. **Claim-style titles that name the mechanism.** "The asymmetry is not rarity — it follows the image / base
       gradient"; "Same images — but the adapter keeps far less of each"; "Near-duplicates are nearly invisible;
       sensitivity rises with distance". Mine were object labels ("What matters is the image itself: class identity").
    5. **Split dense slides.** Direct inversion → 3 slides; ladder → ladder + "the valley: how far a swap must travel";
       full-FT-vs-LoRA → mechanism slide + result slide. One idea per slide, enforced by splitting rather than shrinking.
    6. **Answer the supervisor's own input explicitly.** A dedicated slide on the paper he sent (SimuDy — "same
       primitive, different question"), placed right after the direct-inversion slides.
    7. **TOC as ask → answer → where.** "five asks, five answers — and one new instrument", with the slide number
       each answer lives on.
    8. **Appendix carries method honesty, not just formulas.** "first, the ruler had to be honest" (the winner's-curse
       story as a narrative) and "why these knobs" (every knob grid-searched, gate-checked or swept, so no headline
       rests on one setting).
    9. **A real closing slide** ("Thank you — let's talk about any of it") that lists what the appendix holds.
    ACTION TAKEN: rules 1–9 written into the slide contract; `add_reading_block()` and `add_caveat_block()` added to
    `deck/helpers.py` so the patterns are one call; the notes template now requires a "READING" line for figure slides.
    ONE THING TO CONFIRM: slide 1's title is now "More Work" (subtitle "LoRA adapters, private images, and what we can
    measure") and Gal's name was dropped from the byline — intentional, or a working title left in?

12. **Content/honesty audit of the final deck (2026-08-31, sibling audit, docs/sessions/v21_audit_content.md).** Applied to the
    spec -> `figures/supervisor_meeting_2026_08_31_v24.pptx`: the gallery/ceiling slide now says the reconstructions use ORACLE
    coefficients (the code path is `compute_known_coefficients`, i.e. c_i from the true images - it was labelled "free c_i",
    which broke the project's standing oracle-vs-realistic rule); the SimuDy slide no longer claims the reframe was "agreed"
    (proposed; it is decision 1 on the close slide), says "the optimiser produced recognisable images" rather than "the decoder
    worked" (SimuDy has no decoder), says "known - or grid-searched - recipe", and attributes the ~22 GB / ~15 h to their
    120-image CIFAR cell; the direct-inversion title drops "works" and the lead states the known-recipe upper-bound scope plus
    both bars; the estimator slide says "+44% (the 2-way estimator)" instead of "(retracted)" (it was a bug, not a withdrawn
    claim); notes: "World A is proven" -> "measured/placed (scoped, local)", the ViT-faces numbers now quote STATUS job 976038
    (0.60/0.67/0.71) and flag that the figure's printed 0.38/0.26/0.52 use another convention and that its columns say
    "Person 1/2/3" while the subject is one person, and slide 14 carries the standing caveat that absolute q_eff counts are
    provisional pending the bias-corrected re-run (quote the differences 23 -> 13 -> 0). The "3 of 4" direct-inversion claim is
    now traceable: `results/direct_inversion/n4_slot_correlations.json`.
    STILL OPEN FOR THE USER: (a) slide 1 title "More Work" + byline without Gal; (b) the ViT-faces figure's column titles.
