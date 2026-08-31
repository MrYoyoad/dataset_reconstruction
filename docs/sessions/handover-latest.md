# Handover — 2026-08-30 16:12

## State
Branch `step1-activation-rescore-retrieval`. The supervisor deck for the **2026-08-31 meeting is BUILT and committed**
(commit 9d13e6b): `notes/supervisor_meeting_2026_08_31.pptx` (28 slides, 4.1 MB; gitignored) + copy
`figures/supervisor_meeting_2026_08_31_v1.pptx`. Flow = answers to Gal's May asks (Aug 23–29 crux runs, DI wall, ceiling)
→ theory chain (measurement system → rank → spectrum → noise floor) → the secret-swap instrument (d², 3-way cross-fit,
pre-registered arm table) → battery results → three worlds + decisions → 5 appendix slides. Every slide has speaker
notes (WHAT / WHY THIS FUNCTION / WHY REPRESENTATIVE / GAL-ASK / CAVEATS / PROVENANCE).

## Done this session
- Two sibling audit rounds integrated (content: numbers/honesty/math/clarity; visual: layout/figures/consistency/equations); deck rebuilt twice; findings in docs/sessions/deck_audit_*.md; visuals page https://claude.ai/code/artifact/263fb1e8-bd26-4f1c-b6df-f34a9c8a8197
- Plan (approved after two reframes) → `scripts/deck/` modular python-pptx generator (config, helpers incl. set_notes,
  mathtext eq_render, six slides_*.py modules built by parallel sub-agents, orchestrator with chunked spire audit renderer).
- `scripts/deck/make_deck_figures.py` → 14 clean figures in `figures/deck_2026_08_31/` from the same result files as the
  analysis generators (NEW plots for arms B/C/D/E, null-diag, ViT, d*, atlas 2-panel).
- Docs: STATUS.md (top entry), LESSONS_LEARNED.md (deck-generator lessons), CLAUDE.md (deck generator section),
  NEW `docs/presentation-remarks-log.md`.

## Overnight 2026-08-31 (this session, after the deck)
- Free-c showcase sweep (jobs 323866/323867/336206/341742; hgn29 excluded) → figures/recon_showcase, v1/v2 zips sent; v3 waits
  for the MNIST T=10 full-FT row (job 341742) — then `make_recon_showcase.py --tsweep` (rec env), zip, delta visual check (yoado-90).
- Generator build (29 slides) has slide 8 = A₀=0 span-leakage. The PRESENTED deck is the hand-finished lineage: v20 -> imported to
  figures/deck_v20_spec/ -> v24 (= v20 + six speaker notes + the content-audit corrections). The span slide is NOT in v24.
- Sibling audits: docs/sessions/{showcase_audit_method,showcase_audit_visual,gallery_recovery_audit}.md.

## Next step(s)
- Mac-side v2 pptx exists (Mac: supervisor_meeting_2026_08_31_v2.pptx); the CANONICAL build is the generator here — its content fixes were ported (commit below); the Mac copy differs only in strapline placement. Present from whichever, but rebuild from scripts/deck/ for any further edits.
- Present tomorrow. If Gal gives slide feedback: log it in `docs/presentation-remarks-log.md`, edit the relevant
  `scripts/deck/deck/slides_*.py`, rebuild with `python scripts/deck/build_deck_2026_08_31.py --render <dir>`.
- Optional polish: slide 7 (DI stack) has a wide gap between the ten-image rows; slide 18 tag wording
  ("your ask: direct inversion → KKT?") could be clearer.
- Unchanged open science items (from previous handover): activation crux dataset-dependence (flowers band), g₀ canonical ρ
  + USPS counterexample, instance-level atlas zoo (the --same_digits run never actually ran), SimuDy reply sent/unsent
  contradiction between thesis_scientific_summary.md and next_experiment_plan.md.

## Open threads / gotchas
- spire free tier renders only 10 slides per file → renderer splits into chunks; previews carry an "Evaluation Warning".
- Default python3 lacks scipy; the atlas figure uses an inline numpy re-implementation (do NOT import atlas_analyze).
- Audit banned-strings: 0/40, ‖ΔW‖/‖W₀‖, 0.226, 1.07, ssim_norm 0.6x, confirmed/settled (except "settled on your side").
- `.tmp_pptx/` (May generator) is gitignored; the new generator lives in tracked `scripts/deck/`.

## Pointers
- Build: `python scripts/deck/make_deck_figures.py && python scripts/deck/build_deck_2026_08_31.py --render /tmp/deck_render`
- One module: `python scripts/deck/preview_module.py deck.slides_measure /tmp/prev`
- Contract: `scripts/deck/SLIDE_CONTRACT.md`; plan: `.claude/plans/help-me-plan-a-keen-wigderson.md`
- Story sources: `notes/thesis_note_v2.md`, `notes/identifiability_feasibility_revision.tex:41-153`,
  `notes/dataset_sensitivity_program_plan.md` (§II rules, §III table), `notes/meeting_prep_2026-08-31.md`.
