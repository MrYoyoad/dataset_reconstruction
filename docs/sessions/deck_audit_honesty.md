# Deck honesty / scoping audit — 2026-08-31 supervisor deck

- **Date:** 2026-08-30
- **Auditor:** yoado-ba (per yoado-72 request)
- **Audited:** `scripts/deck/deck/slides_{answers,theory,measure,results,close,appendix}.py` — all 21 story slides + 5 appendix slides, both on-slide text and `set_notes(...)` speaker notes. Checked against `notes/thesis_note_v2.md` (stance + §5), `notes/dataset_sensitivity_program_plan.md` (§II reporting rules), `notes/meeting_prep_2026-08-31.md`, and the `LESSONS_LEARNED.md` retractions.
- **Read-only:** no slide modules or pptx were modified.

## Headline

The deck is honest at the level of its speaker notes: caveats, small-n grades, retractions (a whole retractions panel, A4), ceiling-vs-attack labels, Jang √N, sub-linear β, and the weakest-attacker stance are all present and correct **in the notes**. The systematic issue is that several load-bearing scope qualifiers live *only* in the notes and are **absent from the visible on-slide text**, so an audience reading slides (not hearing the notes) sees unscoped leakage claims. No BLOCKER-level false claim was found; the findings are should-fix scoping gaps and nits.

---

## Findings — worst first

### 1. [should-fix · check 2] Weakest-attacker / lower-bound scope is absent from every visible story slide
- **Slides:** all leakage-bearing story slides — crux (slide 3), g₀ (slide 17 `r3_g0`), class (slide 16 `r2_class`), atlas (slide 21 `r7_atlas`), knobs (slide 15 `r1_knobs`). On-slide text.
- **Claim:** The stance *"Every leakage number bounds the WEAKEST attacker (prior-free, adapter-only, per-image)"* appears **only** in the title-slide `set_notes` (`slides_answers.py:111`). The only visible lower-bound scoping anywhere in the deck is one line on the measure-plan slide (`slides_measure.py:310`: "magnitude = lower bound at fixed K") and appendix A2 rule 3. Slides 3/15/16/17/21 show leakage magnitudes ("kinked ≈ 5× smooth", "leak ~3× more", "β≈0.23", "+0.989") with no visible "lower bound / weakest attacker" scope.
- **Issue:** Check 2 requires every leakage number to be scoped as a LOWER bound. Mitigated because the deck is presented live and the presenter speaks the stance — but a reader of the slides alone sees unscoped numbers.
- **Suggested fix:** Put the stance on a visible surface — e.g. a one-line stance under the title slide's subtitle ("every number is a lower bound on the weakest — prior-free, adapter-only, per-image — attacker"), or a persistent small footer token on the results slides. This single change resolves most of check 2.

### 2. [should-fix · check 10] "a ceiling on every attack, reconstruction included" over-reaches on the same slide that says it bounds adapter space, not pixels
- **Slide:** slide 11 `slide_m2` ("Why this function: the best any attacker can do"). On-slide text (`slides_measure.py:187`).
- **Claim:** blue callout "⇒ a ceiling on every attack, reconstruction included" — while the lead (line 146) and the amber caveat chip (line 200) say d² "bounds adapter space, not pixels / bounds the adapter-space change Δμ, not pixel error."
- **Issue:** The two visible statements are in tension. d² is a necessary-condition ceiling on *detecting the dataset change* (and thus on recovering the change), but "reconstruction included" reads as bounding pixel-level reconstruction, which the caveat chip explicitly denies (the pixel bound needs the composed Fisher bridge, "scheduled, not built" per notes). An NTK theorist will pick at this.
- **Suggested fix:** "⇒ a ceiling on **detecting the change** — every attack, reconstruction included" (ties the strong word to detection, consistent with the caveat chip), or drop "reconstruction included" from the callout and leave it to the notes.

### 3. [should-fix · checks 7/10] Direct-inversion slide shows recovered images with no visible known-recipe / upper-bound label
- **Slide:** slide 7 `slide_direct_inversion` ("Direct inversion: works at N=4, superposes at N=10"). On-slide text.
- **Claim:** Title "works at N=4"; grids of private-vs-recovered digits shown. The "known recipe (lr, T, full batch) = best-case upper bound" scope is **only** in the notes (`slides_answers.py:427`).
- **Issue:** DI requires the known training recipe — it is a best-case / strong-attacker result, the analogue of the ceiling on the neighbouring slide 8 (which *is* labelled "this is the CEILING — not the adapter-only attack"). Slide 7 has no equivalent visible scope, so "works at N=4" can read as the realistic attack succeeding. (Check 9 — SimuDy — is satisfied: the lead visibly says "SimuDy did the same primitive — we reframed".)
- **Suggested fix:** Add a visible scope tag mirroring slide 8, e.g. lead → "...known-recipe upper bound; the joint inversion is the bottleneck (SimuDy did the same primitive — we reframed)", or a small "best-case / known recipe" chip.

### 4. [should-fix · check 4a] g₀ slide: the n=24 INDETERMINATE grade is not visible; "predictable" is stated flatly
- **Slide:** slide 17 `slide_r3_g0` ("…and its base gradient: predictable from the public model"). On-slide text.
- **Claim:** Title asserts "predictable from the public model"; lead "an attacker ranks which images will leak before seeing the adapter — from θ₀ and the candidate image alone." The pre-registration grade (n=24 INDETERMINATE, ρ=+0.777, CI [0.53,0.91]; n=12 gave +0.857 but is not canonical) is only in the notes.
- **Issue:** Check 4a: n=24 is indeterminate by pre-registration and must not be presented as established. The visible card does hedge correctly ("strong where g₀ is small, saturates where it is large"), but the title + lead read as an established predictor.
- **Suggested fix:** Soften the title to "…and its base gradient: predictive at low g₀, saturating" or add a visible "n=24, indeterminate; strong at low g₀" note; keep the honest card.

### 5. [nit · checks 3/1] r5 title licenses the word "leakage" off an n=12 spot-check; "leak/leakage" is used on slides 15–18 that precede it
- **Slide:** slide 19 `slide_r5_hgate` ("Detection tracks behavioural memorisation — so we may say 'leakage'"). On-slide text; and the "leak" usages on slides 15–18.
- **Claim:** The H-gate title licenses "leakage"; slides 15/16/17/18 (`r1`–`r4`) already say "leaks / leak / how much it leaks" earlier in the running order. The A2 rule-6 wording is "'sensitivity' until the H gate **closes**", and the notes state the full H gate at scale is "still required — open" (only a spot-check, n=12, passed).
- **Issue:** Minor tension: the visible slides use "leak" before the gate slide, and the gate is only spot-check-passed, not closed at scale. The "so we **may** say" hedge and the honest r5 notes largely cover it.
- **Suggested fix:** Optionally footnote r5 ("spot-check n=12; full gate at scale pending"), or accept as-is given the "may" hedge.

### 6. [nit · checks 1/10] "proven unbiased" overstates the null-diag evidence
- **Slides:** `slide_m3` notes (`slides_measure.py:252`, "proved the fixed one unbiased") and `slide_a2` notes (`slides_appendix.py:111`, "proven unbiased on signal-free data"). Speaker notes.
- **Issue:** The support is empirical (null-diag reads ≈0 at every K); "proven unbiased" is a stronger word than the evidence (observe-don't-conclude).
- **Suggested fix:** "consistent with unbiased — reads zero on no-signal data at every K."

### 7. [nit · check 1] "A is proven by the ruler" on the close slide
- **Slide:** slide 22 `slide_c1_close`, on-slide text (`slides_close.py:63`) and the world-A card "→ proves A when it happens".
- **Issue:** "proven / proves" is conclusion-language, but here it is methodological (what the ruler *would* do when World A occurs) and the visible verdict chip reads "not our case so far", so it does not assert a current conclusion. Acceptable; flag only for word choice.
- **Suggested fix:** If softening: "the ruler *would establish* A when it happens (attack-independent, scoped)".

---

## Per-check result

- **1 — observe-don't-conclude:** No visible "confirmed / settled / demonstrates / establishes" on any story-slide text. Visible "proves/proven" appears only on the close slide as a *methodological* statement with an honest verdict chip (finding 7). "confirms/confirm" appears only in notes (arm-D context, atlas rebuttal) and is factually scoped there. "proven unbiased" in notes = finding 6. Essentially clean on visible text.
- **2 — weakest-attacker scope:** VIOLATED as a visible-surface gap — finding 1. The scope is correct everywhere in notes.
- **3 — "leakage" word / H-gate licensing:** Disciplined — the reporting rule ("no word 'leakage' before the H gate") is visible on slide 14, rule 6 is visible in A2, and slide 19 explicitly ties the word to the H gate. Minor ordering tension = finding 5.
- **4 — small-n caveats:**
  - g₀ n=24 indeterminate: **partially violated on-slide** — finding 4 (correct in notes).
  - full-FT-vs-LoRA valley: CLEAN — slide 20 lead says "at about the same resolution" (not "narrower"); notes give geomean 1.02 / median 0.86 / target-dependent / "never quote the arithmetic mean".
  - activation crux two-cluster/not-monotone/MNIST-only: CLEAN in notes; visible text uses a two-cluster framing ("kinked ≈ 5× smooth", red/green), not a monotone-smoothness law. OK.
  - ViT n=3: CLEAN (notes state 3 targets; slide claims only "a real ViT").
  - atlas ≥content-level, same-digits zoo never ran: CLEAN and exemplary — slide 21 lead visibly says "which digits… ; which exemplar is open", card says "content-level: which digits — not which exemplar", notes state the instance-level zoo "NEVER actually ran".
- **5 — duplication wording:** CLEAN. Visible text uses "sub-linear", "β≈0.23≪1", "sub-linear, β≈0.2". "duplication-invariant limit" appears only in notes, correctly describing the max-margin limit they do NOT reach.
- **6 — Jang bound:** CLEAN. Visible slides carry no bare "r ≳ N"; slide 6/t2 notes and A4 state r ≳ √N and mark the K·N form as "OUR extrapolation, not Jang's bound"; A4 lists "Jang bound r ~ N → it is r ~ √N" as a retraction.
- **7 — full-gradient gallery + faces = ceiling:** CLEAN and strong — slide 8 title + lead both say "the CEILING (true weight change) — not the adapter-only attack"; faces are on that slide; notes add "Do not read this slide as the adapter attack succeeding." (The neighbouring DI slide 7 lacks an equivalent label — finding 3.)
- **8 — 0/40 absent, not replaced by implied positive adapter-only reconstruction:** CLEAN. 0/40 and ssim_norm 0.61 are absent (A5 notes explicitly forbid putting them on a slide); slide 8 disclaims adapter-only success; the atlas is scoped as content-level detection, not pixel reconstruction. The only residual is the DI slide's missing scope (finding 3), which is a known-recipe result, not an adapter-only pixel claim.
- **9 — SimuDy overlap on DI slide:** CLEAN — visible on slide 7 lead ("SimuDy did the same primitive — we reframed"); notes give the full SimuDy citation and reframe.
- **10 — general overclaim (NTK-theorist lens):** Two visible items — finding 2 ("reconstruction included") and finding 3 (DI as realistic attack). Theory statements are otherwise well-hedged: t2 notes call σ_min(J) collapse "the note's strongest empirical claim and still to be tested"; A1 scopes the rank theorem as "necessary, not sufficient" with the frozen-known-G caveat.

## Clean slides (no findings)

Slides 1–2 (title, what-changed), 4 (feature-stability), 5 (anchor), 6 (mechanism), 8 (more-data ceiling), 9–10 (t1, t2), 12–14 (m1, m3, m4), 15 (knobs), 16 (class), 18 (ladder), 20 (beyond-MLP), and appendix A1/A3/A4/A5 pass all ten checks on both surfaces. A4 (retractions panel) and A5 (provenance) are model honesty artifacts.

## Counts by severity

- **BLOCKER:** 0
- **should-fix:** 4 (findings 1–4)
- **nit:** 3 (findings 5–7)

Net: the deck is honest in substance; the fixes are about promoting four scope qualifiers from the speaker notes onto the visible slides (weakest-attacker stance, DI known-recipe label, g₀ indeterminate grade) and softening one visible callout ("reconstruction included").
