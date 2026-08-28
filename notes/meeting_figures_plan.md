# Meeting figures + science-summary — coordinated plan (DRAFT for yoado-18 audit)

**Role split (Gal, via yoado-18):** this session (figures/plots owner) SPECS → yoado-1f (executer) BUILDS
→ yoado-6d (stats) CHECKS the numbers → yoado-18 (auditer) audits clarity + honest scoping.
**Posture (Gal, swarm-wide):** OBSERVE, don't conclude. Every figure caption + summary block says *what we
observe / what this measures / what's open* — never "the result that confirms X." No CONFIRMED/SETTLED/DONE.
**Gal's content priorities:** similar images, dataset SIZE (distinct-image N), shared perturbation.
**Explicitly deprioritized:** duplicate-count / dilution effects.

**Scope frame to carry on every leakage figure (Gal, via yoado-18):** the valley / whitened-sensitivity d²
bounds only the WEAKEST attacker (prior-free, recipe-blind, adapter-only, per-image). It is A measurement
(a detection floor), NOT the reconstruction limit. Four ways past it: priors (Direction-3 SDS), known-recipe
inversion (direct inversion), structural leakage (class / shared-perturbation / N), stronger decoder. So
"0/40 fails baseline" = the adapter-only DECODER/recipe failing, NOT an information wall (identifiability
shows the info is present ⇒ decoder/recipe-limited, not information-limited).

---

## CURRENT MEETING BUILD (Gal's decisions, via yoado-18)
**Build now (data-ready):** F1 (declutter, DONE — this session), F2, F3 (existing-data fixes → yoado-1f)
+ the CRUX in-progress block (final crux figure HELD until free-c ladder 392821 lands).
**F4 (recover-N / dataset-size): FULLY DEFERRED** — Gal: all 3 readings (recover-N / per-image-dilution /
whole-set-leaks-more-at-N) are open and interesting but "we didn't really check it yet — wait until there
are results." A future experiment, NOT a meeting figure (building one now would assert an unobserved result,
against the posture). recover-N is the default re-spec for whenever it runs; all 3 variants stay open.
**F5 (shared-perturbation): SCOPED, compute-timing = Gal's call** — Gal cares about it, but it's a new run
with no data yet; don't build a figure before it has data. Scaffold only (yoado-1f), runs pending Gal's
compute approval (yoado-18 has asked him run-now-vs-defer).

## FIGURE SET (specs)

### F1 — main combined (identifiability + reconstruction)   ·  builder: this session (I own the generator)
- **Science / what it measures:** two axes on one page — (left) identifiability = q_eff|col(J) vs attacker
  budget ε, binary vs 10-class (how many private directions a weakest-attacker can recover); (right)
  reconstruction = decoded-vs-baseline pixel SSIM per gb_e2e cell (can the adapter-only decoder turn that
  geometry into pixels).
- **Data:** `results/jacobian_j1_roundB_*.pth` (identifiability) + `results/gb_e2e_*.pth` (reconstruction);
  generator `experiments/plot_leakage_combined.py`.
- **Fixes:** (a) neutral SHORT title — drop the "HIGH…confirmed…NOT amplified" paragraph; (b) move the two
  dense annotation boxes into the figure CAPTION (keep the panel faces clean); (c) thin the 12 rotated
  x-labels (group or abbreviate); (d) re-tone "confirmed/settled"→"observed"; (e) ADD the weakest-attacker /
  decoder-limited line (0/40 = prior-free/recipe-blind/adapter-only decoder fails, NOT an information wall;
  info present per identifiability). Generator already partly re-toned + decoder-limited line added — yoado-1f
  to finish the title/label/caption cleanup.
- **Observe-caption:** "What we OBSERVE: the weakest-attacker direction-count is high for both bases; the
  adapter-only decoder does not yet clear the trivial pixel baseline. OPEN: whether priors/known-recipe/
  stronger decoder cross it."

### F2 — similarity ladder (SIMILAR IMAGES)   ·  builder: yoado-1f  ·  owner-of-source: similarity_ladder.py
- **Science / what it measures:** how sensitivity s changes as two training images are made more similar
  (distance d ↓) — the "similar images leak differently" story.
- **Data:** `experiments/dataset_sensitivity/similarity_ladder.py` → `figures/similarity_ladder/*.png`.
- **Fixes:** (a) the image columns are NOT ordered by distance d (defeats the s-vs-d story) → ADD a companion
  **s-vs-d LINE plot, sorted by d, with the valley/floor marked**; (b) add an overall title + a d/s legend
  (viewer cannot currently tell what s and d are); (c) **fix `d=nan` on the blur rung** (metric/först
  undefined — yoado-6d to confirm the correct d for blur); (d) LABEL the two rows (two different digit-5
  targets on different s-scales).
- **Observe-caption:** "What we OBSERVE: s vs d as images approach each other; whether s rises/falls into the
  valley. OPEN: mechanism (is it distance or a specific shared structure)."

### F3 — margin_at_scale = WHO leaks (g0 vs atypicality)   ·  builder: yoado-1f  ·  KEEP as the template
- **Science / what it measures:** which images leak — per-image sensitivity vs the margin/gradnorm predictor
  g0, ρ_spearman(sens, g0). This is WHO-leaks, NOT dataset size.
- **Data:** `experiments/dataset_sensitivity/margin_at_scale.py` (n_targets=24, stratified across g0).
- **Fixes:** (a) soften any "confirmed" → "observed"; (b) **RETITLE** so "at scale" isn't misread as
  dataset-size — e.g. "Which images leak: sensitivity vs the g0 predictor (per-image, n=24)".
- **Stats (yoado-6d):** report ρ with the bootstrap 95% CI and the per-stratum sign; state the CI width
  honestly (the MVP was ρ=+0.857 but CI ~±0.4 at n=12 → n=24 tightens it, report the actual CI).

> **Framing for BOTH new figures F4 + F5 (yoado-6d/yoado-18, theory-endorsed):** these are STRUCTURAL
> leakage — what a stronger-than-weakest (structural) attacker gets that the prior-free/adapter-only/per-image
> attacker does NOT. Frame them as **EXTENSIONS PAST the 0/40 weakest-attacker floor, NOT contradictions of
> it.** The caption MUST say "this is leakage beyond the weakest attacker's reach (a different, often easier
> target)" — else a reader hits "0/40 fails reconstruction, yet these leak?" as an apparent contradiction.
> **Guards (yoado-6d):** each new analysis needs its OWN trivial-baseline gate (the analogue of the 0/40
> mean-image baseline — a result at/below its own trivial predictor carries no structural info); and any
> N-sweep correlation must guard against pseudo-replication (report over DISTINCT units, not correlated rows).

### F4 (NEW) — RECOVER N (infer dataset size from the adapter)   ·  builder: yoado-1f  ·  METRIC HELD pending Gal
- **RE-SPEC (yoado-18 audit fix):** the earlier metric ("per-image sensitivity d² vs N") was the per-image
  DILUTION object the swarm DEPRIORITIZED — dropped. The intended question (Gal: "learn the NUMBER of data
  points that was used") is a STRUCTURAL leak: can an attacker INFER N from the adapter, above a structure-blind
  baseline?
- **Science / what it measures:** is there a signal in the adapter (ΔW / A,B) that predicts the training-set
  size N — recovering N as a quantity, NOT how each image's fingerprint scales.
- **Metric (HELD until Gal clarifies (a)/(b)/(c)):** candidate = a predictor of N from adapter features
  (e.g. ΔW spectrum / hard-rank / a small learned regressor) scored as N-recovery error, vs a structure-blind
  baseline (guessing the prior-mean N). NOT per-image sensitivity vs N (that is the excluded dilution). yoado-18
  is flagging the (a) dilution / (b) recover-N / (c) "does the whole set leak more at bigger N" ambiguity to
  Gal; his "number of data points" wording points to (b). **Do not build until Gal picks the reading.**
- **Observe-caption:** "What we OBSERVE: whether N is recoverable from the adapter above a structure-blind
  baseline — leakage BEYOND the weakest attacker's reach (recovering the dataset SIZE is a different, easier
  target than the per-image pixels the 0/40 measures). OPEN: the reading (pending Gal) and the recovery floor."
- **Guards:** own structure-blind baseline gate (recovery must beat guessing prior-mean N); any correlation
  over DISTINCT configs, no pseudo-replication.

### F5 (NEW) — recover the SHARED PERTURBATION   ·  builder: yoado-1f + spec-confirm this session/yoado-18
- **Science / what it measures:** a different (often easier) threat model — recover the COMMON transformation
  applied to ALL training images (not the per-image content). The similarity ladder only uses rot/blur as
  per-image rungs; there is no "recover the common transform" figure yet.
- **Experiment (spec):** apply a fixed shared transform T (e.g. a common rotation θ or blur σ) to all N
  images, fine-tune, then attack to recover T's parameter (not the images). Readout: recovery error of the
  shared-transform parameter vs its true value, across transform strength. New run — executer to scaffold on
  the similarity_ladder harness (it already applies rot/blur); yoado-6d to define the recovery-error metric.
- **Observe-caption:** "What we OBSERVE: how accurately the common transform is recoverable vs its strength —
  leakage BEYOND the weakest attacker's reach (recovering the shared transform is a different, easier target
  than the per-image content the 0/40 measures). OPEN: whether it's easier than per-image recovery."
- **Guards:** own trivial-baseline gate (recovery vs a transform-blind guess, e.g. identity/mean-θ); report
  recovery error with a CI.

### (Track note, not in Gal's F-list) — CRUX closure figures, in-progress from this session
`figures/crux/activation_ranking_857271.png` (oracle first-pass, re-toned observational),
`feature_stability_vs_T.png` (job 390026), `freec_ladder_ranking.png` (job 392821, the realistic ranking).
**Inclusion = Gal's call (yoado-18 surfaced it with a rec).** It's "Gal's #1" per STATUS so it should be
REPRESENTED, but it's the most provisional thread (oracle-mode, the "REFUTED" being walked back, free-c ladder
392821 still running and it could FLIP the ranking). **Decision (adopted, pending Gal):** include as a clearly
labeled in-progress OBSERVATION block ("first-pass, ranking test running, open"), and HOLD the final crux
figure until the free-c ladder lands so nothing is premature. Prominence is Gal's.

---

## SCIENCE SUMMARY (meeting doc: `notes/thesis_scientific_summary.md` — yoado-6d's, committed 25bb63a)
**Do NOT fork it.** It is already observe-scoped + weakest-attacker-caveated. Point its figure references at
the SAME canonical PNGs this figure set produces (the group deduped the combined figure — one canonical set,
no second fork). yoado-6d's stat-pass re-verifies every doc number against source (job IDs, g0 ρ=0.857/0.777,
0/40, rank gap 23→13→0, crux oracle/wc-provisional caveats).
Structure per Gal: one BLOCK per experiment, each plainly **science → experiment → metric → what we OBSERVE
(with CIs / confounds / what's OPEN)**. No thesis-level conclusions. yoado-18 audits clarity + honest scoping.
Block order (proposed): (1) identifiability/q_eff, (2) reconstruction/decoder-limited, (3) who-leaks/g0
[F3], (4) similar-images/similarity ladder [F2], (5) dataset-size [F4], (6) shared-perturbation [F5],
(7) crux/activation-smoothness [if included]. Each block ends with an explicit "OPEN:" line.

---

## ROUTING & SEQUENCE
1. yoado-18 audits THIS plan (scope + observe-framing) → returns edits.
2. On approval: yoado-1f builds F1 cleanup, F2 companion+fixes, F3 retitle, F4, F5 (F1–F3 first — cheap
   fixes to existing data; F4/F5 need new runs, so spec-lock them with yoado-6d before compute).
3. yoado-6d pins F4 metric + F5 recovery-error metric + checks F3 CI / F2 d=nan / all Spearmans.
4. yoado-18 audits the built outputs for clarity + honest scoping.
5. This session integrates the crux closure figures (390026/392821) when they land + folds the summary
   blocks; wires F1.
**Compute gate:** F4/F5 new runs are the executing user's compute-scope call (bsub, WEXAC) — flag before firing.
