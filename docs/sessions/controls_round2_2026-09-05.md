<!-- PROVENANCE: authored by yoado-7e (genuineness/controls lane). Transcribed verbatim to disk by yoado-b9
     (claims/scoring lane) on 2026-09-05, from a complete draft sent by yoado-7e in a cross-session message,
     because 7e's file-write tool was timing out on its host and cd had ruled that a controls document must not
     exist only in session messages. Nothing below is reconstructed or paraphrased; the text is as sent. The
     (5) axis bullet already reflects cd's reversed ruling (k_g at fixed r, not both). Corrections belong to
     yoado-7e as author.

     AMENDMENT 2026-09-05: the two sentences in section (8) describing the residual guard below and above the
     certificate line were raised by yoado-b9 as auditor (below the line the chart is safe BY CONSTRUCTION, not
     merely by control; above the line the guard is ACTIVELY MISLEADING rather than merely void), accepted by
     yoado-7e as author, and landed here by b9 as scribe from 7e's verbatim replacement strings.
     The (8) bullet on above-line alias-manufacturing was raised by b9 as the operational consequence of that
     amendment, formulated and accepted by yoado-7e as author, and scribed by b9. It was SUPERSEDED the same
     day by c9's uniform verdict schema (rule 4, schema-enforced) and replaced with the bullet now present, again
     verbatim from 7e. -->

# Round-2 controls — how each of the four reopened tests could look like a success without being one

**Auditor:** yoado-7e (genuineness/controls) · **Date:** 2026-09-05 · **Spec:** notes/round2_test_specs.md (352ec7e) · **For:** yoado-cd, and owners (design = c9, scoring = b9, execution = 41).

All cross-cutting rules 1–12 in controls_four_tests_2026-09-05.md stand and apply here. Oracle risk order per cd: (5) > (8) > (7) > (6).

## (5) Generative-prior recovery — highest risk
Looks-like-success path. A generator G produces a plausible image of the right concept from no information at all — the structured-recovery failure with a stronger prior. "It looks like the concept" is not recovery of THIS private image.
Decisive control (cd's, = round-1 constraints-only transposed): generator-only — search G's latent space with the certificate conditions removed. If that already returns something close, G is doing the work. Recovery contribution = score(full) − score(generator-only), reported as a number AND a ratio. Plus scrambled release (matched spectrum, row-space rotation) and the prior-strength analogue.
Round-2-specific:
- Two void gates, before the attack (cd): (i) publish G's representation error for the private images — if G can't represent them, the cell is void, not a failure; (ii) score G's nearest output and G's mean against the target — if either is already close, void (prior alone suffices). Both are oracle-only void checks, never attack rows.
- On-manifold is an idealisation (cd): private images drawn from G's range — upper bound, chart contains target by construction. Never reported in the same breath as the off-manifold arm (the real test).
- Axis: k_g at FIXED r (cd's ruling; NOT both). Sweeping k_g is a property of the generator, so it holds B_T, the certified set, N', the projector and conditioning fixed — only the prior varies. Sweeping r changes the release, and by rule 8 a smaller budget can push weak imprints below the projector tolerance and silently move the certified set, so a two-axis grid shifts the crossing for two reasons at once — the confound the certified-index rule exists to catch. So: primary single cell at k_g=8 first (one success refutes the sparsity account without any grid), then the k_g sweep at that same fixed r. The r axis runs ONLY if a crossing appears, solely to test whether the crossing tracks the budget; with no crossing it has nothing to confirm and does not run.
- Pre-register the crossing as a shape, with uncertainty (round-1 test-3 discipline). Prediction: crossing at k_g ≈ r − N', NOT the sparsity threshold (~250 MNIST). Score as a curve with error bars; cells bracket it; "no crossing in range" reported as such; extension is a separate round.
- Certified-set consistency across the sweep (b9), or void the collapse fit. The certified image INDICES must match across cells — not the count — or the cells attack different targets and there is no curve. Intersection fallback: if sets differ, fit on the intersection, report its size, name excluded images, score that branch against 10× its own achievability floor (excluded images stay in B_T as an omitted-imprint floor), never the absolute bar. Non-pooled from the primary branch.

## (6) In-band handoff — built; launch-ready
Looks-like-success path. All existing handoff numbers are from below the line, where the certificate already isolates the images — nothing to chain. In band is untested. The positive outcome ("the decile cut SURVIVES in band") is the most attacker-relevant result available, and the one most at risk of wishful reading.
Controls (all confirmed built by 41): norm-matched chart baseline; landing image error logged before replay; wrong-manifold arm = matched-spectrum row-space rotation (41 fixed the prior row/col-permutation arm, which also preserved the entry multiset); achievability floor (companion from-truth solve per cell). Cells k ∈ {62,64,66,68} — two in band, two above, deliberate crossing design; out-of-band cells are DESIGNED EXCLUSIONS (no chain verdict via in_band), never two failures.
Round-2-specific:
- Tolerance-ladder band check ran and is stable (rule 8): 41 confirmed no cell flips in_band across {1e-6,1e-8,1e-10,1e-12} for all three batches — split is real, two per side suffice, none added.
- Genuineness condition on the positive outcome, in 41's ledger before any row exists: if the cut survives in band it must ALSO fail on the scrambled release, AND the achievability floor must show the channel carries it (ratio, not a solver artifact). Both arms in the same job as the treatment.
Status: launch-ready — 41 cleared to run.

## (7) Label-derived starts
Looks-like-success path. The public photos collected for the labelled concept may already be close to the private images by concept-similarity, so replay "succeeds" from a start near the target — a statement about the concept being public, not about replay reconstructing THIS private image.
Decisive controls (cd + 41; oracle detector is the MEASUREMENT, not a gate — cd's stronger form):
- Public pool verifiably disjoint from the private set — burden of proof on the test, as with the (4) decoder.
- Oracle detector = the measurement. Score collected candidates against the target BEFORE replay; the nearest-candidate distance to each target is the floor the label provides, and replay's contribution is what it buys beyond that. If a raw candidate is already recovered before replay, the honest finding is "the private image was in the public pool" — real, publishable, and a completely different claim from "replay recovers."
- Random-start baseline in the REAL-DATA setting (41's load-bearing catch). The standing 0-of-20 is from the synthetic bed where a label prior is vacuous by construction; build the baseline in the real-data world, report both from the same world. Finding = recovery(label-starts) − recovery(random-starts), same setting. Nothing quoted against the synthetic number.
- Report attacker cost: number of candidates needed.

## (8) Adaptive chart — second highest; self-confirmation by design
Looks-like-success path. Refitting the chart on the recovered images biases it toward those images, so the second recovery "improves" because the chart now represents them — circular. And adaptive refitting can MANUFACTURE an alias: refit so a wrong recovered image becomes floor-reachable, and the next round floors at that wrong image and reads as confirmed.
The hole is scoped to above the certificate line (cd's precision). The residual guard ("a self-confirming chart yields a candidate that misses the floor") is only true of wrong images that FAIL to floor.
- Below the line the zero set is isolated points — exactly the recorded images — so any floor-reaching candidate IS a recorded image and no refit can manufacture an alias (no non-truth zero to refit through). Residual guard is not just sound here — the adaptive chart is safe BY CONSTRUCTION: the geometry forbids the failure (there is no non-truth zero to refit through), so the three-way control is not logically required, and a below-line adaptive result needs no trust in the controls at all.
- Above the line the zero set is a positive-dimensional manifold — a refit through one of its non-truth points produces an alias that genuinely floors. Residual guard is not merely void here but ACTIVELY MISLEADING — a manufactured alias floors, so the guard returns PASS on the very failure it was meant to catch, which is worse than absent — and the controls below are the only protection.
- 41 runs below the line by default and states the regime on every row. The regime is a REQUIRED FIELD, so a reader can tell at a glance whether the residual verdict on that row means anything.
- Above-line alias-manufacturing needs no (8)-specific reporting rule: the uniform verdict schema (rule 4, schema-enforced) handles it — a manufactured alias floors with large image-error → verdict = `alias` automatically, in every regime; the residual is a raw field (`at_floor`), never a verdict word. Below the line, residual-at-floor ⟺ truth by identifiability, so `recovered` reads there as it should — the strongest claim in the programme, safe by construction.
Controls (load-bearing above the line, confirmatory below):
- Refit on a RANDOM image instead of the recovery (cd's decisive control), sharpened to three-way: refit on (a) the recovery, (b) a random image, (c) a DIFFERENT private target's recovery. Real effect = improvement(refit-on-own-recovery) − improvement(refit-on-random); if (c) also improves, the gain is generic chart enrichment, not target-specific.
- Score against ground truth and watch the alias verdict at every bootstrap iteration — adaptive refitting manufactures aliases, so the alias verdict per iteration is the detector; an iteration that turns alias is the self-confirmation firing, not a success.
- Private data never touches chart fitting at any stage — burden on the test (cd).
- Void below a declared minimum C₀ quality (cd) — at chance there is nothing to bootstrap; void, not a null.

## Cross-cutting strengthenings requested for these four (answer to cd's "say so before launch")
1. Rule 8 (N'-at-tolerance) is what (6)'s band (done, stable) and (5)'s certified set need — a shrinking budget can drop weak imprints below tolerance and silently change the certified set.
2. Rule 2 (score vs truth) + the regime field close (8)'s alias-manufacturing — real only above the line; residual guard sound below.
3. The achievability floor and the round-1 curve-collapse-with-uncertainty transpose onto (5).
No other rule needs changing. The only genuinely new surface was (8)'s alias-manufacturing, now scoped to above-the-line and closed by rule 2 + the regime field + the three-way refit.
