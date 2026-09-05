# Controls for the four pre-launch tests — how each could look like a success without being one

**Auditor:** yoado-7e (genuineness/controls) · **Date:** 2026-09-05 · **For:** yoado-cd, and owners
(run specs = yoado-41, control designs = yoado-c9, scoring = yoado-b9).

The principle for all four: **a result is a success only when the treatment beats its own control by a decisive
margin.** The scoring target differs by test type (b9's scope, and it matters):
- **Recovery tests (3), (4):** score by **error against the true image** — never by appearance, never by
  residual/consistency with the release (consistency is exactly what an alias achieves).
- **Membership tests (1), (2):** the certificate residual **is** the statistic and the ground truth is the
  **membership label**, not an image — score the residual against the label. **A membership result must never be
  reported as a recovery:** a rank-1 membership hit says the image was *in the training set*, not that it has been
  *reconstructed*. That is the slide where the two get merged.

For each test below: the failure mode, the decisive control (must launch *with* the test, not after), and the scoring
rule. The two highest-risk for a silent oracle are (3) and (4).

---

## (3) Structured recovery — certificate conditions + box/sparsity convex program  *(highest risk)*

**Looks-like-success path.** A box-constrained (`x∈[0,1]`) or sparsity-regularised solve can return a plausible digit
from almost no information — the prior draws it onto the digit manifold. "It looks like the digit" is precisely the
failure this project has been bitten by.

**Decisive control — the ablation ladder, all four arms, same solver/budget:**
- (a) **full**: certificate conditions + box/sparsity.
- (b) **constraints only**: conditions *removed*, box/sparsity *kept* — cd's control. If (b) recovers, the constraints
  did the work and the release contributed nothing. **The release's contribution is `score(a) − score(b)`; report
  that number, not `score(a)` alone.**
- (c) **conditions only**: existing certificate baseline (no box/sparsity), for reference.
- (d) **scrambled release + constraints**: replace `B_T` with a random-row-space rotation of matched spectrum, keep
  box/sparsity. If (d) ≈ (a), the specific release content is irrelevant — the prior is recovering a generic digit.

**Two extra knobs that separate signal from prior:**
- **Prior-strength sweep.** Plot recovery vs box/sparsity weight for (a) and (b). A genuine result is *robust to
  weakening the prior* (the release carries the information); a hallucination collapses as the prior weakens while (b)
  tracks it.
- **Mismatched-truth control.** Score the (b) solve against a *wrong* held-out digit too. If the "recovery" is about
  as close to a random other digit as to the true one, it is the manifold, not this image.

**Scoring (b9):** per-image error vs the **true image** (`err_vs_chart_per_image` / relative L2), with the (b) control
score reported on the same row. Verdict `recovered` only if (a) beats (b) decisively *and* lands below tolerance vs
truth. SSIM/appearance is not admissible as the success criterion here. Report `score(full) − score(constraints-only)`
**as a ratio too** (b9): a 0.02 gap means something different at baseline 0.03 than at 0.6, and the ⅓-of-best-baseline
bar is stated in ratio units.

**Dataset scope → compressed-sensing prediction test (81's threshold + b9's scoring discipline).** The sparsity prior
is MNIST's, not "images'" — 81's threshold turns this from a guard into a prediction. Recovery of an `s`-sparse signal
needs `m ≳ s·log(n/s)` measurements; the certificate supplies `m = r − N′`. The requirement scales with sparsity `s`,
`r` does not. cd has fixed the **rank** at one marginal value (`r = 256`, pixel-input layer, supplied conditions ≈248
— below that the counting answer is already decisive), so the experiment does **not** fan across ranks; instead the
compressibility axis is swept **at that fixed rank by varying the basis on fixed data** (see 2 below). The four arms
and the prior-strength sweep run per cell. What makes the compressibility axis *scoreable*:

1. **Publish the dose before the response.** For each `(data, basis)` cell, compute the **best `s`-term approximation
   error** (or top-`s` energy fraction) from **public data alone, committed before any attack runs.** Otherwise the
   compressibility axis is chosen after the outcomes — the same fault as picking a threshold from the rows it scores.
   **Declare `s` as a scalar convention before placing cells (b9):** best-`s`-term error is a *curve* in `s`, so the
   ratio `(r−N′)/(s·log(n/s))` moves with which `s` you read — fix e.g. `s` = smallest term count whose best-`s`-term
   error ≤ 1% of signal energy, per `(data, basis)` in the pre-published dose table. Any fixed convention will do; what
   matters is it is declared *before* cell placement, so placement and scoring use the same `s` and the crossing
   cannot be moved afterward by re-reading it.
2. **Basis-at-fixed-data is the primary (experimental) axis; the dataset table is arithmetic, not a run (cd's
   ruling).**
   - **Basis at fixed data** (pixel, DCT, wavelet, learned dictionary on the *same* images) is the **mechanism test
     and the primary axis** — it holds margins, imprints and the recorded set exactly fixed and moves only `s`, the
     one variable 81's threshold is stated in. This is both the cleaner and the *more direct* test of the threshold.
   - **81's MNIST/CIFAR/224² required-rank table is arithmetic** — derived from published sparsity estimates, a
     deployment-relevance statement that needs **no run**. It does not compete with the basis axis and 81 does not
     need a dataset axis to be primary. Mechanism → basis on fixed images; deployment → the derived table.
   - **If a non-sparse dataset arm is run at all, it is the (confounded) scope check**, labelled as such, and it
     carries `N′` and the imprint spectrum beside the compressibility number so the recording confound is *measured*:
     comparable recording ⇒ the compressibility reading survives; different recording ⇒ you know which way it runs and
     say so.
3. **Two guards on the basis axis (cd), both burden-on-the-test:**
   - **A learned dictionary must be fitted on public data only.** Fitted on the private images it is an oracle
     ingredient and the cleanest possible way to fake this result — the same disposition as the (4) decoder. It must
     be demonstrable that no private-correlated signal entered the dictionary, not asserted.
   - **Report each basis's recovery conditioning beside its `s`.** Different bases change the *conditioning* of the
     solve as well as the sparsity, so the claim "recovery tracks `s`" needs the rival "recovery tracks the basis's
     conditioning" visible in the same table, not argued away.
   - **The cells must bracket the predicted transition** (some below, some above the ratio `(r−N′)/(s·log(n/s)) ≈ 1`),
     which is the operational form of "span ≥1 OOM in `s`": it forces informative failures *and* recoveries onto the
     curve. This also dissolves b9's uninformative-success worry — at fixed `r=256` a very sparse basis may sit
     comfortably above the requirement (recovery expected, uninformative *alone*), but as one point on a bracketing
     curve it is informative; the pre-registered finding is the **curve crossing**, not any single cell's outcome.
4. **Pre-register the shape, not the direction.** "More compressible recovers better" is confirmed by any two ordered
   points — unfalsifiable. The CS statement is quantitative: the recoverable sparsity boundary is
   `s ≲ (r − N′)/log(n/s)`, constant unknown, functional form fixed. Score whether the measured boundary **follows
   that curve** (one free constant), equivalently whether recovery collapses onto one curve in the ratio
   `(r − N′)/(s·log(n/s))` crossing ~1 — not merely whether it slopes the right way. **Report the crossing with its
   uncertainty** (cd): one free constant fitted to ~4 cells gives it real error bars — quoting the crossing as a point
   overstates it. **If no crossing appears in the tested range (b9):** the CS constant is unknown, so a ~3× miss can
   land all four cells on one side. That is not a failure — but it must **not** licence adding cells until a crossing
   appears (that is fitting the axis to the outcome). Report it as **"no crossing in the tested range"**; any
   extension is a **separate, separately-reported second round with its own declared range**, never folded into the
   first. Same disposition as the void rules: a mis-placed axis is recorded as mis-placed, not repaired in place.
5. **≥4 `(data, basis)` cells whose best-`s`-term error spans ≥1 order of magnitude.** Below that the dose-response is
   **not scoreable** and only the per-cell verdicts stand — a thin axis is recorded as thin, not fitted.
6. **Fixed transforms are legitimate per-dataset; only the learned dictionary needs the public-fit guard (c9).**
   TV/wavelet/DCT are *universal* compressibility priors, not distributional access to the private set, so choosing
   one per dataset is legitimate — the guard against a fixed prior "doing too much" is the scramble control (arm d),
   not avoiding the prior. The learned dictionary is the sole exception: it is fitted, so it carries the public-data
   burden in guard 3.

**What a positive result costs (write it this way).** If recovery tracks compressibility, that confirms the CS
prediction *and* bounds the attack: it works where the private image is compressible **in a basis the attacker
chose.** That is the chart result in new clothes — yesterday a prior's job was to build the chart, here to choose the
basis; both are the same statement about where the attacker's knowledge enters. One mechanism, not two caveats, and
it is the honest reading of a win.

---

## (1) Truncated certificate

**Looks-like-success path.** A truncated projector `C_trunc` may separate recorded from non-recorded truths because
it is **close to the identity** (kept most directions), not because the **discarded tail is genuinely small** (the
truncation is near-lossless). Only the second justifies the claim.

**Decisive control (cd's, sharpened):** the same truncation applied to a **scrambled `B_T` of matched spectrum** —
a random orthogonal rotation of `B_T`'s row space (matches "closeness to identity" exactly, destroys data alignment).
Plot separation vs truncation rank ρ for **true vs scrambled**; the true gap must exceed the scrambled gap decisively
across ρ, or the separation is a generic near-identity artifact.

**Direct check to add (c9):** report `‖discarded tail‖ / ‖C‖` at each ρ. If the tail is not small, "truncated
separation" is an artifact regardless of the scramble result — the claim *requires* a provably small tail.

**Scoring (b9):** report the separation metric (certificate residual at recorded vs non-recorded truths) for true and
scrambled side by side, plus the tail fraction. The finding is the *gap between true and scrambled*, not the true gap.

---

## (2) Live-regime attack at N = 1

**Looks-like-success path — and the criterion replacement (b9; this test is already running, so the wording matters
now).** The obvious framing "the member's certificate residual is near zero, so it is identified" is **vacuous** at
`N′ = 1`: `row(B_T) = span(A₀h)` and `A_T h ∝ A₀h`, so `Ch = 0` **as an algebraic identity, regardless of image
content** — the member side cannot fail, and separating "member" from noise is detecting *any* signal, not
membership. So the rank-1 member criterion is retired and **the measurement moves entirely to the negative side.**
The claim narrows from *"the certificate identifies the member"* to *"the certificate is specific."*

**The bar is a fixed 1e-2 on the normalised objective — NOT the numerical floor (b9's self-correction).** `C`'s
numerical floor in FP64 is ~1e-15 (so "2 orders above" ≈ 1e-13), which would pass on essentially anything, including
a certificate that had gone vacuous — wrong anchor. Pin the bar at **1e-2 on `‖Cφ(x)‖ / ‖A_T φ(x)‖`** (the
*normalised* form, per the blank-image audit catch where a constant-normalised objective let a blank image win). It
is anchored between two measured bands, not chosen:

| band | normalised residual | disposition |
|---|---|---|
| member / recorded, clean cells | 1e-16 … 1e-8 | member-like (below bar) |
| marginally recorded (in-band tail) | 1e-5 … 1e-8 | must **fail** the bar → non-member-like |
| true non-members, every cell | 0.1 … 1 | non-member-like (above bar) |

1e-2 sits one order below the lowest measured non-member and three above the top of the marginal-recorded band, so a
working certificate clears it by 1–2 orders and one degraded into the marginal band fails it — that middle band is
exactly where yesterday's in-band gate landed.

**Score it as a rate, not a bare minimum (with 1000 non-members the minimum is an extreme statistic; one unlucky draw
would void a working certificate):**
- **Primary: false-positive rate = fraction of non-members below 1e-2, pre-registered ≤ 1%** — a specificity measure,
  robust to a single outlier, directly meaningful as an attack property.
- **Secondary: the minimum non-member residual, reported** — a lone non-member at 1e-6 is worth seeing even inside a
  1% budget; it is the row that would say the zero set is catching natural images.
- **Per-draw null (same image, previous draw's release): same 1e-2 bar, and it must land in the non-member band.** As
  the paired control it outweighs the population — if the null falls below 1e-2 the certificate is annihilating an
  image it never saw, and that draw is a **failure regardless of the population rate**.
- **Report the member residual, labelled as the identity** — no reader mistakes the algebraic zero for a discovered
  separation. **State it as membership, not reconstruction** (`N′ = 1` has no "which member").

**Across the 20 draws (b9):** fraction of draws where the null clears 1e-2 **and** the population FPR ≤ 1%, reported
with the **exact binomial interval.** Unchanged: margin stratification (predicted margin ~15 is in the
marginal-imprint regime — margin 15.6 → imprint 8.2e-7, 17.7 → 4.8e-8), **no pooling across strata**, gate first with
failure voiding the draw, threshold never set on the tested draw.

---

## (4) Learned initialiser  *(highest risk — this is the route that would promote replay from "identifiability" to "attack")*

**Looks-like-success path (the oracle).** The decoder has seen something **correlated with the private set** and
leaks the answer through the init. And "reached the floor" scored against the *release* is consistency, not
correctness — a floor-reaching **alias** is a failure, not a recovery.

**Data hygiene (c9) — non-negotiable before launch:** the decoder's training data must be provably disjoint from *and
uncorrelated with* the private set — different distribution/class where possible, documented split, a held-out
private set the decoder never touched. This is where an oracle slips in unnoticed; treat the burden of proof as on
the test, not the auditor.

**Cheap oracle-detector (c9), run before the residual solve:** score the decoder's **raw output vs the private set
directly, before replay.** If it is already close to the truth before the exact residual runs, correlation leaked in
regardless of what the split's paperwork says — the paperwork proves intent, this proves absence.

**Decisive controls, same solver budget:**
- **Random-init baseline.** The decoder earns its place only if it beats a random start at equal iterations. Report
  `recovery(decoder-init)` vs `recovery(random-init)`.
- **Shuffled-init control.** Feed image *j*'s decoder-init to the solve for image *i≠j* (permute inits across the
  batch). If recovery survives the permutation, the init carries no image-specific information (the recipe route is
  doing the work — fine, but then the decoder is not the story). If recovery collapses under permutation *and* the
  decoder trained on correlated data, the init was the oracle.
- **Near-truth ceiling.** `truth+10%` is the known-working basin; report how close the decoder-init lands to it from
  *attacker-buildable* inputs — that distance is the actual claim.

**Scoring (b9):** error vs the **true image**, and every "reached the floor" paired with err-vs-truth so an alias
cannot pass as a recovery (verdict `recovered` / `alias` / `optimisation-failure`, never merged).

---

## Cross-cutting — applies to all four rows

1. **Start-model flag, blocking (not just mandatory) — b9's upgrade.** Every row records whether its start was
   **attacker-buildable**, and **a row without it gets no verdict** (same disposition as `in_band`). A flag that must
   be set before a verdict exists beats a convention that must be remembered — it is the mechanical form of yesterday's
   defect, where "the recipe is verifiable" read as an attacker capability because its probes started from truth+10%.
   Where a test has no start at all (truncated certificate, N=1 membership), the flag reads **not-applicable with the
   reason**, never blank (41's handling). (3) and (4) are where an oracle ingredient — a prior tuned on the truth, a
   decoder trained on correlated data — slips in as if it were the release doing the work.
2. **Score by error vs ground truth.** Never appearance/SSIM alone; never residual/consistency with the release alone
   (residual is the objective, not success). `fwd_check` at machine precision first, or no downstream number means
   anything.
3. **Report the control's score on the same row as the treatment.** The generalised lesson from (3): publish the
   ablation, not just the treatment, so the release's *marginal* contribution is always visible.
4. **Verdict semantics literal:** `recovered` / `alias (residual zero, wrong image)` / `optimisation failure
   (residual not zero)` — never collapse the last two into "it didn't work".
   **Schema-enforced (c9):** the verdict is a uniform function of BOTH residual AND image-error-vs-truth, never
   residual alone: `recovered` = at-floor AND error < threshold; `alias` = at-floor AND error large;
   `search-failure` = not-at-floor. The residual becomes a RAW field (value + an `at_floor` boolean) carrying no
   verdict word. This makes `recovered` structurally impossible without a correct image in every regime —
   enforcing rules 2 and 4 in the schema rather than trusting the reader — and it dissolves (8)'s above-line
   alias-manufacturing: a manufactured alias floors with large image-error, so the verdict is `alias`
   automatically, no regime special-case. One harness change across every recovery test (round 1 and 5–8).
   Raised by c9, accepted by 7e.

   **Four-cell verdict (cell caught by b9; schema and bidirectional framing c9; additions cd; threshold-tie to
   b9's locked bars; accepted 7e).** Verdict = f(two booleans: `at_floor`, `error_small`), never residual alone:

       at_floor      & error-small  ->  `recovered`
       at_floor      & error-large  ->  `alias`                 (residual certifies, truth denies — OVER-count)
       NOT at_floor  & error-small  ->  `unverified-recovery`   (truth confirms, residual denies — UNDER-count)
       NOT at_floor  & error-large  ->  `search-failure`

   - Residual is a RAW field (value + `at_floor`), no verdict word. `recovered` is structurally impossible
     without a correct image, in every regime.
   - **Verdicts stay categorical; magnitude lives in raw fields (c9).** `unverified-recovery` at 3e-7 and at
     9.2e-3 are both `unverified-recovery` with different raw `image_error`. The spread is in *every* cell —
     `recovered` spans 3e-7 to 9.2e-3 too — so grading the fourth cell would force grading all four.
   - **`at_floor` is relative to the CELL's achievability floor** (the from-truth solve's residual), NOT a fixed
     1e-30. **Factor pinned at 1.5× (b9):** a row is at-floor iff `residual ≤ 1.5 × res_at_truth` **in the same
     units**. Anchored empirically over the 311 rows carrying both: rows that reached their arithmetic floor span
     ratios **0.55–1.33** (fp64 converged 1.20, 1.29; fp32/fp16/bf16 at their format floors 0.55–1.33), while the
     lowest deliberately early-stopped row sits at **1.51**. 1.5 is the tight side of that gap, and tight is the
     safe direction: a borderline row falling to `unverified-recovery` under-counts attacker-claimable (no
     over-claim) while still counting for information-carried (no lost leakage signal). **The gap 1.33→1.51 is
     narrow — revisit the factor if a genuinely converged row above 1.5 appears.**
   - **UNIT HAZARD, must be checked before the ratio is taken (b9).** In many files `residual` is the *objective*
     (~1e-31) while `res_at_truth` is the *residual* (~1e-15) — the square. Over the corpus this shows as a
     spurious ratio cluster at ~5e-16. Any `at_floor` computed as `residual / res_at_truth` across those rows
     compares an objective to a residual and is wrong by fifteen orders. Take both from the same field, or square
     one, and mark a row **indeterminate** rather than guessing when the units cannot be established.
   - **`error_small` is NOT a new number** — it is the already-locked success bars (1e-2 absolute,
     ≤⅓-of-best-baseline, achievability-floor multiple where that governs). For a multi-image cell it is the
     **worst image under the one-to-one assignment**, consistent with the locked "all `N′` clear" criterion;
     `partial (j/N′)` remains a separate reported field and is not a verdict.
   - **The verdict is the EXPERIMENTER's label (cd, c9 converged):** it needs image-error-vs-truth, which the
     attacker never has. The attacker sees only `at_floor`. **Below** the certificate line `at_floor` ⟹ recorded
     image, so the attacker's own verdict is sound; **above** it `at_floor` does not imply correct, so the
     attacker cannot distinguish `recovered` from `alias` at all — one structure, two symptoms, the same boundary
     as the residual guard from the scoring side.
   - **Three honest metrics, not two (c9's bidirectional framing).** `{residual at floor}` = {recovered, alias}
     and `{true recovery}` = {recovered, unverified-recovery} overlap only in `recovered`. So
     **information-carried** = {recovered, unverified-recovery} (ground-truth); **attacker-claimable** =
     {recovered, alias} (residual at floor, alias-contaminated); **verified-true** = {recovered} (the
     intersection, isolable only with ground truth). The residual — the attacker's only instrument — diverges from
     true recovery in **both** directions: it certifies aliases it should not and misses recoveries it should
     catch. `unverified-recovery` counts toward information-carried, **never** toward an attack success rate.
   - Applies to every recovery test (round 1 and 5–8), one harness change; the derived 4-cell verdict sits
     alongside the preserved recorded verdict, so no historical row is rewritten.
5. **Any learned component is measured against the private target before it is used for anything (cd's standing rule,
   generalising c9's (4) detector).** Score the component's raw output vs the private set and report that number on
   the row *before* the component feeds a solve — covers the decoder in (4) and the learned dictionary in (3) with one
   rule. If it is already close before anything runs, correlation leaked in regardless of the split's paperwork; the
   paperwork proves intent, this number proves absence.
6. **The achievability floor — every solve reports a companion solve started at the truth, and the ratio (b9).**
   One gate solve **per cell** (not per start), started at the ground-truth parameters, using the **same solver,
   tolerance and iteration budget** as that cell's attack rows. Report the objective and image error at the truth
   *before any step*, the endpoint after the matched budget, and `ratio = achieved error / floor error`.
   - **Name it exactly: the achievability floor under solver S**, with S named on the row. It is **not** an
     information-theoretic floor — a better solver could sit lower. *"Within 10× of the floor"* must never be readable
     as *"within 10× of what the release permits"* (the same distinction we enforce on the fitted CS constant, from
     the other side).
   - **The gate row is a near-truth start:** `start_attacker_buildable = false`. It is a floor measurement and must
     not read as an attack result. N/A-with-reason where no truth exists.
   - **What the pair buys (the point of the rule):** two numbers separate two failure modes that keep getting
     conflated — **ratio ≫ 1** = the solver fell short of what this configuration offered; **floor itself large** =
     the channel does not carry it, *under this solver*. A shortfall with one number is un-attributable.
   - **Report the endpoint even when it is worse than the start** — a from-truth solve can move *away*; a gate that
     degrades is solver instability, worth seeing, not clipped to the starting value.
   - **Two exclusions, so the number is not over-read:** a low floor does **not** mean the attack works (it is
     measured from the truth); a ratio near 1 with a large floor is a **channel limit under this solver**, not an
     information bound.
7. **Non-pooling across branches (test 3, and any test with a fallback branch).** In a primary branch both bars are
   about recovery; in a fallback branch the floor bar is **solver efficiency** and only the ⅓-of-best-baseline clause
   is about recovery. A summary line reading *"recovered in N of M cells"* across both branches is false — the
   verdicts are **not commensurable** and are reported separately.

---

## Rules 8–10 — appended 2026-09-05

**Provenance.** Author **yoado-7e** (owner of this file); **transcribed by yoado-b9** from cross-session messages
because 7e's file-write tool was timing out on its host and rules 8 and 9 otherwise existed only in transient
session messages. Rules 8 and 9 were drafted by b9, accepted by 7e as drafted, and are reproduced here from the
drafts 7e accepted; the clauses marked *non-soften* are the ones 7e stated it was keeping verbatim. **Rule 10's
number is b9's** — it is a standing rule of cd's that had not been given a number in this list, and 7e may
renumber. Anything below that is transcribed rather than authored is marked as such rather than smoothed.

8. **The standing definition of `N′` (cd's ruling; drafted by b9, accepted by 7e).**
   > **`N′` is `rank(B_T)` evaluated at the tolerance actually used to build the projector `C`.** Nothing else.
   > It is reported on every row **alongside that tolerance**, and `in_band` is derived from it mechanically.

   *Why this and not the raw recorded count.* The certificate **is** the projector, and the projector removes
   exactly the directions it resolves at its own tolerance. An image whose imprint sits at 1e-9…1e-10 is not
   removed by a projector built above that level — it is not certifiable, it reads as a non-member, and the budget
   genuinely is `r` minus the number of directions actually projected out. A raw count including directions the
   projector never removed states a budget the certificate does not have.

   - ***Non-soften — companion clause.*** `N′` now depends on a chosen tolerance, so **the projector tolerance must
     be declared before the run and held fixed across every cell in a comparison.** Selecting a tolerance that puts
     a cell inside the band is the same fit this apparatus exists to prevent, and the definition alone does not
     forbid it.
   - **Corollary, made concrete.** A cell whose `in_band` status changes under a plausible alternative tolerance
     must report both. Evaluate `N′` and `in_band` across a **declared ladder — {1e-6, 1e-8, 1e-10, 1e-12}** — and
     **flag any cell whose `in_band` is not constant across it**, so a boundary cell shows as one on its own row.
   - **First use, recorded as such:** 41 ran the ladder on test (6); `in_band` is stable across all four orders with
     no cell flipping, so the band split is a property of the release rather than an artefact of where the projector
     was cut. **The ladder was added to be able to detect the *unfavourable* case; this is what a guard looks like
     when it passes** (cd's note).

9. **A claim carries the parameter range actually measured, inside the claim itself.**
   Not *"at a deployable rank"* but *"at rank 8, the only rank tested"*. Not *"recognisable"* but the metric and the
   threshold. **Corollary:** any claim quantified over a parameter must **name the values swept**.

   *Why mechanical rather than vigilant.* Both failures this week — *"the one-image regime and the band are mutually
   exclusive at a deployable rank"* (false at rank 64 within a day, job 351056) and *"the alternatives are
   recognisable"* (never assessed) — were caught by someone reading rows **for a different question**, not by anyone
   re-reading the claim. Re-reading is therefore not the mechanism that catches this; if the range is in the
   sentence, the over-reach cannot be written in the first place.

   - ***Non-soften (a)*** — **the rule must bite where there is no scope word at all.** The commoner failure is
     **implied universality by omission**: *"the certificate vanishes at recorded truths"* has no qualifier and is
     measured at particular `m`, `r`, `N′` and chart. Require the range whether or not a qualifier is present.
   - ***Non-soften (b)*** — **operational test:** *could a reader identify, from the sentence alone, a cell that
     would falsify it?* If not, the range is missing.
   - ***Non-soften (c)*** — **name the boundary, not only the values.** *"Measured at k = 8, 16, 32"* does not tell a
     reader the claim was never tested above 32.
   - **Enforcement point:** applied **at promotion** — when a claim moves into STATUS.md, the `.tex`, or a document
     meant for the supervisor. Retroactive application across the whole record would be a large audit done badly;
     promotion is where an over-reach becomes durable and where the check costs least.

10. **Negative rate mandatory (cd's standing rule; number assigned by b9, 7e may renumber).**
    **No membership-style claim may rest on a positive-side criterion.** With a zero certificate the member residual
    is still ~1e-14, so a vacuous test and a perfect test are **identical on positives** and differ only on
    negatives. A negative rate is required on every row.
    - **b9's added clause:** **the negative must be paired wherever a pairing is constructible.** A population
      negative is confoundable — by class, as the same-class cell showed — while a paired negative (same image,
      different release) holds image-specific factors fixed and cannot be. A **population-only negative must state
      what it does not control for**.
    - This rule would have caught the broken Adam arm, the vacuous hidden-layer certificate and the graded-statistic
      ceiling (cd).

11. **Realised pool size reported beside requested (cd's standing rule; promoted to a numbered rule by 7e,
    2026-09-05).** Wherever a run draws from a pool — non-members, public candidates, random starts — the row
    records what was **realised**, not only what was asked for. A requested 1,000 that realised 340 changes every
    rate computed from it, and a rate quoted against the requested number is wrong by however much the shortfall
    was.

12. **Report the curve alongside the chosen point (cd's standing rule; promoted 2026-09-05).** Where a scalar is
    selected by a rule — a regularisation weight by the discrepancy principle, a truncation by the largest spectral
    gap, a stopping point — report the **quantity across the swept grid**, not only the value at the chosen point.
    It makes tuning visible rather than merely forbidden: a result at a sharp optimum reads differently from one on
    a plateau, and the reader can see which it is without taking the selection rule on trust.

**Citation check (rule 9, job 351056) — VERIFIED by b9, 2026-09-05.** `results/exact_inversion/step121_probe_351056.jsonl`,
git `e55a306`, `part: PROBE` rows: `confident` `N′`=7 lines 57/67 · **`hard1_diff` `N′`=1 lines 63/73** ·
`repeated` `N′`=6 lines 58/68. The `hard1_diff` row is a single-recorded-image cell whose band (63…72) contains
chart dimensions in the swept range, which is what falsifies *"the one-image regime and the band are mutually
exclusive at a deployable rank"* at `r = 64`. The citation is correct as written.
