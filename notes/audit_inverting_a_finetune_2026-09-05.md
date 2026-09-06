# Audit — "Inverting a Fine-Tune" (artifact 46b9465a), primitives and pitch

**Date:** 2026-09-05 · **Auditor:** GM session (yoado-78) · **Scope:** the math of the primitives on the page,
what the page presents versus what the ledger now says, and the position against SimuDy (Tian et al., ICLR 2025)
and the supervisor's own papers (Haim et al. 2022; Smorodinsky, Vardi, Safran 2025; Oz et al. 2024;
Gronich & Vardi 2026). Sources read at source: the artifact HTML, `STATUS.md` (top through the four-model
surface table), `experiments/exact_inversion/RESULTS.md` (counting rule, capacity table, three counts),
`notes/related_work_simudy.md`, `notes/simudy_decision_brief.md`, and pypdf text of the four papers.

## 1. Primitives that check (verified by hand)

| primitive | statement on the page | check |
|---|---|---|
| closure | `B_t = P_t (A_0H)^T`, `A_t = A_0(I + H M_t H^T)`, with the `P`, `M`, `D_t` recurrences | correct by induction: `∇_B ∝ D_t (I + G M_t^T)(A_0H)^T`, `∇_A ∝ A_0H P_t^T D_t H^T`; logits `W_0H + s P_t Q (I + M_t G)` |
| seed reduction | `A_T U = XΩ`, `A_T V = Y`, `B_T = P_T R_H^T X^T`; `Q` sees `A_0` only through `X^T X` | correct; `U^T V = 0` kills the deformation on `V`; rotation gauge on `X` is covariant, not a gauge of the release |
| faithfulness | `ρ = 0` iff some init reproduces the release, via `A_0' = X U_c^T + (A_T V_c)V_c^T` | correct: `A_0' U_c = X`, `A_0' V_c = A_T V_c`, both factors reproduce |
| replay count | release variety has dimension `N((m−1) + r − N)`; line `k < m + r − N` | correct: rank-`N` matrices with columns in `1^⊥` are rank-`N` in `R^{(m−1)×r}`; the second block pays exactly for `X` |
| imprint law | `B_T = Σ_i C_i`, `‖C_i‖ ≤ ηs max_t‖A_t h_i‖ Σ_t‖D_t[:,i]‖` | correct; per-example split of `D_t H^T A_t^T` is exact, bound is the triangle inequality |
| certificate kernel | `ker C = recorded ⊕ ker A_0`, codimension `r − N'`, line `k < r − N'` | correct **given exactness (see 2.1)** |
| simplex cap | `1^T B_T = 0` so `rank B_T ≤ min(m−1, r, N')` | correct |
| full-FT comparison | `ΔW` rows in the span of recorded features; line reads with `n` in place of `r` | correct **given `W_0` public (see 2.6)** |
| Adam | closure fails (entrywise division leaves the gradient subspace) | correct |

## 2. Holes in the primitives as stated

**2.1 The certificate is exact only when every batch member is recorded.** `A_t h_i = A_0 H (I + M_t G) e_i`
mixes every feature direction, including unrecorded ones, with weight of the order of the feedback. So
`row(B_T)` equals `col(X)` only when `rank B_T = N`; when `N' < N` the recorded directions are not exactly in
`row(B_T)` and `C h_i` is small, not zero — its size is predicted (feedback × unrecorded imprint), not
arbitrary. The page's own evidence agrees, and more strongly than first quoted here (verified at the job rows by
yoado-93, 2026-09-05): at `N' = N` the certificate residual sits at ~1e-16 to 1e-10; at `N' < N` it reaches order
one. On the 26-logit head (job 725918, `step74_widehead_725918.jsonl`, checked by the GM at the row) the worst
recorded image's residual (`cert_residual_recorded_max`) is 0.23, 0.38, 0.38, 0.45, 0.45 at recorded counts 19, 17,
16, 15, 14; the median over the twenty recorded images climbs three orders, 1.2e-6, 1.3e-4, 6.7e-4, 6.3e-4,
1.3e-3, and the number of recorded images above one percent runs 1, 4, 6, 8, 7 of twenty (yoado-64, at the row) —
the loss of exactness concentrates in a handful of images and grows with `N − N'`. (A first GM reading of the
median as "near 1e-3 in every row" was a rounding artefact: printing a log-scale quantity to three decimals hid
three orders of variation.) (An earlier "0.05 → 0.20" quoted here came from the
page's own prose, not the job; withdrawn.) Quote it as "~1e-16 at `N' = N`; at `N' < N` the worst recorded image
climbs toward order one with `N − N'` while the median climbs from 1e-6 to 1e-3 and most images stay nearly exact". "The algebra holds as stated" should read: exact at
`N' = N`, approximate otherwise, with the residual scale stated. The same caveat applies to "the row space of
`B_T` is the column space of `X`, legible off the release with no work at all".

**2.2 "The truth is the only zero" is isolation, not uniqueness.** The analytic-minor argument gives local
isolation almost everywhere. Permuting the batch gives `N!` zeros trivially; other discrete zeros are not
excluded by the argument. Global uniqueness below the line is *measured* (floor fraction = landing fraction in
one cell), not proven. The four-cell verdict's "below the line consistency is correctness" rests on this and
should be labelled as measured.

**2.3 The dropped `1/N`.** The gradient formulas omit the mean's `1/N` while the batch-size corollary (R5)
depends on it entering as `ηs/N`. Write `ηs/N` in the recurrence or say the mean is folded into `η`.

**2.4 "The complement carries no information" is an exact-constraint statement only.** `A_T V_c(w)` must look
like an iid Gaussian draw, so it is a weak statistical test on the candidate span. Not a hole in the theorem;
an overstatement in prose.

**2.5 Analyticity needs an analytic chart.** PCA and tanh generators qualify; a ReLU-decoder VAE does not.
State it in the assumption table.

**2.6 The full-FT comparison assumes the base weight is public.** For a head trained from scratch (the Oz et
al. 2024 setting) `W_0` is unknown and the exact route does not apply; only the KKT route does. The page's
"LoRA is the mitigation" line is correct for public-`W_0` fine-tuning and silent otherwise.

**2.7 Minibatching is provable, not conjectural** (independently re-derived by yoado-93, 2026-09-05). Each minibatch gradient lies in `col(A_0 H_b) ⊗ row(H_b^T)`
with `H_b ⊂ H`, so the induction closes with `H` the union of features seen and `M_t` updated blockwise. The
cost is that the batch order becomes part of the recipe. Two lines; upgrade the assumption table.

## 3. Holes in what the page presents (page vs ledger, 2026-09-04/05)

**3.1 The channel's surface is absent — the largest hole.** Since job 273322 the ledger states that the
recipe-free certificate is identically zero on weight-shared modules at deployed rank: one image records one
direction per token or position, so a single image floods a rank-8-to-64 adapter on any transformer block
linear (197 tokens) and early convolutions are vacuous at every rank. It exists on heads, dense non-shared
layers, and deep convolutions with few positions (the capacity table, job 296789). The page's
"which fine-tunes leave anything to find" table therefore over-promises: attention-LoRA, the dominant
deployment, is outside the channel. This must be on the page, and it is the best defender-facing tool the
project has (the counting rule `min(r, d) − min(N·p, d)`, evaluable from architecture and batch size alone).

**3.2 No membership baseline.** Against LiRA on a ViT-B/16 head (job 287241) the certificate wins on
assumptions (no shadows, no recipe), not on separation. The page's "membership needs no fidelity" paragraph
should carry that sentence.

**3.3 Replay counts are single-layer, frozen-input.** The deployment-gap result (jobs 218345/218346): the
release determines the image only if adaptation reaches near the input, and at deployed rank on pixel-input
adaptation it pins about 7% of the image. The page's per-example budget `m−1+r−N'` is the frozen-input
number; multi-layer adapters count (image, step) directions and the budget shrinks.

**3.4 Lead result: attacker-verifiable fraction unmeasured.** The page says so, correctly. Keep the caveat in
the caption as well as the prose; do not let "all eight" travel without it.

**3.5 The start problem is stated honestly but not positioned.** Every replay recovery starts near the truth;
no attacker-buildable start reaches the floor. SimuDy starts from noise and reaches SSIM ≈ 0.2 on ResNet-18.
Say plainly that on the start problem SimuDy is ahead and the chained route (certificate proposes, replay
pins) is the untested answer.

**3.6 The arithmetic section stays out of the pitch.** Keep it on the page for completeness; it does not go to
the supervisor.

## 3b. Coherence audit of the published page (GM, 2026-09-05)

Read end to end as a theorist would. Three deduction defects were found and sent to the fixer, all now repaired:
the count switches from `N` to `N'` mid-argument in three forms without a stated handover; the surface section
(which decides whether the certificate exists at all) sat after every claim depending on it; and a spliced
sentence in the imprint block. The fixer moved the surface section behind the certificate, stated the
`N → N'` handover once with the `k < m+r−N'` / `m−1+r−N'` equivalence spelled out, and rejoined the sentence.

**One substantive defect, found by re-verifying at the raw rows what had been checked against ledger prose
(job 273322).** The page states `margin = min(r,d) − min(N·P, d)` and then tabulates the transformer row as
"0 at every rank tried and every batch size, a single image included". The job tried ranks 8, 16, 64 **and 256**;
`margin_by_rank` at `N = 1` reads `{8:0, 16:0, 64:0, 256:59}` for every block module of ViT-B/16 and ViT-S/16,
and the page's own formula gives `min(256,768) − min(197,768) = 59`. **The general rule and its own instance
disagree on the same screen.** Correct scope (as STATUS already had it): zero at every *deployed* rank, 8–64,
for every batch size including one; at rank 256 it survives only at `N = 1` and dies at `N = 2`. The error runs
in the defender's favour, so the honest version strengthens the argument — the boundary moves exactly where the
formula says it should.

**Verified clean at the raw rows** (do not re-litigate): head margins 8 at `r`=16 and 56 at `r`=64 for `N`=8
(279182); LiRA 0.994–1.000 precisely in the `rank C = 0` batch-equals-rank cells (287241); the live ResNet cell
`N'`=49, `rank C`=15, false-positive rate 0.0000, 19 of 20 (307760 — the count is `null_clears_bar`, 19 true and
1 false, confirmed by the summary row at successes 19 / scored 20; all 20 rows carry `gate_passed` true, which is
a different field and **not** the scoring criterion — a field-name trap worth remembering); Adam control at
`rank C` = 0, false-positive rate 1.0, 6 of 6 voided.

**Coherence fixes verified live (GM, at the artifact, not at a copy).** Surface is section 8 of 13, directly after
the certificate section; the splice is repaired and a lowercase-after-period sweep returns nothing page-wide; the
`N → N'` handover is present verbatim with the `k < m+r−N'` / `m−1+r−N'` equivalence stated. An audit report calling
these three still-live was a stale read — see LESSONS, instance 5. **One straggler remains**: after the handover,
the trade paragraph reads "carries `r−N'` equations per image where replaying carries `(m−1)+r−N`" — two symbols in
the one sentence that compares the two channels. Sent to the fixer.

**The lesson for the sweep:** the highest-value class of defect here is a stated general rule contradicting its own
tabulated instance. Prose checks cannot catch it, because the ledger prose is correct; only the rows can.

## 3c. The other two documents (GM, 2026-09-05)

The argument lives in three places and the corrections had reached only one. Both others carry **the same two
serious defects**, and they are the two documents a theorist actually reads.

**The theorem-first companion** (artifact `50af0d1a`) — **REBUILT AND VERIFIED 2026-09-05.** The explainer lane
retyped it rather than patching (its read returned source inline, so there was no file to edit); the GM read the
rebuild end to end at source, since a retype of a technical document risks silent content loss. **Nothing lost:**
every theorem, "why it holds", the membership test and its three sub-points, the conservation law, the retracted
column reading, and every job id survive. Now 17 items: the surface is item 5 as a **proposition with hypothesis
A7** and the margin formula, stating the rank-256 boundary as *the rule checking itself*; item 16 carries the four
cells and the three counts, with its rules list pointing at the third row rather than at a case the schema could
not express; item 17 states the from-nothing results as **landed**, naming the attacker-verifiable quantity as
unmeasured. **One straggler:** item 13's table uses bare `N` where item 12's uses `N'`, and item 14's handover
names items 6, 10 and 12 but not 13 — a third instance of the rule-versus-table class, produced by the edit that
fixed the other two. As it stood before that rebuild, dated 2026-09-03: Six divergences; two serious. (i) No surface, no counting rule anywhere — a reader would
conclude LoRA-on-attention is attackable, the reverse of what was measured. (ii) Its closing status says the
from-nothing attack "has not landed", which predates 706721, 728592 and the letters cells: **the page denies the
project's own lead result**. Also: three-cell verdict, no exactness scale, no membership baseline, no R5. Its
mathematics is sound (closure, seed reduction, gauge, simplex, Theorem I including the careful point that rank
deficiency at the truth alone would not give a fibre, the `T=1` witness, Adam) — it is the **scope and the status**
that moved. Assigned to the explainer lane; controls lane auditing.

**`notes/exact_channel_rev10.tex`** — the thesis text, compiled via Overleaf, the version that becomes the paper.
Its line-1002 survivor is **already fixed** and carries the full experimenter-vs-attacker caveat. Two defects remain,
verified by reading the file:
- **Three-cell verdict** (~line 561): *recovered* / *alias* / *search failure*, the last defined as "residual above
  the floor" with **no condition on the image error** — so a run returning the correct image without residual
  confirmation is classified as the opposite of what happened. `unverified recovery` occurs **nowhere** in 2,895 lines.
- **No surface, no counting rule**: zero occurrences of *attention*, *positions*, *deployed rank*, or the margin
  formula. The only nearby passage concerns the frozen encoder's bottleneck at a ViT stem, which is a **different
  argument** and must not be mistaken for the counting rule.

**Standing check, from two instances in two days:** wherever a document states a rule and also tabulates instances of
it, verify the table against the rule. On the main page a formula contradicted its own table (`273322`); in the
companion, item 15's rules list warns against a thresholded verdict while its table has no cell to express one.

## 4. Against SimuDy (same primitive, four deltas, one deficit)

Same primitive: replay the recipe from `θ_0` on candidate data and match `θ_T`. SimuDy: full fine-tuning,
full unroll (memory-bound, 22 GB and 15 h for 120 CIFAR images on ResNet-18), cosine loss plus TV plus
gradient clipping, minibatch, grid-searched learning rate, starts from noise, no identifiability statement,
no verdict on a solve.

Where this page is ahead:
1. **The unroll is batch-sized, not model-sized** — but only for one adapted layer with a frozen input.
   Multi-layer adapters break the closure and return to SimuDy's cost.
2. **An exact residual with a floor gives a verdict** (alias versus search failure). SimuDy's loss
   "fluctuates around 0.2 while reconstructions are good" (their §4); it cannot certify anything.
3. **A counting theorem** for the chart budget. SimuDy says only "over-determined".
4. **Recipe identifiability**, including R5 (only `ηs/N` is identifiable). SimuDy's Fig. 7 — wrong batch-size
   guesses give equal quality once the rate is re-paired — is that corollary observed empirically; cite it.

Where SimuDy is ahead: from-noise starts, real ResNet-18 and ViT, N up to 120. On the replay route this page
has no attacker-available start.

## 5. Against the supervisor's papers (where to hang the story)

- **Haim et al. 2022.** KKT stationarity at `t → ∞` with unknown multipliers; its residual has spurious
  solutions (Smorodinsky et al. say so in their introduction: such networks "could have been trained on many
  datasets"). This page is the finite-`T`, exact-dynamics version for LoRA: the residual is zero at the truth
  by construction, so the floor certifies. Its count (dimension of the reachable release variety) is the sharp
  form of Haim's "p equations, nd unknowns".
- **Smorodinsky, Vardi, Safran 2025.** Provable membership: `|Φ(x)| = m` for members against `o(m)` for fresh
  points, needing near-orthogonality (their Assumption 4.1) and a known or bounded margin. The certificate is
  the same shape of statement — exact for members, order one for non-members — with different hypotheses:
  SGD-class, `B_0 = 0`, non-shared module, `N' < min(m−1, r)`, and no distributional assumption. Their
  univariate reconstruction returns a finite candidate set with a constant fraction of training points; our
  "argmin lands on a recorded image, basins uneven" is the same flavour. **Present the certificate in their
  format: Assumption, Theorem, Algorithm, with the counting rule as the Assumption-4.1 analogue.** That is the
  tool.
  **Realised end of day, and this is the sharpest thing to say to him.** Their Assumption 4.1 is a *checkable
  condition on the data distribution* (near-orthogonality) under which a clean membership statement follows.
  A7 plus the counting rule is a *checkable condition on the architecture* — `N·P` against the input width,
  evaluable from a config file before any release exists — under which the certificate's clean statement
  follows and without which it is not weak but **identically vacuous**. Same paper shape, one axis over: they
  buy a theorem with an assumption about the data, we buy one with an assumption about the module. And where
  their assumption is only checkable in expectation, ours is checkable exactly, in one forward pass. That is
  the analogy to open with, because it says *we did the thing your paper does*, not *we measured some things*.
- **Oz et al. 2024.** An MLP on frozen DINO/CLIP embeddings is exactly the head surface where the channel
  survives. Oz trains from scratch (unknown `W_0`) and needs KKT; a head fine-tuned from a public head exposes
  the recorded-embedding span exactly. State both cases. **Sharper after the counting rule lands:** Oz's setting
  is not merely compatible with the live surface, it *is* the live surface — `P = 1`, the one configuration in
  which the recipe-free channel exists at deployed rank. His paper is the argument that the head-on-frozen-
  features regime is the realistic one; ours is the statement of exactly how much that regime leaks and when.
  The two compose rather than compete.
- **Gronich & Vardi 2026.** Momentum keeps the gradient subspace, so the closure should extend to momentum SGD
  (unmeasured). Under Adam only the KKT route survives, which is their regime. One line, no more.

## 6. Solvency, in one table (revised end of day, after the corrections landed)

| claim | standing | goes to the supervisor as |
|---|---|---|
| closure, seed reduction, faithfulness | theorem, verified by hand | the reduction |
| replay count and its sharp line | theorem plus measured sharpness | the identifiability statement |
| imprint law, "what leaks is what it had to learn" | derived plus measured, robust | the mechanism |
| **counting rule / where the channel exists** | **theorem-shaped (a proposition with hypothesis A7) + measured on four pretrained models** | **the defender's meter — evaluable from architecture and batch size before any release exists** |
| certificate, kernel count, simplex cap | theorem **at `N' = N`**; approximate otherwise, at a measured scale | the tool, correctly scoped |
| letters from random starts | experimenter-verified; attacker-verifiable fraction **unmeasured** | an observation, not an attack rate |
| replay from attacker-buildable starts | **none of twenty** — clean, and for a principled reason | an open problem; SimuDy ahead on the start |
| membership vs LiRA | equal separation, fewer assumptions | one sentence |

### What changed today, and why it improves the pitch

1. **The counting rule moved from a caveat to a result.** It was absent from both theorem-first documents; it is now
   a proposition with its own hypothesis, placed *before* every claim that depends on it, carrying **both**
   directions — where it forbids a channel (transformer block linears, early conv, depthwise) and where it predicts
   one and the measurement returns it (the live ResNet cell, `rank C = 64−49 = 15`). Read only as a prohibition it
   would be special pleading; carrying both directions makes it a law. **This is now the strongest defender-facing
   output the project has**, and it is the one thing on the page a supervisor can check against their own
   configuration in a minute.
2. **The replay bound is clean.** "One start reached 0.9% but unverifiably" is withdrawn: the chart-relative
   definition puts it three orders above its floor, so it is a search failure and the bound is **0 of 20**. The
   honest number is also the simpler one.
3. **The verdict schema cannot now be used to overclaim.** Four cells, three counts reported separately, and
   `unverified-recovery` excluded from every attack rate by construction.
4. **One constant is still open**, and it is the only hole left: what it means for an image to *reach* its chart
   floor (`err_vs_chart_max ≤ c · chart_repr_err`). Being pinned against pre-chosen anchors with the disputed cell
   held out. Until it is, the dependent cells hold at *search failure* — the conservative reading.

### The one thing to lead with

Not a reconstruction number. **The counting rule**, stated as the proposition it now is, with the surface table
under it: the recipe-free channel is vacuous on the dominant deployment pattern and alive on heads, dense layers
and low-position deep convolutions — decided by architecture and batch size alone, before any release exists, and
never a privacy guarantee. That is a tool, which is what the supervision has been asking for; the reconstruction
cells are its demonstration, not the claim.


## 7. What the small cells are proof of concept FOR — the calibration

Written because it was nowhere: the individual facts are scattered across the three documents, but the statement
of *what they establish and what they do not* is not in any of them, and it is the first thing a supervisor asks.
All of it is existence, none of it is magnitude.

### They do establish, at small scale

1. **The channel can be read from nothing.** No recipe, no labels, no `N`, no start near the truth — random
   public-scale coordinates only. Digits at `k=6` and letters at `k=16`/`k=32`.
2. **The result can be self-verified without ground truth, and this is the part to lead with.** Not the landing
   rate — the *agreement* between certification and landing:

   | cell | starts certified at the floor | starts that genuinely landed |
   |---|---|---|
   | digits, `k=6`, below the line (two batches) | 0.51 / 0.4455 | 0.51 / 0.4455 |
   | letters FP64, `k=16` | 0.824 | 0.824 |
   | letters FP64, `k=32` | 0.384 | 0.384 |
   | digits, `k=10`, **at** the line | **0.2995** | **0.0475** |

   Below the line certification produced **no false positives** in any cell; at the line it certified a third of
   starts against a twentieth that landed. That contrast is the demonstration. It is what separates an attack from
   a demonstration, because the attacker can tell success from failure without ever seeing the data.
3. **The predicted condition transfers off the synthetic bench.** On a pretrained ResNet with one private
   photograph the counting rule predicted `rank C = 64 − 49 = 15`; the measurement returned 15, with a
   false-positive rate of 0.0000 across 20 draws and the adaptive-optimiser control vacuous as required.

### They do NOT establish

- **That any of this is a privacy attack.** Every from-nothing recovery returns the *projection onto the
  attacker's chart*, exact to machine precision, while that projection sits far from the raw image.
  **Recognisability was never assessed.** This is the load-bearing gap: if reaching a chart floor never yields
  something a person would identify, every theorem survives and the privacy framing does not.
- **That self-verification survives realistic arithmetic.** Train the same letters in FP32 rather than FP64 and
  16.2% of starts still land at `k=32` while **none** reach the floor. Audited against the correct per-image
  floor, the lead cell gives ~3% certifiable against ~29.7% landing. Found and attacker-confirmable come apart
  exactly where the arithmetic gets realistic.
- **Scale.** MLP heads, `N ≤ 14`, one real-network cell at a single private example.

### The chaining question, split by what is provable

- **Provable, and near-trivial:** any candidate reproducing the release also satisfies the certificate, so
  replay's zero set ⊆ the certificate's. Between the two lines the outer set is a manifold of dimension
  `k − (r − N')` per example, and replay's extra `m−1` conditions per example are exactly what would pin a point
  on it. This is counting, and it is derived.
- **Not provable from here:** whether the certificate's solution manifold *intersects replay's basin*. That is a
  geometry question about the basin, the least-characterised object in the framework — its edge was never located,
  and the one measured fact about it is negative: eliminating the seed in closed form shrank the search
  substantially and **did not widen the basin** (job `605718`), so the obstruction is the landscape, not the
  dimension. Chaining is well-posed, untested, and gated on what we understand least. **Do not claim it will work.**

### One place the routes provably decouple

Under an adaptive optimiser the closure fails, the certificate ceases to exist entirely, and replay still
identifies. The two are not two views of one object that must stand or fall together — which is the kind of
statement that makes a framework look like a framework rather than a single trick.

### The sentence this licenses

*Not* "we have a criterion". **"We have a derived condition that has predicted correctly everywhere it has been
tested, including on a real pretrained model, and small-scale existence demonstrations that the channel it
predicts can be read and checked from nothing. Whether what comes back is an image is the next measurement."*


## 8. The pitch after the A8 ruling (2026-09-06) — supersedes §6's "what to lead with"

Ruled by the math/science approver (yoado-0a) in `notes/math_rulings_2026-09-06.md`, R1 and R8. Three of my own
claims were corrected; the net effect **strengthens** the pitch, and §6 above now over-promises in one specific way.

### What was wrong in my derivation, recorded because I relayed it

- **"The representer route's degeneracy is strictly larger."** Wrong. The two zero sets are essentially the same
  size; their kernels differ only by training drift in `A`, and **neither contains the other** — the certificate
  *discards* the per-image error-size conditions the representer keeps and gains kernel slack in exchange. The
  unification is the **shared breaker**, not a size comparison.
- **"Free parameters that can absorb a recombination."** Hand-waving until named. The well-posed quantity is the
  **local degeneracy dimension at the truth**: `dim ker` of the Jacobian of the full residual w.r.t. all unknowns,
  evaluated at ground truth, modulo permutations. It is already `d_parallel` from primitive 2. One SVD, either route.
- **The oracle-coefficient arm is not a sound discriminator.** It removes gauge, scale and conditioning together,
  so a success is unattributable; the scale confound is live; and its own floor is worse than the free arm's above
  `T=1`. Replaced by three oracle-free tests: `N=1` vs `N=2` at `T=1` (superposition empty at `N=1`,
  misspecification zero at `T=1`, so the contrast isolates superposition), a direct measurement of `d_parallel`,
  and fitting the returned points in the span of the truths so superposition is **observed** rather than inferred
  from a failure.

### What survived, and is now a lemma

Both routes' zero sets are *the private span, plus a kernel, intersected with the chart*, and **one condition breaks
both**: the chart meets the private span only at the private points. That is **A8**. A defence satisfying it closes
both routes at once; a setting violating it defeats both.

### THE CORRECTION THAT CHANGES THE PITCH

**The counting rule is ONE-SIDED.** It is sound when it says the channel is **closed** and **silent** when it says
open, because closure needs A8 as well. §6 calls it "the defender's meter"; as written that promises more than it
delivers, and a defender who checks their configuration, reads "open", and relaxes has been misled.

This is not a demotion. A8 is *also* architecture-checkable, so **the tool becomes two checks rather than one**, and
it yields a positive architectural statement the project did not have:

> An adapter on the **input layer** read through a **linear chart** is provably non-identifying for two or more
> images **by either route**. Identifiability begins when a nonlinearity separates the chart from the adapted
> layer's input.

Checkable before any release exists, **positive rather than prohibitive**, and new. That is a better thing to lead
with than a rule that only ever forbids.

### A confound that constrains the head-to-head, and my own prediction with it

The certificate arm is **separable** (one start solves one image; 8 of 8 accumulates over 200 starts); the linearised
arm is **joint** (one start places all eight at once). Joint success is bounded by the smallest per-image basin, and
measured basins are skewed from 53/89 down to 1/89, so the product over eight is ~1e-6 or worse. **`0 of 8` from 200
joint starts is expected even with perfectly identifying equations.** Superposition and arity predict the same row.
So my expectation that the linearised route fails at `T=1` from superposition may be registered as primary **only
with the `N=1` control attached**; registered bare it cannot be falsified by the run meant to test it.
