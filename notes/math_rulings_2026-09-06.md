# Math/science rulings — 2026-09-06 (yoado-0a, approver lane)

Standing role: every scientific or mathematical claim gets a ruling here before it lands in RESULTS.md, STATUS.md,
LESSONS_LEARNED.md, the .tex, the deck, or anything for Gal. Verdicts are APPROVE / REVISE / REJECT / HOLD.
Rulings are recorded **here**, at a citable location, not only in messages (LESSONS_LEARNED item 6: a flag carried
by a person dies with that person's context).

---

## R1 — APPROVE, and UPGRADE from hypothesis to lemma: `k < r − N'` is necessary, not sufficient

Claimed by yoado-64 (CLAIM 1) and yoado-72 independently. Both stated it as a hypothesis supported by three
measured cases. It is stronger than that: in the affine case it is a **theorem**, and the demonstrations are then
illustrations rather than evidence.

**Lemma (blend degeneracy).** Let `Ψ = φ ∘ G : R^k → R^n` be the composition from chart coordinates to the input of
the adapted layer, and let the privates be on-chart, `h_i = Ψ(z_i)`. If `Ψ` is affine then for every affine
combination `Σ c_i = 1`,
```
        Ψ(Σ c_i z_i) = Σ c_i h_i ∈ col H ⊆ ker C     hence     C Ψ(Σ c_i z_i) = 0  exactly.
```
So the certificate's zero set on the chart **contains an affine subspace of dimension `min(N−1, k)` through the
truths**. The truths are not isolated zeros, at any `k`, however far below `r − N'`.

**Why the count does not see it.** The capacity line comes from a transversality count:
`dim(M ∩ ker C) = k − (r − N)` generically, negative below the line, hence isolated. That count is valid off
`col H` and **invalid on it**, because `col H ⊆ ker C` is forced by construction, not generic. The blend
directions are exactly where genericity fails.

**The condition is not "nonlinearity"; nonlinearity is how you buy it.** State the hypothesis as

> **(A8, transversality to the private span).** `M ∩ span{h_i} = {h_1,…,h_N}` — the chart meets the private span
> only at the private points.

An affine `Ψ` violates A8 automatically for `N ≥ 2, k ≥ 2`. A nonlinear `Ψ` satisfies it generically whenever
`k + N < dim(feature space)`, which is why moving the adapter behind two GELU layers fixes it
(`k + N = 40 ≪ 256`). Depth is a **means**, A8 is the condition. Report it that way — a lane that writes
"the composition must be nonlinear" will later be embarrassed by a nonlinear chart that still blends.

**The dimension count was never the binding constraint here, and that is the sharpest way to say it.** On the pixel
layer `k + N = 40` against `n = 3072`; the count is satisfied by two orders of magnitude and the cell is still
degenerate. So this is not a capacity shortfall dressed up — it is a *different* failure, and it has its own name
already: `d_∥` in primitive 2 of the framework (`rank L_x = min(r − N, k − d_∥)`, stated "a.s. for each fixed x").
The affine chart is the measure-zero exception that the "a.s." quietly excluded and that we then built on.
**Record it as `d_∥ > 0`, not as a new lesson** — it costs the framework nothing and gains a connection.

**Corollary, APPROVED as stated: the sanity checks are necessary and never sufficient.** `‖CH‖`, `rank C`, the
excitation gap and the residual at the truths are functions of `C` and `H` alone. **None of them mentions the
chart `G`.** A quantity that does not mention `G` cannot certify a property of `M ∩ ker C`. That is a one-line
proof and it is much stronger than "we measured that they all held in the degenerate case".

**REVISE, at source (yoado-64):** `experiments/cifar/RESULT.md` line 15 says the pixel-layer replica recovers
**0 images**; the study's own table at line 44 and `figures/cifar_charts/table.md` line 3 say **2 of 8**, 30 of 400
starts. The headline table overstates its own result. Fix the headline, not the table.

**Offered, not required — an attacker-available isolation test.** At a found point `z*`, form `J = C · DΨ(z*) ∈
R^{r×k}`. If `rank J = k`, `z*` is an isolated zero and the recovery is identified; if `rank J < k` the zero set is
locally positive-dimensional and `z*` is a blend. This needs only the release and the chart — no ground truth, no
`H` — so it is a genuine attacker-side certificate of isolation, and it is `d_∥` measured per start. It would have
caught the pixel-layer cell from inside the attack.

---

## R2 — APPROVE as an ORDERING statistic; REVISE the wording; REJECT the frozen-`A` redefinition

`σ_i = ‖B_T ã_i‖` with `ã_i` the normalised component of `a_i = A₀h_i` orthogonal to `span{a_j : j≠i}`.

- **Basis-independent:** yes. `ã_i` is defined by a canonical orthogonal projector, not by a chosen basis. Clears
  the per-image-quantities rule.
- **Per-image:** no, and the measured 28–42% cross term is the proof. Every claim `σ_i` supports is **monotone**
  (`σ_i` against `u_i` over 30 orders; recovery error flat in `σ_i` over 11). A ≤40% multiplicative contamination
  cannot invert an ordering that spans decades, and the own-term ordering was measured to agree. So the claims
  survive; the **magnitude** does not. Wording: "record strength, up to a measured O(1) cross term" — never
  "image i's record strength is X".
- **REJECT frozen-`A`.** Freezing `A` changes the recipe, so a frozen-`A` σ is a different experiment's quantity,
  not a cleaner measurement of this one (ground rule 4 — and it applies to diagnostics, not only to releases).
  Decompose within the run instead: you already have `C_i` from the traced release, so report `σ_i^own = ‖C_i ã_i‖`
  as primary and `σ_i` as the release-computable proxy, with the ratio as the caveat.
- **One check I want before this travels:** the cross-term share is reported as a range over cells (0.3–37%,
  28–42%). Report it **for the extreme low-σ recorded cells specifically**. If the cross term dominates at
  `σ_i ≈ 4e-10` the way it does at `4e-16` for the invisible example, then the bottom of the "eleven orders" is
  partly measuring other images' drift, and the span shortens. Cheap, decisive, and it is the number the claim
  rests on.

**Adjudicating yoado-c6's REVISE against this claim (both lanes are partly right).** c6 says the eleven orders
inherit the projector tolerance that defined the recorded count. That bites **wherever `N'` or a "recorded set"
appears**, and there c6 is right and a tolerance ladder is owed. It does **not** bite on the claim's real content,
which can be stated tolerance-free:

> the joint distribution of (σ_i, recovery error) over 32 example-cells is **bimodal with a four-order gap**, and
> the low mode spans eleven orders of σ_i.

That is a statement about measured pairs. No spectrum is cut to make it. Restate it in that form and the objection
dissolves; keep the ladder for every sentence that says `N'`.

---

## R3 — HOLD on "NTK free-c recovers 0 of 8 where the certificate recovers 8 of 8". Three findings, one is a confound that would have made the headline wrong

**(i) The 22.9 is not a floor for the free arm, and the script already computes the right one.**
`ntk_vs_certificate.py` (docstring, and `model_floor_at_truth`) defines the floor as
`min_R ‖target − model(R, H_true)‖_F / ‖target‖_F`, least squares in `R`, closed form. That **is** the honest
floor: the free attack restricted to the true images. The 22.9 / 94.4 / 72.1 numbers quoted to me are the
**oracle-coefficient** misfit, which bounds the oracle arm and says nothing about the free arm. yoado-64's own
narrow reading is the correct one — adopt it and drop the wide one.
- Answer to (a): correct, `lin_err` bounds the oracle arm only. Use `model_floor_at_truth`.
- Answer to (b): normalise by `‖ΔW‖` (or `‖B_T‖`), which is what the script does. Symmetric normalisation is for
  comparing two objects of unknown scale; here you are asking what fraction of the release the model explains.
  Separately, a relative error of **22.9 is mostly scale, not direction** — residuals decay, so
  `Σ_t r_i^(t) ≪ T·r_i^(0)` and the one-step model overshoots. Report `min_α` over a scalar and the cosine; if the
  direction is fine, that is one more reason the free arm absorbs it entirely.
- Answer to (c): neither `lin_err` nor `‖ΔW‖/‖W₀‖`. The regime coordinate for a **free-coefficient** route is the
  misspecification free coefficients cannot absorb, and for the `dW` form it is closed-form:
  `‖ΔW (I − P_Φ)‖_F / ‖ΔW‖_F`, `P_Φ` the projector onto `span{φ_i}`. Pre-register the crossing in **that**.

**(ii) At T = 1 the `lora` form is exact, so a failure there cannot be misspecification.** `B_0 = 0` kills the `A`
gradient, so `A_1 = A_0`, and `B_1 = −(η/N) Σ_i d_i (A_0 φ_i)^T` is *exactly* the model with `r_i = −(η/N)d_i`.
Floor zero. The docstring already says this and it is right.

**(iii) THE CONFOUND — the head-to-head as built compares two search arities, not two equations.**
The certificate arm is **separable**: one start solves a `k`-dimensional problem for **one** image, and `8/8` is
accumulated over 200 starts. The NTK arm is **joint**: one start must place all `N` images at once in an
`N(k+m)`-dimensional space, and it scores `0/8` if it fails to land all of them together. A single start's joint
success is bounded by the *smallest* per-image basin, and behaves like the product if the basins are near
independent. The project's own measured per-image basin fractions are heavily skewed — 53/89 down to 1/89 in the
letters cell, 414/2000 down to 11/2000 elsewhere. **A product over eight such fractions is `1e-6` or smaller, so
`0 of 8` from 200 joint starts is the expected outcome even if the equations were perfectly identifying.**

So the comparison cannot presently distinguish "the linearised equations carry less information" from "one search
is separable and the other is not". The script's own docstring names separability as the mechanism; the **claim
wording** is what would mislead. Required before this is stated in any form:
1. **`N = 1` cell.** Arity equal, superposition empty, misspecification zero at `T = 1`. If the free arm recovers
   at `N = 1` and fails at `N = 2` with everything else fixed, the obstruction is superposition — and that is an
   oracle-free demonstration.
2. **Report the certificate under the same joint burden** — the fraction of single starts that recover *all eight*.
   Predicted small. If the certificate also scores ~0 under joint scoring, the headline is arity, full stop.
3. **State the advantage for what it is:** the certificate is separable and the representer route is not. That is
   a real, structural, quotable advantage. It is not "a better-specified equation".

**On the last question (would 0/8 at T=1 prove superposition):** it would prove the obstruction is not the
linearisation. It would **not** prove superposition until (1) and (2) are in, because arity is the competing
explanation and it predicts the same row.

---

## R4 — REVISE the framing: the "exchange rate" is an identity, and the measurement is that it is achieved

The attack returns the chart projection. Its error against the raw image is therefore **at least** the chart's own
representation error, and landing at a bar τ requires chart error < τ. The 1:1 rate and the "≈2% for exact
landing" are consequences of the definition, not discoveries. Stating a derivable identity as a measured exchange
rate invites a supervisor to ask what was learned.

What **is** measured and worth saying: the search **attains** the chart floor rather than falling short of it
(attack SSIM equals the ceiling at every ε), and the SSIM ladder translates chart error into perceived fidelity
against a flat same-class control near 0.47. Lead with attainment; carry the rate as its corollary.

---

## R5 — APPROVE with wording: added classes survive over-training

Numbers internally consistent, controls present (wrong-release 0/200 both). Two wording requirements: images found
falls in two of four classes (8/8 → 7/8), and the certificate residual at the truths rises from 5e-15 to
4e-14…2e-10 — five orders. "Survives" is right; "survives undiminished" would not be. The mechanism as written
(over-training suppresses what the model already knows; the new class is what it did not) is the imprint law and is
consistent with the rest of the ledger.

---

## R6 — SSIM in the CIFAR `k = 32` cell: keep it, bound it, never headline it

The honest form is the one already written, with two hard constraints. The attack scores 0.58, the chart ceiling is
0.58 and the same-class control is **0.60** — SSIM there favours the control. So:
1. The raw 0.58 may **never** appear in a summary table without the 0.60 control in the same row. A reader
   scanning a column of SSIMs will otherwise average it against cells where SSIM is the evidence.
2. In that cell report SSIM only ceiling-relative (attack/ceiling = 1.00, i.e. the search attains the chart) and
   state plainly that identification rests on image error (1e-14 against 0.25) and the residual ranking.
3. Window-3 SSIM on 32×32 blurred projections is the wrong instrument for "would a person identify this". The
   project's own calibration note names recognisability as the load-bearing untested gap. The measurement that
   closes it is an **identification test** — can the recovered image be matched to its source among a large
   candidate pool by a person or a perceptual metric — not a better similarity scalar.

---

## R7 — APPROVE with two amendments: the verified-coverage claim (yoado-c6, jobs 857268 / 303157)

Witness validity (experimenter, per-image floors from a from-truth solve) and precision-at-k (attacker, blind
ranking) are correctly separated and correctly named; withholding the attacker figure until the disjoint-release
null returned was right, and the null is the correct control for **metric validity**.

1. **The null is not the informativeness control, and both are needed.** The disjoint null answers "does the
   landing metric fire on anything that converges" — 0.00, good. It does not answer "does the ranking do work".
   That comparison is precision-at-k against the **base rate**, `89/300 = 0.297`. Ranking lifts 0.297 → 1.00.
   State both; they are different claims.
2. **Drop the `/8` denominator on the attacker side.** The attacker does not know `N = 8`; the projector reads
   `N' = 11`. The attacker-side statement is "5 distinct private images in the top 50 starts, at 300 starts", with
   no denominator. `5 of 8` is the experimenter's row.

Scope already carried by the lane and endorsed: on-chart, "at B starts", skew.

---

## Cross-check assignments (nobody audits their own rows)

- **yoado-64** owns the CIFAR and record-strength experiments and the head-to-head. Audits **yoado-c6's** step126
  attacker arm at the rows.
- **yoado-c6** owns the two published pages, the .tex, and the coverage measurement. Audits **yoado-64's** CIFAR
  headline table at the rows (R1's 0-vs-2/8 discrepancy is exactly the class of defect this catches).
- **yoado-72** owns the narrative and the audit note. Gates what leads and what reaches the user; relays no
  unapproved number as settled; does not verify numbers itself and marks as unverified anything read from prose.
- **yoado-0a** (this lane) approves the science and math, and records every ruling here.

---

## R8 — Ideation ruling (yoado-72): the two degeneracies ARE one phenomenon. Correct "strictly larger", and replace the oracle discriminator

**The joining is right, and it is provable rather than suggestive.** Take the LoRA-aware representer form, which is
the fair one: `B_T = Σ_i r_i α_i^T` with `α_i = A_T φ(x_i)`, i.e. `B_T = R α^T`, `R ∈ R^{m×N}`, `α ∈ R^{r×N}`. With
`rank B_T = N`, every exact solution is `R' = R S`, `α' = α S^{-T}` for `S ∈ GL(N)`. So the representer route's
zero set is **exactly a `GL(N)` gauge orbit on the private span**, `N²` dimensions, and the only thing that breaks
it is the requirement that each `α'_i` be realisable on the chart: `α'_i = A_T Ψ(z'_i)`.

Now the certificate. Its zero set per point is `M ∩ ker C`, `ker C = col H ⊕ ker A₀`.

Set them side by side:

| route | slack per candidate | broken by |
|---|---|---|
| certificate | `col H ⊕ ker A₀` | the chart meeting the span only at the privates (A8) |
| representer, free coefficients | `col H ⊕ ker A_T` | the same condition, pushed through `A_T` |

**So they are the same phenomenon and, contrary to the suspicion, essentially the same size** — the two kernels
differ only by the training-induced drift in `A`. Correct "strictly larger" before it travels. What actually
differs is the *other* direction: the certificate **discards** the `m−1` per-image error-size conditions the
representer keeps, and gains `ker A₀` slack in exchange. Neither degeneracy contains the other. **The unification
is the shared breaker, A8 — not a comparison of sizes**, and the shared breaker is the stronger statement anyway:
one condition governs both routes, so a defence that satisfies A8 closes both at once and a setting that violates
it defeats both.

**"Free parameters that can absorb a recombination" — make it countable.** It is well-posed once you name it as
the local degeneracy dimension at the truth:
```
        d∥  =  dim ker J ,      J = Jacobian of the full residual w.r.t. all unknowns, at the ground truth
```
(modulo the discrete permutation symmetry). Certificate: `J = C·DΨ(z)`, `d∥ = k − rank(C DΨ)`. Representer: `J`
includes the `∂/∂r` block, and `d∥ ≥ N²` whenever the gauge directions stay on the chart — which is exactly the
affine case, and exactly zero for a generic nonlinear `Ψ`. **One number, computable in both routes by one SVD, and
it is already the framework's `d∥` from primitive 2.** Use it and the hand-waving disappears.

**The oracle-coefficient arm is NOT a sound discriminator — three reasons, and there are better ones available.**
1. It removes the gauge **and** the scale **and** the conditioning at once. A success cannot be attributed to any
   of them. The scale confound is live: the one-step residual overshoots by more than an order (R3).
2. Its own floor is worse than the free arm's at `T > 1` (that is what the 22.9 measures), so at the interesting
   step counts it can fail for a reason that has nothing to do with the question.
3. Standing user rule: oracle mode is an upper bound, never a result. As a labelled tertiary diagnostic on a cell
   already running it costs nothing — keep it if you like — but it cannot settle this.

**Use these instead, all oracle-free and all cheap:**
- **`N = 1` versus `N = 2`, everything else fixed, at `T = 1`.** Superposition is empty at `N = 1` and
  misspecification is zero at `T = 1`. Recovery at `N = 1` and failure at `N = 2` isolates superposition with no
  oracle anywhere. This is the discriminator.
- **Measure `d∥` at the truth.** If `d∥ > 0`, superposition is present by measurement and no arm is needed. If
  `d∥ = 0` and it still fails, the obstruction is the landscape.
- **Fit the returned points in the span of the truths** — the same blend diagnostic that cracked the CIFAR cell.
  Blends returned ⇒ superposition observed directly, rather than inferred from a failure.

**And the same arity confound in R3 applies to the prediction itself.** "The free arm fails at `T = 1` for `N > 1`"
is predicted by superposition *and* by the joint-search argument, which needs no degeneracy at all. So expectation
(iv) may be registered as **primary only with the `N = 1` control attached**; registered bare, it cannot be
falsified by the run that is meant to test it.

**One consequence worth having in the pitch.** If A8 governs both routes, then the architectural statement is not
about the certificate at all: **an adapter on the input layer, read through a linear chart, is provably
non-identifying for `N ≥ 2` by either route; identifiability begins when a nonlinearity separates the chart from
the adapted layer's input.** That is checkable from the architecture before any release exists, it is a *positive*
statement rather than a prohibition, and it is new.

---

## R3-AMENDED — I withdraw the scoring premise of R3. yoado-64 caught it in the code; the rivalry it named collapses, and two solver confounds replace it

Recorded here rather than only in a message, and left beside the original rather than editing it away, because the
original was relayed to three lanes and one of them held a claim on it.

**What I got wrong.** R3 asserted that the linearised arm "scores zero if it misses any", making `0 of 8` the
product-of-basins artefact of a joint search. **False, verified in `run_ntk`:** candidates are
`Xc.permute(1,0,2).reshape(D, P*N)` and `found` counts truths whose minimum error over **all `P·N` slots** is under
`1e-2`. It is a per-image union metric, the same one the certificate arm uses.

**And the correction runs against me, which is the part worth stating.** The linearised arm contributes
`P·N = 1600` candidate slots; the certificate arm contributes `P = 200`, one per start. So the arm that lost had
**eight times the draws**. The comparison does not flatter the certificate through scoring; if anything it
understates it.

**What survives, restated.** The asymmetry is in the optimisation, not the scoring: slots descend on a coupled
objective, so a slot that would land alone can be dragged off by the other seven. **That is not a rival to
superposition — it is one of superposition's two mechanisms.** The live distinction is instead the project's own:

| | mechanism | signature in rows already logged |
|---|---|---|
| (a) identifiability superposition | `d∥ > 0`: the zero set contains recombinations, so the minimiser is a blend | residual **at** the free-coefficient floor, images wrong ⇒ **alias** |
| (b) landscape superposition | coupled descent never reaches a zero that exists | residual **above** the floor ⇒ **search failure** |

**This is readable from rows in hand, with no new run.** Compare `ntk_res.min()` against `model_floor_at_truth`.
The reported plateau of 0.46 sits far above any floor, and in the `lora` form at `T = 1` the floor is **zero by
construction** — so on present evidence this is (b), a search failure, and the honest headline is *"the joint solve
does not converge at this budget"*, not *"the linearised route carries less information"*. A landscape claim is
beatable by a better solver, and a reviewer will say so.

**Two solver confounds must be removed before any headline, and the first is a real bug-shaped hazard.**
1. **`Rc` is initialised at zero, which makes the latent gradient identically zero at step 0.** The model is
   `pred = Rc · F(Z)`, so `∂pred/∂Z ∝ Rc`. At `Rc = 0` the latents receive no gradient at all and the early
   trajectory is entirely coefficient-driven. The certificate arm has no such pathology — its gradient is
   generically nonzero at a random start. **The two arms are not on equal footing at initialisation.**
2. **The fair algorithm is variable projection.** The model is *linear* in `Rc`, so eliminate it in closed form and
   optimise over `Z` alone. That removes the zero-gradient start, removes the two-block scale mismatch under a
   single Adam learning rate, and cuts the unknowns from `N(k+m)` to `Nk`. The eliminated objective is
   `‖(I − P_{F(Z)}) target‖` — **the same shape as the certificate's**, which is what makes the head-to-head
   apples-to-apples. This is the strongest fair version of the linearised route, and only if it still loses is the
   comparison worth stating.

**Answer to "a failure at `N=1, T=1` would be none of the three and I would not know what it is".** It would be
**conditioning or parameterisation**, and the two items above are the first suspects, in that order. Diagnostics:
the condition number of the joint Jacobian at the truth, and per-block learning rates or variable projection.

**The `N = 1` controls (jobs 307866, 307867) remain correct and are still the discriminator** — they are simply now
separating (a) from (b) and from conditioning, rather than separating superposition from a scoring artefact that
does not exist.

---

## R9 — Ruling on the head-to-head: decline all three search-based forms, and run the identifiability comparison instead, which has no solver and therefore no conflict of interest

yoado-72 offered three options — do not compare; run their arm at its published implementation and
hyperparameters; or report ours alone and cite theirs. **My ruling is a fourth, and it is available precisely
because the objection is about *search*.**

### The scientific reason to decline a search comparison, which does not depend on anyone's motives

A search comparison between the two routes has **no controlled variable**. They differ in objective, in arity
(one image per start against `N` at once), in solver (Levenberg–Marquardt against Adam), in parameterisation
(latents alone against latents plus a coefficient block), and in initialisation (generic against a first step with
zero latent gradient). Whatever such a run returns, no single difference is isolated, so no version of it answers
the identifiability question it was built to answer. That is a design fact and it holds even with perfectly
disinterested tuning. **The conflict of interest is a second, independent reason, and it is real** — R3-AMENDED's
zero-initialisation confound is exactly what it produces: not misconduct, a handicap nobody was motivated to find.

### Option 2 is the trap, and it is worth naming specifically

"Run their published implementation at its published hyperparameters, change nothing" **looks** maximally
scrupulous and is the worst of the three. Those hyperparameters were chosen for that method's own regime — a
binary MLP under full fine-tuning at its own data scale — not for a LoRA release on a head read through a public
chart. Transplanting them and reporting the failure is a straw man **with a paper trail**, which is harder to walk
back than an untuned run, not easier. And the supervisor is the lineage's author: he will know within a sentence
that those settings were never meant for this. Do not do this.

Of the three as offered, **option 3 is correct**: report ours, cite theirs as published, compare nothing.

### But there is a comparison that is fair by construction, and we should make it

The conflict lives entirely in the search. **Identifiability does not need a search.** It is a rank question
evaluated at the ground truth: no optimiser, no hyperparameters, no basin, no budget, nothing to tune, and nothing
a better solver can overturn. Compute, on the *same* release, chart and images:

| route | quantity | cost |
|---|---|---|
| certificate | `d∥ = k − rank(C · DΨ(z*))`, per image | one SVD of an `r×k` matrix |
| representer, free coefficients | `d∥ = dim ker J` at `(Z*, R*)`, `J` the Jacobian of `R·F(Z) − target`, joint over all `N` | one SVD, `N(k+m)` columns |

Read out as **"is the truth a locally isolated solution, and if not, by how many dimensions"**. The two kernel
dimensions are not directly subtractable — the certificate's is per image and the representer's is joint over the
set, and the gauge contributes up to `N²` to the latter — so compare the **isolated / not isolated** verdict and
report both dimensions beside it.

**A8 predicts the answer, which is what makes this worth running.** If both routes are broken by the same
condition, then wherever A8 holds, both should come back isolated, and wherever it fails (an affine composition),
both should come back degenerate. **That is a negative comparison result and it is far stronger than a horse
race**: it says the two routes carry the *same* identifiability and differ only in search separability. It is also
exactly the sentence yoado-72 wants — "the two routes answer different questions under different conditions" —
made quantitative instead of diplomatic, and it cannot be attacked as a tuned foil because there is nothing in it
to tune.

**Keep jobs 307866 / 307867 (the `N = 1` cells), with their purpose relabelled.** They are no longer producing a
comparison. They diagnose *our own* understanding of superposition — whether the coupling we attribute the failure
to is the coupling that is actually there. That is worth having and must not be presented as a head-to-head.

### Standing conditions on any mention of the other route

1. Its search performance is **not reported as a number by us** in any form.
2. Where our result needs a foil, the foil is `d∥`, not a landing rate.
3. Separability is stated as our route's structural advantage — one start returns one image and the other `N−1`
   never enter — which is true, checkable from the objective, and needs no run at all.

---

## R10 — Process, adopted at yoado-72's proposal and generalised: rulings carry a register, not just a source

yoado-72's remedy for R3 was that a ruling should carry a provenance mark saying which artefact was read. Adopted,
and the general form is sharper than the mark:

> **A ruling about what an experiment DID requires the artefact that did it — the function, or the row. A ruling
> about what the mathematics IMPLIES requires no artefact, but must state its hypotheses.**

R3's failure was applying the second register to a first-register question: I reasoned from a docstring that was
accurate about mechanism and silent about the metric, and issued it in the confident voice. The docstring is
neither the function nor the row; it is prose beside a number, which is the object this project has now been
misled by five times.

Every ruling here therefore carries one of: **[read: function]**, **[read: rows]**, **[derived]** (with its
hypotheses named), or **[read: prose — provisional]**. Retro-marking the rulings above: R1 [derived], R2 [derived
+ read: rows], R3 [read: prose — WITHDRAWN], R3-AMENDED [read: function], R4 [derived], R5 [read: rows], R6 [read:
rows], R7 [read: rows], R8 [derived], R9 [derived].

---

## R11 — Adjudicating yoado-c6's CIFAR row audit. All four findings accepted; F3 inverts a published argument, F4 corrects a test I offered, and the decisive cell was never run

**F1 — upheld, and I confirmed it at the row myself** [read: rows].
`experiments/cifar/charts/L1_ae_lm_onchart_k32/result.json`: `starts 400, landed 30, images_found 2,
landings_per_image [0,29,0,1,0,0,0,0], top20_by_residual_landed` all ones. The detail table is right, the headline
is wrong.

**F2 — accepted.** §1's "379 of 400" is the original replica, a different run from the §2 table row. Both numbers
are individually right and the paragraph joins them into one cell. Split them.

**F3 — accepted, and it inverts the argument the paragraph makes.** The distinction the write-up lost is between
two different objects:

| object | certificate residual | what it shows |
|---|---|---|
| the **ideal blend** (exact linear combination of the privates) | 2.86e-14, **equal to the truths'** | the lemma, confirmed numerically |
| the **attractor the solver reached** (mean of 379) | **5.90e-03**, 2.1e11 × the truths' | the solver stalled *near* the blend subspace, never on it |

**The lemma is untouched and does not need the measurement** — the ideal blends are exact zeros, so the truths are
not isolated at any `k`, proved rather than observed. What does not survive is the empirical sentence carrying it.
Consequences, all required:
- "the blend's own certificate residual is 3e-15 — as low as the truths'" is **withdrawn** as written. It is true
  of the ideal blend and false of the reached attractor, and the paragraph means the second.
- **99.4% becomes 98.9%, with its formula stated.** `1−(r/x)²` gives 0.9886 and `1−r/x` gives 0.893; a
  normalisation-dependent number that does not reproduce under any tried normalisation must carry its definition.
- **The failure on the pixel layer costs COVERAGE, not PRECISION.** The 14 landings sit at ~1e-06 against the 379
  collapsed at ~5.8e-04, a factor of ~600, which is exactly why top-20 is 20/20 and images found is 1–2 rather
  than 0. The page may not say the checks were defeated by a blend sitting at the floor, because in this run
  nothing reached the floor.

**And the reconciliation, which points at a cell that was never run.** The tested cell used a **conv-autoencoder**
chart, so `Ψ = φ∘G` is **nonlinear** and the blend subspace lies only *approximately* in `M`. The lemma's exact
degeneracy therefore does not apply to it — which is precisely why the solver stalls at 5.9e-03 instead of
reaching 2.9e-14, and why the residual still separates.

> **The affine case — the pixel layer read through a PCA chart — has never been run.** The charts directory
> contains `L1_ae`, `L2_ae`, `L2_pca`, `L3_ae`, `L3_oracle`: **no `L1_pca` cell exists.**

That single cell is the lemma's direct test and the only configuration in which **precision** should fail. Its
pre-registration writes itself: exact zeros reachable at the blends, the residual ranking **not** separating, the
top-20 populated by blends rather than truths, and images found 0. It is cheap, it uses machinery already built,
and it is the honest way to recover the strong form of "necessary and not sufficient" that F3 has just taken away.
`cifar_certificate.py`'s own docstring predicted it and it was never tested.

**F4 — accepted; my offered test was stated two-sided and only one direction holds** [derived]. With
`g(z) = CΨ(z)`, full column rank of `Dg(z*)` makes `g` an immersion at `z*`, hence locally injective, hence the
zero is isolated. **The converse fails:** a degenerate isolated zero (first-order term vanishing, isolation only at
higher order) also has `rank J < k`. So the test **certifies isolation and never certifies its absence**. State it
one-sided. The attacker-side value is entirely in the positive direction, so nothing is lost.

**And c6's strengthening of the R1 corollary is adopted over my own wording.** I wrote that the sanity checks are
insufficient because none of them mentions the chart. c6's form is an impossibility rather than an observation:

> Fix `C` and `H`. Then `G` may be varied freely while `M ∩ ker C` changes from isolated to positive-dimensional.
> Therefore **no function of `(C, H)` alone can decide isolation** — not these three checks, and not any check of
> that class.

That covers every future check anyone proposes, which mine did not. Use c6's.

**One cross-check obligation before the pages change.** F3 rests on a recomputation from tensors, and it reverses
a published argument. Under this lane's own rule that nobody audits their own rows, the **owner of those tensors
(yoado-64) confirms c6's recomputation** before the page is edited. Not because it looks wrong — the ideal blend
landing exactly on the truths' residual is strong internal evidence it is right — but because a reversal is the
one direction where a single reader is not enough.

---

## R12 — APPROVE the zero-floor theorem, and it implies something stronger: the two routes have the SAME ZERO SET, exactly

[derived; hypotheses named below. Checked step by step, and offered for checking — I was wrong once today in the
confident voice.]

**Their argument is correct.** `V^T H = 0` because `V` spans the complement of `col H`, so the untouched Gaussian
block never appears in `A_T H`:
```
        A_T H  =  XΩU^T H + c_T Y V^T H  =  XΩ (U^T H)        hence  col(A_T H) = col(X),
        B_T    =  P_T X^T                                     hence  row(B_T)   = col(X)   [rank P_T = N]
```
Target rows and design columns span the same subspace, so an exact `R` exists at **every** `T`. The floor is zero
by theorem, and their measurement (2.89e-16 at `T=1`, 8.05e-16 at `T=400`) is a confirmation, not the evidence.
**The regime-boundary framing is dead, correctly.**

### The stronger statement it implies

`B_T = R G^T` with `G = A_T Ĥ` is solvable in `R` iff `row(B_T) ⊆ col(G)`. With exactly `N` candidates,
`dim col(G) ≤ N = dim col(X)`, so containment forces **equality**, so every candidate reading lies in `col(X)`,
i.e. `P_{col(X)^⊥} A_T φ(x̂_i) = 0`. And `P_{col(X)^⊥} A_T` **is the certificate**. Conversely, if every candidate
is a certificate zero and their readings are independent, the span is `col(X)` and an exact `R` exists. So:

> **The free-coefficient representer model admits an exact fit if and only if every candidate is a zero of the
> certificate and their readings are linearly independent.**

**The two routes do not merely have degeneracies of the same size — they have the same zero set.** The certificate
is the *per-candidate* form of the representer condition; the representer is its *joint* form. R8's unification
was approximate ("essentially the same size, up to which copy of `A`'s kernel appears"); this is exact, and it
supersedes that wording.

**Hypotheses, all standing and all falsifiable:** `rank P_T = N` (framework hypothesis 3 — if it fails, `row(B_T)`
is a proper subspace of `col(X)`, `C` is contaminated, and the equivalence weakens rather than reverses); `Ω` and
`U^T H` invertible; exactly `N` candidates; `B_0 = 0` and step-invariant `H` inherited from closure.

### Two consequences that change what is planned

1. **R9's identifiability comparison will return "identical", by theorem rather than by measurement.** Run the
   `d∥` computation anyway as a check on the derivation, but the headline is now the equivalence itself, which is
   a better result than the measurement was going to be: *the two routes are the same test, written per-candidate
   and jointly.* That is the sharpest possible form of "different questions under different conditions" — they are
   not even different questions. All of the observed difference is solver and arity, which is exactly what
   yoado-64's rows show (residual thirteen orders above a floor of `2.89e-16`, i.e. a search failure).
2. **The one thing the joint form buys is DISTINCTNESS, and it is worth naming.** The equivalence needs the
   candidate readings *independent*. The certificate, applied per candidate, accepts `N` copies of the same image;
   the representer cannot. So the representer **structurally enforces coverage** where the certificate must earn
   it from basin luck — and coverage is exactly the certificate's measured bottleneck (53 of 89 landings on one
   image; 1–2 of 8 on the pixel layer). **That is a real complementarity rather than a consolation:** the
   certificate wins the search because it is separable, the representer wins distinctness because it is joint, and
   the hybrid is to generate candidates with the certificate and then use the representer's independence condition
   to select a spanning subset. That is the "assemble a joint start" step the chaining plan named and never
   specified.

### On the proposed 3× verdict bar — do not introduce a second threshold

`derived_verdict.py` already pins `FLOOR_FACTOR = 1.5` over 311 corpus rows. A `3×` bar here would be a **second,
inconsistent threshold for the same concept**, which is the exact failure the "a threshold must be scoped to the
physical scale it separates" lesson records, plus its corollary that a fix needing a threshold of its own must not
fill the gap silently. Required instead:
- report `residual / floor` as a **raw ratio on every row** — here it is `3.6e13`, and no bar is doing any work;
- use the existing **1.5** for the verdict field, so the project carries one threshold;
- mark any row whose ratio lands between the bar and `10×` as **undetermined**, not classified. A verdict that
  depends on the bar is a verdict we do not have.

Choosing "visibly loose rather than tuned" was the right instinct; the right expression of it is to make the bar
irrelevant and show that it is.

---

## R13 — The retrospective may ship. R9 does not reach it. But it predates R12 and gets the reason for the direction change wrong

[read: document, `notes/ntk_vs_certificate_comparison.md`, in full]

### The line yoado-72 could not draw alone

It is **not** "our numbers against theirs". It is **a claim about a method** against **a claim about our own
decisions**.

- *"Route X performs worse than route Y"* is a claim about X. It needs a controlled comparison, which does not
  exist here, and R9 forbids it.
- *"We tried X, these are the numbers we got, this is why we moved"* is a decision log. It is a claim about our own
  history, and the user is entitled to it — he is the one making the direction call, and **withholding our own
  measurements from him because they might be misquoted later would leave him deciding blind.** That is
  paternalism, not rigour.

Same numbers, different claim, different bar. **R9 does not reach this document**, whose subject is this project's
own prior phases — Experiment B, the anchor sweep, direct weight inversion, the gradient bridge — every one of them
our own implementation. The safeguard is framing and caveats, not suppression.

### But it predates R12, and its central explanation is now wrong

Four corrections, and they strengthen the conclusion while replacing its reason.

1. **"The linearization is never valid where the signal lives" is true only of the form we happened to pick.**
   Measured today (job 308859): the full-weight form's floor is 0.96 at `T=1` and 0.72 at `T=400` — misspecified,
   as the document says. The **LoRA-aware form's floor is machine precision at both ends**, and R12 shows that is a
   theorem, not a lucky cell. **The wall was our parameterisation, not the route.** Leaving this uncorrected tells
   the user a method is dead when what died was one way of writing it.
2. **"Exact rather than fitted" is not the distinction either.** Under R12 the representer route is also exact —
   the two routes have the **same zero set**. Whatever separates them, it is not exactness.
3. **The batch-size wall is a property of the JOINT SEARCH, not of the information**, and this is the document's
   best observation once reframed. Read its own table again: the NTK route collapses with `N` (0.922, 0.605, 0.536,
   0.252); direct weight inversion collapses with `N` **despite knowing the entire recipe** (0.57, 0.27, 0.15); the
   gradient bridge does the same. **Every route that solves for all `N` images at once collapses with `N`. The one
   route that solves per image does not.** That is a clean, unifying, and previously unstated axis, and it is
   supported by R12: since identifiability is shared, what differs is arity. The direction change was right, but
   we moved from **joint to separable**, not from fitted to exact.
4. **The last paragraph is the best thing in the document and is now partly out of date.** "The fraction of
   recoveries an attacker can verify end to end has not yet been measured" was answered today by yoado-c6's
   step126: on the letters cell, blind ranking gives five distinct private images in the top fifty starts at 300
   starts, precision 1.00 against a disjoint-release null of 0.00, base rate 0.297. Update it and keep the rest of
   the paragraph exactly as written — it is the most honest passage in the bundle.

### Conditions on shipping

- **Retitle it as a direction retrospective / decision log.** "Side by side" is the phrasing that invites the
  lift into a comparison slide later.
- **A framing line at the top:** these are our runs of our own implementations, the comparison is uncontrolled, and
  no number here is evidence about any method other than our own attempts at it.
- **Carry the verdict and the handicap.** R3-AMENDED's reading — search failure, not an information limit — and the
  variable-projection result showing the joint solver was handicapped by a factor of 19 at `T=1`. Omitting a
  handicap that flattered our own conclusion is the failure mode this whole day has been about.

### Ordering, for the ETA the user asked for

The comparison document is **cleared by this ruling** once the four corrections are in — minutes of editing, no
compute. The headline fix is trivial and already partly done. **The F3 tensor recheck is the only real blocker**,
and it belongs to yoado-64. If it will not clear quickly, ship the record-strength half alone: it is clean, its
reproduction gate passes, and its one open item is a caveat rather than a correction.

---

## R12-CORRECTED — yoado-64 checked the identification and found two limits. Both are right, and R12 as committed overstates

I asked them to check the step where I identify the complement projector composed with `A_T` as the certificate.
They did, and returned two refinements. **Both are correct and both narrow my claim.**

**1. The biconditional needs `N′ = N`, and `N′ < N` is this project's COMMON case, not its edge case.**
My "only if" direction ran: `dim row(B_T) = N` equals the number of candidates, so containment forces equality,
so each candidate reading lies in `col(X)`. That forcing **requires `rank P_T = N`**, i.e. every private image
recorded. With `N′ < N` recorded, `row(B_T)` is only `N′`-dimensional, the design has to cover just that subspace,
and `N` candidates have room to spare — **so candidates need not be certificate zeros at all.** The "if" direction
survives (independent certificate zeros span `col(X) ⊇ row(B_T)`, so an exact fit exists); the "only if" does not.

> **Corrected statement.** Certificate zeros with independent readings ⇒ exact representer fit, always.
> The converse holds **only when every private image is recorded** (`N′ = N`). Below that the routes' zero sets
> come apart, and the representer's is strictly larger.

This is not a technicality here: the recorded-versus-unrecorded split is the project's own central phenomenon, so
**the shared-zero-set result is the special case and the divergence is the general one.** R12's headline stands for
the fully-recorded cells and must carry `N′ = N` wherever it is quoted.

**2. The distinctness advantage is narrower than I claimed, and my "enforces coverage" was wrong.**
I wrote that the joint form structurally enforces coverage. It does not. `N` **independent blends** of the private
images satisfy the independence clause *and* are certificate zeros, so they are exact fits. **The representer
excludes duplicate candidates; it does not exclude blends, and it does not escape superposition.** Coverage is not
bought. Withdraw "structurally enforces coverage" and replace with "excludes duplicates".

The hybrid construction in consequence 2 survives, but for a smaller reason: it removes the certificate's
duplicate-landing waste, not its blend exposure.

**Process note.** This is the check working in the direction it is supposed to: I derived R12 an hour after being
wrong in the same register, flagged it as underchecked, asked the owner to verify, and two of its three claims came
back narrowed. The refinements are theirs and are recorded as theirs.

---

## R14 — The top-k breadth inversion (yoado-c6, job 304540). The finding is right; here is the statistic that replaces it

[read: rows, relayed; the mechanism is derived]

**The finding.** Distinct images in a fixed top-50 window falls from 5 at 300 starts to 2 at 3000, while the witness
arm improves to 8 of 8 and precision holds at 1.000 with a null of 0.000. c6 declined to pick a replacement
normalisation, which was right.

**The mechanism, stated exactly.** The top-`k` window is not a random sample of landings — it is the `k` *lowest
residuals*. Where residual correlates with which image was landed, the window concentrates on the easiest images,
and as the landing pool `L` grows with `k` fixed the window becomes a smaller and more extreme slice of it. At 300
starts, 50 was 56% of 89 landings and had to reach into the tail; at 3000 it is 5% of 972 and fills with the two
easiest. **Fixed-`k` distinct-image count therefore measures window size against skew, not attacker capability, and
it is not an attacker statistic.** Confirmed by the skew being stable across budgets (image 0 at 59% in both).

**The two statistics that replace it, both attacker-computable:**
1. **Distinct images after deduplication.** The attacker can cluster their own recovered candidates against each
   other — that needs no ground truth — and count clusters. This is the honest breadth number and it is the one to
   put on a page.
2. **Starts to first landing, per image, reported at its maximum over images.** This is the budget question stated
   properly: breadth here is a **coupon-collector problem with unequal probabilities**, so the cost of full
   coverage is set by the rarest image, not by the mean. It is monotone, it carries its budget by construction
   (which R7 already required), and it converts the skew from a caveat into the quantity being measured.

Both need `per_start_nearest`, which is exactly the field c6 identified as missing — so **the harness gap and the
statistical gap are the same gap**, and their proposed fix (save `per_start_nearest`, scale the grid with the start
count) is the right one. Not touching the script while the job was live was correct.

**What may be said meanwhile.** The witness arm's 8 of 8 and the top-50's 2 are not in tension: the release records
all eight and the attacker's cheapest window surfaces two. **That gap is the result** — recovery breadth is limited
by landing skew, not by what the release carries — and it is more interesting than either number alone. It also
sharpens R12-CORRECTED's second item: the certificate's duplicate-landing waste is now measured, and removing it is
what the hybrid buys.

---

## R15 — The two breadth instruments as built (yoado-c6, job 319712): four refinements, one of which is a bug test available now

[derived; implementation described by the owner, not read by me — marked accordingly]

**1. The dedup count measures distinct ATTRACTORS, not distinct images, and must be labelled that way.** Clustering
the attacker's own candidates against each other is exactly right *because* it needs no ground truth — but a
cluster can be a blend, and the attacker cannot tell. Calling the output "distinct images" silently reasserts the
ground truth the statistic was built to avoid. Name it distinct attractors; it is an **upper bound** on distinct
private images recovered, and its being an upper bound is the honest part.

**2. Greedy single-pass clustering is order-dependent, so the order is part of the definition.** Fix it as
ascending final objective — the attacker's own natural order, and the one already used for ranking — and state it
beside the number. Otherwise the count moves with an implementation detail nobody records.

**3. A consistency test that is available BEFORE the job lands, and that turns the pre-registration into a bug
test.** The witness arm already reports 8 of 8 found at 3000 starts. Every recorded image therefore landed at least
once, so `starts_to_cover_all_recorded` **cannot be null at that budget**. If 319712 returns null, the two
harnesses disagree about the same rows and that is a defect, not a finding. c6's stated willingness to report the
null rather than buy more starts is the right instinct and should be kept for budgets where no image is found — it
just cannot fire on this cell, and knowing that in advance is what stops a bug being written up as a result.

**4. The empirical first-hitting time is ONE geometric draw per image, and the max over images inherits its
noise.** For the rarest image, 11 landings in 3000 starts gives `p̂ ≈ 3.7e-3`; a first-hitting time drawn from that
geometric has a coefficient of variation near 1, so the realised value could be anywhere from a few starts to
several thousand while looking like a precise measurement. Required: report the **per-image landing rate `p̂_i`
with its counting error** as the estimate — 11 events is roughly 30% relative error, far better determined than
the order statistic — and derive the coverage budget from it, with the realised first-hitting time beside it
labelled as **one draw, not an estimate**. This is the same class as the "a count needs its gap" rule: a number
whose sampling distribution is wider than the effect must carry that width.

**Endorsed as written:** the `k` grid scaling with the start budget rather than stopping at 50; `per_start_nearest`
saved; the mechanism paragraph carried in the source so the statistic travels with its own warning; and the
coupon-collector reason for taking the max rather than the mean placed in the code comment where the next reader
will meet it.

**R15 addendum — the 815-start figure is the budget for ONE image, not for coverage.** [derived; arithmetic checked]
`ln(0.05)/ln(1−p̂)` at `p̂ = 11/3000` gives **816**, and that is correct for *the rarest image alone* reaching a 95%
chance of at least one landing. **Coverage means all `N` land**, which is the whole reason R14 took the max over
images, and the multiplicity is exactly the part that gets dropped. Requiring each image to fail with probability
at most `0.05/8` gives `ln(0.05/8)/ln(1−p̂)` = **1382 starts** — 1.7× larger. Publishing 815 as the coverage budget
would undercount by that factor, and consistency supports the larger figure: all eight did land by 3000.

**And the counting error propagates into the budget, so it must be quoted as a band.** At `p̂ = 3.67e-3 ± 30%` the
joint-95% budget runs **1060 to 1975**. Quote it as *"of order 1400 starts, between about 1100 and 2000"*, never as
a point. Same rule as the estimate it comes from: a derived number inherits the width of what it was derived from.

State plainly which question each number answers — one image, or all of them — because they differ by more than
the error band and the labels are interchangeable-looking.

---

## R16 — The breadth rows (job 319712): both instruments work, the identity of the binding image is noise, and dedup turned into a stop signal

[read: rows, relayed by the owner; derivations mine]

**The fixed-window artefact is confirmed and closed.** With the window scaled to the budget, distinct images climb
1 → 2 → 5 → 7 → 8 as the window opens, and the eighth arrives only where precision falls to 0.648. **The last image
is not bought with more starts; it is bought by accepting that a third of the list is wrong.** The earlier "5 at
300, 2 at 3000" was entirely the pinned window, as R14's mechanism predicted.

**R15's point 4 is confirmed harder than I argued it, and the stronger form is the one to keep.** I warned that the
realised first-hitting time is one geometric draw whose spread is as wide as the quantity. The rows show something
sharper: the **rarest** image (11 landings) was found **sixth**, at start 26, while the image that finished last at
228 is 3.7× more common. So the realised order statistic did not rank by rate at all.

> The warning is therefore not "the number 228 is imprecise". It is that **the identity of the binding image is
> itself noise.** Reading a coverage budget off one run and concluding "image 3 is the hard one" gets the wrong
> image, not merely the wrong duration.

That is a qualitative failure mode, not a precision caveat, and it is the version that belongs in the write-up.
The rate-derived percentiles (5th 109, median 297, 95th 840) and the realised 228 at the 33rd percentile are
consistent; no contradiction with the 840 budget, which is a 95% quantity and not a prediction of the draw.

**And dedup did something neither of us designed — it is a ground-truth-free STOP SIGNAL.** The attacker's own
cluster count equals the experimenter's distinct-image count at every window where precision holds
(1,1,1,2,2,5,5,7), and at the window where precision breaks it returns **222 attractors against 8 real images**.
It does not degrade quietly, it explodes. So:

> **Count your own haul as you widen the window. When the count blows up, you have run past the attack.** No
> ground truth, no labels, no knowledge of `N`.

This fell out of the upper-bound property rather than being designed in, which is the best kind of instrument. It
is the breadth analogue of the residual ranking — a self-check the attacker can run on their own output — and it
deserves to be presented that way rather than as a diagnostic.

**One requirement before it is quoted as an instrument.** The cluster count is computed at the landing tolerance
and in a stated order, so the elbow's location is threshold-dependent by construction. Run the **tolerance ladder**
on the dedup count, exactly as was done for the in-band split, and show the elbow does not move. If it moves, the
stop signal is a property of a chosen cut rather than of the attack, and it must be reported with the cut attached.

**Both budgets on the page with their questions attached and never sharing a name** — 816 for the rarest image
alone, 840 for all eight — is the disposition R15's addendum asked for.

### R16b — the affine-chart cell's pre-run prediction (job 331384) CONFIRMED at the algebra

Their observation is right and I checked it rather than accepting it. For an **affine** chart `DΨ` is constant, so
the isolation test `J = C·DΨ` is **one number for the whole chart** rather than one per point — the only case in
the study where that is true. Their predicted value:
```
    ker J ⊇ span{z_i − z_j},  dim = N − 1 = 7      (differences map to h_i − h_j ∈ col H ⊆ ker C)
    no extra kernel generically, since Ψ_lin(R^k) ∩ ker A₀ = {0} whenever k ≤ r  (12 ≤ 64)
    ⇒  rank J = k − (N − 1) = 12 − 7 = 5   exactly
```
**Confirmed.** This is a construction check available *before* either arm runs, and a deviation from 5 means the
cell is not the affine case it claims to be. Building it as a new module rather than editing a testbed whose chart
and encoder are both tanh — and therefore cannot reach the affine case at all — was the right call.

---

## R17 — The affine cell (job 331384): the blend degeneracy is ROUTE-SPECIFIC, and the refusal was load-bearing

[read: rows, relayed; construction check derived and confirmed]

**Construction check passed before either arm:** isolation rank 5 against a predicted 5. The cell is the affine case
it claims to be, so both arms are interpretable.

| arm | outcome |
|---|---|
| certificate (recipe-free) | **0 of 60** starts land; every returned point an exact affine combination |
| replay (needs the recipe) | **19 of 60** starts recover **all eight** images to 2e-15; bimodal, no partial recoveries |
| alias test | 18 starts drove the residual below 1e-20 and **every one recovered the whole batch — zero aliases** |
| failures | median residual 0.055, i.e. optimisation failure, separated from the alias mode by ~30 orders |

**This confirms the three-object scoping rather than complicating it.** R12's equivalence is between the certificate
and the linearised representer; replay is the third object and sits *strictly inside* both. The affine degeneracy
therefore hits the two that share a zero set and misses the one that does not. **The nesting predicted exactly this,
and the nesting was already on the page.**

**The refusal was load-bearing, and that is the lesson worth keeping.** The "cannot identify by EITHER route" form
was refused on the direction of the inclusion — `ρ = 0 ⇒ Ch = 0` and never the converse — one line of algebra
already written down. Had it been published it would have been falsified within the hour by this cell. **It was
caught by reading the inclusion the right way round, not by caution**, which is the more reproducible virtue.

**Refinement to the "the route prefers the blend" wording.** The finding is right — the objective at the found
points (max 2.15e-28) sits *below* the residual at the truths (9.4e-15) — but the mechanism should be stated
precisely or it invites a wrong reading. In exact arithmetic a blend and a truth are *both* exact zeros; neither is
preferred. What happens is that the zero set is `N−1 = 7`-dimensional, the truth is a single point in it, and an
optimiser that minimises a computed residual will systematically settle wherever roundoff is smallest — which is
generically not the one point of measure zero we care about. **The truth is not disfavoured; it simply has no
advantage, and a minimiser on a set where the truth is not the unique minimum lands elsewhere by construction.**
Say that rather than "prefers the blend", which suggests something about blends specifically.

**The coefficient sum is 1.000000000 at every start, min equal to max to nine figures** — the affine hypothesis of
R1's lemma appearing in the measurement at every start rather than on average.

**yoado-c6's framing is adopted and is stronger than the pair statement.** Identifiability is a property of the
release, the chart **and the route** together: the same chart that makes one channel provably blind leaves another
exact. That supersedes the release-and-chart pair wording that yoado-72 and I had converged on.

**Scope, carried:** replay needs the recipe, a solver and starts, and 19 of 60 is a basin fraction, not a
guarantee. The certificate needs none of those and gets nothing here. The trade is now measured at both ends on one
release, which is the first time that has been true.
