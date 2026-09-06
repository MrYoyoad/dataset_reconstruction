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
