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
