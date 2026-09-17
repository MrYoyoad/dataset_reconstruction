# Plan audit — the 7 Sept certificate+replay brief (approver lane, yoado-8b)

Audited before any compute. Seven findings: two would have made a day-one experiment ill-posed, one upgrades a
"may resolve" to a derivation with a registered prediction, one is a wording hole with teeth given what nearly
shipped last night, and three are protocol.

## A1 — E1B is ILL-POSED WHEN q < N, and the brief's own instruction creates that case

§12 day-one says: *"Unknown = H in R^{d×N}, N taken as q = r − rank C (report q and the true N)."* Then it asks for
per-image `‖ĥᵢ − hᵢ‖` and `‖ĥᵢ‖/‖hᵢ‖`.

**If q < N those metrics do not exist.** With q columns you are solving for a spanning set, not for the training
representations; two images with dependent representations are not separable objects, and there is no
correspondence between the q recovered vectors and the N true ones to difference. Worse, the brief instructs
`N := q`, so the very substitution it prescribes is what breaks the metric.

**Fix, and it costs nothing:** run E1B-tiny on a cell where **q = N is verified first** (assert `rank H = N` before
the solve; the fp64 synthetic N=8 release should satisfy it). Then the per-image metrics are well defined and the
decision line means what it says. Report `q` and `N` side by side as rule 12 already requires, and if a later cell
has `q < N`, the recovery metric there is a **subspace distance** (principal angles between span(Ĥ) and span(H)),
never a per-image norm. State which metric is in force on every row.

## A2 — The seed-free scale symmetry IS broken, and by an identifiable mechanism. Register the prediction

§7 and rule 8 leave scale as "replay *may* resolve it", and flag the compensating rescaling `h → αh`,
`A₀ → A₀/α`, which leaves `A₀h` fixed. The brief asks whether the base path breaks it. **It does, and the argument
is one line:**

The adapted layer computes `W₀h + B A h`. The rescaling leaves the adapter path `A₀h` invariant but sends the
frozen base path to `αW₀h`. In the head-adapted setting that shifts the logits, hence the softmax residuals, hence
`P_T`, hence the released `B_T = P_T Xᵀ`. So the released factors are **not** invariant under the rescaling and the
scale is identifiable whenever `W₀ ≠ 0`.

**Registered prediction for E1B, before the rows:** the seed-free arm recovers `‖ĥᵢ‖/‖hᵢ‖ ≈ 1`, and the residual
surface has curvature along the joint `(α, 1/α)` direction. **Falsifier:** if the seed-free arm returns a flat
family of solutions differing by a common scale, then the base path does not break it in this configuration and
the derivation is wrong or its hypothesis (`W₀` acting on the same `h`) fails. A cheap pre-check that needs no
solve: evaluate the replay residual at `(αh*, A₀/α)` for a few `α` and confirm it is not flat.

## A3 — "Not ruled out by count" must be stated as NEVER meaning identifiable

Rule 10 fixes the language to *ruled out by count / not ruled out by count / marginal*. Good. But §5 nowhere
carries the consequence of **Lemma 15** into that language, and this is exactly the hole that a claim went through
last night before being caught: the count can be satisfied with room to spare and the truth still not be isolated,
because `C` is linear and every element of `span(H)` passes. Measured last night: a pixel-layer cell with a linear
chart, count satisfied by two orders (`k + N = 40` against `d = 3072`), **0 of 8 recovered**, found points blends at
fraction 1.000, and the residual ranking that works everywhere else failing too.

**Required wording:** "not ruled out by count" is followed, in the same sentence, by *"identifiability additionally
requires the chart to meet span(H) only at the training points (Lemma 15); the count does not test that."*

## A4 — ker-C starts should be predicted to HELP on a nonlinear chart and NOT on a linear one

§8 use 2 proposes starts in `ker C ∩ range(ψ)` as a fix for the 0-of-20 initialiser problem, uniformly. But
`ker C ⊇ span(H)`, so on a **linear** chart `ker C ∩ range(ψ)` **contains the whole blend subspace**, and starts
drawn there concentrate on exactly the points Lemma 15 says are indistinguishable. Measured last night on the
affine cell: certificate 0 of 60 with every returned point an exact affine combination, coefficients summing to
1.000000000 at every start.

**Registered prediction for E2:** ker-C starts beat random starts on a **nonlinear** chart and do **not** beat them
on a linear one. If they help on the linear chart too, my reading of the blend geometry is wrong. This makes E2's
`starts random / in ker C` axis interact with the chart axis, so **E2 must not be run at a single chart
nonlinearity** or the arm's effect is confounded with the chart's.

## A5 — E1B's "distance from the public feature manifold" measures the wrong thing on its own

The stated measure is the projection residual onto the E4a PCA charts at `k = 32/64/128`. That conflates two
different failures: *not a real feature* and *not inside a low-dimensional chart*. §9's own zero-set statement is
about `range(Φ⁰)`, not about a chart.

**Add the direct test:** for each free solution `ĥ`, solve `min_x ‖Φ⁰(x) − ĥ‖` and report that residual beside the
chart residual. The first says whether any image produces it; the second says whether the chart can express it.
Report both; the decision line in §12 ("dynamics invertible from H") should not rest on the chart residual alone.

## A6 — The audit protocol needs a register, not just re-derivation

§12 asks auditors to *re-derive every number from the logged job outputs*. Last night that was not sufficient three
times: a route inferred from a docstring rather than the function, a chart dimension inherited from adjacent prose
rather than read from the header, and an audit finding built on a general prior about what a value "usually" is.

**Add to the protocol:** every audit entry carries its register — **read: function** / **read: rows** / **derived
(hypotheses named)** / **read: prose — provisional**. A claim about what an experiment *did* requires the function
or the row; a claim about what the mathematics *implies* requires no artefact but must name its hypotheses. An
entry marked *read: prose* cannot be one of the two PASSes.

## A7 — E10's bound has a free constant and cannot be tested until it is fixed

E10 measures `‖C̃_ℓ Φ⁰_ℓ(xᵢ)‖` against `K_ℓ ε`. `K_ℓ` is undefined, so any measurement can be accommodated by
choosing it. **Fix `K_ℓ` in the proof before the measurement runs** (a Lipschitz constant of a named map, or a
measured operator norm with its own job id), or the experiment cannot fail and is not a test.

## Also checked and CORRECT — no action
- §5.2's block-diagonal argument: per-image `Σ_ℓ s_ℓ` against `k`, ranks adding but capped at `k` per image, and
  multiplicity changing nothing unless the chart shares parameters. Correct as written.
- Rule 6's asymmetry: projection onto `ker C` is legitimate when the representation itself is the variable and
  illegitimate through a nonlinear chart. E1B's ker-C starts are the legitimate case; consistent.
- The mandatory three-arm metadata control, and the rule that anything the metadata-only arm also produces is not
  a reconstruction. This is the single most important control in the document.
- Rule 9 keeping precision out of supervisor-facing material.

---

## Addenda after the auditors' first pass (yoado-e1)

### A8 — E5 is NOT new. It ran last night as job 331384 and matches its spec clause for clause

Retag **done** and point it at `results/exact_inversion/affine_two_routes_331384.jsonl`. Recomputed from the rows
and the run's own tensors [read: rows]: linear chart in feature space, `N=8`, `k=12` below both capacity lines and
above `N−1`; certificate landings **0 of 60**; coefficient sum of every returned point **1.000000000**, min equal
to max rather than a median; isolation rank 5, i.e. deficit exactly `N−1`; replay on the **same release and the
same starts** recovers all eight from **19 of 60**, with **zero aliases** — all 18 starts reaching residual below
1e-20 recovered the whole batch — and the 42 failures at median 5.5e-02, optimisation failure rather than
ambiguity. **Lemma 15 has its measurement.** The only thing E5 as written would add is a second chart
nonlinearity, which is A4's point and belongs to E2.

### A9 — E2's failure criterion needs a convergence column, and this is the sharper form of A4

Mine said E2 must not ship at a single chart nonlinearity. yoado-e1's addition is better and I am adopting it:
**E2 must report which arm the starts CONVERGED in, not only whether they recovered.** On the linear chart the
certificate arm converges *beautifully* — objective 2.15e-28, **below the residual at the truths** — while
recovering nothing.

> An arm that reaches a **lower objective** and a **worse image** is the exact signature of this confound, and a
> table of objectives alone scores it as a win.

So every E2 row carries objective *and* image error, and the verdict is a function of both — which is cross-cutting
rule 4 from last night's ledger, arriving in a new place. **Fail any E2 file that reports objectives without image
errors beside them.**

### A10 — W1's spine is NOT gated on the new experiments, and should start now

Several W1 links are already proved and measured and depend on nothing in Tracks I–III: the certificate stated with
`q`, the Gaussian quotient-sensing form, the rank and capacity line, the affine-hull lemma (now with A8's
measurement), and the one-way nesting `{ρ=0} ⊆ Z_C` with its falsifying cell. That spine is in
`notes/exact_channel_rev10.tex`, compiling at 46 pp with A7 and A8 stated and the necessary-not-sufficient
correction in place.

**Ruling: start W1 on that spine immediately.** It is item 1 of §1.3 and the answer to Q4 depends on it existing.
The gate stays on everything downstream — no reconstruction claim from count alone, and nothing from Tracks I–III
enters until its experiment file has two PASSes.

---

## A11 — E4a audit (jobs 674521 DINO, 674524 CLIP). The shared-concept regime is REFUTED, and the headline ratio needs a conditional

[read: rows, relayed by the executor; arithmetic and both findings derived here]

### The result that was measured but not drawn out: §6.2's shared-concept regime is worse, not better

§6.2 proposes `x_i = G(z_shared, z_i)` as the regime that **reduces** the unknown count, illustrated with
`k_shared = 32, k_nuisance = 8, N = 8` giving 96 unknowns instead of 256 — and states explicitly that *"whether
reality has this structure is what E4a measures."* **E4a measured it.** Concept axis **15–16** directions at 95%
variance over 20 concepts; within-concept nuisance **205–208**.

```
    shared-concept   k_shared + N·k_nuisance = 15 + 8·205 = 1655
    per-image target            N·k          = 8·128      = 1024
```

**The shared-concept parametrisation is worse by about 60%.** The brief's illustration has the ratio inverted: it
assumed a large concept axis and a small nuisance axis, and the measurement says the concept axis is tiny and the
nuisance axis is enormous. **Pose, lighting, crop and background are where the dimension lives, not identity.**

This **removes a regime from the plan** rather than adding one, which is worth more than the residual table because
it changes what would be built. Precondition before it lands: both counts must be dimensions at the **same**
variance threshold or the comparison is void.

### Required conditional on "16 to 44 times too coarse"

The arithmetic is right — `0.196/0.0124 = 15.8`, `0.196/0.0045 = 43.6` — but it divides a **feature-space**
residual on DINO/CLIP embeddings by a gate measured in **pixel space** on the CIFAR MLP and CNN releases.
Different space, different data, different release, different search. **The gate is a property of a release and
its search, not a universal constant**, and its transfer to a feature-space attack on a foundation backbone is
unmeasured. State the ratio conditionally with the transferability named open in the same sentence.

**Let the like-for-like claim carry the weight instead**, since it needs no conditional: a 32-dimensional public
chart represents private data **better in pixels (0.25) than in a frozen ViT embedding (0.60)**. Same quantity,
same `k`. Moving to a foundation-model embedding does not dissolve the chart problem — it worsens it. That is the
sentence that answers the realism objection.

### Endorsed as reported
- **Lemma 15's signature reproduces in feature space** at every cell of both backbones and both regimes — blends
  about twice as close to the public chart as the privates themselves (DINO target k=32: 0.4091 against 0.2014).
  The affine-hull degeneracy is a property of **what public charts represent well**, not a pixel artefact. Make it
  a numbered claim.
- **The nonlinear-chart caveat is correctly handled.** Non-monotone in `k` proves an optimisation failure rather
  than a property. "Not competitive as trained here (300 epochs, 2000 samples)" is the right form and it must not
  be cited about nonlinear charts in either direction.
- **The linear-autoencoder assertion passed** to three decimals at every `k` in every regime — the theory check
  the arm exists for. Report as passed, not assumed.

### A11 CORRECTED — both of my numbers were wrong, and the second is an error in the brief itself

The executor rejected two figures I supplied. Both rejections are right; the second traces to a mislabelled
quantity in §6.1 of the brief, which I propagated without checking.

**1. My 60% was computed at unmatched fidelity.** I set the shared-concept count at 95% variance
(`15 + 8·205 = 1655`) against a per-image chart at `k = 128` (`1024`), which is *below* that threshold — DINO
target residual at `k=128` is 0.288, not the 95% point. The fidelity-matched comparison is
```
    shared-concept   15 + 8·205 = 1655
    per-image             8·205 = 1640
```
**Weaker as a percentage and stronger as an argument, which is the executor's point and it is correct:** the
shared part amortises *nothing*; it simply **adds** `k_shared` to a cost already dominated by per-image nuisance.
And it generalises without any measurement — `k_shared + N·k_nuisance > N·k_nuisance` for any positive concept
axis — so **the shared-concept parametrisation can only pay when the nuisance axis is small, which is exactly the
condition the measurement refutes.** My 60% would have been the first thing an opponent recomputed.

**2. The 0.25 pixel figure is not a projection residual at all — it is a CERTIFICATE residual.** The executor
found it had no job row and lives only in a docstring, and refused to quote it back to me as circular. The
provenance is worse than that. Its origin is `experiments/cifar/RESULT.md:212`:

> *"their chart projections have **certificate residual** 0.25, while the blend floor sits at 0.013"*

§6.1 of the brief restates that as *"private images sit off a 32-dimensional public PCA chart by a **projection
residual** of 0.25 while blends of privates sit within 0.01."* Same two numbers, **relabelled as a different
quantity.** A certificate residual at a chart projection and a chart's projection error are not comparable, so the
headline I asked for would have divided one by the other.

**The executor's substitution is correct and better founded than it knew:** the oracle ladder's own public rows —
**0.2432** on the keyboard release and **0.3176** on the motorcycle release — *are* measured projection errors of
the true photographs onto a public PCA chart, with job ids. So C1 reads 0.2432–0.3176 in pixels against 0.5991
(DINO) and 0.4078 (CLIP), like-for-like, every figure sourced to a row.

**Carry the correction back into the brief**, since §6.1 is the user's document and the mislabelling will otherwise
be quoted again: the 0.25/0.01 pair there is a certificate residual and a blend floor, not projection residuals.

---

## A13 — My A12 derivation is REFUTED in both directions, and the check exposed a design defect (job 697344)

[read: rows. The check was written in response to A12 and ran 2026-09-07; nobody acted on it.]

I predicted the unreduced seed-free arm has a 32-dimensional solution family and the reduced arm has none. **Both
halves are wrong**, and the measurement is a Jacobian nullity with a twelve-order singular-value gap, so it is not
a threshold artefact.

| parametrisation | equations fitted | Jacobian | nullity |
|---|---|---|---|
| unreduced | product `A@H` only (**what the harness does**) | 672 × 2048 | **1704** |
| unreduced | full `A_T` (available) | 2016 × 2048 | **232** |
| reduced | product `A@H` only (**what the harness does**) | 672 × 1025 | **681** |
| reduced | full `A_T` (available) | 2016 × 1025 | **232** |

**Where I was wrong.** The unreduced fibre is 232, not 32 — my count assumed all released numbers are independent
constraints and 200 further degenerate directions exist that counting does not see. And **the reduction does not
make the truth isolated**: reduced and unreduced both sit at 232 once the same equations are fitted. My claim that
the reduction is "the difference between a solution family and a point" is **withdrawn**. It shrinks the search; it
does not change the identifiability.

### The defect the check exposed, which is worth more than the correction

**The harness fits the product `A@H` — 192 equations — when the release contains `A_T` in full, 1536.** Using what
is already available drops the nullity from **1704 to 232** unreduced and **681 to 232** reduced, a factor of seven,
at no cost. The running E1B arms were solving a far weaker problem than the release supports.

### And the consequence that changes what E1B can conclude

**Nullity 232 > 0 in every cell means the truth is not locally isolated in either arm.** Recovery of `H` from this
release is **not unique**, by measurement — an information property, not a solver property. So E1B's decision line
cannot read "dynamics invertible from H: yes" in this configuration however well the solver performs, and a low
residual there is expected rather than evidential. Under the verdict rule this is the **alias** side, not the
optimisation-failure side, and the two must not be merged.

**Required before E1B is quoted at all:** refit against the full `A_T`, re-measure the nullity in the configuration
actually used, and if it remains positive, report E1B as an identifiability negative rather than as a solver
result. The 232 needs its own explanation — it is 200 beyond the counting deficit and is presumably a structural
symmetry — but the decision does not wait on that.

## A14 — The H3 question is UNTESTED, not settled (job 696469)

The sweep varies the step size looking for a cell where the invertibility hypothesis fails. The only cell that
breaks it, `lr = 5`, has a conditioning of 7.9e17 **and a diverged trajectory** — `‖ΠA_T − ΠA_0‖` at 4.95e+67 —
and `rank B_T` collapses from 5 to 2, so **H4 fails in the same cell**. Every other cell has conditioning between
1.0 and 3.0 with both hypotheses holding.

**So no cell separates H3 from H4**, and the conjecture that H3 can be dropped is neither confirmed nor refuted by
this. It is also the wrong instrument: a hypothesis that is never *used* cannot be falsified by making it false —
falsifying it requires a cell where H3 fails, H4 holds, and the conclusion still breaks, and the sweep produced
none. The proof sketch in the W1 audit stands on its own; the sweep is consistent with it and is not evidence for
it. **Report as untested and settle it in the proof, not the sweep.**

---

## A15 — A13's headline attributes the non-uniqueness to the WRONG OBJECT. Corrected by yoado-1d (job 350944)

[read: rows, relayed with the decomposition printed per cell]

A13 said: nullity > 0, therefore *"recovery of H from this release is not unique."* **True as stated and wrong in
what it blames.** The sweep decomposes the family:

```
    UNREDUCED  full A_T   family dim 72   of which MOVES H: 72   moves H with the SEED HELD FIXED: 0
    CHART k=12, seed KNOWN, full A_T                            nullity 0
```

**Zero directions move `H` with the seed held fixed.** So the whole family is a **seed-versus-H trade-off**, not
slack in `H`. Hold the seed and `H` is pinned; add a chart at `k=12` with the seed known and the truth is locally
isolated outright.

> **Corrected claim: `H` is not identifiable JOINTLY WITH THE SEED from this release.** That is far narrower than
> "recovery of `H` is not unique", and it points at the **seed-side** work rather than at the chart.

**The cell that decides whether E1B is a negative at all** has not printed: *chart k=12, seed FREE, full `A_T`.*
Under the product it is 1344. If the full-`A_T` version returns at or near zero, **a chart plus the full release
pins the data even with the seed free, and E1B is not a negative in the attack's actual configuration.** If it
stays large, the negative stands and is specifically a **seed**-identifiability negative. Nothing about E1B's
verdict may be written until that number exists.

Also confirmed a second time at a different shape: reduced equals unreduced (72 = 72) once the same equations are
fitted — A13's withdrawal of my reduction claim holds at `m = 40` as well as `m = 20`.

## A16 — The two decompositions reconcile; my "N factors out" was wrong (yoado-b3)

Both splits of the seed give the same formula — one splits by the `r` index, the other by the `d` index, the pinned
block cancels one-for-one either way, and both leave `dN − N·(bracket)`. **No cancelling errors on either side.**

`nullity = N·[d − (m − 1 + r − N)]`, zero exactly when `m + r ≥ d + N + 1` — the capacity line with the chart
dimension replaced by the representation dimension.

**My claim that `N` factors out and the batch cannot help was wrong.** `N` appears twice, as multiplier and inside
the bracket, so `∂(nullity)/∂N = d − m + 1 − r + 2N` = 37 here: shrinking the batch **does** lower the nullity, to
100 at `N=4` and 22 at `N=1`. It never reaches zero, because the bracket at `m=20` is `21 + N > 0` for every
`N ≥ 1`. **So the conclusion stands — only the head closes it — and my reason for it did not.**

Sweep 350944 is submitted with predictions fixed in the script header before any row: head widths 40/48/49/64
predicting 72/8/0/0, and batches 4/1 predicting 100/22. The **8 → 0 step between adjacent head widths** is the
sharp falsifier.

## A17 — Operational: a job that prints only at the end is how 697344 sat unread for ten days

`experiments/e1b/e1b_lm.py` emits one line after every start and every iteration completes, so three arms will be
silent for hours and then speak once. That is the exact shape of the ten-day failure: nothing to watch while it
runs and nothing prompting anyone when it lands. **A per-start flush makes a job self-announcing instead of
dependent on someone remembering it exists.** Worth doing once, in the harness, not per job.

---

## A18 — A numerical check that implements the CORRECTED theorem tests the code, not the theorem (T2 Prop. A)

[relayed from yoado-1d's audit; the hazard is derived]

The audit FAILs T2 Prop. A's stated rank law: `rank C = r − N′` is wrong whenever `n_l < r`, and contradicts the
proposition's **own kernel clause**, `max(0, min(r−N′, n_l−N′))`. **The harness already implements the correct
law.** So the reported 116-of-116 agreement is agreement between the harness and the corrected law — **it could not
have failed**, and it says nothing about the theorem as written.

> **The general hazard, and it is new to the list.** The track's rule says *a numerical check agreeing never
> promotes a status to PROVED*. This is sharper: **a check whose harness implements the corrected form of a
> theorem is not weak evidence, it is no evidence** — the disagreement it exists to detect has been engineered
> away. A theorem check must implement the statement **as written**, independently of the production code, or it
> is a regression test on the implementation wearing a theorem's name.

**Required:** re-label the 116/116 as verifying the implementation; correct the stated law to the kernel clause;
and re-run the check against the statement as written before any claim rests on it. This is the self-confirmation
risk the audit spec named — the checks were written by the author of the theorems — arriving in its most
disguised form.

## A19 — Route B (depth) has a closed form, and it is proved in the regime we are NOT in

`theory/T5.3`: if `r − N ≥ k_1` one layer already saturates and depth adds nothing; if `r − N < k_1`, depth closes
the gap and `L ≥ k_1/(r−N)` layers suffice generically. At `r=24, N=8` that makes `k_1 = 128` need **L ≥ 8 adapted
layers** — set against a **113-output head** for the same chart width by the other route. **Real released adapters
are deep, not wide-headed**, so if it survives, the depth route reaches an adequate chart in a configuration that
exists and the head route does not. That is a better realism answer than the head number.

**Three reasons it is the next AUDIT and not the next claim.** T5.2/T5.3 are unaudited — the auditor covered T2
Prop. A, Cor. A.1 and T5.1 only and says so. T5.2 holds at **zero drift** and needs **independent per-layer
initialisations**, violated by a shared seed or tied adapters. And T5.4, the drift case, is a **CONJECTURE** whose
own caveat is the sharp one: under drift the honest object is **effective rank at a stated tolerance, not rank**.

**Our setting has drift.** So the formula that decides Route B is proved only where we are not, and the quantity it
is stated in is not the quantity that survives there. **Effective rank at a tolerance requires a tolerance ladder**
— the same treatment the dedup stop signal needed, and for the same reason: an elbow that moves with the cut is a
property of the cut.

**T5.1 does NOT block Route B**, and the near-miss is worth recording: `rank J_F ≤ k_1 ≤ min(k, n_1)` reads as a
ceiling on depth, but `k_1` is bounded by `k` itself — the quantity being pinned — so **the ceiling sits AT the
requirement, not below it.** Anyone meeting that line out of context will read it as a blocker.

## A20 — Rank death is attacker-visible; contamination is not. The lifetime result is DEFENDER-facing

`rank C_full` is computable from the release alone, so an attacker sees the channel die. **Contamination is
`rank B_T = N′`, and the attacker lacks `N′`** — so they cannot verify it. **This is the computable-versus-
verifiable rule again**, third instance: naming the quantity on each side of an equality and asking whether the
attacker holds both. The consequence is a scope, not a caveat: the certificate-lifetime result is a statement a
**defender** can act on and an attacker cannot.

---

## A21 — Two corrections to me, and they are the same failure three times tonight

**1. The retrodiction is ONE measurement, thirteen days old — not two, not months apart.** Verified at source:
`STATUS.md:323` (2026-09-04, jobs 218345/218346) and `notes/assumption_relaxation_program.md` §19a carry the
identical setup — four adapted layers, `r = 256`, everything below frozen — and the identical table, 747 / 717 /
445 / 138. **Same jobs written up twice.** My phrasing, *"a different experiment, months apart in intent"*,
overstated it in exactly the way an expert reader tests first.

**What survives is still the strongest confirmation of the corrected law**, and it should be said in this form: the
measurement **predates the theory work**, was **not built to test T5.2**, was written up as a **bare empirical
ceiling with no theory behind it**, and the corrected law predicts it while T5.2 predicts no such ceiling.
**One unforced retrodiction is worth more than twelve purpose-built seeds. It is not worth two.**

**2. I quoted one number where the table has four.** The chart shortfall is **14.9× and 45.7× at the
identifiability cap**, and **9× and 26× at k = 384** — a spread of **9× to 46%** depending on release and on
whether the cap is respected. I gave a single "≈9×". Give the table, never a midpoint and never one corner.

### The pattern, and it is mine

Three corrections tonight, all the same shape: **a true finding amplified one notch past its evidence.** "Removes
precisely the directions" (it removed those and 184 others). "The identifiability negative is dead" (dead in the
oracle configuration). "A different experiment months apart" (the same jobs, thirteen days). None was a wrong
number; each was a true result stated tighter or wider than the rows support.

That is the same family as the phrase this lane corrected in its own file earlier — *a claim stated tighter than
the data supports does not read as more confident, it reads as unchecked.* **Applied to the approver seat: the
amplification usually happens in the sentence that frames a result, not in the result.** Check the framing sentence
against the rows with the same care as the number it frames.

### And the squeeze is no longer a hypothesis

The one-setting sweep I specified had already been run and sits at `STATUS.md` 14–27: same axis, same space, same
eight photographs, **attacker-buildable public PCA chart**, zero widths satisfying both walls on either release,
and the fidelity wall alone carrying it at k = 384. **It is measured, not a cross-construction hypothesis.** The
carried caveat — the cap applied to CIFAR by formula with a trained `φ` in the path — is correctly not
load-bearing.

**The right "open or closed" statement is neither:** public PCA **plateaus**, so it cannot be made to work by
choosing `k`; what remains open is the chart **family**, not its width. That turns agreed step 2 from a preference
into a requirement, with a measurement saying why.

---

## A22 — The chart-shortfall discrepancy: NEITHER figure is canonical, and both compress silently

[derived; arithmetic reproduced from the ladder rows]

Ledger **G3** says the public chart is *"20 to 35× coarser than the gate"*. `STATUS.md`'s two-wall table says
**14.9× and 45.7×** at the cap and **9× and 26×** at `k = 384`. Both are right about different cells, and they
differ in **two** places, only one of which either lane noticed.

**Difference one — the chart width.** G3 reads the ladder's own public row at `k = 32`; the table reads the chart
at the identifiability cap (`k = 66`) and far past it (`k = 384`). Chart error falls with width, so these are not
the same quantity.

**Difference two, and nobody named it — the gate is a BRACKET, not a point.** The ladder measures landings at one
width and none at the next, so the gate is an interval: `(0.0124, 0.0186]` on motorcycle, `(0.0045, 0.0090]` on
keyboard. **Any ratio quoting it as a scalar has already compressed an interval**, and the two figures compress it
differently: **G3 divides by the bracket MIDPOINT** (reproduced exactly: 0.3176/0.0155 = 20.5, 0.2432/0.0067 =
36.0), while the table divides by the **near end only**.

**The honest picture, carrying both the width and the bracket:**

| chart width | motorcycle | keyboard | overall |
|---|---|---|---|
| `k = 32` (ladder's public row) | 17.1–25.6× | 27.0–54.0× | **17–54×** |
| `k = 66` (identifiability cap) | 9.9–14.9× | 22.9–45.7× | **10–46×** |
| `k = 384` (far past the cap) | 5.9–8.8× | 13.2–26.4× | **6–26×** |

**Ruling: neither is canonical and no scalar can be.** The quantity is *chart error at a named `k`, divided by the
gate as a bracket*, and it is a range at every width. G3 gets a scope line naming `k = 32` and disclosing the
midpoint step; the table gets the bracket added. **The single source of truth is the table with `k` stated, not a
number.**

**And the conclusion is robust to every one of those choices**, which is what should actually be said: at the most
generous reading available — widest chart, far end of the bracket — the shortfall is still **5.9×**. It never
approaches 1 anywhere in the grid. *That* is the claim, and it survives whichever convention a reader prefers.

**The failure mode is the freshness rule's own:** one quantity, two sources, no single place saying which governs —
and underneath it a silent compression of an interval into a point, which is the same family as reading a median
where the rule says worst-case.

---

## A23 — Tonight's capacity formula carries an unstated hypothesis: q = N

[raised by the archive lane from `notes/gal_2026-09/` Appendix A.5 and Remark 3.9; checked against my own derivation]

The archive's chart note **explicitly declines** to convert the capacity line into a theorem by substituting `q`,
and Remark 3.9 states that `q_l` is **not interchangeable** with the recorded-span object. **I set `q = N`
explicitly in the derivation of tonight's formula** and never wrote the hypothesis down.

> `nullity = N·[k − (m − 1 + r − N)]` **assumes the private representations are linearly independent**, `q = N`.
> Where `q < N` it must be restated in `q`, and which slots take `q` and which take `N` is exactly what Remark 3.9
> warns is not interchangeable.

**It does not invalidate anything measured**, because every cell asserted `rank H = N` before solving — that was
A1's precondition. But the statement travels without the hypothesis, and it is the most-quoted result of the
night. **Attach it wherever the formula appears**, and say that `q < N` is unrestated rather than assumed benign.
The same applies to every repo statement written `k < r − N`.

## A24 — Four archive conflicts, ruled

**C1 — the "searched, not found" cancellation: reword, and it is a UNIFICATION rather than a correction.** A claim
about a search is refuted by exhibiting one, so `theory/T4-C5`'s wording must go. The suggested replacement is
right — no cancellation for generic instances, a structural one under the range condition. **But the interesting
part is that the range condition keeping first-order drift inside the span is the same mechanism as T3.2's
"in-span drift is free".** Two files describing one phenomenon in different vocabularies. Merge them rather than
patching one.

**C2 — FAIL on `experiments/multilayer_cert/RESULTS.md` §3, and it has two faults.** The attribution is wrong: 36
of 64 such rows sit at hidden layers 1–2, only 28 at the last layer. And the last-layer rows are **additionally**
capped at `m − 1` by a proved softmax bound the file does not mention — which is **the simplex cap this project
already has** (`N′ ≤ m − 1`), so the file attributes to one cause what demonstrably has two, while the second is
already in our own ledger unreferenced. Correct at source. **One reader so far** — the re-aggregation came from a
sub-agent — so the multilayer lane confirms before the edit.

**C3 — the R2 row is stale, and this is now THREE independent routes agreeing.** A counterexample with independent
Gaussian seeds that still fails; the measured `[9,18,20,20]` identical under shared and independent seeds; and the
archive's Thm 10.2 requiring only a **density** on the allowable covectors rather than independence. Independence
was never the hypothesis. Remove the row; the refutation already in T5 is the correct reading.

**C4 — check the three hand-computed coefficients.** Marked PROVED, never checked, and the repo's existing check
validates a different instance. A twenty-line FP64 evaluation of each is cheap and closes it. **A hand-computed
constant carrying a PROVED status with no independent evaluation is the same shape as a numerical check whose
harness implements the corrected law** — a status resting on the author's own arithmetic.

**And the independence finding is correctly scoped.** `theory/T1..T6` cite neither the archive note nor its lemmas
and agree on every shared statement. That is a **consistency signal, not a joint proof**, and recording it as such
is right — two derivations agreeing is evidence about the derivations, not a second proof of the result.

---

## A25 — No bundle number has a reproducing script here. Scope it: unreproduced NUMBERS ≠ unsupported STATEMENTS

[read: rows — repo-wide content hash of 43,832 `.py` files including the vendored tree, zero matches against the
37 bundle scripts]

The archive lane's rescan upgrades an earlier four-directory check to an exhaustive one and the verdict holds:
**none of the provisional numbers in `framework_rev10/11/12` or its audit has a reproducing script in this
repository.** CLAIM.md's ground rule 3 anticipated exactly this — *every number from the external bundle is
†provisional until reproduced by a committed script with a recorded seed* — and it is now confirmed rather than
assumed.

**But the finding must carry a distinction or it will over-reach, and it is the same one the archive lane got
right about independent derivations.** "No reproducing script" is a statement about **numbers**, not about
**claims**:

| | status |
|---|---|
| the bundle's **numbers** — basin fractions, the exact-inversion residuals, the phase-diagram percentages | **unreproduced here**, †provisional, no script |
| several of the bundle's **statements** — the certificate annihilating the training span, `rank C = r − q`, the quotient-sensing form, the capacity line | **independently derived and measured here**, by code that does not descend from the bundle |

Those are different things and conflating them would either overstate the problem (calling proved statements
unsupported) or understate it (treating a measured statement as licensing the bundle's numbers).

**The operational consequence is better than a caveat.** For anything we have measured ourselves, **cite our own
measurement rather than the bundle** — it is independent, it has a job id, and it removes the provenance question
entirely. Reserve the dagger for the numbers that remain the bundle's alone. A claim sourced to our own rows is
strictly stronger than the same claim sourced to an unreproduced document, so this costs nothing and closes the
exposure.

## A26 — The window question, answered with evidence and qualified honestly

**DERIVED, and checkable rather than asserted:** the first submission crashed inside that cell on its first step
value and its rows file records only the other two, **so no number from that cell ever existed to react to.** The
window was set from the fourth-power criterion while repairing the crash.

**And the qualification is the valuable half:** the criterion was computed at the *second* draft — the original
window would have put two of its four points below the representable range. So *"derived before the cell ever
ran"* is true and *"derived before the script was written"* is not, and both sentences are in the file. **A lane
volunteering the reading that weakens its own answer is what makes the answer usable.**

**Margins reported as a table rather than three verdicts:** 3353× and 35014× against 2.0× on the coefficient,
240× and 583× against 2.0× on the order. The flattening was doing real work — four to five orders, not a rounding
difference.
