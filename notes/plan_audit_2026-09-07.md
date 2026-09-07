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
