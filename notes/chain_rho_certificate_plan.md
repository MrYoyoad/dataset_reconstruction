# Plan — chaining the certificate (`Ch = 0`) with replay (`ρ = 0`)  (draft v0, 2026-09-04, yoado-cd; peers to extend)

**User directive (Yoad):** "let's make a plan on how to chain ρ and C together to work; ask peers too."

## 0. Why chain at all — the band
`{truth} ⊆ {ρ = 0} ⊆ {Ch = 0}`. The certificate is the A-block of ρ with the seed block and the trajectory
projected out; it drops the `m−1` error-size conditions per image that live in the B-block.
- Below the certificate line `k < r − N′`: certificate alone returns the exact on-chart image from random starts.
  Nothing to chain.
- **Band** `r − N′ ≤ k < (m−1) + r − N′`: certificate zero set `Z_C` is a manifold of dimension `k − (r − N′)`
  per image; replay's extra `m−1` conditions pin the point on it. Replay lands from near-truth starts only;
  the certificate lands from random starts only. The chain is "certificate to reach, replay to pin."
- At low rank the band IS the value: r=8, N′=1, m=10 → band k ∈ [7, 16) = class-prototype → recognisable.
  At r=64 the band is 56–64 and the certificate is already instance-level.

**Prior evidence:** one anchor point on `Z_C` (above its line, synthetic bed) used as a replay start reached
`Ch=0` to 5e-8 and landed a full image away (jobs 408560-63). One point, not a search along the manifold.

## 1. Candidate designs (all on-chart, fp32 SGD release, one adapted layer, B₀ = 0)

**D0 — basin-along-the-manifold diagnostic (run FIRST, cheapest, decisive for everything below).**
From the truth, move along `Z_C` (project random tangent steps back onto `Ch = 0`) to increasing distance; run
replay from each. Measures replay's basin radius *within* `Z_C` vs the 0.86 ambient radius (job 459111).
If the in-manifold basin is a large fraction of the manifold, random certificate landings will fall inside
it and design D2 will work; if it is tiny, no handoff from certificate landings can work and the start must
come from a prior — decided before spending the landings.

**D1 — many-anchor handoff.** ~500 random certificate landings → replay (unconstrained) from each. The
existing experiment done properly (many points, not one). Baseline for D2: shows whether the *constraint*
does any work.

**D2 — manifold-constrained replay (the executor's design).** LM steps projected onto the tangent space of
`Z_C` at the current point (null space of the certificate Jacobian `∂(Ch)/∂w`, recomputed per step or once
per cell if the chart is linear). Unknowns per image drop from `k` to `k − (r − N′)`; only the B-block of ρ
(and the `X`-block) remain to pin them.

**D3 — penalty continuation (homotopy).** Minimise `ρ² + λ · ‖Ch‖²/‖A_T h‖²` with λ annealed from huge to 0.
At large λ the landscape is the certificate's (benign, random starts land); as λ → 0 the solution slides along
`Z_C` toward ρ's zero. Soft version of D2 — tolerates a numerically thin manifold and lets the path leave `Z_C`
slightly. Free hyper-parameter: the λ schedule; falsifier: the path leaves `Z_C` by more than the certificate's
own floor.

**D4 — seed handoff (already partly done).** The certificate gives `row(B_T) = col(X)`, so replay's seed
unknowns shrink from `r·N′` to an `N′×N′` mixing. Job 605718: shrinks the search, does not widen the basin.
Keep as a component of D2/D3, not a design on its own.

**D5 — peel with the certificate, pin with replay.** In the band each recorded image's `Z_C` is a manifold, but
the single-image subset solve has the largest replay budget `k < (m−1) + r − 1`. Chain: certificate isolates the
dominant image's manifold (spectrum truncation, σ₂/σ₁ small) → replay on the subset {that image} at the rescaled
step `η·N′_kept/N` → subtract its imprint from `B_T` → repeat. Needs the imprints separable (cosine ~0.04, not
aligned). Extends the range of D2 to several recorded images at low rank.

**D6 — B-block-only refinement.** Since the certificate already satisfies the A-block's direction part, run the
refinement on the manifold with the B-block residual alone (`‖P_T R_Hᵀ Xᵀ − B_T‖`). Smaller Jacobian; tests
directly whether the error-size information pins the remaining `k − (r − N′)` unknowns. A variant of D2.

## 2. Pre-registration (yoado-b9's three-branch form, adopted)
Cell: strong 98% model, one low-margin image in a confident batch (most-leaking cell), r = 8, N′ = 1, on-chart
fp32 SGD release, k ∈ {8, 10, 12, 14} (band 7 ≤ k < 16), ~500 random certificate landings as starts.
Log per start: certificate residual at landing, replay residual, image error vs on-chart truth, iterations, stop.
- **(1) Floor at the truth** — residual ≤ 1e-28 AND image error ≤ 1e-10 in ≥ **X%** of landings [X to be fixed by
  41/b9 before the run; proposal: ≥ 10% at k = 8, any nonzero fraction at k = 14] ⇒ the chain closes the low-rank
  attack; the certificate is a start generator for replay.
- **(2) Floor at a wrong image** ⇒ `Z_C` contains release-consistent aliases replay cannot separate: the missing
  ingredient is a CHART/prior constraint, not a start.
- **(3) Residual above the floor** ⇒ basin failure: the missing ingredient is the INITIALISER.
- **(4) No landings** ⇒ test did not run; not scored.
`fwd_check` at machine precision before any row counts. Controls: unconstrained replay from the same landings
(D1); unconstrained replay from random starts (known 0/20); one cell below the certificate line (k = 6) where the
certificate alone must already succeed; one above replay's line (k = 16) where (2) is the predicted outcome.

## 3. Order of work
D0 → (D1 + D2 in one job) → D3 only if D2 stalls (branch 3) → D5 only if D2 works and N′ ≥ 2 is wanted.

## 4. Open questions for the lanes (answers appended below by each session)
- Is `Z_C` well-conditioned enough to walk on? (certificate Jacobian σ_min along the manifold vs across it)
- Does the truth's replay basin have measurable extent *along* `Z_C`? (D0)
- For a PCA chart the encoder is nonlinear (GELU MLP), so `Z_C` is a curved manifold in `w`; is a fixed
  null-space projection per cell adequate, or per step?
- Off-chart: `Ch ≠ 0` at raw images. Does a soft manifold (certificate residual ≤ τ) still help replay?
- Is there a formulation where the certificate is not a constraint but a *coordinate system* (D2/D6), and does
  that expose a counting statement for the chain (e.g. the B-block alone pins `k − (r − N′) < m − 1`)?

---

## §4 answers — executor lane (yoado-41, 2026-09-04)

**Status: the run is held.** I had launched the D2 cell before this plan existed (job 115318); it is killed. Nothing
runs until c9's pre-audit and b9's instrumentation note are in and the thresholds in §2 are fixed.

**On the per-step vs per-cell null-space projection (the question asked).** Per-step, and per-cell is not merely
less accurate — it is wrong here. `Z_C = {w : C φ(ψ(w)) = 0}` is the preimage of a linear condition under a
*nonlinear* map: ψ is a linear PCA decoder, but φ is a GELU MLP, so the constraint's Jacobian
`J_g(w) = C · Dφ(ψ(w)) · Ψ` varies with w through `Dφ` alone. Its variation is exactly the curvature of `Z_C`, and
a projection fixed at the landing point is a projection onto the tangent space *there*, which is only valid to
first order in the step. Two consequences: (i) with a fixed projector the iterate drifts off `Z_C` at second order
and the certificate residual grows monotonically along the run — the very quantity that certifies we are still on
the manifold; (ii) the drift is largest exactly where the chart is rich (more coordinates, more curvature), i.e. in
the band the test is about. The implemented form recomputes `J_g` each step (cost: one `jacfwd` of an `r × k` map,
negligible beside the unrolled replay Jacobian) **and** adds a Gauss–Newton pull-back `w ← w − J_g⁺ g(w)` after each
accepted step, so the iterate is re-projected onto the manifold rather than merely stepped tangentially. The row
logs `cert_norm_at_end`; if a fixed projector is ever wanted as a cheap variant, that field is the falsifier for it.

**On D0, and why I agree it comes first.** It is the only one of the six that can be *decisive in the negative* for
a few minutes of compute: if replay's basin has no measurable extent along `Z_C`, then D1, D2, D3 and D5 are all
dead by the same argument and the answer is "the start must come from a prior", with no landings spent. One design
note: the walk must move along `Z_C` and not merely near it, so each step is a random tangent direction followed by
the same Gauss–Newton pull-back, with the certificate residual logged at every station — a station whose residual
has drifted is not a point of `Z_C` and its replay outcome says nothing about the manifold.

**Cost and timeline.** D0 ≈ 20 minutes on one GPU (a few dozen stations × a 300-iteration LM at r = 8, N′ = 1).
D1 + D2 in one job ≈ 2–3 hours for four k values (the 500-start certificate sweep dominates; the replays are small).
Both are already implemented in `constrained_replay.py` except D0's walk, which is ~30 lines. So: design frozen
today, D0 tonight, D1+D2 immediately after if D0 permits.

**An objection to §2's controls, and a request.** The k = 16 control ("above replay's line, outcome (2) predicted")
is the one I would drop or relabel. At r = 8, N′ = 1 the replay line is `(m−1) + r − N′ = 16`, so k = 16 is *at* the
line, not above it, and the counting argument says nothing about equality — the earlier sweeps found the boundary
sharp to one unit but always read the last full-rank k, never the line itself. Use k = 18 for a control that is
unambiguously above, or state k = 16 as "at the line, outcome not predicted".

**One thing the plan does not yet say, which I think is its real risk.** All six designs assume the certificate's
zero set is where the truth lives. That is exact in FP64 and on-chart; it is *approximate* whenever the certificate
residual at the truth is not ~0 — which is already known to happen on a wide head (m = 26: residual 0.05 … 0.20 at
recorded truths, RESULTS) and under half-precision training. In those regimes `Z_C` as computed does not contain
the truth at all, and every design here silently optimises within a manifold that misses it. The cell chosen
(strong model, m = 10, FP64, on-chart) is safely inside the exact regime, so this is not a problem for the test —
but the chain's *scope* should be stated as "wherever `C h_i ≈ 0` holds to machine precision", not as a general
method, and D0's station log gives the diagnostic for it (`cert_norm` at the truth itself).
