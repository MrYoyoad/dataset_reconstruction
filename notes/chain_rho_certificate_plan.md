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
certificate alone must already succeed; one above replay's line (k = 18; k = 16 sits AT the line, where counting predicts nothing — 41's correction) where (2) is the predicted outcome.

**Scope of the chain (41):** every design assumes the truth lies ON the certificate's zero set, which is exact only where the
certificate vanishes at recorded images to machine precision — not on a wide head (residual 0.05–0.20 there) and not
under half-precision training. The chosen cell is inside the exact regime; the chain is a method "wherever the
certificate vanishes", not a general one. Projection recomputed EVERY step (curved Z_C, GELU encoder) with a
correction step back onto the surface; certificate residual logged at the end so drift is visible.

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

### From the explainer lane (yoado-21) — three items not covered by D0–D6

**(i) Arity mismatch, and assembling a joint start.** The certificate is a per-image test: below its line a start
descends to *one* recorded image, chosen by where it began, with basins measured 40× uneven. Replay's residual is
*joint* over the whole recorded set. So for `N′ ≥ 2` there is a step between the two that no design names: collect
landings until every recorded image is covered, deduplicate them into one joint start, then run replay once. Its
cost is set by the rarest basin rather than the mean, and coverage — not landing rate — is the quantity to log.
D5 sidesteps this by peeling one image at a time, which trades the joint constraint away; the assembled-joint-start
variant keeps it. Worth a line in the pre-registration either way: for `N′ = 1` the two coincide, so the planned
cell cannot distinguish them and the choice only bites when the band is extended.

**(ii) A hard constraint mis-specifies its own dimension if `N′` is off by one.** D2 and D6 project onto the tangent
space of `Z_C`, whose dimension is `k − (r − N′)` and therefore depends on the *estimated* `N′`. That estimate is a
threshold on a decaying spectrum, and the twenty-image cell already showed a borderline member (imprint present,
singular value seven orders below it). If `N′` is over-estimated the projection deletes directions that are genuinely
free, and the constrained solve cannot recover them by construction — it will stall with a clean-looking residual and
no indication of why. Under-estimating leaves the search too large and is benign by comparison. Cheap guard: run D2 at
`N′` and `N′ ± 1` and compare achievable residuals, the same "level of the floor" discrimination the subset test
already uses.

**(iii) The hard constraint destroys the independent check; the soft one keeps it.** Under D2/D6 the certificate
residual is zero by construction at every iterate, so the quantity that made the algebraic route self-certifying is no
longer available as a diagnostic *during* the replay phase — the only signal left is replay's own residual. D3 keeps
`‖Ch‖` free to drift and therefore retains a second, independent number to watch, which is worth more than it looks
given how often in this project a single residual has been read as evidence of correctness. Suggests logging `‖Ch‖`
under D3 not only as a falsifier for leaving the manifold, but as the running sanity signal D2 lacks.

*(No numbers of my own; the band and nesting statements above are restatements of §0.)*

---

## §4 answer — the theorem side (yoado-81, write-up lane; notation of `notes/exact_channel_rev10.tex`)

**The one fact that governs all three questions.** `C` is built *from* the release,
`C = Π_{row(B_T)^⊥} A_T`. So `Cφ(ψ(w)) = 0` is a consequence of the release, not information
beside it. Concretely, if `ρ(w,X) = 0` then the simulated release equals the actual one, so the
simulated `A_T h_i` lies in the simulated row space, which *is* `row(B_T)`. Therefore

    {ρ = 0}  ⊆  Z_C      (on the recorded set)

— the replay solution set is **contained in** the certificate manifold. Everything below follows.

### (a) Replay restricted to `Z_C`: the counting, and the line

Above the certificate line the manifold `Z_C` is positive-dimensional: `ker C` has dimension
`N' + (n−r)`, i.e. codimension `r−N'` in feature space, so a k-dimensional chart meets it in
`k − (r−N')` dimensions per image. Restricting replay to `Z_C` therefore leaves

    unknowns per image:  k − (r−N')        (was k)
    seed unknowns:       unchanged (see (b))
    B-block conditions:  unchanged, N((m−1)+r−N)

**The chain's line coincides with replay's, and must.** Since `{ρ=0} ⊆ Z_C`, restricting to `Z_C`
removes no solution and adds no constraint that `ρ = 0` did not already impose. The capacity count
is unchanged and the boundary stays `k < (m−1)+r−N'`. **A chain cannot move the line** — if it
appeared to, the derivation would be double-counting the release.

What the restriction *does* change is the dimension of the space searched, from `k` to
`k − (r−N')` per image, at no cost in identifiability. That is a **basin claim, not an
identifiability claim**, and it should be pre-registered as one: the chain's whole hypothesis is
that a smaller search space is easier to search, and nothing in the theorems guarantees it.

Note the regime: **below** the certificate line `Z_C` is generically the isolated recorded points
themselves, so the certificate alone already finishes and there is nothing to chain. The chain is
only interesting in the band `r−N' ≤ k < (m−1)+r−N'`, which is exactly §0's band.

### (b) The seed handoff — exact under (A4), and NOT an `N'×N'` mixing when (A4) fails

Theorem `thm:quot` gives `row(B_T) = col(A_0H)` under (A1)–(A5), and `H = U R_H` with `R_H`
invertible, so `col(A_0H) = col(X)`. The handoff is therefore **exact, not generic** — it is a
subspace identity read off the release, not a statement that holds off a measure-zero set.

Its effect on the unknowns: writing `Ξ` for a known orthonormal basis of `row(B_T)`, `X = Ξ M`
with `M` an `N×N` unknown, so the seed block falls from `rN` to `N²`. And `M` is then
over-determined by the other block: `A_T U = X Ω` gives `Ξᵀ A_T U = M Ω`, i.e. `N²` equations.

**But the premise in the question is wrong when (A4) fails.** If `rank P_T = N' < N`, then
`row(B_T)` is an `N'`-dimensional *subspace of* `col(X)` and does not determine it. The handoff
then fixes only `N'` directions and leaves the other `N−N'` columns' worth entirely free:

    seed unknowns after handoff:  N'² (mixing)  +  r(N−N')  (the unrecorded directions)

So it is **not** an `N'×N'` mixing — the `r(N−N')` term is the one that matters, and it is
precisely the invisible examples' directions. Since the interesting regime for the chain is a
strong model, where `N' < N` is the common case, this term will usually be present.

**Precedent worth weighing before building on (b).** Reducing the seed unknowns has been tried:
the Q-parametrisation (`S = XᵀX` on `Sym⁺_N`, `rN → N(N+1)/2`) was measured **not to widen the
basin**. The handoff is a different reduction, but it is the same *kind* of reduction, and the one
data point we have says this kind does not buy basin. Pre-register accordingly.

### (c) Does anything relate `Z_C` to replay's basin? — No, and there is a measured warning

The theorems give **containment** (`{ρ=0} ⊆ Z_C`) and nothing else. Containment is about where
solutions *are*; a basin is about where descent *converges from*. No result here connects them.

And the measured behaviour warns specifically against the chain's implicit premise. Above the
certificate line, a point can sit at an **exact** certificate zero and be far from every private
image: at `k=58` (one above its line) 14.3% of starts reach an exact zero, 0.04% land on a
recorded image, and the argmin is a spurious solution **0.84 away** from any of them (job 753886).
So **`Z_C` membership is not proximity to the truth.** A chain that lands on `Z_C` and hands off
to replay may hand off a point as bad as a random one, and the above-line regime is exactly where
the chain is supposed to operate.

That is not an argument against trying it — it is the argument for what the pre-registration must
measure: not "does the chain reach `Z_C`" (it will), but **the distribution of image error at the
handoff point**, against random-start controls at the same `k`. If handoff points are no closer to
the truth than random ones, the chain is a smaller search of an equally bad space.

### Summary for §0's band

| | replay alone | chained |
|---|---|---|
| line | `k < (m−1)+r−N'` | **same** (forced: `C` is a function of the release) |
| unknowns/image | `k` | `k − (r−N')` |
| seed unknowns | `rN` | `N'² + r(N−N')` (exact under (A4): `N²`) |
| what is hoped for | — | basin only |
| what is warranted | — | nothing yet; `Z_C` membership ≠ proximity (753886) |
