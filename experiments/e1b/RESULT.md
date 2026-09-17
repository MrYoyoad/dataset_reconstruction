# E1B — free feature replay: forget the images, make H the variable, ask whether the dynamics invert

## Setup

The release §12 names: the affine two-routes cell (the job-331384 generator), `d = P = 64`, `N = 8`, `r = 24`,
`m = 20`, `k = 12`, `T = 400` SGD steps, `lr = 0.05`, seed 1, **fp64**, rebuilt deterministically. The unknown is
`H` in `R^{d×N}` directly — no chart — and the question is whether replaying the training dynamics determines it.

Three arms: **seed known** (`A_0` given), **seed free** (`A_0` unknown, initialised at the released `A_T`, which is
attacker-available), and **reduced seed free** (`A_0 = C/c + P_b M`, exploiting that the release pins the seed's
off-span component). 60 Gaussian + 60 ker-`C` starts per arm, matched across arms by construction.

## Job ids + configs + precision

| job | arm | solver | status |
|---|---|---|---|
| **670990** | seed known | Adam, 400 iters, lr 5e-2 | done |
| **670993** | seed free (unreduced baseline) | Adam | done |
| **675031** | reduced seed free, 513 unknowns | Adam | done |
| **688520** | rank-threshold verification | — | done |
| **693342** | arm-separation check | — | done |
| **350928 / 350940** | **fibre dimension (identifiability)** | — | done |
| 350929 / 350930 / 350931 | the three arms again under **Levenberg–Marquardt** | LM | running |

All fp64. Rows in `results/e1b/rows.jsonl`; tensors in `results/e1b/*.pth`.

## Raw numbers

**The two no-solve checks, both of which ran before anything was searched.**

*A1* — `rank H = 8 = N` and `q = r − rank C = 24 − 16 = 8 = N`, asserted before the solve, so the per-image metric
legitimately exists. Verified against an absolute rank floor (job 688520): the cuts sit in spectral gaps of
6.2e14 (`B_T`) and 2.1e14 (`C`), so no rank here is threshold-sensitive.

*A2, the scale pre-check* — the replay residual along `(αh*, A_0/α)`:

| α | 0.5 | 0.8 | 0.9 | 0.95 | **1.0** | 1.05 | 1.1 | 1.25 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|
| residual | 2.8e-1 | 9.2e-2 | 4.4e-2 | 2.1e-2 | **7.5e-16** | 2.0e-2 | 3.9e-2 | 9.2e-2 | 2.8e-1 |

Not flat. A sharp machine-precision minimum at the true scale, rising **linearly** on both sides — two-sided slope
**0.414**, and halving the offset divides the residual by 2.05. Linear rather than quadratic means the zero is
non-degenerate in that direction: **the scale is first-order identifiable**, pinned to order 1e-15 against the fp64
floor, not merely identifiable in principle.

*The E3 gate* — `Pi A_T = c_T Pi A_0` holds to **3.409e-15** with **`c_T` = 1.000000000000**, which is derivable
rather than fitted (`Pi` annihilates `col(A_0 H)`, so the update term vanishes under plain GD with no decay).

**The three Adam arms: zero landings from 120 starts each.**

| arm | landings | objective min | (truth) | best max per-image error | median cosine to assigned truth | `‖ĥ‖/‖h‖` |
|---|---|---|---|---|---|---|
| seed known | **0 / 120** | 3.580e-02 | 8.750e-16 | 1.0155 | 0.484 G / 0.418 K | 0.967 / 0.981 |
| seed free | **0 / 120** | 3.533e-02 | 8.750e-16 | 1.0174 | 0.323 G / 0.314 K | 0.669 / 0.669 |
| reduced | **0 / 120** | 3.565e-02 | 8.750e-16 | 0.9863 | 0.385 G / 0.340 K | seed err 1.28 |

Every arm stalls **fourteen orders above** the residual at the truth, so nothing reached a zero at all. Per the
verdict rule this is `optimisation failure (residual not zero)` and **not** `alias` — the two must not be merged.

**The identifiability measurement (jobs 350928 / 350940).** The local dimension of the family of `(H, seed)`
reproducing the release, as the nullity of the residual Jacobian at the truth. This owes nothing to any solver.

| parametrisation | seed | objective | unknowns | rank | **nullity** |
|---|---|---|---|---|---|
| free `H` | known | full `A_T` + `B_T` | 512 | 512 | **0** |
| free `H` | free | full `A_T` + `B_T` | 2048 | 1816 | **232** |
| reduced seed | free | full `A_T` + `B_T` | 1025 | 793 | **232** |
| chart `k=12` **(ORACLE)** | known | full `A_T` + `B_T` | 96 | 96 | **0** |
| chart `k=12` **(ORACLE)** | free | full `A_T` + `B_T` | 1632 | 1632 | **0** |
| free `H` | free | the v1 objective | 2048 | 344 | **1704** |
| chart `k=12` | free | the v1 objective | 1632 | 288 | **1344** |

Of the 232-dimensional family, **all 232 directions move `H`**, and **0** move `H` with the seed held fixed.

## Claims

**C1 — The answer to E1B's decision line is PARTIAL, and the partition is exact.** *Dynamics invertible from a
free `H`:* **yes with a known seed** (nullity 0 — `H` is locally determined), **no with a free seed** (a
232-dimensional family reproduces the release exactly, so no solver can pick the truth out of it). The entire
ambiguity is a **seed-against-`H` trade**: every direction of the family moves `H`, and none moves `H` alone.

**C2 — A chart that CONTAINS the private representations makes the seed-free problem well posed, and that is why
job 331384 recovers.** Constrain `H` to the `k=12` chart and the seed-free nullity falls from **232 to 0**.
**SCOPE, and it is load-bearing: the chart measured is the release's OWN generating chart** (`H = LW + b` with the
`L`, `b` that produced the data), so the truth lies in it exactly by construction. **It is an ORACLE chart and is
not attacker-available.** What is established is the mechanism — a chart containing the truth restores
identifiability by removing the trade directions — and NOT that any attacker-buildable chart does so. Whether a
public chart does depends on whether the private data lies in it, and the oracle ladder measures that as badly
violated: public PCA charts sit at a projection error of 0.2432–0.3176 against a landing gate of 0.0124 or lower.
The two results compose: this one says what a chart must do, the ladder says public charts do not do it. 331384's 19-of-60 recovery is therefore
not evidence that replay is strong on free features; it is evidence that the chart removes exactly the `H`
directions that trade against the seed. **This is the sharpest statement in the file**: the chart's role is not
merely to shrink a search space, it is to restore identifiability.

**C3 — The reduced parametrisation is a conditioning gain, NOT an identifiability gain.** Nullity 232 both
reduced and unreduced. The reduction removed 1023 unknowns and exactly 1023 of the Jacobian's rank, leaving the
fibre untouched, because every point of the fibre already satisfies `Pi A_0 = Pi A_T` — **the family lies inside
the reduced slice rather than transverse to it**. An intersection argument (`32 + 1025 − 2048 < 0`, therefore
isolated) assumes a genericity that the construction itself destroys, and equation counting also got the fibre's
size wrong: predicted 32, measured 232, because 200 of the 2016 equations are dependent.

**C4 — The v1 objective was wrong twice, and both are worth recording.** It matched `A_s @ Hc` against `A_T @ H`
rather than `A_s` against the released `A_T`. That is (a) only `r*N = 192` equations instead of `r*d = 1536`,
leaving a nullity of **1704**, and (b) **not attacker-available** — its target is built from the true `H`, the very
unknown being solved for. The corrected residual uses only released quantities and is the one the LM arms use.

**C5 — The registered ker-`C` prediction held.** On this linear chart, ker-`C` starts did **not** beat random
ones — median objective 5.34e-02 against 4.48e-02 in the seed-known arm, i.e. slightly worse, in every arm.

**C6 — The capacity line `k < m + r − N` IS the identifiability boundary, confirmed to the unit (job 350967).**
Two lanes independently derived that the family has dimension `N · max(0, k_i − (m + r − N − 1))`, where `k_i` is
the number of unknowns **per image**. The mechanism is that `B_T` is not `m·r` free numbers: it lies on the
rank-`N`, zero-column-sum variety of dimension `N(m−1+r−N) = 280` here, so the attainable Jacobian rank is
`r·d + 280 = 1816` rather than 2016 — which is exactly the rank measured, and the 200 "missing" equations are
exactly that difference. Swept over chart width with the truth held inside the chart at every `k`:

| k | 8 | 12 | 20 | 30 | 34 | **35** | **36** | 37 | 40 | 48 | 56 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nullity | 0 | 0 | 0 | 0 | 0 | **0** | **8** | 16 | 40 | 104 | 168 | 232 |
| predicted | 0 | 0 | 0 | 0 | 0 | **0** | **8** | 16 | 40 | 104 | 168 | 232 |

**12 of 12**, including a **one-unit discontinuity** from 0 to exactly 8 between `k = 35` and `k = 36`, which no
smooth artefact can produce — and a self-check at `k = d = 64` returning exactly the 232 measured independently
for free features **in a different experiment that the formula was never fitted to** — which is the only kind of
confirmation that cannot be a coincidence of construction. So the capacity line is not a counting heuristic: it is
the width at which the private data stops being identifiable, and above it the ambiguity grows by exactly `N`
dimensions per unit of `k`.

**C7 — On the real CIFAR releases the two walls DO NOT OVERLAP: no public-chart width satisfies both (job
350993).** The chart is squeezed from both sides — identifiability caps `k` from above, fidelity pushes it from
below — and this is the first measurement of both **on the same axis, in the same space, for the same eight
photographs**. Chart: the top-`k` PCA of public train images of the added class, which is what the ladder actually
uses and is attacker-buildable. Gate: the ladder's own measured landing threshold on these releases.

| release | identifiability cap | best fidelity available **at** the cap | landing gate | shortfall | widths satisfying both |
|---|---|---|---|---|---|
| motorcycle / MLP | `k ≤ 66` | 0.1845 at `k = 66` | 0.0124 | **14.9×** | **0** |
| keyboard / CNN | `k ≤ 66` | 0.2058 at `k = 66` | 0.0045 | **45.7×** | **0** |

**And the fidelity wall alone is sufficient for the conclusion**, which is what makes it robust: even ignoring the
identifiability cap entirely and pushing to `k = 384`, the projection error is still 0.1094 and 0.1191 — about 9×
and 26× the respective gates. **A public PCA chart of this family cannot be made to work on these releases by
choosing `k`** — and the scope of that sentence is exact: *this chart family*, not charts in general.

**C8 — The obstruction in the seed-known arm is the SUPPLY OF STARTS, and the requirement on a start is
ALIGNMENT with a norm caveat (jobs 351007, 352266, 353524, 353562).** Nullity 0 said the truth is unique. Two
explanations remained — ill-conditioning or a small basin — separated without random starts, since random-start
landings conflate the solver with the geometry.

Both seed-known setups are **well conditioned** (free `H` 1.201e+02, chart `k=12` 1.789e+01), so conditioning is
excluded. The requirement on a start is then measured directly, with cosine and norm set **independently by
construction** — `ρ‖H‖(c·Ĥ + √(1−c²)·u⊥)` has cosine exactly `c` and norm exactly `ρ‖H‖` for any pair:

| cosine \ norm ratio ρ | 0.9 | 1.0 | 1.35 | 1.7 |
|---|---|---|---|---|
| 0.00 | fail | fail | fail | fail |
| 0.30 | fail | fail | fail | fail |
| **0.50** | **converges** | **converges** | **converges** | fail |
| 0.60 | converges | converges | converges | fail |
| 0.70 | converges | converges | converges | converges |

All convergences are to machine precision (max per-image error 3e-15 to 8e-15). **Alignment is the dominant
requirement**: at cosine ≤ 0.30 no norm converges, so norm alone cannot rescue an unaligned start. But the
boundary is **not vertical** — it tilts at large norm, where the cosine needed rises from 0.5 to 0.7.

**This overturns the ray-based threshold this file previously reported, and the reason is instructive.** The
from-near-truth family lies on `H + η‖H‖u`, where `cos = 1/√(1+η²)` and `ρ = √(1+η²) = 1/cos` are locked — so
**that family can only ever sample the line `ρ = 1/cos`** and is structurally incapable of separating the two
quantities. Its apparent threshold (converges at cos 0.597 / ρ 1.674, fails at cos 0.581 / ρ 1.721) sits exactly
where the plane's **norm** boundary lies, near ρ ≈ 1.7. **The ray was measuring the norm wall and attributing it
to cosine**, and it overstated the alignment requirement: on the plane, cosine **0.5** suffices at any sane norm.

**What an initialiser must supply, stated correctly:** a cosine of about **0.5** with the private representations,
while **not inflating the norm** beyond roughly 1.35×. The arms' random starts have a perfectly acceptable norm
ratio (0.897) and a fatal cosine (≈ 0) — they fail on direction alone.

**No single scalar explains the boundary — verified, so the obvious objection is foreclosed.** The three natural
candidates all have overlapping ranges across the twenty cells (recomputed here from the construction, independently
of yoado-a8 who raised it):

| candidate scalar | converging cells | failing cells | separates? |
|---|---|---|---|
| relative distance to the truth | 0.742 – 1.229 | 1.127 – 1.972 | **no** |
| component along the truth | 0.450 – 1.190 | 0.000 – 1.020 | **no** |
| component orthogonal to it | 0.643 – 1.214 | 0.859 – 1.700 | **no** |

There are outright inversions: a start at distance 1.127 fails while one at 1.229 converges. **So the boundary is
genuinely two-dimensional and the plane is the reportable object** — not a caution about a small grid. Cosine is
**necessary and not sufficient**: nothing below 0.30 converges at any norm, and 0.50 converges only up to a norm
ratio of about 1.35.

**The chart improves conditioning by about 7×** (18 against 120), so the chart's second job is real but modest,
and it is not what stands between a random start and the truth.

## What is NOT claimed

**The plane is measured at four norm ratios and five cosines, on the seed-known arm, at `T = 400`.** The tilt is
located between ρ = 1.35 and ρ = 1.7 and the alignment boundary between cosine 0.30 and 0.50; neither is resolved
more finely than that, and no functional form is fitted to five points. **Condition numbers depend on `T`** (the
same setup gives 3.539e+02 at `T=20` against 1.201e+02 at `T=400`), so they are never compared across `T`.

**No isotropic basin radius is claimed, and the distance bracket that looked available is NOT valid.** It is
tempting to combine "converges at 1.343" with "all 120 random starts at 1.30–1.42 failed" and conclude the edge
lies between — but those are different families of points (cosine 0.60 against cosine 0), so the inference is
void. The cosine threshold (converges at 0.598, fails at 0.555) is measured only along rays from the truth; it is
a statement about that family, and whether it transfers to arbitrary starts at the same cosine is untested.

**The Adam arms do not show that recovery is impossible.** In the seed-known arm the nullity is 0, so recovery is
possible in principle and Adam's total failure there is a **solver** result, not an information result. The LM
arms (350929–350931) test exactly that, and their predictions are registered in `results/00_map.md` §4b-bis
before they land: P3 seed-known should land; P4 seed-free should show the **alias** signature (low residual, large
per-image error); P5 reduced should behave like seed-free and not like seed-known.

**The `c_hat` values from the reduced Adam arm test nothing.** They came out at 2.4–5.7 against a theorem that
says exactly 1. That is **not** the harness bug the pre-registration warned about, because no start converged —
these are stalled iterates, not solutions, and the prediction applies to solutions. P1 remains **untested**.

**A5, the direct manifold test, is VACUOUS on this release and is reported as vacuous rather than as a zero.**
`phi = identity` here, so `range(Phi_0) = R^d` and `min_x ‖Phi_0(x) − ĥ‖` is identically zero for every candidate
by construction. The real-backbone cell is a separate addition, not this one.

**C9 — The law SURVIVES a nonlinear `phi`, measured with a matched affine control (job 355535).** The cap's
weakest joint was that the law is derived where the chart-to-adapted-layer map is affine, while real releases put
a trained network in that path. Built small enough to measure exactly rather than fought at CIFAR scale, since
what is under test is the nonlinearity and not the size: `n = 32`, `m = 11`, `r = 12`, `N = 8`, so `cap = 14`, on
a GELU MLP trained on CIFAR-10, with the identical shapes re-run using `phi = ` a fixed linear map as control.

| k | 8 | 12 | 14 | **15** | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| predicted | 0 | 0 | 0 | **8** | 16 | 48 | 80 |
| trained `phi` (nonlinear) | 0 | 0 | 0 | **8** | 16 | 48 | 80 |
| identity `phi` (affine control) | 0 | 0 | 0 | **8** | 16 | 48 | 80 |

**7/7 in both arms**, including the one-unit step at the cap. The control matching matters as much as the trained
arm: it shows the harness reproduces the law, so a hit in the trained arm is not a harness that would have said
"hit" regardless. The rank saturates at 496 = `r·n + N(m−1+r−N)` in both, exactly the variety bound.

**What this does and does not license.** The *mechanism* — a nonlinear `phi` does not reduce the Jacobian's rank,
so the cap survives it — is now measured rather than assumed, which was the open joint. It is measured at **one
scale with a weakly trained backbone** (42% train accuracy: "genuinely nonlinear", not "well trained") and not at
the CIFAR releases' own shapes (`n = 256`–`1000`, `r = 64`, cap 66). So C7's cap is no longer a bare extrapolation,
but the specific CIFAR numbers remain formula-applied and are still labelled so.

**C7's identifiability cap is a FORMULA APPLIED, not a measurement on these releases.** The `N·(k − (m+r−N−1))`
law is derived and confirmed on the *synthetic affine* release, where the map from chart coordinates to the adapted
layer's input is affine. The CIFAR releases put a trained network `phi` in that path, and whether the same rank
argument survives a nonlinear `phi` is **not established here**. The cap is therefore reported as an extrapolation
and labelled as one. **C9 now measures the first of the two arguments below directly, at one scale.** They were recorded as reasoning
(yoado-a8's) and **not** as measurement when written: the dimension count is unchanged by a nonlinear `phi` — unknowns are still `N·k` and the
equations are the same — so what a nonlinearity could change is whether the Jacobian attains full rank, and a
nonlinearity generically does not *reduce* rank; and separately, the affine degeneracy this project has been
worrying about is a statement about the **certificate's** zero set (every blend is an exact zero), whereas this
sweep measures the **replay** residual's Jacobian, whose zero set is strictly smaller and does not contain the
blends. Neither argument is asserted here: one cell on a real backbone settles it. The fidelity column is measured directly on the CIFAR privates and needs no such caveat —
which is why C7 is stated so that the fidelity wall alone carries it.

**Scope of any landing here** (carried in every row as a `scope` field): `Z_feature` and `Z_image` coincide on this
release, so a landing certifies that the dynamics invert from a free `H` and certifies **nothing** about free
feature replay landing on representations no image produces — this cell cannot exhibit that failure.

## Kill-criterion check

The decision line §12 asks for is answered (C1) rather than killed. No kill criterion was met: the reproduction
gates passed (residual at the truth 8.75e-16), A1's precondition held, and the A2 pre-check came back not flat, so
the derivation it guards is intact. Had A2 been flat, the instruction was to stop before spending the solve; it
was not.

## Audit log

- **A2 was run before any solve and reported regardless of outcome**, as instructed. It passed, and the *shape* of
  the pass (linear, not quadratic) turned out to carry more than the pass itself.
- **The two arms reported an identical median objective at their first checkpoint**, which would have been the
  signature of the seed-free arm being a relabelled copy. Checked directly (job 693342) rather than by re-reading
  the code: the seed block's relative gradient is 3.40e-01 against the `H` block's 1.33e-02 and the seed moves by a
  relative 1.15–1.25, so the arms are separated. The identical medians were a **format string** — `%.2e` shows
  three significant figures, so 4.4512e-02 and 4.4548e-02 both render as `4.45e-02`.
- **The approver's counting argument was checked rather than taken**, and both of its conclusions failed: the fibre
  is 232-dimensional, not 32, and the reduction does not isolate the truth. Measured, not re-derived.
- **Rank thresholds were verified against an absolute floor** after a sibling lane found that a relative-only
  threshold calls a numerically zero matrix full rank. Clean here, and the hazard is now recorded as a prohibition
  for defence evaluation, where a merged release gives `C = 0` identically.
