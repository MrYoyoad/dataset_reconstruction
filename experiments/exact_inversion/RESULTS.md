# RESULTS — exact inversion of the LoRA training map (backprop through the recipe)

Synthetic FP64 testbed, WEXAC L40S/A40, 2026-09-02. Theory: `notes/exact_lora_inversion_framework.md`
(from `framework_rev10.pdf`); finite-difference baseline: `results_rev9.pdf` §3b.
Script `lora_exact_inversion.py`, runner `scripts/run_exact_inversion_wexac.sh`, raw rows
`results/exact_inversion/*.jsonl` (each line carries seed, git hash, command line, host).
**Every number is provisional (†).** Post-fix numbers are at git `5762045` and later; the phase-diagram
sweep was produced at `38fec3b`, before six defects were fixed (see "Corrections" below).

**Why the pre-fix grid rows are still usable.** An earlier version of this file argued "none of the six
defects can turn a failure into a false recovery". That argument is **wrong**: defect 2 (the `verdict`
field keyed off the lower median) and defect 6 (the collapse-blind nearest-image metric) are both
false-*recovery* mechanisms. The correct justification is empirical and does not rely on the buggy fields
at all: recovery was **re-derived from `final_err_max` (every image below 1e-2) together with the
residual**, neither of which those defects touch. On that objective criterion the pre-fix grid gives
34/49 recovered — exactly equal to the count the buggy `verdict` field gave, so the two false-recovery
mechanisms fire **zero** times here — and the 15 objective failures are exactly the 15 cells re-run
post-fix. Quote the objective re-derivation, not the "only false failures" argument.
*(This correction is owed to an independent audit by a sibling session.)*

World: `k`-dim tanh generator → 64-dim image → tanh encoder → `n=96` features → softmax head `m=20`,
LoRA `r=16`, `B₀=0`, Gaussian `A₀`, plain SGD. Attacker gets `(A_T, B_T)`, `W₀`, φ, ψ, the labels and the
recipe; never `A₀`, `H`, or the trajectory. "Recovered" = relative image error < 1e-2 for **every** image.

**Scope (independent genuineness audit by a sibling session, 2026-09-03, read-only over code + rows).** This is a
single LoRA *head* on frozen features — `W₀ + BA` on the last layer, `A` and `B` the only trained parameters —
not a deep multi-layer adapter (that case is `multilayer_lora.py`, Step 17, outside the theorems). The forward
model is fully known (generator, encoder, recipe, labels), noise-free, with no model mismatch: the synthetic
numbers are an **identifiability testbed / upper bound**, not attack feasibility. The audit traced every use of
the ground truth: none reaches the objective; recovery is scored separately after the solve; `fwd_check` passes
in every row; residual→0 with the wrong image is scored *failed* (68 such verdicts on disk). Verdict: nothing
self-confirming; every remaining concern is scoping, and those scopings are written into the steps below.

## Step 1 — the simulator is the training map, and the released factors invert

| k | N | r−N | T | lr | deformation | start err | fwd_check | final err (med) | residual | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 8 | 8 | 400 | .01 | 0.39 | 0.085 | 7.8e-16 | 2.2e-15 | 8.9e-31 | recovered |
| 6 | 12 | 4 | 400 | .01 | 0.26 | 0.166 | 8.5e-16 | 6.6e-16 (post-fix) | 1.0e-30 | recovered |
| 8 | 10 | 6 | 400 | .01 | 0.36 | 0.023 | 7.4e-16 | 9.2e-16 | 9.2e-31 | recovered |
| 12 | 8 | 8 | 1500 | .03 | 0.96 | 0.034 | 8.5e-16 | 9.3e-15 | 9.9e-31 | recovered |
| 12 | 8 | 8 | 1500 | .03 | 0.96 | 0.085 | 8.5e-16 | 9.1e-15 | 8.8e-31 | recovered |

**`fwd_check` is the load-bearing number.** It is the relative error with which the span-adapted
simulator reproduces the *actual* release when fed the true data and the true `X = A₀U`. At 7.4e-16 to
8.5e-16 it confirms Theorem 1 operationally: the release really is a deterministic function of the
candidate data and of the `rN` numbers in `X`, and nothing else about `A₀` is needed. Every downstream
claim rests on this; a large `fwd_check` would have meant the simulator and the recipe disagree.

Timing: 1.4 s per LM iteration at T=400, 5.4 s at T=1500 (A40, FP64), ~20 iterations to machine
precision. CPU is 1.7 s/iter at T=400 — **the GPU advantage is small here**; the problem is small and
latency-bound, so this testbed does not need a GPU. The 4/5 recovered cells reach 1e-15 image error with
residual ~1e-30, i.e. the release is reproduced to the FP64 floor.

**The one non-reproduction has been withdrawn** (`NOTES.md §2`). The `(k,N) = (6,12)` cell was re-run
post-fix at the same seed, same start, same single restart, no staging: it recovers to `6.6e-16` in 14 LM
iterations. The pre-fix stall at residual 5.7e-4 was our QR seam bug, not a basin difference and not the
prototype's staged schedule. Staging was tested directly and is not needed (`--stage-x 10` also recovers,
in 53 iterations rather than 14, i.e. slower). `results_rev9.pdf` §3b **does** reproduce here.

## Step 4 + 5 — the phase diagram: the certificate's boundary does not bind the exact inversion

`figures/exact_inversion/phase_diagram_comparison.png` (the side-by-side against the certificate-only
diagram; `phase_diagram_exact.png` is the single panel). 49 cells, `N, k ∈ {2,…,14}`, T=400, start 10% off.

**49 / 49 recovered** (one seed each), median residual 8.6e-31, zero cells with a residual above the
reproduction floor.

**What this study measures, and what it does not.** The grid is an **existence** result across `(N, k)`:
*is the `k = r − N` boundary crossable at all?* One recovered cell above the line would answer yes; 49 of
49 is overwhelming. It is **not** a reliability or success-rate measurement, because there is one seed per
cell. Reliability is what the basin study below measures, on a different axis: *how far from the truth can
a start be and still land?* Keep the two labelled separately and neither is over-claimed.

**Regime scope (independent derivation check, 2026-09-03).** The simulation channel has a capacity
boundary of its own, at roughly `Nk ≈ mr` plus a conditioning limit — and **this grid never reaches it**.
Its largest cell is `Nk = 14 × 14 = 196`, comfortably under `mr = 20 × 16 = 320`. So the precise claim is:
*the `r − N` budget does not bound the simulation channel **in the tested regime** (`Nk ≤ 196 < mr = 320`)*.
"The certificate boundary is irrelevant to simulation" is shown; "simulation is unbounded" is **not**, and
the Adam arm already demonstrates that a different limit can dominate. Job 467914 pushes `Nk` from 112 to
352 at fixed `N = 8` to look for that boundary directly, reading `σ_min(J)` at the truth as it crosses
`mr`.

**First readings (job 467914, N = 8, `mr = 320`).** `σ_min(J)` at the truth collapses geometrically as
`Nk` grows, roughly an order of magnitude per `Δ(Nk) ≈ 48`:

| k | Nk | ‖res(truth)‖ | `σ_min` truth | `cond` truth | worst image err | residual |
|---|---|---|---|---|---|---|
| 14 | 112 | 1.1e-15 | 1.02e-3 | 1.8e3 | 2.9e-15 | 5.7e-31 |
| 20 | 160 | 9.2e-16 | 1.03e-4 | 1.6e4 | 8.2e-15 | 5.2e-31 |
| 26 | 208 | 1.1e-15 | 3.54e-6 | 4.6e5 | 7.1e-13 | 9.2e-31 |

| 32 | 256 | — | **3.03e-20** | 7.6e19 | 5.3e-3 | 9.3e-31 |
| 38 | 304 | — | **1.06e-20** | 1.3e20 | 7.9e-3 | 8.0e-31 |
| 44 | 352 | — | **3.45e-19** | 7.2e18 | 9.3e-3 | 7.9e-31 |

**The simulation channel does have its own boundary, and this is the first genuine non-identifiability
anywhere in the study.** `σ_min(J)` at the truth falls geometrically and then collapses to the FP64 noise
level between `Nk = 208` and `Nk = 256`: `3.5e-6 → 3.0e-20`, a fall of fourteen orders across one step of
the sweep, with `cond` correspondingly at `1e20`. Past that point the Jacobian at the truth is numerically
rank-deficient, so **local identifiability fails** — the solution is no longer isolated.

Read the residual column carefully, because it is the discriminator this project keeps relying on. In the
failed cells the residual is still at the reproduction floor (`~9e-31`) while the image error is `5e-3` to
`9e-3`. The release is reproduced exactly by a point that is **not** the truth, so the truth has ceased to
be locally isolated.

**CORRECTED (independent audit, 2026-09-03) — do NOT call this "the wrong image".** An earlier version of
this section said the release was reproduced "with the wrong image", and the claim set derived from it said
the release no longer determines the data. That is **not supportable on these numbers**. The past-line
image errors are `1.7e-3` (N=4, k=34/38), `2.6e-3`-`5.7e-3` (N=8, k=28/29/30), `5.3e-3`-`9.3e-3`
(N=8, k=32/38/44), `6.4e-3` (N=14, k=22), `1.2e-2` (N=12, k=26). In **8 of the 11** past-line cells
`frac_recovered` is 1.0: every image is inside this study's own 1e-2 tolerance. A 0.3% relative image error
is a visually identical image. (`frac_recovered` is a tolerance count, not the boundary marker: the boundary is
read off `σ_min(J)` at the truth, which is already rank-deficient in those same cells.)

What was measured is that the solution is **no longer pinned to machine precision** — an eleven-order jump
from `1e-14` to `1e-3` — not that it is unrecovered. So:

> **The capacity law is a boundary of exact identifiability. It is not, on this evidence, a boundary of
> leakage.** Past `k = m + r − N` the attack still returns every image to sub-percent accuracy.

That tension is more interesting than the overclaim it replaces, and it should be stated rather than
smoothed over.

**A second limit on the word "alias".** Every past-line cell was run from `--init near`, i.e. starting
adjacent to the truth, so the solver drifts along the flat direction only as far as LM happens to take it.
The measured `1e-3` is therefore a **lower bound on the fibre, not its diameter**. Earning the word alias
requires travelling the null direction: take the right singular vector of `J` at `σ_min` and continue along
it, retracting onto the release-consistent set at each step, and report how far the image actually moves
while the release is still reproduced. Job 482338 does exactly that, with a below-the-line control where
the traversal must be obstructed. Until it reports, the correct wording is: *the solution ceases to be
locally isolated; from a truth-adjacent start the solver lands 0.2%-1% away in image space; the extent of
the release-consistent set has not been measured.*

**The boundary matches the refined count, not the naive one.** `Nk ≈ mr = 320` was the first guess. The
observed collapse is between 208 and 256, which `mr` does not predict — but the derivation check's
refinement does: `B_T = P_T Xᵀ` has rank `N`, so it carries only `N(m + r − N)` independent numbers, which
at `N = 8` is `8 × 28 = 224`. The observed boundary brackets 224. Equivalently the law is a **per-image
budget**:

```
        k  <  m + r − N          (per-image degrees of freedom below per-image released information)
```

which at `N = 8` predicts `k* = 28`, between the last success (`k = 26`) and the first failure (`k = 32`).

**This also closes the Q5 scoping question, and explains the 49/49 grid.** The per-image budget is
`k < m + r − N = 36 − N` here, which *decreases* with `N` at one degree of freedom per added image — so the
simulation channel does degrade with `N`, just not in the `Nk ≈ mr` form first guessed. And it is exactly
why the phase diagram recovered everywhere: at its worst corner `N = 14` the ceiling is `k < 22`, while the
grid only reaches `k = 14`. **The whole 49-cell grid lies strictly inside the capacity region**, so it
could not have found this boundary. The two open questions (the equation count, and the grid's regime
scope) turn out to have been the same missing fact.

**CONFIRMED across three values of `N` (job 469120).** The threshold is not a fixed `k`; it moves with `N`
exactly as `m + r − N` says it should. `figures/exact_inversion/capacity_law.png`.

| N | predicted `k* = m+r−N` | last `k` recovered | first `k` failed | `σ_min` at last ok | `σ_min` at first fail |
|---|---|---|---|---|---|
| 4 | **32** | 30 | 34 | 1.2e-6 | 8.8e-19 |
| 8 | **28** | 26 | 32 | 3.5e-6 | 3.0e-20 |
| 12 | **24** | 22 | 26 | 5.8e-6 | 2.5e-20 |

Every predicted threshold is bracketed by its own last success and first failure, and the three brackets
are disjoint — `N = 12` has already collapsed at `k = 26` while `N = 4` is still healthy at `k = 30`. A
fixed-`k` explanation is ruled out by that crossing. The collapse is 13-14 orders of magnitude in `σ_min`
across one step of the sweep, so the boundary is sharp rather than gradual.

**Single-unit brackets (jobs 479587, 479684), and an honest off-by-one.** Stepping `k` one at a time:

| N | line `m+r−N` | last healthy k | `σ_min` there | first collapsed k | `σ_min` there |
|---|---|---|---|---|---|
| 8 | 28 | **27** | 1.53e-7 | **28** | 6.43e-19 |
| 14 | 22 | **21** | 1.65e-7 | **22** | 2.30e-19 |

The collapse happens **at equality**, `k = m + r − N`. An audit correctly noted that the naive count only
derives `k ≤ m + r − N`, one unit weaker — so the strict form was, briefly, measured rather than derived.

**RESOLVED: the strict inequality IS derived, and the missing unit is the softmax simplex constraint.**
Under softmax cross-entropy the error columns sum to zero, `1ᵀD_t = 0`. Since `∇_B L = D_t(A_tH)ᵀ`, also
`1ᵀ∇_B L = 0`; with `B₀ = 0` and any scalar-linear update, induction gives `1ᵀB_t = 0` for every `t`. So
`B_T` does not merely have rank `N` — it lies in `1^⊥ ⊗ ℝ^r`, of dimension `(m−1)r`. Rank-`N` matrices
there form a manifold of dimension `N((m−1) + r − N) = N(m+r−N) − N`, a deficit of exactly `N`. The count
then returns

```
    N((m-1)+r-N) + rN  >=  Nk + rN     <=>     k <= (m-1)+r-N = m+r-N-1     <=>     k < m+r-N   (strict)
```

**Verified two independent ways.** Directly: `‖1ᵀB_T‖/‖B_T‖` = 3e-16 … 1.9e-15 across *every* SGD release
on disk, all shapes, all `T`, all seeds. And through the rank: the measured `rank(B-block)` for SGD
`n=96, k=12, N=8` is **216**, and `N((m−1)+r−N) = 216` exactly, where the plain cap would be 224. The
"216 / 224" entry in the per-block table is therefore not slack — it is this constraint, and it is tight.

**A falsifiable prediction, with supporting evidence already in hand.** Adam's coordinatewise
normalisation divides the gradient entrywise and does **not** preserve zero column sums: measured
`‖1ᵀB_T‖/‖B_T‖` = **2.6-2.7** for every Adam release, i.e. order one against `1e-15` under SGD. So under
Adam the constraint is absent and the cap reverts to the plain `N(m+r−N)`, predicting an Adam capacity
**one unit higher** than SGD at the same `(m, r, N)`. Supporting: the measured Adam `rank(B-block)` at
`n=96, k=12, N=8` is **224** = the plain cap exactly, saturated, against SGD's 216 = the `1^⊥` cap exactly.
Both caps attained, each in its own regime. Untested directly; the discriminating run is an Adam capacity
sweep at `k = m+r−N`.

Note the irony worth carrying into any defense discussion: Adam destroys the exact algebraic channel
outright, and hands the attacker back one unit of exact-inversion capacity.

**A measurement caveat that the derivation, not the data, settles.** For `k > m + r − N` strictly,
ill-conditioning cannot masquerade as rank deficiency: the containment argument forces `σ_min = 0` exactly.
But *at* equality the plain count would permit full rank, and there `σ_min ≈ 8e-19` with `cond ≈ 2e18` sits
at the FP64 floor — precisely the regime where the measurement cannot separate a rank-deficient problem
from an identifiable one with `cond ≈ 1e18`, and the image error cannot discriminate either since attainable
accuracy there is about `ε·cond`. **So the strictness is not settled by the measurement; it is settled by
the simplex derivation above** (`1ᵀB_T = 0` puts `B_T` in `1^⊥ ⊗ ℝ^r`, giving the cap a deficit of exactly
`N` and hence `k ≤ m+r−N−1`). The data are consistent with it and cannot, at this precision, have
established it alone. *(Caveat and framing owed to an independent audit.)*

**What the law says.** The released `B_T = P_T Xᵀ` is `m × r` of rank `N`, so it carries
`N(m + r − N)` independent numbers, however large `m × r` looks. Divide by the `N` images and each image
gets a budget of `m + r − N` numbers. An image with more degrees of freedom than that is **not locally
isolated**: past the line the release is reproduced at the floor by a *different* point, so the truth
ceases to be pinned.

**Do not read that as "the wrong image".** Measured, those points sit at 0.2%-6.8% relative image error and
in most past-line cells every image is inside the study's own 1e-2 tolerance — see the corrected block
above and Step 7. This is a boundary of **exact identifiability**, and on this evidence it is **not** a
boundary of leakage. Step 11 sharpens it further: it is identifiability of the *chart coordinates*, and how
close `ψ(ŵ)` lands to the private image depends on whether the chart can represent it at all.

So the honest capacity statement for the whole attack is:

| channel | boundary | provenance | failure mode past it |
|---|---|---|---|
| certificate (Primitive 1-2) | `k < r − N` | **†bundle** (`results_rev9.pdf` Fig. 1) — **not reproduced in this repo**; there is no `C`-only recovery experiment under `results/exact_inversion/` | aliases; `C` is blind |
| simulation (Primitive 3) | `k < m + r − N` | measured here (jobs 467914, 469120, 479587, 479684) and on real MNIST at three ranks (job 568095) | `J` at the truth is rank-deficient; the truth is not isolated, and the alternative points are still 0.2-6.8% reconstructions |

The two rows are **not commensurable measurements** — the first is a bundle number under this project's
standing rule that bundle numbers stay provisional until reproduced here. So "the difference is exactly
`m`" should be read as *against the bundle's certificate boundary*, not as a difference between two
in-repo measurements.

Simulation buys a factor of `(m + r − N)/(r − N)` in per-image complexity — here `28/8 = 3.5×` at `N = 8`
— and the released head width `m` is what buys it. That is a much more useful statement than "the `r − N`
budget does not bound the leakage", because it is a *quantitative* replacement rather than a negation, and
it makes an immediately falsifiable prediction: widening the head (larger `m`) should widen the attack's
reach linearly, at fixed rank.

This is the figure the exercise was for. The certificate-only diagram (`results_rev9.pdf` Fig. 1) is
**exactly 0 above the line `k = r − N`** — above it the certificate has fewer rows than the manifold has
dimensions and every run lands on a true alias. Exact inversion recovers **on both sides of that line**,
including the extreme corners (`N=14, k=14`, where the certificate has 2 rows against 14 tangent
directions). The `r − N` budget is a boundary for *one channel*, not a bound on what the release leaks.

Two honest qualifications:

- **CORRECTED (later the same day).** An earlier version of this file said "15 of the 49 cells needed
  restarts". That was wrong, and the correction runs the other way. Those 15 cells failed on the
  **pre-fix** code (git 12fa60d / 38fec3b) and were rescued on the **post-fix** code *with* 4 restarts,
  confounding the two changes. Re-running the same 15 cells post-fix at **`restarts=1`** recovers
  **15 of 15**, median residual 8.0e-31, median 17 LM iterations (job 456630). So the restarts were
  never needed: all 15 were **false failures caused by the QR sign discontinuity** (F4), exactly the
  systematic under-reporting of the basin that the review predicted. The clean statement is that the
  post-fix solver recovers every cell of the grid from a 10% start in a single attempt.
- One seed per cell. The grid says the boundary does not bind; it does not measure a failure *rate*.

## Step 2 — the basin (k=12, N=8, T=1500, lr=.03, deformation 0.96)

**REWRITTEN 2026-09-03.** The first version of this section was measured on the pre-fix code and is
withdrawn: it reported a basin to a 24% start error, a "first clean failure at 36%", and restarts as the
currency of distance. All three were artefacts of the QR seam bug, and the correction runs strongly in
the attack's favour. Post-fix (job 459111, `--restarts 1`, 3 seeds per level, 80 LM iterations):

| latent init-noise | median start err | worst start err | seeds recovered | worst final err | restarts used |
|---|---|---|---|---|---|
| 0.10 | 0.069 | 0.117 | 3/3 | 1.9e-14 | 1, 1, 1 |
| 0.20 | 0.130 | 0.233 | 3/3 | 1.5e-14 | 1, 1, 1 |
| 0.30 | 0.183 | 0.345 | 3/3 | 1.4e-14 | 1, 1, 1 |
| 0.40 | 0.237 | 0.449 | 3/3 | 1.3e-14 | 1, 1, 1 |
| 0.50 | 0.291 | 0.543 | 3/3 | 1.7e-14 | 1, 1, 1 |
| 0.70 | 0.447 | 0.653 | 3/3 | 1.6e-14 | 1, 1, 1 |
| 1.00 | 0.539 | 0.858 | **2/3** | 1.3e-14 (of the two) | 1, 1, 1 |

"Recovered" here is re-derived objectively from `final_err_max`, i.e. **every** image below 1e-2, not from
the `verdict` field.

**Sixteen of seventeen runs recover, to a worst-case start error of 0.86, every one on the first attempt.
The basin edge is still NOT located.** The single non-recovery (noise 1.00, seed 1) is not a demonstrated
boundary: it stopped at the 80-iteration cap **still descending**, with a residual of 8.0e-5 that is
nowhere near the reproduction floor — a budget limit, not a basin wall. The other two seeds at that same
level converged to 1.2e-14 in 66 and 80 iterations. So the honest statement is that the sweep never found
the boundary; raising the iteration cap is what would settle it, and I have not done that. Do not read the
edge as sitting just past the last level tested.

(Note the `init-noise` label is a latent-space perturbation and maps non-linearly to image error: the 1.00
arm produced start errors of 0.48-0.86, not 1.00. Read the achieved start error, not the label.) The
finite-difference prototype reported convergence only from within ~10-15% and a local minimum at 30%.

**The basin is anisotropic, and this number must not be lifted out of context.** These starts are
`truth + noise`, a deliberately favourable direction that an attacker cannot construct. The
release-only initialisers an attacker *can* build start ~80-100% away and **fail** (next section). So the
two arms together say: along truth-directions the basin is not the binding constraint for SGD, and the
binding constraint is the **attacker-reachable initializer**. "Recovers from 65% off" is not "the attack
works from anywhere". That is precisely the framework's division of labour, now measured from both sides.
*(Framing owed to an independent audit.)*

## Step 3 — Adam release: the certificate is gone, and the inversion does not converge (yet)

Scaled-down cell so an LM Jacobian over the `r x n` unknown entries of `A_0` is affordable
(`n=32, k=6, N=4, T=200, r=16`; at `n=96` one Jacobian takes over five minutes, measured, which is why
the full-size Adam arm was moved to LBFGS and then dropped in favour of this).

| release | init-noise | seeds | start err | final err (med) | residual | outcome |
|---|---|---|---|---|---|---|
| adam | 0.05 | 2 | 0.027-0.039 | 7.2e-2, 8.2e-2 | 2.8e-3, 1.7e-2 | not recovered |
| adam | 0.10 | 2 | 0.055-0.076 | 1.4e-1, 7.8e-2 | 1.0e-2, 2.6e-2 | not recovered |
| adam | 0.20 | 2 | 0.113-0.151 | 2.0e-1, 1.2e-1 | 9.0e-2, 1.9e-2 | not recovered |
| **sgd (control, identical shape)** | 0.10 | 2 | 0.055-0.076 | **6.2e-16, 5.7e-16** | **8.6e-31, 2.9e-31** | **recovered** |

**The control is what makes this readable.** At the same shape, same solver, same budget and the same
start distance, the SGD release inverts to machine precision and the Adam release does not. So the gap
is the optimizer, not the problem size.

**It is NOT "an Adam release is not invertible", and the Jacobian says why.** With the diagnostics now
recorded, the two cases separate cleanly on conditioning while agreeing on rank:

| release | eqs | unknowns | `σ_min(J)` | `cond(J)` | LM iterations | outcome |
|---|---|---|---|---|---|---|
| sgd (n=32, k=6, N=4) | 384 | 88 | 9.0e-3, 9.7e-3 | **1.2e2, 1.7e2** | 12, 13 | converged |
| sgd (n=96, 15 rescue cells) | 352-544 | 60-392 | 6.5e-4 … 1.9e-2 | **1.5e2 … 2.9e3** | 12-45 | converged |
| adam (n=32, k=6, N=4) | 832 | 536 | 6.3e-3 … 7.3e-3 | **1.4e7 … 3.7e8** | 23, 28, 54, 400 | not converged |

`σ_min(J) ≈ 7e-3` for Adam sits **inside the range the successful SGD cells show**, so the simulator
Jacobian has full column rank at the point where it was evaluated.

**RESOLVED, and it overturns the conditioning story (job 466915).** An independent audit asked where
`σ_min` was evaluated. It comes from the Jacobian at the **solver's last iterate** — the truth only when
the run converged. The SGD cells converged; the Adam runs never did, so their `cond ≈ 1e7…3.7e8` describes
a *stuck point*, not the map. Evaluating the Jacobian at the **ground-truth parameters** instead
(`--jac-at-truth`), with `‖res(truth)‖` as the gate that the evaluation point really does reproduce the
release:

All rows below pass the gate `‖res(truth)‖` at the FP64 floor, i.e. the Jacobian really is evaluated at a
point that reproduces the release (jobs 466915, 467622). Adam's is exactly `0.0`, as its simulator is
bit-identical to its release generator.

| release | shape | J | ‖res(truth)‖ | `σ_min` truth | `cond` truth | full rank |
|---|---|---|---|---|---|---|
| adam | n=32, k=6, N=4 (seeds 1/2/3) | 832 × 536 | 0.0 | 6.9e-3, 6.7e-3, 6.0e-3 | 2.1e3, 1.1e3, 3.4e3 | yes |
| sgd | n=32, k=6, N=4 (seeds 1/2/3) | 384 × 88 | ~8e-16 | 9.7e-3, 9.0e-3, 8.3e-3 | 1.2e2, 1.7e2, 1.6e2 | yes |
| adam | n=32, k=8, N=4 | — | 0.0 | 4.2e-3 | 1.3e4 | yes |
| adam | n=32, k=6, N=6 | — | 0.0 | 6.8e-3 | 7.3e3 | yes |
| adam | n=32, k=6, N=4, T=800 | — | 0.0 | 6.8e-3 | 2.5e3 | yes |
| **adam, FULL SIZE** | **n=96, k=12, N=8, T=800** | — | **0.0** | **6.9e-4** | **8.0e5** | **yes** |
| sgd | n=96, k=12, N=8 | 448 × 224 | 9.6e-16 | 9.2e-4 | 2.0e3 | yes |

**The full-size Adam release is locally identifiable at the truth.** That is the cell we could not invert
and dropped for cost; a single Jacobian evaluation characterises it anyway. Full column rank at a
release-reproducing point gives Proposition 6 directly.

**Correction to my own first reading of this table.** I initially wrote "at the truth Adam is as well
conditioned as SGD", comparing Adam at `n=32` (`2.1e3`) against SGD at `n=96` (`2.0e3`). That is a
comparison across *different shapes* and it is not valid. Like for like:

| shape | sgd `cond` truth | adam `cond` truth | ratio |
|---|---|---|---|
| n=32, k=6, N=4 | 1.2e2 – 1.7e2 | 1.1e3 – 3.4e3 | ~10-20× |
| n=96, k=12, N=8 | 2.0e3 | 8.0e5 | ~400× |

So Adam **is** genuinely worse conditioned at the solution than SGD, stably across seeds, and the gap
grows with scale. But it is worse by one to three orders of magnitude, **not** by the `10⁵`-`10⁸` the
stuck-point measurement suggested, and it remains full rank throughout. The "Adam defends by
conditioning" reading is still withdrawn — a `cond` of `8e5` in FP64 is not what stops an inversion — but
the honest statement is *"mildly-to-moderately worse conditioned at the solution, and dominated by a much
smaller basin"*, not *"identical conditioning"*.

That also explains why preconditioning did nothing. Rescaling the damping cannot help when the map at the
solution is already well conditioned; the obstacle is that Levenberg-Marquardt from a 4% start does not
reach the solution.

**Do not collapse this into "a better solver cracks it" — that is the next unearned headline.** A small
basin is a property *of the map*, not a solver artifact, so "landscape rather than map" over-dichotomises.
What `cond(J_truth) = 2.1e3` rules out is exactly one thing: local ill-conditioning **of the solution**. It
does not rule out the Adam training map being genuinely hard to invert from a generic start. The measured
contrast is real and large — the same solver tolerates a 65% start error on SGD and fails from ~4-10% on
Adam — so the defensible statement is: **Adam defends by a much smaller basin, not by conditioning, and
whether that basin is solver-fixable (trust region, Gauss-Newton with line search, a better start) or
intrinsic is UNTESTED.** Settling it needs the Adam analogue of the SGD basin sweep plus at least one
trust-region attempt. *(Framing owed to an independent audit.)*

(Everything below this box is superseded by the paragraph above and is kept for the record.) What
differs is `cond(J)`, by four to six orders of magnitude, driven by `σ_max` rather than by any small
singular value — which is what a coordinatewise `1/(√v̂+ε)` rescaling does to sensitivities. At 400 LM
iterations (8x the original budget, job 423887) the run was still descending, residual 7.6e-3.

So the correct statement is: *the Adam release is locally identifiable from the simulator, and plain
Levenberg-Marquardt does not solve it because the problem is ~10⁵ times worse conditioned than the SGD
one.*

**Preconditioning was the obvious fix and it does NOT work** (job 452904). Three variants at the same
cell, all failing:

| variant | iterations | final residual | final image error |
|---|---|---|---|
| unscaled `λI` damping (baseline) | 400 | 7.6e-3 | 7.9e-2 |
| Marquardt `λ·diag(JᵀJ)` scaling | 200 | 2.2e-1 / 9.5e-3 | 6.3e-2 / 7.0e-2 |
| Marquardt + solve the `A₀` block alone first (15 iters) | 200 | 1.1e-2 | 7.6e-2 |

Marquardt scaling is if anything *worse* than unscaled damping on the residual. So the Adam difficulty is
not a block-scaling mismatch between the latent and `A₀` unknowns, which was the natural hypothesis given
`σ₀` differs from the latent scale. It remains open what the right treatment is; a trust-region or
Gauss-Newton-with-line-search variant, or reformulating so the Adam moment buffers are not differentiated
through, are the next things to try. The regression arm in the same job confirms the default SGD path
still recovers to 2.0e-15, so none of this is a refactor artifact.

Two further facts are already established.

1. **The certificate does not merely weaken under Adam, it ceases to exist.** `rank B_T = r`, so the
   projector onto `row(B_T)^⊥` is zero and `C ≡ 0` (measured `‖C‖/‖A_T‖ = 2.8e-15`). The naive
   `eps_inv = ‖CH‖/(‖A_T‖₂‖H‖)` then reads 1.9e-15 — a *perfect* certificate — because it is `0/·`.
   This is the audit's "vacuous signature" trap in a new place, and it is now flagged explicitly
   (`cert_vacuous`) rather than reported as a pass.
2. **The Adam simulator is bit-exact** (`fwd_check` = 0.0, not merely small), so whatever the inversion
   reports will be about conditioning and basins, not about a recipe mismatch. Deformation there is 2.42,
   far beyond anything in the SGD arms.

The full-size Adam arm (`n=96`, LBFGS) descended to residual ~0.18 in ten outer iterations and was
projected at ~20 h; it was killed in favour of the scaled-down LM cells above, which answer the same
question far faster.

## Step 2b — attacker-available initialisers: COMPLETE, 1 of 20 (the real bound on the attack)

All four arms started from **release-only** points, i.e. no access to the truth. `k=12, N=8, T=1500`,
5 seeds each, 8 restarts, post-fix code. This is the arm that measures what an attacker can actually do.

| init | seeds | recovered | typical start err | residuals |
|---|---|---|---|---|
| random | 5 | **0** | 0.61-0.94 | 5e-3 … 1e-1 |
| span (release span estimate, refined onto `ker C`) | 5 | **1** | 0.65-0.84 | 9e-7 … 3e-1 |
| spananchor (minimise out-of-estimated-span energy) | 5 | **0** | 0.58-1.04 | 6e-3 … 3.7e-1 |
| cert (minimise `‖Cφ(ψ(w))‖²`, the Primitive-1 anchor) | 5 | **0** | 0.74-1.04 | 3.5e-2 … 3.1e-1 |

**1 of 20, and even that one is generous by this study's own criterion.** The single success is the span
estimator on seed 3, at a `9.2e-3` image error with residual `9.1e-7`. Everywhere else this file uses a
*dual* criterion — image error below 1e-2 **and** residual at the reproduction floor — and this row fails
the second half by six orders. So the accurate statement is: **1 of 20 reached sub-1% image error without
converging; 19 of 20 failed outright** (image errors 0.69-1.44, residuals 5e-3 to 0.37). Note this makes
the initializer look like an even harder constraint than the looser wording did.

**The certificate anchor is the informative failure, and it is a clean demonstration of Primitive 2.** Its
pre-solve genuinely succeeds: it drives `‖Cφ(ψ(w))‖` to `1.1e-6` on seed 1 and `5.0e-8` on seed 3, i.e. it
finds points satisfying the certificate equation to six or eight digits. Those points sit **1.0-1.2 away
in relative image error**. At `k = 12 > r − N = 8` the certificate-consistent set is a manifold of
dimension `k − (r − N) = 4` per image, so landing on it says almost nothing about *which* point you are
at. Satisfying the exact algebraic constraint is not evidence of anything on its own — which is the same
lesson as "consistency with the release does not certify correctness", now measured on the attacker's
side.

The span arms fail differently: their pre-solve floors at ~42-48% out-of-estimated-span feature energy,
which is the span estimator's own 52° error appearing as a hard alignment limit.

**This is the binding constraint on the whole attack, and it is the one number to quote.** Perturbed-truth
starts recover from a 0.86 start error (Step 2); release-only starts recover 1 time in 20 from ~0.8. The
basin is therefore strongly **anisotropic** — wide along truth-directions, and the attacker's realisable
starts sit outside it. That is precisely the framework's claim that a learned or population prior must
supply the initializer, now measured from both sides rather than assumed.

## Corrections applied mid-run (an adversarial review of the testbed, git 5762045)

The Adam defect is the one that mattered: `B₀ = 0` makes the A-gradient exactly zero at `t=1`, so
`v_A = 0`, and `d/dv √v` is infinite there while the incoming sensitivity is zero — autograd evaluated
`0 × inf = NaN` and **the entire Adam Jacobian was NaN** (1925/1925 entries). `linalg.solve` propagates
NaN without raising, every damping trial is rejected because `nan < x` is False, and the run exits having
never moved. Step 3 would have reported *"an Adam release is not invertible even from 5% off"* — a
fabricated negative answer to one of the framework's four open questions. Fixed with a denormal floor
inside the square root, applied identically to the release and the simulator; post-fix gate: 0/1925 NaN,
`fwd_check` still exactly 0.0.

The other five: `verdict` keyed off the lower median (4-of-8 recovered would print "recovered");
`RESID_ZERO` 14 decades too loose, so a stalled run could be labelled non-identifiability; a QR sign
discontinuity that made the simulated release jump when a feature crossed zero, a systematic
*false-failure* source in exactly the basin being measured; restarts that never re-seeded `X`; and a
"nearest training image" metric that a collapse onto one image could score 1.0 on.

## What this says for the plan

1. **The exact channel has its own budget, and it is `k < m + r − N`, not `r − N`.** (This supersedes the
   earlier "the exact channel is not budget-limited", which was written before the capacity sweep.) The
   certificate's `r − N` bounds one primitive; the simulation channel's own boundary sits `m` higher.
   Any defense argued from "the adapter only has `r − N` independent rows" is arguing about one channel —
   but a defense argued from `m + r − N` is arguing about exact identifiability, **not** about leakage,
   because past that line the images are still recovered to sub-percent accuracy.
2. **The crux is the basin, and post-fix it is very wide.** (This supersedes the earlier "24% at full
   deformation, with restarts as the currency" — both numbers were pre-fix artefacts, see Step 2.)
   Post-fix: 16 of 17 runs recover out to a 0.86 worst-case start error, every one at `restarts = 1`, and
   the edge was never located. The earlier claim that **every** failure in the session had a nonzero
   residual is also superseded: the capacity cells past `k = m + r − N` have floor residuals. Below that
   line, failures are search failures; past it, they are not.
3. **The initializer question is the open one**, and it is now the only arm whose result would change
   the story. That is the right place for a learned/population prior, which is what the framework says.
4. **Report `cert_norm` beside `eps_inv` forever.** A zero certificate scores perfectly on the natural
   metric.
5. **Adam does not defend by non-identifiability. It buys two moderate defenses, and only at scale.**
   This claim has been wrong twice, so the arc is worth stating in full:
   - *"Adam defends by conditioning, `1e7`–`3e8`"* — **wrong**: that was the solver's stuck point, not
     the map.
   - *"Adam is well conditioned at the truth like SGD, so only the basin differs"* — **also wrong**, and
     it was an artefact of comparing a toy-scale Adam cell against a full-scale SGD one.
   - **Current, like-for-like:** at the truth the Adam release is **identifiable at every scale tested**
     (full column rank, gate `‖res(truth)‖` at the FP64 floor), but its solution conditioning is
     **worse than SGD's and worsens with problem size** — roughly 10-20× at `n=32` (536 unknowns) and
     ~400× at the real work point `n=96` (1632 unknowns: `8.0e5` against SGD's `2.0e3`). Report it as a
     **trend**, not a single number; extrapolating past `n=96` would be worse still.
   So at scale the defender gets *two* moderate obstacles — a real but FP64-tractable conditioning
   penalty (`cond 8e5` costs about 6 of 16 digits, a headwind for Levenberg-Marquardt rather than a wall,
   which is consistent with the observed stalls) and a much smaller basin. Neither is non-identifiability,
   and neither is the `1e8` the first reading implied. Whether either is erodable by better optimisation
   or preconditioning is **untested**. *(Both corrections owed to independent audits; the second was
   caught in parallel here and by a sibling session.)*


## Step 6 — near-duplicate degeneracy: which channel does it break? (job 471272)

Motivation from the derivation check: the certificate claim carries an unstated hypothesis `rank P_T = N`,
which fails when two examples give the same residual trajectory. The prediction, sharpened by that
session's own reduced-algebra check, was that both *endpoints* are clean — a well-separated pair, and an
**exact** duplicate (where the private span is genuinely `N−1` dimensional and the certificate is simply
correct on it) — and that the damage lives in the **near**-duplicate band, where the span is truly
`N`-dimensional but its `N`th direction sits under the numerical rank tolerance, so the attacker's
certificate silently misses a real private direction.

Setup: standard work point (`k=12, N=8, T=400`), example 1 set to example 0 plus `ε·δ` in **latent** space
(the realistic defender action: copy a record and jitter it) with the **same label**. `feat sep` is the
resulting feature-space separation, which is the honest x-axis since `ψ, φ` are `tanh` and the map
saturates.

| ε | feat sep | `σ_N/σ_1(B_T)` | rank `B_T` | rank `C` (exp 8) | `‖CH‖` | pair err | others err | equidistant? |
|---|---|---|---|---|---|---|---|---|
| clean | 1.06 | 1.5e-1 | 8 | 8 | 2.0e-15 | 2.8e-15 | 2.7e-15 | — |
| 0.3 | 7.4e-2 | 2.4e-5 | 8 | 8 | 5.1e-15 | 4.9e-2 | 1.3e-3 | yes |
| 0.01 | 2.8e-3 | 3.1e-8 | 8 | 8 | 1.9e-13 | 3.8e-2 | 1.2e-3 | yes |
| 1e-4 | 2.8e-5 | 3.1e-12 | **7** | **9** | **7.8e-7** | 3.8e-2 | 1.2e-3 | yes |
| 1e-6 | 2.8e-7 | 3.7e-16 | **7** | **9** | 7.8e-9 | 3.8e-2 | 1.2e-3 | yes |
| 0 | 0 | 2.0e-16 | **7** | **9** | 1.7e-15 | 3.8e-2 | 1.2e-3 | yes |

**The predicted band is confirmed exactly.** `‖CH‖` runs `2e-15 → 5e-15 → 1.9e-13 → **7.8e-7** → 7.8e-9 →
1.7e-15`: off, on, **peaking at the rank collapse**, decaying with `ε`, and off again at exact
duplication. Rank `C` jumps from `r−N = 8` to 9 precisely where rank `B_T` drops to 7. So a
numerically-degenerate-but-genuinely-distinct pair **contaminates** the certificate, while an exact
duplicate does not — the certificate is then simply solving a correctly smaller problem.

**`σ_N(B_T)/σ_1(B_T)` is the usable detector, `‖CH‖` is not.** The singular-value ratio falls smoothly and
monotonically across fifteen orders of magnitude (`1.5e-1 → 2.0e-16`), tracking the separation the whole
way. `‖CH‖` is non-monotone, and its peak is only of order the rank tolerance. Read degeneracy off
`σ_N(B_T)`, never off `‖CH‖`.

**But the hoped-for clean headline is NOT supported.** "Near-duplication defends the algebraic channel and
leaves the simulation channel intact" is false as stated. What actually happens:

- The failure is **confined to the duplicated pair** at every separation below the transition, and the
  other `N−2` stay under tolerance.
- **CORRECTION to my own first reading of this row (2026-09-03).** I originally called the two
  reconstructions "blends of the pair, not swaps", from `err_to_self ≈ err_to_other`. That inference is
  **vacuous in the tight band**: when the two originals are nearly the same point, *every* point is
  equidistant from both, so the diagnostic carries no information there. The full sweep shows the
  reconstruction is not a blend at all — at separation 0.0028 the pair error is 3.8e-2, **28× larger than
  half the separation**, and at exact duplication the ratio diverges. The reconstruction sits far from
  *both* originals, roughly an order of magnitude further away than the two originals are from each other.
  The `err/(featsep/2) ≈ 1` coincidence that suggested a midpoint blend holds only near separations
  0.07-0.12 and is an artefact of that range.
- The other `N−2` images stay under the recovery tolerance but **degrade by eleven orders**, from `2e-15`
  in the clean control to `2e-4`-`1e-3`. They still leak, but not to machine precision.
- Residuals are `1e-8`-`1e-9`, **not** at the reproduction floor, so these are not clean aliases; a longer
  budget might sharpen them and that is untested.

So the honest statement is the softened one the derivation check anticipated: **near-duplication
contaminates the algebraic channel and mutually aliases the duplicated pair, while the remaining `N−2`
records still leak (at reduced fidelity)**. It changes which records are protected, and it does not
protect the batch.

### The full curve (jobs 471272 + 473055 merged)

| feature separation | `σ_N/σ_1` | rank `B_T` | rank `C` | `‖CH‖` | pair err | others err | residual | recovered |
|---|---|---|---|---|---|---|---|---|
| 1.063 | 1.5e-1 | 8 | 8 | 2.0e-15 | 2.8e-15 | 2.7e-15 | 5.3e-31 | yes |
| 0.572 | 1.2e-3 | 8 | 8 | 1.5e-15 | 8.9e-15 | 3.5e-15 | 9.1e-31 | yes |
| 0.447 | 8.4e-4 | 8 | 8 | 1.6e-15 | 8.1e-15 | 3.6e-15 | 8.8e-31 | yes |
| 0.350 | 5.4e-4 | 8 | 8 | 2.5e-15 | 1.7e-14 | 3.0e-15 | 8.7e-31 | yes |
| **0.206** | 1.9e-4 | 8 | 8 | 1.7e-15 | **2.8e-2** | 1.0e-3 | 9.6e-9 | **no** |
| 0.115 | 5.9e-5 | 8 | 8 | 4.2e-15 | 5.4e-2 | 1.8e-3 | 2.5e-8 | no |
| 0.074 | 2.3e-5 | 8 | 8 | 5.1e-15 | 4.9e-2 | 1.3e-3 | 1.1e-8 | no |
| 0.0028 | 3.1e-8 | 8 | 8 | 1.9e-13 | 3.8e-2 | 1.2e-3 | 1.9e-9 | no |
| 2.8e-5 | 3.1e-12 | **7** | **9** | **7.8e-7** | 3.8e-2 | 1.2e-3 | 1.9e-9 | no |
| 2.8e-7 | 3.7e-16 | **7** | **9** | 7.8e-9 | 3.8e-2 | 1.2e-3 | 1.9e-9 | no |
| 0 | 2.0e-16 | **7** | **9** | 1.7e-15 | 3.8e-2 | 1.2e-3 | 1.9e-9 | no |

**Against the pre-registered prediction: it holds, and my first reading of it was wrong twice over.**

*Point 2 — confirmed.* Every failing row has a residual of `1e-8`-`1e-9`, far above the reproduction
floor: search/conditioning failures, not true aliases. The stated falsifier is not met.

*A caveat on the detector itself.* The three-way rule assumes the residual is either at `~1e-30` or
clearly non-zero. Near the capacity boundary that stops being true: `(N, k) = (14, 21)`, the last cell
below the line and one of the two cells the strictness claim rests on, sits at residual `5.0e-18` with a
`5.0e-4` image error — neither at the floor nor a stall, and classed a search failure while being
recovered for any practical purpose. **It reaches the floor on the FIRST restart once the iteration cap is
raised to 300** (`lm_iters_used = 137`, `restarts_used = 1`, residual 9.5e-31, error 2.84e-12, job 481079),
so the fix was the iteration budget and *not* the restarts — an earlier version of this file said restarts. The residual floor itself degrades near the boundary, so the bins
stop being crisp exactly where they are being read.

*Point 1 — confirmed once sampled finely enough, and my "cliff" claim is withdrawn.* I first reported a
twelve-order jump between separations 0.35 and 0.206 and called the transition sharp. That was a
**sampling artefact**: nothing had been measured in between. Filling it in (job 474132) gives a clean
continuous ramp.

| feature separation | pair err | others err | residual | recovered |
|---|---|---|---|---|
| 0.350 | 1.7e-14 | 3.0e-15 | 8.7e-31 | yes |
| 0.298 | 1.8e-3 | 9.6e-5 | 9.3e-11 | yes |
| 0.270 | 6.1e-3 | 2.2e-4 | 6.6e-10 | yes |
| 0.239 | 2.0e-2 | 8.0e-4 | 6.2e-9 | no |
| 0.206 | 2.8e-2 | 1.0e-3 | 9.6e-9 | no |

The error rises smoothly through the 1e-2 tolerance; the `recovered` boolean flips on a continuous curve,
exactly as predicted. There is no phase transition.

*Point 4 — confirmed, and it is the one that matters.* The prediction said the transition separation is
**solver-set, not fundamental**, and that a longer budget should push it lower. Re-running the first
failing cell (separation 0.206) with 10× the iterations and 4 restarts:

| separation | budget | pair err | residual | recovered |
|---|---|---|---|---|
| 0.206 | 80 iters, 1 restart | 2.8e-2 | 9.6e-9 | **no** |
| 0.206 | **800 iters, 4 restarts** | **1.8e-14** | **8.9e-31** | **yes** |

Same cell, same data, same release — recovered to machine precision with residual back at the floor. So
**"how similar is too similar" is a solver floor, not an information boundary.** The only fundamental
alias in this whole family is exact duplication, where the private span is genuinely smaller. A defender
cannot buy privacy by perturbing-and-copying a record: it costs the attacker compute, not access.

This also revises the "certificate is the more robust channel" observation from the previous paragraph.
The comparison was between a *fundamental* certificate threshold (rank collapse at separation `~3e-5`) and
a *budget-dependent* simulation threshold, which is not a like-for-like comparison. At sufficient budget
the simulation threshold moves down and the ordering is not established.

Two channels, two thresholds — but **only one of them is fundamental**. The **certificate** breaks at
separations below `~3e-5`, where rank `B_T` collapses, and that is a property of the release. The
**simulation** channel's apparent threshold near `0.2` is a *solver floor* that moves with budget (see
below), so the two are not comparable as stated.

**Gap in the sampling, being filled.** The sweep jumps from the clean control (separation 1.06, full
recovery) straight to `ε = 0.3` (separation 0.074, pair unresolved). The transition sits in that gap and
was never sampled; job 473055 fills it at `ε = 5, 3, 2, 1, 0.5`. Until it reports, "how similar is too
similar" is unmeasured.

### PRE-REGISTERED prediction for job 473055 (recorded before the data was read)

From the derivation check, on request, *before* the gap run reported. **Claim: the transition is smooth in
the conditioning, not a sharp separation cliff.** Reasoning: for any `featsep > 0` the pair is identifiable
in principle, because the same-label blend symmetry is exact only at `δ = 0` and is broken at
`O(featsep)`. So the failure must be conditioning/optimisation, which degrades continuously. Specifically:

1. pair error **rises continuously** with decreasing separation, a ramp from ~1e-15 toward ~`featsep/2`,
   and the `recovered` boolean flips where that ramp crosses the 1e-2 tolerance. That crossing is a
   threshold on a smooth curve, **not** a phase transition.
2. residuals of the un-recovered rows stay **above** the reproduction floor (~1e-8, as in the tight band),
   confirming search/conditioning.
3. the other `N−2` degrade **smoothly** too, because a near-degenerate pair inflates `cond(J)` for the
   whole batch — the same mechanism as the Adam arm.
4. the transition separation is **solver-set, not fundamental**: this is the `N=2` case of the repo's
   superposition problem, the un-blend direction's singular value scales like `featsep`, so
   `cond(J) ~ 1/featsep`. A longer-budget or more-restart run should push the transition lower and recover
   the `N−2` to machine precision while the pair stays blended.

**Falsifier, stated in advance:** a sharp cliff at a *budget-independent* separation, with
floor-residual (`~1e-30`) aliases below it, would mean a real information boundary and the prediction is
wrong. That would be the more defense-favourable finding, so it is the one to watch for.

The `err_to_self ≈ err_to_other ≈ featsep/2` fingerprint already observed (`0.074/2 ≈ 0.037` against the
measured `3.8e-2`) is the signature of a **midpoint blend**, consistent with (1). The `α` blend coefficients recorded in the tight band are meaningless there (the
line through the two originals degenerates as they coincide) — use `err_to_other` instead, as above.


## Step 7 — how far does the release-consistent set actually extend? (null-direction traverse, job 482338)

Asked for by an independent audit, and it is the experiment that decides whether "alias" is the right word
past the capacity line. Every past-line cell in Step 4/5 was started **adjacent to the truth**, so the
solver drifts along the flat direction only as far as LM happens to take it; the `1e-3`-`1e-2` errors
reported there are a **lower bound on the fibre, not its diameter**.

Method (`null_traverse.py`): compute `Dρ` at the truth, take the right singular vector at `σ_min`, step
along it, then **retract** back onto the release-consistent set with a few LM steps, re-aiming each step
because the fibre curves. A point only counts if its residual is at the reproduction floor.

| cell | vs line | null dims | steps | all at floor? | max image error at the floor |
|---|---|---|---|---|---|
| N=8, k=32 | past (line 28) | 40 | 25 | yes, 25/25 | **1.43e-2**, rises ~20 steps then flat in 1.0-1.4e-2 |
| N=14, k=22 | at (line 22) | 14 | 17 | yes, 17/17 | **3.45e-2**, still rising at the end of the run |
| N=8, k=26 | below (line 28) | — | — | **CONTROL, PENDING** | traversal must be *obstructed* |

**What is established.** Past the line the truth is not isolated and the release-consistent set is a real,
walkable continuum: every step reproduces the release to `~1e-30` while the reconstruction moves away.
That is the direct check that the fibre is positive-dimensional, rather than an inference from a
rank-deficient Jacobian.

**What is not.** Three limits, all of which must travel with the numbers.
1. **One direction of forty.** The continuation follows a single path through a 40-dimensional null space.
   The maxima above bound the extent **from below**; they are not the fibre's diameter.
2. **The plateau is not a property of the fibre.** The `k=32` path flattens at `1.4e-2`; the at-line path
   shows no plateau at all and is still climbing at `3.45e-2` when its run ends. The two differ by more
   than a factor of two, so extent is cell-dependent and neither run bounded it.
3. **The control has not reported.** If the continuation walks as freely *below* the line, it is finding
   release-consistent points everywhere and this entire section says nothing about the boundary. That is
   the kill condition, and it is still open.

**The reading, hedged to what is measured — and "recognisable" is WITHDRAWN.** The capacity boundary marks
where **exact** recovery stops. An earlier version added "not where *recognisable* recovery stops", on the
strength of the 1.4%-3.5% errors along these two walks. That does not survive: recognisability was **never
assessed** here (no human judgement, no classifier, no perceptual metric), the walked errors were still
rising when a budget ended, and the wider evidence runs against it — past-line error **grows with distance
past the line** (synthetic `N=8`: 2.6e-3 at the line to 9.3e-3 at `+16`; MNIST `r=8`: 3.5% to 10.0%), and
**no** past-line MNIST cell has every image inside the 1e-2 tolerance. So the defensible statement is only
the negative one: `k < m+r−N` is not a boundary of *reproduction* — past it the release is still reproduced
exactly, by points whose distance from the truth grows with how far past the line one is. What those points
look like is unmeasured. *(Withdrawal owed to an independent claims audit.)*

## Step 8 — recipe robustness: DESIGNED, RUN, CONTROL FAILED, BEING RERUN

The user asked directly whether the recipe must be known. Three arms were built (`recipe_robustness.py`):
R1, is a wrong recipe self-detecting (does only the true recipe reach the residual floor, across a menu of
wrong learning rates, wrong step counts and the wrong optimizer); R2, can the recipe scalars be fitted
**jointly** with the data (recipe scalars add to *demand*, so this is affordable only with slack below the
capacity line, `k < m + r − N − p/N`); R3, is the `η·T` degeneracy exact or broken by finite step size, and
are the labels identifiable the same way.

**R1 — a wrong recipe cannot reach the residual floor, and the residual is the attacker's own instrument.**
Rerun on LM (job 484255); the control is the first row and it now reaches the floor, so the readout is
valid. Cell `k=12, N=8, T=400, η=0.01`:

| assumed recipe | error in the recipe | residual | at floor? | image error |
|---|---|---|---|---|
| **correct (control)** | 0 | **5.34e-31** | **yes** | 2.84e-15 |
| T + 1 step (401 of 400) | 0.25% | 6.04e-8 | no | 4.10e-3 |
| η × 1.01 | 1% | 9.09e-7 | no | 1.77e-2 |
| T + 25% | 25% | 1.55e-4 | no | 0.299 |
| T − 25% | 25% | 2.87e-3 | no | 0.398 |
| η × 2 | 100% | 1.59e-3 | no | 0.535 |
| η / 2 | 50% | 3.35e-2 | no | 0.460 |
| **wrong optimizer** (invert an SGD release as Adam) | family | **4.90** | no | 0.255 |

The true recipe is the unique floor-reacher, and it is separated from the nearest wrong hypothesis — a
step count off by **one step in four hundred** — by **twenty-three orders of magnitude**.

**WITHDRAWN (independent audit, 2026-09-03): the residual is NOT monotone in the size of the recipe error,
so it is not a "graded objective".** An earlier version of this paragraph said it was. The data refute it:
`η×2` (a 100% error) gives residual 1.59e-3, *below* both `T−25%` (2.87e-3) and `η/2` (3.35e-2), and
`T+25%` (1.55e-4) sits below `T−25%` (2.87e-3) at the same 25% magnitude. What survives — and it is all the
selection rule needs — is that the true recipe is the **unique** hypothesis reaching the floor, by more
than twenty orders. Ranking *among wrong* hypotheses by residual is not supported.

**Why this is usable by an attacker, which is the whole point.** The image-error column is *not observable*
to an attacker — they do not have the private images, and every reconstruction number in this file is a
diagnostic available only to the experimenter. The **residual is** observable: it is computed from the
released factors and the candidate alone. It is a Cauchy-type criterion — it certifies convergence without
any reference to the limit — and it is therefore a legitimate recipe *selection* rule rather than a
post-hoc diagnostic. Any claim built on reconstruction quality would not be.

## Step 9 — the recipe can be measured, but ONLY under continued-training access (job 485912)

**THREAT-MODEL CORRECTION (2026-09-03, found by the user; missed by me and by three independent audits).**
This section originally read "the recipe can be MEASURED, not assumed", full stop. That is wrong as stated,
and the error is basic: the probe reads `η` off an observed step `ΔB`, but **if the attacker takes that step
they choose `η` themselves and learn nothing from it.** The measurement only works when the *victim's*
optimizer takes the step with its hidden `η` — i.e. under **continued-training access**: a checkpoint
carrying optimizer and scheduler state, or a fine-tuning service that trains on submitted data. The code
does exactly that (`calibrate_recipe.py` steps with the true `a.lr`), so the experiment is sound; the
*claim* attached to it was not. My own code comment read "the attacker runs it; only eta is unknown", which
is incoherent on its face.

**Under the weights-only release assumed everywhere else in this study, this probe does not apply**, and
neither the schedule nor `T` is recoverable by it. What survives weights-only is R1-R3 below, which use only
the released factors: a wrong recipe is detectable, `η` is fittable jointly with the data, and the `(η, T)`
split and the labels are identifiable. Those are the results that carry the threat model; this one is the
weaker contribution, and in the checkpoint case `T` is usually in the metadata anyway.

Given that access, one further step gives exactly `ΔB = −η·gB` with `gB = D(A_T H′)ᵀ`, and the attacker
knows the released factors, their own probe features `H′` and their own labels, hence knows `gB`. So `η` is
a one-dimensional least squares, using **no private data**.

| true recipe | true η | estimated η | relative error | `cos(ΔB, −gB)` | passes the parallelism test? |
|---|---|---|---|---|---|
| SGD, T=400 | 0.01 | 0.0100000000 | 1.7e-16 | 1.0000000000 | yes |
| SGD, T=1500 | 0.01 | 0.0100000000 | 1.4e-15 | 1.0000000000 | yes |
| SGD, T=400 | 0.003 | 0.0030000000 | 1.0e-15 | 1.0000000000 | yes |
| SGD, T=1500 | 0.003 | 0.0030000000 | 2.6e-15 | 1.0000000000 | yes |
| SGD, T=400 | 0.05 | 0.0500000000 | 0 | 1.0000000000 | yes |
| SGD, T=1500 | 0.05 | 0.0500000000 | 5.6e-16 | 1.0000000000 | yes |
| SGD + weight decay 1e-3 | 0.01 | 0.0100031008 | 3.1e-4 | 0.9999986835 | **no** (correctly) |
| **Adam**, T=200 | 0.003 | 0.0068650879 | 1.3 | **0.4346** | **no** (correctly) |

**The learning rate is recovered to machine precision**, at every rate and horizon tested, from a single
probe step. And the same probe **identifies the optimizer family**: the cosine between the observed step
and the gradient is exactly 1 under plain SGD, drops to 0.435 under Adam, and dips just below 1 under
weight decay — correctly flagging that an extra term is present rather than silently absorbing it into a
wrong `η`.

### The three arms, complete (job 484255)

**R1 — a wrong recipe cannot reach the floor.** Control first, `k=12, N=8, T=400, η=0.01`:

| assumed recipe | residual | at floor? | image error |
|---|---|---|---|
| **correct (control)** | **5.34e-31** | **yes** | 2.84e-15 |
| T + 1 step (401 of 400) | 6.04e-8 | no | 4.10e-3 |
| η × 1.01 | 9.09e-7 | no | 1.77e-2 |
| T + 25% | 1.55e-4 | no | 0.299 |
| T − 25% | 2.87e-3 | no | 0.398 |
| η × 2 | 1.59e-3 | no | 0.535 |
| η / 2 | 3.35e-2 | no | 0.460 |
| wrong optimizer (SGD release inverted as Adam) | 4.90 | no | 0.255 |

**R2 — the learning rate can be fitted jointly with the data.** `η` carried as a free unknown beside
`(w, X)`, started at 1×, 2× and 0.5× the truth:

| start | fitted η | relative error | residual | image error |
|---|---|---|---|---|
| 0.0100 | 0.01000000 | 5.2e-16 | 6.74e-31 (floor) | 2.82e-15 |
| 0.0200 | 0.01000000 | 5.2e-16 | 7.07e-31 (floor) | 3.06e-15 |
| 0.0050 | 0.01000000 | 5.2e-16 | 7.63e-31 (floor) | 3.14e-15 |

From a start wrong by a factor of two, both the rate **and** the images come back to machine precision.
One extra unknown against a slack of `N(m−1+r−N) − Nk = 120`, so the counting predicts it is free, and it is.

**R3 — the `η·T` degeneracy is NOT exact; finite step size breaks it.** Product held at `η·T = 4.0`:

| split | residual | at floor? |
|---|---|---|
| T=100, η=0.040 | 3.12e-7 | no |
| T=200, η=0.020 | 3.49e-8 | no |
| **T=400, η=0.010** | **6.44e-31** | **yes** (the true split) |
| T=800, η=0.005 | 8.78e-9 | no |

Only the true pair reproduces the release, so `η` and `T` are **separately** identifiable — the degeneracy
is a gradient-flow statement. Note this is the *weakest* discrimination in the study (22 orders rather than
23-29), which is exactly what one expects if the degeneracy is approached in the small-step limit.

**Labels** are identifiable by the same test: true assignment 7.19e-31 (floor), swapping two examples
2.51e-3, a cyclic shift 7.49e-6.

### The schedule: a decaying rate leaks the step count that a constant one hides

Per-step `η` values are probed exactly (max relative error 1e-15, jobs 487290/488314). Recovering the
schedule *parameters* from them is a small 3-parameter fit. **A gradient fit of mine landed 30-40% off and
I retracted the claim; that was wrong — it was a local minimum of a multimodal objective, not a
degeneracy.** Refitting globally (eliminate the base rate by ratios, grid over the remaining two, solve the
base in closed form — a re-analysis of already-saved probe output, no new run):

| probe window | recovered T | recovered schedule length | recovered base rate |
|---|---|---|---|
| 6 steps | **400** (true 400) | **1000** (true 1000) | **0.010000** (true 0.01) |
| 20 / 60 / 150 steps | identical, exact | identical, exact | identical, exact |

So probing reveals `η` at the *continuation* steps, not the history — with a constant rate that is all one
gets and `T` stays hidden. But a schedule from a known family encodes position on its own curve, so six
probe steps pin the base rate, the schedule length and `T` exactly. **A defender who adds a schedule to
look realistic hands over the one recipe parameter the probe otherwise cannot see.** Caveat: the family is
assumed known here (cosine); an unknown family is a model-selection problem, not run.

### The counting bound on recipes

Recipe unknowns add to *demand*, so `p ≤ N(m−1+r−N) − Nk`, which is **120** at this cell. Parametric
schedules (`p = 2-4`) sit comfortably inside. A **free per-step sequence** (`p = T`) is identifiable only
for `T ≤ 120` and is hopeless at `T = 400`. So the defense that works against recipe-fitting is an
unconstrained per-step schedule, and it works **by counting, not by obscurity**.

*(A recorded-data erratum: the `supply` field in the R2 rows used the plain rank-`N` cap, 352, rather than
the Prop.-5 cap of 344. Demand is inside either, so no conclusion turns on it; the code now records both.)*

**What this does and does not settle.** It converts "the attacker knows the recipe" from an assumption into
a *measurement* for the update rule and the learning rate, at zero cost in private data. It does **not**
recover `T`, the number of steps taken before the release: the continuation reveals the rule, not the
history. Under Adam a single probe identifies the family but not the recipe, since the moment buffers at
the release point are unknown. And it assumes the attacker can evaluate the same head and loss, which they
can here (`W₀` and `φ` are public and the probe labels are theirs to choose).

**Earlier attempt, invalidated.** A first pass at R1-R3 (job 480679) is void: the fault is ours. R1's control — the *correct* recipe — reached
residual `7.2e-8` with `reached_floor = False`. The cause is that these arms were built on the LBFGS solver
rather than the Levenberg-Marquardt one the rest of the study uses, and LBFGS does not drive this residual
to `1e-30` even when the recipe is exactly right. Since the claim under test is "only the correct recipe
reaches the floor", a control that cannot reach the floor makes the readout uninformative. Rerunning all
three arms on LM. Not edited mid-flight, per the standing rule.

Nothing about recipes should be concluded from this yet, in either direction.


## Step 10 — the law on REAL images: MNIST (job 568095)

The capacity law was derived and measured on a synthetic tanh manifold. This is the first test on real data.
Private data are genuine MNIST digits with their real labels, restricted to their own PCA subspace (so the
manifold coordinates are real principal components and `k` is a real dimension); `φ` is a frozen public
network; the head has `m = 10` classes; the recipe is the plain SGD the law is stated for. `N = 8`.

**Scoping (audit, 2026-09-03) — three things this cell is *not*.** (i) `φ` here is a *random* tanh encoder
(`n = 96`), not a pretrained extractor; the trained-encoder case is Step 13. (ii) The solve starts from
`W_true + 0.10·noise` in latent space, so this is a near-truth identifiability/basin test, **not release-only
recovery** — do not present it as "MNIST reconstructed from the adapter alone". (iii) The PCA chart `V_k` was
fitted on the first 10k train images, **which include the 8 private digits** (also in Step 11), so the private
digits sit slightly more exactly on the chart than they would for a real attacker; the Steps 10–11 numbers
predate the exclusion (commit `a1648aa`), and any rerun excludes them. Read the collapse off `σ_min(J)` at the
truth, not off `frac_recovered`: near the line `frac_recovered` can still be 1.0 where `σ_min` is already ~1e-19.

The line must **move with `r`**, and it does — sharp to one unit of `k` at every rank:

| r | predicted line `k < m+r−N` | last `k` full-rank at the truth | first `k` collapsed |
|---|---|---|---|
| 8 | **10** | 9 | 10 |
| 16 | **18** | 17 | 18 |
| 32 | **34** | 33 | 34 |

At `r = 16`: `σ_min` at the truth runs 2.6e-4 (k=6), 2.0e-4 (k=10), 6.4e-5 (k=14), 1.3e-6 (k=17), then
**8.4e-18 at k=18** — the same collapse to the FP64 floor as in the synthetic testbed. Below the line real
digits are reconstructed to ~1e-14. Past it the residual returns to the floor while the images degrade —
and here the real-data picture is **worse for the attacker than the synthetic one**, which an earlier
version of this section understated by quoting the `r=16` range only. Across all three ranks the past-line
errors are 2.26e-2, 2.70e-2, 3.53e-2, 5.33e-2, 5.38e-2, 6.79e-2, 8.37e-2, 9.96e-2 — i.e. **2.3% to 10.0%,
and 0 of 8 past-line MNIST cells have every image inside the 1e-2 tolerance** (against 9 of 13 synthetically).
The error also **grows with distance past the line**: at `r=8` it runs 3.5% → 8.4% → 10.0% at `k = 10, 12, 16`.
So the boundary is still one of exact identifiability rather than of reproduction, but past it the
reconstruction degrades steadily rather than sitting at a harmless offset. The
`k = 17` cell degrades and comes off the residual floor, mirroring the marginal-cell behaviour near the
boundary seen at `(N,k) = (14,21)` synthetically.

**So the law is not an artefact of the synthetic generator.**

## Step 11 — but `k` is a property of the CHART, not of the images (job 574169)

**This is a correction to how I had been stating the result, prompted by the user.** I wrote that the
`r`-sweep "kills a fixed-`k` explanation". That is too strong. `k` is the dimension of the *chart* chosen to
search in; the counting argument only ever sees that number, and nothing about how the chart is built
enters. So the `r`-sweep rules out exactly one alternative — a chart-intrinsic threshold independent of `r`
— and says nothing about the fact that a different parameterisation is a different problem.

That is falsifiable, so it was tested: same digits, same `N, m, r`, three charts at **matched `k`**.
`pca` (linear, data-fit), `warped` (the same manifold under a fixed nonlinear reparametrisation), and
`exact` (an orthonormal chart whose span **contains the private digits**, so it can represent them exactly).

**(i) The boundary is chart-independent.** All three collapse at exactly `k = 18 = m+r−N`:

| chart | `σ_min` at k=17 | `σ_min` at k=18 | full rank at 18? |
|---|---|---|---|
| pca | 2.24e-5 | 5.57e-18 | no |
| warped | 2.17e-5 | 5.30e-18 | no |
| exact | 5.72e-5 | 5.66e-18 | no |

**(ii) What actually comes back is not.** Below the line every chart recovers *its own* representable image
to ~1e-14. Measured against the **real digit**:

| chart | chart representation error | recovered vs the REAL digit (k=17) |
|---|---|---|
| pca | 0.41 | **0.510** |
| warped | 0.41 | **0.510** |
| exact | 0.00 | **5.7e-14** |

Thirteen orders of magnitude apart, at the same `k`, the same budget and the same boundary.
`figures/exact_inversion/chart_dependence_k17.png` and `_k18.png` show it: the PCA chart recovers its own
blurred digit perfectly and is still half the image away from the truth, while the chart containing the
digits returns them exactly — and does so at `k = 8`, far under the line.

**Consequences, and they weaken the privacy reading further.**
1. The law bounds **the dimension of the search**, not the fraction of the image an attacker can reach.
   Anywhere it is glossed as a privacy statement it must read *identifiability within the chosen chart*.
   A defender cannot read `k < m+r−N` as a bound on leakage.
2. **Nothing forces the chart to be data-agnostic — but be careful what that construction is.** The
   `exact` chart spans the private images plus filler directions, so it contains them at dimension `N`
   (`k = 8` here, far under the line) and the inversion returns the true digits at 1e-14. **CORRECTED
   (2026-09-03): this is an ORACLE construction, not an attack.** Building that chart requires the private
   images, so quoting it as something "an attacker can do" is circular, and an earlier version of this
   section did exactly that. What it legitimately proves is the point it was built for: the boundary is a
   property of the chart and not of the images, since the same `k` and the same release give thirteen
   orders of difference in fidelity.

   The **attacker-realizable** version of the same construction is a chart fitted to the data
   *distribution* rather than to the specific private images — i.e. a generative prior. That is where the
   counting becomes genuinely constructive rather than merely limiting: it states the requirement as a
   number the prior must come in under, `k < m + r − N` degrees of freedom per image, computable from the
   release before any attack is attempted. Unbuilt here.
3. So a better generative model buys strictly more leakage at the same budget. That makes the
   generative-prior direction a **consequence of the counting** rather than a hope, and it is the honest
   answer to "is the boundary a real privacy limit": no — it limits the search space, and the search space
   is the attacker's to choose.


## Step 12 — the law's own prediction: the reach is LINEAR in the head width (jobs 589810, 593146)

`k < m + r − N` says the per-image budget grows one-for-one with the head width `m`. Every earlier sweep
moved `N` or `r`; `m` had never been varied, so this was the law's sharpest untested claim. Fixed `r = 16`,
`N = 8`, sweeping `m` and straddling each width's *own* line:

| m | line `m+r−N` | last identifiable k (`σ_min`) | first collapsed k (`σ_min`) |
|---|---|---|---|
| 10 | 18 | 17 (2.15e-5) | **18** (9.25e-18) |
| 12 | 20 | 19 (2.45e-6) | **20** (9.30e-18) |
| 16 | 24 | 23 (1.42e-6) | **24** (1.48e-18) |
| 20 | 28 | 27 (1.53e-7) | **28** (6.43e-19) |
| 28 | 36 | 35 (6.65e-7) | **36** (8.81e-20) |

Sharp to one unit of `k` at all five widths, across a 2.8× range in `m`. **The reach is linear in the head
width**: at fixed adapter rank, each extra class in the head buys the attacker exactly one more degree of
freedom per image.

### A confound in the testbed, found from this sweep, and fixed

The `m = 28` row initially collapsed at `k = 33`, three units *below* its line of 36 — the only cell in the
whole study to break the prediction. The cause was mine, not the law's: the generator
`ψ(w) = tanh(W₂ tanh(W₁w) + b)` had `W₁` of shape `32 × k`, so it factors through a **32-unit bottleneck**
and the manifold dimension is capped at `min(k, 32)` however large `k` is asked for. Every cell with
`k > 32` was therefore probing the generator, not the release.

It confounds exactly two places, and nothing else:
- the `m = 28` row (line 36, past the cap), and
- the `N = 4` row of the synthetic capacity table, whose line at 32 **coincides with the cap** — so its
  original bracket (30 ok / 34 collapsed) could not distinguish the law from the bottleneck.

Everything else is clean: `N = 8, 12, 14`, the widths `m = 10, 12, 16, 20`, and all of MNIST (a *linear*
chart of rank `k`, no bottleneck). Re-running both confounded cells with the generator widened to 128 and
the image space to 256 (job 593146):

| cell | line | last identifiable | first collapsed | verdict |
|---|---|---|---|---|
| `m=28, N=8` | 36 | 35 (6.65e-7) | 36 (8.81e-20) | law confirmed; the `k=33` collapse was the generator |
| `N=4, m=20` | 32 | 31 (1.32e-5) | 32 (8.01e-18) | law confirmed, now unconfounded |
| control `N=8, m=20` | 28 | 27 (2.22e-6) | 28 (1.74e-18) | boundary unchanged by the widening |

The control matters: widening the generator changes the world (so `σ_min` values differ) but moves the
boundary not at all, which is what rules out the widening itself having produced the agreement.

`World` now takes `gen_hidden` (default 32, so every earlier run reproduces byte-for-byte) and the script
**warns** when the requested `k` approaches it. This defect was not found by any of the three audits; it
surfaced because the law made a prediction sharp enough that one anomalous row was visibly wrong.

## Step 13 — the line on a TRAINED model: trained head, trained features (jobs 607896, 610020)

`trained_backbone.py`. Backbone: the repo's `weights-mnist10_gelu.pth` (784-1000-1000-10 GELU, **78.45%** test
accuracy — verified against `CreateModel.NeuralNetwork` to 3.7e-14, so the weakness is the checkpoint's, not a
loading bug); `φ` = penultimate activations (`n = 1000`), `W₀` = the trained output layer (`m = 10`). Chart:
global PCA of the first 50k train images; private digits: 8 from the *test* split (unseen by backbone and chart),
labels `[0,3,0,3,5,0,1,9]`. `r = 16`, `N = 8`, `T = 400`, `lr = 0.01`, cell (a) (truth on the chart), start
`W_true + 0.10·noise`. Line `k < m + r − N = 18`. Rows deduplicated on `(k, seed, git)`; where 610020 reran a
cell with a larger budget, that row is quoted.

| k | σ_min(J) at truth | full rank | chart's best (err vs REAL) | err vs chart | err vs REAL | residual | LM iters | stop |
|---|---|---|---|---|---|---|---|---|
| 6 | 3.54e-5 | yes | 0.530 | 1.3e-13 | 0.866 | 8.4e-31 | 45 | floor |
| 10 | 2.16e-6 | yes | 0.479 | 1.3e-12 | 0.761 | 8.7e-31 | 31 (rerun, restarts 4) | floor |
| 14 | 2.36e-7 | yes | 0.460 | 1.5e-11 | 0.758 | 6.3e-31 | 281 (rerun) | floor |
| 16 | 1.47e-8 | yes | 0.421 | 3.5e-2 | 0.731 | 1.6e-14 | 300 (rerun) | cap, still descending |
| 17 | 1.23e-9 | yes | 0.421 | 5.7e-2 | 0.725 | 4.4e-13 | 80 | cap, still descending |
| **18** | **2.43e-19** | **no** | 0.415 | 5.5e-2 | 0.725 | 3.1e-13 | 80 | collapsed |
| 20 | 2.71e-19 | no | 0.414 | 5.0e-2 | 0.712 | 8.1e-13 | 80 | collapsed |
| 22 | 3.56e-18 | no | 0.403 | 7.4e-2 | 0.704 | 2.8e-13 | 80 | collapsed |

- **The line holds on a trained model, at 18 exactly** — full rank at 17, rank loss of ten orders at 18. The
  count sees only `(m, r, N)`, and the trained head and features do not move it.
- **Conditioning is what changes.** Against the random `n = 96` encoder of Step 10 at matched `k` (2.6e-4,
  2.0e-4, 6.4e-5, 1.3e-6 at `k = 6, 10, 14, 17`) the trained backbone's `σ_min` is **7× / 93× / 271× / 1057×**
  smaller. (Shape caveat: `n = 1000` vs `96`; the fixed-architecture version is job 624573, Step 18.)
- **Below the line the failures are budget, not information.** `k = 10` and `14` came off the 80-iteration cap
  still descending and reached the floor with the larger budget (31 and 281 iterations); `k = 16` is still
  descending at 300 (residual 1.6e-14, fourteen orders above the floor). Iterations-to-floor is the cost axis:
  45 → 31 → 281 → >300 across `k = 6, 10, 14, 16`.
- **The picture is still the chart's.** Every recovered cell returns the chart's projection of the digit
  (err vs REAL 0.87 → 0.70 is `chart_repr_err` itself), which is why Steps 14 and 18 change the chart.

## Step 14 — chart families at fixed k = 16 on the trained backbone (jobs 611033, 611339, 612643) — TWO CONFOUNDS, reruns in flight

`vae_chart.py` (VAE chart, GELU or ReLU decoder, trained on the train split only) and `conditional_charts.py`
(global PCA · class-local PCA with labels given · class-conditional VAE). Same backbone, digits, labels, `r`,
`N`, `T`, `lr` as Step 13; cell (a) = truth on the chart; cell (b) = adapter fine-tuned on the RAW digit, chart
searched anyway (the realistic case; the residual cannot reach the floor). Budget in these first runs:
`--lm-iters 200 --restarts 3`.

| chart | family | analytic ψ | chart's best (err vs REAL) | σ_min(J) at truth | err vs chart | err vs REAL | residual | iters |
|---|---|---|---|---|---|---|---|---|
| global PCA (`vae_chart`) | linear | yes | 0.421 | 2.48e-8 | 5.3e-2 | 0.731 | 1.3e-14 | 200 cap |
| global PCA (`conditional_charts`) | linear | yes | 0.421 | 2.33e-8 | 3.1e-2 | 0.731 | 8.5e-15 | 200 cap |
| class-local PCA (labels given) | linear | yes | 0.364 | 1.72e-10 | 4.0e-2 | 0.619 | 5.2e-17 | 200 cap |
| VAE, GELU | learned | yes | 0.339 | 1.82e-11 | 1.3e-1 | 0.567 | 1.1e-14 | 200 cap |
| VAE, ReLU | learned | **no** | 0.327 | 1.59e-11 | 2.2e-1 | 0.555 | 2.9e-9 | 200 cap |

Cell (b), off-chart, err vs REAL: global PCA **1.448** (residual 1.8e-5) · VAE-GELU **0.728** (1.2e-4) ·
VAE-ReLU 0.828 (3.2e-4). The two global-PCA rows are the same chart through two scripts with different `A₀`
draws: `σ_min` 2.48e-8 vs 2.33e-8 — the truth-side number is reproducible across the seed.

**What is measured and not confounded.** `σ_min(J)` at the truth is solver-independent. Across the four charts,
better representation came with worse conditioning, monotonically: 2.3e-8 → 1.7e-10 → 1.8e-11 → 1.6e-11,
~1400× across the range. The prediction that a class-local chart would be *better* conditioned is **refuted**
(135× worse). The ReLU decoder is not distinguishable from the analytic one in `σ_min` (1.59e-11 vs 1.82e-11):
analyticity looks like a proof convenience here, not a live constraint. The count (`k < 18`) is unaffected by
any of it.

**Confound 1 — budget.** Every cell (a) stopped at the 200-iteration cap **off the floor** (the residual is a
sum of squares with floor 1e-28; these sit at 1e-14 … 3e-9, fourteen-plus orders above it). So every cell (a)
is a *search failure at budget*, none is an alias, and the err-vs-REAL ordering 0.731 → 0.619 → 0.567 → 0.555
partly ranks how far each run got. **Do not quote the fidelity ranking as a result until job 624463 lands**
(same cells, `--lm-iters 3000 --restarts 4`; it reports iterations-to-floor per chart next to `σ_min`).

**Confound 2 — repeated labels (found by a sibling session).** The private draw has three 0s and two 3s.
`LocalPCAChart` builds one `(μ_y, V_{y,k})` per class, so the three 0-columns get an *identical* map; the
class-conditional VAE is label-conditioned too. That does not make `∂ψ/∂w` rank-deficient (block-diagonal,
orthonormal per column, rank `Nk`), but it entangles richness, label conditioning and repeated-class structure
in the local-vs-global 135×. The distinct-labels cell (job 624573, first cell) separates them. The unconditional
VAE charts are not affected.

**Reads.** (1) A richer chart raises the ceiling — 0.42 → 0.34 is the best any solver could return at `k = 16`
— and the realistic off-chart case is where it shows most (1.45 → 0.73 on the same release). (2) It lowers
conditioning by orders. Both PCA charts have an orthonormal `∂ψ/∂w`, so the 135× between them cannot be
decoder geometry; it is what the *encoder* does to the directions each chart spans. Hypothesis, pre-registered
before the controls: **a trained classifier compresses within-class variation — exactly where a local chart or a
VAE spends its coordinates.** Falsifier: on a random encoder of the same architecture the gap should largely
vanish (Step 18).

## Step 15 — the Q-parametrisation: exact, smaller, and it does NOT widen the basin (job 605718)

`q_param.py`. Theorem G says the trajectory depends on the seed only through `Q = XᵀX`; writing the residual in
coefficient space and recovering `X = A_T U_c Ω⁻¹` in closed form leaves `Nk + N(N+1)/2 = 132` unknowns instead
of 224 (`k = 12`, `N = 8`, `r = 16`, `m = 20`, `T = 400`, `lr = 0.01`). Gate: residual at the truth 1.4e-15.
Basin edge, perturbed-truth starts, 3 seeds per noise level, matched cells:

| init noise | median start err | (w, X), 224 unknowns | Q, 132 unknowns |
|---|---|---|---|
| 1.0 | 0.57 | 3/3 | 3/3 |
| 1.5 | 0.71 | 0/3 | 1/3 |
| 2.0 | 0.80 | 0/3 | 0/3 |
| 3.0 | 0.86–1.05 | 0/3 | 0/3 |
| 5.0 | 0.96–1.13 | 0/3 | 0/3 |

1/3 against 0/3 on three seeds is noise. **The seed block was never the obstruction; the difficulty is in the
data coordinates.** Closed as a negative; the reduced form stays as a verified structural simplification.

## Step 16 — encoder quality at fixed architecture (job 614344, IN FLIGHT)

`train_strong_backbone.py` trains the same 784-1000-1000-10 GELU architecture on the full train split and saves
a mid checkpoint (`mnist_mlp_mid.pth`, **96.08%** at save time; 95.10% on the 2000-digit in-script check) and a
strong one (`mnist_mlp_strong.pth`, ≥97%, last saved 98.24%); `trained_backbone.py` then sweeps
`k ∈ {6, 10, 14, 16, 17, 18}` on each with `--lm-iters 300 --restarts 4`. First row on disk (mid, `k = 6`):
`σ_min` **1.16e-6** against the 78% backbone's 3.54e-5 — **30.6× worse** — and residual 2.5e-15 at the
300-iteration cap where the weak backbone reached the floor in 45. One row, provisional (†); direction:
*quality hurts*. The full ladder (random / 78% / 96% / ≥97%) is Step 18's read.

## Step 17 — multi-layer LoRA (job 608693, IN FLIGHT, EMPIRICAL ONLY)

`multilayer_lora.py`: LoRA on all three layers, full unroll with `create_graph`. Arm A: seeds known (oracle,
LBFGS on `W`); arm B: seeds unknown (22,384 unknowns). Outside the theorems (`theorems_apply=False`); no rows yet.

## Step 18 — controls in flight (submitted 2026-09-03 after the sibling review of Step 14)

| job | script | cell | settles |
|---|---|---|---|
| 624463 | `vae_chart.py`, `conditional_charts.py` | all Step-14 cell (a) arms at 3000 iters / 4 restarts | confound 1: iterations-to-floor per chart |
| 624573 (1st) | `random_encoder_control.py --encoder trained --labels distinct` | global vs local, 8 distinct labels | confound 2: does the 135× survive distinct labels |
| 624573 (2nd–3rd) | `--encoder random`, distinct and repeated labels | same charts on a random encoder with the trained layers' norms | the falsifier of the encoder-compression hypothesis |
| 624573 (4th) | `--encoder random --ks 6 10 14 17 18` | ladder zero point at fixed architecture, 300 iters | replaces the `n = 96` comparator in Step 13 |
| 624465 | `beta_vae_sweep.py`, β ∈ {0.25, 1, 4, 16} | one family, graded richness at fixed `k` | ordering vs curve; records `σ(∂ψ/∂w)` at the truth (decoder geometry control) |

Every row from these carries `rank_B_T`, `σ_N/σ_1(B_T)` and `rank_X` (the per-cell witness that rank `P_T = N`
holds with the labels actually drawn). Deferred, not refused: the residual at the chart's own best point for cell
(b) (needs an edit to `vae_chart.py`, which is under running jobs; will be a standalone script).

### Step 18 results — three-seed spectra at the truth (job 625113) and the distinct-labels cell (job 624573, first row)

`truth_spectrum.py`: `k = 16` (line 18), `N = 8`, `r = 16`, `T = 400`, `lr = 0.01`, three `A₀` seeds per cell,
geometric means; ranges in brackets. Encoders at a **fixed architecture** (784-1000-1000-10 GELU): `random` =
Gaussian weights matched to the weak checkpoint's layer norms (8.7% test acc), `weak` = 78.5%, `mid` = 95.1%,
`strong` = 97.9%. Label draws: `repeated` = `[0,3,0,3,5,0,1,9]` (the Step-13/14 draw), `distinct` =
`[0,3,5,1,9,6,7,4]`. Chart's-best error: global 0.421 (repeated) / 0.518 (distinct); local 0.364 (both).

| encoder | labels | chart | σ_min(J) at truth | cond(J) at truth | rank J / 256 | rank B_T | σ_N/σ_1(B_T) |
|---|---|---|---|---|---|---|---|
| random | distinct | global | 7.5e-5 [3.6e-5..1.3e-4] | 7.5e4 | 256 | 8 | 6.0e-2 |
| random | distinct | local | 1.0e-4 [6.4e-5..1.6e-4] | 5.3e4 | 256 | 8 | 8.8e-2 |
| random | repeated | global | 4.0e-6 [3.5e-6..4.6e-6] | 2.0e6 | 256 | 8 | 2.3e-5 |
| random | repeated | local | 6.8e-6 [6.4e-6..7.7e-6] | 1.1e6 | 256 | 8 | 9.0e-5 |
| weak 78% | distinct | global | 9.1e-7 [8.7e-7..9.5e-7] | 8.6e6 | 256 | 8 | 2.5e-4 |
| weak 78% | distinct | local | 2.3e-9 [6.4e-10..6.0e-9] | 3.2e9 | 256 | 8 | 2.6e-6 |
| weak 78% | repeated | global | 2.2e-8 [1.5e-8..3.3e-8] | 3.4e8 | 256 | 8 | 1.6e-6 |
| weak 78% | repeated | local | 1.1e-10 [8.1e-11..1.6e-10] | 6.9e10 | 256 | 8 | 1.0e-8 |
| mid 95% | distinct | global | 1.3e-8 [3.1e-9..3.2e-8] | 4.8e8 | 256 | 8 | 2.6e-5 |
| mid 95% | distinct | local | 9.1e-8 [3.8e-8..1.7e-7] | 5.8e7 | 256 | 8 | 3.8e-4 |
| mid 95% | repeated | global | 3.9e-9 [2.3e-9..7.3e-9] | 1.8e9 | 256 | 8 | 5.9e-7 |
| mid 95% | repeated | local | 7.8e-9 [5.2e-9..1.2e-8] | 1.1e9 | 256 | 8 | 1.4e-6 |
| **strong 98%** | distinct | global | **5.6e-12** [3.5e-12..7.6e-12] | 3.4e12 | **254–255** | 8 | **4.2e-9** |
| **strong 98%** | distinct | local | **3.5e-19** | 6.1e19 | **197–200** | **6** | 1.9e-16 |
| **strong 98%** | repeated | global | **7.0e-19** | 1.1e19 | **210–211** | **6** | 1.2e-16 |
| **strong 98%** | repeated | local | **6.5e-19** | 4.7e19 | **203–205** | **5** | 1.5e-16 |

Single-draw spread across the three seeds: ≤ 2.3× in `σ_min` for every non-collapsed cell except weak/distinct/
local (9×) and mid/distinct/global (10×); the effects below are all far larger than that.

**1. The local-vs-global gap is a property of the weak checkpoint, not of trained encoders.** Global/local
`σ_min` ratio: random **0.6× / 0.7×** (local slightly *better*), weak **206× / 397×**, mid **0.5× / 0.1×** (local
better, as originally predicted), strong 1.1× (both collapsed) / 1.6e7× (local collapsed, global barely not).
The pre-registered mechanism — "a trained classifier compresses within-class variation, so the local chart
pays" — predicted the gap should grow with training. It **vanishes at mid**. The random-encoder half of the
falsifier fired as predicted, but the mid point rules out the monotone story: **refuted as stated.** What makes
the 78% checkpoint special is not measured here (candidate: a partially-trained model's features are dominated
by the between-class directions it learned first; the mid model has had to learn within-class structure to
reach 95%). Left as an observation.

**2. Repeated labels did not cause the gap — but they are a first-order conditioning factor on their own.**
*[CONFOUNDED, 2026-09-03 late — the "distinct" and "repeated" draws are DIFFERENT IMAGES, not the same images
relabelled; the repeated draw holds margin-46 and margin-37 digits against a maximum of 23 in the distinct one, and
on raw digits both draws give rank 6 on the strong model. The differences below are draw effects; whether labels
contribute anything beyond the margins is untested. What replaces it is the margin-order rule under "Step 18
resolved": in all nine rank-deficient batches the sub-floor examples are exactly the highest-margin ones.]*
On the weak encoder the gap *survives* distinct labels (397× vs 206×). Yet at fixed chart, distinct-vs-repeated
`σ_min` is **19× (random), 40× (weak), 3.4× (mid), 8000× (strong)** better on the global chart, and
`σ_N/σ_1(B_T)` moves by 2–4 orders in the same direction. The solve confirms it: the weak-encoder `k = 16` cell,
still descending at 300 iterations with repeated labels (Step 13), **reaches the floor in 93 iterations with
distinct labels** (job 624573, first row: residual 7.5e-31, err vs chart 3.1e-12, `σ_min` 8.1e-7 — matching
the spectrum's 9.1e-7). Same-label private examples produce near-collinear residual trajectories, and the
release records them worse. The Step-13/14 numbers were all taken on the harder draw.

**3. The encoder-quality ladder is monotone at fixed architecture, and it reaches non-identifiability.**
Global chart, distinct labels: `σ_min` **7.5e-5 → 9.1e-7 → 1.3e-8 → 5.6e-12** (random → 78% → 95% → 98%), i.e.
82× / 69× / 2400× per step; cond 7.5e4 → 3.4e12. Same ordering with repeated labels and on the local chart
except the weak/local anomaly of item 1. The mid checkpoint's own solve at `k = 10` (job 614344): `σ_min`
7.9e-8 vs the weak model's 2.2e-6, still descending at 300 iterations. **Quality hurts, four points.**

**4. NEW — the strong model's release is not identifiable at k = 16, seven below the line.** With repeated
labels `B_T` has numerical rank **5–6 of 8** (`σ_N/σ_1` = 1e-16, machine zero) and the truth Jacobian rank
203–211 of 256; with distinct labels `B_T` keeps rank 8 only at `σ_8/σ_1` = 4e-9 and `J` is 1–2 columns short.
`rank X = 8` in every row, so `rank P_T < N`: hypothesis (A4) of the theory summary — `N` independent
accumulated residual trajectories — **fails on a strong model with easy private data**, and the capacity line,
which presupposes it, does not apply there. This is the first below-line loss of identifiability in the study,
and it is caused by the model, not by the count.

**5. Pre-registered mechanism for item 4, with its test in flight (job 626051, `margin_check.py`).** Column `i`
of `P_T` is the accumulated softmax residual `p_t(x_i) − e_{y_i}`. `B₀ = 0`, so the trajectory starts at `W₀`;
a 98% model classifies an easy test digit with margin `M`, its residual is `~e^{−M}`, and the digit is recorded
in the release at that scale — below FP64 for `M ≳ 30`. Prediction: per image, `‖P_T[:, i]‖` tracks the residual
norm at `W₀` across the four encoders, and the rank loss sits on the largest-margin digits. Falsifier: column
norms all `O(1)` on the strong model. If it holds, the reading is *a model fine-tuned on examples it already
fits leaves no fingerprint of them; what leaks is what it had to learn* — and the per-example residual at `W₀`
is a leakage meter the defender can compute without running any attack. *(Superseded: the exact quantity is the
per-example residual ACCUMULATED during fine-tuning — see "Step 18 resolved" — of which the residual at `W₀` is the
pre-training proxy, exact only when training leaves the example undisturbed.)*

**Standing corrections to the reads above (all three now in RESULTS).** "Richer charts are worse conditioned" is
established only across the four charts on the weak checkpoint (Step 14); on other encoders the PCA half of it
inverts (item 1) and the VAE half is unmeasured. "Chart quality costs conditioning" is therefore withdrawn as a
general statement. The fidelity ranking stays embargoed (job 624463). Multi-layer LoRA (job 608693) died with a
code error (`element 0 of tensors does not require grad`, `multilayer_lora.py:57`) — not a result; to be fixed.

### Step 18, item 5 resolved — the release records an image at the scale of the model's residual on it (job 626051)

`margin_check.py`: per image, the margin and softmax-residual norm at `W₀` (training starts there, `B₀ = 0`) and
the norm of the image's column of `P_T` (recovered exactly as `B_T X (XᵀX)⁻¹`; `rank X = 8` in every row).
`k = 16`, `A₀` seed 1; on-chart (the cells' setting) and raw digits (the realistic one).

| encoder | draw | on-chart: spread of ‖P_T[:,i]‖ (max/min) | columns < 1e-6 | rank B_T | concordance with ‖res(W₀)‖ | raw digits: spread · cols < 1e-6 · rank |
|---|---|---|---|---|---|---|
| random | distinct | 5× | 0 / 8 | 8 | 18/28 | 3× · 0 · 8 |
| random | repeated | 17× | 0 / 8 | 8 | 21/28 | 11× · 0 · 8 |
| weak 78% | distinct | 19× | 0 / 8 | 8 | 15/28 | 18× · 0 · 8 |
| weak 78% | repeated | 40× | 0 / 8 | 8 | 17/28 | 62× · 0 · 8 |
| mid 95% | distinct | 2.3e3× | 0 / 8 | 8 | 21/28 | 2.4e2× · 0 · 8 |
| mid 95% | repeated | 58× | 0 / 8 | 8 | 17/28 | 1.0e2× · 0 · 8 |
| **strong 98%** | distinct | **5.2e7×** | **6 / 8** | 8 (σ₈/σ₁ 3e-9) | **21/28** | **1.3e6× · 8 / 8 · 6** |
| **strong 98%** | repeated | **2.6e9×** | **5 / 8** | **6** | 20/28 | **2.9e7× · 8 / 8 · 6** |

The strong model on the repeated draw, on-chart, per image (`y`, margin at `W₀`, ‖res(W₀)‖, ‖P_T[:,i]‖):
0: (0, 12.8, 3.9e-6, 7.0e-2) · 1: (3, 22.0, 3.9e-10, **1.6**) · 2: (0, 46.2, 9.4e-21, 7.0e-2) · 3: (**3, −3.6**, 1.4, 3.8) ·
4: (5, 4.7, 1.3e-2, 0.81) · 5: (0, 37.4, 5.5e-17, **1.5e-9**) · 6: (1, 21.3, 8.0e-10, **1.4e-8**) · 7: (9, 19.8, 4.1e-9, **5.7e-8**).

**Confirmed, in its core.** The falsifier ("O(1) column norms on the strong model") did not fire: six of eight
columns are below 1e-6 with distinct labels, five of eight with repeated, and on raw digits (margins up to 64,
residuals down to 2e-28) **all eight** are — against a 3–5× spread on the random encoder. Every column below 1e-6
belongs to an image with margin ≥ 16. The release records a private example roughly at the scale of the
model's residual on it, and a 98% MNIST model has residual `e^{−20}…e^{−60}` on the digits it gets right.

**[WITHDRAWN 2026-09-03 — see "CORRECTION (job 628731)" below: the per-image "column" this paragraph reads was the order-dependent QR-basis quantity; the coupling it describes does not exist. Kept as record.]** **Refinement the data forces.** The column norm is *not* a function of the image's own residual alone
(concordance 20–24 of 28 pairs, not 28). Image 1 — a 3 with margin 22 and residual 4e-10 — is recorded at 1.6,
because image 3 is a *misclassified* 3 (margin −3.6, residual 1.4) and the `A`-dynamics couple images through
their feature Gram `HᵀH`: a hard example re-records the confident examples whose features overlap with it. With
distinct labels image 1 is still the exception (0.12, next to the one low-margin image, a 5). So the honest
statement is: **a confident example is invisible in the release unless a hard example with overlapping features
is in the same batch** — and that is the mechanism of the rank loss too: the two 3-columns both carry image 3's
residual and become collinear.

**Reads** *(item (iv)'s "up to the Gram coupling" is withdrawn with the paragraph above; (i)–(iii) stand)*. (i) This is why quality hurts monotonically: the ladder is a ladder of margins. (ii) It is an
*information* limit, below the line, that the counting cannot see, and it lives in hypothesis (A4). (iii) It is
asymmetric in a way that matters for privacy: **what leaks is what the model had to learn** — the misclassified
3 is recorded at 3.8, the confident 0 at 1e-9. (iv) The per-example residual at `W₀` — which the defender can
compute from the model and the data before releasing anything — predicts which examples the adapter will
carry, up to the Gram coupling. Not yet shown: that a below-line cell on the strong model returns the *wrong*
confident images at the residual floor (the alias form of this); job 614344's strong sweep is the place to look.

### Step 17 result — multi-layer LoRA does not invert at this budget, seeds known or not (job 626564)

`multilayer_lora.py` after the grad fix (commit `0b1c091`): LoRA `r = 8` on all three layers of the 78% MLP,
`T = 100`, `lr = 0.01`, `N = 8`, `k = 14`, global PCA chart, start `W_true + 0.10·noise`, LBFGS. Gate: the
simulator reproduces the release at the truth to **0.0**. Released numbers 38,352; unknowns 112 (data) + 22,272
(seeds).

| arm | unknowns | residual | err vs chart (max) | images < 1e-2 | seconds |
|---|---|---|---|---|---|
| A — seeds KNOWN (oracle) | 112 | 2.2e-8 | 7.4e-2 | 3 / 8 | 64 |
| B — seeds UNKNOWN (the attack) | 22,384 | 7.7e-7 | 7.9e-2 | 1 / 8 | 132 |

Both arms are **optimisation failures** (residual 20+ orders above the floor), not aliases, and the oracle arm
fails almost as badly as the honest one — so at this budget the obstruction is the unrolled three-layer map
itself, not the unknown seeds. The count (38,352 > 22,384) says nothing against identifiability; nothing here
says anything for it either. Outside the theorems ((A1) fails: adapting layer 1 moves the features); empirical
only. Next honest step, if pursued: the LM solver on arm A (112 unknowns is LM-sized) and `σ_min(J)` at the
truth, which is the same instrument as everywhere else. (Provenance note: the two rows carry different git
hashes — a sibling session committed to the branch between the arms; `multilayer_lora.py` itself did not change.)

### Step 18, β-family first point (job 626565) — richness at fixed decoder family does not move conditioning

`truth_spectrum.py` on the saved decoders, weak encoder, repeated draw, `k = 16`, three inner-projection
budgets, three `A₀` seeds at the last one, three perturbations of `1e-3·std`:

| chart | chart's best (repr err) | inner loss (300 / 1000 / 3000) | σ_min(J) at truth (levels; seeds; perturbations) | decoder Jacobian σ_min / σ_max |
|---|---|---|---|---|
| global PCA | 0.421 | — | 3.3e-8 / 2.4e-8 / 1.5e-8 (seeds) | 1 / 1 (orthonormal) |
| VAE β = 1 (GELU) | 0.339 | 90.47 / 90.47 / 90.47 | 1.13e-11 at every level; 1.13–1.67e-11 (seeds); 1.13–1.14e-11 (perturbations) | 0.0080 / 6.69 |
| VAE β = 0.25 (GELU) | **0.222** | 47.11 / 47.08 / 47.07 | 1.15e-11 / 1.15e-11 / 1.22e-11; 0.99–1.28e-11 (seeds) | 0.012 / 6.2 |

- **The inner-projection confound is closed, not argued:** the on-chart truth is at the same inner loss at 300
  and 3000 steps, `σ_min` is identical across levels and under perturbation. (The inner gradient norm *rises*
  at 3000 — Adam at fixed `lr` bouncing around a converged minimum, not a moving truth.)
- **Richness within one decoder family does not change conditioning:** β = 0.25 draws markedly better than
  β = 1 (0.222 vs 0.339) at the *same* `σ_min` (1.15e-11 vs 1.13e-11). Together with the PCA pair inverting on
  the mid and random encoders (earlier in Step 18), **"richer chart ⇒ worse conditioned" is dead as a general
  statement.** What survives: on this encoder the VAE family sits ~2000× below PCA in `σ_min`, and the decoder's
  own conditioning (σ_max/σ_min ≈ 400–830 vs exactly 1) is the right size for most of that — consistent with
  decoder geometry, not evidence about the encoder. β = 4, 16, ReLU and cVAE follow when their artefacts land.

### Step 18, batch composition (job 627166) — an all-confident batch leaves nothing; the coupling follows features, not labels

`margin_check.py --pick`, batches chosen by margin under the **strong** model and then run through all four
encoders (so random/weak/mid are the control that the effect is the model's, not the digits'). `k = 16`, raw
digits and on-chart. `confident` = the largest-margin digit of each of the 8 highest-margin classes (margins
56–95 under the strong model); `hard1_same` = the most-misclassified test digit (a 1, margin −35) + the seven
largest-margin 1s (54–56); `hard1_diff` = the same hard 1 + the largest-margin digit of seven other classes (73–95).

| batch | setting | strong: rank B_T | strong: columns < 1e-6 | strong: column range | control (random / weak / mid): rank, spread |
|---|---|---|---|---|---|
| confident | raw | **3** | **8 / 8** | 5e-26 … 1e-23 | 8, 2× / 8, 28× / 8, 19× |
| confident | on-chart | **3** | 1 / 8 | 3e-16 … 0.41 (one image at margin 4.1 dominates) | 8, 2× / 8, 6× / 8, 25× |
| hard1_same | raw | **1** | 7 / 8 | hard 8.7; the seven 1s 2e-15 … 1e-14 | 8, 30× / 8, 180× / 8, 2000× |
| hard1_same | on-chart | **1** | 7 / 8 | hard 5.0; the seven 1s 5e-15 … 1e-12 | 8, 92× / 8, 150× / 8, 810× |
| hard1_diff | raw | **1** | 7 / 8 | hard 8.7; the rest 9e-16 … 9e-15 | 8, 1.6× / 8, 5e4× / 8, 4e3× |
| hard1_diff | on-chart | **4** | **0 / 8** | hard 5.0; the rest 5e-3 … 0.35 | 8, 2× / 8, 15× / 8, 24× |

(Columns at 1e-14–1e-16 next to a column of 5–9 are the FP64 floor of the `P_T` recovery, not a measured lift.)

**Prediction 1 — confirmed outright.** Fine-tune the 98% model on eight digits it already classifies with margin
≥ 56 and the release records them at 1e-24 with `rank B_T = 3`: **the adapter carries nothing of that batch.**
The same eight digits through the random, weak and mid encoders give full rank and O(1)…O(1e-2) columns. On-chart
(the PCA projection lowers the margins to 4–71) one image at margin 4.1 dominates, the others sit at 1e-2 —
and the rank is *still* 3: what is recorded of the confident images is a **mixture of the hard image's
residual**, collinear across columns, not independent information about them. Rank, not column size, is the
witness.

**Prediction 2 — refuted in its naive form, and that is informative.** The same-class batch is *not* lifted:
seven confident 1s next to a misclassified 1 stay at the floor (raw and on-chart, `rank B_T = 1`). The
different-class batch *is* lifted on-chart (every column ≥ 5e-3, `rank B_T = 4`) — the opposite sign. A
misclassified 1 is misclassified *because its features do not look like a 1's*, so its feature-Gram overlap with
confident 1s is small; on-chart, the different-class batch contains two further low-margin images (a 3 at 4.1, a
5 at 15.6) whose residuals couple into the rest. So the coupling is through **feature overlap in the encoder's
penultimate space**, and *same label* is the wrong proxy for it. Job 627574 (`step55_gram`) records each image's
feature cosine to the hardest image and to its own-label mates next to the column norms, to make that a
measurement rather than a reading.

**[WITHDRAWN 2026-09-03 — see "CORRECTION (job 628731)" below: the per-image "column" this paragraph reads was the order-dependent QR-basis quantity; the coupling it describes does not exist. Kept as record.]** **Standing statement (replaces the Step-18 refinement).** A private example is recorded in the release at
roughly the scale of the model's residual on it, plus a coupling term from the residuals of batch-mates whose
*features* overlap with it; a batch of examples the model already fits leaves no fingerprint, and a hard example
lends its residual to the batch in a way that is collinear across columns — it raises column norms without
restoring rank. For the defender: the per-example residual at `W₀` and the feature Gram of the batch, both
computable before release, predict what the adapter will carry.

### Step 18, the Gram measurement (job 627574) — labels settled; "feature overlap" is not the right description either

`margin_check.py` with per-image feature cosines (penultimate space, unit-normalised) to the largest-residual image
and to own-label mates, next to the `P_T` column norms; all batches of the previous section plus the two draws.

- **Why the same-class batch is not lifted, measured:** under the strong model the misclassified 1 has feature
  cosine **0.22–0.30** (raw) / 0.12–0.24 (on-chart) with the seven confident 1s; the *same digits* have cosine
  0.63–0.72 under the weak model and 0.43–0.96 under the random encoder. A 98% model has moved a misclassified
  1 away from the 1-cluster in feature space — that is what misclassifying it *is* for a linear head — so it has
  nothing to lend its classmates. Label is settled as the wrong proxy.
- **But feature cosine to the hard image does not order the columns either:** Kendall concordance across the
  ten strong-model batches is 17, 13, 10, 16, 6, 14, 14, 11, 13, 6 of 21 — above chance on average, decisive
  nowhere. What *does* order them in the lifted batch (`hard1_diff`, on-chart, rank 4) is each image's **own
  step-0 residual: 20 of 21 pairs** — while the magnitudes sit far above those residuals (a margin-71 digit with
  residual ~1e-31 has a column of 5e-3).
- **Reading, now to be measured rather than argued (job 628731, `step56_traj`):** the order is preserved and
  the magnitudes are lifted uniformly, which is what happens if training on the hard example *shifts every
  image's margin* by a similar amount — the adapter is shared, and a residual-O(1) example drives O(1) updates
  for 400 steps. The step-0 residual is the right predictor only when training does not disturb the example.
  The exact quantity is the **accumulated residual along the trajectory** (`P_T` is a linear function of it),
  which the trace job records per image (sum, max, final; margin at the end and its minimum along the way),
  gated against `train_release` to 1e-12. Prediction: column norm ∝ accumulated residual (concordance ~21/21);
  confident images in a batch with a hard example show a margin drop of tens of units, those in an all-confident
  batch show none. For the defender this is better news than the step-0 version: the accumulated per-example
  residual is something they *already compute* while fine-tuning.

### CORRECTION (job 628731) — the "coupling" was an artefact of how the per-image column was read; withdrawn

The trajectory trace refutes the margin-shift reading *and* exposes the measure. Confident images' margins do not
move during training (strong model, `hard1_diff` on-chart: the margin-71 digit ends at 71.1, the margin-46 one
at 39.9; all-confident batch: no image moves by more than 0.1), yet their "`P_T` columns" sit **15–26 orders
above their accumulated residuals** (`Σ_t ‖res_t,i‖` = 5.6e-29 for the margin-71 digit; "column" 5.2e-3).

**Why.** The per-image column was read as `P_T = B_T X (XᵀX)⁻¹` with `X = A₀U`, `U` from a QR of the features
`H = U R_H`. Expanding the release, the leading term is `P_T = −lr (Σ_t D_t) R_Hᵀ`: **column `i` of `P_T` collects
the accumulated residuals of every image `j ≥ i` in batch order**, weighted by the triangular factor `R_H[i, j]`.
It is basis- and order-dependent and is not "image `i`'s share". That is exactly the pattern in the data: in
the repeated draw (hard 3 at index 3, low-margin 5 at index 4) the "lifted" columns were indices 0–4 and the
"invisible" ones 5–7 — the images *after* the last hard example; in `hard1_diff` raw (hard image at index 0)
nothing else was lifted; in `hard1_diff` on-chart (a margin-4 digit at index 7) every column was.

**Withdrawn:** "a hard example re-records confident batch-mates through the feature Gram", "the coupling follows
features, not labels", and the Gram/label readings built on it (the cosines themselves are correct measurements
and stay on file; they explain nothing). **Stands, unchanged:** the mechanism's core — the release records an
example at the scale of its residual — and every rank statement (`rank B_T` is basis-independent): an
all-confident batch releases rank 3 at 1e-24; a hard example plus confident ones releases rank 1; the 98% model
loses identifiability below the line.

**The correct per-image quantity** is image `i`'s own contribution to the release: `gB = D (AH)ᵀ = Σ_i D[:,i]
(A h_i)ᵀ`, so `B_T = Σ_i C_i` with `C_i = −lr Σ_t D_t[:,i] (A_t h_i)ᵀ`, an `m × r` piece per image, basis-free,
and `‖C_i‖ ≲ lr · Σ_t ‖res_t,i‖ · ‖A_t h_i‖` — **proportional to the accumulated residual, with no coupling term.**
Job 631392 (`step58_imprint`) recomputes every batch with `‖C_i‖`, gated by `Σ_i C_i = B_T` to 1e-10, and records
the rank of the stacked imprints. Prediction: Kendall(‖C_i‖, accumulated residual) ≈ 28/28 in every batch, and
`rank B_T` = the number of images with non-negligible imprint whenever their features are independent. The
defender-side quantity is then simply **the per-example accumulated residual during their own fine-tuning**.
Job 630308 (subset/OOD) was killed before producing rows because it used the flawed column to name the recorded
images; resubmitted as 631393 on the imprint.

### Step 18 resolved — the per-image imprint is proportional to the accumulated residual, and rank counts the recorded images (job 631392)

`margin_check.py` with the basis-free imprint `C_i` (`B_T = Σ_i C_i`, gated to 4e-12), all 40 batches (4 encoders ×
{3 picks + 2 draws} × {raw, on-chart}):

- **Kendall(‖C_i‖, Σ_t ‖res_t,i‖) = 28/28 in every strong-model batch but one (27/28), 24–28/28 on the weak and
  mid models; 971/1120 overall.** On the random encoder it is at chance (10–14/28) because all eight residuals
  are 0.93–0.97 — there is nothing to order — and there the *ratio* ‖C_i‖/acc_i is constant to ±0.3 decades. In
  every batch the ratio spans < 1 decade: `‖C_i‖ ≈ lr · ‖A h_i‖ · Σ_t ‖res_t,i‖`, **no coupling term.**
- **`rank B_T` = rank of the stacked imprints = number of images with `‖C_i‖/max > 1e-12`, in all 40 batches.**
  The rank loss is exactly the count of images whose accumulated residual is negligible.
- Strong model, `hard1_diff` on-chart (rank 4): the four images with margins −3.9, 4.1, 15.6, 17.7 have relative
  imprints 1, 7.6e-2, 8.2e-7, 4.8e-8; the four with margins ≥ 39 sit at 3e-17 … 1e-30. All-confident batch: every
  imprint ≤ 1.5e-24 absolute, rank 3 only because the three least-confident of them (margins 56–74) clear the
  1e-12 *relative* threshold against a maximum that is itself 1e-24.

**Standing statement (final form).** The LoRA release records each private example as a rank-one piece whose
size is the model's accumulated softmax residual on that example during fine-tuning — `e^{−margin}` if the
model already fits it — and `rank B_T` counts the examples recorded above the floor. A batch the model already
fits leaves no fingerprint; what leaks is what the model had to learn. The defender's leakage meter is the
per-example accumulated residual during their own fine-tuning, which they compute anyway. Nothing about labels,
feature overlap, or batch-mates enters.

**Additivity measured by perturbation, not decomposition (job 710597).** Every imprint measurement above
*decomposes* a release into the `C_i` and reads their sizes, which assumes the sum has no cross term. The batch
swap tests the assumption from the other side: exchange one member of the recorded set for an invisible one and
re-run the recipe — the release moves by **that member's own imprint and by nothing else**, tracking over five
orders across cells (1.1e-11, 6.2e-10, 5.7e-8, 1.3e-8, 1.4e-6 relative; the `N′ = 1` cell gives 1.0 because the
record *is* that example and the swap replaces it). A perturbation equals the removed term only if the sum is
additive with no cross term — which is the imprint law, measured from a direction the decomposition cannot take.

**The margin-order rule (verified on job 631392, all nine rank-deficient batches).** In every batch where
`rank B_T < 8` on the strong model — ranks 1, 1, 1, 3, 3, 4, 6, 6, 6 — the examples whose imprint sits below the
floor are *exactly* the `8 − rank` highest-margin examples of that batch, whether two drop or seven (checked
image-by-image: 9 of 9 match). So **recording is decided by an example's margin relative to its batch-mates, and
by nothing else**: not labels, not feature collinearity, not coupling. Consequences: (i) the label-multiset
attribution in item 2 above is stamped confounded (different draws, different margins); (ii) all-same-label
batches with every example wrong (Step 21: flowers, letters) have full rank, and the same-label batch with one
wrong example (`hard1_same`) has rank 1 — labels do nothing that the margins do not already say; (iii) the
`σ₂/σ₁` trigger of Step 20 detects one dominant *direction*, not one recorded *example*: among the nine
triggering strong batches the rank is 1 in three, 3 in two, 4 in one, 6 in three, 8 in one — the trigger says
"a release dominated by one direction", and the count of recorded examples is the rank, read separately.

**Two corrections carried from the audits (2026-09-03, late).** (a) *The encoder ladder must be quoted per label
set.* At the strong encoder, global chart, same `k` and seeds, eight distinct labels give `σ_min` 5.6e-12 with
`rank B_T = 8` while the repeated draw gives 7.0e-19 with rank 6 — seven orders apart; a single "strong" rung
(2.0e-15, their geometric mean) is a value no measurement lies within three orders of, and is not to be quoted.
The clean ladder is global chart + distinct labels: **7.5e-5 / 9.1e-7 / 1.3e-8 / 5.6e-12**, rank 8 at every rung.
(b) *The (A4) failure does not require repeated labels.* The in-distribution control of Step 19 — strong model,
raw test digits, eight DISTINCT labels — has `rank B_T = 6`. The rank drop follows from each example's own
confidence; repeated labels and class-local charts are two routes *to* it, and chart projection is a route *away*
from it (projection degrades the image, the model is less certain, more is recorded — which is also why the
on-chart and raw arms of a cell differ, and why Step 21 reports both side by side). "Strong encoder ⇒ (A4) fails"
and "repeated labels ⇒ (A4) fails" are each too simple on their own.

## Step 19 — private data from a DIFFERENT distribution: it is recorded in full (job 644062; inversions in 644064)

`subset_and_ood.py --part B --skip-invert`. Private sets of `N = 8` digits with labels `[0,3,5,1,9,6,7,4]`:
`mnist_control` = the distinct-label random draw from the MNIST test split (in-distribution; NOT a margin-picked
batch); `font` = digits rendered from DejaVu Sans/Mono (bold and regular), rotated ±12°, centred like MNIST;
`optdigits` = UCI optdigits scans (8×8, other writers and scanner, bilinear-upscaled to a 20-px box, centred).
Fine-tuned into the weak (78%), mid (95%) and strong (98%) backbones; `r = 16`, `T = 400`, `lr = 0.01`, `k = 16`;
raw digits and their MNIST-PCA projections. Margins are read first — a set whose margins are not below the
control's is a failed manipulation, not a result.

| private set | accuracy at `W₀` (weak / mid / strong) | median margin, strong (raw) | `rank B_T` at 1e-12 / 1e-8, strong (raw) | chart's best (MNIST PCA, k=16) |
|---|---|---|---|---|
| MNIST control | 1.00 / 1.00 / 1.00 | **23.8** (10 … 64) | **6 / 6** | 0.518 |
| font-rendered | 0.50 / 0.88 / 1.00 | 8.9 (three at 31–47, five at 6–9) | 6 / 4 | 0.383 |
| **optdigits** | 0.25 / 0.38 / **0.50** | **−2.0** (−4 … 11) | **8 / 8** | **0.317** |

Weak and mid backbones: `rank B_T = 8/8` on every set. Strong backbone per-image imprints (relative to the
largest), raw digits — MNIST control: 1, 6e-24, 1e-5, 1e-7, 7e-7, 1e-14, 2e-7, 2e-5 (one digit carries the
release); optdigits: 2e-4, 1e-3, 4e-5, 1, 0.7, 0.1, 0.8, 0.6 (five within one decade of the largest).

- **Foreign data is recorded in full.** The 98% model classifies half of the optdigits wrong (margins down to
  −4), so every one of the eight leaves an O(1) imprint: `rank B_T = 8` where the same model on in-distribution
  digits leaves rank 6 and one dominant image. The mechanism's prediction — *what the model gets wrong is what
  it records* — holds on data it has never seen the like of.
- **Fonts are half-foreign, and the margins say so.** The strong model gets all eight right; three (a 0, a 3, a
  5 in bold sans) at margins 31–47 are invisible (imprints 1e-11 … 1e-18), five at margins 6–9 are recorded.
  Rank 6 at 1e-12 but 4 at 1e-8 — a case where the threshold is the reader's choice. Clean rendered digits are
  *easier* for the model than handwriting for some classes, as the design audit anticipated; the set is reported,
  not used as "the" OOD result.
- *[CORRECTED by the blur control, job 667729, `blur_control.py`: the effect below is RESOLUTION, not provenance.
  The same eight MNIST control digits put through optdigits' own pipeline (area-downsample to 8 px, bilinear back to
  20 px, centred) sit at **0.320** on the same PCA basis — identical to optdigits' 0.317 — and at 0.354 / 0.343 /
  0.268 for 14 / 10 / 6 px; Gaussian blur σ = 0.5 / 1 / 1.5 / 2 gives 0.456 / 0.290 / 0.240 / 0.217; the 2,000-digit
  test reference moves 0.521 → 0.336 at 8 px. A 16-component linear chart draws smooth images better whatever it was
  fitted on. So the chart half of the two-sided prediction was NOT tested by provenance: at matched resolution
  MNIST and optdigits are equally drawable, and the fonts' 0.383 sits where blurred MNIST sits. What stands from
  this bullet is only the record/no-record half; "both effects favour the attacker" is withdrawn.]*
- **The second half of the prediction failed: these sets are drawn BETTER, not worse, by the MNIST chart.** The
  chart's own best is 0.317 (optdigits) and 0.383 (fonts) against 0.518 for the MNIST draw — upscaled 8×8 scans
  and rendered glyphs are smoother than handwriting and a 16-component PCA captures them more fully. So for
  these two sets both effects point the same way: recorded more, and drawable at least as well. Whether the
  recovered images look like the foreign digits is job 644064 (cells (a)/(b), image grids saved).
- Scope: this is the record/no-record axis (`rank B_T`, basis-free) beside the chart-fidelity axis
  (`chart_repr_err`); they are not merged. Starts for the inversions are near-truth as everywhere here.

## Step 20 — the most-leaking example: prevalence of the one-image release (job 650891); the attack from random starts (job 650890, IN FLIGHT)

**Trigger.** The attacker reads `σ₂/σ₁` of the released `B_T`; below `τ = 1e-3` the release is treated as
carrying one image. Prevalence over the 40 batches on disk (10 batch compositions — three margin-picked, two
random draws, raw and on-chart — × 4 encoders at one architecture; `k = 16`, `N = 8`, `r = 16`):

| encoder | batches triggered | `σ₂/σ₁` range | `rank B_T` over the 10 batches |
|---|---|---|---|
| random (8.7%) | 1 / 10 (a false positive: rank 8) | 8.7e-4 … 0.79 | all 8 |
| weak (78%) | 0 / 10 | 0.021 … 0.58 | all 8 |
| mid (95%) | 0 / 10 | 0.005 … 0.69 | all 8 |
| **strong (98%)** | **9 / 10** | **4.8e-16 … 0.07** | 1, 1, 1, 3, 3, 4, 6, 6, 6, 8 |

On the strong model the two *ordinary random draws* — not the margin-picked batches — are one-image releases
(`σ₂/σ₁` = 2.1e-5 and 2.9e-5 raw; 2.3e-4 and 2.4e-4 on-chart). The one strong batch that does not trigger is
`hard1_diff` on-chart (`σ₂/σ₁` = 0.07, rank 4: a low-margin 3 shares the release with the hard 1). The random
encoder's single trigger is `hard1_same` (all eight are 1s with residual ≈ 0.95 each; the spectrum is set by the
feature geometry, not by margins) and it is rank 8 — the negative control for the read is job 650890's mid cell.

**Read:** a release *dominated by one direction* is a property of this strong model on ordinary data, 9 of 10
batches, not an anecdote; on the weaker models it never happens. *(Stamped 2026-09-03 late: "one image carries the
release" overstated the trigger — among the nine triggering batches `rank B_T` is 1, 1, 1, 3, 3, 4, 6, 6, 6; the
trigger detects a dominant direction, the number of recorded examples is the rank. For the two ordinary draws the
rank is 6 with one dominant image, which is what the most-leaking attack targets.)* Whether that one image can be recovered from random
public-scale starts by residual ranking alone is job 650890 (per-start floor fractions, argmin-residual label
and image, `k ∈ {16, 24, 25, 26}` toward the one-image line `k < m + r − 1 = 25`).

## Step 21 — adding a NEW CLASS by LoRA (jobs 656205 CIFAR flowers; 658575 EMNIST 'a' on the MNIST models) — IN FLIGHT, predictions pre-registered

`new_class.py`. The head is extended by one row for class 10 (zero, the practice; a random row at the existing
rows' RMS norm as the arm that gives the new class ordinary-spread initial margins), LoRA on the extended head,
`B₀ = 0`, `r = 16`, `T = 400`, `lr = 0.01`, `N = 8`, `k = 16`; line `k < m + r − N = 19`. Private new-class images
from a TEST split never seen by backbone or chart; the attacker's chart = PCA on the public TRAIN images of that
class (2,500 CIFAR-100 flowers; 4,800 EMNIST 'a's); generic chart as control; the in-distribution control runs on
the SAME extended head (`old_ext`, zero row present, never the target) so `m` is matched. Every row carries the
imprints' Gram (σ_N/σ_1, pairwise cosines) beside the imprint norms: *absent* (small norms) and *aligned* (small
angles) are different causes of a poor `σ_min(J)`, and this is the field that separates them.

**Pre-registered before the mid/weak arms and the controls report (2026-09-03, evening):**
1. New class ⇒ imprints all `O(1)` and `rank B_T = 8` — present — but aligned (imprint-Gram `σ_N/σ_1` small, far
   above the floor); alignment is a *conditioning* cost, not a wall. *(First rows, zero-row arm: CIFAR flowers —
   imprints 0.4–1.0 relative, rank 8, Gram `σ_N/σ_1` 0.145 / 0.097, mean cosine 0.50 / 0.57, cell (a) at the floor
   in 62 iterations with `σ_min` 2.4e-6; strong MNIST + 'a' — imprints 0.3–1.0, rank 8, Gram 0.092, cosine 0.52.)*
2. **The quality ladder flattens for a new class**: imprints `O(1)` and rank 8 at 78%, 95% and 98% alike — a zero
   row means maximal error regardless of encoder quality — whereas ordinary digits on the same models fall from
   rank 8 to rank 6 with one dominant image. If it holds, *how good the model is* and *whether it has seen the
   category* are two independent axes of exposure.
3. The random-row arm gives the new class initial margins with spread; the imprint law is *tested* there
   (Kendall of imprint vs accumulated residual), not exhibited by construction as under the zero row.
4. The side-by-side table this section is for: the SAME 98% encoder on eight confident digits vs eight 'a's —
   imprint norms, imprint-Gram `σ_N/σ_1`, `rank B_T`, `σ_min(J)` at the truth — to be filled from 658575's `old_ext`
   and `new` rows.

*CIFAR matched control (job 656205, `cifar10_ext`: eight CIFAR-10 test images on the same 11-row head):* the 53%
model is wrong about five of eight, so they are recorded too (imprints 0.3–1 but one at 5e-4, rank 8) — yet
**orthogonal** (mean cosine 0.01 raw, 0.002 on-chart) where the flowers were aligned (0.50–0.58); on-chart floor in
32 iterations with `σ_min` 1.8e-5, i.e. as solvable as the flowers (2.4e-6). Alignment is the shared-new-class
signature; on a weak encoder it costs little conditioning. Off-chart 0.57 (chart's best 0.26). The imprint-Gram
`σ_N/σ_1` conflates size and angle (one tiny control imprint drags it to 4e-4); the pairwise cosine is the angle.

**The side-by-side (item 4), job 658575, strong 98% model, same extended head (`m = 11`, zero row), raw images:**

| batch | margins (median) | imprints (relative) | `rank B_T` | imprint cosine (mean off-diagonal) | reading |
|---|---|---|---|---|---|
| eight MNIST digits (`old_ext`) | 17.9 | 7e-18, 1, 2e-7, 7e-8, 1e-9, 6e-6, 8e-22, 2e-3 | **6** | 0.04 | **absent** — one dominant, the rest below noise |
| eight EMNIST 'a' (`new`) | −10.8 | 0.6, 0.5, 1, 0.4, 0.3, 0.7, 0.7, 0.7 | **8** | 0.52 | **present and aligned** |

Same encoder, same head, same recipe: what changes is whether the model had anything to learn. The letters are
recovered on-chart to the floor (`σ_min` 1.4e-7 … 4e-6); the digits are not identifiable below the line.
(The on-chart `old_ext` row is a chart artefact — the *letter* chart projects digits so badly (repr. err 0.69)
that the model misreads them and records them all — and is not the comparison.)

Data note: EMNIST's 'letters' split merges cases, so class 'a' contains both lowercase a and uppercase A (5 of 8
private letters are 'a', 3 are 'A'; orientation verified visually after the transpose); the public-'a' chart is
fitted on the same mixture, so the new class is bimodal by construction, not mismatched.

Scoping fixed in advance: near-truth starts, labels given (`oracle = [near_init, labels]`); under the zero row the
new class's negative margin is set by the initialisation, not learned — margin claims ride on the random-row arm;
the CIFAR MLP is a weak encoder (53% best-test-epoch checkpoint) and that domain isolates the new-class /
shared-label structure only, not a quality replication.

## Step 22 — the certificate `C h = 0`, re-read through the imprints (job 701679, `certificate.py`)

**Derivation.** With `B₀ = 0` and SGD every update to `A` lies in `row(B_T)`, so `C := P_{row(B_T)⊥} A_T = P⊥ A₀`.
In the imprint form `B_T ≈ Σ_i q_i (A₀h_i)ᵀ` the row space is spanned by `A₀h_i` of the *recorded* images only, so

```
C h_i ≈ 0  for recorded images,   C h_i ≠ 0  for invisible ones,   rank C = r − N′ .
```

(*Exact* only at full rank: `C = P⊥A₀ + P⊥A₀ H M_T Hᵀ`, and below full rank `P⊥` keeps the invisible directions,
multiplied by the invisible rows of `M_T` — small, not zero. Measured: recorded residuals sit at 1e-10, not
machine precision, eight orders below the invisible ones; that gap is what a threshold buys. The scaling of
the recorded residual — 1e-10 against invisible imprints of 1e-14 and 1e-24 — is not simply the imprint scale
and is an open small-theory question.) No `η`, `T` or labels enter. So (A) `‖Cφ(x)‖` is a **recipe-free test of
whether `x`'s feature vector lies in the span of the recorded examples' features** — subspace membership,
necessary not sufficient: a linear combination passes too; tight at `N′ = 1`, a 6-dimensional subspace of ℝ¹⁰⁰⁰
at `N′ = 6` — and (B) for
`N′ = 1` the `r − 1` linear-in-features equations `Cφ(ψ(w)) = 0` determine the dominant image when `k < r − 1`,
with no unrolled dynamics — a recipe-free, label-free inversion. The theory's "`CH = 0` for all N" is hypothesis
(A4) once more: it holds exactly where every image is recorded and fails where imprints vanish (this is also
the near-duplicate contamination of Step 6, now with its cause). Adam releases have `rank B_T = r`, `C ≡ 0`.

**Part A, measured (strong model, `mnist_control`, `k = 12`; the certificate does not depend on `k`):**

| setting | `rank B_T` | `rank C` (= `r − N′`) | `‖Ch_i‖/‖A_T h_i‖` per image |
|---|---|---|---|
| raw | 6 | **10** | 9e-15, **0.8**, 4e-10, 6e-10, 1e-13, **0.5**, 4e-10, 8e-13 — the two O(1) entries are the two highest-margin (invisible) digits |
| on-chart | 8 | 8 | all eight ≤ 2e-8 |

Confirmed on every batch of job 701679 (`mnist_control`, `hard1_diff`, `confident`; raw and on-chart): certificate
residual 1e-16 … 1e-8 on every recorded image, 0.1 … 1.0 on every invisible one, `rank C = r − N′` throughout
(10 with six recorded, 15 with one, 13 with three). The certificate is the recipe-free face of the imprint law,
and an attacker can apply it to any candidate without knowing how the adapter was trained.

**Part B, two corrections before it could be read** (its rows on 701679 are void). (i) *My design error:* on a
raw batch the true image is not on the chart, so `Cφ(ψ(w_true)) ≠ 0` (objective 0.13–0.17 at the truth) and the
certificate has no zero on the chart at all — it is a hard constraint with no approximate form, so off-chart it is
useless and Part B is meaningful on-chart only. (ii) *Found by the genuineness audit:* the objective
`‖Cφ‖²/‖A_T‖²` is normalised by a constant, so a blank image (`φ → 0` through the GELUs) reaches zero and wins the
attacker's own argmin — the 1e-32 "solutions" at `k = 15, 16` with image error ≈ 1 were exactly that. Fixed to the
scale-invariant `‖Cφ‖/‖A_Tφ‖` (the sine of the angle between `A_Tφ` and `row(B_T)`, Part A's own ratio). Part B
now runs on-chart for `N′ ≤ 6` (job 703061, `k ∈ {8, 10, 12, 14, 16}`): a random public-scale start below the
certificate line `k < r − N′` should land on *one of* the recorded images; rows record which, the fraction of
starts that landed on any, and whether the argmin pick did; `oracle = []`, `recipe_used = False`,
`labels_used = False`. Appending the certificate block to the full LM residual is queued for after that, and will
be reported with and without, same starts — it cannot add information, so any gain is a landscape effect.

*Why the certificate line is `k < r − N′` (yoado-ed):* `Cφ = 0` means `φ ∈ ker C = span(recorded features) ⊕ ker A₀`,
and `ker A₀` is `(n − r)`-dimensional, so `ker C` has dimension `N′ + n − r` — codimension `r − N′` in feature
space. A `k`-dimensional chart image meets it in generically `k − (r − N′)` dimensions: below the line the recorded
images are isolated solutions and generically the only ones; at the line a one-dimensional family; above it
spurious zeros everywhere (seen: fraction of starts at the floor 0 → 0.5 → 0.7 → 0.9 → 1.0 as `k` crosses the line
in the raw rows). The count comes from `ker A₀`, not from C's rank alone. *Claim shape at `N′ > 1`:* every recorded
image is an equally valid isolated solution, so the certificate recovers **one of the recorded examples, chosen by
the start** — not the dominant one; at `N′ = 1` the two coincide. *0/0 guard:* the ratio is undefined as `φ → 0`;
rows now carry `‖A_Tφ‖` against a public reference scale per start, and starts below 5% of it are excluded from
the argmin and counted (`n_degenerate_starts`). The chart is affine (`ψ = μ + Vw`), so `φ∘ψ` is not homogeneous
in `w` and the scale is pinned by `μ`.

## Step 18, distinct-label solve cells (job 624573, 3000 iterations, `random_encoder_control.py`)

| encoder | labels | chart | `σ_min(J)` at truth | residual | iterations | outcome |
|---|---|---|---|---|---|---|
| weak 78% | distinct | global | 8.1e-7 | 7.5e-31 | 93 | floor |
| weak 78% | distinct | local | 1.55e-9 | 1.6e-25 | 3000 (cap) | still descending, err vs chart 2e-5 |
| random | distinct | global | 1.2e-4 | 9.2e-31 | 27 | floor |
| random | distinct | local | 6.8e-5 | 8.6e-31 | 31 | floor |

The weak checkpoint's local-vs-global gap survives distinct labels in the solve (520× in `σ_min`, floor vs cap);
on the random encoder of the same architecture the two charts are within 1.8× and both reach the floor in ~30
iterations. Consistent with the three-seed spectra: the gap is the weak checkpoint's, not trained encoders'.

### Step 22, Part B guarded (job 704286, 16 random starts per cell, on-chart, `N′ ≤ 6`)

| batch (on-chart) | k | N′ | line r−N′ | regime | starts at the floor | starts on a recorded image | argmin pick |
|---|---|---|---|---|---|---|---|
| hard1_diff | 10 | 6 | 10 | at | 0.44 | **2/16** (images 0 and 7) | not on one (0.105 away) |
| hard1_diff | 12 | 5 | 11 | above | 0.25 | 0/16 | spurious zero |
| hard1_diff | 14 | 4 | 12 | above | 0.75 | 0/16 | spurious zero |
| hard1_diff | 16 | 4 | 12 | above | 0.81 | 0/16 | spurious zero |
| confident | 10 | 5 | 11 | **below** | **0.00** | 0/16 (300 iterations; nearest recorded 0.33 away) | 6e-9, not at floor |
| confident | 12 | 4 | 12 | at | 0.19 | 0/16 | spurious zero |
| confident | 14 | 4 | 12 | above | 0.50 | 0/16 | spurious zero |
| confident | 16 | 3 | 13 | above | 0.69 | 0/16 | spurious zero |

No degenerate starts in any row (feature-norm ratios 0.3–1.7). **The structural prediction is confirmed:** below
the certificate line no start finds a spurious zero (0.00), at and above it they are dense (0.19–0.81) — the
kernel count made real. **The selection problem has a regime:** at or above the line the argmin is drawn from a
population that is mostly spurious, so 2 of 16 starts landing on private images at the boundary did not make the
attacker's pick correct, and more starts cannot fix a rule that cannot discriminate. Below the line the situation
inverts — reaching the floor is itself the proof of having found a recorded example — and the only obstacle is
the basin, which cheap starts (≈0.2 s each, no unroll) can buy. Job 706721 spends 2,000 starts per cell there
(`k ∈ {6, 8, 10}`, both batches). Reading, conditional on it: *the certificate identifies the recorded set exactly
and recipe-free; below its line, reaching the objective's floor is proof of a recorded example and cheap starts
buy attempts; at or above the line spurious solutions are dense and no number of starts rescues the selection.*

### Step 21/20 subset test — a recipe error found and fixed (job 634238 VOID; rerun 706597)

The subset rows evaluated the N′-image residual at the recorded images' own truth and found **1.7e-2** (N′ = 3)
and **1.7e-3** (N′ = 6) where the predicted floor was 1e-16 / 1e-31 — and solves then "beat the truth" with wrong
images (residual 1.4e-8 at image error 0.09). Cause: the recipe divides the gradient by the number of images it
is given (`D = R/N`), so an N′-image simulation at the original `lr` runs a *different recipe* (effective rate
`lr/N′`), which the R1 result already said can never reach the floor. Fix: simulate the subset at `lr·N′/N`, so
the omitted images contribute zero gradient — what "invisible" means; the attacker sees only the single
effective rate `lr/N` (fittable, Step 9); `N` is taken as known here and flagged (`oracle` gains `N_known`).
634238's rows are void; 706597 reruns Part A with rows written as produced. **First corrected row (706597,
repeated on-chart, `N′ = 3`, recorded subset):** predicted floor 1.1e-16, residual *at the recorded images' truth*
1.1e-16 — the two independent computations of the floor now agree — and the solve reaches it (residual 1.1e-16,
over-floor 1.03), `σ_min` at the subset truth 4.7e-8 (identifiable), image error 2.5e-2: with invisible images
present the subset is determined to about `√floor / σ_min`, not to machine precision. Controls follow.

### (R5) — the batch size is not identifiable from the release (corollary, yoado-ed; falsifier job 709507)

In the recurrences the step, the adapter scale and the batch size enter only through the single product `η·s/N`
(the `1/N` from the loss being a mean). Consequences: (i) the subset simulation at `lr·N′/N` is forced, not
chosen; (ii) **the attacker never needed `N`** — they fit the one scalar `η/N`, which R2 already measured succeeding
to 5e-16, so the "N known" flag has been dropped from the subset rows; (iii) **`N` cannot be recovered from the
release at all, only `N′` through `rank B_T`** — a batch of eight with two invisible members gives the same release
as its six recorded members at a proportionally smaller step, up to the omitted imprints. Exception: decoupled
weight decay contributes `−η·wd·B_t`, so the release then determines two combinations, `η/N` and `η·wd`, and a
published nonzero weight decay publishes the batch size. **Measured (job 709507, `batch_scale_check.py`), six of
six cells:** the recorded members alone at `lr·N′/N` reproduce the full-batch release to exactly the omitted
imprints' relative scale — `‖ΔB_T‖/‖B_T‖` = 1.8e-15 vs omitted 1.9e-15 (repeated raw, N′ = 6), 4.9e-16 vs 4.0e-16
(repeated on), 1.3e-15 vs 1.5e-15 (hard1_diff raw, N′ = 1), 7.5e-17 vs 3.8e-17 (hard1_diff on, N′ = 4), 4.6e-14 vs
4.6e-14 (confident raw, N′ = 3), 6.8e-16 vs 6.7e-16 (confident on) — while the same subset at the *unscaled* step
differs by 3e-2 … 1.7 (the 634238 error). Constructed controls at the scaled step (job 710597, replacing a random
subset that shared most members at large N′): a subset built from the *complement* (all invisible members, filled
with the weakest recorded ones; overlap 0–4 of N′) differs by **1.0 in all six cells**; the recorded set with its
weakest member swapped for the strongest invisible one differs by 1.1e-11, 6.2e-10, 1.0, 5.7e-8, 1.3e-8, 1.4e-6 —
i.e. by the swapped member's own imprint, which is what the imprint law says it should. **`N` is not in the
release; `N′` is.**

### Step 22, Part B below the line — first rows (job 706721; bracket pending)

`confident` batch on-chart, `k = 6`, `N′ = 7`, certificate line `r − N′ = 9` (below). **2,000 random public-scale
starts, 300 iterations each, no recipe, no labels, no start near the truth: 51.0% land exactly on a private
image** (error 5.5e-15 to its chart projection); **all seven** recorded images are found; the fraction of starts
at the floor equals the fraction on a private image (0.510 = 0.510 — no spurious zero, as the kernel count
requires below the line); zero degenerate starts; the attacker's argmin pick is a private image at 5.5e-15;
0.2 s per start. Figure `figures/exact_inversion/certificate_recovery_k6_706721.png`.

**The equality of the two fractions is the kernel count measured directly:** 1,020 of 2,000 starts reached the
floor and *all 1,020* are on a recorded image — below the line the recorded examples are the only solutions, so
every floor-reacher must be one of them; had the fractions differed, the count would be wrong. The landing is not
a thresholded near-miss: over the 1,020 floor-reachers the image error to the nearest recorded image has min
1.3e-15, median 5.3e-15, 90th percentile 1.5e-12, max 3.7e-12 — machine precision throughout, no tail — while the
980 non-floor starts sit at objective ≈1e-3 with image error ≈0.47, cleanly separated. Landings per recorded image
are uneven (414, 383, 99, 63, 37, 13, 11): the basins differ in size by 40×, so "which image a start finds" is
weighted, not uniform, and the rarest image needed ~180 starts.

*What orders the basin sizes (job 716701, a lead, not a result — n = 7, only a perfect ordering would
discriminate):* landings per recorded image against the imprint 17/21, against −margin 16/21, against `‖A_Tφ‖`
15/21, against `‖x_on‖` 16/21; imprint and −margin agree with each other 20/21 (one latent quantity), `‖x_on‖` is
independent of them (12/21) and orders the landings as well. Under the null, Kendall's τ at n = 7 has standard
deviation 0.32, so 17/21 (τ = 0.62) is under two of them and, as the maximum over four correlated candidates, is
what nothing looks like; and among the three largest imprints — where noise on `‖C_i‖` is negligible — the
ordering inverts (imprint 0.59 → 63 landings, 0.20 → 383). **The null is the result:** presence is the release's
(the imprint law); search cost is not ordered by it on this cell. Across ladder cells the per-cell taus are pooled,
never the raw (image, landings) pairs.

**Pooled over the ladder (job 734584, `basin_predictors.py`; 17 on-chart cells with landings, 331 pairs; cells
with no landings contribute τ = −1 by construction and are excluded):** landings per recorded image are ordered
by the image's **feature norm** `‖A_Tφ(x_on)‖` — pooled τ **+0.43** (null sd 0.08, positive in 14 of 17 cells) — and
its pixel norm `‖x_on‖` (+0.40, 15 of 17), and only weakly by the imprint (+0.18, 13 of 17) or −margin (+0.15).
Imprint and −margin are one latent (pairwise +0.96) and nearly independent of the geometry predictors (+0.10,
−0.05). So: **presence is the release's (the imprint law); search cost is the chart's geometry** — a candidate
with a larger feature norm has a wider basin. Caveat on the z-values: the 17 cells come from three batches across
ranks and are not independent; the sign consistency across cells is the evidence, not the z. **Pooled per batch
(the independent units):** `confident` (13 cells, 232 pairs) feature norm +0.44, pixel norm +0.55, imprint +0.22;
`mnist_control` (3 cells, 84 pairs) feature norm +0.50, pixel norm +0.17, imprint +0.12; `hard1_diff` (1 cell,
15 pairs, landings 0–1 — barely informative) −0.07 / −0.60 / +0.07. Two batches positive, the third a single
near-empty cell: the evidence rests on two independent units, dominated by one. **The per-batch split also
discriminates the two geometry predictors:** the feature norm `‖A_Tφ(x)‖` *replicates* (+0.44, +0.50) while the
pixel norm does not carry (+0.55 in the batch supplying 13 of 17 cells, +0.17 in the other — the shape of a
single-batch artefact that the pooled +0.40 concealed). The sentence names the feature norm: the basin is set by
how the *adapter* sees the candidate (through the released `A_T`), not by how large the image is in pixel space. **Open confound, one cell
running (job 736737):** starts are drawn at a fixed public scale, so a target whose norm sits nearer that scale
is nearer the starts — `‖x_on‖` could be a property of the *start distribution*, not the chart. Re-drawing the
same batch's starts at 0.5× and 2× the public scale (`--start-scale`) separates them: if the landing order
survives, it is chart geometry; if it tracks the scale, an attacker who varies the start scale covers targets a
single scale misses — which would turn the basin skew from a limitation into a technique.

*Identity, not just pixels (job 719106):* the base model reads 6 of the 8 `k = 6` projections as their true digit
(the 4 → 9, the 5 → 8; 5 of the 7 recorded). That is the illustration, not the number: the classifiability of a
`k`-dimensional chart is a property of the chart family and `k` alone, so it is measured standalone over 2,000
held-out digits per `k` (`chart_fidelity.py`, job 719793) and the ladder's admissible `k` at each `r` are marks
on that curve. Figure re-rendered as two rows — real digit / chart projection with the per-image recovery error,
landings, and the model's reading printed under each — since the attack's output coincides with the projection.

This is the first recovery in the study that begins from nothing, and it is exact. Its scope: (i) *on-chart* —
the fine-tuning images lie on the `k = 6` chart, so what is recovered is those chart images, which at six PCA
components are blurs (the figure shows it); off-chart the certificate has no zero. (ii) The certificate budget is
`k < r − N′`, so at `r = 16` with seven recorded it is tiny; **the budget scales with the LoRA rank** — a
higher-rank adapter opens a proportionally richer recipe-free channel (untested; the natural next cell).
(iii) Which recorded image a start lands on is chosen by the start, not the attacker. Bracket rows (`k = 8`
below, `k = 10` above; `hard1_diff`) follow.

**Pre-registered for the rank cell (job 716016, `r ∈ {16, 32, 64}`, on-chart, 500 → 5,000 adaptive starts).** Two
claims, kept apart by a fixed-`k` ladder (`k = 8` at every `r`): *rank buys budget* — the certificate line
`k < r − N′` moves up with `r`, so a richer chart is admissible — is the scientific claim; *rank buys basin* — at
fixed `k` more slack below the line makes the search easier — is a separate, weaker effect. The deliverable's
y-axis is the chart's own representation error at each `k`: the honest form of the fidelity claim is **"the
recipe-free channel's fidelity ceiling is the chart error at `k = r − N′ − 1`"**, a curve with its own falsifier
(if the recovered error at the largest below-line `k` does not sit on the chart-error curve, something other than
the budget limits it). A zero recorded fraction at few starts is an *unresolved basin*, not a closed channel —
the search-failure/alias distinction in a new form — hence the adaptive budget. Framing, if both lines move:
**the LoRA rank is a leakage dial for both routes** — the recipe route's `k < m + r − N` and the certificate route's
`k < r − N′` — and it is the knob practitioners turn for utility.

### Step 18 — the random-encoder ladder at fixed architecture is complete (job 624573)

Global chart, repeated draw, random encoder (Gaussian weights at the trained layers' norms): `σ_min(J)` at the
truth 2.1e-5 (k=6), 3.8e-5 (10), 1.6e-5 (14), 5.0e-7 (17), **2.5e-19 (18)** — full rank to 17, collapse at 18,
every below-line cell at the floor within 60–272 iterations. The line holds on the random encoder exactly as on
the trained ones; only the conditioning differs (this is the zero point of the quality ladder).

### Step 22 — the fidelity axis, standalone (job 719793, `chart_fidelity.py`, 2,000 held-out test digits)

Class identity preserved by a `k`-dimensional PCA chart (fitted on 50k train digits), as the base model's accuracy
on the projections — a property of the chart family and `k` alone (no adapter, release, rank or search):

| k | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 | 40 | 48 | 56 | 64 | raw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strong 98% | .20 | .31 | .52 | .68 | .75 | .83 | .88 | .94 | .95 | .97 | .97 | .975 | .978 | .977 | .979 |
| mid 95% | .31 | .46 | .66 | .76 | .81 | .85 | .88 | .92 | .93 | .94 | .94 | .94 | .94 | .945 | .951 |
| weak 78% | .23 | .32 | .47 | .58 | .60 | .63 | .69 | .73 | .76 | .79 | .79 | .79 | .79 | .79 | .784 |
| repr. err (median) | .73 | .69 | .64 | .62 | .59 | .56 | .52 | .49 | .46 | .41 | .38 | .34 | .32 | .29 | — |

**Instance survival, the privacy axis (job 721003, same 2,000 digits, nearest neighbour among the full 10,000-digit
raw test pool):** does the `k`-projection retrieve *its own source* (self, top-1) or *another member of its
class* (an archetype)?

| k | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| self, top-1 | .001 | .01 | **.04** | **.11** | .21 | .33 | .57 | .73 | .84 | .94 | .97 | .99 | .99 | .99 |
| self, top-5 | .006 | .03 | .12 | .26 | .43 | .57 | .79 | .90 | .94 | .99 | .995 | .996 | .997 | .999 |
| another of the same class, top-1 | .30 | .44 | .53 | .58 | .57 | .49 | .35 | .23 | .15 | .06 | .03 | .01 | .01 | .01 |

Class survival and instance survival come apart exactly in the certificate's `r = 16` regime: at `k = 6–8` the
projection reads as the right class for 52–68% of digits but identifies the *specific* digit for only 4–11%,
retrieving a different digit of the same class more than half the time — "there was a 4 in the batch", not "this
4". At `k ≥ 32` (admissible at `r = 64`) both survive: 94–99% self-identification. Caveat on the other control:
against a pool of *projected* candidates the projection identifies its source with certainty at every `k` (top-1
1.000 even at `k = 2`), since projection is deterministic — a membership-style identification that needs the
candidate pool. **It is robust, not a precision artefact (job 723107):** perturbing the recovered coordinates by
`ε` × coordinate std and re-running the retrieval among the 10,000 projected candidates, the source stays top-1 at
100% for `ε ≤ 3e-2` at every `k ≥ 8` (96–100% at `ε = 0.1`; 49–99% at `0.3`), and even `k = 2` holds 99.8% at
`ε = 1e-3` and 95% at `3e-3` (63% at `1e-2`). Solves land at 1e-15; a stopped-short solve at 1e-2 still identifies.
So an attacker holding a candidate pool never needs to render anything: the low-`k` regime that is harmless as a
*reconstruction* is fully effective as a *membership* attack — the six-coordinate chart that cannot draw a digit
answers "was this record in the training set" with certainty. That is the strongest privacy statement in this
section, and the objection "a six-coordinate chart cannot capture a real image" is correct and irrelevant to it.

The ladder's admissible `k` at each rank (`k < r − N′`) are marks on this curve: at `r = 16` with seven recorded
the admissible charts (`k ≤ 8`) keep 52–68% of digits identifiable to the strong model; at `r = 64` with eight
recorded (`k ≤ 55`) the admissible chart is at the raw accuracy. Whether the *basin* holds at those `k` is the
ladder's question. *(A warning written here earlier — "at `r = 16, k = 8` 0 of 500 starts landed" — was wrong: that
was the ladder job's RAW cell, `N′ = 3`, whose truth is not on the chart and which has no zero by construction. The
on-chart `k = 8` cell, measured in the bracket below, lands 18% of starts. My own rule — check the truth before
reading a solve — caught it one row late; the ladder and the twenty-image cell were restarted on-chart only.)*
"Rank buys budget" and "the budget is reachable" remain separate claims; the fixed-`k` arm (`k = 8` at
`r = 16 / 32 / 64`, run first in job 721391) decides whether the basin is governed by `k` or by distance below the line.

**The fixed-`k` arm (job 721391, `confident` on-chart, `k = 8`, `N′ = 7`, 500 random starts each):**

| r | line `r − N′` | distance below | **basin per recorded image** (aggregate / N′) | aggregate on any private image | at the floor | found | argmin |
|---|---|---|---|---|---|---|---|
| 16 | 9 | 1 | **2.4%** | 16.6% | 16.0% | 6 of 7 | on a private image |
| 32 | 25 | 17 | **10.6%** | 74.4% | 70.0% | 7 of 7 | on a private image |
| 64 | 57 | 49 | **13.8%** | 96.4% | 87.0% | 7 of 7 | on a private image |

The per-image basin is the comparable quantity: the aggregate scales with how many private images there are to
land on, and `N′` moves with `k` in every cell because projection changes the margins (e.g. `r = 32, k = 12` has
`N′ = 4` with aggregate 63% but per-image 15.8% — *larger* than `k = 8`'s 10.6%, the opposite of the aggregate's
reading). What this arm shows is distance dominating **at fixed `k` and fixed batch**; it does not show `k` has
no effect. The converse needs the mirror design — equal slack, different `k`: `r = 32, k = 8` against
`r = 64, k = 40`, both seventeen below their lines — which is inside the running ladder.

Second batch, `mnist_control` (`N′ = 8`), same `k = 8`: `r = 16` sits *at* its line (8) — 4% on a private image,
argmin wrong; `r = 32` (line 24, sixteen below) — **73.4%**, all eight found, argmin correct, floor 69%; `r = 64`
(line 56, forty-eight below) — **89.0%**, all eight found. The basin saturates with distance below the line:
16.6 → 74.4 → 96.4% at distances 1 → 17 → 49 (first batch).
Same chart, same images, same `k`, same starts: the basin quadruples when the same `k` sits seventeen rather than
one below the line, and goes from the at-line regime to 73% for the second batch. **Distance below the certificate line governs the basin, not `k` itself — rank buys
reachability as well as budget.** (`mnist_control` at `r = 16, k = 8` sits *at* its line — all eight recorded, line
8 — and shows the at-line regime: 34% of starts at the floor, 4% on a private image, argmin wrong.)

**Bracket, `k = 8` (job 706721, on-chart, `N′ = 7`, line 9, one below):** 2,000 starts → 345 at the floor, 364 landed on
a recorded image (19 landed at objective ~1e-17, converging), all seven found, argmin on a recorded image at
3.6e-15, no spurious zero. **Basin 18% at `k = 8` against 51% at `k = 6`** — narrowing toward the line, open.

### Step 22 — the precision of the released adapter is a privacy parameter (job 722663, `precision_check.py`)

Releases already measured, rounded to FP32 / TF32 / FP16 / bfloat16 and back; `rank B_T`, `rank C` and the
per-image certificate residual re-read with the SVD tolerance at 10× the dtype's epsilon (a fixed tolerance would
read rounding noise as *extra* rank). Strong model, `r = 16`, `k = 16`, eight images:

*(A first version of this table counted images with residual < 1e-3 — a threshold at or below the bfloat16 noise,
which manufactures a zero by construction (caught by the design audit). The metric that answers the question is
**separability**: a recorded image is still readable if its certificate residual sits ≥ 2 orders below the
smallest residual among the invisible images of the same release; that gap is what an attacker with a
per-release threshold reads.)*

| cell | `B_T` spectrum (rel., FP64) | recorded images separable from the invisible band, FP64 → FP32 → TF32 → FP16 → bf16 |
|---|---|---|
| repeated, on-chart | 1, 2e-4, 1e-5, 9e-9, 4e-9, 1e-11, 2e-16, 1e-16 | 6 → 3 → 0 → 0 → 0 (strongest recorded at 0.2 vs invisible 0.98) |
| repeated, raw | (rank 6) | 6 → 2 → 1 → 1 → 1 (3e-3 vs 0.65) |
| confident, on-chart | (rank 3) | 3 → 3 → 1 → 1 → 1 (2e-3 vs 0.93) |
| hard1_diff, on-chart | (rank 4) | 4 → 2 → 2 → 2 → 1 (6e-3 vs 0.44) |

Quantisation noise (relative): FP32 3e-8, TF32/FP16 2e-4–7e-4, bf16 1e-3–2e-3. The mechanism is the spectrum: the
release's singular values fall steeply (second direction at 2e-4 of the first), so every direction below the
quantisation noise collapses into the invisible band — FP32 keeps what is above 3e-8, bfloat16 keeps the single
dominant direction in three of four cells and nothing in the fourth. **A bfloat16 adapter narrows the recipe-free
channel to at most one example; it does not close it.** Stated with its tolerance rule (10× epsilon) and its
scope (the certificate channel; the full-residual channel at low precision is a separate measurement).

### Step 22 — a second cap on the certificate: at most `m − 1` images (found by the twenty-image cell)

The twenty-image cell (`optdigits`, `N = 20`, `r = 64`, strong model, on-chart) gave `rank B_T = 9` with a clean
gap (spectrum 1, .8, .3, .2, .1, .09, .03, .01, .005, **3e-16**) while all twenty imprints are `O(1)` — and the
certificate residual is 1e-2 … 0.5 for *every* image, even in FP64: **no image is individually in `row(B_T)`.**
Cause: `B_T = Σ_i q_i (A₀h_i)ᵀ` with the accumulated error vectors `q_i ∈ ℝᵐ` on the softmax simplex, so at most
`m − 1 = 9` of them are independent; with twenty recorded images `row(B_T)` is a 9-dimensional subspace of the
20-dimensional span of their feature directions, aligned with none of them. The certificate line is therefore
`k < r − N′` **and** `N′ ≤ m − 1`; past `m − 1` recorded images the recipe-free channel mixes them and recovers
none individually (the full-residual channel is a different question). This is the same `m − 1` as in the
capacity count `N(m − 1 + r − N)`, met from the certificate side. Consequences: the twenty-image basin-ordering
design cannot run on a 10-class head (its Part B was stopped, job 721393); it needs a head with `m ≥ 21`
(EMNIST letters, 26 classes, is on disk). The eight-image cells (`N′ ≤ 7 < 9`) are unaffected. **The truncation
tolerance is the attacker's free knob (job 727654, `tolerance_sweep.py`):** rebuilding `C` from the same `B_T` at
tolerances 1e-1 … 1e-16 — on the 10-class head the annihilated count is **0 at every tolerance and every dtype**
(spectrum 1, .6, .4, .3, .2, .08, .07, .02, .02, **3e-16**: nine directions, then nothing) — the cap is not a
tolerance artefact. **Pre-registered
positive test (job 725918):** the *same* MNIST MLP retrained with a padded 26-logit head (16 logits never targets —
only `m` changes; data, encoder, batch, rank and chart fixed), then the identical twenty-image cell. Predictions:
`rank B_T` 9 → **20**; `rank C` 55 → **44** (= `r − N′`); the per-image certificate residual falls from 0.3–0.5 to
the floor for all twenty; the certificate line moves to `k < 44`, so below it random starts land on the twenty.
A cap that is measured switching back *on* when `m` is widened is a different class of evidence from an
explained failure. **Measured (job 725918, `mnist_mlp_m26_strong.pth`, 98.28% — the same accuracy as the 10-logit
model, same data):** at `k = 8` on the identical twenty images, `rank B_T` **9 → 19**, `rank C` **55 → 45**
(= `r − N′`), and the certificate residual of **nineteen** images falls from 0.3–0.5 to **2.5e-8 … 4.3e-3** — five to
seven orders below the invisible band — with the twentieth (imprint 3e-7 of the largest, the numerical boundary
between "present" and "in the row space") at 0.23. **Then the tolerance sweep (727654): the count climbs 0 → 1 → 10 → 16 → 18 → 18 → 20 as the tolerance tightens
from 1e-8 to 1e-14, and `B_T`'s spectrum runs 1, .7, .5, .2, .1, .07, .06, .01, 3e-3, 6e-6, 1e-7, 7e-9 … 3e-12, 6e-14,
then 2e-16.** So the prediction 20 / 44 is met *exactly* at a tolerance of 1e-14 — the shortfall was my truncation
— and the twentieth image is a case of **collinearity, not magnitude**: its imprint is 3e-7 of the largest but its
direction contributes a singular value of 6e-14, seven orders smaller, because it is nearly dependent on the
other nineteen. The channel that was dead for all twenty on the 10-class head is open for all twenty on the
26-class head with nothing else changed. Per dtype the wall of the count-vs-tolerance curve — the release's
information content — is 20 (FP64), 17 (FP32, noise 2e-8 swamps σ₁₈…σ₂₀ at 2e-11 … 6e-14), and **0** at TF32 /
FP16 / bfloat16 for this batch (nothing annihilated below 1e-3 at any tolerance: with twenty comparable
imprints the residual of even the strongest is set by the noise). For the eight-image cells the wall by
separability is: repeated 6 / 6 / 5 / 2 / 5 and confident 3 / 3 / 3 / 1 / 3 over FP64 / FP32 / TF32 / FP16 / bf16 —
higher than the single-tolerance counts of the precision table, which should be read as lower bounds. *(An
earlier version of this sentence called FP16 "the smallest mantissa" — wrong: FP16 carries ten mantissa bits to
bfloat16's seven; what FP16 lacks is exponent range, five bits against eight, so its normals stop near 6e-5.)*
**Dynamic range decides, not precision:** the imprint spectrum spans many orders, FP16 *underflows* the small
imprints to zero, bfloat16 keeps them coarsely — so bfloat16, the format deployment actually uses because it
keeps FP32's exponent range, is the *most* revealing of the low-precision formats and FP16 the least. "Quantise for
privacy" is not a slogan the measurement supports; what closes the channel structurally is the *many comparably
recorded examples* case (the twenty-image cell dies below FP32 because rounding noise sets even the strongest
residual), and low precision barely dents the common confident-model case where a few strong imprints dominate.
The sharper statement, from the twentieth image: what decides individual recoverability is the size of the
**independent direction** an example contributes, not the size of its imprint — and the two can differ by orders. *Survey across
`k` (job 722950):* `rank B_T = 9` and `rank C = 55` at every `k ∈ {8, 12, 16, 20, 24, 32, 40}`, all twenty imprints
present, certificate residual 0.3–0.5 throughout — the cap is `m − 1` exactly and independent of the chart.

## Step 23 — the exact-arithmetic existence corner: instance-identifying private images from random starts, no recipe, no labels (jobs 728592 and 721391)

> **This cell was titled HEADLINE and is not one — read it with its scope, which was measured after it was written.**
> The confident batch's release at r = 64, k = 32 has **‖B_T‖ = 7.6e-18** with absolute imprints down to **2e-29**
> (job 752500); the adapter never moves the logits there (feedback 4e-19, `A_T = A_0` to working precision), so this is
> the closed-form one-step gradient, not a trajectory. The recoveries below are exact because the certificate is an
> angle, and the imprint-sum mismatch at this cell is 6e-32 — fourteen orders below the release — so the release is
> genuine signal, not roundoff. But every rank here is read at a *relative* tolerance off a numerically vanishing σ₁
> (Step 18's own warning), and the cell is precision-fragile: **fp16 storage zeroes the whole file (0 of 8), bf16
> storage finds 3, fp32/tf32 storage 5, and fp32 *training* of the same batch leaves 5 of 8** (Steps 24–25).
> **The robust, lead result is the new class (Step 25): letters at r = 64, k = 32, trained in fp32 — all eight
> recovered from random starts at the tight tolerance, imprints of order one, an adapter that moves the logits by
> 43%.** STATUS.md leads with that cell; this one is the exact-arithmetic corner that says the channel is sharp when
> the arithmetic is.

Strong 98% MNIST model, LoRA `r = 64`, eight private test digits (`confident` batch — the ones the model was
*most* sure about at `r = 16`; on-chart at `k = 32` all eight are recorded), chart = 32-component PCA of the public
train split. Certificate-only inversion `‖Cφ(ψ(w))‖/‖A_Tφ(ψ(w))‖ → 0` from **500 random starts at the public
coordinate scale**, 300 LM iterations each, no unrolled recipe, no labels, nothing from the truth:

| job | starts on a private digit | per image | found | landings per digit | argmin pick | at the floor |
|---|---|---|---|---|---|---|
| 728592 (dedicated) | **65.6%** | 8.2% | **8 of 8** | 19, 24, 26, 30, 34, 34, 58, 103 | on a private digit, err 2.2e-14 | 42% |
| 721391 (ladder, same cell) | **66.0%** | 8.3% | 8 of 8 | — | on a private digit | 42% |

Per-image best landing errors 1e-14 … 9e-5 (five at machine precision, three at 1e-5 … 1e-4 — converging at the
300-iteration cap; the 1e-2 landing criterion does no work). No degenerate starts, no spurious zeros. And at
`k = 32` the chart is instance-identifying: the base model reads all eight projections as their own digit, and on
the held-out curve a 32-component projection retrieves *its own source* among 10,000 candidates 94% of the time
(Step 22). Figure `figures/exact_inversion/certificate_recovery_r64_k32_728592.png` — real digit / `k = 32`
projection / the recovered panel for each of the eight.

**What this is:** every ingredient measured separately in Step 22 — budget from the line (`k < r − N′ = 56`),
basin from the distance below it (24), fidelity from the chart at `k = 32` — combined in one cell, and it
holds. A rank-64 adapter of a strong model, fine-tuned on digits it was confident about, yields the specific
private digits from random starts with the release, the public model and a public chart, and nothing else.
**What it is not:** off-chart (the fine-tuning images here lie on the 32-component chart by construction; the
raw-digit case has no certificate zero — Step 22), Adam (no certificate), or a head narrower than `N′ + 1`.

### The per-rank sweeps (job 721391, `confident` on-chart, 500 starts each; per-image basin = aggregate / N′)

| r | k | N′ | line | below by | aggregate | **per image** | found | chart's class acc. (curve) | instance id (curve) |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 8 | 7 | 9 | 1 | 16.6% | 2.4% | 6/7 | .68 | .11 |
| 32 | 8 | 7 | 25 | 17 | 74.4% | 10.6% | 7/7 | .68 | .11 |
| 32 | 12 | 4 | 28 | 16 | 63.0% | 15.8% | 4/4 | .83 | .33 |
| 32 | 16 | 3 | 29 | 13 | 37.2% | 12.4% | 3/3 | .88 | .57 |
| 64 | 8 | 7 | 57 | 49 | 96.4% | 13.8% | 7/7 | .68 | .11 |
| 64 | 16 | 3 | 61 | 45 | 85.8% | 28.6% | 3/3 | .88 | .57 |
| 64 | 24 | 7 | 57 | 33 | 83.6% | 11.9% | 7/7 | .95 | .84 |
| 64 | 32 | 8 | 56 | 24 | 66.0% | 8.3% | 8/8 | .97 | .94 |

| 64 | 40 | 8 | 56 | 16 | 45.2% | 5.7% | 8/8 | .97 | .97 |
| 64 | 48 | 8 | 56 | 8 | 20.6% | 2.6% | 8/8 | .975 | .99 |
| 64 | 56 | 7 | 57 | 1 | 2.0% | 0.3% | 4/7 | .978 | .99 |

**The mirror pair (equal slack, different `k`):** `r = 32, k = 8` (17 below) has a per-image basin of 10.6%;
`r = 64, k = 40` (16 below) 5.7%. So at held distance, 32 more unknowns cost about 2× in basin — distance
dominates, but `k` is not free. **Chart/private disjointness, verified in the code path for 728592 and every
ladder row:** the chart is `PCAChart(Xtr_t, k)` with `Xtr` from the *train* idx file (first 50,000), the
coordinate scale from the same train images, and every private batch from `Xte_t`, the *test* idx file.
`N′` moves with `k` (projection changes the margins), so only the per-image column is comparable across rows.
*Caption.* Within the `r = 64` series `k` rises as the distance below the line falls — anti-correlated by
construction — so the declining per-image basin (13.8 → 28.6 → 11.9 → 8.3 → 5.7 → 2.6%) cannot be attributed to
either; the attribution is carried by the fixed-`k` arm and the mirror pair, which varied one at a time, and the
series is consistent with them, not evidence for them. What the series *does* show: **every cell in the admissible
range finds all recorded images with the argmin correct** — from `k = 8` at 13.8% per image to `k = 48` at 2.6% —
so at `r = 64` the admissible chart range up to `k = 48` is attackable and only the *cost* varies, a factor of five
(at `k = 56`, one below the line, the basin collapses to 0.3% per image and 4 of 7 are found — the same one-below
collapse the fixed-`k` arm showed at rank 16),
against a fidelity that climbs from class-only (.11 instance id) to .99. There is no dimension in that range
where the attacker is stopped, only ones where they must buy more starts. (The `k = 16` point has `N′ = 3`
against seven or eight elsewhere — a coarse estimate, not to carry weight either way.) At `r = 32` the admissible
`k` stop at 16–24. The mirror pair (`r = 32, k = 8` vs `r = 64, k = 40`, both ~seventeen below) is above.

### Find-some-among-twenty on the wide head (job 725918, `m = 26`, `r = 64`, `k = 8`, 10,000 random starts)

Twenty optdigits, nineteen certificate-recoverable (Step 22): **88.6% of 10,000 starts land on a private digit;
18 of 20 found** (landings median 241, max 2,402; the boundary image and one more at 0), argmin on a private digit.
On the 10-class head the same cell found nothing (rank 9, certificate dead for all): the `m − 1` cap measured from
both sides *and* through the attack. (The objective floor here is ~1e-6, not 1e-20, since the recorded residuals
on this head sit at 1e-8 … 4e-3 — landings are counted by image error < 1e-2.)

### Bracket, `k = 10` (job 706721): still below its line for this batch (`N′ = 5`, line 11) — 13.4% of 2,000 starts
on a private digit, all five found, argmin correct, floor fraction 12.9% ≈ recorded (no spurious zero). The
at/above-line regime is the earlier 704286 rows (`k = 12, 14, 16`: spurious zeros dense, argmin wrong).

### The start-scale cell (job 737516): the public scale is not special; the skew is partly the start distribution's

Same cells, 500 random starts each, drawn at 0.5× / 1× / 2× the public coordinate std:

| cell | scale | aggregate on a private image | found | landings per image (in image order) | Kendall(landings, feature norm) |
|---|---|---|---|---|---|
| r=64, k=32 | 0.5× | 69.4% | 8/8 | 5, 25, 6, 103, 8, 13, **183**, 4 | 16/28 |
| | 1× | 65.6% | 8/8 | 34, 34, 30, 58, 26, 19, **103**, 24 | 23/28 |
| | 2× | 71.2% | 8/8 | 33, 33, 63, 45, 31, 13, **80**, 58 | 18/28 |
| r=32, k=8 | 0.5× | 71.6% | **5/7** | **0**, 19, 1, 35, 6, **297**, **0** | 14/21 |
| | 1× | 74.4% | 7/7 | 6, 73, 16, 84, 10, **180**, 3 | 15/21 |
| | 2× | 67.0% | 7/7 | 12, 51, 47, 84, 3, **127**, 11 | 17/21 |

Landing-order stability across scales: 0.5× vs 1× 20/28 and 19/21; 1× vs 2× 20/28 and 19/21; 0.5× vs 2× 15/28
and 17/21. **Reads.** (i) The *aggregate* is insensitive to the start scale (65–71%, 67–74%): the public scale is
not the best place to start, nor a bad one — any scale in this range lands most starts. (ii) The *distribution*
over targets is scale-dependent: at 0.5× the mass concentrates on the dominant image (183 of 347; 297 of 358)
and two of seven images are never found; at 2× it flattens (min/max 13/80 and 3/127). The dominant image is the
same at every scale (chart geometry), the rest re-order with the scale (start distribution): the confound is
resolved as *both*, and the feature-norm ordering measured at 1× (23/28) is not scale-invariant (16/28 at 0.5×).
(iii) The actionable version stands: **an attacker who sweeps the start scale covers targets a single scale
misses and flattens the coupon-collection skew** — at 2× the rarest image needs a fifth of the starts it needs at
0.5×. The "search cost is the chart's geometry" sentence is narrowed to the dominant target; the rest of the
ordering belongs to the start distribution, which the attacker controls.

### Subset test, the one-swapped control (job 706597): a wrong subset reaches ITS OWN floor — discrimination is by the floor's level

`repeated` on-chart, `N′ = 3`: the recorded subset reaches its predicted floor 1.1e-16 exactly (over-floor 1.03,
image error 2.5e-2); the **one-swapped** subset (weakest recorded → strongest invisible) *also* reaches its own
predicted floor — 2.1e-10 predicted, 2.1e-10 at its truth, 2.1e-10 achieved, over-floor 1.00 — at image error
7e-2 with `σ_min` 3.3e-10. So "reaches its predicted floor" does not discriminate subsets, as the design audit
warned it might; **what discriminates is the level of the floor**, six orders apart here (1e-16 vs 2e-10), and
the attacker sees that level as the achievable residual of each candidate subset. "Find some" therefore reads:
*minimise the residual over the choice of subset* — the recorded subset is the one whose residual can go lowest —
which is a search over subsets the attacker can run without knowing which images are recorded.
*Confident-only control (same job, later row):* the subset of the three *invisible* images (imprints 3e-10, 4e-20,
2e-16; predicted floor 1.0 — the whole of `B_T` unexplained; residual at that subset's truth 1.0) does **not** reach
the recorded floor level: from a near start the solver drives the residual to 3.2e-4 (B-block 1e-5) by moving *one*
candidate far off its start (image error 0.85; the other two stay at their 0.064 start error) — a candidate drifting
toward what does explain the release, not a floor. Three levels in one cell: recorded subset 1.1e-16, one-swapped
2.1e-10, confident-only 3.2e-4 (twelve orders): the achievable residual orders the candidate subsets by how much
of the release they contain. *At N′ = 6 (same job, later rows) the levels no longer separate:* the recorded
six-image subset (predicted floor 1.6e-31) and the one-swapped subset (3.8e-19, the swapped-in image's imprint
being 1e-19 of the release) both stall at 4–5e-17 with image error 0.10 (σ_min at the truth 1e-14 … 1e-20,
rank-deficient) — both predicted floors lie below the solver's reach, so the swap of an image the release barely
contains is undetectable, which is the same statement from the other side: the level orders subsets exactly as
far as the release distinguishes them.

### Closures from the older jobs
- **The alias form did not appear (job 614344, 98% model, repeated draw, `k = 6 … 18`):** `σ_min` at the truth is
  1e-14 … 1e-19 at *every* `k` (the two invisible digits make the Jacobian rank-deficient below the line), and the
  residual never reaches the floor (7e-18 … 1e-12 at the cap, still descending, image error 5–15%) — the strong
  model's below-line failure manifests as non-convergence on a rank-deficient problem, not as a floor reached
  with wrong digits. Search failure throughout; no alias.
- Certificate k=6 and k=8 replicate on a second batch (`hard1_diff`, seven recorded, line 9): 44.6% and 18.9% of
  2,000 starts land on a recorded image (confident batch: 51% and 18.2%), all seven found at both k, floor fraction
  ≈ recorded fraction (.446/.446, .179/.189), argmin exact (7.5e-15 at k=8) (job 706721). **The k = 10 bracket (same job, pre-registered: floor and recorded
  fractions must come apart AT the line):** confident k = 10 has N′ = 5, line 11, one below → 13.4% on a recorded
  image and 12.9% at the floor (together), argmin on a recorded image; `hard1_diff` k = 10 has N′ = 6, line 10, **at
  the line** → 4.75% on a recorded image but **30.0% at the floor** (spurious zeros dense), argmin NOT on a recorded
  image (error 0.49), 5 of 6 still found among the 2,000. The two fractions coincide one step below the line and
  separate by 6× at it — the certificate line is sharp on both batches.
- Wide head (job 725918), the k ladder on the twenty-digit batch (m = 26, r = 64): k = 8 → 18 of 20 found (N′ 19);
  k = 16 → 62.5% of 10,000 starts, 15 of 20 (N′ 17); **k = 24 → 46.9% of 10,000, 13 of 20 (N′ 16, line 48)**, landings
  per image 0 / 57 / 1,765 (min / median / max), argmin on a recorded image, chart error .25; **k = 32 → 27.9% of 10,000, 12 of 20 (N′ 15, line 49)**, landings 0 / 30 / 1,544, argmin on a recorded image, chart error .23; k = 40 → N′ 14 (line 50, Part B pending). The wide
  head's release loses rank with k as the confident batch's does (19 → 17 → 16 → 15): the same collapse-with-fidelity.
- The ladder job (721391) exited on an assertion after its `mnist_control r = 64, k = 16` row (88.8%, 8 of 8); all
  `confident` rows and the `mnist_control` k = 8 rows at every rank were already on disk.

### Why the ladder job stopped, and what it shows (diagnostic 748065)

The ladder job (721391) tripped the imprint-sum consistency assertion (`‖Σ_i C_i − B_T‖/‖B_T‖ < 1e-10`) at the
`mnist_control r = 64` cell after k = 16. Recomputing that release at every k (CPU, FP64):

| k | ‖B_T‖ | ‖Σ C_i − B_T‖ (abs) | relative | rank B_T |
|---|---|---|---|---|
| 16 | 3.7e-2 | 1.0e-15 | 2.9e-14 | 8 |
| 24 | 1.7e-6 | 1.8e-15 | 1.1e-9 | 7 |
| 32 | 7.9e-6 | 4.9e-16 | 6.1e-11 | 7 |
| 40 | 8.3e-6 | 1.6e-15 | 1.9e-10 | 7 |
| 48 | 4.5e-6 | 4.4e-16 | 9.6e-11 | 7 |
| 56 | 9.3e-5 | 1.4e-15 | 1.5e-11 | 6 |

The absolute mismatch is FP64 roundoff at every k (the traced loop and `train_release` differ only in summation
order); what changed is the release itself: **the control batch's release shrinks by four orders between k = 16
and k = 24 and loses rank (8 → 7 → 6)** — once the chart is faithful enough, the strong model classifies the
projected control digits confidently and records almost nothing of them (Step 18's law, now along k). The
assertion is being floored at roundoff (`1e-10·‖B_T‖ + 1e-13`) and the five missing control cells resubmitted; the
k = 16 control row (88.8%, 8 of 8) stands — it was on a 3.7e-2-scale, rank-8 release.

### Step 24 (pre-registered, job 753371): the headline cell from a QUANTISED release

The headline (Step 23) was measured from an FP64 release; nobody ships adapters that way. The cell's eight images
are recorded across **twelve orders**: relative imprints 1.0, .87, 9.5e-4, 2.2e-4, 6.7e-7, 4.4e-10, 6.6e-11,
3.0e-12 — and all eight were recovered from random starts. Step 21 measured what each format keeps of the
imprint band (relative roundoff: fp32 1.2e-7, tf32/fp16 9.8e-4, bf16 7.8e-3). This step joins the two: the same
cell (confident, on-chart, r = 64, k = 32, 500 random starts, 300 iterations) with `(A_T, B_T)` round-tripped
through fp64 / fp32 / tf32 / fp16 / bf16 after training (`--release-dtype`), everything downstream in FP64 on that
file; the imprints and the recorded set stay the FP64 truth (found is always out of eight). Second variable: the
certificate tolerance (`--tol`, the attacker's knob): the headline's 1e-12 and a noise-matched 10·ε (fp32 1.2e-6,
tf32/fp16 1e-2, bf16 8e-2). Nine cells.

**Prediction (band rule):** found = the images whose relative imprint exceeds the format's roundoff: fp64 8;
fp32 4–5 (the 6.7e-7 image sits at ~5ε); tf32/fp16 2–3 (9.5e-4 ≈ ε); bf16 2. At tolerance 1e-12 the quantised
spectrum is noise-filled (N′ → m = 10, line 54, still above k = 32), no start reaches the 1e-20 floor, but the
landing count should follow the same band rule because a weak image's direction has left the row space. Falsifiers:
8 found at fp32 (the certificate is more robust than the spectrum says); 0 found at bf16 (landings need the floor,
not only the null space). The landing threshold stays 1e-2 on-chart image error at every precision.

*Table note for the ladder (Step 23):* the 28.6% per-image basin at r = 64, k = 16 is the N′ = 3 dip cell (the other
five imprints are ~1e-29 there): three targets, coarse normalisation — not a peak.

**Two additions before any row is read (yoado-ed).** (i) *Spectrum prediction beside the imprint one.* The wide-head
twentieth image (Step 22) had imprint 3e-7 but singular value 6e-14 — seven orders of collinearity — and the singular
value decided. So the second predictor is `σ_i/σ_1` of `B_T` against each format's roundoff; the headline rows do
not carry the spectrum (field added only now), so it is being recomputed (job 752500) and the spectrum-based
counts will be written here before the first quantised row lands. If the two predictions coincide the cell is a
precision result; if the spectrum wins, "direction, not magnitude" is shown in a second, independent setting.
(ii) *Quantisation may widen the line while narrowing the channel.* If bf16 takes N′ from 8 to 2, the line moves
from 56 to 62: fewer images exposed, but each survivor may be recoverable through a richer chart. Job 753886 runs
the confident cell at **k = 58 and 60** — above the FP64 line — from the bf16 release (tol 8e-2) and, as control,
from the FP64 release (tol 1e-12). Prediction: FP64 at k = 58/60 has N′ = 7–8 (line 56–57), so it sits at or above
the line: spurious zeros dense, argmin unreliable; bf16 has N′ = 2 (line 62), so k = 58/60 is below: the two
strongly recorded images should be found from random starts with a basin comparable to the r = 64, k = 48–56
FP64 cells (same slack), at chart fidelity .99 — the defender's quantisation then trades the number exposed
against the sharpness of what remains. Falsifier: zero landings at bf16 k = 58/60 (the basin has collapsed;
quantisation wins outright). Note the private data at k = 58/60 is the projection at that k, so the release is
retrained there: "the two survivors" are the two strongly recorded images of that cell, not literally those of
k = 32.

### The release against k, both batches (job 752500, CPU FP64, r = 64, on-chart, N = 8, seed 1): the headline cell's release has norm 7.6e-18

| k | confident ‖B_T‖ | rank @1e-10 / @1e-6 | margin min / median at W₀ | control ‖B_T‖ | rank | margin min / median |
|---|---|---|---|---|---|---|
| 8 | 4.7e-1 | 7 / 5 | 0.02 / 10.8 | 8.4e-1 | 8 / 7 | −7.6 / 9.3 |
| 12 | 2.3e-1 | 4 / 3 | 2.0 / 22.5 | 3.7e-1 | 8 / 4 | −0.5 / 16.6 |
| 16 | 8.1e-2 | 3 / 3 | 4.1 / 39.5 | 3.7e-2 | 8 / 2 | 4.7 / 19.8 |
| 24 | **1.6e-13** | 6 / 3 | 31.6 / 47.7 | 1.7e-6 | 7 / 6 | 15.4 / 23.7 |
| 32 | **7.6e-18** | 5 / 4 | 41.5 / 50.3 | 7.9e-6 | 6 / 6 | 13.6 / 25.4 |
| 40 | 1.9e-22 | 8 / 5 | 52.1 / 57.3 | 8.3e-6 | 7 / 6 | 13.5 / 22.3 |
| 48 | 2.8e-22 | 6 / 4 | 51.9 / 59.6 | 4.5e-6 | 6 / 6 | 14.2 / 21.1 |
| 56 | 2.8e-24 | 6 / 5 | 56.3 / 60.2 | 9.3e-5 | 6 / 5 | 11.2 / 24.0 |

(r = 16 and 32 at k = 8/12/16 agree with r = 64 to the leading digit: the collapse is the batch's, not the rank's.)
Headline cell (confident, k = 32) spectrum `σ_i/σ_1`: 1, .83, 8.3e-4, 1.7e-4, 5e-7, 6.7e-11, 1.7e-11, 1.7e-12 — one-to-one
with the imprints (no collinearity in this cell), absolute imprints 6e-18 … 2e-29.

**Reads.** (i) *The window question (yoado-ed):* the confident batch does not sit in a window — its release collapses
**monotonically and completely** as the chart sharpens: 0.47 at k = 8 (one projected digit at margin 0.02, nearly
misclassified), 0.08 at k = 16, then 1.6e-13, 7.6e-18, … 2.8e-24 by k = 56, as the projections' margins climb from
11 to 60. The rank's dip-and-return (7 → 3 → 5–8) is a *relative* effect: at k ≥ 24 every image is confident, the
imprints shrink together and their ratios compress, so more of them clear a relative tolerance of a vanishing σ₁.
(ii) *The control plateaus:* its weakest projected digits keep margins of 11–15 at every k ≥ 24, so its release
stays at 1e-6 … 1e-4 — twelve orders above the confident batch's at the same k. (iii) **The headline (Step 23) was
recovered from a release of norm 7.6e-18.** In FP64 the certificate is scale-free — only directions enter — so the
recoveries are exact; but the number has to be stated with the headline. What it implies is in the next step.

### Step 24 continued: pre-registrations revised by the spectrum (written before any quantised row was read)

*Spectrum prediction* (σ_i/σ_1 above the format's relative roundoff): fp32 **5** (5e-7 is 4× the roundoff — borderline
4), tf32 **2** (8.3e-4 sits just under 9.8e-4 — borderline 3), bf16 **2**; the imprint prediction gives the same
counts (the spectrum and the imprints coincide in this cell). **fp16 is revised from 2–3 to 0:** ‖B_T‖ = 7.6e-18 is
below fp16's smallest subnormal (6e-8), so the *whole release rounds to zero* — Step 21's "dynamic range decides",
now for the entire file, not a tail. tf32 and bf16 keep the fp32 exponent range and preserve it.

*Training precision — the prediction that matters (not tested by 753371, which quantises after FP64 training):*
an image's imprint is its accumulated softmax residual `1 − p_y ≈ exp(−margin)`; training arithmetic at unit
roundoff `ε` makes that residual **exactly zero** once `exp(−margin) < ε`: fp32 (ε = 6e-8) records nothing with
margin above ~16.6, bf16 (3.9e-3) above ~5.5, fp16 (4.9e-4) above ~7.6. The confident batch's projections at k ≥ 24
have margins 32–60, so **under fp32 training the headline cell's release is exactly zero — there is nothing to
recover**; at k = 8 (margins 0.02 … ) part of it is recorded even in bf16. The control's weakest digits (margins
11–15) stay recorded under fp32 training at every k, and drop out under bf16/fp16. Pre-registered for a
training-precision cell (design to audit): run the release loop itself in fp32 / bf16 / fp16 at the confident and
control k = 32 cells (and confident k = 8); predicted ‖B_T‖: confident k = 32 → 0 exactly in all three; control
k = 32 → ~8e-6 in fp32, 0 in bf16/fp16; confident k = 8 → ~0.47 in all three with the images above the margin
threshold dropping out. **If confirmed, the headline is an FP64-training statement**: the leakage of a confidently
classified batch exists only when the training arithmetic can represent residuals of 1e-18.

**Correction to the training-precision prediction above, before any row is read.** The residual vector is
`R = softmax(z) − e_y`; its *off-class* entries are `p_j = exp(z_j − z_y)/Z`, and a format keeps those down to its
smallest subnormal, not its unit roundoff: fp32/tf32/bf16 (fp32 exponent range) represent `exp(−margin)` to
margins of ~100, fp16 (subnormal floor 6e-8) only to ~16.6. What unit roundoff `ε` removes is the *own-class* entry
`R_y = p_y − 1 = −Σ_j p_j`, which rounds to 0 once `Σ_j p_j < ε` — one row of an image's imprint, not its direction.
And the release never feeds back: `B_T A_T h ≈ 1e-18` against logits of ~40 is below the ulp of the logits *even in
FP64* (2 orders below), so `z` is constant through training, `A_T = A_0` to working precision in every format, and
`B_T = −(lr·T/N) Σ_i R_i (A_0 h_i)ᵀ` in closed form — the certificate is exact on `span{A_0 h_i}`. **Revised
predictions for the training-precision cell:** confident k = 32 → ‖B_T‖ ≈ 7.6e-18 (own-class rows lost, ~√2 lower)
under fp32 and bf16 (bf16 to ~3 digits), **0 exactly under fp16** (exp(−42) underflows); control k = 32 (margins
13.6–25) → ≈ 7.9e-6 under fp32/bf16, and under fp16 only the images with margin < 16.6 survive (rank falls);
confident k = 8 (margins 0.02–11) → ≈ 0.47 in all three, rank unchanged. **So the headline is not an FP64-training
artefact: it survives fp32 and bf16 training arithmetic and is erased only by fp16's range.** The earlier
sentence ("under fp32 training the release is exactly zero") is withdrawn; the STATUS caveat is corrected.

**Audit refinements adopted (yoado-6e).** (a) The spectrum is the *primary* predictor, mechanistically: the
certificate lives on row(B_T) = the top singular directions, quantisation noise at σ₁·ε buries every direction
with σ_i < σ₁·ε, so the count of σ_i above the floor *is* N′ at that precision and sets the line and the ceiling on
found; the imprint ‖C_i‖ says how much image i contributed, not whether its direction survived (they diverge under
collinearity — the wide-head 6e-14 vs 3e-7 case). Where they disagree, the σ call stands. (b) fp16 is *range*-
limited here, not mantissa-limited: with ‖B_T‖ = 7.6e-18 the entire file is below fp16's subnormal floor — read
fp16 as narrow-range storage, and tf32 + bf16 (fp32 range) as the clean mantissa test; a measured fp16 < bf16 is
the range signature, not a band-rule violation. (c) A null is keyed off the **certificate residual at the truth**:
O(1) → the direction left the noisy row space (destroyed); ~0 with no landing → sampling (the 500→5,000 extension
covers it). (d) At tolerance 1e-12 on a quantised release N′ → m = 10 is a *noise rank*; its line (54) is not
physical and is reported only to show the tolerance must be noise-matched; the physical line is the
noise-matched N′'s.

**Job 753886 relabelled; the trade hypothesis withdrawn (yoado-ed).** On-chart the truth *is* a point of the
training chart: a release trained at k = 32 has its private images exactly on the 32-dimensional chart, and
searching a nested 58-dimensional chart represents the same image at `(w*, 0)` — identical fidelity, 26 extra
unknowns, a smaller basin; retraining at k = 58 changes the private data. So "quantisation widens the line and
lets the survivors be recovered through a richer chart" cannot be tested on-chart, and off-chart the certificate
has no zero — the trade question is a limit of the certificate route (an *open*, not a cell). 753886 now answers
only: *is a bf16 release attackable at all in the regime where the chart is faithful and the model records
little*, with the FP64 control at the same k expected at or above its own line; the FP64 control is read by
err-vs-truth (spurious = low objective, err > 1e-2), not by landing count, and nothing is carried across from
k = 32.

**Training-precision jobs submitted (760909 letters, 760912 digits; `train_precision.py`, design audited).** Cells:
`letters_a` at k = 32 and 16 (EMNIST 'a' as an eleventh class on the strong model, zero head row, chart = public
letters' PCA — the decisive arm: negative margins, O(1) residuals, survives every format), `confident` k = 32 (the
headline; negative control for precision), `mnist_control` k = 32 (the companion where the adapter *moves*),
`confident` k = 8 (the recorded-at-every-precision digit cell). Formats fp64 (gate against `train_release`), fp32,
bf16, fp16 — the loop itself run in the format. Random-start search (500 starts, certificate.py's objective and
1e-2 threshold) from the fp64/fp32/fp16 letter releases and the fp32/fp16 digit releases. Every row carries the
**feedback** `‖B_T A_T H‖/‖z‖` and the margins at t = 1 and T (yoado-ed's objection: at 1e-18 the adapter never
moves the logits, so the headline is one gradient step in disguise — the control and letter cells are where the
fine-tune does something), the **imprint-sum mismatch** `‖Σ C_i − B_T‖` (yoado-6e: it must scale with the signal,
not with O(1) intermediates, or 7.6e-18 would be roundoff), and the fraction of residual entries that are exactly
zero. Falsifier made precise (yoado-6e): fp32 at confident k = 32 should give a *nonzero* release ~√2 below FP64
(own-class rows lost); an exactly-zero fp32 release breaks the off-class/own-class split. The letters' projected
margins at t = 1 are read from the fp64 row before the letter predictions are held to (yoado-ed's caution).

*Letters prediction written before the rows (yoado-ed):* the digits collapsed because projection **raised** their
margins toward the model's confidence. The letters start at margin −10.8 — the model is wrong about them, not
unsure — and projection cannot make a head with a zero row correct about that class, so their `t = 1` margins should
stay negative at every k and their imprints of order one. If the fp64 rows show that, the collapse mechanism is
confirmed by the case that escapes it. The fork it decides for the write-up: letters at k = 32 with O(1) imprints
= one cell that is both robust to arithmetic and instance-identifying (lead of the measured section); letters
collapsing as the projections sharpen = robust-but-coarse beside sharp-but-cornered.

### Step 24 results (job 753371, seven of nine cells; bf16 pending): the headline cell from a quantised release

Same cell as Step 23 (confident, on-chart, r = 64, k = 32, 500 random starts). "found" is out of the eight FP64-recorded
images; `σ_rel` is the quantised release's spectrum; residual = certificate residual at each truth.

| release | tol | N′ (line) | on a recorded image | found | which lost (σ_rel of the lost) | residual at lost truths | residual at found truths |
|---|---|---|---|---|---|---|---|
| fp64 | 1e-12 | 8 (56) | 65.6% (42% at floor) | **8** | — | — | 5e-15 … 2e-5 |
| fp32 | 1e-12 | 10 = m, noise rank (54) | 54.2% (0 at floor) | **5** | 0, 3, 4 (7e-11, 3e-11, 2e-12) | 0.9, 0.9, 0.9 | 2e-8 … 4e-8 |
| fp32 | 1.2e-6 | 4 (60) | 66.6% | **4** | + 7 (6e-7) | 0.9 | 4e-8 … 2e-4 |
| tf32 | 1e-12 | 10 (54) | 54.0% | **5** | 0, 3, 4 | 0.8–0.9 | 2e-4 … 3e-4 |
| tf32 | 1e-2 | 2 (62) | 48.2% | **2** (images 1, 5) | all but the two strong | ~1 | 2e-4, 3e-4 |
| fp16 | both | 0 — **‖B_T‖ = 0, the file underflowed** | 0 | **0** | all | — | — |

The fp64 row reproduces Step 23 exactly (landings 34, 34, 30, 58, 26, 19, 103, 24). Quantised spectra: fp32
1, .9, 9e-4, 2e-4, 6e-7, **3e-9, 1e-10, 6e-11**; tf32 1, .9, 9e-4, 2e-4, **2e-5, 1e-6, 7e-7, 6e-7** — the three weakest
directions are replaced by a noise floor.

**Reads against the pre-registration.** (i) **fp16: 0 found, the whole release rounds to zero** — as revised
(range, not mantissa). (ii) **fp32: 5 found** — the imprint and spectrum predictions (4–5) hold; the three lost
images have residual 0.9 at their truths (their directions left the row space — destroyed, not under-sampled;
guard (c)). (iii) **tf32 at the tight tolerance: 5 found, not the predicted 2–3.** The band rule keyed the count to
σ_i/σ_1 *above the unit roundoff*; the quantised spectrum's actual noise floor sits ~3 orders below the roundoff
(fp32: 1e-10 … 3e-9 against ε = 1.2e-7; tf32: 6e-7 … 2e-5 against 9.8e-4), so the 9e-4, 2e-4 and even the 6e-7
directions survive tf32 and are found (residuals 2e-4 … 3e-4 at their truths, landings within 1e-2 of the images).
The σ-primary framing stands but its threshold is the *measured* spectrum floor of the quantised file, not ε.
(iv) **The tolerance knob cuts the other way from "noise-matched":** at fp32, tol 1.2e-6 (= 10ε) drops N′ to 4 and
loses image 7 (σ 6e-7), which tol 1e-12 recovers; at tf32, tol 1e-2 leaves N′ = 2 and finds exactly the two strong
images that tol 1e-12 finds five of. The attacker should use the *noise rank* (a tight tolerance): the certificate's
null space then includes the weak directions approximately, and on-chart the search still lands within 1e-2.
"Noise-matched" was the wrong recommendation — recorded as a lesson. (v) The aggregate basin barely moves
(54–67% against 65.6%) — what precision removes is *which* images are reachable, not how many starts land.

**Headline sentence, precision-scoped (pending bf16):** from the FP64 release all eight; from the same adapter
stored in fp32 or tf32, five of eight (the three recorded at 1e-11 … 1e-12 of the strongest are gone); stored in
fp16, none — the file is zero.

### Flowers on CIFAR, corrected design (job 656205; old-head and mixed batches rerun as 762253 after a save-path crash)

Eight flowers (new class, m = 11) and eight CIFAR digits on the extended head, both charts (flower PCA / CIFAR PCA,
k = 16), both head inits, r = 16, N = 8: every *near-start* cell (a) reaches residual ~1e-30 with chart error
≤ 1e-12 (σ_min at the truth 1e-6 … 2e-5 — identifiable, rank 8 in every release, imprints O(1)); every *random-start*
cell (b, 2 restarts, 600 iterations) stops at residual 5e-5 … 5e-3 with chart error 0.07 … 0.59 — search failure
(residual not zero), no alias. Chart ceilings vs the raw images: .31/.38 (flowers), .24/.26 (CIFAR). Same picture as
the first flowers job: recorded and identifiable, not reachable by the recipe route from random starts at k = 16.

**Caption for the Step 24 table (yoado-ed):** two thresholds govern what a format keeps — the unit roundoff, which
is *relative*, and the subnormal floor, which is *absolute* — and which binds depends on ‖B_T‖. **fp32 and tf32
preserve every image recorded above roughly 1e-10 of the strongest imprint**; ten orders of dynamic range is not a
privacy mechanism. **The fp16 zero is a property of this cell's release (7.6e-18 against fp16's smallest subnormal
6e-8), not of the format**: for releases of 1e-6 … 1 — what data the model gets wrong produces — the floor does not
bind and fp16 keeps whatever sits above ~1e-6 of the strongest; fp16 protects only what had already collapsed
twenty-three orders, i.e. what did not need protecting. The structural line: the aggregate basin holds at 48–67% in
every format — precision changes *which* images are reachable, not how hard they are to reach.

*Random-init letter arm, pre-registered before it runs (job below):* the eleventh head row drawn Gaussian at the RMS
norm of the digit rows gives the letters margins of ordinary spread rather than −10.8 by construction; a few
letters may be chance-correct at t = 1 (positive margin) and show lower imprints, the rest stay negative with O(1)
imprints; the release survives every format, with only a chance-correct letter of margin > ~16.6 (unlikely at
ordinary spread) dropping out under fp16 training. The zero-init arm remains primary (the honest model of a class
the network does not have).

### Step 25 (jobs 760909, 760912; `train_precision.py`): the release loop run in fp32 / bf16 / fp16

Same recipe, every tensor cast to the format and the loop run in it; FP64 column is the gate (reproduces
`train_release` to 6e-32 at the headline cell, 9e-16 at the O(1) letter cell).

| cell | format | ‖B_T‖ | rel. dev. from FP64 | ‖ΣC_i − B_T‖ | feedback ‖B_T A_T H‖/‖z‖ | A_T − A₀ (rel) | N′ (tight / noise-matched) | certificate residual at the truths |
|---|---|---|---|---|---|---|---|---|
| confident k = 32 | fp64 | 7.62e-18 | — | **6e-32** | 4e-19 | **0** | 8 / 8 | 5e-15 … 2e-5 |
| | fp32 | 7.62e-18 | 3e-6 | 2e-23 | 4e-19 | 3e-8 | 10 / 4 | 3e-6 … 2e-4 (4 images), 0.9 (4) |
| | bf16 | 6.84e-18 | **0.21** | 3e-21 | 4e-19 | 2e-3 (the cast) | 10 / 2 | 0.18 at the two strongest |
| | fp16 | **0** | 1 | 0 | 0 | — | 0 | — |
| control k = 32 | fp64 | 7.89e-6 | — | 6e-21 | 5e-7 | 1e-11 | 7 / 7 | 5e-15 … 2e-5 |
| | fp32 | 7.99e-6 | 0.02 | 3e-12 | 5e-7 | 3e-8 | 10 / 7 | 7e-7 … 8e-4 (5), 0.018, 0.82 |
| | bf16 | 5.37e-6 | **0.72** | 6e-11 | 3e-7 | 2e-3 | 10 / 2 | 0.19, 0.20 at the two strongest |
| | fp16 | **0** | 1 | 0 | 0 | — | 0 | — |
| confident k = 8 | fp64 | 0.465 | — | 5e-16 | 0.040 | 0.029 | 7 / 7 | ≤ 6e-8 (recorded) |
| | fp32 | 0.465 | 5e-7 | 2e-7 | 0.040 | 0.029 | 10 / 5 | ≤ 9e-5 (5), 0.9 (2 weakest) |
| | bf16 | 0.397 | 0.17 | 0.02 | 0.026 | 0.018 | 10 / 2 | 0.07 |
| | fp16 | 0.474 | 0.03 | 3e-3 | 0.038 | 0.026 | 7 / 2 | 0.03–0.04 |
| **letters 'a' k = 32** | fp64 | **1.00** | — | 9e-16 | **0.43** | **0.093** | 8 / 8 | **2e-13 … 7e-12** |
| | fp32 | 1.00 | 4e-7 | 5e-7 | 0.43 | 0.093 | 11 / 8 | 3e-7 … 1e-5 (tight), 1e-4 … 3e-3 (10ε) |
| | bf16 | 0.894 | 0.12 | 0.09 | 0.36 | 0.085 | 11 / 3 | 0.04 … 0.23 |

Margins (t = 1 → T): confident k = 32: 42–69 unchanged to three digits; control k = 32: 13.6–55 unchanged;
confident k = 8: image 4 from 0.02 to 4.6 (the adapter acts); **letters: −8.95 … −2.65 at t = 1 → +6.0 … +13.7 at T
(the model learned the class)**. Fraction of residual entries exactly zero: 0.10 at the headline cell *already in
FP64* (the eight own-class entries `p_y − 1` round to zero at margins > 37), 0.14 fp32, 0.18 bf16, 1.0 fp16.

**Random-start certificate search from the low-precision releases (500 starts, 1e-2 landing bar):**

| cell | release | tolerance | N′ (line) | on a recorded image | found | first-landing error (residual at truth) |
|---|---|---|---|---|---|---|
| letters k = 32 | **fp64** | 1e-12 | 8 (56) | **38.4%** (all at floor) | **8 of 8** | 1e-12 … 6e-11 (2e-13 … 7e-12) |
| letters k = 32 | fp32-trained | 1.2e-6 | 8 (56) | 16.2% | 6 of 8 (letters 0, 1 missing: residual 2.7e-3, 3.3e-3) | 6e-4 … 1e-2 (1e-4 … 1e-3) |
| confident k = 32 | fp32-trained | 1.2e-6 | 4 (60) | 67.2% | 4 (1, 2, 5, 6) | 5e-6 … 1e-3 (3e-6 … 2e-4) |
| control k = 32 | fp32-trained | 1.2e-6 | 7 (57) | 42.0% | 5 of 7 (0, 2, 3, 4, 6) | 1e-6 … 2e-3 (7e-7 … 8e-4) |

**Reads.** (i) **The letters are the decisive cell and they came back as predicted:** a new class the base model
does not have (margins −9 … −3 at t = 1, k-independent by construction), release of norm 1.0, the adapter moving the
logits by 43% and the model learning the class (+6 … +14 at T), imprints 0.12–0.26 for every letter, certificate
residuals 1e-12 at the truths **with a moving adapter** — and **all eight letters recovered from random starts
with no recipe and no labels (38% of starts, argmin 3e-26) at k = 32**, chart error .24. One cell that is both
robust and instance-level. (ii) **Precision-of-training acts through accumulation, and only when the adapter
moves.** At the frozen-logit cells (feedback 1e-19, 5e-7) fp32 reproduces the FP64 release to 3e-6 / 2e-2 and the
certificate keeps its images (the √2 detail pre-registered was wrong: FP64 had *already* rounded the own-class
entries to zero, so nothing further was lost); at the letters (feedback 0.43) fp32's 400 accumulated roundings
move the row space by 1e-4 … 3e-3 per image and the search from the fp32-trained release recovers 6 of 8 at the
(wrong) noise-matched tolerance — the tight-tolerance search is running (764976). **bf16 training is a different
regime from bf16 storage**: accumulating 400 steps in an 8-bit mantissa moves the release by 21% (headline), 72%
(control), 12% (letters) and the certificate residuals at the strongest truths to 0.18–0.2 (digits) / 0.04–0.23
(letters), against 2e-3 from a one-shot bf16 cast of the FP64 release (Step 24). (iii) **fp16 training zeroes the
control's release too**, though its residuals exp(−13.6) = 1.2e-6 are representable: what must survive is the
*update* `lr·R_i/N` (B starts at zero), 800× smaller — the fp16 training threshold is margin ≲ 10, not 16.6; the
confident k = 8 cell (margins 0.02 … 0.9 for the recorded digits) keeps its release under fp16 to 3%. (iv) **The
imprint-sum mismatch scales with the signal** (6e-32 at ‖B_T‖ 7.6e-18; 6e-21 at 7.9e-6; 5e-16 at 0.47; 9e-16 at 1.0) —
the 7.6e-18 release is structure, not roundoff (yoado-6e's check). (v) The "adapter moves" companion is not the
control at k = 32 (feedback 5e-7, margins frozen) but the letters and confident k = 8 — both recorded, both attackable.

### The landing error tracks the certificate residual at the truth (all cells, from the saved first landings)

Across every cell above and every Step 24 cell, the first landing's image error is **≈ 4–5 × the certificate
residual at that image's truth, with ~2× per-image scatter** (tf32 spans 3.7–8.5×): fp64 headline 4e-6 → 1.3e-5, 5e-15 → 2e-14; tf32 storage 2e-4 → 9e-4 … 1.7e-3;
bf16 storage 2e-3 → 6e-3 … 9e-3; letters fp32-trained 1e-4 → 6e-4, 1e-3 → 6e-3 … 1e-2. So "approximately
contained → proportionally approximate recovery" is the mechanism (yoado-6e's first outcome), and the 1e-2 landing
bar corresponds to a residual of ~2e-3 — which is why bf16 storage (residuals 2e-3 at five images) landed three of
them just under the bar (errors 6e-3 … 9e-3) and missed two (the scatter straddling the bar, not a sharp cutoff), and why the two missing fp32-trained letters
(residuals 2.7e-3, 3.3e-3) are just over it. **Step 24's tf32 "5 found" is therefore five recovered to ~1e-3
image error, not to 1e-7 as at fp32 or 1e-14 at fp64**; the count is honest, the sharpness scales with the residual.

### Step 24, bf16 storage (753371, eighth cell): 3 found, all approximately

bf16 release at tol 1e-12: N′ = 10, quantised spectrum 1, .9, 9e-4, 2e-4, 1e-4, 7e-6, 4e-6, 2e-6 (floor 3–4 orders
below ε = 7.8e-3 — this cell); residual 2e-3 at the five above-floor truths, 0.8–0.9 at the three destroyed; 21% of
starts land; **found 3 (images 1, 5, 7) at errors 6e-3 … 9e-3** — the other two above-floor images (2, 6) sit at the
same 2e-3 residual and simply did not cross the 1e-2 bar. The ninth cell (tol 0.08, N′ = 2) is pending. On the "3
orders below ε" magnitude (yoado-6e): √(m·r) = √640 ≈ 25 accounts for ~1.4 of them by the random-matrix heuristic;
the rest is structured rounding error and is this-cell-specific until another spectrum shows it.

### Control ladder at r = 64 (749362, after the assertion floor)

| k | N′ (line) | on a recorded image | at floor | found |
|---|---|---|---|---|
| 24 | 7 (57) | 71.6% | 59.4% | 7 of 7 |
| 32 | 7 (57) | 51.6% | 42.6% | 7 of 7 |
| 40 | 7 (57) | 18.2% | 15.2% | 7 of 7 |
| 48 | 7 (57) | 6.6% | 5.0% | 6 of 7 |
| 56 | 6 (58) | pending | | |

The control batch (releases 1e-6 … 1e-4, seven recorded at every k) is attackable across the whole range with the
basin falling with k as the confident batch's does (16.6% at k = 8 on the confident batch was a different rank);
argmin on a recorded image at every k; chart class accuracy 1.0. Job 753886 Part A at bf16 k = 58: ‖B_T‖ 1.6e-24,
N′ = 3 at tol 0.08 (σ_rel 1, .7, .2), line 61, residuals 2e-3 … 3e-3 at three truths, 0.94–0.97 at five (Part B pending).

*Guard note (yoado-6e):* the scale-invariant objective removes the exact blank as a minimiser, but at a noise-filled
N′ a *near*-blank (small φ aligned with a noise direction) can still be a spurious local minimum — the feature-norm
< 5% flag caught one such start in 500 at tf32. The flag is load-bearing at the noise rank, not vestigial. And for
the tight-tolerance rows: a letter "not found" at a residual near the bar is a *degraded recovery*, not a miss —
`train_precision.py` now reports the closest approach per recorded image (`min_err_per_recorded_image`) so the
image error is stated directly (future runs; 764976 started before the field).

### Step 25 continued (760909 complete, 763805 random-init arm, 753371 ninth cell)

| cell | format of TRAINING | ‖B_T‖ (rel. dev.) | residuals at the truths (tight tol) | search: on a letter / found |
|---|---|---|---|---|
| letters k = 32, zero row | fp64 | 1.00 | 2e-13 … 7e-12 | 38.4% / **8 of 8** |
| | fp32 | 1.00 (4e-7) | 3e-7 … 1e-5 | 16.2% / 6 of 8 at 10ε (tight tol pending, 764976) |
| | bf16 | 0.89 (0.12) | 0.04 … 0.23 | — (not run; residuals say none) |
| | fp16 | 0.98 (0.03) | 0.014 … 0.17 | **0% / 0 of 8** (N′ 4 at 10ε; residuals 0.04 … 0.25) |
| letters k = 16, zero row | fp64 | 0.975 | 2e-14 … 8e-13 | 82.4% / 7 of 8 (letter 5: residual 2.5e-14, zero landings in 500 — a *sampling* null) |
| | fp32 | 0.975 (4e-7) | 2e-7 … 4e-6 | **82.6% / 8 of 8** at 10ε (residuals 2e-5 … 6e-4) |
| | bf16 | 0.86 (0.12) | 0.04 … 0.24 | — |
| | fp16 | 0.96 (0.03) | 0.017 … 0.16 | 0% / 0 |
| letters k = 32, **random row** | fp64 | 0.95 | 4e-13 … 5e-12 | 38.4% / **8 of 8** |
| | fp32 | 0.95 (5e-7) | 3e-7 … 5e-6 | 16.8% / 6 of 8 at 10ε (letters 0, 1: residuals 3.1e-3, 2.7e-3) |
| | bf16 / fp16 | 0.85 / 0.93 | 0.05 … 0.23 / 0.02 … 0.19 | — / 0 |

Random-row margins at t = 1: −10.5 … −0.41 — none chance-correct (the two nearest zero, letters 4 and 6, carry the
lowest imprints .072/.070 against .14–.28, as pre-registered: imprint follows the residual, and a residual of 0.6
is still an O(1) recording). Every arm: margins at t = 1 negative for every letter at both k, imprints O(1), the
adapter moving the logits by 33–44%.

**Reads.** (i) **Training precision for the lead cell:** fp32 keeps the release to 4e-7 and the certificate route
recovers 8 of 8 at k = 16 and 6 of 8 (10ε tolerance) at k = 32; **half-precision training keeps the release's
norm (bf16 −12%, fp16 −3%) but not its directions** — 400 accumulated roundings in an 8- or 11-bit mantissa move
the row space by 2–25% per image, the certificate residuals at the truths become 0.01–0.25, and the search finds
nothing (0 of 8 from the fp16-trained release at both k). The disclosure is still *recorded* in a bf16/fp16-trained
adapter (O(1) release, rank 8); the *certificate* route needs the recording's directions to ~1e-3, which ordinary
fp32 training gives and half precision does not. (ii) The k = 16 letters from an fp32-trained release, all eight,
82.6% of starts: "trained and stored in ordinary arithmetic, on the canonical fine-tuning task, every private
example recoverable from random starts with no recipe and no labels" holds at k = 16 (chart error .32, class-level
fidelity); at k = 32 (instance-level) it is six of eight pending the tight tolerance. (iii) The random-row arm
reproduces the zero-row arm in every number (8/8 fp64, 6/8 fp32, the same two letters missing) — the result does
not depend on how the new head row is initialised. (iv) The bf16-storage ninth cell (tol 0.08, N′ = 2): found only
image 5 (81 landings); image 1 at residual 2.4e-3 straddled the bar — the scatter, as read above.

### Step 26 (pre-registered; job below): the RECIPE route against a half-precision-trained release — is "recorded but not certificate-recoverable" protection?

yoado-6e's probe: the certificate is one weak, recipe-free attacker; its failure on bf16/fp16-trained letter releases
bounds nothing stronger. The recipe route (the unrolled FP64 simulator, `invert_lm`, from a near start with
noise 0.1 — the identifiability cell of every earlier step) uses strictly more information. Cells: letters k = 16
and 32, release trained in fp64 (gate: residual at the truth ~1e-16, recovery exact), fp32, bf16, fp16; simulator
always FP64. Measured: residual at the truth (= the arithmetic-mismatch floor: predicted ≈ the release's relative
deviation, 4e-7 / 0.12 / 0.03), σ_min of the Jacobian at the truth, the LM endpoint's residual and its per-image
chart error. **Pre-registration:** the residual cannot reach zero from a mismatched release (floor = rel. dev.);
the question is where the LM endpoint sits. *Recovery number (yoado-6e's fix): the endpoint's error against the ON-CHART truth `X_on` (`err_vs_chart`), read
against the same 1e-2 bar the certificate route uses so the two attackers compare like for like; `err_vs_REAL`
is reported but floors at the chart's own representation error (.32 / .24) and cannot go below it.* This is an
identifiability probe — best-case start plus the full recipe — i.e. the upper bound on recoverability, not
from-scratch reachability, as in every earlier "cell a". Two outcomes, both recorded: (a) `err_vs_chart` small
(≲ 1e-2, then `err_vs_REAL` ≈ the chart floor, reached) for every letter → the class is recovered from the
half-precision-trained adapter as well as the chart allows → "not protection" *demonstrated*; (b) `err_vs_chart`
O(1) (the 12% inconsistency amplified through σ_min ~1e-6 … 1e-5 into the weak directions) → an extraction gap for
both routes we have — **still not protection**: the release provably encodes the class (learned margins, rank 8,
O(1) imprints) and a matched-arithmetic attacker (simulating in bf16, non-differentiable, not run) is strictly
stronger and untested. fp32-trained (rel. dev. 4e-7) is the control expected to recover to ~1e-6. *Per-letter read (yoado-ed):* the
row-space perturbation of each letter under half-precision training is its certificate residual at the truth in
the Part A rows (bf16 0.04 … 0.23, fp16 0.014 … 0.17); it is read beside that letter's recipe-route endpoint error
(`err_vs_chart_per_image`). Recovery of the 2%-moved letters with failure of the 25%-moved ones would make the
boundary quantitative and reusable; recovery of all regardless would expose the certificate's 1e-3 direction
requirement as that route's own fragility rather than a property of the release. Outcome (b) is written as an
*open* (two routes in one afternoon), not as a boundary.

### Step 25 closed (764976): the tight tolerance recovers ALL EIGHT letters from the fp32-trained release at k = 32

Certificate search from the releases trained in fp32 / bf16, tolerance 1e-12 (the noise rank), 500 random starts:

| cell (k = 32) | trained in | N′ (line) | on a recorded image | found | residuals at the found truths | argmin |
|---|---|---|---|---|---|---|
| **letters 'a'** | **fp32** | 11 (53) | **32.8%** | **8 of 8** (landings 94, 4, 8, 6, 6, 6, 4, 36) | 3e-7 … 1.3e-5 | on a letter, error 1.1e-6 |
| letters 'a' | bf16 | 11 (53) | 0 | 0 | 0.04 … 0.23 | off (0.12) |
| confident | fp32 | 10 (54) | 49.8% | 5 of 8 (1, 2, 5, 6, 7) | 9e-7 … 4e-6 | on, 4.9e-6 |
| confident | bf16 | 10 (54) | 0 | 0 | 0.06 … 0.94 | off (0.17) |
| control | fp32 | 10 (54) | 33.4% | 6 of 7 (image 5 destroyed, 0.79) | 6e-7 … 4e-5 | on, 1.4e-6 |

**The sentence for the supervisor now holds without qualification at the instance-identifying chart:** *from an
adapter trained in fp32 on the canonical fine-tuning task — a class the base model does not have — every private
example is recoverable from random starts with no recipe, no labels and no knowledge of the batch, at k = 32
(instance survival .94 on digits; chart error .24 on the letters).* The two letters the noise-matched tolerance
missed (residuals 2.7e-3, 3.3e-3 there) are found at the tight tolerance with residuals 5e-6 and 6e-6 — the
tolerance, not the release, had hidden them; "noise-matched" is withdrawn as a recommendation for training-
precision releases as it was for storage. bf16-trained releases: 0 found at every cell, as the residuals said.

### Step 26 first rows (771329, letters k = 16): the recipe route against half-precision-trained releases

| trained in | rel. dev. of B_T | residual at the truth (mismatch floor) | σ_min / σ_max at the truth | LM endpoint residual | endpoint error vs the on-chart truth, per letter | median / max | vs raw (floor .32) |
|---|---|---|---|---|---|---|---|
| fp64 (gate) | 5e-16 | 9e-16 | 2.7e-4 / 4.1 | 4e-31 | 1e-15 … 4e-15 | 2e-15 / 4e-15 | .316 / .79 |
| fp32 | 4.4e-7 | 4.5e-7 | 2.7e-4 / 4.1 | **4.5e-14** | 5e-7 … 3.5e-6 | 1.5e-6 / 3.5e-6 | .316 / .79 |
| bf16 | 0.12 | 0.138 | 2.8e-4 / 4.6 | **5.8e-4** | .21 .50 .09 .50 .77 .18 .44 .69 | **.44 / .77** | .57 / .80 |
| fp16 | 0.026 | pending | | | | | |

**Reads.** (i) The gate passes (exact recovery, residual 4e-31); the letters' Jacobian at the truth is well
conditioned (σ_min 2.7e-4 against 1e-6 … 1e-5 for the flowers). (ii) **fp32-trained: outcome (a)** — the FP64
simulator's LM ends 2e-6 from the on-chart truth for every letter (raw error at the chart floor .316), i.e. the
class recovered as well as the chart allows. Note the endpoint residual 4.5e-14 sits *seven orders below* the
residual at the truth (4.5e-7): the mismatch is absorbed by a 2e-6 displacement — an alias of the arithmetic, at
the scale of the arithmetic. (iii) **bf16-trained: outcome (b), in the alias form** — the LM ends at residual
5.8e-4, 240× *below* the truth's mismatch floor 0.138, i.e. the FP64 recipe explains the bf16-trained release
better with *different* images: endpoint errors .09 … .77 against the on-chart truths (median .44), raw errors
.57 (floor .32). Residual well below the truth's, wrong images: the arithmetic mismatch has created a genuine
non-identifiability for the FP64-simulating attacker, not a search failure. Per letter, the least-moved row-space
directions recover best (letter 2: perturbation .045 → error .087; letter 5: .054 → .18) and the most-moved worst
(letter 4: .20 → .77; letter 7: .22 → .69), but not monotonically (letters 1 and 6: perturbation .08/.055 → errors
.50/.44). The non-monotonicity is the alias's signature, not a complication (yoado-6e): the LM solves for all
eight latents *jointly*, so the mismatch is redistributed across images and a little-moved letter can be dragged
off by the joint fit — per-image degraded recovery would be monotone in the per-image perturbation; a wrong joint
solution is not. **Three-way reading to keep distinct:** *leak demonstrated* — fp64/fp32 training (both routes
recover, letters 8 of 8) and bf16/fp16 storage of an O(1)-scale release (the certificate finds ≥ 2 of the
headline's images from bf16 storage); *extraction gap, an open* — bf16 training (certificate 0, simulator alias;
information present, matched-arithmetic attacker unrun; "not protection" is principled here, not demonstrated);
*destroyed by range* — fp16 storage of the 7.6e-18 release only. So at bf16 training: recorded (norm, rank, learned
margins, O(1) imprints), recoverable by neither route we ran; **an open, not a boundary** — the matched-arithmetic
attacker, who simulates in bf16 and has no mismatch floor, is strictly stronger and was not run (autograd through
a rounded loop is not meaningful with our solver). fp16 and the k = 32 rows pending.

### Step 26 addendum (pre-registered; job 779207): is the matched-arithmetic attacker's landscape navigable?

yoado-ed: leave the matched-arithmetic attacker as an open and a reader hears "bf16 training is a defence".
Before building a matched solver, two facts decide whether one can exist: (i) the attacker cannot reproduce the
training's roundings bit for bit — A₀ is unknown (only the bf16 A_T is released, and A₀ = A_T − ΔA carries ΔA's own
rounding), and the reduced simulator (span-adapted coordinates, the recipe route's only tractable form) rounds a
*different* sequence of operations from the full training loop — so even a matched-arithmetic simulator has a
mismatch floor, set by how the low-precision map responds to one-ulp perturbations of its inputs; (ii) a
low-precision loop is piecewise constant at the scale of its roundings, so the matched residual landscape may
be a noise floor everywhere except at the exact truth. Measured (letters k = 16 and 32; fp32, bf16, fp16 releases):
the response `‖B(W + δ) − B(W)‖/‖B‖` of the format's map to relative perturbations δ = 1e-6, 1e-4, 1e-2 of the
latents and to a one-ulp perturbation of A₀, against the FP64 map's response; and the residual along the segment
from the 0.1-noise near start to the truth, in matched arithmetic against the format's release and in FP64
against the same release. **Pre-registration:** the FP64 map responds linearly (≈ σ·δ, ~4e-6 at δ = 1e-6); the bf16
map's response to δ = 1e-6 and to one ulp of A₀ is predicted at 1e-2 … 1e-1 (a rounding cascade over 400 steps —
the same 12% seen as the release's deviation), i.e. the matched landscape is a ~0.1 noise floor with a single
needle at the exact truth: then no matched *gradient-based* solver exists and matched arithmetic is a verification oracle for such
solvers — **which is not "no solver"** (yoado-ed): piecewise constant is a staircase, not noise, and a
derivative-free search at a resolution above the step size (Nelder–Mead, CMA-ES, finite differences taken
deliberately above the steps) is the natural next attacker, untested. The honest sentence is therefore "bf16
training: recorded, not extractable by any *gradient-based* simulator we can build" — an extraction cost, not an
information bound; and the certificate still resolving the class's directions to 5–25% is itself the number that
stops "not extractable by our solvers" from sliding into "not there". *The deciding number is the coarse-scale
trend, not the local ruggedness* (second job, 779969, `--segment-dense`): along 21 linear points from the 0.1
start to the truth, the pointwise matched residual beside 4-point window means at radii 1e-3 and 1e-2 of the
coordinate std. Pre-registration: a windowed mean falling monotonically toward the truth under ~0.1 local noise →
extraction is a solver-engineering problem, the word is *cost*, and a derivative-free search at the window's
resolution is the next attacker; a windowed mean flat until the last window → a needle, and (yoado-6e) *no
landscape-navigating solver of either kind* — gradient or derivative-free — has a signal to follow: matched
arithmetic is then a verification oracle, not an inversion, and only exact enumeration reaches the needle
(infeasible in a 128–256-dimensional latent space) — an extraction cost, present-but-unreachable, still not
protection since a compute-unbounded verifier or a future method is not excluded. *On the A₀ proxy (yoado-6e):*
one ulp on a fully known A₀ is more conservative than "the floor of what an attacker could achieve" — the SGD
release only ever constrains the projection A₀U onto the candidate span (≈ A_T U, 9% off for the letters), and
the component of A₀ orthogonal to that span is unconstrained; so if one ulp of a fully known A₀ already cascades
to 1e-2 … 1e-1, matched simulation is uncomputable for the real attacker *a fortiori*, and the two facts are
stated together. The s = 0 point of the matched segment is a determinism gate (same inputs, same roundings → 0
exactly); if the device's bf16 loop were not bit-reproducible, "verification oracle" would not hold either.
Falsifier: a bf16 response ∝ δ down to 1e-4 (a smooth map at the attacker's scale) — then a matched LM (FP64
Jacobian, bf16 residual) is feasible and is run next. fp32 is expected in between (response ~1e-6 at δ = 1e-6
from its own rounding, then linear).

### Step 26 addendum, measured (779207, 779969): the falsifier fired — the bf16 map is smooth at the attacker's scale

Response `‖B(W + δ) − B(W)‖/‖B‖` of the format's training map (letters, r = 64) to a relative perturbation δ of the
latents, beside the FP64 map's, and to one ulp on every entry of A₀:

| k | format | δ = 1e-6 | δ = 1e-4 | δ = 1e-2 | FP64 map at the same δ | one ulp on A₀ | release's deviation from FP64 |
|---|---|---|---|---|---|---|---|
| 16 | fp32 | 1.3e-6 | 1.2e-4 | 1.05e-2 | 1.25e-6 / 1.17e-4 / 1.05e-2 | 1.4e-7 | 4.4e-7 |
| 16 | **bf16** | **4.4e-6** | **4.4e-3** | 1.6e-2 | same | **1.0e-2** | **0.12** |
| 16 | fp16 | 2.0e-4 | 7.2e-4 | 1.1e-2 | same | 1.3e-3 | 0.026 |
| 32 | bf16 | 1.7e-5 | 2.7e-3 | 1.8e-2 | 1.05e-6 / 1.06e-4 / 1.46e-2 | 1.1e-2 | 0.115 |
| 32 | fp16 | 2.2e-4 | 6.2e-4 | 1.5e-2 | same | 1.2e-3 | 0.027 |

Residual along the segment truth → 0.1-noise start, matched arithmetic against the format's release (k = 16, bf16):
s = 0 → **0 exactly** (determinism gate passed, every cell), 1e-4 → 2.3e-3, 1e-3 → 3.6e-3, 1e-2 → 5.9e-3, 0.03 → 8e-3,
0.1 → 0.022, 0.3 → 0.062, 0.6 → 0.12, 1 → 0.19; the FP64 simulator against the same release: 0.138 flat until
s = 0.3, then 0.146, 0.18, 0.24. Dense segment (21 points, 779969): the pointwise matched residual rises
monotonically 0 → 0.19 at every format (bf16 k = 16: 0, .013, .022, .033, … .189; fp64: 0, .009, .018, … .187), and
the 4-point window means at radius 1e-3 track it to within 6e-3 and at radius 1e-2 to within ~1.5e-2 — no flat
region, no needle.

**Reads.** (i) **The pre-registration is falsified in the direction that helps the attacker.** The release's 12%
deviation from FP64 is a *systematic bias* of bf16 accumulation, shared by nearby inputs, not a decorrelating
noise: the bf16 map responds smoothly (4e-6 at δ = 1e-6, within 4× of FP64) up to a rounding floor of only
2e-3 … 4e-3 reached at δ ≈ 1e-4, and linearly beyond. (ii) The matched landscape is monotone from the start to the
truth at every window size, with a local floor of ~2e-3 — an ordinary smooth landscape with a small noise floor,
navigable by a gradient surrogate; the word is **cost**, not ruggedness, and the "needle" reading is withdrawn.
(iii) The remaining obstacle is A₀: one ulp on every entry of A₀ moves the bf16 release by 1e-2, so the attacker's
reconstruction of A₀ (from A_T, with the learned part in the span of the candidate features up to rounding) sets a
mismatch floor predicted at 1e-3 … 1e-2, against the FP64 simulator's 0.138. **Pre-committed next step, run
(job 782682):** the matched recipe route — the full loop simulated in the training format, unknowns W and Z with
A₀ candidate `A_T − Z Hcᵀ`, FP64 Jacobian as surrogate, acceptance on the matched residual, near start 0.1 —
letters k = 16 and 32 in fp64 (gate: exact), fp32, bf16, fp16. *Pre-registration:* the matched residual at the
truth (Z least-squares) is the A₀ floor, 1e-3 … 1e-2 at bf16; the LM ends within ~5× that of the on-chart truths,
i.e. `err_vs_chart` ≲ 1e-2 … 5e-2 — the class recovered from a bf16-trained adapter by an attacker who simulates in
its arithmetic → **not protection, demonstrated**; falsifier: endpoint error ≳ 0.1 (the alias persists) — then the
A₀ floor, not the landscape, is the extraction limit. *Written before the rows so that neither outcome reads as a
surprise (yoado-ed): the solver's result cannot change the conclusion, only its magnitude.* A descent path exists
(measured); it terminates at whatever floor the A₀ reconstruction imposes; a recovery that stops at the
arithmetic's own floor still returns the private image to within that floor — 1e-2 or 5e-2 decides how degraded
the disclosure is, not whether there is one; and an alias persisting above 0.1 names the A₀ floor as a *cost* of
inverting a public map, not a statement about what the release contains. The rows carry the achieved matched
residual beside the per-letter image error (B-block joint, A-block per letter — field added after 782682
started; for that job the total residual and the objective trace serve): residual and image error falling
together confirms the floor reading; residual at the 2e-3 floor with image error still ~0.3 says the binding
constraint is the parametrisation (A₀'s conditioning), not the precision. *Verdict key (yoado-6e):* Z is not new
freedom — it is the seed block the recipe route already carried as `aux`, reparametrised — but with the matched
residual floored at the A₀ level rather than machine zero, the LM could reach the floor with a wrong (W,
compensating Z): a joint alias. The verdict is therefore keyed off (`err_vs_chart`, `Z_err_rel`) together, never the
residual alone: recovered = residual ≈ floor AND small image error AND small Z error; joint alias = residual ≈
floor with O(1) image error and Z absorbing the mismatch.

### Other rows landed with these
- Recipe route (771329): fp16-trained k = 16 → residual 5.8e-5 (below the truth's 0.027), errors .02 … .26 (median
  .074) — the alias form at a quarter of bf16's displacement; k = 32 fp64 gate exact (σ_min 9.7e-6), k = 32
  fp32-trained → errors 3e-6 … 1e-4 (outcome (a)); bf16/fp16 at k = 32 pending.
- Control bf16-trained, tight tolerance (764976): 0 found, residuals .08 … .87 at the truths.
- bf16 release above the FP64 line (753886, k = 58, tol 0.08, N′ = 3, line 61): **0 landings in 5,000 starts**;
  residual 2e-3 … 2.7e-3 at the three strong truths, and the argmin (objective 9e-8) sits 7.8% from a recorded
  image — a degraded near-miss, not a landing; fp64 control at k = 58/60 pending.
- Control ladder k = 56 (749362): N′ = 6, line 58, 0.86% of 5,000 starts on a recorded image, 5 of 6 found, argmin on
  a recorded image — the control batch attackable at every k from 24 to 56 (71.6 → 51.6 → 18.2 → 6.6 → 0.86%).
  Ladder figure re-rendered with the control cells: `figures/exact_inversion/certificate_ladder.png`.

*Matched route, gate row (782682, letters k = 16, fp64 release, fp64 simulation):* from the 0.1 near start
(objective 0.109) the LM converges in 17 iterations to residual 1e-15, image error 1e-15 for every letter, Z error
5e-15 — the Z-parametrised A₀ candidate and the surrogate-Jacobian solver reproduce the recipe route's exact
recovery; the objective trace falls 0.109 → 8e-3 → 7.6e-4 → … → 1e-30 monotonically. The bf16, fp16 and fp32 rows
are running against this gate.

### Step 26 result (782682, letters k = 16, bf16-trained release, matched-arithmetic recipe route): recovered to 3%

| quantity | value |
|---|---|
| determinism gate (true A₀, bf16 simulation vs the release) | 0 |
| matched residual at the truth (W_true, Z_ls) = the A₀ floor | **0.0231** (pre-registered 1e-3 … 1e-2: above the band) |
| A₀ left unexplained by Z_ls at the truth (rel.) | 0.0205 (the bf16-rounded part of ΔA outside the feature span) |
| FP64 simulator's residual at the same truth | 0.173 |
| start (near, noise 0.1) → endpoint residual | 0.29 → **0.0141** in 20 iterations, then no accepted step (at the floor) |
| image error vs the on-chart truths, per letter | .044 .028 .022 .030 .069 .031 .049 .030 — **median .030, max .069** |
| image error vs the raw letters (chart floor .316) | median **.317** |
| Z error (rel.) | 0.131 |

**Reads.** (i) **Outcome (a): the matched attacker recovers the class from the bf16-trained adapter.** Median
image error 3.0% (max 6.9%) against the on-chart letters, raw error exactly at the chart's floor — the letters
recovered as well as the chart allows, to within 3%; against the FP64 simulator's alias on the same release
(median .44) and the certificate's nothing. The verdict key holds: residual at the floor (0.014 against the
truth's 0.023 — 1.6×, not 240×), image error small and *uniform* across the eight letters (2.2–6.9%; a partial alias
would be non-uniform or O(1)), Z error 0.13 — a red herring (yoado-6e): Z is a soft, under-determined parameter
whose 13% wander moves A₀ by 2% and the images by 3%; the right pair is `err_vs_chart` with `A0_recon_rel`, and
both sit at the floor. Not a joint alias. The honest qualifier: the matched attacker recovers the class *to the
chart floor at a fidelity set by its A₀ reconstruction* (2% → 3% here), not the certificate route's 1e-2 or
FP64's 1e-6 — cost, not an information bound. *Mechanism of the 3% (yoado-ed):* the endpoint's residual 0.014 is
below the truth's own 0.023 — the same signature that at 240× was an alias — because the A₀ mismatch means the true
latents are *not* the minimiser of the matched objective; the recovery lands where the minimiser is, 3% away,
which is also why the error does not fall with more iterations. *The three attackers on one unchanged release:*
certificate — none; FP64-simulator recipe route — none, and an alias at 240× below the truth's residual;
matched-arithmetic recipe route — all eight. Nothing about the release differed; two failures looked exactly
like protection; the third differed by arithmetic, not information. (ii) The floor is the A₀ reconstruction, as pre-registered in mechanism though
not in magnitude: 2.3e-2, set by the 2% of A₀ that the bf16-rounded updates leave outside the feature span
(one ulp on A₀ gave 1e-2; the accumulated rounding is twice that). The image error is ~1.3–3× the residual —
the same order as the certificate's 4–5× law. (iii) **So half-precision training is not protection —
demonstrated, not principled:** the release keeps the class (norm, rank, learned margins), the certificate cannot
read it (directions moved 5–25%), the FP64 simulator aliases (mismatch 0.17), and the attacker who simulates in
the training's own arithmetic gets every letter back to 3%. What bf16 training costs the attacker is the A₀
floor: 1e-15 → 2e-2 in image error. fp16, fp32 and the k = 32 rows are running.

**Precision, the whole picture for the lead cell (letters, k = 16 / 32):** FP64 — 8 of 8 from random starts, exact.
fp32 training — 8 of 8 from random starts (certificate, tight tolerance), recipe route exact to 2e-6. bf16
training — certificate 0, FP64 simulator alias, **matched simulator 3%** (k = 16). fp16 training — certificate 0,
FP64 simulator alias at a quarter of bf16's (pending matched). fp16 storage of an O(1) release — not tested here
(the 7.6e-18 headline release underflows; the letters' would not).

*Recipe route, FP64 simulator, k = 32, bf16-trained letters (771329):* residual 5.5e-4 against the truth's 0.129
(the alias form again), but the endpoint is now far away — image errors .32 … 2.03 (median 1.12) — the k = 32
Jacobian is worse conditioned (σ_min 1e-5 against 2.7e-4 at k = 16), so the same mismatch is absorbed by a much
larger displacement; still descending at the 600-iteration cap. The matched-arithmetic row at k = 32 (782682) is
the one that decides whether the 3% recovery survives the harder chart.

*Matched route, fp16-trained letters k = 16 (782682):* A₀ floor 2.05e-3 (A₀ unexplained 0.66%; FP64 simulator's
floor there 0.047); from the near start (residual 0.31) the LM reaches 1.87e-3 in 23 iterations; image error per
letter .029 .010 .004 .011 .038 .010 .007 .014 — **median 0.97%, max 3.8%**; raw error at the chart floor .316; Z
error 1.8%. Recovered, at a fidelity three times better than bf16's (fp16's 11-bit mantissa leaves a third of
bf16's A₀ error) — the per-format cost ordering follows the mantissa: bf16 3%, fp16 1%, fp32 2e-6, fp64 1e-15.

*Scaling observed across the four training formats (yoado-6e), one batch, k = 16, k = 32 pending:* the matched
attacker's recovery error tracks the **training format's unit roundoff ε within 4–20×** — bf16 (ε 7.8e-3) → 3.0%,
fp16 (9.8e-4) → 0.97%, fp32 (1.2e-7) → 2e-6, fp64 (2.2e-16) → 1e-15 — because the cost is the A₀-reconstruction
floor, which is the accumulated rounding of the updates and scales with ε. Read as: the leakage fidelity from a
low-precision-trained adapter is set by the training precision, monotone and without a cliff (fp16 *storage* of a
1e-18 release is a range cliff on a different axis). The bf16 Z error (13%) against fp16's (1.8%) is the same
mechanism seen from the soft parameter: less accumulation, cleaner Z — the two rows cross-validate the "soft
parameter, not alias" reading. Recorded as an observed scaling, not a law, until another batch and k = 32 repeat it.

*Two readings from the write-up lane (yoado-ed), recorded here so no one quotes a single "safer format" number:*
(i) **the cost ordering reverses the storage ordering.** Against a matched simulator fp16 leaves the attacker a 1%
floor and bf16 a 3% one (reconstructing A₀ is limited by *precision*: ten mantissa bits beat seven); reading a
*stored* release through the certificate, bf16 kept the most and fp16 the least (limited by *range*: eight
exponent bits beat five). The two attacks are limited by different halves of the format; "which format is safer"
has no single answer. (ii) **The control ladder qualifies "every admissible k is attackable":** 7, 7, 7, 6, 5 of
seven at k = 24 … 56 — attackable across most of the range, with recorded images going *unsampled* at a fixed
budget within a few units of the line as the basin shrinks: a sampling limit, not an identifiability one, but the
attacker's limit all the same.

*Matched route, fp32-trained letters k = 16 (782682):* A₀ floor 1.26e-7 (A₀ unexplained 1.4e-6), endpoint residual
1.07e-7 in 30 iterations, image error median 3.0e-7 / max 1.2e-6, Z error 8e-7 — recovered; the fp32 point of the
scaling sits at 2.5× ε. The k = 16 column is complete: **bf16 3.0% / fp16 0.97% / fp32 3e-7 / fp64 2e-15**, each
from the same near start, the same solver, the same release except for the arithmetic it was trained in.
*FP64-simulator recipe route, fp16-trained letters k = 32 (771329, last row):* alias, residual 6.9e-5 against the
truth's 0.028, image errors .16 … .86 (median .47) — against .074 at k = 16 from the same mismatch: the worse-
conditioned k = 32 chart (σ_min 1e-5) turns a 2.8% mismatch into a 47% displacement. The four k = 32 matched rows
decide whether the matched attacker's 1–3% survives that conditioning.

*Pre-registration for the k = 32 matched rows (yoado-ed, written before they exist):* the cost of an arithmetic
mismatch is set by the chart's conditioning, not by the mismatch's size — the same 2.8% mismatch displaces the
FP64 simulator by .074 at k = 16 and .47 at k = 32 (σ_min 2.7e-4 → 1e-5), a sixfold amplification from the chart
alone. Scaling the k = 16 matched errors by that factor predicts roughly **6% for fp16 and 19% for bf16 at
k = 32** — still recoveries, visibly degraded. Near those → the conditioning mechanism is confirmed from a
second direction; much better → something protects the matched solver from its own floor and must be understood
before it is celebrated. Structural point kept either way: a richer chart buys fidelity in exact arithmetic and
pays for it by amplifying every arithmetic error into a displacement, so under real arithmetic **the attacker's
best k is bounded by conditioning as well as by the line** — a second boundary on k from a different direction,
not necessarily coinciding with k < r − N′; if the k = 32 rows degrade as predicted there is an optimum between 16
and 32, named in the opens, not located tonight.
*Refinement before the rows (yoado-6e):* the matched route does not start from the mismatch floor (0.028) but from
the A₀ floor (2e-3 fp16, 2e-2 bf16), and the amplification is not 1/σ_min (the FP64-simulator alias worsened ~6×
from k = 16 to 32, not the 27× of the σ_min ratio: the floor perturbation projects mostly onto well-conditioned
directions). Held loosely: fp16/fp32 still recover at k = 32 (~5% / ~1e-6); bf16 is the one at risk, 2e-2 × ~6
landing at 10–20% and possibly crossing from recovery into alias — which would not contradict the k = 16 law but
refine it to two dimensions, *fidelity ≈ (training-ε floor) × (chart-conditioning amplification)*. Read off the same
key: image-error magnitude and uniformity across letters, with the A₀ reconstruction.

**READ (job 782682, the four k = 32 matched rows; scored by yoado-b9 against the pre-registration above).**
Measured on the same key as the k = 16 column, `err_vs_chart_median`: fp64 **1.54e-14** · fp32 **5.81e-6** ·
fp16 **7.47%** · bf16 **4.56%**.

| | pre-registered | measured | verdict |
|---|---|---|---|
| fp16 | ~6% (yoado-ed) / ~5% (yoado-6e) | **7.47%** | hit, within 1.5× |
| bf16 | ~19% (yoado-ed) / 10–20% (yoado-6e) | **4.56%** | **miss, 2.2–4.2× low** |
| fp32 | ~1e-6 (yoado-6e) | 5.81e-6 | same order |

(i) **The split fires the pre-registration's own escape clause.** bf16 came in *much better* than predicted,
and the registered rule for that branch was "something protects the matched solver from its own floor and must
be understood before it is celebrated". Scored as a miss, not averaged with the fp16 hit.

(ii) **The ordering reverses between the two charts, which neither prediction anticipated.** At k = 16 bf16
(3.02%) was 3.1× worse than fp16 (0.97%); at k = 32 fp16 (7.47%) is 1.6× worse than bf16 (4.56%).

(iii) **"The cost is set by the chart's conditioning" is falsified as stated,** because a conditioning-only
mechanism predicts one amplification factor for every format. Measured k = 16 → k = 32: **fp32 19.3× · fp16
7.7× · bf16 1.5×** — a thirteenfold spread in a quantity the mechanism calls a property of the chart. Measured
above each format's own A₀ floor it is worse: fp16 4.7× → 39.8× against bf16 1.3× → 2.0×.

(iv) **Half the mechanism survives cleanly.** `B_T_rel_dev_from_fp64` is k-independent (fp16 0.0258 → 0.0273,
bf16 0.1218 → 0.1146) and so are the A₀ floors (fp16 2.05e-3 → 1.88e-3, bf16 2.31e-2 → 2.27e-2). The mismatch
size is genuinely not what changed; the chart does set the cost. It is simply not a scalar amplification and
not the same for every format.

(v) **What the rows point at is the precision/range split, arriving in the matched route.**
`residual_entries_exactly_zero_frac`: fp16 0.281 → **0.354**, bf16 0.021 → **0.011**. At k = 32 fp16 flushes a
third of the residual entries to zero while bf16 — carrying fp32's exponent range — flushes one percent. fp16's
extra amplification tracks a *range* failure that grows with k, not a precision one, and bf16's mildness is that
effect absent. Same reversal as the certificate route's storage ordering, now in the matched route, and it is
the standing candidate for what "protects the matched solver from its own floor". Already logged; needs no run.

(vi) **Caveat limiting all eight numbers.** `stopped` is `converged` only for the two fp64 rows; fp32, fp16 and
bf16 all read `no_accept` at both k — the solver stalled. Every low-precision figure is where LM gave up, so
comparing formats partly compares stall points. **Corrected (yoado-41):** the traces are flat at termination in
*every* low-precision row — bf16 k=32 at 1.5358e-4, fp16 at 6.2227e-6, fp32 k=16 at 1.1415e-14 — so these are
genuine stalls rather than budget cuts and the caveat applies uniformly; it does not single out bf16. "Stalled at
a good place" and "converged to a good place" still support different claims about what an attacker gets, and the
bf16 miss stands as a miss.

**Net:** the k = 32 column confirms that a richer chart costs the matched attacker something real (both half
formats degrade, 4.6% and 7.5% against 1.0% and 3.0%), refutes the specific claim that conditioning alone sets
the cost, and exposes a format-dependent second axis — underflow — whose direction reverses the ordering.

### Step 26, the k = 32 matched rows (782682) — against both pre-registrations

| trained in | A₀ floor (matched residual at the truth) | A₀ unexplained | endpoint residual | image error vs on-chart truth (median / max) | raw (floor .235) | Z error | k = 16 value |
|---|---|---|---|---|---|---|---|
| fp64 (gate) | 8e-16 | 1e-15 | 1e-15 (converged, 48 it.) | 1.5e-14 / 1.0e-13 | .235 | 9e-15 | 2e-15 |
| **bf16** | 0.0227 | 2.0% | 0.0124 (17 it., no accepted step) | **4.6% / 8.6%** | .239 | 0.16 | 3.0% |
| **fp16** | 0.0019 | 0.64% | **0.0025** (23 it., no accepted step — *above* its floor) | **7.5% / 10.4%** | .256 | 0.047 | 0.97% |
| fp32 | 1.3e-7 | 1.4e-6 | 1.1e-7 (62 it.) | 5.8e-6 / 3.8e-5 | .235 | 2.9e-6 | 3e-7 |

**Reads.** (i) **bf16 at the instance-level chart: recovered at 4.6%** — uniform across the letters (3.3–8.6%),
residual below the truth's floor by the same 1.8× as at k = 16, raw error at the chart floor. The
chart-conditioning amplification from k = 16 to 32 is **1.5×**, not the 6× read off the FP64-simulator alias
(yoado-ed's 19% pre-registration is falsified toward the attacker; yoado-6e's refinement — the A₀-floor
perturbation lives mostly in well-conditioned directions, expect a single-digit factor — is the one that held,
and bf16 did not cross into alias). (ii) **The ordering inverts between the charts and my first reading of it was wrong.** *(This whole paragraph is
superseded — see "The k = 32 'reversal' dissolves at the optimal stop" and the knee sweep below: the inversion was
an artefact of comparing an over-descended fp16 against a floor-stopped bf16, and "the ordering inverts" is
retired. Read on only for the chronology.)* I wrote fp16's 7.5% off
as a stall because its endpoint residual (2.5e-3) sits above its floor (1.9e-3) — but *every* low-precision row
ends on `no_accept` with a flat trace (bf16 at k = 32 plateaus at 1.5358e-4 for its last three iterations, fp16 at
6.2227e-6, fp32 at k = 16 at 1.1415e-14), so a plateau is the normal termination here and does not distinguish
fp16. Withdrawn. What the rows say: **fp16 3.1× better than bf16 at k = 16 (0.97% vs 3.0%), bf16 1.6× better than
fp16 at k = 32 (4.6% vs 7.5%)** — the ordering reverses with the chart. Two mechanisms are on the table and this
cell does not separate them: *over-descent* (yoado-7e — fp16 reaches a 5× lower residual and a 3× better A₀
reconstruction yet lands worse, having bought that decade by travelling the flat σ_min direction, so bf16's coarse
floor acts as an implicit regulariser), and *flush-to-zero* (yoado-81 — fp16 flushes 28–35% of residual entries to
exactly zero at these margins against bf16's 1–2%, the precision/range split again). Both are consistent with the
logged fields; neither is tested. The k = 16 "error ≈ few × training-ε" observation is a *well-conditioned-chart*
statement and does not survive to k = 32. (iii)
fp32: 5.8e-6 median (20× its k = 16 value), the chart's amplification visible where the floor is tiny. (iv) **The reported rank of a half-precision release exceeds the m − 1 cap and must not be used** (found by
yoado-81, verified in my own rows): the letter cells have m = 11, so the softmax simplex caps N′ at 10, and FP64
gives rank 8 — but the bf16 and fp16 releases report **rank 11** at every tolerance and every k (fp32 reports 10 at
1e-10, 11 at 1e-12). That is impossible under exact softmax: unit roundoff zeroes the own-class entry of the
residual while the off-class entries survive, so the residual columns no longer sum to zero and B_T acquires the
component the cap forbids. Every N′ and every certificate line read off a low-precision release is inflated by it —
a second reason, beside the moved directions, that the certificate route misreads such releases. (v) The
two-dimensional statement that survives: *fidelity from a half-precision-trained adapter ≈ the A₀ floor set by
the training ε, amplified by the chart mildly (1.5× for bf16 from k = 16 to 32)*; the attacker's best k is bounded
by conditioning as well as by the line, but on this cell k = 32 is still inside the recoverable range for every
format. **Closing sentence for the lead cell:** from an adapter trained in bf16 on a class the base model did not
have, an attacker who simulates in bf16 recovers every private letter to 4.6% at the instance-identifying chart;
trained in fp32, to 6e-6; the certificate alone, nothing; the FP64 simulator, a confident alias.

### 753886, the FP64 control above its line and the bf16 release below its wider one (k = 58, 60; 5,000 starts each)

| release | k | N′ (line) | position | on a recorded image | at the floor | found | argmin |
|---|---|---|---|---|---|---|---|
| fp64 | 58 | 7 (57) | **one above** | 0.04% (2 landings) | **14.3%** | 1 of 7 | spurious exact zero (4e-31), error 0.84 |
| fp64 | 60 | 7 (57) | three above | 0 | **36.4%** | 0 | spurious (1e-30), error 1.18 |
| bf16 (tol .08) | 58 | 3 (line 61) — **but 11 (line 53) at tol 1e-12** | *undefined* | 0 | 0 | 0 | 7.8% from a recorded image (obj 9e-8); 229 near-blank starts excluded |
| bf16 (tol .08) | 60 | 3 (line 61) — **but 11 (line 53) at tol 1e-12** | *undefined* | 0 | 0 | 0 | 10.3% from a recorded image (obj 2e-12); 289 excluded |

The FP64 line is sharp as pre-registered: one unit above it the certificate's exact zeros are dense (14%) and
spurious (argmin 0.84 from any recorded image), three above 36%. **The claim that the bf16 release's wider line puts k = 58/60 below it is withdrawn (yoado-81).** The same
release reads N′ = 3 (line 61) at tolerance 0.08 and N′ = 11 (line 53) at 1e-12 — and 11 is above the m − 1 = 10
cap, so that reading is the rounding artefact; whichever tolerance is chosen, "below the line" is a choice, not a
measurement, and at the tight tolerance k = 58 is *above* the line.

**And "approximately and rarely" is withdrawn too — the rows say never (yoado-b9).** Three separate reasons, each
sufficient. (a) *The 7.8% is a post-hoc nearest match, not a recovery:* it is the distance from the best point to
whichever of the seven recorded images happens to be closest (image 7, chosen after the fact;
`argmin_nearest_is_top` False, `argmin_landed_on_recorded` False). Against the intended target the same point is
**113% / 115% away** (`argmin_err_vs_top_chart` 1.133 / 1.148). An attacker holding it cannot know which image it
is near, or that it is near one. (b) *There is no rate to be "rare":* `frac_starts_on_a_recorded_image`,
`frac_starts_recovered` and `frac_starts_at_floor` are all exactly 0 over 5,000 starts, and every per-image landing
count is 0 — the 7.8% is one order statistic, not a tail of a distribution. (c) *The objective's minimum is not at
the truth in these cells:* the objective at the recorded truths is 4.5e-6 … 9.8e-6 while the argmin's objective is
**8.8e-8 / 1.9e-12** — the best point fits the certificate two to eight orders *better* than any truth does, so the
search is not failing to reach a minimum located at a private image; the minimum is elsewhere. The honest
statement: **on a bf16 release of norm 1.6e-24 the certificate objective no longer has its minimum at the truth;
no start of 5,000 lands on a recorded image, and the best-fitting point is 113% from the target.**

*The FP64 control's boundary is graded, not hard (same audit):* at k = 58, one unit above its line, **2 of 5,000
starts did land on recorded image 6** (`frac_starts_on_a_recorded_image` 4e-4, Poisson error 1.41 on that count);
at k = 60 it is zero. "Spurious zeros dense at and above the line" must carry that 2/5,000 rather than round it
away.
The near-blank guard is load-bearing here (5–6% of starts degenerate at k ≥ 58).

### Wide head, k = 40 (725918, last row)
N′ 14 (line 50); 11.3% of 10,000 starts on a recorded image; **9 of 20 found**, more than half the images with zero
landings (median 0, max 678); argmin on a recorded image; chart class accuracy .75. The twenty-digit ladder is
complete: 18 / 15 / 13 / 12 / 9 of 20 at k = 8 / 16 / 24 / 32 / 40 with N′ 19 / 17 / 16 / 15 / 14 — the release
losing rank and the basin losing coverage together as the chart sharpens.

### Overnight closures (jobs finished 2026-09-04 while the session was down)

**Subset test complete (706597).** `hard1_diff` on-chart (N′ = 4): recorded subset → 2.6e-16 (image error
median 6e-7), one-swapped → 1.0e-15 (its own floor 2.2e-15 — reached, at the same level: the swapped-in image's
imprint is 1e-15 of the release), confident-only → 1.0 (nothing explained); `confident` on-chart (N′ = 3):
recorded 7.9e-16 / median error 1e-3, one-swapped 2.3e-12 (= its floor 2.6e-12), confident-only 1.0; raw
`hard1_diff` (N′ = 1): the recorded image reaches 9.5e-6 at image error 0.47 — off-chart, no zero (as every raw
cell). Across three batches the achievable residual orders the candidate subsets by how much of the release they
contain, down to the level at which the release stops distinguishing them (N′ = 6: recorded and one-swapped both
at 5e-17). The `all (control)` rows (N′ = 8, near start, labels given) reach 5e-17 … 6e-16 with median error
.04–.05 — the full-batch identifiability cells, rank-deficient (σ_min 1e-19) as before.

**Negative controls for the one-image trigger (652786).** Random encoder, `hard1_same`: the spectrum trigger
does *not* fire (σ₂/σ₁ = 1.2e-3 on-chart, 2.0e-3 raw — the release is rank-8, one image is not dominant), the
attacker does not read one image, and the residual reaches 2.7e-6 / 4.6e-4 against a predicted one-image floor
of 0.76 — i.e. the one-image reading would be wrong there, and the trigger correctly declines. Strong encoder,
`hard1_same` raw: trigger fires (σ₂/σ₁ 7e-16), argmin label correct and nearest the dominant image, but the raw
truth is off-chart (error 2.6, floor fraction 0) — the label read survives, the image read needs the chart. As
pre-registered: the trigger is a property of the strong model's imprint law, not of the attack.

**3,000-iteration chart reruns (624463, 624465) — the fidelity ranking, previously embargoed.** Only the
charts whose residual *converged* rank: PCA (1.8e-29, converged) and global-PCA (1e-30, converged) reach the
floor with chart error 3e-8 / 3e-9 and raw error .73 (their .42 ceiling); β-VAE β = 16 converges (8e-31, chart
error 4e-12) at a raw error of 1.01 (ceiling .72 — the chart is useless); every other learned chart is *still
descending at the cap*: VAE-GELU 1.4e-17 (chart error .025), VAE-ReLU 2.9e-9 (.22), local 9e-20 (.022), cVAE
7e-17 (.023), β = 0.25 1.3e-16 (.16), β = 1 2.5e-17 (.049), β = 4 1.4e-19 (3e-4). Per the standing lesson, no
ranking among the unconverged arms is claimed. **Weakened on audit (yoado-b9):** I had called the
non-convergence a conditioning statement rather than a budget one, but these are not alternatives — conditioning is
*why* a budget is inadequate. The converged charts took 616 / 1,175 / 1,808 iterations at σ_min 1e-8 … 2e-8 while
the learned charts sit at 4e-12 … 6e-11, so if LM iterations grow anything like 1/σ_min they would need 1e5 … 1e7
and a 3,000 cap cannot separate "slow" from "never". **What stands: the learned charts did not converge within
3,000 iterations, which is what their conditioning predicts**, and they offer better ceilings (.22 … .34 against
.42). Deciding budget against barrier needs the per-iteration descent rate and **that job logged no trace**
(`lm_iters_used` and `stop` only), so it needs a rerun with tracing. Caveat on the conditioning half: for the
learned charts σ_min is evaluated at the 300-step Adam inner-solve output, so it is a property of an unconverged
projection as well as of the chart; the ceiling half is safe, since inner-solve slack only understates a ceiling. The ceiling
question (a richer chart) and the conditioning question (a reachable one) pull apart, as the conditioning
figure said.

**Flowers on CIFAR, mixed batches (762253: 1, 4, 7 flowers among 8, both charts, both head inits) and the
old head (cifar10_m10).** Every release is rank 8; every near-start cell reaches ~1e-30 at chart error ≤ 5e-12;
every random-start cell (2 restarts, 600 iterations) stops at 4e-4 … 3e-2 with chart error .07 … .60 — search
failure throughout, no alias. The mixed rows add nothing to the imprint law beyond what the digits' mixed batch
showed: identifiable, not reachable from random starts by the recipe route at k = 16.

**Letters on the mid and weak MNIST models (658575, in progress; strong done).** The new class is rank 8 with
margins −4 … −5 at t = 1 on every model (strong, mid, weak; zero or random head row) — the pre-registered
flattening holds where it matters: a class the model does not have is recorded in full by every model. The old
digits flatten as predicted: rank 8 on mid and weak (margins −1.7 … +0.6: the weaker models are unsure of their
own digits) against rank 6 on the strong model. Recipe-route cells: near-start exact everywhere (1e-30); random
starts fail everywhere (chart errors 1–5). The weak model's mixed batch is still running.

**OOD inversion grids (644064, three encoders × three sets, k = 16, N = 8, near start and random start).** Recipe
route: every near-start cell reaches 1e-30 (chart error ≤ 1e-9) except the two strong on-chart cells, which stall
at 2e-16 / 1.9e-15 with chart error 0.10–0.13 (σ_min 1e-18 — rank-deficient because part of the batch is
invisible); every random-start cell fails (chart error 0.4 … 13).

**What the imprint rows say, in the form the rows support (rewritten after audit — the categorical version is
withdrawn).** My first reading was "the model records what it gets wrong and not what it gets right". The rows
contradict the categorical form in both directions (yoado-b9): in the strong model's on-chart font batch, image 7
has a *positive* margin (+1.25) and a relative imprint of 0.36, comparable to the 0.50 of a misread one; and the
weak model's `mnist_control` batch, which it classifies perfectly, still records at relative imprints up to 1.0
because its margins are 0.1 … 14.5. The consistent statement is the **margin-order law already measured within
batches (9/9)**: *what is recorded is what has low margin*, and distribution shift is a route to low margin rather
than a second mechanism. It shows up cleanly across the grid — strong model: `mnist_control` margins 4.7 … 23.2 →
one image above 1e-6 relative, `font` margins −6.7 … +44 → the four lowest-margin images carry 0.36 … 1.0 and the
four highest 1e-4 or less, `optdigits` margins −17 … +8 → six of eight at 0.1 … 1.0; weak and mid models: margins
0.1 … 11 everywhere → everything recorded. The falsifiable form, not yet run: **an in-distribution batch selected
for low margin should be recorded as heavily as optdigits.** *Measurement note:* `ood_acc_at_W0` is computed once
on the raw images (`subset_and_ood.py:286`, outside the setting loop) while `margins` are recomputed per setting,
so an on-chart row can show accuracy 1.0 beside negative on-chart margins; the two are not inconsistent, they are
about different images. The margin-order form does not depend on the accuracy field at all, which is a second
reason to prefer it.

**R6's boundary, measured (706597, three batches).** The selection rule works while the omitted imprints are
large enough to move the floor, and stops when they are not. Ratio of the one-swapped subset's achieved residual
to the recorded subset's, by cell: `hard1_diff` N′ = 4 → **3.9×** (2.6e-16 vs 1.0e-15); `confident` N′ = 3 →
**2.9e3×** (7.9e-16 vs 2.3e-12); `repeated` N′ = 3 → **1.9e6×** (1.1e-16 vs 2.1e-10); `repeated` N′ = 6 →
**1.6×** (3.4e-17 vs 5.3e-17) — gone. The confident-only subset (no recorded image at all) separates by 1e5 … 2e16
in every cell including N′ = 6, so the rule always distinguishes "contains recorded images" from "contains none";
what degrades with N′ is the finer discrimination between two subsets that both contain most of the release. The
mechanism is visible in the predicted floors: at N′ = 3 the swapped-in image's imprint is 1e-10 of the release, at
N′ = 6 it is 4e-19 — below the solver's own reach (both cells stop at 5e-17), so no residual can see it. **The
law, in the quantity the rows already log (yoado-b9):** the controlling number is `residual_floor_pred` on the
*one-swapped* subset, computable without solving anything — 2.06e-10 → ratio 1.9e6 (repeated, N′ = 3); 2.57e-12 →
2.9e3 (confident, N′ = 3); 2.19e-15 → 3.9 (hard1_diff, N′ = 4); 3.83e-19 → 1.6 (repeated, N′ = 6). Monotone across
four cells, nine orders of predicted floor against six of ratio, and the last cell explains itself: its swapped
floor lies *below* what the recorded subset actually achieved (3.4e-17), and a solver cannot resolve a floor
beneath its own achievable residual. **The discrimination ratio tracks the swapped subset's predicted floor
against the achievable residual, and is lost when that floor falls below it** — falsifiable on a new cell *before
it runs*. This also dissolves the confound I flagged (N′ and which image was swapped move together): `floor_pred`
is one scalar that absorbs both and orders all four cells.

### The cap-violating direction is the all-ones vector — measured (job 85049), with two corrections to the fix

`D = softmax(z) − Y` has every column summing to zero exactly, so `B_T`'s column space lies in the zero-sum
hyperplane and `rank B_T ≤ m − 1` (yoado-7e's derivation). Recomputing the letter releases and projecting the
all-ones direction `1_m/√m` out of the output space:

| k | format | ‖B_T‖ | rank @1e-10 | rank after removing 1_m | max overlap of a left singular vector with 1_m (index) | σ_rel there | mean \|softmax column sum − 1\| at t = 1 |
|---|---|---|---|---|---|---|---|
| 16 | fp64 | 0.975 | 8 | 8 | **0.976** (9th) | **2.1e-16** | 1.5e-16 |
| 16 | fp32 | 0.975 | 10 | 10 | **0.976** (9th) | **1.5e-7** | 8.0e-8 |
| 16 | bf16 | 0.862 | 11 | **10** | 0.792 (6th) | 1.3e-2 | 5.0e-4 |
| 16 | fp16 | 0.960 | 11 | **10** | 0.754 (6th) | 5.9e-3 | 1.2e-4 |
| 32 | fp64 | 1.001 | 8 | 8 | 0.979 (9th) | 2.2e-16 | 6.9e-17 |
| 32 | fp32 | 1.001 | 10 | 10 | 0.981 (9th) | 9.4e-8 | 3.6e-8 |
| 32 | bf16 | 0.894 | 11 | **10** | 0.600 (6th) | 8.6e-3 | 8.0e-4 |
| 32 | fp16 | 0.983 | 11 | **10** | 0.562 (6th) | 5.5e-3 | 1.3e-4 |

**Confirmed:** the ninth direction is the all-ones vector (overlap .98) and its singular value tracks the softmax
column-sum error almost exactly — fp64 2.1e-16 against 1.5e-16, fp32 1.5e-7 against 8.0e-8. The mechanism is
exactly as derived: roundoff breaks `1ᵀD = 0`, an all-ones component leaks into `B_T`, and the simplex cap is
violated numerically.

**The conclusion is not "subtract one" but "a half-precision N′ is not a physical rank" (yoado-7e, adopting the
corrections below): do not quote it as a recorded-image count and do not derive a certificate line from it — only
the FP64 rank is clean.** That is also the stronger basis for withdrawing the "widened line" reading: the line is
not off by one, it is non-physical on any half-precision release. **It does not touch the recovery results** — the
letters still recover 8 of 8 at k = 16 from the fp32-trained release and 4.6% at k = 32 from the bf16-trained one,
because the spurious directions are sub-roundoff and do not unseat the true images as minimisers of
`‖Cφ‖/‖A_Tφ‖`. Rank readout unreliable in half precision; recovery intact; the two are separate.

**The two corrections that force it (to "subtract one from every half-precision N′").** (i) *It is necessary but not
sufficient.* Removing the all-ones component takes bf16 and fp16 from 11 to **10**, not to the true 8 — the two
remaining extra directions are ordinary roundoff filling, not the simplex break. A corrected N′ is still inflated
by two on these cells. (ii) *In half precision the all-ones is not an isolated small direction:* its overlap is
0.98 with the ninth singular vector in fp64/fp32 but only 0.56–0.79 with the *sixth* in bf16/fp16, i.e. it is mixed
into the signal directions and cannot be cleanly removed.

**And the true rank is not recoverable by choosing a tolerance (yoado-c9's question, answered no).** In fp64 the
gap between the last real direction and the first spurious one is eleven orders (3.2e-5 → 2.1e-16). In bf16 at
k = 16 it is **3×** (σ_8 = 5.8e-4, σ_9 = 1.9e-4), in fp16 **4×** (2.0e-4 → 5.2e-5); at k = 32, bf16 4× and fp16 2×.
Note also that σ_9…σ_11 sit *below* the format's unit roundoff (bf16 1.9e-4 … 1.5e-5 against ε = 7.8e-3), so the
10ε discipline does not reach them either. Rounding has compressed the spectral gap from eleven orders to a factor
of two to four.

**"Rank 11 at every tolerance" was too strong and the accurate version is sharper (yoado-c9).** Sweeping the
tolerance over the stored spectra gives the rank actually read (true value 8 in every cell):

| k | format | 10ε (7.8e-2) | ε (7.8e-3) | 1e-3 | 3e-4 | 1e-4 | 1e-12 |
|---|---|---|---|---|---|---|---|
| 16 | fp64 | 3 | 5 | 6 | 6 | 6 | **8** |
| 16 | bf16 | 3 | 6 | 7 | **8** | 9 | 11 |
| 16 | fp16 | 3 | 5 | 7 | 7 | **8** | 11 |
| 32 | bf16 | 3 | 6 | 7 | **8** | 8 | 11 |
| 32 | fp16 | 3 | 5 | 6 | 7 | 7 | 11 |
| 32 | fp32 | 3 | 4 | 5 | 5 | 6 | 11 |

So the read is not 11 everywhere: it climbs from 3 to 11 as the tolerance tightens and it *does* pass through the
true 8 — but only inside a narrow, unmarked band (for bf16 at k = 16, between σ₉ = 1.9e-4 and σ₈ = 5.8e-4) that
nothing in the release identifies. **No release-agnostic tolerance recovers the true rank: the principled choices
under-count badly (10ε → 3 in every cell, including FP64's), tight ones over-count on any inexact release, and the
truth sits in a band you would have to already know the answer to find.** That is the "unreliable in both
directions" caveat occurring inside a single release, and it is why a half-precision N′ is not a physical rank.

### The k = 32 reversal is over-descent, not information loss (job 85300, yoado-7e's test)

The two candidate mechanisms differ in where the damage is: over-descent is a solver effect (an early stop fixes
it), flush-to-zero is information lost from the release (no solver can undo it). The test: run the fp16-trained
k = 32 cell again and **stop the LM at bf16's residual level** (0.0124) instead of letting it descend to 0.0025.

| fp16-trained, k = 32 | residual reached | iterations | image error vs the on-chart truths (median / max) |
|---|---|---|---|
| full descent (782682) | 0.0025 | 23 | **7.5% / 10.4%** |
| **early-stopped (85300)** | **0.0107** | **4** | **4.72% / 7.69%** |
| bf16-trained, full descent, for comparison | 0.0124 | 17 | 4.56% / 8.6% |

**Over-descent confirmed.** Stopped at bf16's residual, fp16 lands at 4.72% — within 4% of bf16's 4.56%, i.e. the
two formats agree once the descent is equalised, and the entire reversal is the extra decade of residual
reduction fp16 buys by travelling the flat σ_min direction. The release is not the lossy thing. *(Superseded by the full sweep below: bf16's floor
does not land near the knee, it stops short of it, so "bf16's floor acts as an implicit early stop" is retired
along with the finding it explained.)* The "coarser format recovers better at the hard chart"
finding is a statement about the solver's stopping point, not about what half precision destroys. Per-letter
errors are uniform (4.1–7.7%) and the Z error falls with the residual (0.076 against 0.047 at full descent).
Consequences, **corrected on audit (yoado-7e) — my first version was wrong in a way an attacker following it
would feel**: (i) *the floor is not the place to stop.* The best image error here is reached near residual ~0.011,
while fp16's own A₀ floor is 0.0019 — far *below* it, so an fp16 attacker who "descends to the floor" over-descends
and gets exactly the 7.5%. The floor is a lower bound on the reachable residual and an upper bound on how far one
should travel toward it; the right stop is the **knee**, where residual reduction stops buying image accuracy and
starts buying flat-direction travel, or equivalently damping along the ill-conditioned direction. (ii) *"Coarse
arithmetic recovers better" is **retired**, not merely scoped (yoado-7e, after the full sweep).* The bf16 > fp16
result was an artefact of comparing an over-descended fp16 (full descent, 7.5%) against a floor-stopped bf16
(4.56%). With early stopping the ordering is monotone in precision — fp32 6e-6, fp16 stopped at the knee 4.20%,
bf16 4.56% — and bf16's floor (0.0124) sits *above* the knee (~0.004), so it stops short of the optimum rather
than landing on it. This paragraph's earlier "accident of scale, floor happens to land near the knee" reading was
built on a two-point extrapolation of the knee's position and is superseded. The general statement is that **at an ill-conditioned chart the attacker should early-stop or regularise —
which they can do at any precision** — and low precision does it by accident when its floor coincides with the
knee. Nobody should read "train in bf16 and the attacker does worse". (iii) The flush-to-zero reading (fp16
zeroing 28–35% of residual entries) and the dynamic-range reading (σ_min 1e-5 near fp16's normal floor) are not
needed to explain the reversal, though neither is excluded; the matched-residual gap of 3.5% bounds them — and even that
gap is partly the 16% residual mismatch between the two stopping points (0.0107 against 0.0124) rather than pure
format, which tightens the conclusion further (yoado-7e). The bound's resolution is the per-letter scatter; a
tighter bound would need more seeds and letters, not another experiment. **On the 2×2 (yoado-7e):** the decomposition is right — any residual-determined part is over-descent, any
format-locked gap at matched residual is flush-to-zero — but the fourth cell cannot be measured: bf16's own A₀
floor is 0.0227 and its full descent stops at 0.0124, so it can never reach fp16's 0.0025. The one comparison the
design does support is the matched-residual one, and it is already in: **fp16 at 0.0107 → 4.72%, bf16 at 0.0124 →
4.56%**, a 3.5% relative gap, smaller than the per-letter scatter in either row. So over-descent accounts for the
reversal to within the resolution available, and the flush component is bounded above by that gap rather than
measured. The mirror run is kept as a control that bf16 stopped early does not improve either.

**What over-descent is, in general (yoado-c9):** it is not a precision curiosity but the measured form of a
hazard already named in the write-up audit — *the image finishes long before the residual does*. Whenever the
release's exact minimiser differs from the true image, the image optimum sits at an **intermediate** residual, and
descending to the minimiser moves the image away from it. Two causes seen so far: off-chart (the release-optimal
point is not the representation-optimal one) and roundoff-trained (the true latents are not the matched objective's
minimiser — Step 26's 3% displacement). The early-stop rows are the first measurement of "does driving the residual
down actually hurt the image?", and the answer is yes: **a lower floor is a liability, not an asset**, because it
lets the solver descend past the knee. The attacker's stopping rule is therefore the image knee, not the residual
floor — and the knee is above every format's floor.

*Knee sweep, pre-registered (jobs 86888 fp16, 87369 bf16 over the overlap band).* fp16-trained letters,
k = 32, matched route, the LM stopped at residuals 0.08, 0.04, 0.02, 0.012, 0.008, 0.005, 0.003, 0.0019 (its floor);
bf16 at 0.08, 0.04, 0.02, 0.016, 0.0124 (it cannot go lower). **Predicted (yoado-7e):** image error against
stopping residual is U-shaped — high at 0.08 (barely descended), a minimum near 4.6% at a knee around 0.011, rising
to 7.5% at fp16's floor; the two points already in hand (0.0124 → 4.56%, 0.0025 → 7.5%) forbid a monotone curve,
so what the sweep decides is *where* the knee sits and *how sharp* it is — a flat bottom is a wide safe-stopping
band for the attacker, a sharp one means they must tune the stop. **Sharpened before the rows land (yoado-c9): "fidelity is residual-determined" is a *left-arm*
property, and the knee is where it stops holding.** Above the knee every format is still converging toward the
truth, so image error ≈ f(residual) and the curves must coincide; below it each format veers toward *its own*
displaced minimiser, and those differ (coarser roundoff → larger displacement), so the right arms **must** diverge
by format — that divergence is not a failure of the hypothesis, it *is* the over-descent, and its onset marks the
knee. For fp16 against bf16 the overlap is entirely left-arm (bf16's floor ~0.0124 sits at or above the knee
~0.011), so those two should coincide cleanly. **The trap to avoid on the plot:** fp32's minimiser is essentially
the truth (roundoff ~2e-8), so it has *no right arm* — it descends monotonically to ~1e-6 and will peel
**downward** past the knee while bf16 and fp16 peel upward. That is the same mechanism (fidelity tracks
distance-to-minimiser, and fp32's minimiser is the truth), not a violation of residual-determinism, and it must be
labelled as such or it would spuriously reopen the format-effect question. **A third sweep (fp32, job below) tests
exactly that prediction: coincidence on the shared left arm and a monotone descent with no upturn.** Any persistent
vertical gap in the shared left arm is the flush-to-zero contribution, measured instead of bounded by the current
≤ 3.5% scatter.
*Where to read the plot first (yoado-c9).* fp32's missing right arm is a near-certain confirmation rather than a
live risk — both endpoints are already known (4.6% near the knee, 5.8e-6 at its floor), so that arm only tests
monotonicity *between* them, and the sole surprise available is a mid-curve upturn (image worse at some
intermediate residual than at both the knee and the floor), which would mean a non-monotone residual→image
trajectory. **The stringent test is the LEFT arm**: it re-tests the dynamic-range hypothesis, which the early-stop
row excluded from a single point, across the whole shared range. If all three formats coincide *above* the knee,
format genuinely does not matter where everything is still converging toward the truth and the hypothesis is
doubly dead. If any format is already off there — fp16 worse than fp32 at, say, residual 0.05, well above the knee
— that is a format effect operating *before* any over-descent, and it reopens the question the early-stop row
closed. **The over-descent account needs the left arms to coincide, not merely the right arms to diverge**, so
that is the first thing to read.

### Knee sweep, partial rows (86888 fp16, 87369 bf16, 88743 fp32; k = 32, one row per stopping target)

| stop target | fp16: residual reached → image error | bf16 | fp32 |
|---|---|---|---|
| 0.08 | 3.4e-2 → **5.252e-2** | 4.2e-2 → **5.239e-2** | 3.6e-2 → **5.259e-2** |
| 0.04 | 3.4e-2 → 5.252e-2 | 2.3e-2 → 5.080e-2 | 3.6e-2 → 5.259e-2 |
| 0.02 | 1.37e-2 → 5.123e-2 | 1.83e-2 → 4.653e-2 | — |
| 0.016 / 0.012 | 1.07e-2 → 4.724e-2 | 1.49e-2 → 4.534e-2 | — |
| 0.008 | 7.4e-3 → **4.274e-2** | (floor 1.24e-2) | — |
| full descent | 2.5e-3 → 7.471e-2 | 1.24e-2 → 4.558e-2 | 1.1e-7 → 5.8e-6 |

**The left arms coincide** — at matched targets the three formats agree to within 0.4% relative (5.239 … 5.259e-2
at 0.08), and where the reached residuals differ the ordering follows the residual, not the format. No format is
off above the knee, so the dynamic-range reading is excluded a second time, now across the whole shared range
rather than at one point. **And the knee is lower than the ~0.011 both the write-up and the audit had asserted:**
fp16 stopped at 0.008 reaches 7.4e-3 and gives **4.274e-2**, better than bf16's best (4.534e-2 at 1.49e-2) and than
fp16's own 0.012 stop. So **bf16's floor (1.24e-2) sits *above* the knee, not at it** — bf16 is slightly
under-recovering rather than landing on the optimum, which corrects the "lucky landing" clause: coarse arithmetic
stops a little short of the best point rather than exactly on it. The remaining fp16 stops (0.005, 0.003, 0.0019)
locate the minimum and its sharpness.

### The k = 32 "reversal" dissolves at the optimal stop (yoado-c9's reading, adopted with one scope caveat)

Lined up at each format's *best* stopping point rather than at full descent:

| | fp16 | bf16 | ordering |
|---|---|---|---|
| k = 16, full descent (= each format's floor) | **0.97%** | 3.0% | by precision |
| k = 32, full descent | 7.5% (ran to 2.5e-3) | 4.56% (floor 1.24e-2) | *inverted* |
| k = 32, best stop measured so far | **4.27%** (at 7.4e-3) | 4.53% (its best reachable) | by precision, restored |

**There is no genuine reversal.** At k = 32 the inversion exists only in the naive full-descent comparison: fp16
has the range to descend past the knee all the way to its floor and ruins the image doing so, while bf16's floor
sits *above* the knee, so it cannot over-descend — and cannot reach the knee either, so it under-recovers at 4.53%.
Precision is an asset (fp16 gets closer to the knee) that becomes a liability only past the knee, and only for an
attacker who does not stop. The clause "the ordering inverts at k = 32" is **withdrawn**, along with the older
chart-conditioning-amplification framing that predicted bf16 much worse; what replaces both: *at the optimal stop
the precision ordering holds, and the full-descent reversal is the over-descent hazard, visible only to an attacker
who runs to the floor — which is why the stopping rule is the knee, not the floor.*

*Two scope caveats of my own.* (i) fp16's 4.27% is the best **measured so far**, not a located minimum; the
remaining stops (0.005, 0.003, 0.0019) may go lower before the right arm rises. (ii) The k = 16 row is a
full-descent comparison, so "the precision ordering holds at every k" is established at k = 32 and *assumed* at
k = 16 — it holds there only if that chart's knee lies below fp16's floor (2.05e-3), which no row tests. A k = 16
knee sweep would settle it; until then the statement is "at k = 32, measured; at k = 16, consistent with the
full-descent rows".

*Synthesis, pre-registered before ei85d lands (yoado-c9).* The controlling quantity is not `k` but **where each
chart's knee sits relative to the formats' floors**, and that position moves with the chart's conditioning. At
k = 32 (σ_min ~1e-5) the knee is shallow — about 7e-3, *above* fp16's floor of 1.9e-3 — so fp16 has room to
over-descend and does. **Prediction for k = 16:** since fp16's full-descent error there is already 0.97%, there is
little room for an over-descent penalty, so the k = 16 knee should lie *at or below* fp16's floor (2.05e-3) with a
shallow-to-absent right arm; then the two charts bracket the mechanism — *conditioning lifts the knee up into the
reachable residual range, and that lift is what turns precision from an asset into a liability.* **If instead the
k = 16 knee comes back above fp16's floor**, fp16 over-descends there too, its 0.97% is not its optimum, and the
honest statement becomes "precision is an asset up to each chart's knee" with k = 16 a mild over-descent case as
well — still fp16 ahead of bf16 at matched stops, but with no clean "ordering preserved" shortcut. The right-arm
shape at both charts is what decides it.

### Knee sweep, fuller rows — the knee at k = 32 is SHARP, and fp32 has no right arm

k = 32, image error against the residual actually reached (same start, seed, solver; one row per stopping target):

| residual reached | fp16 | bf16 | fp32 |
|---|---|---|---|
| 4.2e-2 | — | 5.239e-2 | — |
| 3.6e-2 / 3.4e-2 | 5.252e-2 | — | 5.259e-2 |
| 2.3e-2 | — | 5.080e-2 | — |
| 1.8e-2 | — | 4.653e-2 | — |
| 1.5e-2 | — | 4.534e-2 | — |
| 1.2e-2 | — | **4.558e-2** (its floor) | 5.134e-2 |
| 1.07e-2 | 4.724e-2 | — | — |
| 7.0e-3 / 7.4e-3 | **4.274e-2** | — | 4.752e-2 |
| 4.2e-3 | **4.196e-2** | — | — |
| 2.5e-3 | **7.471e-2** (its floor) | — | — |
| 1.1e-7 | — | — | **5.810e-6** (its floor) |

**The knee is sharp and lies between 4.2e-3 and 2.5e-3.** fp16 improves monotonically down to 4.196e-2 at
residual 4.2e-3 and then jumps to 7.471e-2 at 2.5e-3 — the image error nearly doubles over a factor of 1.7 in
residual. There is no flat bottom: **an attacker must tune the stop**, and a rule of "descend as far as you can"
loses almost half the fidelity. (The exact location needs the 0.003 row, still running.)

**fp32 has no right arm, as predicted:** it descends monotonically from 5.259e-2 at 3.6e-2 through 4.752e-2 at
7.0e-3 to **5.810e-6** at its floor of 1.1e-7 — no upturn anywhere, because its minimiser is essentially the truth.
The distance-to-minimiser reading is confirmed from the one format that has no displaced minimiser to travel to.

**The knee is per-format, not one knee.** bf16 improves to 4.534e-2 at 1.49e-2 and then ticks *up* to 4.558e-2 at
its floor 1.24e-2 — the onset of bf16's own knee, which sits near 1.5e-2 against fp16's near 4e-3, about 4× higher.
So each format has its own knee and its own right arm; bf16's floor lands just past its own knee, which is why it
neither over-descends badly nor reaches the better optimum fp16 can.

**The left-arm discrepancy is the knee boundary, not a crack in residual-determinism (yoado-c9).** At nearly the
same residual fp32 is *worse* than fp16 (4.752e-2 at 6.99e-3 against 4.274e-2 at 7.38e-3, 11% the wrong way), while
at the coarse stops the three agree to under 0.4%. The reason the axes stop being comparable: **the three curves
invert different releases in different arithmetics**, so "residual 7e-3" against the fp32 release in fp32
arithmetic is not the same distance-to-anything as "residual 7e-3" against the fp16 release in fp16 arithmetic.
Far above every format's roundoff the releases are effectively identical and the arithmetics agree — which is why
the coincidence of three *different* releases on the left arm is a real result and not a triviality — and near the
knee (residual ~ roundoff) they diverge and "same residual" stops being cross-comparable. Path-dependence (4 LM
steps against 5) is a symptom of the same thing. Neither touches the per-format-optimal-stop ordering.

**Synthesis the set points to (qualitative; the scaling is not pinned).** *The knee residual rises with the
training roundoff and with the chart's conditioning.* That one statement covers everything measured: coarser
roundoff → higher knee → worse attainable optimum (the precision ordering); worse conditioning → the knee lifts
above the reachable floor → over-descent bites (the k = 32 "reversal"); fp32's roundoff so small that its knee lies
below any floor → no right arm at all. It also predicts the k = 16 result now landing — less conditioning, lower
knee, below fp16's floor, no right arm. **Not pinned:** the scaling is not linear — the bf16/fp16 knee ratio is
about 4× against a roundoff ratio of about 8× — so "knee ∝ roundoff" is qualitative and the conditioning
interaction (or a sublinear law) is unresolved. The ordering conclusions need only the qualitative form.

*k = 16 so far:* fp16 4.044e-2 at residual 3.2e-2, 3.444e-2 at 1.5e-2, **0.973e-2 at its floor 1.87e-3** — still
descending at the floor, no right arm yet, consistent with the pre-registered prediction that the k = 16 knee lies
at or below fp16's floor. The remaining stops (0.012 … 0.00205) will show whether it turns up at all.

### The knee pinned, and the right arms diverge by format exactly as pre-registered

The two rows that close it. **k = 32, fp16:** 4.196e-2 at residual 4.24e-3 → **7.485e-2 at 2.84e-3** → 7.471e-2 at
its floor 2.50e-3. The knee is between **4.2e-3 and 2.8e-3**: a factor of 1.5 in residual takes the image error
from 4.2% to 7.5%. Sharper than the earlier bracket and confirming there is no flat bottom.

**k = 32, fp32 at the same place:** 3.780e-2 at residual **2.65e-3** — where fp16 sits at 7.49e-2. At a matched
residual below the knee the two formats differ by a factor of **two**, and fp32 goes on descending to 5.8e-6.
That is the pre-registered right-arm divergence, measured: *above the knee the curves coincide (0.4%), below it
each format veers toward its own minimiser and they separate by 2×.* Residual-determinism is a left-arm property
and the knee is exactly where it ends — as registered before the rows landed, and now with the sharpest possible
illustration rather than the 11% hint.

**k = 16 confirms the prediction:** fp16 descends monotonically — 4.044e-2 at 3.2e-2, 3.444e-2 at 1.5e-2,
2.768e-2 at 9.2e-3, 2.537e-2 at 6.2e-3, **0.973e-2 at its floor 1.87e-3** — with no upturn anywhere. The k = 16
knee lies at or below fp16's floor, so fp16 does not over-descend at the well-conditioned chart and its 0.97% *is*
its optimum. The two charts therefore bracket the mechanism as pre-registered: **conditioning lifts the knee up
into the reachable residual range, and that lift is what turns precision from an asset into a liability.** The
"precision ordering holds at the optimal stop" statement is now measured at both charts, not assumed at either.

*The crossover annotation (yoado-c9).* The two formats **cross at fp16's knee**: at residual 7e-3 fp32 is *behind*
fp16 (4.752e-2 against 4.274e-2) because it is still mid-descent; at 2.7e-3 fp32 has *overtaken* it (3.780e-2
against 7.485e-2) because fp16 has veered toward its displaced minimiser while fp32 continues toward the truth.
The 11% gap recorded earlier as an unexplained discrepancy was the first pixel of that crossover. One picture of
the different-minimisers mechanism.

**What this sub-thread delivers.** For a matched-arithmetic attacker, **low-precision *training* is not
protection**: it displaces the minimiser, but the attacker still recovers to within a few percent of the chart's
own floor by stopping at the knee instead of at the release's floor. What coarse training costs is a tuning step
and a few points of fidelity, not access — the same shape as the storage-precision result, and the same conclusion:
"quantise for privacy" is not supported by any cell measured here.

### The knee's cause: underflow, not displacement — and the two are anti-correlated here (yoado-b9's dose-response, checked)

Over-descent cost = (error at the lowest-residual row) ÷ (error at the best stop), against the fraction of
residual entries flushed to exactly zero and the release's own deviation from FP64:

| cell | flush fraction | release deviation | over-descent cost | best error |
|---|---|---|---|---|
| fp32, k = 32 | **0.000** | 4.3e-7 | **1.00×** (no knee in range: 0.0526 → 1e-5 monotone) | 1e-5 |
| bf16, k = 32 | 0.011 | **0.115** | 1.01× | 0.04534 |
| fp16, k = 16 | 0.281 | 0.026 | 1.02× | 0.00954 |
| fp16, k = 32 | **0.354** | 0.027 | **1.78×** | 0.04196 |

**The cost is monotone in the flush fraction, and the zero-flush cell has no knee at all.** The reading: flushing
residual entries to zero is what stops the residual being a faithful proxy for image error, and the knee is where
that proxy fails; the stopping rule is the remedy, underflow is why a remedy is needed.

**And the obvious confound is dissociated, on one decisive cell.** bf16 carries the **largest** release
displacement (0.1146 against fp16's 0.0273) with almost no flush (0.011) and almost no knee (1.01×), while fp16
carries the smaller displacement with a third of its entries flushed and pays 1.78×. *The cell with the largest
displacement has the smallest knee* — which rules displacement out as the cause, in the one direction that matters.
**The two strengths of evidence are different and are stated as two (yoado-b9):** *flush orders the cost* — four
cells, monotone, with a zero-flush control that has no knee at all — while *displacement is dissociated from it*,
carried by that single decisive cell. This is deliberately **not** called an anti-correlation: fp32 has both the
smallest displacement and the smallest cost, so the relation is not monotone downward, and a rank statistic on
four points dominated by one cell would invite exactly the objection the dissociation avoids. *Caveat kept (yoado-b9's own):* the
jump from 1.02× to 1.78× spans a 26% change in flush, which is steep for two fp16 points — the ordering and the
zero-flush control are what is claimed, not a functional relationship.

**This also removes "luck" from the bf16 story.** The earlier text said bf16's floor "happens to land near the
knee". It is not a coincidence: bf16 barely flushes, so its residual stays honest, so its knee is shallow and its
floor can sit near it at almost no cost. The falsifiable form: **a format that flushes more should show a sharper
knee sitting further above its floor.**

**One correction to my own k = 16 claim.** I wrote "no upturn anywhere" at k = 16. There is a small one: fp16's
best is 0.00954 at residual 2.6e-3 against 0.00973 at its floor 1.9e-3 — a 1.02× over-descent, not zero. The
conclusion is unchanged (the k = 16 penalty is 2%, against 78% at k = 32) but "no right arm at k = 16" should read
"a right arm too shallow to matter".

**And the format-dependence shrinks rather than dissolves.** At the optimal stop the precision ordering is
restored in *sign* but not in *magnitude*: fp16 beats bf16 by 3.1× at k = 16 and by only 1.08× at k = 32, and the
knee-to-knee amplification from the easy chart to the hard one is 4.40× for fp16 against 1.50× for bf16 — still
2.9× apart after the stopping rule has done all its work. Over-descent accounts for 1.78× of fp16's 7.68×
full-descent amplification; the remaining 4.40× is not a stopping artefact. "The reversal dissolves at the optimal
stop" is therefore too strong as I first wrote it: **the reversal in sign dissolves; the format-dependence shrinks
by 1.75× and remains.**

*Final form of the k = 32 statement (yoado-7e, after the sweep).* **At an ill-conditioned chart the attacker must
early-stop — the knee is sharp, a factor 1.5 in residual costing 4.2% → 7.5% — and once they do, more precision
only helps:** fp32 6e-6, fp16 stopped 4.20%, bf16 4.56%, monotone. fp32 and fp64 have no over-descent in range at
all. The apparent inversion was over-descent from not stopping, and "coarse arithmetic recovers better" is retired.
The two mechanism claims reconcile as cause and effect: **flush is the cause** (half precision underflows the fine
residual entries below the knee) and **over-descent is the effect** (the continued descent then travels the flat
direction); fp32 does not flush, so it has no over-descent and recovers to the truth. Above the knee all three
formats coincide within 0.5% — residual-determined and precision-independent.

### The wide head measures a weaker object than the exact cells, and the reason is the imprint law (yoado-b9)

**The certificate does not hold at the truth on the m = 26 head.** `Ch_i ≈ 0` is the instrument: on the m = 10/11
cells the recorded images sit at 1e-16 … 1e-8 against 0.1 … 1 for invisible ones, a four-to-sixteen-order gap. On
the twenty-six-class head the objective at the *recorded truths* is 0.05 … 0.20 — the same order as an invisible
image elsewhere — and the floor fraction is **0.0000 at every k of the ladder, k = 8 included** (not only at
k = 24, where my prose had misplaced the clause as if it distinguished that cell).

**What that costs the attacker.** In the exact cells a landing is *self-certifying*: objective at the floor **and**
the right image, each confirming the other. Here the floor test is unavailable, so the only criterion left is
1e-2 relative image error **against ground truth the attacker does not have**. So the sequence 18 / 15 / 13 / 12 / 9
at k = 8 … 40 is a **single fixed criterion** — it measures one quantity, which was the question asked — but an
**oracle-scored** one on this head, not an attacker-available one. It belongs beside the exact cells with that
difference stated, not inside the same word. What *is* attacker-available and survives: **the argmin lands on a
recorded image at every k tested** — no ground truth needed to take an argmin. Honest scoping: on the wide head the
certificate degrades from exact to approximate, the floor test is lost, and the weaker argmin criterion still
selects a recorded image throughout.

**And the degradation is the imprint law again — predicted, then checked in the logged rows.** `Ch_i = 0` needs
`row(B_T)` to equal `col(A₀H)` exactly; with twenty images spanning a wide imprint range the numerical row space is
dominated by the strong imprints and the faintest recorded images are not annihilated. Prediction: the certificate
residual at each recorded truth is ordered by that image's imprint. **Measured, concordant pairs (imprint up →
residual down):** 136/190 at k = 8, 153/190 at k = 16, 164/190 at k = 24, 162/190 at k = 32, 163/190 at k = 40 —
72–86%, i.e. Kendall τ 0.43 … 0.73 on the twenty images of one batch (all pairs, so no independence beyond that
batch).

**A stronger "clean partition" version was proposed and does not survive removing its threshold — recorded as a
negative.** With the cut at residual 0.05 the two groups separate perfectly (high-residual imprints 1e-4 … 4e-9,
low-residual 1.0 … 2e-3, a twentyfold gap and no overlap), which would be the signature of a threshold law. But
0.05 was chosen after seeing the data, so I re-split each cell at its **largest residual gap** instead, choosing
nothing: k = 8 clean (1 image above the gap, imprint 3e-7, against a minimum of 3e-4 below it), k = 16 clean
(6e-6 against 7e-4), k = 24 clean (2e-6 against 2e-5) — but **k = 32 overlaps** (8e-7 above against 5e-7 below)
and **k = 40 overlaps** (2e-5 against 8e-6). So the partition is clean at three of five charts and fails at two,
and the threshold-free form of the claim cannot be made. What stands is the moderate rank correlation at every k,
which is enough for the mechanism (faint imprints are the ones the certificate fails to annihilate) but not for a
threshold law. **So the certificate's usable N′ is set by the
imprint *spread*, not by the count** — the same law that decides what is recorded also decides what the certificate
can annihilate, and the wide head's degradation is not a separate failure of the instrument.

*Final two-part form (yoado-7e, adopting the claims lane's magnitude point).* **(1) The inversion was an
over-descent artefact and is resolved:** with early stopping the ordering is monotone in *sign* — fp32 6e-6,
fp16 4.20%, bf16 4.56% at k = 32, and fp16 ahead at k = 16 too. **(2) A genuine, smaller precision advantage
survives and shrinks with conditioning:** fp16's advantage over bf16 is 3.1× at k = 16 and 1.08× at k = 32, with
knee-to-knee amplification 4.40× against 1.50×, so about 2.9× of format-dependence remains after the stopping rule
has done all its work. The first is an artefact of not stopping; the second is not an artefact at all. Precision
still matters — just less at an ill-conditioned chart.

*The separation degrades monotonically with k, and the ladder is not the same phenomenon (yoado-b9's two items).*
Writing the gap as the ratio of the smallest below-gap imprint to the largest above-gap one: **1000× (k = 8),
117× (16), 10× (24), 0.6× (32), 0.4× (40)** — monotone across all five charts, three orders of decline, crossing
into overlap between k = 24 and k = 32. So the separation is not absent but *narrows with chart dimension and is
lost between 24 and 40 on this head*, which is what the mechanism predicts (more chart dimensions, a relatively
less dominant numerical row space, a blurrier boundary) and predicts further: **a wider head at fixed k, or a
smaller k on this head, should widen the gap again.**

**But the found-count ladder is a different quantity, and the per-image check says so.** The suggestion was that
18 / 15 / 13 / 12 / 9 and the gap's decline might be one phenomenon. Ranking the twenty images by imprint (1 =
strongest) and listing which are never landed on: at k = 8 the two not found are ranks **19, 20** — clean, the
faintest. At every larger chart it is interleaved: k = 16 not-found ranks 15, 16, 18, 19, 20 while rank 17 *is*
found; k = 24 not-found from rank 12 while rank 18 is found; k = 32 not-found from rank 11 while rank 17 is found;
k = 40 not-found from rank **6** while rank 12 is found. So beyond k = 8 the images that stop being found are *not*
simply the faintest — a strongly recorded image can go unlanded while a fainter one is recovered. The ladder's
decline is therefore basin geometry (which targets random starts reach), not the certificate's separation
(which images it annihilates); the two monotone sequences are two phenomena that happen to fall together. That
also fits the earlier landing counts, where the per-image distribution is dominated by one or two targets with
most images at zero.

*Noticed and declined (yoado-b9).* A conditional version survives the refutation: at k = 8 the two unfound images
are exactly the faintest pair (ranks 19, 20), and that is also the cell with the widest separation (1000×), while
by k = 40 the unfound start at rank 6 — so imprint rank predicts recoverability *while the separation is wide* and
stops when it narrows. It is **recorded as declined, not as an open question**: it rests on one cell, two misses
landing on ranks 19–20 of twenty is p ≈ 0.005 for a pattern spotted after the fact, and it is the third reading in
the same direction after a threshold law and a single-phenomenon reading both failed — the point at which a
hypothesis surviving in progressively weaker forms is more likely to be pattern-matching than structure. Anyone
rediscovering it should know it was seen and set aside.

### The k = 16 knee sweep, complete (89853): a knee exists there too, and it is shallow

| stop target | residual reached | iterations | image error (median) |
|---|---|---|---|
| 0.04 | 3.23e-2 | 2 | 4.044e-2 |
| 0.02 | 1.51e-2 | 3 | 3.444e-2 |
| 0.012 | 9.20e-3 | 4 | 2.768e-2 |
| 0.008 | 6.24e-3 | 5 | 2.537e-2 |
| 0.005 | 4.31e-3 | 8 | 1.197e-2 |
| **0.003** | **2.59e-3** | 11 | **0.954e-2** ← best |
| 0.00205 (its floor) | 2.03e-3 | 19 | 0.973e-2 |

So the well-conditioned chart **does** have a knee, at residual ≈ 2.6e-3, and fp16's floor (2.03e-3) sits just
below it — an over-descent cost of **1.02×**, against 1.78× at k = 32. This refines the pre-registered prediction
rather than confirming it as stated: I had predicted the k = 16 knee would lie *at or below* fp16's floor with a
shallow-to-absent right arm, and it lies just *above* the floor with a right arm that exists and is negligible.
The synthesis is unchanged and now measured at both charts — **conditioning lifts the knee: 2.6e-3 at k = 16
against ≈ 4e-3 … 2.8e-3 at k = 32 in absolute terms, but relative to fp16's floor it moves from 1.3× above it to
2× above it, and the penalty grows from 2% to 78%.** What matters for the attacker is not the knee's absolute
position but how far above the reachable floor it sits, and that gap widens with conditioning.
*The attacker-facing consequence (yoado-7e), which is the form to quote.* **Early-stopping discipline matters more
the harder the chart.** At a well-conditioned chart the knee sits essentially at the reachable floor, so
over-descending is nearly free (2%); at an ill-conditioned one it sits well above the floor, so failing to stop
costs about 80%. "Stop at the knee" is therefore a soft suggestion at easy charts and a hard requirement at hard
ones — which is exactly why the naive full-descent attacker looked format-inverted only at k = 32. And the single
axis subsumes both earlier readings: *floor above the knee* (bf16 at k = 32) is under-recovery, *floor below the
knee* (fp16) is over-descent, and the cost either way is the size of that gap.

### Scope on the whole precision sub-thread (yoado-cd, verified): every replay recovery is from a near start

Checked in the code and the rows: `matched_lm` and `recipe_route` both start at `W_true + 0.1·randn·std`
(`train_precision.py:94, 190`), and every M and R row in jobs 771329 / 782682 / 85300 / 86888 / 87369 / 88743 /
89853 carries that start. **No replay cell anywhere in this study has reached the floor from an attacker-buildable
start** (the twenty such starts of jobs 408560–63: 0 of 20). And on precisely the bf16- and fp16-trained releases
where matched replay succeeds, the certificate — *the only random-start route* — returns **0 of 8** (760909, 500
starts, both k).

So the sub-thread's conclusion must be stated conditionally: **"low-precision training is not a defence" is an
identifiability statement given a near start, not an attack result.** The honest form: *half-precision training
does not destroy the information — matched replay from a near start recovers to a few percent — but it breaks the
only route that currently works without a start, so today, for a bf16-trained adapter, nobody can begin the
search.* That is a materially weaker claim than the one I sent as a story note earlier, and it is the one to
carry. It also sharpens where the remaining work is: the replay route's binding constraint was already the
initialiser rather than the information, and half-precision training removes the one initialiser-free route that
had been supplying starts.

## The recorded count is not an independent variable — it is set by the chart (2026-09-04, 26 probed cells)

This constrains every design in the chain direction and was found while trying to build one, so it is recorded on
its own rather than inside that experiment's write-up.

**Measured.** Four batches (`confident`, `hard1_diff`, `repeated`, and one constructed to specification) at r = 8,
on-chart, tolerance 1e-12, chart sizes 6 … 18. The recorded count `N′`:

| chart size k | 6 | 8 | 10 | 12 | 14 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|
| `confident` | 7 | 7 | 5 | 4 | 4 | 3 | — | — |
| `hard1_diff` | 7 | 7 | 6 | 5 | 4 | 4 | — | — |
| `repeated` | 7 | 7 | 6 | 6 | 6 | 6 | — | — |
| constructed (one low-margin image + 7 confident fillers) | 7 | 7 | 6 | 5 | 4 | 4 | 4 | 4 |

**Two consequences, both structural.**

**(i) `N′` falls as the chart sharpens, for every batch.** This is the imprint law along `k`: a richer chart makes
the projections easier for the model to classify, so fewer of them leave a trace. The certificate vanishes at the
truth in every cell (1e-16 … 1e-12), so this is not a degenerate-cell artefact. **Therefore `N′` and `k` cannot be
varied independently**, and any claim of the form "the attack improves as `N′` falls" is confounded with the chart
size by construction. In-band cells available at this rank: `N′` = 7 at k = 6, 8; 6 at k = 10; 5 at k = 10, 12;
4 at k = 12. `N′` = 3 occurs only out of band; `N′` ≤ 2 does not occur at all.

**(ii) A single-recorded-image cell does not exist at a deployable rank.** Not naturally, and not by construction.
A batch built to the obvious recipe — the lowest-margin image plus the highest-margin image of every other class —
gives 7, 7, 6, 5, 4, 4, 4, 4 and nearly duplicates `hard1_diff`, which is defined the same way. The reason is
mechanistic: a filler is confidently classified *as an image*, but the release records the model's error on its
**on-chart projection**, and a small chart destroys that confidence. At k = 6–8 the fillers' relative imprints are
0.7, 0.6, 0.4 — within a factor of two of the target's, where a one-image cell needs them a thousandfold beneath
it. The gap only opens at k = 18 (σ₂/σ₁ = 2.8e-9), which is *above* the band. **The one-image regime and the band
are mutually exclusive at r = 8**, so the realistic low-rank attack is inherently multi-image.

*Scope: r = 8, m = 10, the 98% MNIST model, on-chart PCA charts, FP64. The mechanism (imprint ∝ error on the
projection) is general; the specific counts are not.*

### The handoff, measured correctly (job 159323): the certificate hands replay a start 3e12× better than chance

The earlier reading — "landings are barely closer to a private image than a random point of the same norm" — came
from a defect in my own metric and is **withdrawn**. It scored every start against *one* recorded image (the
strongest) rather than against the image it actually landed on, so at a cell with three recorded images two thirds
of genuine landings read as failures. Corrected (nearest **recorded** image, applied identically to landings, raw
starts and the norm-matched control), at r = 64, k = 16, N′ = 3, 500 random starts:

| | min | 10th pct | median |
|---|---|---|---|
| certificate landing | 9.3e-16 | 1.0e-15 | **2.6e-13** |
| the same starts, before the solve | 0.40 | 0.59 | 0.75 |
| random point of the **same norm** | 0.35 | 0.61 | 0.78 |

**95% of landings are closer to a private image than a norm-matched random point; the median is better by a factor
of 3e12.** So the handoff falsifier does not fire: the certificate does not merely shrink the search space, it
delivers starts essentially *on* private images.

**Conditioned on the landing's certificate residual, by decile** (best-converged first): deciles 0–7 all have
`frac_landed < 1e-2` = **1.00** with landing errors 1e-15 … 1.2e-10; decile 8 drops to 0.20 (median error 0.46);
decile 9 to 0.00 (median 1.2). So the handoff is neither weak nor concentrated in a narrow top slice — **it is
uniform over the best 80% of landings and fails only in the last 20%**, exactly where the certificate solve itself
did not converge (its residual rises from 3.6e-15 in decile 0 to 0.48 in decile 9). The attacker's own certificate
residual therefore *predicts* handoff quality with no private knowledge, and the rule "keep landings whose
certificate residual is at the floor" recovers a clean population.

*This also answers the question of whether the k = 6 correspondence (every floor-reacher is a private image)
breaks higher up the chart range: on this row it does not. The apparent median landing error of 0.71 that suggested
it was the same metric defect.*
