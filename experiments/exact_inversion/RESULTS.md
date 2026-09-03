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
634238's rows are void; 706597 reruns Part A with rows written as produced.

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
