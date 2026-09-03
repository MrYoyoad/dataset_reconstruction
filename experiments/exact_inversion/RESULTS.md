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
is a visually identical image.

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

**The deciding cell is still not decidable in FP64, and that caveat stands even though the law is now
derived.** For `k > m + r − N` strictly, ill-conditioning cannot
masquerade as rank deficiency: the containment argument forces `σ_min = 0` exactly. But *at* equality the
count permits full rank, and `σ_min ≈ 8e-19` with `cond ≈ 2e18` is precisely the regime where FP64 cannot
separate a rank-deficient problem from an identifiable one with `cond ≈ 1e18`. The image error cannot
discriminate either, since attainable accuracy there is about `ε·cond`. So the law is a theorem; the
strictness is an empirical off-by-one whose deciding cell is undecidable at this precision.

**What the law says.** The released `B_T = P_T Xᵀ` is `m × r` of rank `N`, so it carries
`N(m + r − N)` independent numbers, however large `m × r` looks. Divide by the `N` images and each image
gets a budget of `m + r − N` numbers. An image with more degrees of freedom than that cannot be pinned
down, and the failure is genuine non-identifiability — residual at the reproduction floor, wrong image.
So the honest capacity statement for the whole attack is:

| channel | boundary | provenance | failure mode past it |
|---|---|---|---|
| certificate (Primitive 1-2) | `k < r − N` | **†bundle** (`results_rev9.pdf` Fig. 1) — **not reproduced in this repo**; there is no `C`-only recovery experiment under `results/exact_inversion/` | aliases; `C` is blind |
| simulation (Primitive 3) | `k < m + r − N` | measured here (jobs 467914, 469120, 479587, 479684) | `J` at the truth is rank-deficient; images still sub-percent |

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
recovered for any practical purpose. The residual floor itself degrades near the boundary, so the bins
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

**The reading, hedged to what is measured.** The capacity boundary marks where **exact** recovery stops,
not where **recognisable** recovery stops: on the paths measured the alternative solutions are still
recognisable reconstructions at 1.4%-3.5% relative error. Whether the fibre reaches unrecognisable points
is **not established**, and one cell was still rising when its run ended — at 3.5% the degradation is
visible rather than a rounding difference, so this should not be read as "the alternatives are always
near-perfect".

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
step count off by **one step in four hundred** — by **twenty-three orders of magnitude**. The ordering is
monotone in the size of the recipe error, so the residual is not merely a detector but a graded objective
one could minimise over candidate recipes.

**Why this is usable by an attacker, which is the whole point.** The image-error column is *not observable*
to an attacker — they do not have the private images, and every reconstruction number in this file is a
diagnostic available only to the experimenter. The **residual is** observable: it is computed from the
released factors and the candidate alone. It is a Cauchy-type criterion — it certifies convergence without
any reference to the limit — and it is therefore a legitimate recipe *selection* rule rather than a
post-hoc diagnostic. Any claim built on reconstruction quality would not be.

## Step 9 — the recipe can be MEASURED, not assumed (job 485912)

Proposed by the user: the attacker holds the released adapter and can keep training it on data of *their
own* choosing. Under a scalar-linear update one further step gives exactly `ΔB = −η·gB` with
`gB = D(A_T H′)ᵀ`, and the attacker knows the released factors, their own probe features `H′` and their own
labels, hence knows `gB`. So `η` is a one-dimensional least squares, using **no private data at all**.

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
