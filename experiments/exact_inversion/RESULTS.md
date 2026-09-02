# RESULTS — exact inversion of the LoRA training map (backprop through the recipe)

Synthetic FP64 testbed, WEXAC L40S/A40, 2026-09-02. Theory: `notes/exact_lora_inversion_framework.md`
(from `framework_rev10.pdf`); finite-difference baseline: `results_rev9.pdf` §3b.
Script `lora_exact_inversion.py`, runner `scripts/run_exact_inversion_wexac.sh`, raw rows
`results/exact_inversion/*.jsonl` (each line carries seed, git hash, command line, host).
**Every number is provisional (†).** Post-fix numbers are at git `5762045`; the phase-diagram sweep and
the near-basin arm were produced at `38fec3b`, before six defects were fixed (see "Corrections" below) —
none of those six can turn a failure into a false recovery, but they can and did turn recoveries into
false failures, so pre-fix *failures* are the ones to distrust.

World: `k`-dim tanh generator → 64-dim image → tanh encoder → `n=96` features → softmax head `m=20`,
LoRA `r=16`, `B₀=0`, Gaussian `A₀`, plain SGD. Attacker gets `(A_T, B_T)`, `W₀`, φ, ψ, the labels and the
recipe; never `A₀`, `H`, or the trajectory. "Recovered" = relative image error < 1e-2 for **every** image.

## Step 1 — the simulator is the training map, and the released factors invert

| k | N | r−N | T | lr | deformation | start err | fwd_check | final err (med) | residual | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 8 | 8 | 400 | .01 | 0.39 | 0.085 | 7.8e-16 | 2.2e-15 | 8.9e-31 | recovered |
| 6 | 12 | 4 | 400 | .01 | 0.26 | 0.166 | 8.5e-16 | 6.2e-02 | 5.7e-04 | optimisation failure |
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

The one non-reproduction is written up in `NOTES.md §2`: it is a genuine local minimum (residual plateaus
while LM damping climbs 6 orders with no accepted step), and the finite-difference prototype it is
compared against used a *staged* schedule (X first, then joint) that this run does not. Do not quote the
bundle's "converges from within 10–15%" as reproduced.

## Step 4 + 5 — the phase diagram: the certificate's boundary does not bind the exact inversion

`figures/exact_inversion/phase_diagram_comparison.png` (the side-by-side against the certificate-only
diagram; `phase_diagram_exact.png` is the single panel). 49 cells, `N, k ∈ {2,…,14}`, T=400, start 10% off.

**49 / 49 recovered**, median residual 8.6e-31, zero cells with a residual above the reproduction floor.

This is the figure the exercise was for. The certificate-only diagram (`results_rev9.pdf` Fig. 1) is
**exactly 0 above the line `k = r − N`** — above it the certificate has fewer rows than the manifold has
dimensions and every run lands on a true alias. Exact inversion recovers **on both sides of that line**,
including the extreme corners (`N=14, k=14`, where the certificate has 2 rows against 14 tangent
directions). The `r − N` budget is a boundary for *one channel*, not a bound on what the release leaks.

Two honest qualifications:

- **15 of the 49 cells needed restarts** (marked `*` in the figure). At `restarts=1` they failed, all 15
  with a *nonzero* residual (3e-5 … 1e-2) — optimisation failures, never aliases. Re-run with 4 restarts
  that re-seed the whole unknown vector, all 15 recovered to ~1e-15. So the pre-fix failure pattern was a
  property of the search, not of the release. Failures concentrated at large `N` (4 of 7 at N=12 and
  N=14, 1 of 7 at N ≤ 8), which is the honest statement of where the search gets harder.
- One seed per cell. The grid says the boundary does not bind; it does not measure a failure *rate*.

## Step 2 — the basin (k=12, N=8, T=1500, lr=.03, deformation 0.96, up to 8 restarts)

| latent init-noise | median start err | seeds | fraction of seeds fully recovered | mean restarts used |
|---|---|---|---|---|
| 0.05 | 0.031 | 5 | 1.00 | 1.0 |
| 0.10 | 0.063 | 5 | 1.00 | 1.0 |
| 0.15 | 0.100 | 4 | 1.00 | 2.5 |
| 0.20 | 0.123 | 5 | 0.80 | 2.8 |
| 0.30 | 0.183 | 4 | 1.00 | 3.2 |
| 0.50 | 0.359 | 1 (running) | 0.00 | 8.0 |

`figures/exact_inversion/basin_curve.png`. At a **fully trained** adapter (deformation 0.96) the basin
extends to at least a 24% start error, past the ~10–15% the finite-difference prototype reported, and
the first clean failure appears at a 36% start error. The cost of distance is restarts, not accuracy:
whenever it converges it converges to ~1e-14, never to something in between. Restarts-used is the
better-behaved signal than the binary outcome. Arms at 0.15/0.20/0.30/0.50 were still accumulating seeds
when this was written — treat the per-row seed counts, not the fractions, as the state of evidence.

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
| adam (n=32, k=6, N=4) | 832 | 536 | 6.3e-3, 6.9e-3, 7.2e-3 | **1.4e7 … 3.7e8** | 23, 28, 400 | not converged |

`σ_min(J) ≈ 7e-3` for Adam sits **inside the range the successful SGD cells show**, so the simulator
Jacobian has full column rank there: by Proposition 6 the Adam cell is **locally identifiable**. What
differs is `cond(J)`, by four to six orders of magnitude, driven by `σ_max` rather than by any small
singular value — which is what a coordinatewise `1/(√v̂+ε)` rescaling does to sensitivities. At 400 LM
iterations (8x the original budget, job 423887) the run was still descending, residual 7.6e-3.

So the correct statement is: *the Adam release is locally identifiable from the simulator, and plain
Levenberg-Marquardt does not solve it because the problem is ~10⁵ times worse conditioned than the SGD
one.* The indicated fix is preconditioning, not more iterations: Marquardt's `diag(JᵀJ)` scaling instead
of the unscaled `λI` damping used here, and rescaling the `A₀` block against the latent block. Untested.

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

## Step 2 (attacker-available initialisers) — RUNNING, one row so far, no claim yet

`random` / `span` / `cert` / `spananchor`. A first submission was **killed and discarded** rather than
reported: its restarts re-seeded only the latents, leaving 57% of the unknown vector frozen at a stale
value, so "8 restarts failed" would really have meant "one start, jittered 8 times". Since the
framework's headline is *what a learned decoder must supply is an initializer*, that arm has to be able
to support the claim it is quoted for.

First row in (`cert`, seed 1): the certificate anchor left the start at a **1.04 relative image error**
— i.e. it did not pull a global start anywhere near the truth — and the inversion then failed with a
nonzero residual (1.1e-1). One row of twenty; it is consistent with the framework's claim that an
initializer is the missing ingredient, and it is not yet evidence for it.

One measurement already constrains it: the released span estimator `Ĥ = row(P_{row(B_T)}A_T)` sits at
**52° mean principal angle** to the private span at the N=8 work point (59° at N=12), against ~78–83° for
a random subspace. Informative, but far from the 18–29° the audit reports at n=256/768 — at this work
point the span channel is a weak initializer, and `NOTES.md §3` records why the two are consistent.

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

1. **The exact channel is not budget-limited.** `r − N` bounds the certificate, not the leakage. Any
   defense argued from "the adapter only has `r − N` independent rows" is arguing about one primitive.
2. **The crux is confirmed to be the basin**, and it is more forgiving than the prototype suggested:
   24% at full deformation, with restarts as the currency. Every failure observed anywhere in this
   session had a nonzero residual — **not one alias**. Failures here are search failures.
3. **The initializer question is the open one**, and it is now the only arm whose result would change
   the story. That is the right place for a learned/population prior, which is what the framework says.
4. **Report `cert_norm` beside `eps_inv` forever.** A zero certificate scores perfectly on the natural
   metric.
5. **Adam defends by conditioning, not by hiding the data.** It removes the algebraic channel outright
   (`C ≡ 0`) and leaves a system that is still locally identifiable but ~10⁵ times worse conditioned.
   That is a much weaker kind of defense than non-identifiability, and it is the kind that better
   optimisation erodes. Any claim that "Adam is safe" has to be argued against a preconditioned solver.
