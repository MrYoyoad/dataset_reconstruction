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
`9e-3`. That is the signature of an **alias**: the release is reproduced exactly by a point that is not
the truth. Every failure earlier in this study had a non-zero residual and was a search failure; these are
the first cells where the release genuinely does not determine the data. (Note the errors sit just under
the 1e-2 recovery tolerance, so a naive `frac_recovered` would score them as successes. They are not
successes — they are a flat direction that happens to be locally weak. This is a case where the tolerance
is the wrong instrument and `σ_min` is the right one.)

**The boundary matches the refined count, not the naive one.** `Nk ≈ mr = 320` was the first guess. The
observed collapse is between 208 and 256, which `mr` does not predict — but the derivation check's
refinement does: `B_T = P_T Xᵀ` has rank `N`, so it carries only `N(m + r − N)` independent numbers, which
at `N = 8` is `8 × 28 = 224`. The observed boundary brackets 224. Equivalently the law is a **per-image
budget**:

```
        k  <  m + r − N          (per-image degrees of freedom below per-image released information)
```

which at `N = 8` predicts `k* = 28`, between the last success (`k = 26`) and the first failure (`k = 32`).

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

**What the law says.** The released `B_T = P_T Xᵀ` is `m × r` of rank `N`, so it carries
`N(m + r − N)` independent numbers, however large `m × r` looks. Divide by the `N` images and each image
gets a budget of `m + r − N` numbers. An image with more degrees of freedom than that cannot be pinned
down, and the failure is genuine non-identifiability — residual at the reproduction floor, wrong image.
So the honest capacity statement for the whole attack is:

| channel | boundary | failure mode past it |
|---|---|---|
| certificate (Primitive 1-2) | `k < r − N` | true aliases; `C` is blind |
| simulation (Primitive 3) | `k < m + r − N` | true aliases; `J` at the truth is rank-deficient |

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

## Step 2 (attacker-available initialisers) — RUNNING; 2 of 5 seeds per arm, all four fail

`random` / `span` / `cert` / `spananchor`. A first submission was **killed and discarded** rather than
reported: its restarts re-seeded only the latents, leaving 57% of the unknown vector frozen at a stale
value, so "8 restarts failed" would really have meant "one start, jittered 8 times". Since the
framework's headline is *what a learned decoder must supply is an initializer*, that arm has to be able
to support the claim it is quoted for.

All four arms start from a **global** (release-only) start, not from a perturbed truth. Two seeds each
so far, k=12, N=8, T=1500, 8 restarts, cell where `k > r − N` so the certificate is blind by Primitive 2:

| init | seeds | start err | fraction recovered | residual | pre-solve diagnostic |
|---|---|---|---|---|---|
| random | 2 | 0.94, 0.86 | 0, 0 | 1.6e-2, 9.8e-2 | — |
| span | 2 | 0.82, 0.84 | 0, 0 | 2.4e-1, 3.0e-1 | out-of-span energy 0.477, 0.481 |
| spananchor | 2 | 1.04, 0.58 | 0, 0 | 3.7e-1, 1.3e-1 | out-of-span energy 0.423, 0.438 |
| cert | 2 | 1.04, 0.74 | 0, 0 | 1.1e-1, 3.1e-1 | **‖Cφ(ψ(w))‖ driven to 1.1e-6** |

**The `cert` row is the informative one, and it is a clean demonstration of the alias structure rather
than a solver complaint.** Its pre-solve genuinely succeeded: it drove the certificate residual to 1.1e-6,
i.e. it found a point that satisfies `Cφ(ψ(w)) = 0` to six digits. That point sits at a **1.04 relative
image error** from the truth. This is exactly what Primitive 2 predicts at `k = 12 > r − N = 8`: the
certificate-consistent set is a manifold of dimension `k − (r − N) = 4` per image, so being
certificate-consistent carries essentially no information about *which* point on it you are at. The
subsequent exact inversion, started on that manifold, does not descend to the truth either — it ends at
residual 1.1e-1, nonzero, so a search failure and not an alias of the full release.

The span-based arms fail differently: their pre-solve only reaches ~45% out-of-estimated-span feature
energy, which is the span estimator's own 52° error showing up as a floor on how well any candidate can
be aligned to it.

Preliminary reading, 2 of 5 seeds: **the initialisers an attacker can actually build from the release do
not reach the basin**, while a perturbed-truth start of up to 24% does. That is the framework's division
of labour stated as a measurement rather than as a hope, and it is the strongest argument in this session
for why a learned or population prior is the missing piece. It is not yet a result: 8 of 20 rows.

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

- The failure is **confined to the duplicated pair** at every separation tested below 0.074, and the two
  reconstructions are **equidistant from both originals** (`err_to_self` 3.78e-2 vs `err_to_other` 3.82e-2)
  — blends of the pair, not swaps of it.
- The other `N−2` images stay under the recovery tolerance but **degrade by eleven orders**, from `2e-15`
  in the clean control to `2e-4`-`1e-3`. They still leak, but not to machine precision.
- Residuals are `1e-8`-`1e-9`, **not** at the reproduction floor, so these are not clean aliases; a longer
  budget might sharpen them and that is untested.

So the honest statement is the softened one the derivation check anticipated: **near-duplication
contaminates the algebraic channel and mutually aliases the duplicated pair, while the remaining `N−2`
records still leak (at reduced fidelity)**. It changes which records are protected, and it does not
protect the batch.

**Gap in the sampling, being filled.** The sweep jumps from the clean control (separation 1.06, full
recovery) straight to `ε = 0.3` (separation 0.074, pair unresolved). The transition sits in that gap and
was never sampled; job 473055 fills it at `ε = 5, 3, 2, 1, 0.5`. Until it reports, "how similar is too
similar" is unmeasured. The `α` blend coefficients recorded in the tight band are meaningless there (the
line through the two originals degenerates as they coincide) — use `err_to_other` instead, as above.
