# NOTES — disagreements with the theory documents, and open questions

Rule (task spec item 5): do not edit `framework_rev10.pdf` / `results_rev9.pdf`. Anything that looks
wrong or that does not reproduce goes here with evidence. Every number is provisional (†) and carries
its job id; the JSONL lines under `results/exact_inversion/` hold seed, git hash, command line and host.

## 1. Nothing so far contradicts the theory

`fwd_check` (the simulator reproducing the release at the truth) is 7.4e-16 – 8.5e-16 across the
validation cells†, so the reduced span-adapted simulator of Theorem 1 *is* the training map: the
release really is a function of the candidate data and of `X = A_0 U` alone. `rank C = r − N` and
`eps_inv ~ 1e-15` reproduced in every cell seen so far†. No statement in the framework is contradicted.

## 2. A validation cell does NOT reproduce the finite-difference result — basin, not identifiability

`results_rev9.pdf` §3b reports `(k, N, r−N) = (6, 12, 4)`, `T = 400`, `η = .01`, start error 0.24 →
final error 5e-16, residual 6e-16. Our backprop rerun of that cell from start error 0.166 (init-noise
0.24 in latent space, seed 1) **stalls at residual 5.7e-4 with median image error 6.2e-2**† (job 395496).
The trace is a genuine local minimum, not an early stop: the residual plateaus from iteration ~7 while
the LM damping λ climbs from 1e-2 to 1e+4 with no accepted step.

Two differences from the finite-difference run, both plausible causes, neither yet isolated:

1. **A different world / start draw.** The bundle prototype (`primitive3_exact_inversion.py`) uses a
   numpy RNG stream and a different generator bias; ours is a torch stream. So this is a *different
   cell of the same family*, and the disagreement may be seed-level, which is exactly what a basin
   study is supposed to measure. The other two validation cells at comparable deformation reproduce to
   1e-15†.
2. **The prototype staged the optimisation; ours does not.** `invert2(..., iters_x=10, iters=40)` runs
   ten LM iterations on `X` **alone** (latents frozen) before going joint. Our LM is joint from
   iteration 0. If the staging is what buys the larger basin, that is a *finding about the attack*, not
   a detail: it says the nuisance block `X` should be solved first because it is linear-ish given the
   data, and it makes the reported "converges from 10–15% off" contingent on the schedule.

**Action (queued, deliberately not applied while jobs are running):** add a `--stage-x` option that
mirrors the prototype's schedule and rerun this cell with and without it, same seed. Until then the
honest statement is: *one of the five finite-difference cells does not reproduce under joint LM from a
start of comparable size; the failure is an optimisation failure (residual 5.7e-4, far from zero), not
an alias.* Do not quote the bundle's "converges from within 10–15%" as reproduced.

## 3. The span estimator is much weaker in this testbed than in the audit's own numbers

`audit_rev9.pdf` A1(iv) reports mean principal angles of `Ĥ = row(P_row(B_T) A_T)` to the private span
of 18–29° at `n = 256/768`. In the validation testbed (`n = 96`, `r = 16`) we measure 52.0° mean /
71.6° max at `N = 8` and 59.4° mean at `N = 12`†. Not a contradiction — the audit's bound scales with
`c_T σ_0 (√n + √N) / (σ_min(X) σ_min(Ω))` and its runs used larger deformation (1.5–2.5 vs 0.26–0.39
here) — but it matters for the plan: at this work point the span estimator is a *weak* initializer, and
the `--init span` / `--init spananchor` arms should be read against a random-subspace null (≈78–83°)
rather than against the audit's numbers.
