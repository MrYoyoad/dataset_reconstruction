# Handover — 2026-09-03 00:23

## State
Branch `step1-activation-rescore-retrieval`. Exact-inversion thread (external bundle `framework_rev10.pdf`
/ `results_rev9.pdf` / `audit_rev9.pdf`, still Mac-only, not yet in `papers/`): testbed built, validated,
headline measured, and **two of my own earlier claims withdrawn** after an adversarial review found the
bug that caused them. Full write-up `experiments/exact_inversion/RESULTS.md`; retraction in `NOTES.md §2`.
Commits 38fec3b → 5762045 → 204ec67 → 2763053 → daa4163 → 276cfb3 → 3adb1b3 → 9536898.

## Settled results
- **fwd_check = 7.4e-16 … 8.5e-16.** The span-adapted simulator reproduces the real release at the truth,
  so the normal form holds operationally: the release is a function of the candidate data and of
  `X = A₀U` (rN numbers) alone.
- **HEADLINE: 49/49 phase-diagram cells recovered, one attempt each, median residual 8.6e-31**
  (N,k ∈ 2..14, r=16, T=400, start 10% off). The certificate-only diagram is exactly 0 above `k = r − N`;
  exact inversion recovers on both sides. `figures/exact_inversion/phase_diagram_comparison.png`.
  One seed per cell, so the grid shows the boundary does not bind but does not measure a failure rate.
- **Adam: locally identifiable, but ~10⁵ worse conditioned.** `σ_min(J)` = 6.3e-3 … 7.3e-3 sits inside
  the range of the SGD cells that converge; `cond(J)` = 1.4e7 … 3.7e8 vs 1.2e2 … 2.9e3 for SGD. The
  certificate does not weaken under Adam, it ceases to exist (`rank B_T = r` ⇒ `C ≡ 0`, so `eps_inv`
  reads as a perfect certificate while being vacuous — always read `cert_norm`/`cert_vacuous`).
- **Preconditioning does NOT fix Adam** (job 452904): unscaled `λI` reaches residual 7.6e-3 in 400 iters;
  Marquardt `λ·diag(JᵀJ)` reaches 2.2e-1 and 9.5e-3 in 200; Marquardt + staging the `A₀` block leaves
  1.1e-2. Not a block-scaling mismatch. Next candidates: trust region, or reformulating so the Adam
  moment buffers are not differentiated through.

## Withdrawn — do not resurrect
1. *"A validation cell does not reproduce the bundle"* — RETRACTED. Post-fix, same seed/start/single
   restart/no staging, it recovers to 6.6e-16 in 14 iterations. `results_rev9.pdf` §3b **does** reproduce.
   The staged-schedule explanation is also wrong: `--stage-x 10` recovers too, in 53 iterations (slower).
2. *"15 of 49 cells needed 4 restarts"* — WITHDRAWN. Those failed pre-fix and were rescued by a run that
   changed **two** things (code version and restart count). Varying one: post-fix at `restarts=1` they
   recover 15/15, median residual 8.0e-31, median 17 iterations (job 456630). False failures from the QR
   sign discontinuity, exactly the "systematic false-failure that under-reports the basin" the review
   predicted.

## Next step(s)
1. **Job 459111 (`ei6_basin_postfix`) is the open one.** The basin arm was measured PRE-FIX, so its edge
   (recovers to 24% start error, fails at 36%) is a **lower bound** — the same bug narrowed it. This job
   re-measures post-fix at `restarts=1`, noise ∈ {0.1,0.2,0.3,0.4,0.5,0.7,1.0}, 3 seeds. When it lands,
   update the basin table in RESULTS.md and the basin figure, and correct STATUS.
2. **Initialiser arms** (jobs 408560 random / 408561 span / 408562 cert / 408563 spananchor), 13 of 20
   rows: **one recovery so far**, span estimator seed 3, image error 4.0e-3 at residual 9.1e-7 — inside
   the 1e-2 tolerance but far from the ~1e-15 a perturbed-truth start reaches, so it entered the basin and
   was still converging. The other 12 fail. NOTE these also ran post-fix, so they are clean.
   The certificate-anchor arm is the mechanism demonstration: it drove `‖Cφ(ψ(w))‖` to 5.0e-8 and still
   landed 0.70 away in image error, because at `k=12 > r−N=8` the certificate-consistent set is a
   4-dimensional manifold per image.
3. rsync the three bundle PDFs from the Mac into `papers/`.

## Open threads / gotchas
- **`set +u` before `conda activate`** in any job script here, or the job dies in 9 s with a near-empty
  stdout (`ADDR2LINE: unbound variable`).
- **Never edit the script under a running multi-cell job** — the runner re-launches python per cell.
- The session scratchpad is **not visible from compute nodes**; submit inline scripts via `bsub` stdin.
- **A pre-fix failure is not a result.** Anything measured at 12fa60d / 38fec3b that FAILED is void; the
  successes are still fine (none of the six fixes can turn a failure into a false recovery).
- Pre-existing uncommitted `figures/recon_showcase/*.png` + `results/recon_showcase_sweep.csv` predate
  this thread and were left untouched.

## Pointers
- Write-up `experiments/exact_inversion/RESULTS.md`; retraction `NOTES.md`; theory
  `notes/exact_lora_inversion_framework.md`.
- Analyze: `python experiments/exact_inversion/analyze_exact_inversion.py` (CPU, globs all
  `results/exact_inversion/*.jsonl`).
- Submit: `bsub -q long-gpu -gpu "num=1" -R "rusage[mem=8192] select[ngpus>0]" -J ei_x -o scripts/wexac_logs/ei_x_%J.out -e scripts/wexac_logs/ei_x_%J.err bash scripts/run_exact_inversion_wexac.sh <stage> [arg]`
- New solver flags: `--lm-scale {identity,marquardt}`, `--stage-x N`, `--solver {lm,lbfgs}`.
