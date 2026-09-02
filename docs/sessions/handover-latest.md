# Handover — 2026-09-03 02:31

## State
Branch `step1-activation-rescore-retrieval`. The **exact-inversion thread is COMPLETE** — every arm that was
listed as RUNNING in the previous note has landed, and the session produced a new headline that reframes the
whole thread: **the simulation channel has its own capacity law, `k < m + r − N`.** Authoritative write-up is
`experiments/exact_inversion/RESULTS.md` (read it first; it is more detailed than STATUS); disagreements with the
external bundle in `experiments/exact_inversion/NOTES.md`. STATUS.md's top section has been rewritten to match.
The bundle PDFs (`framework_rev10.pdf` / `results_rev9.pdf` / `audit_rev9.pdf`) are still Mac-only, not in `papers/`.
Every number is provisional (†).

## Done this session
- **NEW HEADLINE — a capacity law, CONFIRMED (jobs 467914, 469120).** The released `B_T = P_T Xᵀ` is `m × r` but
  has rank `N`, so it carries only `N(m+r−N)` independent numbers; per image the budget is `m + r − N`. Confirmed
  at `N = 4, 8, 12` with predicted thresholds 32 / 28 / 24, each bracketed by its own last success and first
  failure, and **the brackets are disjoint** (`N=12` collapsed at `k=26` while `N=4` is healthy at `k=30`), which
  rules out a fixed-`k` explanation. `σ_min(J)` at the truth falls 13–14 orders across one step of the sweep.
  Past the line the failures are **genuine non-identifiability** — residual at the reproduction floor (~1e-30)
  with the wrong image — the **only true aliases found anywhere in this session**.
  Figure `figures/exact_inversion/capacity_law.png`.
  Framing: certificate channel `k < r − N` vs simulation channel `k < m + r − N`; simulation buys
  `(m+r−N)/(r−N) = 3.5×` at `N=8`, and the **released head width `m`** is what buys it. Falsifiable prediction,
  UNTESTED: widening `m` widens reach **linearly** at fixed rank.
- **It also scopes the 49/49 grid.** With `k < 36 − N`, the grid's worst corner (`N=14`) has ceiling `k<22` while
  the grid only reaches `k=14` — the whole grid lies **strictly inside** the capacity region and could not have
  found this boundary. 49/49 stands, as an existence result scoped to that regime.
- **Initialiser arms COMPLETE — †1 of 20, and this is the binding constraint on the attack** (release-only starts,
  `k=12, N=8, T=1500`, 5 seeds × 4 arms, jobs 408560-63). random 0/5, spananchor 0/5, cert 0/5; span 1/5 (seed 3,
  image error 9.2e-3 at residual 9.1e-7 — inside tolerance but still converging, not finished). The **cert-anchor
  failure is the clean demonstration**: its pre-solve reaches `‖Cφ(ψ(w))‖` of 1.1e-6 and 5.0e-8 and those points
  sit **1.0–1.2 away** in image error, because at `k=12 > r−N=8` the certificate-consistent set is a 4-dimensional
  manifold per image. Against perturbed-truth starts recovering from 0.86, the basin is strongly **anisotropic** —
  wide along truth-directions, **not attacker-reachable**. Measured confirmation that a learned/population prior
  must supply the initializer.
- **Basin post-fix (job 459111, `restarts=1`): 16 of 17 recover, worst-case start error 0.86.** The single failure
  hit the 80-iteration cap **still descending** at residual 8.0e-5 — a budget limit; **the edge was NOT located.**
  The old "24% / first failure at 36% / restarts are the currency" table is withdrawn (pre-fix QR-seam artefact).
- **Adam (jobs 466915, 467622, 452904): identifiable at the truth at every scale tested, including the full-size
  `n=96` release** (gate `‖res(truth)‖` exactly 0.0, full column rank). Solution conditioning is worse than SGD's
  and worsens with size: ~10–20× at `n=32`, ~400× at `n=96` (8.0e5 vs 2.0e3). Preconditioning does **not** fix it
  (Marquardt `diag(JᵀJ)` scaling is worse than unscaled damping). So Adam gives the defender **two moderate
  obstacles at scale — a conditioning penalty and a much smaller basin — not non-identifiability.** Whether either
  is erodable by better optimisation is UNTESTED.
- **Near-duplication (jobs 471272, 473055, 474132).** A numerically-degenerate-but-genuinely-distinct pair
  **contaminates the certificate**: `‖CH‖` peaks at 7.8e-7 where `rank B_T` collapses (six orders above clean) and
  returns to ~1e-15 at exact duplication — a non-monotone band with clean endpoints. `σ_N(B_T)/σ_1(B_T)` is the
  graded detector; `‖CH‖` is fragile. But the simulation channel's apparent threshold near separation 0.2 is a
  **SOLVER FLOOR, not an information boundary**: the same failing cell recovers to 1.8e-14 at residual 8.9e-31
  with 10× the budget. A defender **cannot** buy privacy by perturbing-and-copying a record — it costs the
  attacker compute, not access. The only fundamental alias is **exact** duplication.
- **Withdrawn during the session, do not resurrect:** the false non-reproduction of `results_rev9.pdf` §3b; the
  "15 cells needed restarts" caveat; "Adam defends by conditioning"; "only the basin differs"; the vacuous "blend"
  diagnostic; and the near-duplicate "cliff" (a sampling artefact). Each traced either to the one QR-seam defect
  in our own code or to reading a diagnostic at the wrong point. That is why the current numbers are trustworthy,
  not a reason to distrust them.

## Next step(s)
1. **Test the capacity law's falsifiable prediction**: widening `m` should widen the attack's reach **linearly**
   at fixed rank. Same sweep shape as job 469120, holding `r` and `N` fixed and scanning `m`; read `σ_min(J)` at
   the truth for the collapse, not `frac_recovered` (the aliases sit *under* the 1e-2 tolerance).
2. **Settle whether Adam's small basin is solver-fixable**: the Adam analogue of the SGD basin sweep (job 459111's
   shape) plus at least one **trust-region** attempt (or Gauss-Newton with line search, or reformulating so the
   Adam moment buffers are not differentiated through).
3. **Locate the SGD basin edge** with a higher iteration cap — the 17th run was still descending at the cap, so
   the edge is unmeasured and must not be reported as "just past 0.86".
4. **The real-data step**: frozen DINO/CLIP features with an adapter head. The original task spec explicitly
   scoped this as a **separate task**.
5. **rsync the three bundle PDFs** from the Mac into `papers/`.

## Open threads and gotchas
- **`set +u` before `conda activate`** in any job script here, or the job dies in 9 s with a near-empty stdout
  (`ADDR2LINE: unbound variable`).
- **Never edit a script under a running multi-cell job** — the runner re-launches python per cell.
- The **session scratchpad is not visible from compute nodes**; submit inline scripts via `bsub` stdin.
- **A pre-fix failure is not a result.** Anything measured at `12fa60d` / `38fec3b` that FAILED is void (QR sign
  discontinuity, fixed at `5762045`).
- **Always read the residual** to tell a search failure from a true alias: residual at the reproduction floor
  (~1e-30) with a wrong image = alias; residual far above the floor = the solver simply did not get there.
- `figures/exact_inversion/` and `experiments/exact_inversion/` are owned by the user; a doc-only session must
  not edit them.
- Pre-existing uncommitted `figures/recon_showcase/*.png` + `results/recon_showcase_sweep.csv` predate this thread.

## Pointers
- Authoritative write-up `experiments/exact_inversion/RESULTS.md`; retraction/open questions `NOTES.md`; theory
  `notes/exact_lora_inversion_framework.md`.
- Figures: `capacity_law.png`, `phase_diagram_comparison.png`, `phase_diagram_exact.png`, `basin_curve.png`
  (all under `figures/exact_inversion/`).
- Analyze: `python experiments/exact_inversion/analyze_exact_inversion.py` (CPU, globs all
  `results/exact_inversion/*.jsonl`; each line carries seed, git hash, command line, host).
- Submit: `bsub -q long-gpu -gpu "num=1" -R "rusage[mem=8192] select[ngpus>0]" -J ei_x -o scripts/wexac_logs/ei_x_%J.out -e scripts/wexac_logs/ei_x_%J.err bash scripts/run_exact_inversion_wexac.sh <stage> [arg]`
- Solver flags: `--lm-scale {identity,marquardt}`, `--stage-x N`, `--solver {lm,lbfgs}`, `--jac-at-truth`.
