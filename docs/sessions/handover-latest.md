# Handover — 2026-09-02 21:53

## State
Branch `step1-activation-rescore-retrieval`. New thread opened today from an external theory bundle
(`framework_rev10.pdf` / `results_rev9.pdf` / `audit_rev9.pdf`, still Mac-only, not yet in `papers/`):
**exact inversion of the LoRA training map** — simulate the known recipe as the forward model and solve
`Recipe_T(φ(ψ(wᵢ)), X) = (B_T, A_T U)` for the latents and `X = A₀U`. Testbed built, validated, and the
headline measured; two arms still running on WEXAC. Commits: 38fec3b (testbed) → 5762045 (six review
fixes) → 204ec67 (results + figures) → 2763053 (docs/memory).

## Done this session
- `experiments/exact_inversion/lora_exact_inversion.py` — FP64, **true backprop through the unrolled
  training loop**, LM solver with an autograd (`torch.func.jacfwd`) Jacobian (LBFGS fallback), SGD
  (span-adapted, unknowns = latents + X) and Adam (unknowns = latents + full A₀) simulators, attacker-
  available initialisers (span / cert / spananchor) beside near/random, per-line provenance, and a
  `verdict` separating optimisation failure from alias. Runner `scripts/run_exact_inversion_wexac.sh`
  (step1 · step2_near · step2_init · step3 · step4 · step5_rescue). Analyzer + 3 figures.
- **fwd_check = 7.4e-16 … 8.5e-16**: the span-adapted simulator reproduces the actual release at the
  truth — the normal form confirmed operationally. Everything downstream rests on this.
- **HEADLINE: the (N,k) phase diagram is 49/49 recovered** (median residual 8.6e-31, N,k ∈ 2..14, r=16,
  T=400, start 10% off). The certificate-only diagram is exactly 0 above `k = r − N`; exact inversion
  recovers on BOTH sides. Figure: `figures/exact_inversion/phase_diagram_comparison.png`.
  Caveat: 15 of 49 failed at restarts=1, every one with a NONZERO residual (never an alias), and all 15
  recovered at restarts=4. One seed per cell.
- Basin at a fully trained adapter (deformation 0.96): recovers to a **24% median start error**, first
  clean failure at 36% — past the ~10–15% the finite-difference prototype reported. Cost of distance is
  restarts (1.0 → 3.2 mean), not accuracy.
- An adversarial review found six defects, all fixed and gated. The serious one: `B₀ = 0` makes the
  A-gradient exactly 0 at t=1, so `d/dv √v` is infinite with zero incoming sensitivity → `0 × inf` →
  **the entire Adam Jacobian was NaN (1925/1925)**, which would have reported a fabricated "an Adam
  release is not invertible". Post-fix gate: 0/1925 NaN, fwd_check still exactly 0.0.
- Docs: STATUS.md top section, two LESSONS_LEARNED entries (the NaN; a vacuous diagnostic scoring
  perfectly), CLAUDE.md section, next_experiment_plan.md item, `notes/exact_lora_inversion_framework.md`,
  memory files, `experiments/exact_inversion/{RESULTS.md,NOTES.md}`.

## Next step(s)
1. **Collect the initialiser arms** (jobs 408560 random / 408561 span / 408562 cert / 408563 spananchor,
   k=12 N=8 T=1500, 5 seeds × 8 restarts). This is the only arm whose outcome changes the story, because
   the framework's claim is that what a learned decoder must supply is an *initializer*. Then
   `python experiments/exact_inversion/analyze_exact_inversion.py` and fill the placeholder section in
   RESULTS.md. Related measurement already in hand: the released span estimator sits at 52° mean
   principal angle to the private span at N=8 (59° at N=12) vs ~78–83° for a random subspace.
2. **Adam**: job 408559 (LBFGS, n=96) is descending very slowly (residual ~0.18 after 10 outer iters,
   ~20 h to finish) — consider killing it in favour of job **413794** (`ei2_adam_small`, n=32, k=6, N=4,
   T=200, LM affordable at r·n=512 unknowns, with an SGD control at the same shape). 413794 was
   **preempted back to PEND** and will restart from scratch.
3. Test the staged schedule (X-only first, then joint) that the bundle prototype used — see
   `NOTES.md §2`; it is the untested explanation for the one validation cell that does not reproduce.
4. rsync the three bundle PDFs from the Mac into `papers/`.

## Open threads / gotchas
- Running: 396204/396206/396207 (basin 0.15/0.30/0.50 seeds), 408559–408563, 413794 (PEND).
- **`set +u` is required** before `conda activate` in any job script here, or the job dies in 9 s with a
  near-empty stdout (`ADDR2LINE: unbound variable` in the env's activate.d hook).
- **Never edit the script while a multi-cell job runs** — the runner re-launches python per cell, so
  later cells silently pick up the new recipe. One run was killed and resubmitted for this.
- The session scratchpad is **not visible from compute nodes**; submit inline scripts via `bsub` stdin.
- Under Adam `rank B_T = r`, so `C ≡ 0` and `eps_inv` reads 1.9e-15 — a *perfect-looking* certificate
  that is vacuous. Always read `cert_norm` / `cert_vacuous` beside it.
- Pre-existing uncommitted `figures/recon_showcase/*.png` + `results/recon_showcase_sweep.csv` were in
  the tree before today and were left untouched.

## Pointers
- Write-up: `experiments/exact_inversion/RESULTS.md`; disagreements with the bundle: `NOTES.md`.
- Theory: `notes/exact_lora_inversion_framework.md`.
- Submit: `bsub -q long-gpu -gpu "num=1" -R "rusage[mem=8192] select[ngpus>0]" -J ei_x -o scripts/wexac_logs/ei_x_%J.out -e scripts/wexac_logs/ei_x_%J.err bash scripts/run_exact_inversion_wexac.sh <stage> [arg]`
- Analyze: `python experiments/exact_inversion/analyze_exact_inversion.py` (CPU, reads all `results/exact_inversion/*.jsonl`).
