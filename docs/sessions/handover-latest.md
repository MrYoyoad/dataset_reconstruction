# Handover — 2026-08-28 18:59

## State
Branch `step1-activation-rescore-retrieval`. The active research front is the **dataset-sensitivity
program** (whitened-Mahalanobis sensitivity, per-image leakage predicted by base-gradnorm g₀, distance
dial, margin-at-scale, ViT+LoRA) and its extension the **full-FT-vs-LoRA "valley" comparison**
(`notes/fullft_valley_comparison_plan.md` v1.2, both audits PASS). A multi-session swarm runs it:
**executer = yoado-1f**, **theory auditer = yoado-18**, **reconstruction/crux owner = yoado-ed**, and
**this metric-auditer = yoado-6d**. A deep audit of all plans-vs-asks was just done; a consolidated
science summary was written (`notes/thesis_scientific_summary.md` + 10pp PDF, figures embedded).

## Done this session
- Rank sweep (job 581629) DONE + figured: the multi-class "leaks-fewer" reversal is a LOW-RANK effect,
  gap 23→13→0 across r=8/16/32, vanishes at full-FT. `figures/rank_sweep/*`.
- Leakage story consolidated; **reconstruction overclaim CORRECTED to 0/40** (decoded adapter-only never
  beats the mean-image baseline; the "ssim_norm 0.61" was inflated). Canonical honest figure =
  `figures/combined/leakage_identifiability_plus_reconstruction.png` (+ generator `experiments/plot_leakage_combined.py`).
- Metric-rigor audits (this session, as yoado-6d): the whitened metric, the arm-B "sharpens with N"
  ARTIFACT retraction (3-way cross-fit + K-non-convergence), the full-FT valley plan (B1 dimension-invariance
  + B2 SGD-noise gates), all folded and PASS.
- Deep audit → `notes/thesis_scientific_summary.md` (10pp PDF). Fixed STATUS internal contradiction
  (lines 193/323 still claimed "reconstructable ~0.6"). Fixed CLAUDE.md to point at `next_experiment_plan.md`
  (not the superseded `experiment_plan.md`). `scripts/md_to_pdf.py` now embeds images.

## Next step(s)
1. **Full-FT valley wave** — stage-0 was RE-RUN after a calibration-ordering fix (arm C fatal'd: calib
   sequenced after the dial arms → provisional lr=0.05 → metric-starved; fixed to calib-FIRST). Watch
   **job 375314**; the wave (arms C/D/E/G → F → B1) launches on a green stage-0 under the executer's own
   authority. Headline read ONLY after B1+B2 gates pass.
2. **Close the activation crux (supervisor's TOP ask, STALLED)** — job 857271's 21 configs never fully
   analyzed (partial rescore `results/rescored_activations_857271_2026-08-11.csv` exists); feature-stability-
   vs-T and flowers matched-wc band untested. `next_experiment_plan.md` QW1.
3. **Effect of fine-tuning on classification accuracy** — user ask, only PARTIAL (held-acc asymmetry
   measured; no systematic pre/post-FT accuracy study).
4. **Commit the untracked dataset-sensitivity program** — 48 untracked results + the package `__init__.py`
   (module not in git). Flagged to yoado-1f to commit (its live code).

## Open threads / gotchas
- **Uncommitted active program**: `experiments/dataset_sensitivity/{__init__,margin_at_scale,arm_b_*_diag}.py`
  + `results/arm_*/` untracked. Biggest hygiene gap.
- **SimuDy reframe reply to Gal** drafted but never confirmed sent — gates the direct-inversion axis framing.
- **g₀ predictor**: ρ=+0.857 (n=12, 260171) vs +0.777 (n=24, 272504, INDETERMINATE) — no canonical value;
  USPS OOD counterexample (higher g₀, leaks less, n=2) unresolved.
- STATUS.md is 2699 lines with a dead Sprint-3 to-do list buried (~2385-2472) — prune candidate.
- WEXAC nodes to exclude: lgn28, hgn46, hgn45, lgn13 (flaky/NaN). `python -u` in job scripts; bsub-only.

## Pointers
- Consolidated science: `notes/thesis_scientific_summary.md` (+ .pdf). Plans: `notes/dataset_sensitivity_program_plan.md` (v3),
  `notes/fullft_valley_comparison_plan.md` (v1.2), `notes/whitened_sensitivity_metric.md`, `notes/next_experiment_plan.md` (to-do).
- Leakage: `notes/leakage_story_consolidated.{md,pdf}`; figures `figures/{combined,rank_sweep,crux,margin_at_scale,similarity_ladder,h_spotcheck}/`.
- Metric: `experiments/dataset_sensitivity/whitened_metric.py` (3-way cross-fit); full-FT: `experiments/dataset_sensitivity/fullft_valley.py`.
- Job to watch: `bjobs`; stage-0 log `scripts/wexac_logs/fullft_valley_stage0_375314.out`.
