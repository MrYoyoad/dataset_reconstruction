
---

# Handover — 2026-07-21 20:33


## State
Branch `main`, clean, pushed to `myfork/main` @ `e0d10a4` (main now tracks myfork — it had no
upstream before). Gal sent **SimuDy (Tian et al., ICLR 2025)** saying it "already showed an idea we
discussed" — it publishes our direct-weight-inversion primitive, so that headline novelty is taken.
Direction reframed (not abandoned): re-center on LoRA-adapter-only leakage + identifiability theory.
LSF job **435843** (`gal_additions`, long-gpu, host lgn22) is **RUNNING** with Gal's Additions 1+2,
but **has two defects — see gotchas. Decide whether to kill/fix/resubmit before trusting its output.**

## Done this session
- Read SimuDy end-to-end; wrote `notes/related_work_simudy.md` (teardown) and
  `notes/simudy_decision_brief.md` (1→N chain: what they prove vs miss, feasibility, B1/B2 gates).
- Resolved the Part D novelty search in `notes/experiment_plan.md`.
- Added smooth activations (`gelu`/`silu`/`softplus`) to `ACTIVATION_CHOICES` + `make_activation()` —
  they did not exist, silently blocking Gal's top-priority Addition 2.
- Wrote + submitted `scripts/run_gal_additions_sweep.sh` (job 435843), priority-ordered.
- Updated STATUS.md + LESSONS_LEARNED.md; committed and pushed `e0d10a4`.
- Drafted a reply to Gal (in `simudy_decision_brief.md` §13) — **not confirmed sent.**

## Next step(s)
1. **Decide on job 435843** (see gotcha #1). Recommended: `bkill 435843`, fix the filename builder to
   include activation / n_per_class / loss_type, resubmit. Otherwise 43 runs collapse to ~8 files.
2. **Investigate the GELU result** (gotcha #2) — likely an LR confound, not a real negative.
3. Send the reply to Gal if not already sent.
4. Unblock the two missing Gal asks, both need code: **Addition 3** (anchor α-sweep — no
   `--anchor_alpha` flag exists) and **GB-Phase 1** (gradient-bridge decoder — no code at all).

## Open threads / gotchas
- **[CRITICAL] Job 435843 output collides.** `run_experiment_b.py:652-661` builds the filename as
  `exp_b_T{n_steps}_r{rank}[_free]_s{seed}_a{relu_alpha}` — it does **not** include
  `finetune_activation`, `n_per_class`, or `loss_type`. So all 5 activations at a given T overwrite
  the same `.pth`; Stage 2's n_per_class values overwrite by seed; Stage 3's l2/cosine overwrite each
  other. **43 runs → ~8 surviving files.** Metrics are still recoverable from the stdout log.
- **[RESULT?] GELU looks broken, probably a confound.** First config (`ADD2 act=gelu T=1 r=8`, oracle)
  gave **SSIM 0.0414 vs control 0.0203** — i.e. barely above chance, against a LeakyReLU/ModifiedRelu
  baseline of ~0.797. Diagnostics: `weight_change=0.039` (tiny), `delta_w_effective_rank=2`,
  `ntk_passed=False`, `feature_stability=0.965`. The tiny weight change suggests the **fine-tuning LR
  is tuned for ReLU and barely moves a GELU net** — so this is likely a hyperparameter artifact, not
  evidence against the "smoother = better" prediction. Do an LR sweep per activation before reporting.
- **[BUG, logged] Any run overwrites canonical figures.** `generate_experiment_b_figure()` defaults
  `save_dir` to `figures/sprint1/`, and `--save_results` gates only the `.pth`, not the figure. A
  3-epoch smoke test overwrote `figures/sprint1/experiment_b_grid_oracle.png`; it was caught via
  mtime and reverted before committing. Job 435843 is rewriting that same file every config.
- Reply to Gal drafted but unconfirmed. Gmail/Calendar MCP connectors are **not authorized**, so
  Claude cannot check the thread.

## Pointers
- Paper: `papers/Tian_2025_SimuDy_Simulating_Training_Dynamics_ICLR.pdf` (+ `_fulltext.txt` for grep)
- Analysis: `notes/simudy_decision_brief.md`, `notes/related_work_simudy.md`
- To-do source of truth: `notes/experiment_plan.md` (Parts A/B/C; Part D done)
- Job: `bjobs -w` · `tail -50 scripts/wexac_logs/gal_additions_435843.out` · script
  `scripts/run_gal_additions_sweep.sh` · queue `long-gpu` (use it; `short-gpu` had 4208 pending)
- Filename builder to fix: `experiments/run_experiment_b.py:652-661`
- Figure-overwrite source: `experiments/plotting.py:363-372`
- Reading papers here: `pip install pypdf` then extract text (Read tool can't render PDFs; use `grep -a`)
