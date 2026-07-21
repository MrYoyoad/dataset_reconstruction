# Handover — 2026-07-21 20:45

## State
Branch `main`, pushed to `myfork/main` (main now tracks it — it had no upstream before). Gal sent
**SimuDy (Tian et al., ICLR 2025)** saying it "already showed an idea we discussed" — it publishes our
direct-weight-inversion primitive, so that headline novelty is taken. Direction reframed (not
abandoned): re-center on LoRA-adapter-only leakage + identifiability theory. Job 435843 was submitted,
found defective, **killed**, fixed, and **resubmitted as job 440634** (`gal_additions`, long-gpu).

## Done this session
- Read SimuDy end-to-end; wrote `notes/related_work_simudy.md` (teardown) and
  `notes/simudy_decision_brief.md` (1→N chain: what they prove vs miss, feasibility, B1/B2 gates).
  Resolved the Part D novelty search in `notes/experiment_plan.md`.
- Added smooth activations (`gelu`/`silu`/`softplus`) — they did not exist, silently blocking Gal's
  top-priority Addition 2.
- **Found and fixed two defects in the first sweep** (both logged in LESSONS_LEARNED.md):
  1. Output filenames omitted every swept dimension → 43 runs would have collapsed to ~8 files.
     Fixed `base_name` (`run_experiment_b.py`) + added `base_name` to
     `generate_experiment_b_figure()` (`plotting.py`) so figures stop colliding too.
  2. GELU looked catastrophic (SSIM 0.041 vs control 0.020) but `weight_change=0.039`,
     `effective_rank=2` → the ReLU-tuned `lr=0.01` barely moved the net. Confound, not a result.
- Rewrote `scripts/run_gal_additions_sweep.sh`: Stage 0 now **asserts filename uniqueness and aborts**
  if it fails; Stage 1 is an **LR calibration** (activation × LR) before any conclusions.
- Updated STATUS.md + LESSONS_LEARNED.md; commits `e0d10a4`, `8f10fa9` pushed.
- Drafted a reply to Gal (`simudy_decision_brief.md` §13) — **not confirmed sent.**

## Next step(s)
1. **Check job 440634 first thing**: `bjobs -w`; then
   `tail -40 scripts/wexac_logs/gal_additions_440634.out`. Stage 0 must print
   "PASSED: activations run AND filenames are unique". If it aborted, the guard did its job — fix and
   resubmit rather than trusting the output.
2. **Read Stage 1 before anything else**: for each activation, compare SSIM *at comparable
   `weight_change`*, not at fixed LR. That determines whether "smoother = better" holds.
3. Send the reply to Gal if not already sent.
4. Unblock the two missing Gal asks — both need code: **Addition 3** (anchor α-sweep; no
   `--anchor_alpha` flag exists) and **GB-Phase 1** (gradient-bridge decoder; no code at all).
   A ready-to-paste prompt for a parallel session exists in the 2026-07-21 conversation.

## Open threads / gotchas
- Job 440634 runs ~68 configs (~5 min each ≈ 5-6 h). Priority-ordered: Addition 2 first, so partial
  completion still delivers Gal's top ask.
- **Stage 0 guard PASSED at 20:41** — verified `_gelu/_silu/_softplus.pth` are three distinct files
  and figures are uniquely named. Both collision bugs are confirmed fixed in practice, not just in code.
- **Job went `SSUSP` at ~20:45: preempted by higher-priority job 440970.** This is normal LSF
  behaviour on WEXAC, **no progress is lost**, and it resumes automatically when the GPU frees
  (RUNLIMIT is 5760 min, so no timeout risk). If it is *still* `SSUSP` many hours later, requeue with
  `brequeue 440634` or resubmit. **Do NOT submit a second copy in parallel** — both would write the
  same result filenames and recreate the collision bug.
- **Never trust an SSIM without checking `weight_change` and `delta_w_effective_rank`.** Near-zero
  weight change ⇒ the number is meaningless.
- Reply to Gal drafted but unconfirmed. Gmail/Calendar MCP connectors are **not authorized**, so
  Claude cannot check the thread.
- `results/` still holds pre-existing files from older runs under the *old* naming scheme; new files
  carry activation/npc/loss/lr suffixes. Don't mix them when analysing.

## Pointers
- Paper: `papers/Tian_2025_SimuDy_Simulating_Training_Dynamics_ICLR.pdf` (+ `_fulltext.txt` for grep)
- Analysis: `notes/simudy_decision_brief.md`, `notes/related_work_simudy.md`
- To-do source of truth: `notes/experiment_plan.md` (Parts A/B/C; Part D done)
- Job: `bjobs -w` · `tail -40 scripts/wexac_logs/gal_additions_440634.out` · script
  `scripts/run_gal_additions_sweep.sh` · queue `long-gpu` (`short-gpu` had 4208 pending)
- Reading papers here: `pip install pypdf` + extract text (Read tool can't render PDFs; use `grep -a`)
