
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

---

# Handover — 2026-07-21 20:45 (supersedes 20:33 — job fixed & resubmitted)


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

---

# Handover — 2026-08-23 17:20

## State
Branch `step1-activation-rescore-retrieval`, pushed to `myfork` (commit `3f2226c`). Phase J0 of the
Jacobian-spectrum leakage program is **built and submitted**; the deciding FD gate is pending on the
cluster.

## Done this session
- **Audited** the parallel-session plan against live infra; caught two real bugs before submit:
  (1) `direct_inversion.A_rank_shape` hardcodes MNIST dims (784/1000) → would break the toy net; added
  local `_a_shape` reading `in_features` from `frozen`. (2) `generate_target`'s `θ_T` is the ALL-layer
  update, not the single-module target → use it only for `frozen/b0/B0[0]/ds_mean`, define
  `Y0:=forward_Y(0)`.
- **Built Phase J0**: `experiments/jacobian_spectrum.py`. `J = ∂vec(A_T,B_T)/∂a` via
  forward-over-reverse JVP (`exact_jacobian`, `jvp_double`, double `autograd.grad`, composes with the
  create_graph unroll; `retain_graph` freed after the last column). Single LoRA module, GELU, float64,
  ds_mean frozen at a=0. `build_tangents` (qr + svd), `recover_a` (LSQ). J1 scaffolded
  (`estimate_sigma_seed`, `snr_spectrum` via Woodbury, `q_eff`). Did NOT edit `direct_inversion.py`.
- **Rewrote the spec doc** to PhD-readable v3: `notes/jacobian_leakage_experiment_plan.md` (Part 0
  background primer → concrete J0/J1 → J2–J6 goals → build spec → verification).
- **Submitted job 966830** (`short-gpu`). Updated STATUS.md. Saved a memory: never run anything
  locally — always submit a WEXAC job (user rule, emphatic).

## Next step(s)
1. **Read job 966830 first**: `tail -60 scripts/wexac_logs/jacobian_spectrum_966830.out`. Stage 0 is the
   **toy-AD FD gate** — must print `PASSED` with FD rel err `<1e-6` AND `jvp_double` vs `reverse_loop`
   `<1e-8`. If it aborts (`FATAL`), the third-order autograd path is wrong — fix `exact_jacobian`
   before trusting any number.
2. **If gate passes** → Stage 1 real MNIST single-module smoke (`dimY=3568`, `Nk=8`, FD `<1e-4`) →
   Stage 2 J0 coordinate-recovery-vs-ε sweeps (`qr` + `svd` tangents, N∈{2,4}, k∈{4,8,16}). Inspect the
   spectrum figures in `figures/jacobian_spectrum/` and the `svd`-tangent run (σ_i(J) should track the
   injected geometric decay — the deterministic "which coords survive" claim's teeth).
3. **Then J1**: seeds + whitening. `snr_spectrum`/`q_eff` are written but UNVALIDATED — add a whitening
   sanity check (whitened seed samples ≈ isotropic) and report `q_eff` over a range of shrinkage ρ and ε.

## Open threads / gotchas
- **Spec = `notes/jacobian_leakage_experiment_plan.md` (v3).** Self-contained; a fresh session can
  extend from it.
- Deterministic J0 is only non-trivial where J is genuinely rank-deficient OR ε leaves the linear
  regime (the plan's honesty caveat) — that's why the `svd`-tangent + ε-sweep exist.
- With A₀=0, B is stationary at step 1 (∂loss/∂B ∝ A=0) → B-rows of J carry signal only for T≥2 (T=5 ok).
- **User rule: never run anything locally, always submit a bsub job** — gate/smoke live as Stage 0 of
  the script. We run on the WEXAC filesystem directly, so no rsync needed.
- Sibling session yoado-29 co-designed this in plan mode (read-only); it did not touch repo files.

## Pointers
- Code: `experiments/jacobian_spectrum.py` · Job: `scripts/run_jacobian_spectrum_wexac.sh` ·
  log `scripts/wexac_logs/jacobian_spectrum_966830.out` · `bjobs -w`
- Spec: `notes/jacobian_leakage_experiment_plan.md` (v3) · rationale
  `notes/identifiability_feasibility_revision.tex`, `notes/minegrad_analysis.md`

---

# Handover — 2026-08-23 19:30

## State
Branch `step1-activation-rescore-retrieval`, pushed to `myfork`. Phase J0 of the Jacobian-spectrum
leakage program is **built, validated, and complete** (job 982855). AD is exact (toy FD 5.9e-10,
jvp-vs-reverse 3.5e-18; MNIST FD 3.9e-9). **Next task = J1 (seed-whitening / q_eff), the first
privacy-meaningful number.** See STATUS.md "Phase J0 COMPLETE" for the full result table + the
critical honesty caveat (pre-whitening eff_rank is NOT leakage evidence — it conflates magnitude with
recoverability and is confounded by LoRA-rank + T-underfitting).

## Done this session
- **Audited** the parallel-session plan against live infra; caught two real bugs before submit:
  (1) `direct_inversion.A_rank_shape` hardcodes MNIST dims (784/1000) → would break the toy net; added
  local `_a_shape` reading `in_features` from `frozen`. (2) `generate_target`'s `θ_T` is the ALL-layer
  update, not the single-module target → use it only for `frozen/b0/B0[0]/ds_mean`, define
  `Y0:=forward_Y(0)`.
- **Built Phase J0**: `experiments/jacobian_spectrum.py`. `J = ∂vec(A_T,B_T)/∂a` via
  forward-over-reverse JVP (`exact_jacobian`, `jvp_double`, double `autograd.grad`, composes with the
  create_graph unroll; `retain_graph` freed after the last column). Single LoRA module, GELU, float64,
  ds_mean frozen at a=0. `build_tangents` (qr + svd), `recover_a` (LSQ). J1 scaffolded
  (`estimate_sigma_seed`, `snr_spectrum` via Woodbury, `q_eff`). Did NOT edit `direct_inversion.py`.
- **Rewrote the spec doc** to PhD-readable v3: `notes/jacobian_leakage_experiment_plan.md` (Part 0
  background primer → concrete J0/J1 → J2–J6 goals → build spec → verification).
- **Submitted job 966830** (`short-gpu`). Updated STATUS.md. Saved a memory: never run anything
  locally — always submit a WEXAC job (user rule, emphatic).

## Next step(s)
1. **Build/validate J1** (the first privacy-meaningful number). `snr_spectrum` (Woodbury) / `q_eff` /
   `estimate_sigma_seed` are scaffolded in `jacobian_spectrum.py` but UNVALIDATED. Need: a
   `ctx_factory(seed)` that redraws the LoRA B0 init (the ordinary-training randomness source; full
   batch → B0 draw is the main stochasticity) holding data/frozen fixed; estimate `Σ_seed` over
   S∈{16,32,64} seeds; whiten; test (a) whitening sanity (whitened seed samples ≈ isotropic), (b) `q_eff`
   over a range of shrinkage ρ and ε, (c) CRLB per-coordinate scatter. Run as a bsub job.
2. **T-sweep (5/20/50)** on the J0 configs to de-confound the eff_rank readout: eff_rank→Nk with T ⟹
   underfitting; plateau <Nk ⟹ structural. (yoado-29's caution — do this before writing up the
   N-dependence.)
3. **Report σ-spectrum SHAPE** (gap vs gradual decay), not just scalar eff_rank; consolidated
   eff_rank-vs-Nk figure (submit as a small plotting job — do not run locally).
4. Only after J1: revisit the N=2(frac~1.0) vs N=4(frac~0.6) contrast as a possible identifiability
   statement.

## Open threads / gotchas
- **Spec = `notes/jacobian_leakage_experiment_plan.md` (v3).** Self-contained; a fresh session can
  extend from it.
- Deterministic J0 is only non-trivial where J is genuinely rank-deficient OR ε leaves the linear
  regime (the plan's honesty caveat) — that's why the `svd`-tangent + ε-sweep exist.
- With A₀=0, B is stationary at step 1 (∂loss/∂B ∝ A=0) → B-rows of J carry signal only for T≥2 (T=5 ok).
- **User rule: never run anything locally, always submit a bsub job** — gate/smoke live as Stage 0 of
  the script. We run on the WEXAC filesystem directly, so no rsync needed.
- Sibling session yoado-29 co-designed this in plan mode (read-only); it did not touch repo files.

## Pointers
- Code: `experiments/jacobian_spectrum.py` · Job: `scripts/run_jacobian_spectrum_wexac.sh` ·
  log `scripts/wexac_logs/jacobian_spectrum_966830.out` · `bjobs -w`
- Spec: `notes/jacobian_leakage_experiment_plan.md` (v3) · rationale
  `notes/identifiability_feasibility_revision.tex`, `notes/minegrad_analysis.md`

---

# Handover — 2026-08-23 19:40

## State
Branch `step1-activation-rescore-retrieval`, pushed to `myfork`. Jacobian-spectrum program: **J0 AND J1
both built, validated, complete** (jobs 982855, 983139). AD exact (toy FD 5.9e-10, jvp-vs-reverse
3.5e-18; MNIST FD 3.9e-9). **Two decisive de-confounds landed — see STATUS.md "Phase J1 COMPLETE":**
(1) the J0 "N=4 collapse" is largely T=5 UNDERFITTING (eff_rank climbs 9.3→12.7 over T=5→50);
(2) whitening is INOPERATIVE here — B0-init noise is ~orthogonal to J (J-energy in measured noise
subspace = 0.0–0.1%), so q_eff is shrinkage-floor artifact, NOT a valid privacy number yet.

**Next task = add a randomness source that lives in J's column space (minibatch SGD / data-order /
augmentation) so Σ_seed spans J and q_eff becomes measurable.** Then re-run the leakage bracket with T
large enough to converge and S≥4·Nk.

## Done this session
- **Audited** the parallel-session plan against live infra; caught two real bugs before submit:
  (1) `direct_inversion.A_rank_shape` hardcodes MNIST dims (784/1000) → would break the toy net; added
  local `_a_shape` reading `in_features` from `frozen`. (2) `generate_target`'s `θ_T` is the ALL-layer
  update, not the single-module target → use it only for `frozen/b0/B0[0]/ds_mean`, define
  `Y0:=forward_Y(0)`.
- **Built Phase J0**: `experiments/jacobian_spectrum.py`. `J = ∂vec(A_T,B_T)/∂a` via
  forward-over-reverse JVP (`exact_jacobian`, `jvp_double`, double `autograd.grad`, composes with the
  create_graph unroll; `retain_graph` freed after the last column). Single LoRA module, GELU, float64,
  ds_mean frozen at a=0. `build_tangents` (qr + svd), `recover_a` (LSQ). J1 scaffolded
  (`estimate_sigma_seed`, `snr_spectrum` via Woodbury, `q_eff`). Did NOT edit `direct_inversion.py`.
- **Rewrote the spec doc** to PhD-readable v3: `notes/jacobian_leakage_experiment_plan.md` (Part 0
  background primer → concrete J0/J1 → J2–J6 goals → build spec → verification).
- **Submitted job 966830** (`short-gpu`). Updated STATUS.md. Saved a memory: never run anything
  locally — always submit a WEXAC job (user rule, emphatic).

## Next step(s)
1. **Add SGD/minibatch (or data-order/augmentation) noise to the training map** so `Σ_seed` spans J's
   column space. This is THE blocker: with full-batch training the only randomness (B0 init) is
   ~orthogonal to J (0% energy overlap), so q_eff is unmeasurable. Implement a `ctx_factory(seed)` that
   varies minibatch order (needs switching `unrolled_lora_AB` to minibatch SGD) and re-run
   `run_j1`; the energy diagnostic (`noise_subspace_energy`, already in code) should now report a
   meaningful non-zero fraction. Only then is q_eff a real number.
2. **Train to convergence** (T-sweep shows T=5 underfits N=4) and **scale S≥4·Nk** for headline configs
   (adequacy print already flags this).
3. Consolidated figures (eff_rank-vs-T curve; q_eff/q-vs-ε across configs) — submit as a plotting job,
   do NOT run locally.
4. Then J2 (the (N,r,L) phase diagram) once q_eff is trustworthy.

## Infra ready to reuse
`experiments/jacobian_spectrum.py`: `run_j0`, `run_j0_T_sweep`, `run_j1` (+ `--j0/--j1/--T_sweep`
CLI), `exact_jacobian` (jvp_double), `snr_spectrum` (Woodbury), `q_eff`, `noise_subspace_energy`,
`_draw_B0`. Scripts: `run_jacobian_spectrum_wexac.sh` (J0), `run_jacobian_j1_wexac.sh` (T-sweep+J1).
Both put the toy-AD gate as Stage 0 (abort-on-fail). RULE: never run locally — always bsub.

## Open threads / gotchas
- **Spec = `notes/jacobian_leakage_experiment_plan.md` (v3).** Self-contained; a fresh session can
  extend from it.
- Deterministic J0 is only non-trivial where J is genuinely rank-deficient OR ε leaves the linear
  regime (the plan's honesty caveat) — that's why the `svd`-tangent + ε-sweep exist.
- With A₀=0, B is stationary at step 1 (∂loss/∂B ∝ A=0) → B-rows of J carry signal only for T≥2 (T=5 ok).
- **User rule: never run anything locally, always submit a bsub job** — gate/smoke live as Stage 0 of
  the script. We run on the WEXAC filesystem directly, so no rsync needed.
- Sibling session yoado-29 co-designed this in plan mode (read-only); it did not touch repo files.

## Pointers
- Code: `experiments/jacobian_spectrum.py` · Job: `scripts/run_jacobian_spectrum_wexac.sh` ·
  log `scripts/wexac_logs/jacobian_spectrum_966830.out` · `bjobs -w`
- Spec: `notes/jacobian_leakage_experiment_plan.md` (v3) · rationale
  `notes/identifiability_feasibility_revision.tex`, `notes/minegrad_analysis.md`

---

# Handover — 2026-08-23 19:40

## State
Branch `step1-activation-rescore-retrieval`, pushed to `myfork`. Jacobian-spectrum program: **J0 AND J1
both built, validated, complete** (jobs 982855, 983139). AD exact (toy FD 5.9e-10, jvp-vs-reverse
3.5e-18; MNIST FD 3.9e-9). **Findings — see STATUS.md "Phase J1 COMPLETE" (three, in order of correction):**
(1) the J0 "N=4 collapse" is largely T=5 UNDERFITTING (eff_rank climbs 9.3→12.7 over T=5→50);
(2) the "0.1% J-energy overlap = init noise orthogonal to J" reading was a DIMENSIONALITY ARTIFACT —
`eff_rank(Σ_seed)`≈S−1 at S=16/32/64/128 (job 983585) proves the B0-init noise is high-dim (~full-rank
over the ~8000-dim B-block) and undersampled, NOT low-dim/orthogonal. "Random init is not a defense" is
RETRACTED;
(3) honest fallback under an isotropic init-noise model: `q_eff|iso`=0 for ε≤1 ⇒ init noise plausibly
DOES mask at realistic ε (opposite of the retracted claim). True Σ_seed unmeasurable at S≤128.

**Next task = add a randomness source that lives in J's column space (minibatch SGD / data-order /
augmentation) so Σ_seed spans J and q_eff becomes measurable at feasible S.** Then re-run the leakage
bracket with T large enough to converge. (B0-init whitening can't work: Σ_seed needs S≫8000.)

## Done this session
- **Audited** the parallel-session plan against live infra; caught two real bugs before submit:
  (1) `direct_inversion.A_rank_shape` hardcodes MNIST dims (784/1000) → would break the toy net; added
  local `_a_shape` reading `in_features` from `frozen`. (2) `generate_target`'s `θ_T` is the ALL-layer
  update, not the single-module target → use it only for `frozen/b0/B0[0]/ds_mean`, define
  `Y0:=forward_Y(0)`.
- **Built Phase J0**: `experiments/jacobian_spectrum.py`. `J = ∂vec(A_T,B_T)/∂a` via
  forward-over-reverse JVP (`exact_jacobian`, `jvp_double`, double `autograd.grad`, composes with the
  create_graph unroll; `retain_graph` freed after the last column). Single LoRA module, GELU, float64,
  ds_mean frozen at a=0. `build_tangents` (qr + svd), `recover_a` (LSQ). J1 scaffolded
  (`estimate_sigma_seed`, `snr_spectrum` via Woodbury, `q_eff`). Did NOT edit `direct_inversion.py`.
- **Rewrote the spec doc** to PhD-readable v3: `notes/jacobian_leakage_experiment_plan.md` (Part 0
  background primer → concrete J0/J1 → J2–J6 goals → build spec → verification).
- **Submitted job 966830** (`short-gpu`). Updated STATUS.md. Saved a memory: never run anything
  locally — always submit a WEXAC job (user rule, emphatic).

## Next step(s)
1. **Add SGD/minibatch (or data-order/augmentation) noise to the training map** so `Σ_seed` spans J's
   column space. This is THE blocker: with full-batch training the only randomness (B0 init) is
   ~orthogonal to J (0% energy overlap), so q_eff is unmeasurable. Implement a `ctx_factory(seed)` that
   varies minibatch order (needs switching `unrolled_lora_AB` to minibatch SGD) and re-run `run_j1`.
   **Guardrails (yoado-29):** (a) RE-RUN `noise_subspace_energy` on the SGD Σ_seed — minibatch noise
   enters through the gradients (same channel as data perturbations) so it SHOULD overlap J, but verify,
   don't assume (that's the whole lesson). (b) With stochastic training the correct objects are
   `J := ∂E_ξ[Y|a]/∂a` (Jacobian of the MEAN adapter) and `Σ_seed := Cov_ξ[Y|a]` (plan §12) — computing
   J at a single ξ and whitening by Cov(ξ) is valid only if J is stable across ξ, so check that or
   average J over a few ξ. (c) Batch size is now a first-class knob (sets noise magnitude / masking) —
   pick a realistic fine-tuning batch size and report q_eff's dependence on it.
2. **Train to convergence** (T-sweep shows T=5 underfits N=4) and **scale S≥4·Nk** for headline configs
   (adequacy print already flags this). Also: the energy-vs-T check (job [energy-vs-T]) tests whether
   the J⊥Σ_seed orthogonality is robust or a small-T effect (predicted to grow with T).
3. Consolidated figures (eff_rank-vs-T curve; q_eff/q-vs-ε across configs) — submit as a plotting job,
   do NOT run locally.
4. Then J2 (the (N,r,L) phase diagram) once q_eff is trustworthy.

## Infra ready to reuse
`experiments/jacobian_spectrum.py`: `run_j0`, `run_j0_T_sweep`, `run_j1` (+ `--j0/--j1/--T_sweep`
CLI), `exact_jacobian` (jvp_double), `snr_spectrum` (Woodbury), `q_eff`, `noise_subspace_energy`,
`_draw_B0`. Scripts: `run_jacobian_spectrum_wexac.sh` (J0), `run_jacobian_j1_wexac.sh` (T-sweep+J1).
Both put the toy-AD gate as Stage 0 (abort-on-fail). RULE: never run locally — always bsub.

## Open threads / gotchas
- **Spec = `notes/jacobian_leakage_experiment_plan.md` (v3).** Self-contained; a fresh session can
  extend from it.
- Deterministic J0 is only non-trivial where J is genuinely rank-deficient OR ε leaves the linear
  regime (the plan's honesty caveat) — that's why the `svd`-tangent + ε-sweep exist.
- With A₀=0, B is stationary at step 1 (∂loss/∂B ∝ A=0) → B-rows of J carry signal only for T≥2 (T=5 ok).
- **User rule: never run anything locally, always submit a bsub job** — gate/smoke live as Stage 0 of
  the script. We run on the WEXAC filesystem directly, so no rsync needed.
- Sibling session yoado-29 co-designed this in plan mode (read-only); it did not touch repo files.

## Pointers
- Code: `experiments/jacobian_spectrum.py` · Job: `scripts/run_jacobian_spectrum_wexac.sh` ·
  log `scripts/wexac_logs/jacobian_spectrum_966830.out` · `bjobs -w`
- Spec: `notes/jacobian_leakage_experiment_plan.md` (v3) · rationale
  `notes/identifiability_feasibility_revision.tex`, `notes/minegrad_analysis.md`

---

# Handover — 2026-08-23 19:40

## State
Branch `step1-activation-rescore-retrieval`, pushed to `myfork`. Jacobian-spectrum program: **J0 AND J1
both built, validated, complete** (jobs 982855, 983139). AD exact (toy FD 5.9e-10, jvp-vs-reverse
3.5e-18; MNIST FD 3.9e-9). **Findings — see STATUS.md "Phase J1 COMPLETE" (three, in order of correction):**
(1) the J0 "N=4 collapse" is largely T=5 UNDERFITTING (eff_rank climbs 9.3→12.7 over T=5→50);
(2) the "0.1% J-energy overlap = init noise orthogonal to J" reading was a DIMENSIONALITY ARTIFACT —
`eff_rank(Σ_seed)`≈S−1 at S=16/32/64/128 (job 983585) proves the B0-init noise is high-dim (~full-rank
over the ~8000-dim B-block) and undersampled, NOT low-dim/orthogonal. "Random init is not a defense" is
RETRACTED;
(3) SOUND resolution (col(J)-restricted whitening, job 983882): `Σ_J=Cov(Qᵀ(Y−Ȳ))` IS estimable at
S≥r_J. Measured `iso_ratio=tr(Σ_J)/(μ·r_J)` ≈ 0.10 (N=2 k8) / 0.01 (N=4 k8), stable across S≥64 ⇒
init noise is 1–10% of isotropic in the signal directions ⇒ WEAK masking ⇒ `q_eff|col(J)` HIGH
(N=2 k8 11–16/16 at ε≥0.1; N=4 k8 18–22/32 at ε=0.1, 30/32 by ε=3). Net: measured soundly, random
B0-init is a WEAK defense — private coords largely recoverable. (This corrects both #2-artifact and the
over-pessimistic isotropic fallback.) Method = the one to reuse for SGD noise.

**Next task = SGD/minibatch noise phase, reusing `q_eff_colspace` verbatim (just swap the noise
source).** SGD noise enters through the gradients so it SHOULD carry real variance in col(J) — but
VERIFY with `iso_ratio` (that's the whole lesson). Use `J:=∂E_ξ[Y]/∂a` + `Σ_seed:=Cov_ξ[Y]` (plan §12)
with a J-stability check; batch size as a first-class knob; T large enough to converge (T=5 underfits).

## Done this session
- **Audited** the parallel-session plan against live infra; caught two real bugs before submit:
  (1) `direct_inversion.A_rank_shape` hardcodes MNIST dims (784/1000) → would break the toy net; added
  local `_a_shape` reading `in_features` from `frozen`. (2) `generate_target`'s `θ_T` is the ALL-layer
  update, not the single-module target → use it only for `frozen/b0/B0[0]/ds_mean`, define
  `Y0:=forward_Y(0)`.
- **Built Phase J0**: `experiments/jacobian_spectrum.py`. `J = ∂vec(A_T,B_T)/∂a` via
  forward-over-reverse JVP (`exact_jacobian`, `jvp_double`, double `autograd.grad`, composes with the
  create_graph unroll; `retain_graph` freed after the last column). Single LoRA module, GELU, float64,
  ds_mean frozen at a=0. `build_tangents` (qr + svd), `recover_a` (LSQ). J1 scaffolded
  (`estimate_sigma_seed`, `snr_spectrum` via Woodbury, `q_eff`). Did NOT edit `direct_inversion.py`.
- **Rewrote the spec doc** to PhD-readable v3: `notes/jacobian_leakage_experiment_plan.md` (Part 0
  background primer → concrete J0/J1 → J2–J6 goals → build spec → verification).
- **Submitted job 966830** (`short-gpu`). Updated STATUS.md. Saved a memory: never run anything
  locally — always submit a WEXAC job (user rule, emphatic).

## Next step(s)
1. **Add SGD/minibatch (or data-order/augmentation) noise to the training map** so `Σ_seed` spans J's
   column space. This is THE blocker: with full-batch training the only randomness (B0 init) is
   ~orthogonal to J (0% energy overlap), so q_eff is unmeasurable. Implement a `ctx_factory(seed)` that
   varies minibatch order (needs switching `unrolled_lora_AB` to minibatch SGD) and re-run `run_j1`.
   **Guardrails (yoado-29):** (a) RE-RUN `noise_subspace_energy` on the SGD Σ_seed — minibatch noise
   enters through the gradients (same channel as data perturbations) so it SHOULD overlap J, but verify,
   don't assume (that's the whole lesson). (b) With stochastic training the correct objects are
   `J := ∂E_ξ[Y|a]/∂a` (Jacobian of the MEAN adapter) and `Σ_seed := Cov_ξ[Y|a]` (plan §12) — computing
   J at a single ξ and whitening by Cov(ξ) is valid only if J is stable across ξ, so check that or
   average J over a few ξ. (c) Batch size is now a first-class knob (sets noise magnitude / masking) —
   pick a realistic fine-tuning batch size and report q_eff's dependence on it.
2. **Train to convergence** (T-sweep shows T=5 underfits N=4) and **scale S≥4·Nk** for headline configs
   (adequacy print already flags this). Also: the energy-vs-T check (job [energy-vs-T]) tests whether
   the J⊥Σ_seed orthogonality is robust or a small-T effect (predicted to grow with T).
3. Consolidated figures (eff_rank-vs-T curve; q_eff/q-vs-ε across configs) — submit as a plotting job,
   do NOT run locally.
4. Then J2 (the (N,r,L) phase diagram) once q_eff is trustworthy.

## Infra ready to reuse
`experiments/jacobian_spectrum.py`: `run_j0`, `run_j0_T_sweep`, `run_j1` (+ `--j0/--j1/--T_sweep`
CLI), `exact_jacobian` (jvp_double), `snr_spectrum` (Woodbury), `q_eff`, `noise_subspace_energy`,
`_draw_B0`. Scripts: `run_jacobian_spectrum_wexac.sh` (J0), `run_jacobian_j1_wexac.sh` (T-sweep+J1).
Both put the toy-AD gate as Stage 0 (abort-on-fail). RULE: never run locally — always bsub.

## Open threads / gotchas
- **Spec = `notes/jacobian_leakage_experiment_plan.md` (v3).** Self-contained; a fresh session can
  extend from it.
- Deterministic J0 is only non-trivial where J is genuinely rank-deficient OR ε leaves the linear
  regime (the plan's honesty caveat) — that's why the `svd`-tangent + ε-sweep exist.
- With A₀=0, B is stationary at step 1 (∂loss/∂B ∝ A=0) → B-rows of J carry signal only for T≥2 (T=5 ok).
- **User rule: never run anything locally, always submit a bsub job** — gate/smoke live as Stage 0 of
  the script. We run on the WEXAC filesystem directly, so no rsync needed.
- Sibling session yoado-29 co-designed this in plan mode (read-only); it did not touch repo files.

## Pointers
- Code: `experiments/jacobian_spectrum.py` · Job: `scripts/run_jacobian_spectrum_wexac.sh` ·
  log `scripts/wexac_logs/jacobian_spectrum_966830.out` · `bjobs -w`
- Spec: `notes/jacobian_leakage_experiment_plan.md` (v3) · rationale
  `notes/identifiability_feasibility_revision.tex`, `notes/minegrad_analysis.md`

---

# Handover — 2026-08-23 19:40

## State
Branch `step1-activation-rescore-retrieval`, pushed to `myfork`. Jacobian-spectrum program: **J0 AND J1
both built, validated, complete** (jobs 982855, 983139). AD exact (toy FD 5.9e-10, jvp-vs-reverse
3.5e-18; MNIST FD 3.9e-9). **Findings — see STATUS.md "Phase J1 COMPLETE" (three, in order of correction):**
(1) the J0 "N=4 collapse" is largely T=5 UNDERFITTING (eff_rank climbs 9.3→12.7 over T=5→50);
(2) the "0.1% J-energy overlap = init noise orthogonal to J" reading was a DIMENSIONALITY ARTIFACT —
`eff_rank(Σ_seed)`≈S−1 at S=16/32/64/128 (job 983585) proves the B0-init noise is high-dim (~full-rank
over the ~8000-dim B-block) and undersampled, NOT low-dim/orthogonal. "Random init is not a defense" is
RETRACTED;
(3) SOUND resolution (col(J)-restricted whitening, job 983882): `Σ_J=Cov(Qᵀ(Y−Ȳ))` IS estimable at
S≥r_J. Measured `iso_ratio=tr(Σ_J)/(μ·r_J)` ≈ 0.10 (N=2 k8) / 0.01 (N=4 k8), stable across S≥64 ⇒
init noise is 1–10% of isotropic in the signal directions ⇒ WEAK masking ⇒ `q_eff|col(J)` HIGH
(N=2 k8 11–16/16 at ε≥0.1; N=4 k8 18–22/32 at ε=0.1, 30/32 by ε=3). Net: measured soundly, random
B0-init is a WEAK defense — private coords largely recoverable. (This corrects both #2-artifact and the
over-pessimistic isotropic fallback.) Method = the one to reuse for SGD noise.

Note: `q_eff|col(J)` is a conservative LOWER BOUND on true q_eff (Fisher from observing only col(J) ≤
full Fisher; Schur complement) — state leakage as "at least X directions", not "=X".

**Next task = SGD/minibatch noise phase, reusing `q_eff_colspace` verbatim (just swap the noise
source).** SGD noise enters through the gradients so it SHOULD carry real variance in col(J) — but
VERIFY with `iso_ratio` (that's the whole lesson). Use `J:=∂E_ξ[Y]/∂a` + `Σ_seed:=Cov_ξ[Y]` (plan §12)
with a J-stability check; batch size as a first-class knob; T large enough to converge (T=5 underfits).
**The privacy statement is the BRACKET, and the headline number is the MODE COUNT** `#(Σ_J/μ > thr)`
(already printed by `q_eff_colspace`), not the scalar iso_ratio: init couples into col(J) via ~1 mode;
SGD (gradient-channel) should light up MANY (potentially ~r_J). init≈1 vs SGD≈(count) is the privacy
result — instrument for the mode count. init→SGD gap = how much ordinary training randomness protects
the data.
- **Also for J2 (N-sweep):** check whether the masked-mode count stays ~1 while r_J grows with N. If so,
  init-as-defense → 0 fraction is proven (masked fraction 6% at N=2 → ~0% at N=4 already).

## Done this session
- **Audited** the parallel-session plan against live infra; caught two real bugs before submit:
  (1) `direct_inversion.A_rank_shape` hardcodes MNIST dims (784/1000) → would break the toy net; added
  local `_a_shape` reading `in_features` from `frozen`. (2) `generate_target`'s `θ_T` is the ALL-layer
  update, not the single-module target → use it only for `frozen/b0/B0[0]/ds_mean`, define
  `Y0:=forward_Y(0)`.
- **Built Phase J0**: `experiments/jacobian_spectrum.py`. `J = ∂vec(A_T,B_T)/∂a` via
  forward-over-reverse JVP (`exact_jacobian`, `jvp_double`, double `autograd.grad`, composes with the
  create_graph unroll; `retain_graph` freed after the last column). Single LoRA module, GELU, float64,
  ds_mean frozen at a=0. `build_tangents` (qr + svd), `recover_a` (LSQ). J1 scaffolded
  (`estimate_sigma_seed`, `snr_spectrum` via Woodbury, `q_eff`). Did NOT edit `direct_inversion.py`.
- **Rewrote the spec doc** to PhD-readable v3: `notes/jacobian_leakage_experiment_plan.md` (Part 0
  background primer → concrete J0/J1 → J2–J6 goals → build spec → verification).
- **Submitted job 966830** (`short-gpu`). Updated STATUS.md. Saved a memory: never run anything
  locally — always submit a WEXAC job (user rule, emphatic).

## Next step(s)
1. **Add SGD/minibatch (or data-order/augmentation) noise to the training map** so `Σ_seed` spans J's
   column space. This is THE blocker: with full-batch training the only randomness (B0 init) is
   ~orthogonal to J (0% energy overlap), so q_eff is unmeasurable. Implement a `ctx_factory(seed)` that
   varies minibatch order (needs switching `unrolled_lora_AB` to minibatch SGD) and re-run `run_j1`.
   **Guardrails (yoado-29):** (a) RE-RUN `noise_subspace_energy` on the SGD Σ_seed — minibatch noise
   enters through the gradients (same channel as data perturbations) so it SHOULD overlap J, but verify,
   don't assume (that's the whole lesson). (b) With stochastic training the correct objects are
   `J := ∂E_ξ[Y|a]/∂a` (Jacobian of the MEAN adapter) and `Σ_seed := Cov_ξ[Y|a]` (plan §12) — computing
   J at a single ξ and whitening by Cov(ξ) is valid only if J is stable across ξ, so check that or
   average J over a few ξ. (c) Batch size is now a first-class knob (sets noise magnitude / masking) —
   pick a realistic fine-tuning batch size and report q_eff's dependence on it.
2. **Train to convergence** (T-sweep shows T=5 underfits N=4) and **scale S≥4·Nk** for headline configs
   (adequacy print already flags this). Also: the energy-vs-T check (job [energy-vs-T]) tests whether
   the J⊥Σ_seed orthogonality is robust or a small-T effect (predicted to grow with T).
3. Consolidated figures (eff_rank-vs-T curve; q_eff/q-vs-ε across configs) — submit as a plotting job,
   do NOT run locally.
4. Then J2 (the (N,r,L) phase diagram) once q_eff is trustworthy.

## Infra ready to reuse
`experiments/jacobian_spectrum.py`: `run_j0`, `run_j0_T_sweep`, `run_j1` (+ `--j0/--j1/--T_sweep`
CLI), `exact_jacobian` (jvp_double), `snr_spectrum` (Woodbury), `q_eff`, `noise_subspace_energy`,
`_draw_B0`. Scripts: `run_jacobian_spectrum_wexac.sh` (J0), `run_jacobian_j1_wexac.sh` (T-sweep+J1).
Both put the toy-AD gate as Stage 0 (abort-on-fail). RULE: never run locally — always bsub.

## Open threads / gotchas
- **Spec = `notes/jacobian_leakage_experiment_plan.md` (v3).** Self-contained; a fresh session can
  extend from it.
- Deterministic J0 is only non-trivial where J is genuinely rank-deficient OR ε leaves the linear
  regime (the plan's honesty caveat) — that's why the `svd`-tangent + ε-sweep exist.
- With A₀=0, B is stationary at step 1 (∂loss/∂B ∝ A=0) → B-rows of J carry signal only for T≥2 (T=5 ok).
- **User rule: never run anything locally, always submit a bsub job** — gate/smoke live as Stage 0 of
  the script. We run on the WEXAC filesystem directly, so no rsync needed.
- Sibling session yoado-29 co-designed this in plan mode (read-only); it did not touch repo files.

## Pointers
- Code: `experiments/jacobian_spectrum.py` · Job: `scripts/run_jacobian_spectrum_wexac.sh` ·
  log `scripts/wexac_logs/jacobian_spectrum_966830.out` · `bjobs -w`
- Spec: `notes/jacobian_leakage_experiment_plan.md` (v3) · rationale
  `notes/identifiability_feasibility_revision.tex`, `notes/minegrad_analysis.md`

---

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

---
# Handover — 2026-08-30 16:12

## State
Branch `step1-activation-rescore-retrieval`. The supervisor deck for the **2026-08-31 meeting is BUILT and committed**
(commit 9d13e6b): `notes/supervisor_meeting_2026_08_31.pptx` (28 slides, 4.1 MB; gitignored) + copy
`figures/supervisor_meeting_2026_08_31_v1.pptx`. Flow = answers to Gal's May asks (Aug 23–29 crux runs, DI wall, ceiling)
→ theory chain (measurement system → rank → spectrum → noise floor) → the secret-swap instrument (d², 3-way cross-fit,
pre-registered arm table) → battery results → three worlds + decisions → 5 appendix slides. Every slide has speaker
notes (WHAT / WHY THIS FUNCTION / WHY REPRESENTATIVE / GAL-ASK / CAVEATS / PROVENANCE).

## Done this session
- Plan (approved after two reframes) → `scripts/deck/` modular python-pptx generator (config, helpers incl. set_notes,
  mathtext eq_render, six slides_*.py modules built by parallel sub-agents, orchestrator with chunked spire audit renderer).
- `scripts/deck/make_deck_figures.py` → 14 clean figures in `figures/deck_2026_08_31/` from the same result files as the
  analysis generators (NEW plots for arms B/C/D/E, null-diag, ViT, d*, atlas 2-panel).
- Docs: STATUS.md (top entry), LESSONS_LEARNED.md (deck-generator lessons), CLAUDE.md (deck generator section),
  NEW `docs/presentation-remarks-log.md`.

## Next step(s)
- Present tomorrow. If Gal gives slide feedback: log it in `docs/presentation-remarks-log.md`, edit the relevant
  `scripts/deck/deck/slides_*.py`, rebuild with `python scripts/deck/build_deck_2026_08_31.py --render <dir>`.
- Optional polish: slide 7 (DI stack) has a wide gap between the ten-image rows; slide 18 tag wording
  ("your ask: direct inversion → KKT?") could be clearer.
- Unchanged open science items (from previous handover): activation crux dataset-dependence (flowers band), g₀ canonical ρ
  + USPS counterexample, instance-level atlas zoo (the --same_digits run never actually ran), SimuDy reply sent/unsent
  contradiction between thesis_scientific_summary.md and next_experiment_plan.md.

## Open threads / gotchas
- spire free tier renders only 10 slides per file → renderer splits into chunks; previews carry an "Evaluation Warning".
- Default python3 lacks scipy; the atlas figure uses an inline numpy re-implementation (do NOT import atlas_analyze).
- Audit banned-strings: 0/40, ‖ΔW‖/‖W₀‖, 0.226, 1.07, ssim_norm 0.6x, confirmed/settled (except "settled on your side").
- `.tmp_pptx/` (May generator) is gitignored; the new generator lives in tracked `scripts/deck/`.

## Pointers
- Build: `python scripts/deck/make_deck_figures.py && python scripts/deck/build_deck_2026_08_31.py --render /tmp/deck_render`
- One module: `python scripts/deck/preview_module.py deck.slides_measure /tmp/prev`
- Contract: `scripts/deck/SLIDE_CONTRACT.md`; plan: `.claude/plans/help-me-plan-a-keen-wigderson.md`
- Story sources: `notes/thesis_note_v2.md`, `notes/identifiability_feasibility_revision.tex:41-153`,
  `notes/dataset_sensitivity_program_plan.md` (§II rules, §III table), `notes/meeting_prep_2026-08-31.md`.

---

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

---

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

---

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

---

# Handover — 2026-09-03 17:15

## State
Branch `step1-activation-rescore-retrieval`. My lane this session was the **document**, not the compute:
`notes/exact_channel_rev10.tex` (~21 pp) is a self-contained, theorem-first Rev 10 delta section for the
Plan of Record, rebuilt from a prose draft into Definition → Lemma → hypotheses → Theorem → proof form and
audited by three independent sessions. It builds locally to `notes/exact_channel_rev10.pdf` via
`bash scripts/rev10_figs/build_pdf.sh` (static musl tectonic; the PDF is gitignored, the .tex is not).
The exact-inversion **experiments are owned by another session** (executor, currently `yoado-f4`; the
lineage rotated twice today: yoado-6c → yoado-b7 → yoado-f4). Ten of its jobs are still running.

## Done this session
- Rewrote the mathematical core as theorems with proofs; every hypothesis (A1)–(A6) stated as a formula
  where it is used. Added Corollary (general adapted layer: `k < m+r−N+1` without softmax, `k ≤ mr/N` for
  `N ≥ r`) and the Q-parametrisation (seed unknowns `N(N+1)/2` instead of `rN`), both derived-not-run.
- **Theorem I (capacity)** `k < m+r−N`: strict form DERIVED from the softmax simplex constraint
  (`1ᵀB_t = 0`), measured sharp at N=4,8,14, at five head widths, three ranks on MNIST, and on a TRAINED
  784→1000→1000→10 MLP with unseen test digits.
- **Theorem II (imprint)** `B_T = Σ_i C_i`, `C_i = −ηs Σ_t D_t[:,i](A_t h_i)ᵀ`: what the release records per
  example is bounded by that example's own accumulated error. Figure `figures/rev10/fig_imprint.png`.
- Wrote up, with scoping: encoder-quality ladder; where (A4) fails; out-of-distribution data; the
  attacker-side rank trigger and its prevalence; chart-dependence and chart-richness; new-class (CIFAR).
- **Four retractions of my own claims**, all caught by peer audit and all logged in LESSONS_LEARNED
  (items 10–15): per-example P_T columns (basis/order-dependent); the feature-Gram "coupling"; the
  "richer chart costs conditioning" trade-off; the averaged encoder rung (2.0e-15 — a value no measurement
  lies within three orders of).
- Restructured §measured into four labelled groups and rewrote §"What to say to Gal" as a two-theorem
  pitch (the last supervisor meeting failed on "correlations, no tool").

## Next step(s)
1. **The matched control for the EMNIST new-class cell (job 658575) has not landed.** Two `batch=new` rows
   are in: on the 98.2% MNIST model an 11th class (EMNIST 'a') gives `rank_B_T = 8` with all imprints within
   ~3.7× and imprint-Gram σ_N/σ_1 ≈ 0.092 (present and aligned). The pre-registered prediction is that the
   quality ladder FLATTENS for a new class where ordinary data loses rank — i.e. "how good the model is" and
   "has it seen this category" are independent axes of exposure. **Do not write that up until the
   same-job control (MNIST digits on the extended head, same chart and k) is in**, because at the strong
   encoder ordinary digits give rank 8 on-chart/distinct and rank 6 raw/distinct, so the comparison must be
   matched within the job.
2. The attacker-realisable arm (job 650890, `most_leaking.py`) is the first cell that starts from random
   public-scale coordinates rather than near the truth. Headline must be the **selection** — does
   `argmin_residual` pick the right label and image — not the success rate. `k ∈ {24,25,26}` tests the
   one-image line `k < m+r−1 = 25` in the only arm an attacker could occupy.
3. Job 652786 is the negative control that matters: act on a trigger that fires for the WRONG reason
   (random encoder, eight 1s, collinear features). If the argmin pick is wrong there, the .tex footnote on
   the trigger becomes a measured caveat instead of a caution.
4. Still open in §opens: global uniqueness; the fibre's extent past the line (one direction of forty walked,
   one cell still rising); the accuracy at which (A4) begins to fail; multi-layer LoRA (measured, but both
   arms 20 orders off the floor with the ORACLE arm nearly as bad, so no identifiability statement);
   Adam's predicted one-unit-higher line; the schedule family under weights-only release.

## Open threads / gotchas
- **Running jobs**: 614344 (strong/mid sweep), 624463 (chart budget rerun at 3000 iters), 624465 (β-VAE),
  624573 (labels/random-encoder), 634238 + 644064 (subset & OOD inversions), 650890 (most-leaking),
  652786 (negative control), 656205 (CIFAR new class), 658575 (MNIST/EMNIST new class).
- **Three peer sessions in play**: executor `yoado-f4`; theory auditor `yoado-77`; claims auditor `yoado-36`.
  They have both offered further passes. Names rotate — use `ListAgents` and re-identify rather than
  assuming a handle.
- **The tooling failure that bit three times today**: a Python patch script that applies several edits in
  memory and writes once at the end silently DISCARDS everything if a later assert fails — and it once
  produced a commit whose message claimed edits the file did not contain. Write after each edit, and verify
  target strings in the file rather than trusting the script's own "ok" lines.
- **Never read `jac_sigma_min` or `jac_cond`** — those are at the point the solver stopped. The theorems are
  about the Jacobian at the truth: use `jac_sigma_min_truth`, and compute cond as
  `jac_sigma_max_truth / jac_sigma_min_truth`. This substitution produced three separate wrong readings.
- **Never read the jsonl `verdict` field** into a document: it thresholds on image error alone and calls a
  9e-7-residual run "recovered".
- The `labels` field in `step51`/`step58` is the STRING "distinct"/"repeated", not a list — classifying by
  `len(set(...))` silently mis-splits every group.
- All MNIST/trained cells start from truth + 10% noise (`--init near`) and are identifiability tests, not
  attacks; the .tex says so. The only exception is 650890.

## Pointers
- Deliverable: `notes/exact_channel_rev10.tex` (+ wrapper `notes/exact_channel_rev10_main.tex`); build with
  `bash scripts/rev10_figs/build_pdf.sh`; figures `figures/rev10/` from `scripts/rev10_figs/fig_*.py`
  (CPU-only, read committed jsonl). Rev 9 source is Mac-only, so the .tex carries a MERGE NOTE mapping every
  Rev 9 number it uses — the user merges on Overleaf.
- Executor's authoritative write-up: `experiments/exact_inversion/RESULTS.md` (retracted claims are stamped
  in place, not deleted). Durable record: `STATUS.md` top section; pitfalls `LESSONS_LEARNED.md` items 10–15.
- Memory: `project_exact_inversion_capacity_law` carries the corrected framing — the law bounds the CHART
  COORDINATES, never say "wrong image" or "recognisable", and reconstruction factorises as
  capacity × chart quality.

---
# Handover — 2026-09-03 20:04

## State
Branch `step1-activation-rescore-retrieval`, HEAD `de78f94`. The exact-inversion thread (executor session;
write-up session yoado-ed owns `notes/exact_channel_rev10.tex`; genuineness auditor yoado-6e has closed on every
script) produced today the imprint law, the certificate channel, and the first from-nothing recoveries. Thirteen
WEXAC jobs are still running; every one has a Monitor watch in this session, and the results they produce
should be read against the pre-registered predictions in `experiments/exact_inversion/RESULTS.md` (Steps 13–22).
Authoritative record: RESULTS.md (read Steps 18–22 first); STATUS.md top section; LESSONS_LEARNED.md top entries.

## Done this session
- **Imprint law** (Step 18): `B_T = Σ_i C_i`, `‖C_i‖ ∝` the accumulated softmax residual of image i (Kendall 28/28
  on the strong model; rank `B_T` = number of images above the floor in all 40 batches); recording is decided by
  margin order within the batch (9/9). A 98% model records nothing of confident digits (imprints ≤ 1e-24,
  rank 3) and everything of a new class (flowers on CIFAR, letter 'a' on MNIST; rank 8, aligned, cosine ~.5).
  Withdrawn en route: feature-Gram coupling (QR-basis artefact), label-multiset causation (confounded draws),
  "richer chart ⇒ worse conditioned", "foreign sets drawn better" (resolution), and a false "0/500 basin
  collapse" (a raw cell).
- **Certificate** (Step 22): `C = P_{row(B_T)⊥}A_T`, `Ch_i ≈ 0` exactly for recorded images (1e-16…1e-8 vs 0.1–1
  invisible) — recipe-free, label-free. Certificate-only inversion from RANDOM public-scale starts, on-chart,
  below `k < r − N′`: 51% of 2,000 starts land exactly on private images at k=6 (18% at k=8), floor fraction =
  recorded fraction (kernel count measured), argmin reliable below the line, spurious zeros dense at/above.
  Objective must be `‖Cφ‖/‖A_Tφ‖` (a constant-normalised form let a blank image win; audit catch).
- **Rank is a leakage dial three ways** (fixed-k arm, job 721391): at k=8 the basin is 16.6 → 74.4 → 96.4% at
  r = 16/32/64 (1/17/49 below the line) — distance below the line governs, rank buys reachability as well as
  budget; fidelity axis standalone (`chart_fidelity.py`, 2,000 held-out digits): class survival .52/.68/.88/.97 and
  instance survival .04/.11/.57/.94 at k = 6/8/16/32.
- **Membership needs no fidelity**: a recovered projection identifies its source among 10k projected candidates
  with certainty at every k, robust to 3% coordinate noise at k ≥ 8.
- **Second certificate cap `N′ ≤ m − 1`** (softmax simplex): twenty recorded digits on a 10-class head → rank 9,
  certificate dead for all; the SAME MLP with a padded 26-logit head → rank 19 (20 at tolerance 1e-14, rank C 44),
  residuals to 1e-8…4e-3. The twentieth image is collinearity (σ 6e-14 vs imprint 3e-7), not precision.
- **Precision**: dynamic range decides, not mantissa — FP16 underflows the small imprints, bfloat16 keeps them
  coarsely (most revealing low-precision format); structural closure only when many examples are comparably
  recorded. **(R5)** only `lr/N` enters the recurrence — N is not identifiable, only N′ (measured 6/6).
- Subset ("find some") test: recipe error found (batch size is part of the recipe) and fixed (`lr·N′/N`);
  corrected first row: predicted floor = residual at the recorded truth (1.1e-16 both).

## Next step(s)
1. **Read job 728592** (`step76_r64k32_728592.jsonl`): r=64, k=32, confident on-chart — the cell combining budget
   (line 56, 24 below), basin (saturating with distance) and fidelity (instance id .94 at k=32). If random
   starts land on private images there, it is the thread's headline: instance-identifying images from random
   starts with no recipe. Send the row (not a reading) to yoado-ed.
2. Read the rest as they land, against RESULTS' pre-registrations: 725918 (wide-head twenty-image landings;
   extension pinned by the boundary image — read the nineteen), 721391 (per-rank sweeps), 706721 (k=10 bracket:
   floor and recorded fractions must come APART; hard1_diff), 706597 (subset controls: one-swapped and
   confident-only must NOT reach the recorded floor), 650890 (one-image attack: argmin label over 10 must be 0;
   k=24/25/26 line), 652786 (negative controls: random-encoder false-positive trigger must fail), 644064 (OOD
   inversion grids), 656205 (flowers mixed 1/4/7), 658575 (letters on mid/weak — predicted: ladder FLATTENS for a
   new class), 614344 (98% sweep — alias form?), 624463/624465 (3000-iter chart reruns, β family; fidelity
   ranking embargoed until then).
3. When 624463/624465/614344 finish (they hold `lora_exact_inversion.py`, `vae_chart.py`, `conditional_charts.py`,
   `trained_backbone.py`): the deferred edits in one commit — `jac_cond_truth` field per cell script, β kwarg
   fold-back into `vae_chart.train_vae`, off-chart best-point residual, `median_gap_to_chart`, per-iteration
   trajectory trace in `invert_lm`, certificate block appended to the full residual (report with/without),
   subset oracle flag. Run `basin_predictors.py` over all ladder cells (pool per-cell taus, never raw pairs).
4. If wanted: the 26-class head as the basin-with-power cell (built: `models/exact_inversion/mnist_mlp_m26_strong.pth`).

## Open threads / gotchas
- Running jobs (all watched): 728592, 725918, 721391, 706721, 706597, 650890, 652786, 644064, 656205, 658575,
  624463, 624465, 614344. Several share GPU nodes and are slow (3000-iteration LM cells take hours).
- Launch new modules with `python -u -m experiments.exact_inversion.<mod>` (a file path dies on import).
- Part B of `certificate.py` is meaningful ON-CHART only (`--settings on`); raw truths are not on the chart.
- `invert_lm`'s `restarts` = number of ATTEMPTS (0 runs nothing).
- `subset_and_ood.py` and `most_leaking.py` write rows as produced; older runs wrote only at the end.
- Every recovery cell except the certificate arm starts NEAR THE TRUTH (identifiability test, flagged per row).
- Read `setting` and `_truth` fields before any number; check the residual at the truth before reading a solve.
- The auditors' rules that bit today: grep for retraction survivors; matched-nuisance controls for cross-set
  claims; per-image quantities must be basis-independent; a count needs its gap; tolerance is the attacker's knob.

## Pointers
- RESULTS: `experiments/exact_inversion/RESULTS.md` (Steps 18–22 + corrections); STATUS.md top; LESSONS top.
- Scripts: `certificate.py`, `tolerance_sweep.py`, `precision_check.py`, `chart_fidelity.py`, `basin_predictors.py`,
  `margin_check.py` (imprints, traced release), `subset_and_ood.py`, `most_leaking.py`, `new_class.py`,
  `train_strong_backbone.py --n-out 26`, `batch_scale_check.py`, `blur_control.py`.
- Figures: `figures/exact_inversion/certificate_recovery_k6_706721.png`, `newclass_recoveries_preview.png`.
- Submit pattern: `bsub -q long-gpu -gpu "num=1" -R "rusage[mem=8192] select[ngpus>0]" -J <name> -o scripts/wexac_logs/<name>_%J.out -e ... <<'EOF' ... set +u; source .../conda.sh; conda activate .../rec; cd /home/projects/galvardi/yoado; python -u -m ... EOF`
- Siblings: yoado-ed (write-up, uds:/run/user/50309/cc-socks/4170091.sock), yoado-6e (auditor, ...4170067.sock).
