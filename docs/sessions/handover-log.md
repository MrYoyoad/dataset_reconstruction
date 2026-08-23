
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
