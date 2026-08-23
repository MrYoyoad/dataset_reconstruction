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
