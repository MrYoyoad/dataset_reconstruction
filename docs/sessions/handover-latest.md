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
