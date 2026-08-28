# Full-FT Valley Comparison — Plan v1 (2026-08-28, PLANNER draft, pre-audit)

**Parent program:** [dataset_sensitivity_program_plan.md](dataset_sensitivity_program_plan.md) (v3).
This extension slots in as **§III.2-FV** (after S1, feeding S2/§III.6), and inherits every §II metric
rule and §V design lock verbatim. Status: DESIGN — to be adversarially audited (metric-rigor +
theory sessions) before any submission. Nothing here has run.

## 0. Mission (one-liner)

> Measure the **valley width** — the diameter of the indistinguishability class an adapter pins a
> training image to — in the **full-weight regime** (the regime of Haim et al., where pixel
> reconstruction WORKS), with the **same instrument** (3-way whitened sensitivity) and the **same
> distance dial** (similarity_ladder rungs) as the LoRA measurement, and compare the profiles.

**Why.** The LoRA distance dial (job 268959, STATUS 2026-08-28) showed near-duplicate swaps are
nearly invisible (sens 0.03–0.07) while cross-digit swaps are large (8–24): the adapter pins an
image only to an indistinguishability class ("valley"). The working hypothesis for the §I.2 0/40
reconstruction failure vs Haim's success: **in full training the valley is much NARROWER** (all
layers × width supply thousands of measurements), pushing the class diameter below perceptual scale
— Haim's soft/blurry outputs and sharpest-for-outliers pattern being the valley diameter made
visible. This experiment turns "their attack works, ours doesn't" into ONE measured quantity
compared across regimes. It is the empirical companion of the §I.3 Fisher bridge (valley width ≈
the inverse-Fisher diameter, measured rather than derived).

## 1. Design decisions (each with the rejected alternative)

**D1 — WHICH full regime: (a) full fine-tuning from the SAME pretrained θ₀ [PRIMARY] + (b) a small
Haim-faithful anchor [OPTIONAL, 1 config].**
(a) is the controlled comparison: identical θ₀ (`_honest_target` checkpoint), identical D
(get_finetuning_data seed=42, N=16), identical targets/rungs/T=1000/full-batch GD/float64/gelu,
identical metric call — the ONLY change is the parameterization (all-layer full-rank Δθ vs
rank-8 single-layer BA). One variable moves; any profile shift is attributable.
(b) Haim-faithful (from-scratch small init, ~1e6 epochs to near-KKT) is the regime where
reconstruction is PROVEN, but it changes θ₀, the loss trajectory, and the convergence regime all at
once, and costs ~1e3× per training — unusable as the measurement grid (9 rungs × K=50 retrains).
Keep it as ONE anchor config (§2 arm H): budget-Haim (T=50k–100k GD steps, N=16, reduced rungs,
K=20) with β(T)/stationarity diagnostics reported so we state honestly how close to KKT it got.
P3 (reconstruction-fidelity vs valley width) lives only on this anchor and is optional.

**D2 — SEED-NOISE SOURCE for Σ in full FT (make-or-break).** LoRA's reseed randomness is the B0
init: function-preserving at t=0 (A0=0 ⇒ ΔW=0), it randomizes the training PATH/preconditioning of
one fixed function. Deterministic full-batch full FT from fixed θ₀ has NO randomness ⇒ Σ=0 ⇒ the
whitened metric is undefined. Candidates:
- (i) **small random init perturbation θ₀ → θ₀ + ε·ξ_seed [CHOSEN]** — the closest structural
  analogue: like B0, it perturbs the training operator (linearization point / effective NTK) of the
  same starting model, works under unchanged deterministic full-batch GD, and cancels per pair in
  v_j = Δθ(D,s_j) − Δθ(D′,s_j) exactly as B0 does. Unlike B0 it is not exactly function-preserving
  at t=0 — mitigated by ε-calibration + the ε-bracket + the P4 exchangeability control below.
- (ii) minibatch/SGD order — rejected as primary: changes the ALGORITHM (all LoRA arms are
  full-batch GD); at N=16 it also changes memorization dynamics. Kept as a one-config sensitivity
  check only if the auditors demand it.
- (iii) dropout — rejected: changes the architecture and the trained function.
**Gauge:** Δθ = θ_T − θ₀_canonical (attacker-visible; INCLUDES ε·ξ — the injected noise is then
part of Σ honestly, exactly as B0-randomness is part of ΔW=BA). The path-noise-only variant
(subtract ξ) is computed as a diagnostic, never the headline.
**ε-calibration (pre-registered procedure, frozen before any rung runs):** choose ε (per-layer:
ε·std(θ₀,ℓ)·ξ, ξ~N(0,1)) so the trained reseed RMS spread of the LAYER-0 block matches the LoRA
r=8 arm's measured reseed_noise on the same block (the shared readout surface). One calibration
job; ε frozen; the same ε machinery reused for the full-rank-single-layer rung of the ladder.
**Sensitivity check on the choice (mandatory):** rerun {p00_identity, p0_noise, r_cross} at ε/3
and 3ε — pre-registered: the NORMALIZED profile s(d) (§ D3) moves d* by < 2× across the 9× ε
bracket, while raw sens scales ≈ ε⁻² (linear-response gate; a fitted exponent far from −2 flags
that ε is out of the linear regime — shrink ε and redo).

**D3 — VALLEY-WIDTH FUNCTIONAL.** "Invisible" is a MAGNITUDE statement (LoRA near-dups were
already DETECTABLE, p=0.002 at K=50). Per regime and target define the **normalized profile
s(d) = sens(d) / sens(r_cross)** (r_cross = the cross-digit same-parity far anchor, the largest
valid same-label swap) and the **valley width d\*** = the d_pixel value where s crosses **0.1**
(linear interpolation between rungs; d_pixel is the x-axis — it was the BEST predictor in the LoRA
dial, +0.807 pooled, and is model-free; d_encoder reported alongside). Normalization removes the
regime-scale (different Σ floors, different ambient D — the §II rule-3 dimension confound) and
keeps the SHAPE, which is the object of comparison. Robustness: d\* also reported at thresholds
{0.3, 0.03}; conclusions must be threshold-stable in ORDERING. RAW sens at the p0_noise rung is
reported alongside (fixed K=50, stated as lower-bound-at-K per §II rule 3), plus per-rung p-values.
**Normalizer validity gate:** if p(r_cross) > 0.05 in any regime, s is undefined there — the arm is
VOID for that regime (pre-declared, not patched post hoc).

**D4 — PER-LAYER READOUT (the depth question).** Full Δθ blocks: L0 784×1000, L1 1000×1000,
L2 1000×1 (weights only; biases frozen, mirroring LoRA which trains only W). Report the whitened
metric per layer separately AND all-layers-concatenated (the metric is architecture-agnostic —
flattened lists in). Memory: K=50 × D=1.79M float64 ≈ 0.7 GB per stack, thin-SVD on K×D is
O(K²D) — fine on a fat node's CPU; per-layer runs are views of the same saved stacks, so the
concat is optional if memory bites. Every rung SAVES its per-layer Δθ stacks (.pth) so per-layer
rescoring is a CPU job, never a retrain.

## 2. Arms & configs (all bsub-only; stage-0 smoke first; rsync before submit)

Common: mnist, N=16 (seed 42), gelu, T=1000, full-batch GD, float64, K=50 headline / K=12 stage-0,
2 targets (the SAME target positions as job 268959), the EXACT similarity_ladder rungs incl.
p00_identity, whitened_sensitivity(n_folds=5, p_max=3, n_perm=500). Implementation: a
`train_full()` sibling of `arm_b_dilution.train_adapter` (same loop, same forward via
`forward_logits` with all layers trainable and no BA) — swap ONLY the parameterization; import,
never re-implement (design lock). lr per regime tuned ONCE on the baseline to the memorization
gate (max_bce < 1e-3 at T=1000), then frozen (see open question Q2).

| Arm | Parameterization | Noise source | Rungs | Purpose |
|---|---|---|---|---|
| A (exists) | LoRA r=8, layer 0 | B0 reseed | all 9 | job 268959 = the LoRA profile (re-used, not re-run) |
| B | LoRA r=32, layer 0 | B0 reseed | all 9 | rank ladder rung |
| C | FULL-RANK single layer (L0 only trainable) | ε-perturb of L0 | all 9 | rank→∞ at fixed depth |
| D | FULL FT all layers [PRIMARY] | ε-perturb all layers | all 9 | the full-regime profile + per-layer readout |
| E (control) | LoRA r=8, layer 0 | ε-perturb of B0-equivalent scale (B0 fixed, θ₀-perturb) | {p00, p0_noise, r_med, r_cross} | **P4 noise-exchangeability**: same arm, both noise sources |
| F | LOO removal, BOTH regimes (full-all vs LoRA r=8) | per-regime as above | one contrast (drop i), 6–8 targets | **leave-one-out weight footprint** (§2.1) |
| G | Jacobian J = ∂vec(Δθ)/∂a, BOTH regimes | ε / B0 ensembles for SNR-whitening | tangent bases, reduced config | **local valley geometry** (§2.2) |
| H (anchor, optional) | Haim-faithful from-scratch | ε-perturb of the small init | {p00, p0_noise, r_med, r_cross}, K=20 | regime-where-reconstruction-works anchor; P3 |

The A→B→C→D ladder is the design spine: the valley narrowing monotonically with measurement count
is the most elegant version of the whole comparison, and C isolates rank-at-fixed-depth from D's
depth effect. Arms F and G are first-class (user-specified core scope), not optional: F measures
the valley on the REMOVAL functional (the Feldman LOO object), G measures its LOCAL differential
geometry (J's small singular directions ARE the flat valley).

### 2.1 Arm F — leave-one-out weight effect, full vs LoRA

Not (only) swaps: **REMOVE example i entirely** and measure the weight-space footprint
v_j = Δθ(D, seed_j) − Δθ(D\{i}, seed_j), same K-seed pairing, on (i) the FULL network (all layers,
per-layer readout, arm-D parameterization + ε-noise) and (ii) the LoRA adapter ΔW=BA (arm-A
parameterization + B0-noise). Readouts: whitened sensitivity AND the raw coherent norm
‖mean_j v_j‖ per layer. Targets: the 2 dial targets PLUS 4–6 more class-1 slots (LOO is one
contrast per target, so more targets are affordable; more targets = the rank statistics below).
**N vs N−1 caveat (the h_spotcheck confound, encoded up front):** removal changes set size AND
class balance (8→7 class-1) — a constant offset shared by all class-1 targets, NOT a per-target
signal. Mitigations inherited from h_spotcheck: (a) all cross-target and cross-regime comparisons
are RANK/paired comparisons (immune to per-construction constant offsets); (b) class-1-only
targets ⇒ the balance shift is identical across targets; (c) ds_mean stays FROZEN from full D;
(d) no m=1-style degeneracy at N=16 (7 class-1 remain). A matched-N alternative (swap-to-duplicate
instead of drop) is listed as open question Q7.
**Connection:** h_spotcheck (job 272309) measured this object BEHAVIORALLY — ρ(LOO-mem, sens) =
+0.88. Arm F measures the WEIGHT-SPACE side of the same Feldman LOO functional, in both regimes:
does the removal footprint that drives behavioral memorization live early or deep, and is it
proportionally larger in the full-weight regime?

### 2.2 Arm G — Jacobian comparison (the "secret a's")

The jacobian_spectrum machinery (`experiments/jacobian_spectrum.py`: `exact_jacobian` via
forward-over-reverse `jvp_double`, tangent bases via `build_tangents`, `spectrum`/`snr_spectrum`/
`q_eff`) measures **J_LoRA = ∂vec(BA)/∂a** — a = coordinates of a secret perturbation of the
private images along a tangent basis U. Arm G measures **the SAME J for the full-training map**:
J_full = ∂vec(Δθ_full)/∂a through the unrolled full-parameter training (create_graph unroll ⇒
GELU mandatory, never modified_relu — the standing constraint), and compares the two **on the SAME
tangent directions a** (identical U, identical D, identical θ₀): r_J(full) vs r_J(LoRA), the
singular spectra, SNR-whitened q_eff (Σ from each regime's own noise ensemble via
`estimate_sigma_seed`/`snr_spectrum`), and the **per-layer decomposition of col(J_full)** (energy
of each left singular vector across the L0/L1/L2 blocks). Additional targeted readout tying G to
the dial: for a_nn = the tangent direction from T toward its near-duplicate and a_far = toward the
far anchor, compare ‖J·a_nn‖/‖J·a_far‖ per regime — the LOCAL linearized version of s(d).
**Reconciliation duty:** this arm must be reported against the existing Jacobian identifiability
results ([leakage_story_consolidated.md](leakage_story_consolidated.md): r_J/q_eff high = "the
geometry leaks") and the §I.4 rank-reconciliation (r_J saturates in LoRA rank) — G extends that
saturation curve past the rank axis onto the parameterization axis.
**Compute note:** the full-parameter unrolled double-backward is heavier than the LoRA case (Δθ is
all-layer, ~1.79M outputs; J_full is [1.79M × Nk]). Reduced config as PRIMARY: N=4, k=8, T=5
(matching the existing j1 runs so J_LoRA is directly comparable/reusable), **layer-0-only J_full
first** (like-for-like with J_LoRA on the same weight block), all-layer J_full as the headline
stretch. T=5 keeps the unroll graph small; storing J_full at 1.79M×32 float64 ≈ 0.5 GB is fine.
The T=5-vs-T=1000 regime mismatch with the dial is open question Q8.

## 3. Predictions & kills (all signed, pre-registered)

| # | Prediction (signed) | Kill / alternative reading |
|---|---|---|
| P1 | **d\*_full(D) < d\*_LoRA(A)**: s_full(d) rises EARLIER; s_full(p0_noise) ≥ 3× s_LoRA(p0_noise). Consistent with "reconstruction works because the valley is sub-perceptual". | Profiles identical (d\* ratio in [0.7, 1.4], per-rung s CIs overlapping) ⇒ valley width does NOT explain the reconstruction gap — the gap is pipeline/decoder or measurement-count-independent. Informative: kills the valley narrative, redirects to §III.6 Fisher bridge. |
| P1b | Ladder monotone: d\*(A) ≥ d\*(B) ≥ d\*(C) ≥ d\*(D). | Non-monotone ⇒ rank per se is not the valley knob; identify which step breaks. |
| P1c (theory, Jang 2024) | Since r=32 ≥ N=16 (NTK-equivalence regime), the B→C step is SMALL (profiles near-overlap) and the C→D step (adding DEPTH) carries most of the narrowing. | B→C large ⇒ the r≳N equivalence does not govern valley width — a publishable qualification of Jang-style equivalence for privacy. |
| P2 | Per-layer (arm D): L0 profile rises at smaller d than L1, L2 (pixel-carrying layer records the instance; deep layers the concept): d\*_L0 < d\*_L1. | Reversed ⇒ instance information lives DEEP — surprising; revisit the gradient-recording locality story before any write-up. |
| P3 (arm H anchor, optional) | Per-image Haim reconstruction SSIM anti-correlates with per-image valley width: spearman ≤ −0.4. | No association ⇒ the valley-diameter reading of Haim blur is not supported; demote to metaphor. |
| P4 (control) | Arm E: LoRA r=8 normalized profile under ε-noise ≈ under B0-noise (per-rung s within CIs). | Mismatch ⇒ the noise analogue is NOT exchangeable; the cross-regime comparison downgrades to qualitative and D2 must be re-based (this is the honest failure mode of the whole design — surface it, don't bury it). |
| P5 (arm F) | (a) LOO removal detectable at p=0.002 in BOTH regimes; (b) per-target LOO footprints rank-correlate across regimes (spearman ≥ +0.5 over 6–8 targets) — the same images have the biggest removal footprint under either parameterization; (c) the FULL-regime per-layer LOO footprint follows the P2 depth ordering (L0 carries the largest normalized share / rises earliest); (d) normalized LOO footprint (per §D3 normalization by that regime's r_cross swap) is LARGER in full than LoRA — removal is harder to hide when more of it is measured. | (b) fails ⇒ the regimes memorize DIFFERENT images — the LoRA valley is not a widened version of the full-FT valley, and per-image leakage predictors (g₀) do not transfer across parameterization: major, publishable. (c) reversed ⇒ removal footprint lives deep — the depth story fails on the Feldman functional even if it holds on swaps. (d) fails while P1 passes ⇒ swap- and removal-sensitivity dissociate; report both, do not average. |
| P6 (arm G) | **r_J(full) > r_J(LoRA)** at matched (N, k, T, U) — the measurement-count story made exact (r_J(LoRA) sits at its known saturation; the full map has no rank bottleneck); signed magnitude: r_J(full) ≥ 2× r_J(LoRA) at N=4, k=8. | r_J(full) ≈ r_J(LoRA) ⇒ the identifiable direction COUNT is set by the DATA (Nk), not the parameterization — the valley difference (if P1 passed) must then be spectral (direction gains), not directional (counts): reconcile with §I.4 before write-up. |
| P7 (arm G) | The singular spectrum of J_full decays SLOWER along similarity directions than J_LoRA's (J's small singular directions ARE the flat valley — the differential-geometry version of the valley width); concretely ‖J·a_nn‖/‖J·a_far‖ is larger for J_full than for J_LoRA. | Spectra identical up to global scale ⇒ the valley difference is not local-geometric (it is nonlinear/finite-distance); the dial (P1) and J disagree — this dissociation itself is the finding, and the Fisher-bridge (§I.3) inherits it. |

Thesis narrative per outcome: P1 pass ⇒ one unified quantity explains "their attack works, ours
doesn't" (valley width vs measurement count), feeding directly into §I.3's Fisher framing. P1 kill
⇒ equally citable: rank/depth do not narrow the valley, so the 0/40 gap must be pipeline-side.
P5+P6+P7 together triangulate the same object at three scales — global removal (Feldman), finite
swap distance (the dial), infinitesimal (J) — and their agreement/disagreement pattern is itself
reportable.

## 4. Asymptote & sanity checks (mandatory, house standard)

1. **d=0 identity rung ≈ 0 / p≈1 in EVERY regime** (same seed ⇒ v_j = GPU-nondet only). Any
   significantly nonzero d²(0) is an artifact-kill for that regime's entire dial. (LoRA reference:
   sens=0, p=1.000 exactly.)
2. **ε→0 limit:** Σ→0, raw sens→∞ for any d>0; s(d) (a ratio of divergents) should limit finite.
   Report the ε-bracket (ε/3, ε, 3ε) with the fitted raw-sens∝ε^α exponent (expect α≈−2) and
   d\*(ε); pre-registered stability: d\* within 2× across the bracket.
3. **Metric-CI gates on the NEW noise source** (§II standing requirement, re-run because Σ changed
   character): p-value uniformity over ≥20 reseed-vs-reseed redraws with ε-noise, and q_eff≈0 on
   that null — prerequisites for quoting ANY full-FT number.
4. **Memorization gate** every training (max_bce < 1e-3); non-memorized runs excluded with counts;
   NaN drops counted (arm-E lock).
5. **Far-anchor consistency:** arm-D r_cross sens must land at arm-C/D-scale significance
   (normalizer gate, §D3); LoRA arm-A far end already reproduces arm-C/D-scale numbers (S1 check).
6. **{K, 2K} adequacy** on one rung per regime (p0_noise), per §II rule 4.

## 5. Pre-declared plots

1. **HEADLINE — overlaid normalized profiles:** s(d_pixel), log-y, one curve per arm {A r8, B r32,
   C full-single, D full-all}, threshold 0.1 line + d\* crossings marked. Expected shape: curves
   shift LEFT monotonically A→D.
2. **Per-layer fan (arm D):** s(d) for {L0, L1, L2, concat}. Expected: L0 leftmost.
3. **Calibration panel:** raw sens vs ε (log-log, slope −2 guide) for 2 rungs; plus arm-E overlay
   (B0-noise vs ε-noise profiles) — the P4 evidence in one figure.
4. **(arm H anchor)** reconstruction-SSIM vs valley-width scatter, one point per image.
5. **Arm F — per-layer LOO-footprint fan, full vs LoRA:** grouped bars (or paired fan) of the
   normalized LOO footprint per {L0, L1, L2, concat} for the full regime beside the LoRA (single-
   block) value, one panel per target + a pooled panel; expected: L0 dominates in full, and
   full > LoRA on the normalized footprint. Second panel: cross-regime per-target rank scatter.
6. **Arm G — overlaid singular spectra of J_full vs J_LoRA** (same tangent basis, log-y, index and
   SNR-whitened variants) + an **r_J bar pair** (full vs LoRA, with the LoRA rank-saturation line
   from leakage_story_consolidated overlaid) + per-layer col(J_full) energy stack.
7. Ladder image grids per arm (house rule: visual examples, best AND worst, saved .pth tensors).

## 6. Compute & sequencing (WEXAC long-gpu; python -u; stage-0 gates)

Per full-ladder arm: K=50 baseline + 9 rungs × 50 seeds × 2 targets ≈ 950 trainings × T=1000
full-batch steps on a 784-1000-1000-1 MLP — same order as the LoRA ladder job (full-FT backprop
cost ≈ LoRA's; only the update is bigger). Estimate ≈ job-268959 wall-clock per arm; arms B, C, D
submit as parallel jobs (~1 GPU-day total). Arm E ≈ 0.4×; ε-calibration + bracket ≈ 0.3×;
per-layer rescoring is CPU-only on saved stacks. Arm F: one contrast × 6–8 targets × K=50 × 2
regimes ≈ 700–800 trainings + shared baselines ≈ 0.8× a ladder arm. Arm G: Nk=32 J columns × 2
maps at N=4/k=8/T=5 — the unroll is tiny (T=5); the all-layer stretch is memory-, not time-bound;
≈ hours (the LoRA half may reuse existing j1 runs). H anchor: 1 config × (K=20+4×20) × 50–100k
steps ≈ 1–2 GPU-days — submitted LAST, only if P1 lands. Order: (1) stage-0 smoke all arms;
(2) ε-calibration + metric-CI gates job; (3) arms B/C/D/E parallel + G (independent of the dial);
(4) arm F (reuses arm-A/D baselines); (5) CPU per-layer rescore + figures; (6) H decision.
STATUS.md + LESSONS_LEARNED.md updated same-turn per house rule.

## 7. Open questions for the auditors (genuinely contestable — attack here)

- **Q1 (D2, the big one):** is the θ₀-perturbation noise analogue defensible, given it is not
  exactly function-preserving at t=0? Is the P4 exchangeability control + ε-bracket SUFFICIENT to
  license the cross-regime d\* comparison, or must SGD-order noise run as a second full arm?
- **Q2 (lr matching):** lr tuned per-regime to a fixed memorization criterion (chosen) vs fixed lr
  (may not memorize / may diverge in full FT) vs matched trained-distance ‖Δθ‖ (changes the
  Carlini control). Which confound is least bad?
- **Q3 (normalizer):** r_cross (median cross-digit) as the s(d) denominator — selection effects?
  Alternative: normalize by the MAX rung, or by the null floor. Threshold 0.1 for d\* is arbitrary
  (hence the {0.3, 0.03} robustness band) — is ordering-stability enough?
- **Q4 (x-axis):** d_pixel chosen because it out-predicted the DINO encoder in the LoRA dial
  (+0.807 vs +0.399) — but pixel distance is regime-blind by construction; fine for comparing
  profiles, weaker as a semantic claim. Accept?
- **Q5 (gauge):** Δθ includes the injected ε·ξ (attacker-visible gauge). Should the headline
  instead be the path-noise-only variant (ξ subtracted), with the attacker gauge as diagnostic?
- **Q6 (anchor honesty):** at T=50–100k the budget-Haim anchor is NOT at KKT (β(T) evidence says
  even T=1000 LoRA isn't). Under what convergence-diagnostic threshold may arm H carry the label
  "the regime where reconstruction is proven"?
- **Q7 (arm F, N vs N−1):** are the h_spotcheck mitigations (rank/paired comparisons, class-1-only
  targets, per-construction constant offsets) sufficient for the REMOVAL contrast, or does the
  set-size/balance shift demand a matched-N control (e.g. replace-by-duplicate-of-another-member,
  which holds N fixed but changes duplication structure — its own confound)? Which is less bad?
- **Q8 (arm G, T mismatch):** J is exactly computable only at small T (T=5, unroll memory), while
  the dial and arm F run at T=1000. Is the T=5 local geometry representative — or must G report a
  small T-sweep (T=1,5,20) to show the spectra's SHAPE is T-stable before comparing with the
  T=1000 valley? Also: the SNR-whitening of J_full uses the D2 ε-noise Σ, so G inherits the whole
  Q1 dependence — flag any G conclusion as conditional on P4 passing.
