# Full-FT Valley Comparison — Plan v1.2 (2026-08-28)

**v1.1 trim (user directive):** informative core only — the {A, C, D} ladder (rank vs depth is
the whole story), the E validity control, and the user's explicit F/G asks; rungs 9→6; arm B,
arm H(+P3), and full-regime replication of the resolved composition arms DEFERRED (§2.3).
**v1.2 (BOTH audits + trim reconciliations folded — FINAL, pending PASS re-checks):**
- *Metric audit (yoado-6d, CONDITIONAL PASS):* builds/stage-0/calibration cleared; **two BLOCKERS
  (§4.0: B1 dimension-invariance, B2 SGD-order cross-check) gate any cross-regime valley
  HEADLINE**; should-fixes S1–S6 folded inline; ε-bracket 3-point on the load-bearing p0_noise
  rung; d\* reported as an INTERVAL unless two rungs bracket the crossing.
- *Theory audit (yoado-18, NOT-PASS → items folded):* TB1 asymmetric (necessity-only) valley
  narrative; TB2 ABSOLUTE — zero inferential use of Haim's success (the Haim connection lives
  only on deferred arm H; P3 REMOVED); TB3 — Jang/NTK-equivalence framing DROPPED (P1c removed;
  A→C→D = combined rank+depth narrowing, no equivalence claim); TB4 — P6 re-signed
  T=5-conditional and demoted; TF5 δ₀xᵀ mechanism for P2; TF6 local-linearity wording for P7;
  TF7 gauge conditionality.
- Q2–Q8 resolved/retired (§7); Q1 narrowed, remains for the re-check round.

**Parent program:** [dataset_sensitivity_program_plan.md](dataset_sensitivity_program_plan.md) (v3).
This extension slots in as **§III.2-FV** (after S1, feeding S2/§III.6), and inherits every §II metric
rule and §V design lock verbatim. Status: **both audits + trim reconciliations folded — v1.2
FINAL, pending PASS re-checks** (yoado-18: predictions/framing; yoado-6d: §4.0 gates B1/B2).
Nothing here has run beyond stage-0/calibration clearance.

## 0. Mission (one-liner)

> **Does the PARAMETERIZATION narrow the valley?** Full-rank-all-layer Δθ vs rank-8-single-layer
> BA, under matched fine-tuning from the same θ₀: measure the **valley width** — the diameter of
> the indistinguishability class the weight-change pins a training image to — with the **same
> instrument** (3-way whitened sensitivity) and the **same distance dial** (similarity_ladder
> rungs), and compare the profiles.

**Why — and the direction of the logic (TB1, asymmetric by construction).** The LoRA distance
dial (job 268959, STATUS 2026-08-28) showed near-duplicate swaps are nearly invisible (sens
0.03–0.07) while cross-digit swaps are large (8–24): the adapter pins an image only to an
indistinguishability class ("valley"). The load-bearing direction is **NECESSITY**: a WIDE valley
means any reconstruction carries residual error ≥ the valley width, so pixel-level reconstruction
**cannot** succeed — this legitimizes reading the §I.2 0/40 as a *forced* failure in the LoRA
regime. A NARROW valley only **PERMITS** reconstruction; it never establishes or explains one.
**Haim scope caveat (TB2, one line, the whole of it):** a narrow full-FT valley would be
*consistent with* Haim et al.'s reconstruction success; this plan does not measure Haim's regime
(from-scratch, near-KKT — deferred arm H) and is not evidence about it. What this plan measures
is a prediction about the FINE-TUNING regime only. It is the empirical companion of the §I.3
Fisher bridge (valley width ≈ the inverse-Fisher diameter, measured rather than derived).

## 1. Design decisions (each with the rejected alternative)

**D1 — WHICH full regime: (a) full fine-tuning from the SAME pretrained θ₀ [PRIMARY] + (b) a small
Haim-faithful anchor [OPTIONAL, 1 config].**
(a) is the controlled comparison: identical θ₀ (`_honest_target` checkpoint), identical D
(get_finetuning_data seed=42, N=16), identical targets/rungs/T=1000/full-batch GD/float64/gelu,
identical metric call — the ONLY change is the parameterization (all-layer full-rank Δθ vs
rank-8 single-layer BA). One variable moves; any profile shift is attributable.
(b) Haim-faithful (from-scratch small init, ~1e6 epochs to near-KKT) is where Haim et al.
demonstrated reconstruction, but it changes θ₀, the loss trajectory, and the convergence regime
all at once, and costs ~1e3× per training — unusable as the measurement grid (9 rungs × K=50
retrains). It survives only as deferred arm H (§2.3), and **TB2 governs it: the label is
"closest available approximation, NOT-at-KKT" with the β(T)-distance from convergence reported;
NO convergence threshold earns "the regime where reconstruction is proven" — that label is
RETIRED, not gated.** P3 (reconstruction-fidelity vs valley width), which lived on H, is REMOVED
from this plan with it.

**D2 — SEED-NOISE SOURCE for Σ in full FT (make-or-break).** LoRA's reseed randomness is the B0
init: function-preserving at t=0 (A0=0 ⇒ ΔW=0), it randomizes the training PATH/preconditioning of
one fixed function. Deterministic full-batch full FT from fixed θ₀ has NO randomness ⇒ Σ=0 ⇒ the
whitened metric is undefined. Candidates:
- (i) **small random init perturbation θ₀ → θ₀ + ε·ξ_seed [CHOSEN]** — the closest structural
  analogue: like B0, it perturbs the training operator (linearization point / effective NTK) of the
  same starting model and works under unchanged deterministic full-batch GD. **Honest pairing
  statement (audit S6b):** the ε·ξ cancellation in v_j = Δθ(D,s_j) − Δθ(D′,s_j) is exact only at
  t=0; through training the perturbation propagates DIFFERENTIALLY along the D vs D′ paths,
  leaving an O(ε) residual in v_j — which is precisely why the α≈−2 linear-response gate exists.
  Also not exactly function-preserving at t=0 (B0 is) — mitigated by ε-calibration + the ε-bracket
  + the P4 exchangeability control + the §4.0-B2 blocker below.
- (ii) minibatch/SGD order — rejected as PRIMARY (changes the ALGORITHM: all LoRA arms are
  full-batch GD; at N=16 it also changes memorization dynamics) — but **PROMOTED by the audit to
  the mandatory §4.0-B2 cross-check on the full-FT arm**: P4 validates ε only on LoRA
  (necessary-not-sufficient; ε is isotropic-per-layer while B0 noise is structured low-rank path
  randomization — RMS-matching matches a scalar, not the eigenstructure whitening is sensitive to).
- (iii) dropout — rejected: changes the architecture and the trained function.
**Gauge (Q5 RESOLVED, audit-endorsed):** headline = Δθ = θ_T − θ₀_canonical, the ATTACKER-VISIBLE
gauge (INCLUDES ε·ξ — the injected noise is part of Σ honestly, exactly as B0-randomness is part
of ΔW=BA). The path-noise-only variant (subtract ξ) is computed as a diagnostic, never the
headline. **TF7 — inherited conditionality (state on every cross-regime readout):** LoRA's Σ is
PURE path-noise (B0 is function-preserving at t=0) while full-FT's Σ = path-noise + the DIRECT
injection ε·ξ; the ε-calibration matches MAGNITUDE, not covariance STRUCTURE. The cross-regime
comparison is therefore CONDITIONAL on P4 (with its §4.7 power gate) and §4.0-B2 throughout.
**ε-calibration (pre-registered procedure, frozen before any rung runs):** choose ε (per-layer:
ε·std(θ₀,ℓ)·ξ, ξ~N(0,1)) so the trained reseed RMS spread of the LAYER-0 block matches the LoRA
r=8 arm's measured reseed_noise on the same block (the shared readout surface). One calibration
job; ε frozen; the same ε machinery reused for the full-rank-single-layer rung of the ladder.
**Sensitivity check on the choice (mandatory; bracket restored per the audit addendum):** on the
ONE load-bearing rung, **p0_noise, run the FULL 3-point bracket {ε/3, ε, 3ε} per regime** — two
points can compute a slope but cannot VALIDATE linearity (no curvature detection, no
goodness-of-fit), and the α≈−2 gate exists precisely to catch ε leaving the linear regime.
Pre-registered: log-log raw-sens-vs-ε fit on p0_noise has slope −2 ± 0.3 with no significant
curvature (middle point on the fitted line within its CI), and d* moves < 2× across the bracket.
The other bracket rungs ({p00_identity, r_cross}) run 2 points (ε, 3ε) — acceptable there. Gate
failure ⇒ shrink ε and redo the bracket before any rung measurement is quoted.

**D3 — VALLEY-WIDTH FUNCTIONAL.** "Invisible" is a MAGNITUDE statement (LoRA near-dups were
already DETECTABLE, p=0.002 at K=50). Per regime and target define the **normalized profile
s(d) = sens(d) / sens(r_cross)** (r_cross = the cross-digit same-parity far anchor, the largest
valid same-label swap) and the **valley width d\*** = the d_pixel value where s crosses **0.1**
(d_pixel is the x-axis — **Q4 RESOLVED, audit-ENDORSED as the cross-regime axis**: best LoRA
predictor +0.807 pooled and model-free; d_encoder reported alongside and RESERVED for semantic
statements). Normalization removes the regime-scale and keeps the SHAPE — **but (Q3 RESOLVED) the
claim that s(d) cancels the ambient-dimension step EARNS the cross-regime headline only if the
§4.0-B1 dimension-invariance blocker passes; if B1 fails, the ladder trend must be
dimension-corrected per §II rule 3 before interpretation.** Robustness: d\* also at thresholds
{0.3, 0.03}; conclusions must be threshold-stable in ORDERING. RAW sens at the p0_noise rung is
reported alongside (fixed K=50, lower-bound-at-K per §II rule 3), plus per-rung p-values.
**d\* interpolation discipline (audit S6c + addendum, 6-rung spacing):** BEFORE the wave, pre-check
on the DONE arm-A data (job 268959) that ≥ 2 rungs BRACKET the expected s=0.1 crossing in each
regime, and PLACE the "one mid" rung at the expected crossing (swap p3_rot15 for another 9-rung
member if the pre-check says so). Every d\* is reported WITH its interpolation-uncertainty band
from the local rung spacing, and as an **INTERVAL (not a point) whenever only one rung brackets
the crossing** — else d\*_full < d\*_LoRA could be a rung-placement interpolation artifact.
**Normalizer validity gate:** if p(r_cross) > 0.05 in any regime, s is undefined there — the arm is
VOID for that regime (pre-declared, not patched post hoc).

**D4 — PER-LAYER READOUT (the depth question).** Full Δθ blocks: L0 784×1000, L1 1000×1000,
L2 1000×1 (weights only; biases frozen, mirroring LoRA which trains only W). Report the whitened
metric per layer separately AND all-layers-concatenated (the metric is architecture-agnostic —
flattened lists in). **Numerator/denominator decomposition per layer (audit S2, mandatory):**
report ‖Δμ_ℓ‖ (numerator) and the per-layer noise scale (denominator) SEPARATELY — ε is
L0-calibrated then scaled by std(θ₀,ℓ), so a per-layer d\* ordering could be a denominator
artifact; **"L0 records the instance" (P2) must be a NUMERATOR statement.** Memory: K=50 ×
D=1.79M float64 ≈ 0.7 GB per stack, thin-SVD on K×D is O(K²D) — fine on a fat node's CPU;
per-layer runs are views of the same saved stacks, so the concat is optional if memory bites.
Every rung SAVES its per-layer Δθ stacks (.pth) so per-layer rescoring is a CPU job, never a
retrain.

## 2. Arms & configs (all bsub-only; stage-0 smoke first; rsync before submit)

Common: mnist, N=16 (seed 42), gelu, T=1000, full-batch GD, float64, K=50 headline / K=12 stage-0
(K stays at 50 — power is not where we trim), 2 targets (the SAME target positions as job 268959).
**Reduced rung set (trim, 9→6):** {p00_identity, p0_noise, p3_rot15, r_nn, r_far, r_cross} — the
calibration anchor, the near-duplicate, one mid rung, and the retrieved bracket; the EXACT
similarity_ladder constructions, just filtered (`rung_filter`). Arm A keeps its completed 9-rung
data; cross-regime profiles are compared on the SHARED 6 rungs.
whitened_sensitivity(n_folds=5, p_max=3, n_perm=500). Implementation: a
`train_full()` sibling of `arm_b_dilution.train_adapter` (same loop, same forward via
`forward_logits` with all layers trainable and no BA) — swap ONLY the parameterization; import,
never re-implement (design lock). **lr matching (Q2 RESOLVED, audit S5):** lr per regime tuned
ONCE on the baseline to a pre-registered NARROW memorization band — **max_bce ∈ [1e-4, 1e-3] at
T=1000** (not merely <1e-3, which spans an order of magnitude) — then frozen; the ACHIEVED
max_bce is reported per regime alongside every headline.

| Arm | Parameterization | Noise source | Rungs | Purpose |
|---|---|---|---|---|
| A (exists) | LoRA r=8, layer 0 | B0 reseed | 9 done; shared 6 used | job 268959 = the LoRA profile (re-used, not re-run) |
| C | FULL-RANK single layer (L0 only trainable) | ε-perturb of L0 | the 6 | rank→∞ at fixed depth |
| D | FULL FT all layers [PRIMARY] | ε-perturb all layers | the 6 | the full-regime profile + per-layer readout |
| E (control) | LoRA r=8, layer 0 | ε-perturb of B0-equivalent scale (B0 fixed, θ₀-perturb) | {p00, p0_noise, r_far, r_cross} | **P4 noise-exchangeability**: same arm, both noise sources |
| F | LOO removal, BOTH regimes (full-all vs LoRA r=8) | per-regime as above | one contrast (drop i), 6–8 targets | **leave-one-out weight footprint** (§2.1) |
| G | Jacobian J = ∂vec(Δθ)/∂a, BOTH regimes | ε / B0 ensembles for SNR-whitening | tangent bases, reduced config | **local valley geometry** (§2.2) |
| B1 (§4.0 GATE) | arm-D stacks, dimension varied — coordinate subsets at **{~25k (≈LoRA ambient), ~100k, ~450k, 1.8M}** (re-check strengthening: a single 2× point cannot license the 70× A→D span; d\* must be flat across the fractions spanning toward LoRA's dimension) | ε (same) | rescore (multi-fraction) + fresh-seed budget | dimension-invariance of d\* across the ACTUAL comparison span — gates the headline |
| B2 (§4.0 GATE) | FULL FT all layers | SGD minibatch-order | {p0_noise, **mid/crossing rung** (the one the arm-A pre-check places at s≈0.1 — re-check strengthening: endpoints can agree while the d\*-determining crossing differs), r_cross} | noise-source consistency of full-FT d\*/s AT the d\*-determining region — gates the headline |

The **A→C→D ladder** is the design spine — COMBINED rank+depth narrowing, presented with **NO
NTK-equivalence claim (TB3: the Jang framing is dropped)**: A→C is the rank effect at fixed depth
(rank-8 → full-rank on the same single layer; this step CROSSES the r=N boundary and is labeled
as such, nothing more), C→D is the pure depth/measurement-count effect; the two step sizes are
reported separately. Arms F and G are first-class (user-specified core scope), not optional:
F measures the valley on the REMOVAL functional (the Feldman LOO object), G measures its LOCAL
differential geometry (J's small singular directions APPROXIMATE the flat valley under
local-linearity, TF6).

### 2.3 Deferred (explicitly deferred, NOT dropped — one line each)
- **Arm B (LoRA r=32 dial):** lowest marginal information — C already anchors the
  high-measurement end. Un-defer B iff the Jang/NTK-equivalence question is later wanted: it is
  the ONLY rung that can test it (the plan itself makes no equivalence claim, TB3; the former
  P1c retired with it).
- **Arm H (budget-Haim from-scratch anchor):** only if P1 lands AND the user asks. **The ENTIRE
  Haim connection lives here and nowhere else (TB2):** label "closest available approximation,
  NOT-at-KKT", β(T)-distance from convergence reported; no threshold earns "the regime where
  reconstruction is proven" — the label is retired (Q6's honest answer, Q6 itself moot). P3 was
  removed with H.
- **Full-regime replication of the composition arms (duplication/imbalance/rarity):** resolved on
  LoRA to image-identity effects (parent §III results table); no full-regime replication planned.

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
(d) no m=1-style degeneracy at N=16 (7 class-1 remain). **Q7 RESOLVED (audit S3): KEEP LOO as-is;
the matched-N replace-by-duplicate alternative is REJECTED** — it trades a rank-cancelling
constant offset for the KNOWN duplication confound (parent arm E). Readout consequence: **P5b
(cross-regime rank transfer) is the ROBUST HEADLINE**; P5a and P5d carry the N→N−1 offset caveat,
and P5d is DESCRIPTIVE only (it normalizes a removal footprint by a swap anchor — different
constructions, offset structures unverified).
**Connection:** h_spotcheck (job 272309) measured this object BEHAVIORALLY — ρ(LOO-mem, sens) =
+0.88. Arm F measures the WEIGHT-SPACE side of the same Feldman LOO functional, in both regimes:
does the removal footprint that drives behavioral memorization live early or deep, and is it
proportionally larger in the full-weight regime?
**Cheap piggyback readout (not an arm, ~0 extra compute):** correlate the full-regime per-target
removal footprints with the EXISTING g₀ values (margin_vs_sensitivity / job-272504 data for the
same images) — does the base-gradient predictor, established on LoRA (ρ=+0.78), transfer to the
full regime for free? Reported as a spearman with n stated (6–8, SMALL-n, descriptive not
confirmatory).

### 2.2 Arm G — Jacobian comparison (the "secret a's")

The jacobian_spectrum machinery (`experiments/jacobian_spectrum.py`: `exact_jacobian` via
forward-over-reverse `jvp_double`, tangent bases via `build_tangents`, `spectrum`/`snr_spectrum`/
`q_eff`) measures **J_LoRA = ∂vec(BA)/∂a** — a = coordinates of a secret perturbation of the
private images along a tangent basis U. Arm G measures **the SAME J for the full-training map**:
J_full = ∂vec(Δθ_full)/∂a through the unrolled full-parameter training (create_graph unroll ⇒
GELU mandatory, never modified_relu — the standing constraint), and compares the two **on the SAME
tangent directions a** (identical U, identical D, identical θ₀). **Readout priority (audit
S4/TB4): the arm LEADS with P7** — the RAW singular spectra of J_full vs J_LoRA (noise-free = the
clean comparison) plus the targeted ratio ‖J·a_nn‖/‖J·a_far‖ per regime (a_nn = the tangent
direction from T toward its near-duplicate, a_far = toward the far anchor: the LOCAL linearized
version of s(d), a valley statement only under local-linearity, TF6). SECONDARY and
**P4-conditional**: SNR-whitened q_eff (Σ from each regime's own noise ensemble via
`estimate_sigma_seed`/`snr_spectrum` — inherits the D2/TF7 conditionality). CONSISTENCY CHECK
only: r_J counts (the demoted, T-conditional P6). Plus the **per-layer decomposition of
col(J_full)** (energy of each left singular vector across the L0/L1/L2 blocks).
**Reconciliation duty:** this arm must be reported against the existing Jacobian identifiability
results ([leakage_story_consolidated.md](leakage_story_consolidated.md): r_J/q_eff high = "the
geometry leaks") and the §I.4 rank-reconciliation (r_J saturates in LoRA rank) — G extends that
saturation curve past the rank axis onto the parameterization axis.
**Compute note:** the full-parameter unrolled double-backward is heavier than the LoRA case (Δθ is
all-layer, ~1.79M outputs; J_full is [1.79M × Nk]). Reduced config as PRIMARY: N=4, k=8, T=5
(matching the existing j1 runs so J_LoRA is directly comparable/reusable), **layer-0-only J_full
first** (like-for-like with J_LoRA on the same weight block), all-layer J_full as the headline
stretch. T=5 keeps the unroll graph small; storing J_full at 1.79M×32 float64 ≈ 0.5 GB is fine.
**Q8 RESOLVED (audit S4 + TB4): a T-sweep T={1, 5, 20} is MANDATORY** — the spectral SHAPE and
the ‖J·a_nn‖/‖J·a_far‖ ratio must be T-stable before any small-T J is compared with the T=1000
valley; report max_bce at each T, and every G readout carries the explicit caveat
**"early-training Jacobian, not converged-valley."**

## 3. Predictions & kills (all signed, pre-registered)

| # | Prediction (signed) | Kill / alternative reading |
|---|---|---|
| P1 | **d\*_full(D) < d\*_LoRA(A)**: s_full(d) rises EARLIER; s_full(p0_noise) ≥ 3× s_LoRA(p0_noise). **Read ASYMMETRICALLY (TB1):** the wide LoRA valley makes the §I.2 0/40 a FORCED failure (residual ≥ valley width — the necessity direction); a narrow full-FT valley only PERMITS reconstruction and is stated as consistent-with, never as the cause of, any attack's success. | Profiles identical (d\* intervals overlapping, per-rung s CIs overlapping) ⇒ the parameterization does NOT narrow the valley ⇒ the 0/40 cannot be blamed on valley width alone — the gap is pipeline/decoder-side. Informative: redirects to §III.6 Fisher bridge. |
| P1b | Ladder monotone over the spine: d\*(A) ≥ d\*(C) ≥ d\*(D), with the A→C (rank-at-fixed-depth) and C→D (depth) step sizes reported separately — combined narrowing, NO equivalence claim (TB3). | Non-monotone ⇒ rank per se is not the valley knob; identify which step breaks. |
| P2 | Per-layer (arm D): L0 profile rises at smaller d than L1, L2: d\*_L0 < d\*_L1 — **stated on the NUMERATOR ‖Δμ_ℓ‖ (S2); a denominator-driven ordering does not count.** Mechanism (TF5, KKT-free and fine-tuning-valid): the layer-0 gradient has the **δ₀xᵀ outer-product structure at EVERY training step**, so L0 accumulates pixel-specific information continuously. | Deep-first ordering **REFUTES the pixel-recording-locality premise underlying the entire reconstruction/valley narrative** (TF5) — a program-level finding, not a curiosity: halt the narrative write-up until reconciled. |
| P4 (control) | Arm E: LoRA r=8 normalized profile under ε-noise ≈ under B0-noise (per-rung s within CIs), **subject to the §4.7 power gate (S1): the design must be able to detect a 2× s-difference at K=50, else the pass is INCONCLUSIVE.** | Mismatch ⇒ the noise analogue is NOT exchangeable; the cross-regime comparison downgrades to qualitative and D2 must be re-based (the honest failure mode of the whole design — surface it, don't bury it). |
| P5 (arm F) | **(b) THE HEADLINE (S3-robust):** per-target LOO footprints rank-correlate across regimes (spearman ≥ +0.5 over 6–8 targets) — the same images have the biggest removal footprint under either parameterization. Secondary, N→N−1-offset-caveated: (a) removal detectable at p=0.002 in BOTH regimes; (c) the full-regime per-layer footprint follows the P2 NUMERATOR ordering. **(d) DESCRIPTIVE ONLY:** normalized LOO footprint larger in full (normalizes a removal footprint by a swap anchor — different constructions, offset structures unverified). | (b) fails ⇒ the regimes memorize DIFFERENT images — the LoRA valley is not a widened full-FT valley, and per-image predictors (g₀) do not transfer across parameterization: major, publishable. (c) reversed ⇒ the P2 kill applies on the Feldman functional. (d) discrepant while P1 passes ⇒ swap- and removal-sensitivity dissociate; report both, do not average. |
| P6 (arm G — DEMOTED consistency check, **EXPLICITLY T=5-CONDITIONAL**, TB4) | In the early-training rank-deficiency window (eff_rank ≈ 9–13 < Nk=32 per J0), r_J(full) MAY exceed r_J(LoRA). The LIKELY CONVERGED outcome is both counts saturating at the data ceiling Nk — **count is set by the DATA, not the parameterization** (§I.4 / rank-sweep banked result). No ≥2× claim. | Either count outcome is reported descriptively; the informative object is the SPECTRUM (P7). A converged r_J < Nk in ONE regime only would be the sole surprising count result — flag and investigate. |
| P7 (arm G — **LEADS the arm**) | The singular spectrum of J_full decays SLOWER along similarity directions than J_LoRA's; concretely ‖J·a_nn‖/‖J·a_far‖ is larger for J_full. **J's small singular directions APPROXIMATE the flat valley UNDER LOCAL-LINEARITY over the valley scale — which the dial-vs-J comparison itself tests (TF6).** | Spectra identical up to global scale ⇒ the valley difference is not local-geometric (it is nonlinear/finite-distance); the dial (P1) and J disagree — this dissociation itself is the finding, and the Fisher-bridge (§I.3) inherits it. |

(P1c removed with the Jang framing, TB3; P3 removed with arm H, TB2.)

Thesis narrative per outcome (asymmetric, TB1/TB2): P1 pass ⇒ the parameterization narrows the
valley, and the wide-valley reading of the LoRA 0/40 stands as a FORCED failure (necessity
direction); nothing here is evidence about Haim's regime. P1 kill ⇒ equally citable: rank/depth
do not narrow the valley, so the 0/40 gap must be pipeline-side. P5b + P7 (with the demoted P6
count check) triangulate the same object at three scales — global removal (Feldman), finite swap
distance (the dial), infinitesimal (J) — and the agreement/disagreement pattern is itself
reportable.

## 4. Blockers, asymptote & sanity checks (mandatory)

### 4.0 HEADLINE-GATING BLOCKERS (metric audit yoado-6d — run BEFORE any cross-regime valley claim)

**B1 — DIMENSION-INVARIANCE CONTROL.** The A→D axis is collinear with a 25k→1.8M
ambient-dimension step (~70×) — the exact §I.4 confound; the claim that s(d)-normalization
cancels it is otherwise untested against a rung×dimension interaction. Run: ONE regime (D),
signal fixed, dimension varied — measure s(d)/d\* on all-layers vs a RANDOM HALF of the
coordinates (seeded; primarily a rescore of arm-D's saved stacks, with fresh seeds budgeted if
independence demands — ~350 trainings reserved). d\* invariant ⇒ the normalization is validated
on this axis; d\* moves ⇒ the ladder trend must be dimension-corrected (§II rule 3) before any
interpretation.

**B2 — SGD-ORDER-NOISE CROSS-CHECK (full-FT, one config).** P4 validates ε only on LoRA
(necessary-not-sufficient; ε is isotropic-per-layer, B0 noise is structured low-rank path
randomization — RMS-matching matches a scalar, not the eigenstructure whitening is sensitive to).
Run the full-FT arm under SGD minibatch-order noise on {p0_noise, r_cross} (~200 trainings):
full-FT d\*/s must be CONSISTENT between ε-noise and SGD-order noise, else "narrower valley" is a
synthetic-noise artifact.

### 4.1 Standing checks (house standard)

1. **d=0 identity rung ≈ 0 / p≈1 in EVERY regime** (same seed ⇒ v_j = GPU-nondet only). Any
   significantly nonzero d²(0) is an artifact-kill for that regime's entire dial. (LoRA reference:
   sens=0, p=1.000 exactly.)
2. **ε→0 limit:** Σ→0, raw sens→∞ for any d>0; s(d) (a ratio of divergents) should limit finite.
   Report the ε-bracket per the D2 spec — **3 points {ε/3, ε, 3ε} on p0_noise** (slope −2 ± 0.3,
   no significant curvature), 2 points on the other bracket rungs — and d\*(ε); pre-registered
   stability: d\* within 2× across the bracket.
3. **Metric-CI gates on the NEW noise source** (§II standing requirement, re-run because Σ changed
   character): p-value uniformity over ≥20 reseed-vs-reseed redraws with ε-noise, and q_eff≈0 on
   that null — prerequisites for quoting ANY full-FT number.
4. **Memorization gates:** baseline lr tuned into the pre-registered band max_bce ∈ [1e-4, 1e-3]
   (S5); every training then gated at max_bce < 1e-3; non-memorized runs excluded with counts;
   NaN drops counted (arm-E lock); ACHIEVED max_bce reported per regime.
5. **Far-anchor consistency:** arm-D r_cross sens must land at arm-C/D-scale significance
   (normalizer gate, §D3); LoRA arm-A far end already reproduces arm-C/D-scale numbers
   (parent-plan S1-dial consistency).
6. **{K, 2K} adequacy** on one rung per regime (p0_noise), per §II rule 4 — checked on the
   REPORTED d\* ratio as well as on raw sens (S6a).
7. **P4 power pre-registration (S1):** before arm E is read, compute the minimum detectable
   s-difference at K=50 from the sign-flip null spread on arm-E data; the design must have the
   power to catch a **2×** s-difference — a P4 "pass" below that power is INCONCLUSIVE, not a
   pass, and the cross-regime comparison stays conditional.

## 5. Pre-declared plots

1. **HEADLINE — overlaid normalized profiles:** s(d_pixel) on the shared 6 rungs, log-y, one
   curve per arm {A r8, C full-single, D full-all}, threshold 0.1 line + d\* crossings marked.
   Expected shape: curves shift LEFT monotonically A→C→D (a deferred arm-B curve slots between
   A and C if it ever runs).
2. **Per-layer fan (arm D):** s(d) for {L0, L1, L2, concat}. Expected: L0 leftmost.
3. **Calibration panel:** raw sens vs ε (log-log, slope −2 guide) — the 3-point bracket on
   p0_noise, 2-point on the others; plus arm-E overlay (B0-noise vs ε-noise profiles) — the P4
   evidence in one figure.
4. **Gate panel (§4.0):** B1 bars (d\* all-layers vs random coordinate-half) + B2 bars (full-FT
   s/d\* under ε-noise vs SGD-order noise). (The former arm-H reconstruction scatter is REMOVED
   with P3/H, TB2.)
5. **Arm F — per-layer LOO-footprint fan, full vs LoRA:** grouped bars (or paired fan) of the
   normalized LOO footprint per {L0, L1, L2, concat} for the full regime beside the LoRA (single-
   block) value, one panel per target + a pooled panel; expected: L0 dominates in full, and
   full > LoRA on the normalized footprint. Second panel: cross-regime per-target rank scatter.
6. **Arm G — overlaid singular spectra of J_full vs J_LoRA** (same tangent basis, log-y, index and
   SNR-whitened variants) + an **r_J bar pair** (full vs LoRA, with the LoRA rank-saturation line
   from leakage_story_consolidated overlaid) + per-layer col(J_full) energy stack.
7. Ladder image grids per arm (house rule: visual examples, best AND worst, saved .pth tensors).

## 6. Compute & sequencing (WEXAC long-gpu; python -u; stage-0 gates)

Per trimmed dial arm (C, D): K=50 baseline + 6 rungs × 50 seeds × 2 targets ≈ 650 trainings ×
T=1000 full-batch steps on a 784-1000-1000-1 MLP (full-FT backprop cost ≈ LoRA's; only the update
is bigger) ≈ well under the job-268959 wall-clock. Arm E: 4 rungs × 50 × 2 targets + its own
ε-noise baseline ≈ 450 trainings. Arm F: one contrast × 6–8 targets × K=50 × 2 regimes ≈ 700–800
trainings, minus baselines shared with A/D. Arm G: Nk=32 J columns × 2 maps at N=4/k=8/T=5 — the
unroll is tiny; the all-layer stretch is memory-, not time-bound; ≈ hours (the LoRA half may
reuse existing j1 runs), and the mandatory T-sweep T={1,5,20} triples a small cost — still hours.
ε-calibration + the bracket (3-point on p0_noise, 2-point elsewhere) ≈ 350 trainings. Blockers:
**B1 ≈ 350 trainings budgeted** (largely a CPU rescore of arm-D's saved stacks; the budget covers
fresh seeds if independence demands) and **B2 ≈ 200 trainings**. Per-layer rescoring and the F
g₀-piggyback are CPU-only on saved stacks/existing data. **Total ≈ 3.3k trainings + G — still
about one GPU-day for the whole wave**, split into 3–4 parallel bsub jobs of a few hours each.
Deferred arm H (1–2 GPU-days) is out of this wave by construction. Order: (1) stage-0 smoke all
arms; (2) ε-calibration + bracket + metric-CI gates + the arm-A rung-bracket pre-check (CPU, on
job-268959 data); (3) arms C/D/E + B2 parallel, + G (independent of the dial); (4) arm F (reuses
arm-A/D baselines) + the g₀ piggyback; (5) CPU rescores incl. B1 + figures; (6) **headline read
ONLY after B1 + B2 pass**; (7) deferred-item decisions (B, H). STATUS.md + LESSONS_LEARNED.md
updated same-turn per house rule.

## 7. Audit resolutions & remaining open items

**Resolved / retired by the two audits (all encoded above):**
- **Q2 (lr matching) → S5:** fixed-memorization-criterion matching, tightened to the
  max_bce ∈ [1e-4, 1e-3] band; achieved values reported per regime.
- **Q3 (normalization validity) → §4.0-B1:** the s(d) normalization EARNS the cross-regime claim
  only if B1 passes; otherwise the §II rule-3 dimension correction applies first.
- **Q4 (x-axis):** d_pixel ENDORSED as the cross-regime axis; d_encoder reserved for semantic
  statements.
- **Q5 (gauge):** headline = attacker-visible Δθ (incl. ε·ξ); path-noise-only as diagnostic; the
  TF7 conditionality is stated in D2 and applies to every cross-regime readout.
- **Q6:** MOOT — retired with deferred arm H; no threshold earns the "reconstruction proven"
  label (see §2.3).
- **Q7 (arm F construction) → S3:** keep LOO; matched-N replace-by-duplicate rejected; P5b is the
  headline, P5a/P5d downgraded.
- **Q8 (arm G T-mismatch) → S4/TB4:** mandatory T-sweep T={1,5,20} + the early-training caveat;
  P6 T=5-conditional and demoted; P7 leads the arm.

**Q1 — RESOLVED (metric re-check, yoado-6d PASS):** the conditional structure (P4 + §4.1.7 power
gate + §4.0-B2 + TF7) IS sufficient license; SGD-order noise is NOT demanded as co-primary — and
deliberately so: making the full-FT arm SGD would change parameterization AND algorithm at once,
breaking the single-variable design that is the plan's core strength. B2's role is exactly the
scoped adjudication: if full-FT d\*/s flips ε↔SGD the headline dies; if it agrees, the noise
source is not driving the result. Strengthenings folded: B1 widened to the multi-fraction span
{~25k, ~100k, ~450k, 1.8M} (a 2× null cannot license the 70× headline span) and B2 gains the
mid/crossing rung (endpoints can agree while the d\*-determining crossing differs). SCOPE
STATEMENT (explicit): the claim is the FULL-BATCH full-FT valley under a calibrated init-noise
analogue, B2-confirmed not a synthetic-noise artifact at the d\*-determining rungs; the
SGD-regime (ecological) full-FT valley is a separate question, out of scope / future work.
