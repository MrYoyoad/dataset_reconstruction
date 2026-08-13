# Thesis Experiment Plan & TODO — single source of actionable work

> **⚠ SUPERSEDED FOR STATUS (2026-08-13).** This 2026-05-14 plan pre-dates the work that landed
> Jul–Aug 2026. The current actionable to-do is **[next_experiment_plan.md](next_experiment_plan.md)**
> (the coupled activation×anchor×linearization study), and demonstrated results live in
> [../STATUS.md](../STATUS.md). Reality vs the checkboxes below: **Addition 3 (anchor α-sweep) ✅,
> DI-Phase 0 ✅, GB-Phase 1 ✅ (0.951 cosine)** are all demonstrated — the Part A/B/C boxes here are
> stale. This file is kept for its rationale/taxonomy; do not treat its ☐/⏳ marks as current.

**Last updated:** 2026-05-14 (after first supervision meeting with Gal Vardi + follow-up planning)
**Rationale / canonical record:** [thesis_update_briefing.md](thesis_update_briefing.md) (the briefing — read it for full context)
**Direction rationale & attack taxonomy:** [unified_direction_analysis.md](unified_direction_analysis.md) → "Direct Weight Inversion — New Primary Axis"
**Demonstrated results & sprint log:** [../STATUS.md](../STATUS.md)

> This is the **one place** for the actionable to-do. The other planning docs
> ([GRADIENT_BRIDGE_PLAN.md](GRADIENT_BRIDGE_PLAN.md) syllabus + decoder background,
> [STATUS.md](../STATUS.md) Sprint sections, [reconstruction_approaches.tex](reconstruction_approaches.tex)
> approach catalog) now point here for what to *do next*; they retain their background material.

---

## Status & honesty legend

- **☐ proposed** (planned, not started) · **⏳ in progress** · **✅ demonstrated**
- Honesty conventions (keep everywhere): "demonstrated" vs "proposed"; "best single run" vs
  "multi-seed mean"; "empirically consistent with" vs "proven".
- The LoRA-**adapter** attack (taxonomy **row 3**) is **proposed, not done.** Direct weight inversion
  is **proposed** — only an Approach-G / S3.4 sketch exists (see Precursor note below). The
  demonstrated results so far are **full-gradient** (row 1) and **ΔW** (row 2), not LoRA-adapter.
- **Do not invent numbers.** Anything the briefing does not fix is an explicit `TODO`.

## Phase-naming key (resolves the overloaded "Phase 0")

| Label used in docs | Meaning | Status |
|---|---|---|
| **ViT-gate Phase 0** | The *completed* ViT full-gradient inversion gate (D1–D7, faces, N=3) — STATUS.md | ✅ demonstrated |
| **DI-Phase 0…3** | Direct-weight-inversion track (briefing §5) — this file, Part B | ☐ proposed |
| **GB-Phase 0…2** | Gradient Bridge decoder roadmap (GRADIENT_BRIDGE_PLAN.md) — this file, Part C | ☐ proposed |

**Precursor note.** Direct weight inversion is *not* from scratch: it generalizes **Approach G
"Differentiable Unrolling"** ([reconstruction_approaches.tex](reconstruction_approaches.tex), §G) and
**S3.4** ([../STATUS.md](../STATUS.md) Sprint 3) — same `‖θ_T − F(θ₀,x̂)‖²` outer loss, reduces to
the single-step NTK result (Experiment B) at T=1. The briefing generalizes it to any `F` (incl.
LoRA / Adam / schedulers), adds the two attacker-knowledge regimes, the NTK identifiability theory,
and the anchor tradeoff.

## Where each track stands today (demonstrated — see STATUS.md / briefing §2 for numbers)

- **Row 1 (full gradient):** ViT-gate Phase 0 crossed on Flowers102 + faces; N=3 same-person joint. ✅
- **Row 2 (ΔW):** NTK reconstruction on MNIST (full FT + LoRA r=8, multi-step survival). ✅
- **Row 3 (LoRA adapter):** ☐ proposed — the thesis target. Direct inversion + Gradient Bridge attack it.
- **Row 4 (composed W₀+BA, KKT extraction):** ✗ closed — structural negative result (Experiment A).

---

## Part A — The three meeting additions (briefing §4)

### Addition 1 — More LoRA inversion examples, varied data ☐

Broaden the LoRA / ΔW inversion beyond the current N=2 MNIST result — show the attack isn't
sample-specific, and supply the varied data the direct-inversion experiments need.

- [ ] **More within MNIST:** different digit classes, more samples/run, harder/ambiguous digits,
      varied stroke styles. Move beyond the two cherry-able samples.
- [ ] **Other datasets** (if MNIST exhausts cleanly), in order of complexity: Fashion-MNIST (same
      resolution, harder structure) → CIFAR-10/100 (color, real images) → possibly SVHN or a small
      face set.
- [ ] **Report distributions** (multi-seed where possible), not single best runs.
- `TODO`: exact digit sets, samples per run, number of seeds, which datasets make the final cut.

*Why:* hardens the row-2 claim and feeds SSIM-vs-N and SSIM-vs-data-complexity for direct inversion.

### Addition 2 — Smooth activations (GELU is top priority) ☐

Rerun the NTK / inversion experiments with genuinely smooth activations. LeakyReLU is only
piecewise-linear (C⁰, kinked at 0) — not smooth in the sense NTK theory wants. **GELU is C^∞ and is
what real ViTs use**, so it is both the theoretically correct and the deployment-realistic test.

Activation family, ordered by smoothness × deployment relevance:

| Activation | Smoothness | Note |
|---|---|---|
| **GELU** | C^∞ | real ViTs/BERT/GPT — **top priority** |
| SiLU / Swish | C^∞ | EfficientNet, some ViT variants |
| Softplus | C^∞ | canonical smooth ReLU; sharpness knob interpolates ReLU↔smooth (cleanest for theory) |
| Mish | C^∞ | some detection models |
| ELU | C¹ at 0 | intermediate |
| tanh / sigmoid | C^∞ | classical; smoothest-baseline endpoints |
| LeakyReLU / ReLU | C⁰ / kinked | kinked baselines (ReLU already shown to break at T≥50) |

- **Predicted (testable) result:** NTK survival (feature stability across T) and reconstruction SSIM
  **improve monotonically with activation smoothness**. LeakyReLU is the worst of the "good" cases;
  GELU/SiLU/Softplus survive longer T and reconstruct better; ReLU is the worst case.
- [ ] Rerun **feature-stability-vs-T** (slide-10 experiment) per activation.
- [ ] Rerun **linearization-error-vs-SSIM** (slide-12 experiment) per activation.
- [ ] Headline figure: feature-stability-vs-T and SSIM-vs-T, **one line per activation**, ordered by smoothness.
- [ ] (Optional, theory-clean) Sweep the Softplus sharpness from sharp (≈ReLU) to soft for a continuous
      smoothness→survival knob.
- `TODO`: T grid, seed count, Softplus sharpness values.

### Addition 3 — Anchor at the average of θ₀ and θ_T (α-sweep) + two-curve validation ☐

The NTK inversion linearizes around an anchor `θ_anchor` and uses `∇Φ(θ_anchor; x)`. Currently
`θ_anchor = θ₀`. Gal suggests the midpoint, generalized to a family:

```
θ_anchor(α) = (1−α)·θ₀ + α·θ_T,   α ∈ [0, 1]
α=0 → θ₀ (current) · α=0.5 → midpoint (Gal's suggestion) · α=1 → θ_T (final weights)
```

**The key tradeoff (the crux — get it right).** Two opposing forces as α→1:
1. **Linearization error decreases** (anchor closer to the trajectory ⇒ more accurate Taylor expansion).
2. **Identifiability of x_i degrades** — `θ_T` was *produced by training on {x_i}*, so as α→1 the anchor
   absorbs the training signal and `∇Φ(θ_anchor; x_i)` becomes contaminated by x_i. (This is exactly why
   we do **not** anchor at θ_T directly.)
   The bet: an interior optimum α* (plausibly near the midpoint) balances the two.

**Validation protocol (the deliverable — answers Gal's likely first question before he asks):**
measure two separate curves vs α —
- [ ] **Linearization error(α):** `‖Φ(θ_anchor+δ; x) − [Φ(θ_anchor; x) + ∇Φ(θ_anchor; x)·δ]‖` on the
      known displacement `δ = θ_T − θ_anchor`. Pure approximation-quality; does **not** reconstruct x_i.
- [ ] **Reconstruction SSIM(α):** how well the actual training images are recovered.
- [ ] **Plot both on one x-axis.** Legitimate win = lin-error bottoms out and SSIM peaks at ≈ the same α
      (improvement *explained by* better linearization). **Red flag** = SSIM keeps climbing past where
      lin-error bottoms out (the gain is the anchor leaking x_i, not better approximation).
- [ ] Sweep α ∈ {0, 0.25, 0.5, 0.75, ~0.9}, **cap α below the identifiability-degrading regime**, and
      report the α=0.5-vs-α=0 linearization-error reduction explicitly.
- `TODO`: where exactly to cap α (set empirically once the red-flag onset is visible).

**Richer anchor options (possibilities, not first experiments):** trajectory-time average
`(1/T)Σ_t θ_t` (Regime B only); curvature-weighted anchor (flattest-Hessian region);
two-sided / split linearization (θ₀ early, θ_T late).

---

## Part B — Direct-weight-inversion sequence (briefing §5) · DI-Phase 0…3

Treat fine-tuning as a deterministic differentiable map `θ_T = F(θ₀, {x_i})` and recover `{x_i}` via
`{x̂_i*} = argmin ‖θ_T − F(θ₀, {x̂_i})‖²`. See [unified_direction_analysis.md](unified_direction_analysis.md)
"Direct Weight Inversion" for the formulation, regimes, and concerns.

### DI-Phase 0 — direct-inversion toy + Addition 3 baseline (immediate, ~1 wk) ☐

- [ ] MNIST MLP, **N=4**, LoRA **r=8**, **GELU** (use the realistic smooth activation from the start).
- [ ] `F` = SGD unrolled for **T ∈ {1, 2, 5, 10, 20}**, full batch, known recipe, **Regime A** (endpoint matching).
- [ ] x̂_i init from noise; **Adam** on x̂_i.
- [ ] Plot **SSIM vs T**.
- [ ] **Confirm T=1 reproduces the existing single-gradient NTK baseline** (Experiment B sanity).
- [ ] Anchor at **α=0 and α=0.5**; report linearization-error and SSIM for both (Addition 3 first cut).
- **Deliverable:** SSIM-vs-T curve + the α∈{0,0.5} first-cut numbers.

### DI-Phase 1 — breadth + anchor sweep + activations (~2 wk) ☐

- [ ] **Addition 1:** more MNIST classes/samples; begin Fashion-MNIST. **SSIM-vs-N curve.**
- [ ] **Addition 3:** full α-sweep {0, 0.25, 0.5, 0.75, ~0.9}; the **two-curve validation plot** (lin-error vs α, SSIM vs α).
- [ ] **Addition 2:** activation sweep (GELU, SiLU, Softplus, LeakyReLU, ReLU) on feature-stability-vs-T and SSIM-vs-T.
- [ ] **Identifiability sanity:** do multi-restart inversions converge to the same x̂_i?
- [ ] **LoRA-rank ablation.**
- [ ] **Regime B** (per-step matching) as a faster sanity check if checkpoints are available.

### DI-Phase 2 — harder data + stress tests (after first clean figures) ☐

- [ ] CIFAR-10/100 (Addition 1, harder data).
- [ ] **Recipe-mismatch stress tests** (wrong η, wrong T) — frame the clean version as a best-case leakage upper bound.
- [ ] **Different F** (Adam, scheduler) — test recipe-generality of the attack.
- [ ] Scale to **ViT-B/16** with checkpointed backprop through F.

### DI-Phase 3 — theory, in parallel ☐

- [ ] NTK closed-form for F analytically; identifiability conditions in terms of **feature-gradient rank**
      and **(I−ηH) eigenstructure**.
- [ ] Formalize the **anchor tradeoff** (linearization error vs anchor-contamination as a function of α);
      look for an analytically derivable **α\*** that minimizes total error subject to an identifiability constraint.
- [ ] Match theory predictions against the empirical **SSIM(T)** and **SSIM(α)** curves.

**Two open theory questions this track exists to answer (the "Gal-shaped" core):**
- **Q-A (stability / identifiability):** for a fixed inversion algorithm/prior, when is the
  reconstruction map `R: g → x̂` well-posed?
- **Q-B (pretraining / finetuning overlap):** when θ₀ already (partially) fits x_i, does inversion
  recover x_i itself or only the residual-error directions? (Feature-map injectivity of `φ(x)=∇Φ(θ₀;x)`.)

---

## Part C — Gradient Bridge track (preserved, complementary axis) · GB-Phase 0…2 ☐

**Not replaced by direct inversion — complementary.** Direct inversion = leakage *upper bound* under
best-case (known-recipe) knowledge; the Gradient Bridge = how leakage *degrades* under realistic,
weaker assumptions (only the published adapter, no recipe). Full roadmap + reading syllabus:
[GRADIENT_BRIDGE_PLAN.md](GRADIENT_BRIDGE_PLAN.md); slide-ready concept: [r2f_bridge_concept.md](r2f_bridge_concept.md).

- [ ] **GB-Phase 0 — perfect-signal scaffold:** reconstruct from an exact ("cheating") full gradient to
      validate the inversion engine. *(Substantially covered by the completed ViT-gate Phase 0.)*
- [ ] **GB-Phase 1 — the bridge:** train the R2F-style gradient decoder `f_φ: (A,B) → ∇_W L` on a public
      proxy dataset (~50k single-step LoRA updates, per-layer MLP, cosine-similarity loss). Milestone: >0.9
      cosine sim on held-out proxy.
- [ ] **GB-Phase 2 — end-to-end attack:** freeze decoder → victim adapter → approximate gradient →
      gradient inversion → reconstructed image. Enhancement: SDS / diffusion prior to clean decoder noise.

---

## Part D — Literature search (briefing §6) · a DI-Phase 0 task ☐

- [ ] Thorough search before claiming novelty for vision few-shot LoRA + direct deterministic weight
      matching `θ_T=F(θ₀,x_i)` + NTK identifiability theory.
- [ ] Keywords: "differentiable training" + reconstruction · "unrolled SGD" + data reconstruction ·
      "training trajectory" + privacy · "implicit bias" + fine-tuning + reconstruction.
- [ ] Check **Geiping's** recent work (most active author in gradient-inversion space).
- **⚠ COLLISION FOUND (2026-06-29): SimuDy, Tian et al., ICLR 2025** (`openreview ZJftXKy12x`) **does
  the full-FT version of our det-`F` matching** — unroll SGD through dummy data, match `θ_f−θ₀` by
  cosine sim + TV. This **takes the "direct weight inversion of full fine-tuning" headline novelty.**
  It does **not** touch PEFT/LoRA, weaker-knowledge regimes, or any identifiability theory, and it is
  the expensive full-unroll (22 GB / 15 h for 120 CIFAR-32² imgs; ViT only N=10). **Reframe** (don't
  abandon): cite as closest prior + feasibility de-risker + baseline; re-center novelty on LoRA-only
  leakage / Gradient Bridge + weak-knowledge regimes + identifiability/anchor-α theory + the
  NTK-linearized *efficient* counterpart. Full analysis: [related_work_simudy.md](related_work_simudy.md).
- Other nearest (none does exactly this): ReCIT (text), TLDR/Network Inversion (vision, not det-F),
  DSiRe (recovers dataset *size*), Spectral DeTuning (recovers θ₀, opposite direction), R2F (text unlearning).
- **Deliverable:** a literature paragraph for the next meeting — SimuDy is now the lead citation.
- **Decision brief + gated plan (B1→B5):** [simudy_decision_brief.md](simudy_decision_brief.md) —
  reconstruction-chain teardown of what SimuDy proves/misses, feasibility, paper-worthiness, and the
  fail-fast gates (B1 adapter-only recovery, B2 linearized-vs-unroll) that gate the whole direction.

---

## Deliverable for the next supervision meeting (briefing §7)

A clean **SSIM-vs-T** curve (direct inversion) + the **anchor two-curve plot** (Addition 3) + at least
**GELU** results (Addition 2) + **broader MNIST** examples (Addition 1) + a **literature paragraph** (Part D).

> Framing to preserve: "enough independent work to justify a thesis *proposal*" — working pipeline, a
> clear failed path (KKT, row 4), a plausible PEFT bridge, and two theory questions where Gal's guidance
> matters. The thesis frame is **proposal, not finished work.**
