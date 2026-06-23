# Thesis Project Briefing & Update — Gradient Inversion Attacks on PEFT/LoRA

**Author:** Yoad Oxman (MSc thesis, Weizmann Institute, advisor Gal Vardi)
**Last updated:** 2026-05-14, after first supervision meeting + follow-up planning
**Purpose of this doc:** Single source of truth for the project. Hand this to any assistant (VS Code Claude, chat Claude, etc.) at the start of a session so it has full context. Update it as the project evolves.

---

## 0. Instructions for the assistant reading this (VS Code update task)

If you are being asked to **update project files** with the meeting details:

1. Read this whole document first. It is the canonical record.
2. The **three new meeting additions** are in Section 4. These are concrete experimental tasks Gal requested. They should be reflected in: the experiment plan / TODO file, any research-direction notes, and (if present) the thesis-proposal draft.
3. The **direct weight inversion** idea (Section 3) is the new primary research direction. If there is an existing notes/proposal file that only mentions the "Gradient Bridge," add direct weight inversion as a complementary second attack axis — do not delete the Gradient Bridge material.
4. Preserve all existing experimental results (Section 2). Do not overwrite or contradict them.
5. Keep the voice: direct, technical, allergic to overclaiming. Distinguish "demonstrated" from "proposed," "best single run" from "multi-seed mean," "empirically consistent with" from "proven."

---

## 1. Project overview

The thesis studies **privacy attacks on parameter-efficient fine-tuning (PEFT), specifically LoRA**: can an attacker reconstruct the private fine-tuning images from artifacts a victim publishes (the LoRA adapter, or the final weights)?

### The attack taxonomy (the orienting frame)

Four possible attacker objects, each a different inverse problem with different identifiability:

| # | Object attacker has | Evidence in deck | Status |
|---|--------------------|-----------------|--------|
| 1 | Full gradient ∇L | ViT inversion on Flowers + faces | ✓ done |
| 2 | Fine-tuning update ΔW (full FT or LoRA-induced) | NTK reconstruction on MNIST | ✓ done |
| 3 | LoRA adapter (A, B) — the actual PEFT artifact | Proposed attacks (see §3) | → proposed |
| 4 | Composed weights W₀ + BA | Haim-style KKT extraction | ✗ failed structurally |

- **The thesis target is row 3** (and the new direct-inversion idea attacks row 2/3 under strong assumptions).
- **Row 4 fails structurally:** two-image rank-2 BA cannot interpolate weights satisfying KKT over hundreds (~110 in the run shown, hundreds typical) of pre-train + fine-tune support points. This is a clean negative result, not a tuning failure.

### Two open theoretical questions (the thesis "Gal-shaped" core)

- **Q-A (stability/identifiability):** When is the inversion map R: g → x̂ stable under perturbations of g? Better framed as: for a fixed inversion algorithm/prior, when is the reconstruction map well-posed? Identifiability + stability.
- **Q-B (pretraining/finetuning overlap):** When θ₀ already (partially) fits xᵢ, does inversion recover xᵢ itself, or only the directions where θ₀ still has residual error? Feature-map injectivity of φ(x) = ∇Φ(θ₀; x).

---

## 2. What's already established (existing empirical results)

These are demonstrated and should not be contradicted by new work.

**NTK reconstruction on MNIST (row 2 / ΔW attack):**
- MNIST MLP, N=2 samples, one SGD step, LoRA r=8 (38K params, 47× compression vs full).
- Full FT recovers near-perfectly: SSIM 0.990 / 0.998.
- LoRA r=8: SSIM 0.686 / 0.496 — well above the 0.30 recognizability gate.
- Setup: known labels, free-c inversion.

**Instance-level leakage (not class memorization):**
- LoRA reconstruction closer to its own ground-truth than to any same-class digit.
- Δ = +0.220 (S1), +0.182 (S2) over mean of n=20 random same-class controls.
- Framed as: evidence for instance-level leakage, not only class-level memorization. (Sanity control, n=20 — not a full statistical study.)

**Multi-step NTK survival:**
- Feature stability cos(∇Φ(θ₀), ∇Φ(θ_T)) ≈ 0.75, flat through T=100 for LeakyReLU / ModifiedReLU.
- ReLU breaks at T ≥ 50 due to gradient kinks.
- Pixel-level reconstruction at T=10: Sample 1 SSIM 0.73, Sample 2 0.75. Full FT decays 1.00→0.80 across T=1→100; LoRA r=8 nearly flat 0.80→0.77.

**NTK assumption quantification (slide 12):**
- Linearization error vs reconstruction SSIM across 3 activations. LeakyReLU stays SSIM≈0.8 at T=100; ReLU collapses to ~0.5 at T=50.
- Width dependence: Jacobian Lipschitz **empirically consistent with** O(1/√width). (Not proven.)

**Batches/epochs math (slide 13):**
- ΔW ≈ −η Σᵢ c′ᵢ ∇θ Φ(θ₀; xᵢ), where c′ᵢ = Σ_t 𝟙[i∈ℬ_t] · cᵢ(t).
- Under stable residuals collapses to c′ᵢ ≈ nᵢ · cᵢ.
- Framed as: in the NTK / small-motion approximation, stochastic fine-tuning reduces to the same inverse problem with sample weights.

**ViT scaling (row 1 / full-gradient attack):**
- ViT-B/16 (86M params), Flowers102. Gate ≈ 0.30 SSIM; best run 0.55, median across D2 sweep 0.35–0.45.
- Final target-gradient match cos_sim ≈ 0.95 across configs. **Input: full ViT gradient**, not LoRA adapter.

**Face recovery (row 1):**
- Three people, single ViT gradient each. **Best single run SSIM 0.52 / 0.67 / 0.58.**
- face1 has 5-seed validation: mean 0.42 ± 0.03. face2/face3 are D3-winner config transferred, **not yet multi-seed**.
- Input: full ViT gradient.

**Multi-image (N=3):**
- Joint recovery of 3 faces; per-image SSIM 0.60 / 0.67 / 0.71 (mean 0.66).
- **But cross-matrix shows partial centroid attraction:** best-match assignment [1,1,2], not [0,1,2]. High SSIM ≠ clean identity separation. Honest superposition territory.

**Key methodological finding (Q-A motivation):**
- In the saturated cos_sim regime (0.94–0.96), SSIM varies 0.18–0.55. ρ(cos_sim, SSIM) = +0.43 — weakly predictive within the saturated regime. TV penalty and total loss are stronger predictors. **The prior is the lever, not gradient match.**

**Conventions:**
- SSIM via `kornia.metrics.ssim`, window_size=3.
- 0.30 SSIM = operational recognizability gate (visually calibrated for this deck, NOT a universal threshold).
- Compute: WEXAC GPU cluster.

---

## 3. The new primary direction: direct weight inversion

This is Gal's idea from the first meeting. It is the new main research thrust, complementary to (not replacing) the earlier Gradient Bridge proposal.

### Core formulation (general, F as a black box)

Whatever the fine-tuning procedure — SGD, Adam, schedulers, anything — the final weights are a **deterministic function** of the base weights and the training samples:

```
θ_T = F(θ₀, x₁, ..., x_N)
```

In LoRA the attacker has θ₀ (public base) and (A, B) (published adapter), so θ_T = θ₀ + BA is fully reconstructible. The only unknowns are the training samples {x_i}.

**The attack:**

```
{x̂_i*} = argmin_{x̂_i}  ‖θ_T^observed − F(θ₀, {x̂_i})‖²
```

F is treated as a known, deterministic, differentiable map. In PyTorch, F is whatever fine-tuning loop you'd normally run, with x̂_i marked as a learnable `nn.Parameter`; autograd backprops through F to the candidate data. **Crucially, this is NOT specific to trajectory unrolling** — trajectory unroll (SGD with known η, T, full batch) is just one concrete F. Adam, schedulers, etc. all fit the same formulation.

### Two attacker-knowledge regimes

- **Regime A — endpoint only (realistic):** attacker has only θ_T. Loss = ‖F(θ₀, x̂_i) − θ_T‖². This is the real LoRA threat model — HuggingFace publishes the adapter, not intermediate checkpoints.
- **Regime B — full trajectory (best-case):** attacker has all intermediate θ_t. Per-step matching loss, shorter backprop chains, easier — but stronger assumption.

### Why few-shot LoRA is the sweet spot

Small N (5–50 unknowns), small T (10–50 steps, tractable unroll), full-batch realistic at small N, recipe metadata often on the model card, θ₀ public by construction. Maps onto real HuggingFace usage.

### Theoretical hook (NTK regime)

When F is SGD with small η, F has closed-form structure:

```
θ_T − θ₀ ≈ −η Σ_{s<T} (I − ηH)^(T−s−1) · Σ_i c_i(s) · ∇Φ(θ_anchor; x_i)
```

(H = NTK Hessian / feature-gradient Gram matrix; c_i(s) = per-sample residual at step s.) Inverting F in the NTK regime is then a **structured linear inverse problem**, with identifiability conditions writable analytically in terms of feature-gradient rank and the eigenstructure of (I − ηH). This is where Gal's theory expertise plugs in.

### Concerns to manage

- **Memory** (only when F is an unroll): gradient checkpointing, truncated BPTT, implicit gradients, or Regime B. Start small (MNIST T=5–10).
- **Non-convexity / non-uniqueness:** multiple training sets may produce the same θ_T. Identifiability ↔ feature-gradient rank in NTK regime; check empirically with multi-restart elsewhere. Characterizing the non-uniqueness map is itself a contribution.
- **Known-recipe assumption is strong:** frame the clean version as a best-case leakage upper bound; stress-test η/T/recipe mismatch as separate experiments.
- **Stochasticity:** few-shot is often full-batch; if not, known seed restores determinism, else marginalize/optimize over batch order.
- **θ₀ must be exactly right:** errors compound through F. Same-checkpoint cleanest; checkpoint-drift is a stress test; cross-architecture probably breaks.

### Relationship to the Gradient Bridge

- **Gradient Bridge:** decoder f_φ: (A,B) → ∇W L trained on proxy data, output fed to single-gradient inversion. Attacks row 3, weaker assumptions, more general, proxy-generalization risk. (Inspired by R2F — Liu et al., arXiv 2512.07374, Dec 2025 — which is **text-only LLM unlearning**; the vision attack + threat-model reversal is the novel part. Direct weight inversion does NOT depend on R2F.)
- **Direct weight inversion:** invert F directly, no decoder, no proxy. Attacks row 2/3 under known-recipe assumptions, stronger assumptions, more exact.
- **Combined thesis story:** direct inversion = leakage upper bound under best-case knowledge; Gradient Bridge = how leakage degrades under realistic (weaker) assumptions. Two complementary axes + theoretical core.

---

## 4. The three new meeting additions (action items)

These are Gal's specific requests from the planning discussion. Each becomes a concrete experiment.

### Addition 1 — More LoRA inversion examples, varied data

**Request:** broaden the LoRA inversion beyond the current N=2 MNIST digits.

**Concretely:**
- More examples **within MNIST**: different digit classes, more samples per run, harder/ambiguous digits, varied stroke styles. Move beyond the two cherry-able samples to show the attack isn't sample-specific.
- **Other datasets too** (if MNIST exhausts cleanly): Fashion-MNIST (same resolution, harder structure), then CIFAR-10/100 (color, real images, bigger jump), possibly SVHN or a small face set. The point is to show the LoRA inversion generalizes across data complexity, not just on toy digits.
- Report distributions (multi-seed where possible), not single best runs — consistent with the honesty conventions.

**Why it matters:** the current LoRA result is N=2 on MNIST. To make the row-2 claim robust, it needs more breadth. This is a credibility-hardening task, and it feeds the direct-inversion experiments (which also need varied data to characterize SSIM-vs-N and SSIM-vs-data-complexity).

### Addition 2 — Test genuinely smooth activations (GELU and others)

**Request:** rerun the NTK / inversion experiments with **GELU** and other genuinely smooth activations. LeakyReLU is only piecewise-linear (C⁰, kinked at 0) — it is NOT smooth in the sense the NTK theory wants. GELU is C^∞ smooth **and is what real ViTs actually use**, so it's both the theoretically correct test and the deployment-realistic one.

**Activation family to test, ordered by smoothness × deployment relevance:**
- **GELU** — C^∞, used in real ViTs/BERT/GPT. The most important one to add. Top priority.
- **SiLU / Swish** (x·σ(x)) — C^∞, used in EfficientNet and some ViT variants. Real-world.
- **Softplus** (log(1+e^x)) — C^∞, the canonical smooth ReLU approximation. Cleanest for theory (it's literally the smooth limit of ReLU; you can sweep a temperature to interpolate ReLU↔smooth).
- **Mish** (x·tanh(softplus(x))) — C^∞, used in some detection models. Smooth, less common.
- **ELU** — smoother than LeakyReLU (C¹ at 0), exponential negative branch. Intermediate.
- **tanh / sigmoid** — classical, fully smooth, not used in modern vision but good as smoothest-baseline endpoints.

**Predicted result (testable):** NTK survival (feature stability across T) and reconstruction SSIM should **improve monotonically with activation smoothness**. LeakyReLU (kinked) should be the worst of the "good" cases; GELU/SiLU/Softplus should survive longer T and reconstruct better. ReLU (already shown to break at T≥50) is the worst case. If Softplus has a tunable sharpness, sweeping it from sharp (≈ReLU) to soft gives a clean continuous knob showing the smoothness→survival relationship directly.

**Experiments:** rerun the existing slide-10 (feature stability vs T) and slide-12 (linearization error vs SSIM) experiments for each activation. The headline figure: feature-stability-vs-T and SSIM-vs-T curves, one line per activation, ordered by smoothness.

### Addition 3 — Anchor the linearization at an average of θ₀ and θ_T (not just θ₀)

**Request:** in the reconstruction/inversion computation, instead of anchoring the NTK linearization at θ₀, try anchoring at the **average of θ₀ and θ_T** (the midpoint), and more generally explore a family of anchor points.

**The setup.** The NTK inversion linearizes the network around an anchor point θ_anchor and uses the feature map ∇Φ(θ_anchor; x). Currently θ_anchor = θ₀. Gal suggests θ_anchor = (θ₀ + θ_T)/2.

**The anchor family to sweep:**
```
θ_anchor(α) = (1−α)·θ₀ + α·θ_T,   α ∈ [0, 1]
```
- α = 0 → θ₀ (current)
- α = 0.5 → midpoint (Gal's suggestion)
- α = 1 → θ_T (the final weights)

**Richer options (mention as possibilities, not first experiments):**
- **Trajectory-time average:** θ_anchor = (1/T)Σ_t θ_t — mean over all intermediate weights (Regime B only; needs checkpoints). Weights the anchor toward where the trajectory actually spent time, which differs from the endpoint midpoint if motion is non-uniform.
- **Curvature-weighted anchor:** anchor where the Hessian is smallest (flattest region → best linearization). More expensive; later refinement.
- **Two-sided / split linearization:** expand around θ₀ early, θ_T late, stitch. More complex; future option.

**THE KEY TRADEOFF (this is the crux — get it right):**

There are two opposing forces as α increases from 0 to 1:

1. **Linearization error decreases** as α → 1. The anchor moves closer to the actual trajectory the weights took, so the Taylor expansion is more accurate across that trajectory. (A Taylor expansion around the midpoint of an interval is more accurate across the interval than one around an endpoint — same principle.) This pulls toward larger α.

2. **Identifiability/reconstructability of x_i degrades** as α → 1. The anchor θ_anchor(α) increasingly **depends on the very x_i being recovered**, because θ_T was *produced by* training on {x_i}. At α=0, the anchor is θ₀ — public, pre-training, completely x_i-independent — so the recovered x̂_i is a genuine inversion from a clean feature map. As α → 1, the anchor absorbs the training signal; the feature map ∇Φ(θ_anchor; x_i) is contaminated by x_i, and you can no longer cleanly attribute the recovered x̂_i to the inversion rather than to the anchor already "knowing" the answer. **This is exactly why we do NOT use θ_T directly in the inversion** — it would minimize linearization error but destroy identifiability.

**The bet:** there is an interior optimum α* (plausibly near the midpoint) that balances these — small enough linearization error for the NTK inversion to be accurate, anchor still clean enough that x_i remains genuinely identifiable. Sweeping α and finding where reconstruction is best *and still trustworthy* is the experiment.

**Validation protocol (do this — it answers Gal's likely first question before he asks):**

Measure two separate curves as functions of α:

1. **Linearization error(α):** ‖Φ(θ_anchor + δ; x) − [Φ(θ_anchor; x) + ∇Φ(θ_anchor; x)·δ]‖, measured on the known trajectory displacement δ = θ_T − θ_anchor (or per-step δ). This is a pure approximation-quality measurement; it does NOT involve reconstructing x_i.
2. **Reconstruction SSIM(α):** how well the actual training images are recovered.

Interpretation:
- If the midpoint helps for the **legitimate** reason, linearization error should bottom out and SSIM should peak at **roughly the same α**. The reconstruction improvement is *explained by* the linearization improvement. Clean, defensible story.
- If SSIM keeps climbing toward α=1 **past** where linearization error bottoms out, that's a **red flag**: the extra gain isn't from better approximation, it's from the anchor leaking x_i (the identifiability problem above). That signals you've pushed α too far and the "improvement" is an artifact, not a real attack gain.

So the deliverable for Addition 3 is: **a plot of linearization-error(α) and reconstruction-SSIM(α) on the same x-axis**, showing where they jointly optimize, with α capped below the regime where identifiability degrades. Compare midpoint (α=0.5) explicitly against the baseline (α=0) and report the linearization-error reduction.

---

## 5. Experimental sequence (consolidated)

**Phase 0 (immediate, ~1 week) — direct weight inversion toy + Addition 3 baseline:**
- MNIST MLP, N=4, LoRA r=8, **GELU** (per Addition 2 — use the realistic smooth activation from the start, not LeakyReLU).
- F = SGD unrolled for T ∈ {1, 2, 5, 10, 20}, full batch, known recipe, Regime A (endpoint matching).
- x̂_i init from noise; Adam on x̂_i.
- Plot **SSIM vs T**. Confirm T=1 recovers the existing single-gradient NTK baseline.
- Anchor at α=0 and α=0.5; report linearization-error and SSIM for both (Addition 3 first cut).

**Phase 1 (~2 weeks) — breadth + anchor sweep + activations:**
- **Addition 1:** more MNIST classes/samples; begin Fashion-MNIST. SSIM-vs-N curve.
- **Addition 3:** full α-sweep α ∈ {0, 0.25, 0.5, 0.75, ~0.9}; the two-curve validation plot (lin-error vs α, SSIM vs α).
- **Addition 2:** activation sweep (GELU, SiLU, Softplus, LeakyReLU, ReLU) on feature-stability-vs-T and SSIM-vs-T.
- Identifiability sanity: do multi-restart inversions converge to the same x̂_i?
- LoRA-rank ablation.
- Regime B (per-step matching) as a faster sanity check if checkpoints available.

**Phase 2 (after first clean figures):**
- CIFAR-10/100 (Addition 1, harder data).
- Recipe-mismatch stress tests (wrong η, wrong T).
- Different F (Adam, scheduler) — test recipe-generality of the attack.
- Scale to ViT-B/16 with checkpointed backprop through F.

**Phase 3 (theory, in parallel):**
- NTK closed-form for F analytically; identifiability conditions in terms of feature-gradient rank and (I−ηH) eigenstructure.
- Formalize the anchor tradeoff: linearization error vs anchor-contamination as a function of α. This is a genuinely theoretical question (Gal-shaped) — there may be an analytically derivable α* that minimizes total error subject to an identifiability constraint.
- Match theory predictions against the empirical SSIM(T) and SSIM(α) curves.

---

## 6. Literature position (brief; full search is a Phase 0 task)

Searched once; nothing found doing exactly this (vision few-shot LoRA, direct deterministic weight matching θ_T=F(θ₀,x_i), with NTK identifiability theory). Nearest:
- **ReCIT (arXiv 2504.20570, Apr 2025):** PEFT gradient attack, recovers PII from LoRA LLMs (Bloomz-3B). **Text only**, uses malicious pretraining + token filtering, not direct weight matching.
- **TLDR / Network Inversion (NeurIPS 2024, arXiv 2410.16884):** training-data reconstruction from CNN weights via learned generators + memorization. Vision, but not deterministic-F inversion.
- **DSiRe (HUJI 2024):** recovers dataset *size* from LoRA weights spectrally. Nearby; doesn't recover content.
- **Spectral DeTuning (Feb 2024):** recovers θ₀ from θ_T (opposite direction). Precedent for weight-space inverse problems.
- **R2F (arXiv 2512.07374, Dec 2025):** adapter→gradient decoder for unlearning. Text only. Inspired the Gradient Bridge, orthogonal to direct inversion.

Do a thorough search before claiming novelty. Keywords: "differentiable training" + reconstruction, "unrolled SGD" + data reconstruction, "training trajectory" + privacy, "implicit bias" + fine-tuning + reconstruction. Check Geiping's recent work (most active author in gradient-inversion space).

---

## 7. Meeting / supervision status & communication notes

- First supervision meeting with Gal Vardi went **well**. He agreed to supervise on a **thesis-proposal basis**, pending a follow-up meeting in a few weeks after Yoad prototypes the new experiments.
- The framing that worked: "enough independent work to justify a thesis *proposal* — working pipeline, a clear failed path (KKT), a plausible PEFT bridge, and two theory questions where his guidance matters" — NOT "please approve a finished thesis."
- This is an **MSc thesis**, not a PhD. The bar for "agree to supervise" is correspondingly lower; the work already over-delivers for the stage.
- Gal proposed the direct weight inversion idea (Section 3) and the three additions (Section 4). He thinks in terms of differentiable simulators / dynamical systems / NTK — the theory framing in Section 3 and the anchor-tradeoff in Addition 3 are deliberately pitched to his expertise.
- Deliverable for the next meeting: a clean **SSIM-vs-T** curve (direct inversion) + the **anchor two-curve plot** (Addition 3) + at least GELU results (Addition 2) + broader MNIST examples (Addition 1). Plus a literature paragraph.

**Voice / honesty conventions to preserve everywhere:**
- "demonstrated" vs "proposed"; "best single run" vs "multi-seed mean"; "empirically consistent with" vs "proven."
- Don't claim "reconstruct private images from LoRA adapters" — that's the proposed bridge, not done. Current results are full-gradient and ΔW.
- Don't say "no privacy leak" for low SSIM — say "not visually recognizable in our examples."
- KKT support points: "~110 in this run, hundreds typical," never "500."
- The thesis frame is "proposal, not finished work."
