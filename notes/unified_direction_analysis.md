# Unified Thesis Direction Analysis

**Report Date:** March 25, 2026
**Status:** Reconciling three contradictory thesis-direction analyses after Sprint 1-2 experiments

---

## Executive Summary

Three competing analyses recommended different thesis directions:
- **Run 1 (Thesis_Direction_Analysis.tex):** KKT-LoRA + phase transition formula r* — "direct approach, most testable"
- **Run 2 (Reconstruction_Approaches.tex + Sprint 2b results):** NTK (Experiment B) → scale to ViT — "empirically working"
- **Run 3 (GRADIENT_BRIDGE_PLAN.md + Inversion_Feasibility.tex):** Gradient Bridge (R2F decoder + gradient inversion) — "most sophisticated, highest risk"

**Root cause of disagreement:** Each analysis was written at different points in the experimental timeline and weighted evidence differently. Run 1 (pre-Sprint 1) was optimistic about KKT theory. Run 2 (post-Sprint 2b) was excited about empirical NTK success. Run 3 (earliest) outlined an ambitious pipeline without empirical validation.

**Unified recommendation:** NTK-LoRA is the primary track (empirically proven, low risk). KKT-LoRA as a parallel theoretical track. Gradient Bridge deferred until Phase 0 validates.

---

## 1. Evidence Inventory

### 1.1 Proven Results (with citations)

#### Experiment B: NTK Reconstruction (1-Step, Oracle Coefficients)
- Full model SSIM: 0.9999 (seed=42, T=1, oracle c) — STATUS.md:121
- LoRA r=8 SSIM: 0.797 (oracle) — STATUS.md:122
- LoRA r=32 SSIM: 0.826 (oracle) — STATUS.md:123
- Control images (same class): SSIM 0.582–0.693 — proves instance-specific leakage
- Multi-seed (200 seeds): 22/200 (11%) produce strong signal
- CSV: `results/experiment_b_sweep_20260222_115334.csv`

#### Experiment B: Free-Coefficient Extraction (Honest Attack)
- Full model, SGD + α=1: SSIM = 0.997 (matches oracle) — LESSONS_LEARNED.md:237
- LoRA r=8 free-c (ReLU, L-BFGS): SSIM = 0.617 (gap 0.08 from oracle)
- LoRA r=64 free-c: SSIM = 0.635 (gap 0.08)
- LoRA r=16, r=32: Stubbornly bad (~0.42, coeff error ~0.28) — optimization landscape bottleneck

#### Sprint 2b: Multi-Step NTK (Phases 0–2)
- Phase 0: LeakyReLU stable through T=100; ReLU NaN's at T≥50
- Phase 2: LoRA r=8, r=32 nearly match full model through T=100 (gap 0.01–0.03)
- CSV: `results/sprint2b_phase2_20260223_072927.csv`

#### Experiment A: KKT-LoRA (Compose + Reconstruct) — FAILED
- KKT loss ~460, NaN at epoch 7-8 — STATUS.md:110
- Root cause: composed W = W₀ + BA satisfies KKT over all ~502 samples; 2 images can't explain 502

### 1.2 Speculated Claims

| Claim | Source | Basis | Status |
|---|---|---|---|
| KKT-LoRA achieves 70-80% success | Thesis_Direction_Analysis.tex | Parameter-counting theory | SPECULATED — no experiments |
| NTK scales to ViT at 60-75% SSIM | Reconstruction_Approaches.tex | Extrapolation from MNIST | SPECULATED — no ViT data |
| Gradient decoder achieves >0.9 cosine sim on vision | GRADIENT_BRIDGE_PLAN.md | R2F proved it for LLMs | SPECULATED for vision |
| 0.9 cosine sim sufficient for pixel reconstruction | Inversion_Feasibility.tex | Uncertainty — doc says "enormous noise" | SPECULATED — calls for Phase 0 |
| r* = N×d_input/(d_in+d_out) is sharp | Thesis_Direction_Analysis.tex | Linear algebra | SPECULATED — never tested |
| Gradient Bridge works multi-step | GRADIENT_BRIDGE_PLAN.md | Assumption | SPECULATED — severe information bottleneck |

---

## 2. Why the Three Analyses Disagreed

| Document | Written | Based On | Recommendation |
|----------|---------|----------|---|
| Thesis_Direction_Analysis.tex | ~Jan 25 (pre-Sprint 1) | Literature + theory | KKT-LoRA + r* |
| Reconstruction_Approaches.tex | Mar 19 (post-Sprint 2) | Sprint 1-2 empirical results | NTK as baseline, Approach G |
| Inversion_Feasibility.tex | ~Feb 15 (pre-Phase 0) | Risk assessment | Phase 0 mandatory; 30-40% for Bridge |
| GRADIENT_BRIDGE_PLAN.md | ~Jan 20 (earliest) | R2F blueprint | Full pipeline |

**Key dynamics:**
1. Thesis_Direction_Analysis weighted theory elegance; didn't account for Experiment A's structural failure
2. Reconstruction_Approaches was pragmatic post-Sprint-1/2, correctly identified NTK as empirically winning
3. Inversion_Feasibility was risk-aware, called for Phase 0 validation (still pending)

---

## 3. Error Cascade: What the Documents Actually Say

From Inversion_Feasibility_Analysis.tex:

| Cosine Similarity | Error Fraction | Interpretation |
|---|---|---|
| 0.99 | 2% | Acceptable |
| 0.95 | 9.75% | Challenging |
| 0.90 | **19%** | Very challenging |
| 0.85 | 27.75% | Likely fails |

For ViT-B/16 query projection (589K dims): 0.90 cosine sim = ~112K effective dimensions of noise.

**Quote (Inversion_Feasibility.tex):** "Nobody in the gradient inversion literature has systematically studied what happens when gradients are approximate (as opposed to exact + DP noise). This is both a gap and an opportunity — but you are walking into unknown territory."

**Multi-step accumulation:** Single-step decoder vs T-step adapter. Information bottleneck is severe for batch>1.

---

## 4. NTK Scaling: Evidence Assessment

**On MNIST (proven):** Full model 0.9999, LoRA r=8 0.797, multi-step stable through T=100.

**On ViT:** NO EXPERIMENTAL DATA. The "0.6-0.75" claim has no basis beyond heuristic extrapolation.

**Concern:** 28×28 MNIST (784 pixels) → 224×224×3 (150K pixels) is a 190× increase in search space.

---

## 5. Approach G: Role and Priority

- Mathematically exact for any T (no linearization error)
- Reduces to NTK at T=1 (consistency check)
- Memory-intensive (T computation graphs)
- Requires knowing η, T exactly

**Verdict:** Secondary. Only implement if NTK fails at T>100 on ViT.

---

## 6. Phase Transition (r*): Feasibility

Formula: r* ≈ N × d_input / (d_in + d_out)

**Very testable** (sweep r × N, 1-2 weeks). But Sprint 2 results (r=16, r=32 stubbornly bad) suggest transition may not be sharp.

**Recommendation:** Treat r* as secondary theoretical contribution, not primary attack method.

---

## 7. Unified Recommendation

### 7.1 Primary Track: NTK-LoRA (70-80% confidence)
Empirically proven on MNIST. Scale to ViT. Phase transition heatmap for thesis core result.

### 7.2 Parallel Track: KKT-LoRA Theory (50-60% confidence)
Prove r* for linear case. Test on MLPs. Lower risk, adds theoretical depth.

### 7.3 What to Build This Week

**Must-Do (Gate-Clearing):**
1. `experiments/phase0_vit_gradient_inversion.py` — ViT + LoRA, 1 image, 1 step, exact gradient → inversion (3 days)
2. `experiments/phase0b_noise_tolerance.py` — Add noise to exact gradient, plot SSIM vs cosine sim (3 days)

**Should-Do (Continue Sprint 2):**
3. Complete Sprint 2c Track B (Phase 3-4 on WEXAC)
4. Update `run_experiment_b.py` for ViT support

---

## 8. Risk Matrix

| Direction | P(Success) | Primary Risk | Time to Results |
|---|---|---|---|
| NTK-LoRA (Primary) | 70-80% | ViT scaling unknown | 6-8 weeks |
| KKT-LoRA (Parallel) | 50-60% | Formula may not hold nonlinearly | 4-6 weeks |
| Gradient Bridge | 30-40% | Noise cascade; Phase 0 unknown | 10-12 weeks |
| Approach G | 40-50% | Memory/compute; incremental if NTK works | 2-3 weeks |

---

## Conclusion

The three analyses weren't wrong — they were written at different stages and weighted evidence differently. NTK-LoRA is the primary track because it's empirically proven. KKT-LoRA is a parallel theory track. Gradient Bridge requires Phase 0 clearance first.

**By running Phase 0 (ViT gradient inversion) this week, you'll know within days whether to invest in all three or focus on one.**

---

# Direct Weight Inversion — New Primary Axis (2026-05-14 meeting)

> **Added 2026-05-14**, after the first supervision meeting with Gal Vardi. This supersedes the
> March-25 prioritization above on one point: the ViT gradient-inversion gate has since been **crossed**
> (see [../STATUS.md](../STATUS.md)), and Gal proposed a new primary research axis — **direct weight
> inversion**. It is **complementary to, not a replacement for, the Gradient Bridge.** §5 above
> ("Approach G: Role and Priority") called the embryonic form of this idea *secondary*; the meeting
> **re-prioritized its generalized form to primary.** Status throughout this section: **proposed**
> (only an Approach-G / S3.4 sketch exists). Actionable to-do lives in
> [experiment_plan.md](experiment_plan.md); full rationale in [thesis_update_briefing.md](thesis_update_briefing.md).

## The attack taxonomy (the orienting frame)

Four possible attacker objects, each a different inverse problem with different identifiability:

| # | Object the attacker has | Evidence | Status |
|---|---|---|---|
| 1 | Full gradient ∇L | ViT inversion on Flowers + faces | ✅ demonstrated |
| 2 | Fine-tuning update ΔW (full FT or LoRA-induced) | NTK reconstruction on MNIST | ✅ demonstrated |
| 3 | LoRA adapter (A, B) — the actual PEFT artifact | proposed attacks (direct inversion + Gradient Bridge) | ☐ **proposed** (thesis target) |
| 4 | Composed weights W₀ + BA | Haim-style KKT extraction | ✗ failed structurally (Experiment A) |

- **Thesis target is row 3.** The new direct-inversion idea attacks rows 2/3 under strong (known-recipe) assumptions.
- **Row 4 fails structurally:** a rank-2 BA cannot interpolate weights satisfying KKT over the hundreds of
  pre-train + fine-tune support points (~110 in the run shown). A clean negative result, not a tuning failure.

## Core formulation (F as a black-box differentiable map)

Whatever the fine-tuning procedure — SGD, Adam, schedulers — the final weights are a deterministic
function of the base weights and the training samples:

```
θ_T = F(θ₀, x₁, …, x_N)
```

In LoRA the attacker has θ₀ (public base) and (A, B) (published adapter), so `θ_T = θ₀ + BA` is fully
reconstructible. The only unknowns are the samples {x_i}. The attack:

```
{x̂_i*} = argmin_{x̂_i}  ‖θ_T^observed − F(θ₀, {x̂_i})‖²
```

F is treated as a known, deterministic, **differentiable** map: in PyTorch, F is whatever fine-tuning
loop you'd normally run, with x̂_i marked as a learnable `nn.Parameter`; autograd backprops through F to
the candidate data. **This is not specific to trajectory unrolling** — unrolled SGD (known η, T, full
batch) is just one concrete F; Adam, schedulers, etc. fit the same formulation.

## Two attacker-knowledge regimes

- **Regime A — endpoint only (realistic):** attacker has only θ_T. Loss = `‖F(θ₀, x̂_i) − θ_T‖²`. The real
  LoRA threat model — HuggingFace publishes the adapter, not intermediate checkpoints.
- **Regime B — full trajectory (best-case):** attacker has all intermediate θ_t. Per-step matching loss,
  shorter backprop chains, easier — but a stronger assumption.

**Why few-shot LoRA is the sweet spot:** small N (5–50 unknowns), small T (10–50 steps, tractable unroll),
full-batch realistic at small N, recipe metadata often on the model card, θ₀ public by construction.

## Theoretical hook (NTK regime)

For SGD with small η, F has closed-form structure:

```
θ_T − θ₀ ≈ −η Σ_{s<T} (I − ηH)^(T−s−1) · Σ_i c_i(s) · ∇Φ(θ_anchor; x_i)
```

(H = NTK Hessian / feature-gradient Gram matrix; c_i(s) = per-sample residual at step s.) Inverting F in
the NTK regime is then a **structured linear inverse problem**, with identifiability conditions writable
analytically in terms of **feature-gradient rank** and the **eigenstructure of (I − ηH)**. This is where
Gal's theory expertise plugs in — and where the **anchor α-sweep** (briefing Addition 3) lives: the choice
of θ_anchor trades linearization error against identifiability of x_i.

## Concerns to manage

- **Memory** (only when F is an unroll): gradient checkpointing, truncated BPTT, implicit gradients, or
  Regime B. Start small (MNIST T=5–10).
- **Non-convexity / non-uniqueness:** multiple training sets may produce the same θ_T. Identifiability ↔
  feature-gradient rank in the NTK regime; check empirically with multi-restart elsewhere. Characterizing
  the non-uniqueness map is itself a contribution.
- **Known-recipe assumption is strong:** frame the clean version as a best-case leakage upper bound;
  stress-test η/T/recipe mismatch as separate experiments.
- **Stochasticity:** few-shot is often full-batch; if not, a known seed restores determinism, else
  marginalize/optimize over batch order.
- **θ₀ must be exactly right:** errors compound through F. Same-checkpoint is cleanest; checkpoint-drift is
  a stress test; cross-architecture probably breaks.

## Relationship to the Gradient Bridge (complementary, not competing)

| | Direct weight inversion | Gradient Bridge |
|---|---|---|
| **Mechanism** | invert F directly (no decoder, no proxy) | decoder `f_φ: (A,B) → ∇_W L` trained on proxy data, fed to single-gradient inversion |
| **Attacks** | rows 2/3 under known-recipe assumptions | row 3 |
| **Assumptions** | stronger (known recipe, exact θ₀) | weaker, more general (proxy-generalization risk) |
| **Thesis role** | leakage **upper bound** under best-case knowledge | how leakage **degrades** under realistic weaker assumptions |

**Combined story:** two complementary attack axes + a shared theoretical core. The Gradient Bridge
material ([GRADIENT_BRIDGE_PLAN.md](GRADIENT_BRIDGE_PLAN.md), [r2f_bridge_concept.md](r2f_bridge_concept.md))
stands unchanged; direct inversion is added beside it. Direct weight inversion does **not** depend on R2F.

## Precursor note

Direct weight inversion generalizes **Approach G "Differentiable Unrolling"**
([reconstruction_approaches.tex](reconstruction_approaches.tex), §G) and **S3.4** ([../STATUS.md](../STATUS.md)
Sprint 3): same `‖θ_T − F(θ₀, x̂)‖²` outer loss, reduces to the single-step NTK result (Experiment B) at
T=1. The briefing adds the general-F view, the two regimes, the NTK identifiability theory, and the anchor
tradeoff — and elevates it from "secondary fallback" to primary axis.

## Two open theory questions (the "Gal-shaped" core)

- **Q-A (stability / identifiability):** for a fixed inversion algorithm/prior, when is the reconstruction
  map `R: g → x̂` well-posed? Identifiability + stability under perturbations of g.
- **Q-B (pretraining / finetuning overlap):** when θ₀ already (partially) fits x_i, does inversion recover
  x_i itself, or only the directions where θ₀ still has residual error? Feature-map injectivity of
  `φ(x) = ∇Φ(θ₀; x)`.

## Updated recommendation (supersedes §7.1–§7.2 prioritization)

- **Primary axis:** Direct weight inversion (DI-Phase 0…3 in [experiment_plan.md](experiment_plan.md)) +
  the three meeting additions (varied-data breadth, smooth activations / GELU, anchor α-sweep).
- **Complementary axis:** Gradient Bridge (GB-Phase 0…2), unchanged.
- **Theory core:** the two questions above, plus the analytic anchor tradeoff (α*).
- Status: all **proposed**; the demonstrated base (rows 1–2) is what these build on.
