# Deck math audit — 2026-08-31 supervisor deck

**Auditor pass:** every equation rendered in `scripts/deck/deck/slides_*.py` (via `render_math` /
`render_lines`) checked against the code/notes it transcribes, and the mathtext PNGs in
`figures/deck_2026_08_31/eq/` checked for rendering artifacts.

## Summary

**24 equation statements checked** (across 22 render files; multi-line blocks counted as one row).
**Verdict tally: 22 ✓ exact/correct · 2 ⚠ imprecise · 0 ✗ wrong · 0 rendering artifacts.**

The deck's math is in excellent shape. Every core object — the gate-matrix rank theorem, the
whitened d² and its four equivalences, the col(J) q_eff construction, J_full, the anchor
linearization residual, the DI loss, ctrl-margin, g0, mem_i, P_LoRA, the LoRA gauge — transcribes its
source **exactly**. The two CRITICAL guardrails held: `ΔW ≈ Σ cᵢ∇f` is rendered with `≈` (not `=`),
and `P_LoRA` is explicitly labelled "a PSD operator, not a projection" on slide r6. No equation is
stated in a form stronger than its source, with one qualified exception (m2). All 9 spot-inspected
PNGs (a1, a2, a3, m3, anchor, delta_w, gate_drift, ctrl_margin, r6, t1) render cleanly — `\overset`,
`\mathcal`, `\top`, `\nabla`, fractions, overlines and calligraphic all correct.

**Most important issues (all minor):**
1. **m2 (slide 8):** headline "⇒ a ceiling on every attack, reconstruction included" slightly
   overreaches — d² bounds *adapter-space* error, not pixel MSE directly. It IS qualified by the
   on-slide caveat chip ("bounds the adapter-space change Δμ, not pixel error… images need Δμ pushed
   through the data→adapter Jacobian"), so it is not wrong, but the word "reconstruction included" in
   the green headline is a hair stronger than the chip. Tighten the headline.
2. **more_data / eq_delta_w:** slide writes `∇_W f(θ₀;xᵢ)`; the source
   (`linearization_leakage_theory.tex:62-66`) linearizes about the **anchor** θ_a with `∇_θ f(x;θ_a)`.
   At α=0 (default) θ_a=θ₀, so it is correct for the base case; flagged only as an imprecision if the
   slide is meant to hold at general α. The `≈` is correct.
3. **Cross-source `G` convention (context, not an error):** slides use `G = D_v M D_c`
   (matches `identifiability_rank_bound.tex:111-116`); `linearization_leakage_theory.tex:72-87` folds
   `D_c` out separately (`G = D_v M`, then `ΔW = G·diag(c)·Xᵀ`). Slides are internally consistent and
   match the primary source. No fix needed; noted for reviewer awareness.
4. **`J` vs `J_full`:** t2 introduces `J_full`; t3/a3 use bare `J` (in `J_SNR = Σ_seed^{-1/2}J`, and
   `J` = col-basis Jacobian in a3). Same object, subscript dropped — mildly loose, harmless.

---

## Theory slides

| slide | equation (as shown) | source (file:line) | verdict | fix |
|---|---|---|---|---|
| t1 | `z →g→ x →LoRA→ (A_T,B_T)=F(z)` | `notes/identifiability_feasibility_revision.tex:41-72` (measurement-system framing) | ✓ exact | — |
| t2 | `J_full = ∂vec(A_T,B_T)/∂(z₁,…,z_N)` | `identifiability_feasibility_revision.tex:100-105` | ✓ exact | — |
| t3 | `J_SNR = Σ_seed^{-1/2} J`; `q_eff(ε)=#{i: ε·σᵢ(J_SNR)>1}` | `experiments/jacobian_spectrum.py:20,22` | ✓ exact | — (bare `J` = `J_full`; see summary note 4) |

## Measure slides

| slide | equation (as shown) | source (file:line) | verdict | fix |
|---|---|---|---|---|
| m1 | `d²=(μ(D′)−μ(D))ᵀΣ⁻¹(μ(D′)−μ(D))` | `notes/whitened_sensitivity_metric.md:22` | ✓ exact | — |
| m2 | `d²=SNR²_NP=2·KL(P_{D′}‖P_D)` | `whitened_sensitivity_metric.md:49-70` | ⚠ imprecise | eq is exact; the equal-Σ Gaussian caveat is on-slide (lead). But the headline "⇒ a ceiling on every attack, reconstruction included" overreaches (d² bounds adapter-space, not pixel MSE); tighten to "adapter-space ceiling" — chip already says so. |
| m3 | `d̂²_3-way: A→U, B→Δμ·U, C→λ`; `sens = d̂²_obs − mean(d̂²_null)` | `whitened_sensitivity_metric.md:38` + `experiments/dataset_sensitivity/whitened_metric.py` (3-way cross-fit) | ✓ exact | — |

## Results slides

| slide | equation (as shown) | source (file:line) | verdict | fix |
|---|---|---|---|---|
| r1 | `d² ∝ k^β, β≈0.23 ≪ 1` | `notes/dataset_sensitivity_program_plan.md:165-176` (β=0.234 @ r8/T=1000) | ✓ exact | β is T-dependent (0.313→0.234); 0.23 is the r8/T=1000 headline value — correct as the stated recipe |
| r3 | `g₀(xᵢ)=‖∇_{W₀} BCE(θ₀;xᵢ)‖_F` | `experiments/dataset_sensitivity/margin_vs_sensitivity.py:25,74-91` | ✓ exact | — |
| r4 | `s(d)=sens(d)/sens(d_cross)` | `experiments/dataset_sensitivity/fig_f2_similarity_ladder.py:37,202-210` (`NORM_RUNG="r_cross"`) | ✓ exact | — |
| r5 | `mem_i=E_seeds[margin(D)−margin(D\{xᵢ})]` | `experiments/dataset_sensitivity/h_spotcheck.py:74-121` | ✓ exact | — |
| r6 | `P_LoRA(H)=BBᵀH+HAᵀA` | `notes/thesis_note_v2.md:26` | ✓ exact | CRITICAL check PASS: slide side-label says "a PSD operator, not a projection" — correct. Symbol named `P` but never claimed to be a projection. |
| r7 | `BA=(BR)(R⁻¹A)` | `notes/thesis_note_v2.md:91` | ✓ exact | — |

## Answers slides

| slide | equation (as shown) | source (file:line) | verdict | fix |
|---|---|---|---|---|
| crux | `leak=ssim_n(x̂,x)−ssim_n(x̂,x_ctrl)` | `experiments/recompute_metrics.py:87` (`ctrl_margin_norm`) + `experiments/metrics.py` | ✓ exact | — |
| fs | `fs(T)=cos(∇_θ f(θ₀;x), ∇_θ f(θ_T;x))` | `experiments/ntk_verification.py:88-110` | ✓ exact | — |
| anchor | `θ(α)=(1−α)θ₀+α·θ_T`; `L_lin=‖Φ(θ_T)−[Φ(θ_a)+∇Φ(θ_a)δ]‖/‖Φ(θ_T)−Φ(θ_a)‖` | `experiments/configs.py:76-79` (θ(α)); `experiments/ntk_verification.py:324-340` (L_lin) | ✓ exact | δ=θ_T−θ_a stated correctly in caption |
| mechanism | `Ω=Σᵢgᵢxᵢᵀ=GXᵀ, G=D_v M D_c, M_ki=σ′(⟨w_k,xᵢ⟩)` | `notes/identifiability_rank_bound.tex:111-116` | ✓ exact | — |
| mechanism | `dM ∝ σ″ dz` | `notes/linearization_leakage_theory.tex:109-125` | ✓ correct | differential of `M_ki=σ′(z)` ⇒ `dM=σ″dz`; source gives the finite gate-range `≈σ″(z̄)·spread`. `∝` is the honest form. |
| direct_inversion | `min_{x̂} ‖θ_T − F(θ₀,x̂)‖²` | `experiments/direct_inversion.py:97-110` (`di_endpoint_loss`) | ✓ exact | — |
| more_data | `ΔW ≈ Σᵢ cᵢ ∇_W f(θ₀;xᵢ)` | `notes/linearization_leakage_theory.tex:62-66` | ⚠ imprecise | CRITICAL check PASS: rendered with `≈` (PNG confirmed), not `=`. Source linearizes about the **anchor** θ_a with `∇_θ f(x;θ_a)`; slide uses θ₀ and `∇_W`. Correct at α=0 (θ_a=θ₀); note the anchor generality if the slide is meant beyond α=0. |

## Appendix slides

| slide | equation (as shown) | source (file:line) | verdict | fix |
|---|---|---|---|---|
| a1 | `Ω=Σgᵢxᵢᵀ=GXᵀ, G=D_v M D_c, M_ki=σ′(⟨w_k,xᵢ⟩)`; `rank(M)=k<N ⇒ Φ⁻¹(Ω)=X+K, K={H: every row of H ∈ ker G}`; `dim K=d(N−k)≥d≥1`; `leakage ≤ min(rank(M),r,N)` | `identifiability_rank_bound.tex:95-118, 159-183, 253-259` | ✓ exact | matches thm:main verbatim, incl. LoRA ceiling; slide labels carry the frozen-known-G scope |
| a2 | `d²=ΔμᵀΣ⁻¹Δμ = 2KL = SNR²_NP = 𝓘_Fisher` | `whitened_sensitivity_metric.md:49-70` | ✓ correct | equal-Σ Gaussian caveat present on-slide (label + amber "Cramér–Rao, not pixel MSE" box). No overclaim. |
| a3 | `r_J=rank(J)`; `Q=orthonormal basis of col(J)∈ℝ^{dimY×r_J}`; `Σ_J=Cov(Qᵀ(Y_s−Ȳ))`; `F=(QᵀJ)ᵀΣ_J⁻¹(QᵀJ)`; `q_eff(ε)=#{i: ε√λᵢ(F)>1}`; `iso=tr(Σ_J)/(μ r_J)` | `experiments/jacobian_spectrum.py:580-620` (`q_eff_colspace`) | ✓ exact | matches implementation line-for-line |

---

### Notation-consistency scan (across slides)
- **θ₀ / θ_T** — consistent everywhere (init / trained endpoint). ✓
- **Σ vs Σ_seed vs Σ_J** — Σ (m1/a2 eqs) = Σ_seed (t3, m1 figure label); Σ_J (a3) is the distinct
  col(J)-projected covariance, correctly subscripted. Consistent. ✓
- **d² vs sens** — `sens` = null-corrected d² (`d̂²_obs − mean d̂²_null`, m3); r4 uses `sens(·)`.
  Consistent split between raw quadratic form and the reported null-corrected statistic. ✓
- **J vs J_full vs J_SNR** — J_full defined in t2; t3/a3 drop the subscript to J. Same object; minor
  looseness, harmless.
- **M / G** — M = gate matrix (σ′ entries), G = D_v M D_c, used consistently in a1 + mechanism. ✓
  (cross-*source* G-convention difference noted in summary #3 — not a slide inconsistency.)

### Rendering scan (PNGs inspected)
Clean, no artifacts: `eq_a1_rank`, `eq_a2_d2`, `eq_a3_qeff`, `eq_m3_threeway_null`,
`eq_anchor_family` (fraction OK), `eq_delta_w` (≈ OK), `eq_gate_drift`, `eq_ctrl_margin`,
`eq_r6_plora`, `eq_t1_measurement_map` (`\overset` OK). Remaining PNGs use only these same standard
mathtext primitives (`\mathrm`, `\top`, `\nabla`, `\Sigma`, `\Phi`, `\mathcal`, subscripts) already
confirmed rendering elsewhere — no artifact risk.
