# Handoff brief — re-do the LoRA-leakage working note (for the Mac Claude that made the 26-page PDF)

*This brief carries every correction and new result since you made the 26-page note. Pair it with the two attached PDFs (see "Files" at the bottom).*

## What to produce
Re-do the working note you made ("From the Mathematics of the Update to the Experiments"), but:

- **Much TIGHTER — target ~8–10 pages.** The 26-page version was too detailed to actually use. Give each experiment ONE block: *what it asks · a one-line status · what we found · the caveat.* **Cut** the per-experiment "How to run it" and "Telling Gal" blocks (fold the Gal-facing points into a single short playbook section at the end), compress the glossary to the essential symbols, keep a short provenance table, and **drop the editorial changelog** entirely.
- **Keep your design** — the typography, colour, and hand-redrawn figures were excellent; that's exactly what to preserve.
- **Redraw the figures in your style from the corrected numbers** in the "Figure numbers" section below (as you did originally). Don't rely on the attached plain PDF's raw plots — redraw clean.
- **Apply every fix in ERRATA** — several were genuine errors in the 26-page version.
- **Fold in the new result** (E6 flipped from indeterminate to a positive — see "New result").
- Hold the stance throughout: **observe, don't conclude; every leakage number is a lower bound on the *weakest* attacker (prior-free, adapter-only, per-image) — never a limit on what a stronger attacker could do.**

The attached **thesis_note_v2.pdf** is the corrected, compressed content + length target (build on that content). The attached **26-page original** is your design reference.

---

## ERRATA — corrections to the 26-page version (do NOT reintroduce these)

1. **E3 activation-crux mechanism was BACKWARDS** (this is Gal's #1 figure — get it right).
   WAS: "the gate-change term D_v·dM·D_c·Xᵀ *vanishes* for ReLU → so linearization fidelity lives in that term (smooth wins)."
   PROBLEM: a *vanishing* gate-change term means frozen gates → the update stays *more* linear → that points at ReLU as the *better* linearizer, the opposite of the data.
   NOW (correct): the two links are different *kinds* of object. **Informativeness (kinked wins)** = the *static* rank/rigidity of M — a step-like σ′ gives crisp near-binary gate codes → distinct rows → high rank ρ → mixtures separate. **Linearization fidelity (smooth wins)** = the *continuity* of the gate drift dM under fine-tuning — smooth σ has bounded σ″ → small continuous dM → features barely rotate; ReLU is frozen *within* a region but jumps *discontinuously* at every kink-crossing, and those jumps break fidelity where training crosses them. So fidelity tracks the **continuity of dM**, NOT "dM vanishes." Two terms, opposite directions.

2. **Jang citation was wrong, and its attribution must be paper-safe.**
   WAS: "r ≳ N" (and, in a later draft, "r(r+1)/2 > K·N attributed to Jang Thm 4.1").
   NOW: Jang, Lee & Ryu (ICML 2024, arXiv:2402.11867) prove LoRA needs **r ≳ √N** (rank on the order of √N, NOT N) to kill spurious local minima; full-FT admits a rank-√N solution. Attribute ONLY **r ≳ √N** to Jang. The output-dimension refinement **r(r+1)/2 > K·N (√(K·N))** used to explain the multi-class E2 anomaly is **OUR constraint-counting extrapolation, NOT Jang's stated bound** — label it as ours. (Jang is a loss-landscape result, so it anchors the leakage boundary by analogy; say so.)

3. **§5 pushback table cohort slip.** WAS: "rs = +0.78 at n=12." NOW: the g₀ predictor is **+0.857 at n=12** (strong) and **+0.777 at n=24** (indeterminate, CI [0.53, 0.91]). The 26-pager mixed the two cohorts.

4. **World A vs the stance box — internal contradiction.** WAS: page 1 says "never a limit on what an attacker could do," but the World A row calls the ruler's verdict "a privacy guarantee, the strongest thing we can offer." NOW: scope World A as a **local, per-image, linearized guarantee under Gaussian seed noise** — NOT an unqualified guarantee (E6, the composition channel, is exactly an escape from it). Then it no longer contradicts the weakest-attacker framing.

5. **"‖ΔW‖/‖W₀‖ = 0.23" is mislabeled — do not cite it as a relative-update norm.** It's actually the **weight-space linearization error ≈ 0.23** (from an NTK-flagged full-FT config), a different quantity. If you want a "not strictly lazy" point, use the weight-space (0.23) vs function-space (0.0023) linearization-error contrast, with the NTK-flag caveat — or drop the number.

6. **E3 dissociation is metric-dependent — scope it.** State **Spearman(feature-stability, *control-margin* leakage) ≈ 0** (≈ −0.06, the headline metric where the dissociation is clean); footnote that on SSIM it's mildly positive (+0.08 to +0.28). The dissociation (kinked = worst linearizers yet leak most, ~5× on control-margin) holds on the control-margin metric.

7. **"Records concept, not instance" stays removed as a *blanket* claim** (you already cut it) — because E7's full-gradient ceiling *does* recover instance pixels (strong attacker). BUT E6 now supports a *scoped* version: the *adapter-only composition channel* records content (which digits), not the specific instance. Keep those two separate (different attackers/observables).

---

## NEW result — E6 (composition atlas) flipped to a POSITIVE

WAS (26-pager): "rigorous test INDETERMINATE, cross-fit acc-diff = +0.00, CI [0,0] — the picture is perfect but the rigorous test isn't there yet."

That +0.00 was a **cross-fit fold bug**: with 5 compositions and 5 folds, the naive fold assignment isolated each *whole composition* into its own test fold (none in train) → recovery impossible → acc-diff mechanically 0. **Fixed.**

NOW (verified): **composition IS recoverable from ΔW above the fitted-recipe baseline** — cross-fitted held-out **acc-diff = +0.989, 95% CI [+0.973, +1.005], G = 30** (cluster-robust). ΔW still clusters perfectly by composition (ARI +1.00) and the raw (B,A) by seed (+0.55) — gauge-contrast confirmed.

**Scope (be precise):** the 5 compositions are 5 distinct **digit-subsets** ({1,6,7,8}, {0,1,7}, {1,6,7}, {0,1,4,9}, {3,4,8,9}) under a shared binary task — so +0.989 recovers **which digit-subset = content / concept-level** (which digits were present), the honest **floor**. A stronger *instance-level* reading (which specific images) is **possible but untested** — single-image swaps are near-invisible (0.03–0.07). So: content-level recoverable, graded down to ~0 for single swaps. Zoo = 169 converged adapters of a 180-cell grid (5 comps × 3 activations × 2 lr × 6 seeds; 11 non-converged dropped).

**Cite & differentiate:** *Learning on LoRAs* (Putterman, Lim, Gelberg, Jegelka, Maron; ICLR 2025; arXiv:2410.04207) — GL-equivariant processing of LoRA weights to predict fine-tuning data attributes and membership (our exact gauge + channel). Differentiation: they train a learned *probe* ("can a model extract it"); we ask whether composition is *forced* into ΔW above a recipe baseline (variance decomposition + cross-fit).

---

## Corrected figure numbers (redraw in your style)

- **E2 rank sweep** (job 581629): multi-class "leaks fewer" gap **23 → 13 → 0** across r = 8/16/32. Binary q_eff flat ~58; 10-class climbs 36 → 47 → 58. Story = our √(K·N) extrapolation: binary K=1 clears at all r; 10-class K=10 needs r ≈ 14, so r=8 (dof 36 < 100) below, r=16 (dof 136 > 100) above.
- **E3 dissociation** (jobs 392821 / 390026), two panels: (A) linearization fidelity by activation — sigmoid/softplus highest (~0.98 / 0.86), kinked relu/leaky_relu lowest (~0.67); (B) leakage — kinked ~0.47 vs smooth ~0.09 on control-margin (~5×). Spearman(fidelity, control-margin leakage) ≈ −0.06.
- **E4 g₀** (jobs 260171 / 272504): per-image sensitivity vs g₀; ρ = +0.857 (n=12) / +0.777 (n=24, CI [0.53, 0.91], indeterminate); tercile +0.88 (low g₀) → −0.12 (high g₀). g₀ beats the max-margin dual λ: **0.857 vs 0.538** (n=12 cohort).
- **E5 valley** (job 695782): full-FT ~5× more signal per image (target-median; per-target 3–6×) at ≈ the same resolution — valley-width ratio geomean **1.02**, median **0.86**, narrower on 4/6 targets (n=6). More signal, not finer memory.
- **E6 atlas** (job 838868; 811847 was fold-buggy): ΔW ~ composition ARI +1.00; raw (B,A) ~ seed +0.55; composition recovery acc-diff **+0.989**, CI [0.973, 1.005], G=30.
- **E7 reconstruction**: full-gradient ceiling SSIM up to ~0.99 (MNIST/CIFAR/Flowers); direct weight inversion ssim_norm 0.57 @ N=4 → 0.27 @ N=10 (superposition wall); gradient-bridge decoder 0.951 cosine; q_eff up to 156/160; ViT faces 0.38 / 0.26 / 0.52. (Label these the full-gradient CEILING — the upper bound, NOT the adapter-only attack.)

---

## New citations worth adding (from an expert review)
Wang–Lee–Lei (AISTATS 2023, provable gradient identifiability — §1); Hannun–Guo–van der Maaten (Fisher-information leakage — the §2 ruler); Feldman–Zrnic individual privacy accounting (E4 — the integrated/mid-trajectory gradient norm explains the g₀ saturation; the whitened predictor ⟨gᵢ, Σ⁻¹gᵢ⟩ likely fixes the USPS counterexample); Balunović et al. (Bayesian gradient inversion = the World-C prior formalism); and the *Learning on LoRAs* cite above for E6.

## Open items — flag as "to resolve," don't invent numbers
- Re-measure the max-margin dual **λ** used in the E4 g₀-vs-λ comparison (log its job) or drop the comparison.
- One line on the reconstructed-faces dataset provenance/consent (the only ethics-exposed slide).

## Files to work from
1. Your **original 26-page PDF** — design/structure reference.
2. **thesis_note_v2.pdf** (attached) — the corrected, compressed content + length target.
3. **this brief** — the errata, the new E6 result, and the redraw numbers.
