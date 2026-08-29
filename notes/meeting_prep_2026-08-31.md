# Meeting prep — Monday 2026-08-31 (supervisor: Gal Vardi, implicit-bias theorist)

**Stance (non-negotiable, per the swarm's observe-directive):** early exploratory research —
OBSERVE, don't conclude. Every number below bounds only the WEAKEST attacker (prior-free,
adapter-only, per-image); priors / known-recipe inversion / structural leakage go beyond it. No
"confirmed / done." Small-n and caveats stated up front, not buried.

---

## The one-paragraph story
LoRA fine-tuning leaks individual training images *detectably* (p=0.002) but our reconstruction
attack fails the honest bar (0/40 vs mean-image baseline). We built a validated instrument (whitened
Mahalanobis sensitivity, proven unbiased) to ask *why*. Finding 1: **which images leak is written in
the public base model's gradients** (g₀) — attacker-relevant, and it *transfers* to full fine-tuning.
Finding 2 (the new one, against our own hypothesis): **full training does NOT pin images into a
narrower "indistinguishability valley" than a rank-8 adapter** — so the reconstruction gap is
probably *decoder/pipeline-side, not information-geometry-side*. Finding 3: recovery is **concept-
level, not instance-level** (near-duplicate swaps are ~invisible). Depth: the instance/pixel signal
lives in the **first layer**.

---

## Figures, in meeting order — what each shows, what we EXPECTED, what we GOT, the caveat

### F-A. Margin scatter — "WHO leaks" (fig_f3_margin, margin_at_scale)
- **Shows:** per-image sensitivity vs base-model gradient g₀, n=24, stratified.
- **Expected:** g₀ strongly predicts (MVP was ρ=0.857, n=12).
- **Got:** ρ=+0.777, p=1e-4 — real and strong, but **INDETERMINATE** (CI [0.53,0.91] too wide for the
  pre-registered ±0.15; tercile sign-flip). Correct reading: predictor **STRONG at low g₀ (+0.88),
  SATURATES/reverses at high g₀ (−0.12)** — the relationship saturates as g₀ grows. WHY it saturates
  is OPEN. Partial-ρ survives a θ₀-independent typicality control (+0.78) ⇒ not just image atypicality.
- **Caveat:** n=24; indeterminate verdict; g₀ needs base model + candidate image (NOT "base alone").

### F-B. Distance dial — instance vs concept (fig_f2_similarity_ladder)
- **Shows:** swap-sensitivity vs graded visual distance; the d=0 self-swap control reads exactly 0.
- **Expected:** sensitivity rises with distance; near-duplicates ~null ⇒ concept not instance.
- **Got:** supported — near-dup rungs ≈ floor, sensitivity climbs to the cross-digit anchor. Adapter
  records "a kind of image," not the exact pixels. Privacy statement: attacker recovers the *concept*.
- **Caveat:** ~9 rungs/target, small-n; d_pixel axis (semantic axis is secondary).

### F-C. The valley ladder — full-FT vs LoRA (fig_valley_ladder) — THE NEW HEADLINE
- **Shows:** normalized profile s(d) for LoRA(A) / full-single-layer(C) / full-all-layers(D), + d* bars.
- **Expected (P1):** full training pins a NARROWER valley (d*_full < d*_LoRA; s(near-dup) ≥ 3× LoRA) —
  which would explain why Haim reconstruction works and LoRA's doesn't.
- **Got — AGAINST hypothesis:** d*_full ≈ d*_LoRA (2.6 vs 2.7 / 2.2 vs 2.0); near-dup ratio ~1, not ≥3.
  **Two independent methods agree** (finite-swap dial + noise-free Jacobian P7 ratio full≈LoRA). So the
  parameterization does NOT narrow the valley ⇒ the reconstruction gap is likely **decoder-side**.
- **Guards:** B1 dimension-invariance PASS (d*≈d* is NOT a 70×-dim artifact). B2 ε-vs-SGD noise
  DIVERGENT (SGD ~30% narrower) ⇒ read the comparison QUALITATIVELY; the qualitative equality survives
  because the noise-free Jacobian shows it too.
- **Caveat:** 2 dial targets (scale-up to 6 running, job 695782); B2 divergence; qualitative not precise.

### F-D. Removal cross-regime + g₀ transfer (fig_removal_crossregime) — arm F, the robust one
- **Shows:** (a) full LOO footprint vs LoRA LOO footprint per image; (b) full footprint vs g₀.
- **Expected:** same images imprint most in both regimes; g₀ predictor transfers.
- **Got — clean:** rank corr ρ=+0.943 (same images), ρ(footprint, g₀)=+0.829 (predictor transfers to
  full FT). Absolute footprint ~5× bigger in full — the reconciliation: **more signal, not finer
  resolution** (feeds the "decoder-side gap" reading in F-C).
- **Caveat:** n=6; absolute-magnitude comparison is descriptive (N→N−1 offset).

### F-E. Depth fan — "all layers of it" (fig_valley_depth) — arm D per-layer
- **Shows:** per-layer numerator ‖Δμ_ℓ‖ vs distance for the full network.
- **Expected (P2):** the pixel-carrying first layer reacts to near-duplicates earliest (instance early).
- **Got:** L0 ‖Δμ‖ largest at the near-dup rung (0.022 > 0.013 > 0.003), fading with depth — instance/
  pixel signal concentrated early. Directly answers "how does the imprint distribute across layers."
- **Caveat:** read on the NUMERATOR (per-layer d* is denominator-confounded); K-scale plumbing.

### F-F. Activation crux — supervisor's TOP ask (activation_crux_summary) [analysis in flight]
- **Shows:** reconstruction fidelity (baseline-relative ssim_norm) vs activation smoothness / feature-
  stability / NTK-regime, across ~21 configs (+ running feature-stability-vs-T).
- **Expected (prior read):** "smoother ⇒ more leakage" is REFUTED (ρ≈0); smoothness sets fidelity, not
  direction-count; r_J is β-independent.
- **Got:** pending the analysis agent + the two running jobs (390026/392821). Report at meeting-time.

---

## The most interesting figures (rank for the talk)
1. **F-C valley ladder** — the surprising, against-hypothesis result; reframes the reconstruction gap.
2. **F-A margin scatter** — the strongest positive (attacker predicts exposure from the public model).
3. **F-B distance dial** — the deepest privacy statement (concept, not instance).
4. **F-D removal + g₀ transfer** — the robustness that ties WHO-leaks across parameterizations.
5. **F-E depth fan** — answers the mechanistic "which layer" question directly.
6. **F-F crux** — the supervisor's own axis.

## What to have ready for Gal's likely pushback
- "Is this the KKT/max-margin regime?" → No — we reframed the spine to NTK/gradient-recording (g₀ beats
  the max-margin dual λ: 0.78 vs 0.51); LoRA is NOT strictly lazy (‖ΔW‖/‖W₀‖=0.23), so it's gradient-
  *structure* stability, not laziness. Convergence diagnostics gate any KKT language.
- "Detectability ≠ reconstruction; your 0/40." → Owned. The valley result says the gap is decoder-side;
  the J-composed Fisher → Fano bridge (scheduled) is the rigorous connection.
- "n is tiny." → Yes; margin scale-up (n=24) done, valley scale-up (n=6) running; stated as exploratory.

## Open / not finished (honest)
Full validation gate at scale (only the spot-check ρ=0.88 ran); the crux jobs (in flight); reconstruction
itself (0/40, unsolved); F5 shared-perturbation (scaffold, compute-gated — awaiting Gal). Figures F2/F3
self-audited this session; the full multi-figure set is being rendered from committed data.
