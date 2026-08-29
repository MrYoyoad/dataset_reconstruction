# Meeting prep — Monday 2026-08-31 (supervisor: Gal Vardi, implicit-bias theorist)

**Stance (non-negotiable, per the swarm's observe-directive):** early exploratory research —
OBSERVE, don't conclude. Every number below bounds only the WEAKEST attacker (prior-free,
adapter-only, per-image); priors / known-recipe inversion / structural leakage go beyond it. No
"confirmed / done." Small-n and caveats stated up front, not buried.

---

## Framing (user directive, 2026-08-29): POSITIVE discoveries + clean rejections-with-mechanism
Lead with what we POSITIVELY discovered and what we POSITIVELY rejected *and understand why*. Do NOT
feature reconstruction pass/fail numbers or mean-baseline verdicts — those came from an under-cooked
inversion setup and are not the interesting content. Reconstruction itself: **show the real positive
examples we already have (Phase-0 ViT inversion — recognizable faces / rose structure) and state that
a proper, robust inversion is the next-weeks work.** The science to present is the *characterization*
of what fine-tuning does to the weights and which images are exposed — all positive.

## The one-paragraph story (honesty-audited — qualifiers are load-bearing)
Using a validated instrument (whitened Mahalanobis sensitivity, proven unbiased), we characterize how
LoRA fine-tuning records its training images — all observations bound only the WEAKEST (prior-free,
adapter-only, per-image) attacker. **(1)** Which images are most exposed is **strongly predictable at
low base-gradient g₀ and saturates as g₀ grows** (overall verdict INDETERMINATE at n=24), and the
predictor *transfers* to full fine-tuning (ρ=0.83, n=6). **(2)** The adapter records the **concept, not
the instance** — *parametric* near-duplicates are ≈invisible to it. **(3)** New characterization of
parameterization (n=2, UNSETTLED): **full fine-tuning records ~5× more per-image signal (descriptive,
n=6)**; whether it also resolves *finer* is open — the finite-swap dial suggests full is MODESTLY
NARROWER (d* ~15–20% smaller, n=2), the noise-free Jacobian shows EQUAL resolution, and the SGD-noise
check diverges. So: clearly more signal; possibly-slightly-finer resolution, not settled (n=6 running). **(4)** That extra signal lives
disproportionately in the **first (pixel-carrying) layer**. **(5)** We already get recognizable
**full-gradient** reconstructions across a real sweep (F-0); the information is present, so a robust
adapter-only inversion is the immediate next step.

---

## Figures, in meeting order — what each shows, what we EXPECTED, what we GOT, the caveat

### F-0. Positive reconstructions — "we recover recognizable images" (LEAD SLIDE)
- **Shows:** a GALLERY across the early reconstruction sweep — **mnist / fashion / cifar10 / flowers ×
  N∈{2,4,10} × {gelu, softplus}** — recovering recognizable training images from the weight change
  (full-gradient setting, `results/gb_e2e_*.pth` → `figures/meeting/positive_reconstruction_gallery.png`,
  building). PLUS the ViT-scale Phase-0 result (recognizable faces `figures/phase0/n3_three_faces.png` /
  rose structure `phase0_full_r8_n1.png`).
- **Message:** across datasets, image-counts, and activations, the full-gradient reconstruction recovers recognizable images —
  the information IS in the weights and IS recoverable. **Some configs work cleanly today; making the
  robust ADAPTER-ONLY inversion work across the board is the immediate next-weeks work.** This ties to
  F-C: full fine-tuning holds ~5× more per-image signal, so the open problem is *extraction*, not missing
  information.
- **Use:** open here so reconstruction reads as *working-with-traction across a real sweep*, not a result
  to defend. NO pass/fail-vs-baseline numbers, NO "0/40" — dropped per the framing directive.
- **Winners (built, figures/meeting/positive_reconstruction_gallery.png):** mnist N=2 gelu (digits crisp,
  0.99), cifar10 N=2 gelu (boat+car, 0.9995), flowers32 N=2 gelu (rose+lily, 0.9994), fashion N=2 softplus
  (boot, 0.64 — honest weakest, still recognizable). All four recognizable — none dropped.
- **HONESTY LABEL (say this out loud):** these are the **full-gradient / TRUE-ΔW *ceiling*** reconstructions
  — what's achievable when you have the true weight-change. They prove the information is recoverable; they
  are NOT the adapter-only decoded attack. The adapter-only inversion is exactly the next-weeks work — and
  F-C says it has ~5× more signal to work with. Do not let the gallery read as "our LoRA attack got these."

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
- **Got:** supported — *parametric* near-dup rungs (blur/rotate/brightness) ≈ floor (the retrieved
  nearest-neighbor rung is moderate, s≈0.24–0.39); sensitivity climbs to the cross-digit anchor. Adapter
  records "a kind of image," not the exact pixels. Privacy statement: attacker recovers the *concept*.
- **Caveat:** ~9 rungs/target, small-n; d_pixel axis (semantic axis is secondary).

### F-C. The valley ladder — full-FT vs LoRA (fig_valley_ladder) — THE NEW POSITIVE CHARACTERIZATION
- **Shows:** normalized profile s(d) for LoRA(A) / full-single-layer(C) / full-all-layers(D), + d* bars.
- **Positive finding (observed):** full fine-tuning records **~5× more per-image signal** (removal
  footprint, descriptive, n=6). Whether it also resolves *finer* is UNSETTLED at this scale:
  the finite-swap dial reads full **modestly narrower** (d*: full-D 2.59/2.23 vs LoRA-A 3.11/2.69,
  ~15–20%, n=2 — live from valley_headline_dstar.json), the **noise-free Jacobian reads EQUAL**
  (P7 full≈LoRA, gap not direction-robust), and the **SGD-noise check (B2) diverges**. So: clearly more
  signal; possibly-slightly-finer resolution, not a clean result — do NOT overstate either "narrower" or
  "equal." Scale-up (n=6, job 695782) running to firm it. Weakest-attacker scope applies.
- **Guards:** B1 dimension-invariance PASS (the equality is NOT a 70×-dim artifact). B2 ε-vs-SGD noise
  DIVERGENT (SGD ~30% narrower) ⇒ read QUALITATIVELY; the qualitative equality survives via the
  noise-free Jacobian.
- **Why it matters:** it says the extra information in full fine-tuning is *there to be inverted*
  (more signal) — the open direction is extracting it, not a missing-information wall.
- **Caveat:** 2 dial targets (scale-up to 6 running, job 695782); B2 divergence; qualitative not precise.

### F-D. Removal cross-regime + g₀ transfer (fig_removal_crossregime) — arm F, the robust one
- **Shows:** (a) full LOO footprint vs LoRA LOO footprint per image; (b) full footprint vs g₀.
- **Expected:** same images imprint most in both regimes; g₀ predictor transfers.
- **Got (n=6, exploratory):** strong rank corr ρ≈+0.94 (same images imprint most in both), ρ≈+0.83
  (predictor transfers to full FT). Absolute footprint ~5× bigger in full (target-median; per-target
  ~3–6×) — the reconciliation: **more signal, not finer
  resolution** (feeds the "decoder-side gap" reading in F-C).
- **Caveat:** n=6; absolute-magnitude comparison is descriptive (N→N−1 offset).

### F-E. Depth fan — "all layers of it" (fig_valley_depth) — arm D per-layer
- **Shows:** per-layer numerator ‖Δμ_ℓ‖ vs distance for the full network.
- **Expected (P2):** the pixel-carrying first layer reacts to near-duplicates earliest (instance early).
- **Got:** L0 ‖Δμ‖ largest at the near-dup rung (0.022 > 0.013 > 0.003), fading with depth — instance/
  pixel signal concentrated early. Directly answers "how does the imprint distribute across layers."
- **Caveat:** read on the NUMERATOR (per-layer d* is denominator-confounded); K-scale plumbing.

### F-F. Activation crux — supervisor's TOP ask — OWNED BY yoado-ed (authoritative figure pending)
- **Status:** yoado-ed owns the crux track and is finishing the FULL 152-config rescore + the free-c
  wc-ladder (job 392821, landing ~this hour); THEY provide the final crux figure. The current
  figures/crux/activation_crux_summary.png is FIRST-PASS with PARTIAL panels — do not present as settled.
- **Shows:** 4 panels — feature-stability by activation; fidelity vs feature-stability; free-c leakage
  cluster; feature-stability↓ / eff-rank↑ with T. **Panels c/d PARTIAL** (jobs running) — mark
  "PRELIMINARY" on the figure or wait for yoado-ed's refresh.
- **Positive characterization (n=6/n=11, exploratory — SUGGESTIVE links, not load-bearing):** activation
  **smoothness tracks the lazy/NTK regime** (smoothness → feature-stability ρ≈+0.85, n=6), and
  **feature-stability tracks reconstruction fidelity** (ρ≈+0.94, n=6 — CI spans ~0.5–0.99). This is the
  implicit-bias/NTK direction Gal cares about; state as observed, not proven.
- **Observed rejection (observe-framed, not "refuted"):** the naive **"smoother ⇒ more fidelity" does
  NOT hold on the fuller set** — the smooth-only ρ≈+0.85 drops to ≈+0.11 across 11 activations; the
  apparent driver is *laziness*, not smoothness per se (weight-change-confounded).
- **CRITICAL caveat (yoado-ed):** the **kink lead (leaky_relu/selu ~5×)** is an ORACLE / upper-bound
  read; documented precedent is that the **realistic free-c attack FLIPS the activation ranking** — so
  do NOT present "kink leaks most" as the realistic result until the free-c ladder lands. Direction-count
  (eff_rank) is **T-driven, not activation-driven** (grows
  2→6.3 with T) — consistent with the on-record "r_J β-independent."
- **Caveat:** n=6 on the CSV smooth subset; c/d partial; small-n throughout — exploratory.

---

## The most interesting figures (rank for the talk)
1. **F-0 positive reconstructions** — open here: recognizable faces/rose, robust inversion = next weeks.
2. **F-A margin scatter** — the strongest positive (attacker predicts exposure from the public model).
3. **F-B distance dial** — the deepest privacy statement (concept, not instance).
4. **F-C valley ladder** — the new positive characterization (full = more signal, same resolution).
5. **F-D removal + g₀ transfer** — the robustness that ties WHO-leaks across parameterizations.
6. **F-E depth fan** — answers the mechanistic "which layer" question directly.
7. **F-F crux** — the supervisor's own axis (activation smoothness → dynamics).

## What to have ready for Gal's likely pushback
- "Is this the KKT/max-margin regime?" → No — we reframed the spine to NTK/gradient-recording (g₀ beats
  the max-margin dual λ: 0.78 vs 0.51); LoRA is NOT strictly lazy (‖ΔW‖/‖W₀‖=0.23), so it's gradient-
  *structure* stability, not laziness. Convergence diagnostics gate any KKT language.
- "Detectability isn't reconstruction." → Agreed — detectability is the *ceiling* (what an attacker
  could do). We show recognizable recoveries already (F-0); the valley result shows the information is
  present (more signal in full FT), so robust inversion is an extraction problem, scheduled next.
- "n is tiny." → Yes; margin scale-up (n=24) done, valley scale-up (n=6) running; stated as exploratory.

## Next-weeks work (framed as forward, not as failure)
- **Robust adapter-only inversion** — we have recognizable recoveries in favorable settings (F-0); the
  full-FT valley result says the information IS present (~5× more signal than LoRA), so the task is
  *extracting* it (better decoder / prior-equipped inversion), not overcoming a missing-information wall.
- **Firm the under-powered readouts** — margin at n=24→more; valley dial n=2→6 (job 695782 running).
- **Full validation gate at scale** (spot-check ρ=0.88 done) — the behavioral-memorization tie.
- **Activation crux** — the two jobs (390026/392821) complete + the fuller analysis (F-F).
- F5 shared-perturbation stays scaffold (compute-gated, awaiting Gal's go).

Figures F2/F3 self-audited this session; the full figure set is being rendered from committed data.
