# Program: Dataset-composition sensitivity of LoRA adapters (+ adapter-space geometry)

## THE PROGRAM (what is being tested — read first)

Stop perturbing a continuous knob `a`. Instead measure, **directly and empirically, how sensitive the
released LoRA adapter is to the composition of the private fine-tuning set** — which images, how many,
and their distribution — and then **map the geometry of the resulting population of adapters.**

**The unifying quantity = empirical per-record SENSITIVITY, normalized by the seed-noise floor.** This is
literally the differential-privacy notion (global sensitivity Δf = max change in the output from swapping
one record; Dwork-McSherry-Nissim-Smith 2006), measured on the *parameters* (the adapter) rather than on
predictions. It sits on one theoretical triangle: **sensitivity (Δ adapter) → memorization (Δ behavior,
Feldman-Zhang) → membership-inference (attacker-observable Δ).**

**Headline metric.** For a dataset change D → D′ (a one-image swap, a resize, a reweighting):
  ρ = E_seeds‖ΔW(D′) − ΔW(D)‖ / E_seeds‖ΔW_reseed(D) − ΔW(D)‖
- numerator = adapter change from the composition change; denominator = adapter change from *retraining
  the identical set with a different seed* (the noise floor).
- **ρ > 1 with the change outside the reseed CI** ⇒ the change leaves a real fingerprint in the adapter
  above training noise. **ρ ≈ 1** ⇒ indistinguishable from reseeding (privacy-good, important to know).
- Report it as an **effect size + a significance test** (is ‖ΔW_change‖ outside the distribution of the K
  reseed deltas?) — the f-INE template (arXiv:2510.10510).

**Five non-negotiable metric rules (from the literature + this repo's scars):**
1. **Measure ΔW = BA, never raw A or B.** LoRA factors are gauge-non-identifiable (BA invariant to
   B→BR, A→R⁻¹A) — raw-factor distances are pure gauge noise. Use the product ΔW, and/or a gauge-invariant
   subspace (Grassmann) / spectral distance.
2. **Report paired-seed (oracle) AND independent-seed (realistic).** Same seed for change-vs-baseline
   removes init variance = optimistic; independent seeds = the honest, harder test.
3. **DECOMPOSE the noise floor into TWO parts (M2 — the most important add; repo has HARD evidence).**
   GPU-atomic nondeterminism makes even *fixed-seed* re-runs differ here (this session: fashion-10class
   unrolls NaN run-to-run at fixed seed; FD gate flips). So measure **σ_repeat** (D fixed, seed fixed,
   re-run K×) SEPARATELY from **σ_reseed** (init varies). Report both + the ratio σ_repeat/σ_reseed. If
   σ_repeat is a non-trivial fraction of σ_reseed, **ρ≈1 is UNINTERPRETABLE** and the paired-seed "oracle"
   numerator is INVALID (same-seed no longer ⇒ same ΔW). Bitwise reproducibility on the L40S is NOT assumed.
   **σ_repeat must DROP non-finite (NaN/Inf) re-runs and report the dropped count (D3, mirror
   `estimate_sigma_seed`)** — a single NaN poisons σ_repeat and fakes a huge ratio.
4. **The readout for the swap/minority/OOD arms is a SLOPE, not ρ>1 (S3).** Swapping any 1-of-N image gives
   a ~1/N *mechanical* ΔW change regardless of atypicality, so ρ>1 holds trivially. The result is the
   (partial) **correlation/slope of ρ vs atypicality**, controlling for gradient norm (M1) — not ρ crossing 1.
5. **Phase-1 ρ = relative-Frobenius on ΔW=BA (the scalar the significance test needs) PLUS a SEPARATE
   principal-angle/subspace term** — do NOT collapse them (a swap can rotate col(ΔW) without moving
   Frobenius much). Full Bures–Wasserstein is Phase-2 only.

---

## PHASE 1 — the ablation battery (each arm = retrains → an adapter → ρ). Signed predictions from theory.

Retraining is CHEAP at our N-image / small-net / LoRA scale, so Phase-1 uses **actual swap-and-retrain
ground truth** — retrain IS the ground truth. **Skip DataInf for Phase-1 (S5):** it estimates *behavioral*
influence (Δ-loss), while ρ is *parametric* (Δ-weights), so they need not rank-correlate — a low LDS would
just mean the two quantities differ, not that either is wrong. Reintroduce DataInf only if behavioral
influence becomes a separate, explicitly-labeled deliverable.

| # | Ablation | What varies | Signed prediction (theory) | Novel? |
|---|---|---|---|---|
| A | **Swap-one vs typicality** | swap image i→i′; vary atypicality of i′ | ρ **↑ with atypicality** (rare/ambiguous/outlier imprint more) | parametric angle |
| B | **Subset size (dilution)** | N ∈ {2,4,8,16,32,64} | mean per-image ρ **~1/N** | **exact exponent is open** |
| C | **Class imbalance** | skew ratio (even → 8:1:1 → all-one-class) | minority per-image ρ **>** majority | yes (low-rank) |
| D | **Context-rarity (same image, vary CONTEXT — D1)** | ONE FIXED image as (a) 1-of-k same-class neighbors [typical] vs (b) lone minority among another class [rare] | the SAME image's ρ is **higher in the rare context** | yes — the clean headline |
| E | **Duplication** | dup an image k∈{1,2,4,8}× | imprint **grows with k**; LLMs super-linear | **linear vs super-linear in a low-rank adapter is open** |
| F | **Cross-dataset composition** | same (mnist) vs mixed (mnist+fashion) private set | mixing changes the sensitivity structure / subspace | yes |
| G | **OOD-style injection** | inject other-style digits (SVHN/USPS/EMNIST/printed-font) into an mnist set | the OOD (atypical) digit imprints **more** than an in-dist one | yes |
| **H** | **Triangle-closing validation (M3 — REQUIRED)** | same records as A/D | high-ρ records have a **higher leave-one-out mem score** (MIA secondary) | the gate |

- **Arm H is non-optional (M3).** ρ measures Δ-parameters; that this = "privacy leakage" is ASSUMED — the
  exact trap that just sank the reconstruction claim. One Phase-1 arm must empirically CLOSE the
  sensitivity→memorization→MIA triangle: does high-ρ predict a higher attacker-observable on the same
  records? If not, ρ is a parametric curiosity, not privacy. **H's PRIMARY metric = the leave-one-out
  Feldman-Zhang mem score (D2)** (does the model classify record i correctly when trained WITH vs WITHOUT
  it) — tractable at N=8–64, genuinely independent of Δ-parameters (it's *behavioral*), directly closes the
  triangle. Classical shadow-model / loss-threshold MIA is UNDERPOWERED at tiny N (too few members, huge
  variance, many models needed) → **secondary, only if feasible.**
- **Arm D is context-CONTROLLED (D1).** The naive "lone minority among a majority" conflates context-rarity
  with class/image IDENTITY — high ρ could be "sevens imprint more than threes," not "being the lone
  minority raises ρ." So hold the IMAGE FIXED and vary ONLY its context (typical same-class neighbors vs
  lone-minority-in-another-class); that isolates context-rarity and IS the long-tail claim. D is the
  *extreme of C* (imbalance) — frame it as "the cleanest instance of C," not a separate law.

- **A "typicality" score is required for A/D/G, and airtight independence is MAKE-OR-BREAK (M1).** k-NN
  density in θ₀'s *own* features is CIRCULAR: an image atypical to θ₀ produces large θ₀-gradients → large
  ‖ΔW‖ almost BY CONSTRUCTION, so "atypical→more imprint" would be a gradient-magnitude tautology, not
  memorization — and D (the headline) is where it bites hardest. BOTH required: **(a)** typicality from a
  **θ₀-INDEPENDENT extractor** (a *different* pretrained net, NEVER θ₀); **(b)** report ρ-vs-atypicality as a
  **PARTIAL correlation controlling for the swapped image's gradient norm ‖∇_image L(θ₀)‖**.
- Two genuinely NEW quantitative results LoRA can deliver (the literature is all full-model / LLM-scale):
  **(i) the dilution exponent in N (arm B); (ii) whether duplication is linear or super-linear in a
  low-rank update (arm E).**

**Controls / confounds the plan must bake in (each an assert or a measured column):**
- **Seed-noise floor — decomposed (M2) + its N-scaling measured (S2):** K reseeds per reference set (K≥~20,
  tied to a {K,2K} stability check) → the denominator + null distribution; report σ_repeat vs σ_reseed
  (rule 3). AND: the floor is itself N-dependent, so **measure the floor's N-scaling BEFORE claiming arm-B's
  1/N law** — else the dilution exponent is confounded by the denominator (same class as the
  ssim_norm-vs-raw-baseline error: always check what the normalizer does under the swept variable).
- **Raw-norm / scale confound:** control input norm; report ρ *relative* to ‖ΔW(D)‖ and, ideally, to the
  swapped image's gradient norm (a bigger-pixel image inflates ‖Δθ‖ for boring reasons).
- **Onion effect (memorization is relative):** measure against a FIXED reference set; removing one outlier
  can promote another, so per-image ρ is not additive/context-free.
- **Hold training depth + LoRA capacity FIXED across arms** (memorization scales with T and capacity —
  Carlini's law), or the composition effect is confounded by how hard each arm trained.
- **Rank interaction — you CAN'T get both clean attribution AND the novel low-rank laws (S1).** Clean
  per-image attribution needs **r≳N** (adapter≈full-FT, images separable) → run the signed-prediction arms
  **A/C/D/G at r≳N**. But the two NEW low-rank laws — **B (dilution exponent) and E (duplication scaling) —
  REQUIRE r<N**, because their novelty IS the entanglement of per-image contributions in the shared
  low-rank bottleneck (the superposition problem); at r≳N they're no longer "low-rank" results. So measure
  B/E at r<N with the entanglement acknowledged and the floor capturing the mixing. Don't let "fix r≳N"
  hide that the novelty lives at r<N. (This is exactly where our rank-sweep reversal lives, too.)
- **GPU nondeterminism** (this repo has been bitten by GPU-atomic NaN/nondeterminism): pin determinism or
  it inflates the reseed floor and fakes ρ≈1.

**SEQUENCING — the first spine (yoado-34):** run **arm B FIRST as the circularity-free opener** — it needs
NO typicality proxy (sidesteps the M1 trap entirely) and delivers a novel result (the 1/N dilution
exponent); its only guard is S2 (measure the floor's N-scaling first). B de-risks the core machinery (the
M2 floor decomposition + the N-scaling) on the most tractable arm. THEN ship **context-controlled D as the
headline** once M1's θ₀-independent typicality pipeline is validated on real data. (A/C/G follow; H gates
throughout.)

**Every arm SAVES its adapter (ΔW = BA per layer) to disk** — Phase 1's output IS Phase 2's input.

---

## PHASE 2 — adapter-space geometry (gated on having the population, ~hundreds of adapters)

Once Phase 1 has produced a few hundred adapters spanning the ablations, go unsupervised: **embed and
cluster them to see which are close / related / composed-of-what.**

- **Featurize each adapter (per layer ℓ), GL(r)-invariant:** `ΔWℓ = BℓAℓ`; thin-SVD → top-r left
  singular subspace `Uℓ` + singular spectrum `σℓ`. (Raw A/B/[A;B] are BROKEN — never use.) No git-rebasin
  needed: all adapters share the frozen base θ₀ → one common frame; only GL(r) is nuisance, killed by
  using ΔW.
- **Distance (pick one, both invariant):**
  (1) **Bures–Wasserstein — but TWO-SIDED (S4).** `Cℓ = ΔWℓΔWℓᵀ` (out×out) captures only the LEFT/output
      subspace + energy and DROPS the right/input subspace `ΔWᵀΔW` (in×in = *which input directions
      imprint*) — which for dataset-composition sensitivity likely matters MORE. So use the full ΔW
      distance, or BW on BOTH `C_out=ΔWΔWᵀ` and `C_in=ΔWᵀΔW` (arXiv:2302.14618). OR
  (2) **Grassmann (principal-angle) on col(BA) + spectral on σ** (two-axis, interpretable: "same
      directions" vs "same magnitude/rank"). Aggregate over layers (sum / product-manifold).
- **Cluster + visualize:** spectral or hierarchical (avg-linkage) clustering on the precomputed distance;
  UMAP/t-SNE with `metric='precomputed'`. **Never** Euclidean k-means on flattened raw params.
- **Composition/relatedness:** compute a shared r-dim subspace across the population (EigenLoRAx /
  Compress-then-Merge style) and express each adapter as coordinates in it — mixtures show up as
  near-linear combinations (task-arithmetic / model-soups give the "adapters add" prior).
- **The questions:** do adapters cluster by *composition* (dataset, class-balance, N, which images)?
  Which attribute (subspace vs spectrum vs rank) drives proximity? Are cross-dataset / mixed adapters
  linear combinations of the pure ones? Does ρ (Phase-1 sensitivity) correlate with position/spread in
  the space?

---

## Novelty (what's actually new)

- **Parametric, adapter-space sensitivity** — the influence/attribution literature (IF, TRAK, datamodels,
  Shapley) measures *behavioral* influence (Δ prediction); we measure *Δ adapter weights*. Under-studied.
- **Two open quantitative laws for low-rank updates:** dilution exponent in N; duplication linear vs
  super-linear.
- **A geometry/atlas of adapter-space** organized by dataset composition.

---

## Compute structure (WEXAC, bsub-only)

- Phase 1 is **many short retrains** (small net + LoRA, N-image sets) — cheap individually, but the count is
  **low THOUSANDS, not "hundreds" (Q6, be honest):** arm B alone ≈ 6 N-values × ~10 targets × 20 reseeds ≈
  1200 retrains; the full battery × K-reseed floor pushes several thousand. Cheap per-run, but budget
  walltime AND storage for the saved ΔW tensors explicitly. Each run saves metrics + ΔW=BA per layer.
- Phase 2 is a **single analysis job** over the saved adapter population.
- Datasets/bases: reuse the honest mnist/fashion bases; OOD-digit sources = SVHN / USPS / EMNIST / a
  printed-font digit generator (confirm availability/licensing).
- Reuse existing machinery: `_mnist_ctx` / unrolled LoRA training, the Σ_seed reseed-sampling (already
  built for the noise floor), the honest-θ0 loader. Determinism pinned.

---

## Open questions for the AUDITOR (before this ships to an executor)

1. **Metric:** Bures–Wasserstein (unified) vs Grassmann+spectral (interpretable) as the Phase-2 default —
   and for Phase-1 ρ, is relative-Frobenius on ΔW=BA + a subspace/cosine term the right pair, or do we
   need the full BW there too?
2. **Typicality proxy** for arms A/D/G — which independent measure (frozen-feature k-NN density? a held-out
   mem score? both)? This is the circularity trap.
3. **Reseed budget K** for a trustworthy floor (tie to the {S,2S}-style convergence check we already use).
4. **The N-grid and imbalance ratios** — exact values; and does arm B need r swept with N (r≳N) or fixed?
5. **Determinism** — can we get bitwise-reproducible LoRA training on the L40S, or must the floor absorb
   residual nondeterminism (and how to report it)?
6. **Phase-1 → Phase-2 handoff:** the exact saved-tensor schema (ΔW per layer + composition metadata) so
   the population is directly clusterable with no re-run.
7. **DataInf as fast predictor:** worth wiring as a cheap influence estimate validated against retrains
   (LDS), or skip and stay purely retrain-based for Phase 1?
8. **Scope/first spine:** which arm is the primary headline for a first batch (recommend D, the lone-rare
   minority-slot — cleanest signed prediction), with the rest as breadth?

## Grounding (arXiv)
Sensitivity/DP: Dwork-McSherry-Nissim-Smith 2006. Influence/attribution: Koh-Liang 1703.04730, TRAK
2303.14186, datamodels 2202.00622, DataInf 2310.00902, IF-fragility 2006.14651, f-INE 2510.10510.
Memorization: Feldman 1906.05271, Feldman-Zhang 2008.03703, Carlini 2202.07646, Kandpal 2202.06539,
onion 2206.10469; imbalance/MIA 1911.09777. Adapter geometry: LoRA 2106.09685, intrinsic-dim 2012.13255,
git-rebasin 2209.04836, task-arithmetic 2212.04089, model-soups 2203.05482, DWSNets 2301.12780, EigenLoRAx
2502.04700, Bures-Wasserstein 2302.14618. (Verify Kornblith-2019 CKA + Schürholt model-zoo IDs before citing.)

## BATTERY TRACKER (2026-08-27 — do NOT drop; run the WHOLE thing on the fixed metric)

Gate: fix whitened_metric to 3-way (in flight) -> generalize arm-B into a reusable runner
(composition-fn -> whitened_sensitivity: p-value + 3-way magnitude) -> then run EVERY item below.

PHASE-1 ARMS (composition changes):
- [~] B  subset size / dilution  -- DONE (detection flat in N; magnitude re-confirm pending 3-way metric)
- [x] E  duplication  -- DONE (job 162114): SUB-LINEAR β≈0.24, rank-INVARIANT (not low-rank-protective); ~15x more absolute leakage at r=32; caveat=at convergence, T-sweep is follow-up
- [ ] C  class imbalance  -- minority per-example sensitivity > majority?
- [ ] D  context-rarity (minority-slot)  -- SAME image, vary context (typical vs lone-minority); needs care
- [ ] A  swap-one vs typicality  -- needs a theta0-INDEPENDENT typicality proxy + gradient-norm partial-corr (M1)
- [ ] F  cross-dataset composition  -- mnist vs mixed mnist+fashion private set
- [ ] G  OOD-style injection  -- other-style digits (SVHN/USPS/EMNIST/printed-font) into an mnist set
- [ ] H  triangle validation (REQUIRED gate)  -- does high whitened-sensitivity predict higher leave-one-out
        Feldman-Zhang mem score / MIA? (else it's a parametric curiosity, not privacy)

CROSS-CUTTING AXES (cross the arms with these — the user's reminder, don't forget):
- [ ] RANK r  -- LoRA rank as a sensitivity axis (connect to the rank-sweep reversal)
- [ ] DATA TYPE  -- different base datasets/architectures for the adapter (mnist/fashion/flowers/cifar)
- [ ] BASE/MODEL SIZE  -- width/depth of the frozen base
- [ ] training-length/convergence  -- Carlini control (hold fixed OR sweep deliberately)

PHASE 2 (after ~hundreds of adapters accumulate across the arms):
- [ ] adapter-space atlas  -- embed/cluster ΔW=BA (subspace+spectrum, Bures-Wasserstein / Grassmann);
        do arms/axes cluster by composition? are mixed adapters linear combos of pure ones?

REPORTING RULE (all arms): primary = permutation p-value (robust); magnitude only via the 3-way metric
that passes K-convergence; small-eigenvalue floor mandatory; H gate before any "privacy" framing.

### Arm E design lock (2026-08-27, audit-hardened by yoado sub-agent — verdict FIX-FIRST, all folded)
- FIXED-PREVALENCE construction: k_max class-1 slots held constant for all k; copies_k = context +
  [T]*k + donors[k:k_max], distinct = context + donors[0:k_max]. Kills the minority→balanced confound.
- Distinct baseline IDENTICAL across all k & targets ⇒ Σ (metric denominator) FROZEN by construction
  (and trained once per rank). Kills the k-varying-denominator confound.
- NO analytic β=2 null (nonlinear BCE + low-rank bottleneck break "gradients sum⇒β=2"). EMPIRICAL null:
  β(sensitivity) at rank=8 (bottleneck, r<N) vs rank=32 (full, r>N). β(low)<β(high) ⇒ low-rank
  SATURATES duplication imprint = privacy-protective. Folds in the rank axis.
- Headline β on DEBIASED sensitivity (d2_obs−d2_null); β(d2_obs) diagnostic only. Drop-counters +
  finite-d² assert before fitting. Base is MNIST-train pretrained, targets MNIST-test ⇒ genuinely
  held-out (no target/base overlap). Files: experiments/dataset_sensitivity/arm_e_duplication.py,
  scripts/run_arm_e_{stage0,duplication}_wexac.sh.
