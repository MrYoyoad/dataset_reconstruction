# Literature Grounding: Gallery-Restricted Training-Set Recovery from a LoRA Adapter

**Date:** 2026-08-31
**Purpose:** Ground the proposed attack in prior work so the plan can cite honestly. Scope the novelty claim; list techniques to borrow and pitfalls to avoid.

**The attack (short form):** Attacker holds a released LoRA adapter (B, A; ΔW = BA) for a layer fine-tuned on N private images, and knows the images are a subset of a *known finite gallery* (e.g. MNIST test set). Goal is **selection, not pixel synthesis**: identify the exact N gallery items used. Method = (1) model ΔW ≈ Σᵢ gᵢxᵢᵀ, build a per-gallery-image "atom" gxᵀ through the frozen base; (2) greedy matching-pursuit selection against ΔW (iteration-1 correlation = a membership score); (3) verify by retraining an adapter on the selected set S and comparing ΔW(S) to target ΔW under a gauge-invariant subspace distance; (4) nudge via residual-guided discrete swaps.

> Bottom line up top: **the individual ingredients all exist in the literature; the specific *combination and framing* — greedy weight/gradient-matching pursuit over a *fixed known gallery*, closed as a discrete membership-set recovery with a retrain-consistency certificate and residual-guided swap refinement, applied to a *released LoRA adapter in the image domain* — I did not find published as a single method.** The closest single prior is the **SELECT** attack (Sec. 3), which does greedy gradient-matching selection from a fixed corpus against a weight difference, but for text/utility and without the retrain-verify + discrete-nudge loop or the LoRA/gallery framing. Treat the novelty as **"new composition + new threat framing," not "new primitive."**

---

## Area 1 — Matching pursuit / OMP / sparse subset recovery

The core mechanic (pick atoms from a dictionary that best explain a linear measurement, subtract, repeat) is exactly **Orthogonal Matching Pursuit** and its guarantees carry over directly if you cast ΔW as the measurement and the gallery atoms {gxᵀ} as the dictionary.

- **OMP / greedy sparse recovery** (Tropp; Donoho & Huo 2001; Cai & Wang). Recovery guarantees come in two flavors:
  - **Mutual coherence (MIP):** exact support recovery if `(2k−1)·μ < 1` where μ = max pairwise atom correlation and k = sparsity (here N). Cheap to compute for your dictionary — *use it as a go/no-go diagnostic.*
  - **RIP:** OMP recovers every k-sparse signal if `δ_{k+1} < 1/(1+√k)`; a counterexample exists at `δ_{k+1}=1/√k`. RIP is NP-hard to verify, so mutual coherence is the practical tool.
  ([OMP+MIP](https://liu.diva-portal.org/smash/get/diva2:1199565/FULLTEXT02.pdf), [RIP bound 1401.0578](https://arxiv.org/pdf/1401.0578))
- **Where greedy breaks — coherent dictionaries.** Near-duplicate gallery images ⇒ near-parallel atoms ⇒ high μ ⇒ OMP picks the wrong twin and the guarantees void. This is *the* structural risk for MNIST (many near-identical digits). Mitigations from the sparse-coding literature: block/group-OMP over duplicate clusters, or forward-backward pursuit that can un-pick.
- **Relation to the attack:** Steps 1–2 *are* OMP with a data-derived dictionary. This gives you (a) a coherence-based feasibility test before running anything, (b) a principled reason the residual-nudge (step 4) is needed (greedy OMP has no backtracking; your swap step adds it — essentially a forward-backward / CoSaMP-style correction).

**Adjacent, very close prior — OMP for data selection:** **GradMatch** (Killamsetty et al., ICML 2021) selects a coreset by minimizing gradient-reconstruction error via **OMP** — the identical "select a subset whose summed gradients match a target vector" primitive, but for coreset training utility, not privacy/membership. Cite it as the algorithmic precedent for OMP-over-gradient-atoms.

## Area 2 — Gradient inversion / leakage, and how they handle N>1 mixing

All of these invert *gradients*; your ΔW = BA is a compressed multi-step gradient sum, so they are the direct methodological neighbors. The key axis is **how they de-mix N samples** — your answer (a known gallery turns de-mixing into discrete selection) is the differentiator.

- **Inverting Gradients** (Geiping et al., NeurIPS 2020) and **GradInversion** (Yin et al., CVPR 2021): optimize pixels to match the observed gradient (cosine / L2 + TV/BN priors). N>1 handled poorly — they recover a blurry superposition; GradInversion adds a group-consistency + BN-statistics prior. *Generative, not selective.*
- **Cocktail Party Attack (CPA)** (Kariyappa et al., ICML 2023, [2209.05578](https://arxiv.org/pdf/2209.05578)): frames FC-layer gradient inversion as **blind source separation** and runs **ICA** to separate up to **1024** mixed inputs from an aggregated gradient. This is the reference method for the superposition/N>1 problem and directly relevant to un-mixing Σᵢ gᵢxᵢᵀ *before* or *instead of* selection. Borrowable as a de-mixing front-end.
- **SPEAR** (Dimitrov et al., NeurIPS 2024, [2403.03945](https://arxiv.org/pdf/2403.03945)): **exact** batch recovery for FC+ReLU, exploiting the low-rank gradient structure (SVD) + **ReLU-induced sparsity** to filter candidate directions; exact up to batch b<25, scales to ImageNet dims. **SPEAR++** (Bakarsky et al., NeurIPS 2025 workshop, [2510.24200](https://arxiv.org/pdf/2510.24200)) recasts it as **sparsely-used dictionary learning**, ~10× larger batches, robust to DP noise / FedAvg. These are the most rigorous "exact recovery conditions" references — borrow the low-rank + sparsity filtering ideas and the dictionary-learning framing (your gallery *is* a fixed dictionary; theirs is learned).
- **ARES** (2026, [2603.17623](https://arxiv.org/pdf/2603.17623)): activation-recovery gradient inversion scaling to batch 384. **ReCIT** ([2504.20570](https://arxiv.org/abs/2504.20570)) and **PEFTLeak/MineGrad** (Sami et al.; [2506.04453](https://arxiv.org/abs/2506.04453), MineGrad AISTATS 2026): PEFT/LoRA-specific gradient inversion — **but all rely on a MALICIOUS server** that poisons the base model / adapter init to disentangle samples. Your attack is **passive/honest-but-curious** (no poisoning), which is a *harder and more honest* threat model — state this contrast explicitly; it is a genuine differentiator versus the LoRA-gradient-inversion line.

## Area 3 — Closed-world / candidate-set / gallery-restricted reconstruction (the novelty crux)

This is where the claim lives or dies. Findings:

- **Nobody I found does exactly this for LoRA/images**, but the *idea* that "MIA over a known candidate pool ⇒ reconstruction-by-selection" is explicitly articulated as a threat model in **SoK: Data Reconstruction Attacks** ([2506.07888](https://arxiv.org/html/2506.07888)): "MIA could aid data reconstruction if the adversary has a candidate dataset containing all target samples and can perfectly predict membership … MIA is a decision problem, reconstruction is a search problem." So the *framing* is known-as-a-concept; the *instantiated attack* is what you'd contribute.
- **SELECT — "Approximating Language Model Training Data from Weights"** (Zaman et al., [2506.15553](https://arxiv.org/html/2506.15553)) — **THE CLOSEST PRIOR.** Explicitly "constrain the problem to data *selection instead of generation*: given a large corpus, search for a small set of datapoints that, after training, produce a model close to the final model." Uses **greedy gradient-matching** against the weight difference `∑∇ℓ(x;θ₀)·(θf−θ₀)`, submodular near-optimality, JL dimensionality reduction, synthetic interpolated checkpoints. **This is your steps 1–2 for text.** How you differ: (a) image domain + **LoRA adapter** as the measurement (not full-weight diff); (b) goal is **exact membership-set identity**, not utility approximation; (c) you add **retrain-verify (step 3)** and **residual-guided discrete swap (step 4)** — SELECT stops at greedy selection; (d) your gauge-invariant subspace distance is LoRA-specific (init-direction invariance). Cite SELECT as the primary prior and position your work as the *privacy-attack, LoRA, exact-subset, verified* counterpart.
- **WARP: Weight-Space Analysis for Recovering Training Data Portfolios** ([2607.01686](https://arxiv.org/pdf/2607.01686)) — reverse-engineers the training *mixture* from weight-space geometry; a "which-data" recovery but at distribution/portfolio granularity, not exact instance selection.
- **Dataset Size Recovery from LoRA Weights** (DSiRe, [2406.19395](https://arxiv.org/html/2406.19395v1)) — recovers *N* (the count) from LoRA spectrum. Complementary: it gives you the **N** your greedy loop needs as a stopping criterion; cite as a helper, not a competitor.
- **Haim et al.** (NeurIPS 2022, [2206.07758](https://arxiv.org/abs/2206.07758)) and multiclass follow-up ([2305.03350]) — open-world pixel synthesis from KKT/implicit bias. Your attack is the **closed-world discrete dual**: instead of solving for pixels, you *select* pre-existing pixels. Frame it as "Haim-style leakage, but the search space is a finite gallery, which converts an ill-posed continuous inverse problem into a combinatorial one with verifiable answers."
- **Yao 2024, Risks When Sharing LoRA Fine-Tuned Diffusion Weights** ([2409.08482](https://arxiv.org/abs/2409.08482)) — closest *passive-LoRA-vision* competitor: a learned VAE maps LoRA weights → reconstructed private images. **Generative, identity-level, not selective, not from a gallery.** Your selection framing is orthogonal and arguably a cleaner attack when a gallery exists.

## Area 4 — Membership inference → set reconstruction; retrain/consistency verification

- **Rank-all-candidates-then-refine-jointly** is an established *shape* but not with your exact loop. Shadow-model MIA (Shokri et al. 2017) ranks candidates by a membership score; recent work does **joint** membership inference over many queries (approximate Gibbs sampling, anchor-conditioned shadow retraining) rather than per-sample independent decisions — matches your "iteration-1 score, then joint refinement" intuition.
- **Verify-by-retraining** appears in two nearby lines:
  - **Reconstruction Attacks on Machine Unlearning** (Bertran et al., NeurIPS 2024) — uses *before/after* model differences (analogous to your ΔW) to reconstruct the forgotten point; simple models provably vulnerable. Conceptually a "diff-of-weights ⇒ recover the responsible sample" attack, same spirit as your atom-matching.
  - **"No More Guessing: a Verifiable Gradient Inversion Attack"** (Diana et al., [2604.15063](https://arxiv.org/abs/2604.15063)) — provides an **explicit certificate of correctness** for a reconstruction, motivated by the fact that gradient inversion normally has "no intrinsic way to certify success." **This is the published precedent for your step-3 retrain-verify certificate** — cite it; your subspace-distance-after-retrain is a certificate in the same sense, specialized to LoRA gauge invariance.
- **Net:** "MIA score → joint set refinement → retrain-consistency check" is assembled from known pieces; the *closed loop over a gallery with a LoRA-gauge-invariant certificate* is the assembly you contribute.

## Area 5 — Direct weight inversion (the continuous parent of this attack)

- The proposed attack is the **discrete (gallery-restricted) special case of direct weight inversion**: minimize ‖θ_T − F(θ₀, x̂)‖² over data. Continuous version = the thesis's own primary axis (`notes/thesis_update_briefing.md`).
- **SELECT** ([2506.15553]) again the closest published instance of the *selection* variant; **GradMatch** the OMP-based coreset variant; **"Recovering the Pre-Fine-Tuning Weights" / Spectral DeTuning** ([2402.10208](https://arxiv.org/abs/2402.10208)) inverts LoRA weights but recovers θ₀, not the data — adjacent weight-space inversion.
- Framing to use: continuous direct-inversion is the leakage *upper bound* (unrestricted x̂); the gallery restriction both (a) makes the inverse **well-posed / combinatorial** and (b) yields **certifiable exact answers** (you can check membership, unlike a fuzzy pixel reconstruction). That is the honest value proposition.

---

## NOVELTY VERDICT

**Novel as a composition and as a threat framing; NOT novel in any single primitive.** No published method matches "greedy weight-matching pursuit over a *known gallery* + retrain-verify certificate + residual-guided discrete swap, on a *released LoRA adapter*, for *exact training-subset identification* in the image domain."

- The **greedy-gradient-matching-selection-from-a-fixed-pool** core is **already published (SELECT, 2506.15553; GradMatch OMP)** — do **not** claim it as new. Claim the *LoRA measurement*, the *exact-membership-set goal*, the *retrain/gauge-invariant certificate*, and the *residual-nudge backtracking* as the contribution, and the *closed-world-selection framing of LoRA leakage* as the conceptual contribution.
- The **passive (non-malicious) LoRA threat model** genuinely separates you from ReCIT/MineGrad/PEFTLeak (all malicious-server).
- **Risk to the claim:** a reviewer who knows SELECT will say "this is SELECT for images with LoRA + a verify step." Pre-empt by (i) citing SELECT prominently, (ii) foregrounding the verify+nudge loop and the gallery-coherence analysis (which SELECT lacks), (iii) reporting *exact-subset* accuracy (a metric SELECT does not target).

## TECHNIQUES TO BORROW

1. **Mutual-coherence feasibility test** (Area 1): compute μ over gallery atoms up front; predict when greedy will fail (near-duplicate digits). Cheap, principled, and a paper-worthy diagnostic.
2. **Forward-backward / CoSaMP-style backtracking** to justify and strengthen step 4 (greedy OMP cannot un-pick; your swap step is exactly the fix — frame it in pursuit-algorithm language, not ad hoc).
3. **ICA de-mixing front-end (Cocktail Party Attack)** for the Σᵢ gᵢxᵢᵀ superposition when N is large — separate sources first, then match separated components to gallery atoms.
4. **SPEAR/SPEAR++ low-rank + ReLU-sparsity filtering and dictionary-learning framing** — your gallery is a *fixed* dictionary; import their exact-recovery conditions and DP-robustness analysis.
5. **DSiRe** to estimate **N** (stopping criterion for the greedy loop) directly from the LoRA spectrum.
6. **Verifiable-inversion certificate ("No More Guessing")** — formalize step 3 as a certificate; report a certified-correct rate, not just an SSIM/accuracy.
7. **SELECT's efficiency tricks** — final-layer-only gradients + JL random projection to store per-gallery atoms cheaply at 10k-gallery scale.

## PITFALLS

1. **Dictionary coherence = the dominant failure mode.** MNIST has many near-identical images; high μ voids OMP guarantees and greedy picks the wrong twin. Report coherence and expect degraded exact-subset accuracy on near-duplicate-heavy galleries. This is the single biggest technical risk.
2. **Multi-step / accumulated ΔW ≠ single sum of clean outer products.** ΔW = BA aggregates many SGD steps with evolving activations; the "atom = gxᵀ through the *frozen base*" approximation degrades as fine-tuning moves the activations (the anchor-α issue the thesis already studies). The atom must be computed at a representative anchor, and error grows with T.
3. **LoRA gauge/init invariance.** ΔW = BA is only defined up to the shared init direction; a naive Frobenius ‖ΔW − ΔW(S)‖ mis-scores. You already plan a gauge-invariant subspace distance — verify it is invariant to the *sign/scale/rotation* ambiguity of the B,A factorization, not just to init.
4. **Greedy has no optimality guarantee under coherence** — budget for the nudge/local-search to matter, and note it can still land in local minima (report failure cases per the project's "show best AND worst" rule).
5. **N>1 superposition is not solved by selection alone** — if two gallery atoms jointly mimic a third, matching pursuit can pick a spurious pair. Cross-check with an ICA/SPEAR de-mixing pass.
6. **Retrain-verify is stochastic** — retraining an adapter on S has its own seed/optimizer noise; the certificate threshold must tolerate that, and you must retrain under the *same recipe* (the project's equalized-reference-budget lesson: cross-recipe comparison is unfair).
7. **Don't overclaim passivity as strictly stronger** — malicious-server attacks (MineGrad) recover *more* per the memory's "leakage bounds the weakest attacker" note; your passive result is a *lower bound* on leakage, not the reconstruction ceiling. State it that way.

## Papers I could NOT fully verify / flag

- Exact RIP/coherence constants quoted from search summaries, not re-derived — verify against Cai & Wang / Tropp before putting a specific bound in the thesis.
- **ARES** (2603.17623) details are from the CLAUDE.md summary + search title only (activation recovery, batch 384); did not fetch full text.
- The "joint MIA via Gibbs / anchor-conditioned shadow retraining" claim is from a search summary; find the specific paper before citing it as the joint-refinement precedent.
- Did not locate any paper doing *exact-subset selection over a fixed image gallery from a LoRA adapter* — absence-of-evidence, so search once more with domain-specific terms before asserting first-of-kind in writing.
