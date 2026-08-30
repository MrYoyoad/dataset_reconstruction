# Gallery-Restricted Training-Set Recovery — Plan

**Status: PLANNING (draft, pre-litreview-fold).** A closed-world attack that recovers the EXACT subset of a
KNOWN gallery used to fine-tune a LoRA adapter — turning the failed pixel-reconstruction into a tractable
discrete SELECTION. Compute gated on the user's go. Observe-framed; scope stated as "closed-world,
this-attacker" on every number. Literature grounding: notes/gallery_recovery_litreview.md (in progress).

## 0. Threat model (state it prominently — it is the whole game)
Attacker holds the released LoRA adapter (B, A; ΔW = BA) for one fine-tuned layer, and KNOWS the N private
images were drawn from a KNOWN finite gallery G (e.g. MNIST test, ~10k). Goal: recover the EXACT N-subset
S* ⊂ G that was used. This is RECOVERY of the set (stronger than the per-image MIA detection), NOT pixel
reconstruction. The closed-world (known-gallery) assumption is STRONG and must be flagged on every claim —
it is realistic only when the fine-tuning data is a subset of a known public pool.

## 1. Why closed-world makes it tractable (the core idea)
Continuous pixel reconstruction failed (the dropped 0/40) because 784 free dims/image is huge and
under-constrained. Restricting to a gallery turns reconstruction into SELECTION over C(|G|, N) discrete
options; a rank-r adapter (1000×784 numbers) MASSIVELY over-determines a choice of N≈4 items. The intrinsic
difficulty collapses from "invert an image" to "pick N atoms from a dictionary."

## 2. Method — greedy weight-matching + retrain-verify + residual-nudge
ΔW = BA ≈ Σᵢ∈S* **gᵢ xᵢᵀ** (sum of per-image gradient outer-products at the training anchor). Each gallery
image x has a computable ATOM a(x)=g(x)·xᵀ (g(x) = backprop signal through the FROZEN base at x, given its
label). The task: find S ⊂ G, |S|=N, whose atoms best reconstruct ΔW.
- **(a) Warm-start ranking** — correlate every atom a(x) with ΔW (this IS the C membership score = MP
  iteration 1). Keep a shortlist (top-K).
- **(b) Greedy matching-pursuit select** — pick best atom, subtract its projection from ΔW, correlate the
  RESIDUAL, pick next, repeat N times → candidate S₀. (Orthogonal MP = re-fit coefficients each step.)
- **(c) Retrain-verify** — train a fresh adapter on S₀, compare ΔW(S₀) to target ΔW via a GAUGE-INVARIANT
  SUBSPACE distance (ΔW is init-invariant in DIRECTION, not exact value — atlas finding). Low distance ⇒
  consistent ⇒ likely S*.
- **(d) Residual-nudge (discrete local search)** — if inconsistent, the residual ΔW − ΔW(S) says what is
  missing/extra: swap the guessed image contributing LEAST for the gallery image whose atom best aligns with
  the residual; iterate to a local min. Beam search / restarts to escape local minima.

## 3. Why it should work + the honest subtleties
- **Over-determination** ⇒ S* should be the UNIQUE consistent subset (sharp verify minimum). PRE-TEST this
  with a plant-and-recover oracle sanity check (§5.0).
- **The init/gauge subtlety** — retraining with a different random init yields a different exact ΔW even for
  the same data, so verify MUST use the gauge-invariant subspace distance, NOT raw ‖ΔW − ΔW(S)‖. (Directly
  our clustering result: init ARI≈0.05.)
- **Warm-start is weak alone** — at MIA AUC 0.861, a true member ranks ~1400th against 10k negatives, so
  naive top-N MISSES members. The JOINT residual-guided search does the real work; the marginal score only
  seeds it. Do NOT claim the ranking alone recovers the set.
- **Free-coefficient vs oracle (user's standing rule)** — the REALISTIC attack must not assume the training
  init/recipe. Oracle-init verify = UPPER BOUND only; the realistic verify uses subspace-invariant matching
  and unknown init. Report both, labelled.

## 4. Literature grounding (fold from litreview when it lands)
Matching pursuit / OMP (sparse selection from a linear measurement; RIP / mutual-coherence guarantees, and
where greedy breaks = coherent atoms = near-duplicate images). Gradient inversion: Inverting Gradients,
GradInversion; EXACT batch recovery SPEAR (SVD+ReLU sparsity), Cocktail Party (ICA N-source separation),
ARES, ReCIT — for the N>1 superposition. Direct weight inversion (‖θ_T−F(θ₀,x̂)‖²) — this is its DISCRETE,
gallery-restricted analogue. Haim et al. (dataset reconstruction) as the base. R2F / Yao 2024 / MineGrad for
LoRA-specific leakage. NOVELTY claim to check hard: "greedy weight-matching + retrain-verify + residual-nudge
over a KNOWN gallery" as a closed-world SELECTION attack.

## 5. Baselines, ablations, pre-registered success criteria
- **§5.0 plant-and-recover oracle sanity FIRST:** on a KNOWN planted subset, does the pipeline recover it
  (same init)? If not, the method is broken before any realism. Gate everything on this.
- **Baselines:** random selection (chance exact-recovery ≈ 1/C(|G|,N), ~0); MIA-top-N (marginal only);
  MP-only (no nudge); MP+nudge (full). Report the LADDER.
- **Metrics:** EXACT-set recovery rate (all N correct); partial recovery = Jaccard(S, S*) / precision@N;
  as functions of N, gallery size |G|, and rank r.
- **Pre-registered REAL result:** exact-set recovery ≫ random AND ≫ MIA-top-N at N=4, |G|≥1000, realistic
  (unknown-init) verify. NULL if the nudge adds nothing over the marginal ranking, or only the oracle-init
  version works (then it is an upper bound, not an attack).

## 6. Gates (load-bearing; the ones that bit us before)
1. **Gauge-clean subspace verify** — never raw ΔW equality (init frame). 2. **Plant-and-recover sanity** before
any real claim (§5.0). 3. **Random + MIA-top-N baselines** on every recovery number (exact-recovery has a tiny
chance rate — quote it). 4. **Free-coefficient/realistic vs oracle** labelled separately; oracle = upper bound
only. 5. **Closed-world assumption** stated on every claim; this is RECOVERY (set), be careful not to imply
pixel reconstruction. 6. **Near-duplicate / coherence check** — report gallery mutual-coherence; greedy MP is
known to fail on coherent atoms, so a wrong pick may be a lookalike (tie to the A resolution result).
7. **N-superposition honesty** — as N grows the Σgxᵀ mixing degrades attribution (cite SPEAR/Cocktail-Party
limits); sweep N and report where it breaks. 8. Observe-framed, this-attacker (stronger than weakest — it
uses the gallery + gradient atoms), no "confirmed."

## 7. Failure modes to characterize (not hide)
Near-duplicate gallery lookalikes (coherent dictionary); large-N superposition; greedy local minima (→ beam /
restarts); recipe knowledge for verify (subspace-invariance buys robustness — test cross-recipe verify, per
B1); anchor/linearization error in the atom g(x)·xᵀ (single-step vs T-step accumulation — tie to the anchor
α-sweep + R2F single-step-decoder issue).

## 8. Sequence, roles, compute
Order: (0) plant-and-recover oracle sanity on a small gallery (|G|=200, N=4) → (1) MP-select + subspace-verify,
add the residual-nudge, the baseline ladder → (2) sweep N, |G|, rank; near-duplicate stress → (3) realistic
(unknown-init, cross-recipe) verify = the honest attack. Each: build/reuse a gallery + adapter zoo (bsub GPU),
analysis (bsub CPU), save tensors + a figure (recovery-rate curves, the ladder). Roles as before: this session
specs/co-drafts, **auditor** adversarially reviews (gauge/baseline/oracle-vs-realistic/coherence/N-mixing),
**executer** builds/runs. Compute-gated on the user's go per stage.
