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

## 1.5 Prior work & novelty (litreview: notes/gallery_recovery_litreview.md) — DO NOT overclaim
- **SELECT (Zaman et al., arXiv:2506.15553)** is the CLOSEST prior: greedy GRADIENT-MATCHING to select a
  subset from a fixed corpus that reproduces a finetuned model — essentially our steps (a)+(b), but for TEXT,
  full-weight diff, UTILITY (not privacy), and with NO retrain-verify, NO nudge, NO LoRA/gallery/exact-
  membership framing. **Cite it prominently** — a reviewer who knows it will otherwise call this "SELECT for
  images + a verify step."
- **GradMatch (Killamsetty, ICML 2021)** = OMP over gradient atoms (coreset selection) — the algorithmic
  precedent, not privacy-framed. **VGIA "No More Guessing" (2604.15063)** = the retrain/consistency-certificate
  precedent (our step c). **SoK: Data Reconstruction (2506.07888)** already states the CONCEPT "MIA over a
  known candidate pool ⇒ reconstruction-by-selection" — so the idea exists; the instantiation does not.
- **NOVELTY = the COMPOSITION + threat framing, not any single primitive:** LoRA adapter as the measurement;
  EXACT-subset-membership goal (SELECT targets utility, not this metric); the LoRA-gauge-invariant retrain
  CERTIFICATE; the residual-nudge backtracking; and closed-world SELECTION as a PASSIVE LoRA-leakage attack
  (vs the malicious-server ReCIT/MineGrad/PEFTLeak). Passive is a LOWER bound on leakage, NOT the ceiling —
  don't claim it as strictly stronger. One more domain-specific search before writing "first to select an
  image subset from a LoRA adapter."

## GO / NO-GO — three pre-tests, thresholds PRE-REGISTERED (auditor yoado-d4). No build unless all clear.
**Compute note (auditor correction):** these are NOT all CPU-on-existing-data. #1 and #3 run on existing
tensors + a cheap atom computation; **#2 needs one TINY GPU run** — the atlas has NO single-swap adapter (its
compositions differ by whole digit-subsets; the same-digits zoo never ran), so build S*(N=4)×K seeds +
one-swap×K seeds (minutes) and compute both the subspace distance and the whitened d². The arm-B/similarity-
ladder stacks have single-swap data but at N=16 with whitened d² only — usable as a fallback, wrong regime.

1. **LoRA-frame warm-start (BLOCKER 1).** Member rank under (i) full-gradient atoms g(x)xᵀ vs (ii) one-image-
   LoRA-adapter atoms in the gauge-invariant subspace. **PASS = LoRA-frame median true-member rank ≤ 10% of
   |G| AND ≤ ½ the full-gradient median rank** (frame is the problem, not the marginal score). (The C AUC 0.861
   used the full-weight ‖ΔW·(x−μ)‖ — NOT this ranking.)
2. **Verify resolution gate (BLOCKER 2).** One retrain = one seed-cloud draw. **PASS = median D(S*, ONE-swap) /
   D(S*, S*-reseeded) ≥ 3× at |S|=4** (subspace distance, over K seed pairs). If <3×, switch verify to the
   WHITENED d² with K retrains and PASS = its separation is significant (arm-B-style; report Cohen's d).
   **Also report on the first figure the seed-cloud spread D(S*, S*-reseeded) over K pairs — the NOISE FLOOR
   every later "consistent" verdict is measured against.**
3. **Identifiability / coherence (necessary, not sufficient).** Compute mutual coherence μ of the CORRECTLY-
   FRAMED (LoRA) dictionary (OMP needs μ<1/(2N−1); MNIST same-digit atoms ~0.8, so greedy WILL grab lookalikes).
   **PASS = plant-and-recover exact-rate ≥ 0.80 at LOW coherence (oracle-init, best case)**; report the
   exact-rate-vs-μ curve (high-coherence degradation is CHARACTERIZATION, not a fail — but if even low-coherence
   oracle plant-and-recover is <0.8, the method is dead). Pre-register the CLASS-MATCHED-random baseline (§5).

## 2. Method — greedy SUBSPACE-matching + retrain-verify + residual-nudge
**BLOCKER-1 FIX (auditor yoado-d4): the atoms must be in the LoRA frame, not the full-weight frame.**
ΔW=BA≈Σ gᵢxᵢᵀ is the FULL-weight one-step update. A LoRA step exposes P_LoRA(H)=B₀B₀ᵀH+HA₀ᵀA₀
(thesis_note_v2.md:26), and here A₀=0 / B₀ random, so the adapter sees the gradient through the COLUMN SPACE
of a RANDOM B₀ (unknown to a realistic attacker), and over T steps B,A co-evolve. So correlating full-gradient
atoms g(x)xᵀ against ΔW measures the WRONG object — likely why the marginal warm-start looked weak.
- **ATOM(x) = a one-image LoRA adapter** trained on {x} with the ATTACKER'S OWN init, represented by its
  gauge-invariant SUBSPACE (col/row space of ΔW_x). Matching pursuit runs over SUBSPACES, compared with the
  gauge-invariant subspace distance we already use — NOT raw matrix correlation.
- **(a) Warm-start ranking** — subspace-align each atom to the target ΔW's subspace; rank. (Re-measures the
  membership score in the CORRECT frame — the C AUC 0.861 (job 203683) used the full-weight ‖ΔW·(x−μ)‖ score,
  so it is NOT this ranking; re-measure first, §GO-NO-GO.)
- **(b) Greedy MP over subspaces** — select best-aligned atom, deflate the target subspace by its component,
  re-align the RESIDUAL subspace, repeat N times → candidate S₀.
- **(c) Retrain-verify (3 ARMS, headline = realistic)** — train an adapter on S₀ and compare to the target by
  a gauge-invariant distance, in THREE arms: ORACLE-init (attacker knows the victim's init/recipe = UPPER
  BOUND only) / REALISTIC (K independent attacker inits, whitened d² over the seed cloud) / CROSS-RECIPE
  (different activation/lr). Pre-register the REALISTIC arm as the headline.
- **(d) Residual-nudge (discrete local search)** — swap the guessed image contributing LEAST for the gallery
  image whose atom best reduces the residual subspace; iterate to a local min; beam search / restarts.
See §GO-NO-GO: the verify metric must first be shown to RESOLVE a single swap above the seed cloud.

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

## 4. Techniques to borrow (litreview: notes/gallery_recovery_litreview.md)
1. **Mutual-coherence feasibility test** on gallery atoms — predicts greedy failure, cheap, paper-worthy
   diagnostic (the go/no-go #3). 2. **Forward-backward / CoSaMP backtracking** to frame the nudge in
   pursuit-algorithm language (plain OMP can't un-pick a wrong atom). 3. **ICA de-mixing (Cocktail Party)** +
   **SPEAR/SPEAR++** dictionary-learning for the Σgxᵀ superposition front-end. 4. **DSiRe (2406.19395)** to
   estimate N from the LoRA spectrum (MP stopping criterion / unknown-N case). 5. **Verifiable-certificate**
   formalization (VGIA) for step c — report a CERTIFIED-correct rate, not just an accuracy. Unverified/flagged
   in the note: exact RIP/coherence constants (re-derive before citing); "first to select an image subset from
   a LoRA adapter" needs one more domain search before writing.

## 5. Baselines, ablations, pre-registered success criteria
- **§5.0 plant-and-recover oracle sanity FIRST:** on a KNOWN planted subset, does the pipeline recover it
  (same init)? If not, the method is broken before any realism. Gate everything on this.
- **Baselines (LADDER):** unconstrained-random (chance exact ≈ 1/C(|G|,N), ~0); **CLASS-MATCHED random — the
  one that will bite (auditor):** subsets with the SAME digit labels as S*. Content-level recovery is already
  easy (atlas +0.989), so exact-set rates vs *unconstrained* random are INFLATED by class composition; the
  honest denominator is class-matched random. Then MIA-top-N (marginal); MP-only (no nudge); MP+nudge (full).
- **Metrics:** EXACT-set recovery rate (all N correct); Jaccard(S,S*)/precision@N; **plus a "RIGHT DIGIT, WRONG
  EXEMPLAR" column** (did we get the class right but the specific image wrong — the coherence/lookalike
  failure) — as functions of N, |G|, rank r, and gallery coherence μ.
- **Does the attacker know N?** State it. If not, estimate N from the LoRA spectrum (DSiRe, arXiv:2406.19395)
  as the MP stopping criterion, and report recovery under estimated-N too.
- **Pre-registered REAL result:** exact-set recovery ≫ random AND ≫ MIA-top-N at N=4, |G|≥1000, realistic
  (unknown-init) verify. NULL if the nudge adds nothing over the marginal ranking, or only the oracle-init
  version works (then it is an upper bound, not an attack).

## 6. Gates (load-bearing; the ones that bit us before)
1. **Gauge-clean subspace verify** — never raw ‖ΔW−ΔW(S)‖: invariant to the B,A sign/scale/rotation
FACTORIZATION freedom AND to init, not just init (litreview pitfall 3). 2. **Plant-and-recover sanity** before
any real claim (§5.0). 3. **Class-matched-random + MIA-top-N baselines** on every recovery number (§5). 4.
**Free-coefficient/realistic vs oracle** labelled separately; oracle = upper bound only. 5. **Closed-world
assumption** on every claim; RECOVERY (set), not pixel reconstruction; passive = LOWER bound, not the ceiling.
6. **Near-duplicate / mutual-coherence check** — report gallery μ; greedy fails on coherent atoms (μ<1/(2N−1)),
a wrong pick is a lookalike (tie to the "right-digit-wrong-exemplar" column + A resolution). 7.
**N-superposition honesty** — Σgxᵀ mixing degrades with N (two atoms can mimic a third); cite SPEAR /
Cocktail-Party, cross-check with ICA de-mixing; sweep N, report where it breaks. 8. **3-ARM verify + Q2
structural controls (auditor):** every verify in oracle/realistic-K/cross-recipe; NULL-VERIFY control (oracle
must separate S*-reseeded from best-wrong-S while realistic cannot, if init leaks); SEED-SWAP sanity (realistic
unchanged, oracle drops — if oracle doesn't drop, init leaks via another path); report per-image residual
contributions under both arms. 9. **Multi-step anchor** — the frozen-base atom degrades as activations drift
over T (anchor-α); compute atoms at a representative anchor, report the error. 10. Observe-framed, this-attacker
(uses gallery + atoms — stronger than weakest), no "confirmed."

## 7. Failure modes to characterize (not hide)
Near-duplicate gallery lookalikes (coherent dictionary); large-N superposition; greedy local minima (→ beam /
restarts); recipe knowledge for verify (subspace-invariance buys robustness — test cross-recipe verify, per
B1); anchor/linearization error in the atom g(x)·xᵀ (single-step vs T-step accumulation — tie to the anchor
α-sweep + R2F single-step-decoder issue).

## 8. Sequence, roles, compute
**PHASE 0 — GO/NO-GO on EXISTING data (CPU, atlas tensors, no new zoo, no user-go needed beyond planning):**
the three pre-tests — (1) LoRA-frame vs full-gradient warm-start, (2) verify resolution gate (≥3× or whitened
d²), (3) coherence μ + class-matched baseline pre-registered + plant-and-recover. **No build proceeds unless
all three clear.** THEN, compute-gated on the user's go: (1) MP-subspace-select + 3-arm verify + residual-nudge
+ the baseline ladder on |G|=200,N=4 → (2) sweep N, |G|, rank, coherence; near-duplicate stress → (3) realistic
(unknown-init, cross-recipe) verify = the honest headline attack. Each stage: build/reuse a gallery + adapter
zoo (bsub GPU), analysis (bsub CPU), save tensors + figures (recovery-rate ladder, coherence curve). Roles:
this session specs/co-drafts, **auditor** adversarially reviews (gauge/baseline/oracle-vs-realistic/coherence/
N-mixing), **executer** builds/runs. Cite SELECT prominently in any writeup.
