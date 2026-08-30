# Harder-Identification Plan — kick out the crutches that made instance-1.000 "not impressive"

**Status: PLANNING.** Three experiments that stay in the identification/clustering direction but each REMOVE a
crutch that made the existence-floor result near-tautological. Compute gated on the user's go. Observe-framed,
weakest-attacker (adapter-only, per-attribute, prior-free = a LOWER bound, not the reconstruction limit). Still
small-scale (MNIST-MLP, rank 8) — this hardens the *identification* claim; it is NOT reconstruction.

## 0. What we have and what's weak
Instance recovery = 1.000, but it leans on: (i) CLOSED-SET (you hold the candidates), (ii) DISJOINT /
max-separable candidate sets, (iii) generalize across INIT only (victim's recipe known), (iv) tiny N. Content
recovery +0.989 is even softer (different digits ⇒ different ΔW). The three experiments each remove one crutch.
Mechanism throughout: ΔW = BA = Σᵢ gᵢ xᵢᵀ — a rank-r recording whose SUBSPACE/DIRECTION fingerprints which
images shaped it (gauge-clean ΔW, not raw B,A).

## A. RESOLUTION LIMIT — similar candidates (removes crutch ii: max-separability)
The sharpest "how fine-grained is the fingerprint" test. M candidate sets built as SWAP-k perturbations of a
common base: each set = base with k of its N images swapped for held-out same-class images (k=1..N). Small k ⇒
sets nearly identical (hard); k=N ⇒ disjoint (the current easy case).
- **Measure:** leave-one-INIT-out matching accuracy vs k → the resolution CURVE. Headline claim if it holds:
  "ΔW distinguishes training sets that differ by only ONE image" (k=1 above chance).
- **Design note:** M sets × K init seeds, digits fixed {0,1}, N=4. **Pick M LARGE enough that chance=1/M is
  LOW** (auditor) — else "above chance" at k=1 is a weak bar (e.g. M≥10 → chance≤0.1). Report
  accuracy-ABOVE-chance AND the per-k permutation null so the resolution curve is interpretable, not just
  raw accuracy. (The k=1 test is exactly the SNR of one image's ΔW-contribution vs the init-noise.)
- **Reuse:** `instance_zoo.py` + `instance_recovery.py` (already do LOIO + the Grassmann-only norm-control).

## B. RECIPE-INVARIANT MATCHING — unknown recipe (removes crutch iii; best value/effort)
A real attacker does not know the victim's lr/activation/rank/T. Match a target to its content/instance using
REFERENCE adapters trained with a DIFFERENT recipe.
- **B1 content-level, near-free:** the atlas factorial zoo already varies {activation × lr × composition ×
  init}. Hold out a whole ACTIVATION (e.g. all softplus) as targets; reference = gelu+relu. Predict the
  target's COMPOSITION cross-recipe.
- **B2 instance-level:** a small new zoo — {0,1}, fixed image-samples × {gelu, relu, softplus} × init.
  Cross-recipe instance matching (does the exact-set fingerprint survive an activation change?).
- **HEADLINE = the GRADED recipe-distance curve, NOT a single cross-activation number (auditor).** Report a
  ladder of increasing recipe-distance: same-recipe (baseline) → cross-lr/T (SAME base, mild) → cross-activation
  (DIFFERENT base, strong). Plot matching accuracy vs recipe-distance.
- **The cross-activation NULL is AMBIGUOUS — design around it.** Cross-activation changes the frozen base → σ′
  → gate geometry, so a cross-activation POSITIVE (cross≈same) is genuinely strong (recipe-invariant). But a
  cross-activation NULL does NOT mean "recipe-specific fingerprint" — it may just mean the base-geometry gap is
  too wide for even a real signal to bridge (a TEST-POWER failure). Honest scoping: cross-activation null =
  "not invariant to a FULL base-geometry change", NOT "fingerprint is recipe-specific". The graded curve
  disambiguates: cross-lr holds but cross-activation collapses = clean interpretable result, not a fuzzy binary.
- **Reuse:** atlas zoo (B1, already on disk) + a small instance×activation zoo (B2).

## C. MEMBERSHIP INFERENCE FROM THE ADAPTER — drop closed-set (removes crutch i; literature-comparable)
Reframe "which of a known list" → the standard privacy question: "was THIS image in the private set?"
- **Score:** s(x) = ‖ΔW·(x−μ)‖ (the retrieval score) as an in/out detector; also the θ0-referenced variant
  s(x) = ‖ΔW·(x−μ)‖ with the base-gradient direction removed (LoRA-Leak showed θ0-as-reference helps — test it).
  **μ = the SAME-DISTRIBUTION mean (mean of {0,1} images)** so the score isolates the INSTANCE deviation, not
  "is this a {0,1} image at all" (auditor).
- **CRITICAL — the negative pool (the hole that would silently fake a positive, auditor).** Negatives must be
  non-members of the SPECIFIC adapter being scored. For a MIA pooled/cluster-robust over many adapters, a
  non-member of adapter i can be a MEMBER of adapter j — if the score picks up "was this in SOME adapter" via
  shared {0,1} structure, the AUC inflates WITHOUT true per-adapter membership. **FIX: use a GLOBALLY-held-out
  pool of {0,1} images used in NO adapter's private set** (disjoint from EVERY adapter's training set, not just
  the target's). State this disjointness explicitly. Otherwise it degenerates to content detection.
- **Measure:** MIA AUC over many (adapter, image) pairs, cluster-robust over adapters. Compare to the
  LoRA-Leak anchor (0.775, VERIFY at source before citing) and to a random-image AUC≈0.5 floor.
- **Reuse:** the eco retrieval machinery (`retrieval_auc`, `_auc`) with the globally-held-out negative pool.

## 4. SHARED GATES (load-bearing — every one bit us already)
1. **Fold by the NUISANCE, never by the target.** Cross-fit folds = init (or recipe), NOT the thing being
   predicted — the +0.000 fold-isolation artifact (instance Facet-C) came from folding by composition.
2. **NORM-CONTROL.** Report Grassmann-ONLY (pure direction) alongside the full distance — the atlas distance is
   already scale-free, so a magnitude fingerprint would only show if the full-vs-grass numbers diverge.
3. **ΔW gauge-clean, not raw (B,A).** Raw factors carry the init frame (ARI~init +1.0); all matching on ΔW=BA.
4. **Cluster-robust CI + permutation null** on every headline number; state G (cluster count) honestly.
5. **Trivial baseline for everything:** chance=1/M (A,B), random-image AUC≈0.5 (C), same-recipe baseline (B).
6. **Same-distribution negatives** for MIA (C) — else it's content detection.
7. **Observe-framed, weakest-attacker/lower-bound** scope on every number; no "confirmed/proven". Small-scale
   (MNIST-MLP, N=4, rank 8) stated.
8. **DETECTION / IDENTIFICATION, NOT reconstruction — prominent on EVERY positive (auditor).** "Distinguishes
   sets differing by one image" or "MIA AUC 0.8" is a DETECTION result; it must never read as "recovers the
   image". Put this next to each headline number, not just in the plan preamble.

## 5. What counts as a REAL result vs a null (pre-registered)
- **A:** the resolution curve's breaking-point k*. REAL if k=1 (differ-by-one-image) is above chance with CI
  excluding chance; NULL/weak if recovery needs near-disjoint sets (then the 1.000 really was max-separability).
- **B:** REAL (strong) if cross-recipe ≈ same-recipe (recipe-invariant fingerprint); REAL (also useful) if it
  collapses cross-activation (fingerprint is base-geometry-specific — bounds the attacker's recipe ignorance).
- **C:** REAL if MIA AUC meaningfully > 0.5 on same-distribution negatives (state the number, compare to
  LoRA-Leak); NULL if ≈0.5 (adapter doesn't leak membership at this scale).
All three can null cleanly and honestly — a null in A or C is itself a privacy-bound result.

## 6. Sequence, roles, compute
Order: **B1 first** (near-free, atlas zoo on disk) → **C** (cheap, eco machinery) → **A** (new swap-k zoo) →
**B2** (new instance×activation zoo). Each: build zoo (bsub GPU, ~25s each) + analyze (bsub CPU numpy/scipy).
Roles (as before): this session specs/co-drafts, **auditor** adversarially reviews (fold/norm/baseline/scope),
**executer** builds/runs. Compute-gated on the user's go per experiment. All results save tensors + a figure.
