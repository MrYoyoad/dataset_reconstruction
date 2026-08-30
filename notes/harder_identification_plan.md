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
- **Design note:** M sets × K init seeds, digits fixed {0,1}, N=4. Chance = 1/M. Report accuracy AND a
  permutation null at each k.
- **Reuse:** `instance_zoo.py` + `instance_recovery.py` (already do LOIO + the norm-control).

## B. RECIPE-INVARIANT MATCHING — unknown recipe (removes crutch iii; best value/effort)
A real attacker does not know the victim's lr/activation/rank/T. Match a target to its content/instance using
REFERENCE adapters trained with a DIFFERENT recipe.
- **B1 content-level, near-free:** the atlas factorial zoo already varies {activation × lr × composition ×
  init}. Hold out a whole ACTIVATION (e.g. all softplus) as targets; reference = gelu+relu. Predict the
  target's COMPOSITION cross-recipe. Cross-activation is the STRONGEST cut (different frozen base ⇒ different
  ΔW geometry); cross-lr/T (same base) is milder — report both.
- **B2 instance-level:** a small new zoo — {0,1}, fixed image-samples × {gelu, relu, softplus} × init.
  Cross-recipe instance matching (does the exact-set fingerprint survive an activation change?).
- **Measure:** cross-recipe accuracy vs the same-recipe baseline. Invariant (cross ≈ same) = the strong,
  scary claim; collapse = recipe-specific fingerprint (also a clean result).
- **Reuse:** atlas zoo (B1, already on disk) + a small instance×activation zoo (B2).

## C. MEMBERSHIP INFERENCE FROM THE ADAPTER — drop closed-set (removes crutch i; literature-comparable)
Reframe "which of a known list" → the standard privacy question: "was THIS image in the private set?"
- **Score:** s(x) = ‖ΔW·(x−μ)‖ (the retrieval score) as an in/out detector; also the θ0-referenced variant
  s(x) = ‖ΔW·(x−μ)‖ with the base-gradient direction removed (LoRA-Leak showed θ0-as-reference helps — test it).
- **CRITICAL design:** negatives must be SAME-DISTRIBUTION held-out (same digits {0,1}, images NOT in the
  private set). Otherwise it degenerates to trivial content detection, not membership. Report the negative pool
  composition explicitly.
- **Measure:** MIA AUC over many (adapter, image) pairs, cluster-robust over adapters. Compare to the
  LoRA-Leak anchor (0.775, VERIFY at source) and to a random-image AUC≈0.5 floor.
- **Reuse:** the eco retrieval machinery (`retrieval_auc`, `_auc`) with same-distribution negatives.

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
   (MNIST-MLP, N=4, rank 8) stated; this hardens IDENTIFICATION, not reconstruction.

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
