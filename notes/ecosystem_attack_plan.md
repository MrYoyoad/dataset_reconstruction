# Ecosystem Attack — Plan (does a POPULATION of LoRAs on a shared base amplify a target's leakage?)

**Status: PLANNING ONLY.** No compute authorized by this file; the weak-signal zoo build is gated on the
user's DIRECT go. Observe-framed, weakest→population(stronger)-attacker scope on every number.

## 0. The claim (and what is / isn't already known)
Given a famous shared base θ0 and MANY published LoRAs (different projects/data), can the POPULATION be used
to extract MORE about a target adapter's private samples — via clustering + common-mode subtraction?
- **Premise (already validated): θ0-as-reference boosts leakage.** LoRA-Leak (arXiv:2507.18302): using the
  pre-trained model as a reference raises MIA (search-reported 0.775 AUC — VERIFY at source before
  citing as the premise anchor). So "the base helps" is known.
- **The NOVEL claim = POPULATION amplification: many *other* adapters help recover a *target's* private
  data beyond θ0 alone.** LoRA-Leak does NOT show this. ⇒ **the whole ballgame is the GAIN vs the
  single-adapter baseline** (gate 2). No gain = no result.

## 1. Why the first prototype was a NULL (the lesson that shapes this plan)
On the atlas zoo, subtracting the top-p population PCs collapsed composition recovery (ARI 1.0→0.01,
shared-energy=1.0, SNR÷0). Honest cause: that zoo's composition IS the dominant variance (by design), so the
"top PCs" WERE the private signal — tautological, and saturated (baseline already at ceiling). Two failures
to design out: (a) wrong "shared" component (top-variance = signal, not θ0 common-mode); (b) no headroom.

## 2. Substrate — a WEAK-SIGNAL MULTI-TASK zoo
Many adapters on DIFFERENT tasks (different digit-pairs / datasets) sharing θ0, K seeds each. The private
per-adapter signal must be WEAK so the population has room to help.
- **DISJOINT-CONTENT control (required).** All adapters share θ0, so the LOO-shared subspace *should* be θ0
  common-mode — BUT if other-task adapters share the target's private CONTENT (e.g. overlapping digits), the
  shared subspace absorbs target SIGNAL, not just θ0, and you subtract your own signal (the gate-3 tautology,
  one level up). So the LOO-population must be VERIFIED DISJOINT in private content from the target (distinct
  digit-pairs/datasets); report the target-vs-population content overlap explicitly. This pairs with gate 1's
  projection test.
- **Headroom knob = few images (small N)** [Q1, auditor-preferred]: directly lowers per-adapter signal
  without confounding the geometry. (low-rank confounds the LoRA-rank story; label-noise is messier.)
  **Confirm the single-adapter baseline is MID-RANGE (measurably below ceiling AND above floor) BEFORE any
  subtraction** — else the gain is uninterpretable (gate 2).

## 3. Method — leave-one-out common-mode subtraction
1. For a TARGET adapter, estimate the shared base subspace from the OTHER-task adapters (**leave-one-out**),
   operating on ΔW (gauge-clean, per the atlas finding), NOT raw B,A.
2. Subtract the target's projection onto that subspace → private residual.
3. Recover the target's private data from the residual; compare to the single-adapter baseline. **GAIN = the
   ecosystem effect.**
4. Variant: recurring-sample amplification — a sample in M adapters, does averaging help (∝√M)?

## 4. GATES (audit + gates by yoado-f5 — load-bearing; #1 is the E6 fold-bug one level up)
1. **LOO-leakage diagnostic — the PROJECTION test is PRIMARY (shift test is secondary + near-blind).**
   The real diagnostic: measure the TARGET's private-direction PROJECTION onto the LOO-shared subspace — it
   must be LOW. If the target's private signal lies IN the shared subspace, subtraction removes the very
   signal you're recovering (circular — the E6 fold-bug one level up) **even if the subspace doesn't visibly
   shift.** The with/without-target SHIFT test is SECONDARY and near-BLIND at large population (removing 1 of
   N barely moves the subspace regardless of leakage), so it cannot be the primary gate. LOO is honest only
   if the target's private direction has LOW projection onto the shared subspace.
2. **HEADROOM.** Single-adapter baseline must be mid-range (below ceiling, above floor) — measure it first.
3. **Tautology guard.** "Shared" = what's common across DIFFERENT-TASK adapters (θ0 common-mode), VERIFIED to
   have LOW overlap with the target's private direction. If subtracting it kills the target signal, the split
   is wrong (that was the atlas null).
4. **Cheap-proxy baseline.** The cut-1 proxy (reconstruction/clustering-recovery) needs its OWN trivial
   baseline (mean-image / random-adapter control), like every structural experiment.
5. **√M independence AND signal-coherence.** √M aggregation needs (a) INDEPENDENT per-adapter noise — verify
   not correlated through the shared θ0 — AND (b) the recurring sample's signal to add COHERENTLY (same ΔW
   direction) across the M adapters; different recipes may encode the same sample in DIFFERENT directions →
   incoherent → no √M (the arm-B coherent-vs-incoherent issue). Check BOTH, else the gain shrinks below √M.
6. **Scope.** population(stronger-than-weakest)-attacker on every number; observe-framed; no "confirmed."

## 5. Metric (cut-1 cheap, q_eff later) [Q3]
Cut-1 proxy = **NN-RETRIEVAL** of the target's private images from a pool (NOT "reconstruction" — that
implies inversion, not cheap): for each target private image, is it retrieved as a nearest neighbour off the
LOO-RESIDUAL vs off the raw adapter, scored against the **RANDOM-ADAPTER baseline** (gate 4). **The GAIN =
retrieval on the LOO-residual − retrieval on the raw single adapter** (both vs the same baseline). Escalate
to q_eff-on-residual (the Jacobian machinery — heavy) ONLY if the cheap proxy shows a gain worth the compute.
[Q2] Estimator: LOO-PCA on other-task ΔWs for cut-1; the learned prior (hyper-representations) is a later
upgrade. **Headroom (gate 2) is measured against this SAME retrieval baseline, so "mid-range" is well-defined.**

## 6. Sequence & roles
(1) build the weak-signal multi-task zoo; (2) measure single-adapter baseline + confirm headroom (gate 2);
(3) LOO shared-subspace + the with/without-target leakage diagnostic (gate 1) + tautology check (gate 3);
(4) subtract → cheap-proxy recovery vs baseline (gate 4) = the GAIN; (5) √M variant with the independence
check (gate 5). Roles: this session specs/co-drafts, yoado-f5 audits, executer builds/runs. Compute-gated.
