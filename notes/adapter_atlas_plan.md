# Adapter-Space Atlas — Plan (§III.8 of the dataset-sensitivity program)

**Status: PLANNING ONLY.** Nothing here has been run. **All compute — including the CPU-only
existing-data pre-look, and especially the factorial-zoo GPU build — is GATED on the user's DIRECT
in-session approval.** No `bsub`, no re-save, no save-hook commit is authorized by this file's existence.
No relay-firing. (See "Compute gate" at the end.)

Inherits the whole program's honesty discipline: **OBSERVE, don't conclude** (a PASS is a data point,
not a verdict); **weakest-attacker scoping** (any recovery number bounds only the WEAKEST attacker — a
LOWER bound on leakage, never the reconstruction limit); **small-n stated explicitly**; **baseline gates
with bootstrap-CI lower bound > 0**; **no pseudo-replication** (CI/associations over independent
adapters, never over the rows inside one ΔW).

---

## 1. Mission — variance-decomposition of the adapter clustering (the SPINE)

Sharpened by the user (verbatim intent): *"cluster by the ACTUAL ADAPTER VALUE and see to what extent
it is DIFFERENT than the init and lr etc."* The plan's spine is a **variance decomposition**, not a
"subtract-nuisance-then-hunt-composition" search:

1. **CLUSTER on the actual adapter value ΔW = BA.** This object is ALSO the gauge-clean one — the BA
   product is GL(r)-invariant, raw B,A carry a meaningless internal frame — so **the user's "actual
   adapter value" and our gauge gate COINCIDE.** We cluster on ΔW; raw (B,A) is a contrast only (§4, §5).
2. **DECOMPOSE that partition across the training knobs:** how much of the ΔW-clustering is explained by
   {init/seed, lr, activation, weight-decay} vs how much is RESIDUAL. **The "EXTENT DIFFERENT FROM
   init/lr" IS THE HEADLINE NUMBER.** The residual is where data/composition signal (if any) lives.
3. **Method:** adjusted-Rand / mutual-information between the ΔW-partition and each factor's label, OR
   nested variance-explained (fit {init,lr,activation,wd} first, composition on the residual second).
   yoado-f0's "nuisance-matched-baseline residual" and this "residual beyond init/lr/activation" are
   **THE SAME OBJECT** — they converge; we say so.

**Why this is leakage, and why it EXTENDS (never contradicts) the per-image story.** If — after the
init/lr/activation contribution is accounted for — a RESIDUAL partition still tracks the private
composition (which classes, N, imbalance, shared transform, duplication count), then an attacker who
recovers no pixels can still read the private *makeup* off the weights. This is an **additional leakage
channel stacked ON TOP of the per-image bound**, an EXTENSION of the leakage surface. It does not weaken,
strengthen, or contradict any per-image number. We never say "so images leak"; we say "composition is a
second thing that can leak, measured on its own nuisance-matched gate."

E1/E2/E4 from the original §III.8 sketch **unify into this one decomposition**: E1 = the
init/lr/activation factors; E4/composition = the residual factor; E3 = a laziness sub-analysis (§4).

---

## 2. Grounding / prior work (cited, load-bearing)

- **GL(r) gauge — core obstacle, addressable.** *Learning on LoRAs* (ICLR'25, arXiv:2410.04207):
  **ΔW = BA is gauge-invariant; raw B, A are NOT** (invariant only to B→BR, A→R⁻¹A, R ∈ GL(r)). Any
  distance on raw factors MUST canonicalize first. Consequence: ΔW-based features (F-a, F-c) are
  gauge-safe; raw-(B,A) features (F-b) are not.
- **Nuisance caution — the make-or-break.** *Hyper-representations* (Schürholz et al., NeurIPS'22):
  weight-zoo embeddings cluster by **init & activation**, not just task. "Clusters by composition" is
  CONFOUNDED until init/activation are factored out — this is the whole point of the decomposition.
- **Precedent it can work.** *Modular LLMs / Library of LoRAs* (arXiv:2405.11157): LoRA weight-similarity
  ↔ task transfer; adapters cluster by domain. Existence proof of recoverable task structure in weights.
- **Mixed = linear combo?** Task arithmetic (LoRAHub / ZipLoRA / MoLE; submodule linearity
  arXiv:2504.10902) — motivates the E3 residual, framed as departure-from-laziness (§4).

---

## 3. Featurizations (each tagged with gauge status)

Verified on disk: arms save the DENSE ΔW product only — `dW_ref`/`dW_seed_mean` (arm_b),
`dW_base_mean` (arm_c/d, arm_d per-target stacks), `dW_distinct_mean` (arm_e); shape (1000,784) = the
MLP layer-0 delta. **Raw A,B factors are NOT saved anywhere.** ΔW here is a seed-MEAN (matters for §6).

- **F-a — ΔW directly (gauge-INVARIANT).** SVD spectrum + top-p singular subspaces; compare via
  **principal angles / Grassmann** on top-p subspaces + cosine on the spectrum. *On disk NOW.*
- **F-b — raw (B,A) pairwise (gauge-VARIANT; CONTRAST only).** Needs (i) a **factor-save hook** (a
  REQUIRED code add — arms save only the product) and (ii) **canonicalization per arXiv:2410.04207**
  before any WITHIN-(B,A) distance. Never a standalone "this is data structure" claim (§5).
- **F-c — two-sided Bures–Wasserstein (gauge-INVARIANT; candidate HEADLINE metric).** C_out = ΔWΔWᵀ,
  C_in = ΔWᵀΔW (both from ΔW, both gauge-invariant); Bures–Wasserstein on each, stacked. Symmetric,
  metric, precomputed → feeds precomputed-metric clustering/UMAP. *On disk NOW.*

Clustering/embedding: **precomputed-metric only** (agglomerative/UMAP on the F-a/F-c distance matrix) —
never a raw-vector embedding that silently reintroduces the gauge.

---

## 4. The decomposition, as facets (each: signed prediction + kill + data need)

### Facet A — NUISANCE factors (was E1): fit the partition↔{init/seed, lr, activation, wd} model
Over the factorial zoo (§7), measure association between the ΔW-partition and each nuisance label
(adjusted-Rand / silhouette-by-label). **This step FITS the nuisance model** that Facet C must beat.
- **Signed prediction:** activation & init/seed explain a NON-trivial share of the ΔW-partition
  (hyper-rep warns so) — association well above the label-shuffle null.
- **Kill / branch:** if activation/init explain essentially ALL of the ΔW-partition (residual at the
  null), composition is CONFOUNDED and Facet C is reported as a nuisance artifact, not leakage.
- **Data need:** **NEW compute — the factorial zoo** with per-seed ΔW retained (existing arms fix
  activation/lr, so they cannot attribute anything to those factors; §7).

### Facet C — COMPOSITION residual (was E2 + E4): does composition beat the FITTED nuisance model?
Recover the composition label (N-bucket, minority-count m, copy-count k, class-set / target identity)
from ΔW-features, and test whether it beats **Facet A's fitted nuisance predictor** — NOT a naive prior.
**The E1-fitted nuisance model IS the E4 baseline: E1↔E4 are one pipeline** (E1 fits the nuisance
predictor on {init,activation,seed,lr,wd}; E4 must show signal in the exact fitted RESIDUAL). This is
the same object as "residual beyond init/lr/activation" (§1.3).
- **Signed prediction:** composition recovery beats the nuisance-model baseline with a **bootstrap 95%
  CI lower bound > 0** (resampled over independent adapters, §6); ‖ΔW‖/spectrum track N and k monotonically.
- **Kill:** CI includes 0 (composition ≈ nuisance model) ⇒ at this scale the adapter encodes its RECIPE,
  not its private composition — report the honest null (F5-style).
- **Data need:** factorial zoo for the clean CI; existing arm data supports only a cheap pre-look (§6, §7).

### Facet L — LINEAR-COMBO / laziness (was E3): mixed adapter vs linear combination of pure ones
Fit c* = argmin ‖ΔW_mix − Σ c ΔW_pure,c‖; report the residual fraction vs a **shuffled-combo baseline**.
- **Theory tie (LOAD-BEARING framing):** the residual is **EXPECTED NONZERO** — LoRA is NOT strictly
  lazy (measured ‖ΔW‖/‖W₀‖ ≈ 0.23 in this program). Frame the residual as a **departure-from-laziness
  quantification**, NOT as a refutation of the linear-combo hypothesis. **A ~zero residual would be the
  surprise.**
- **Signed prediction:** residual fraction is SMALLER than the shuffled-combo baseline (CI lower bound
  > 0) — mixed adapters sit NEAR, not AT, the pure-span; the gap measures nonlazy interaction.
- **Kill:** residual ≈ shuffled baseline ⇒ no exploitable linear structure to decompose a mixture.
- **Data need:** pure adapters exist (arm_b/c per-class); a true MIXED (arm-F) adapter may need
  construction → possibly new compute.

---

## 5. Two-clustering contrast — ΔW (primary) vs raw (B,A) (second method)

Per the user: run **TWO parallel clusterings and COMPARE.** (1) ΔW=BA — gauge-clean, PRIMARY, the
"actual adapter value." (2) raw (B,A) — as-is, keeping the internal r×r frame ΔW discards, SECOND method.
The frame is **init-pinned**, so per the weight-zoo result (B,A) should cluster HARDER by init/seed.
**The comparison is a second lens on "how much is the adapter just its recipe."** Auditor-consistency:
**raw (B,A) stays STRICTLY a CONTRAST — never a standalone "this is data structure" claim** — which is
exactly the gauge gate. The user's ask and the gauge gate COINCIDE; any leakage/composition claim rides
on ΔW alone.

**The honest, attributable test (yoado-bd — do NOT use bare partition divergence).** Two clusterings of
the same objects under different metrics differ SOMEWHAT by chance at small N, so "(B,A) and ΔW
partitions disagree ⇒ init contribution" conflates a real gauge effect with metric noise. Instead ANCHOR
both partitions to the KNOWN nuisance labels. The claim "gauge+init live in raw (B,A) and are scrubbed
by ΔW" holds **IFF**:
- association( (B,A)-partition , init-seed/activation labels ) is HIGH, **AND**
- association( ΔW-partition , init-seed/activation labels ) is AT/NEAR the label-shuffle null.

Measure via adjusted-Rand / silhouette-by-label against the KNOWN init/activation labels — **reuses the
exact Facet-A nuisance labels + the label-permutation null (§6); no new machinery.** Net picture: raw
(B,A) lights UP on init/activation, ΔW stays DARK on it. The bare (B,A)-vs-ΔW adjusted-Rand *agreement*
may be reported DESCRIPTIVELY, but **the load-bearing statement rides on the two associations-vs-null,
not on the partitions merely disagreeing** — state this so nobody over-reads the divergence.

---

## 6. Statistics (permutation null on EVERY association; no pseudo-replication)

- **Unit of replication = one independent adapter (one composition × one seed × one knob-cell).** NEVER
  the 1000×784 entries of a single ΔW, NEVER its p singular directions. A ΔW is ONE sample.
- **Label-permutation null on EVERY partition↔factor association — never eyeballed from a UMAP/t-SNE.**
  Report adjusted-Rand / silhouette vs a shuffled-label reclustering with a permutation CI. This is the
  qualitative twin of Facet-C's bootstrap-over-adapters gate; both are mandatory.
- **Meaningful-recovery gate:** every recovery/association number carries a **bootstrap 95% CI resampled
  over independent adapters**; a claim stands only if the CI **lower bound > 0** vs its baseline
  (nuisance model for Facet C; shuffle null for clusters). Straddling 0 ⇒ INDETERMINATE, reported as such.
- **Distinct-units / effective-n check (before ANY clustering claim).** Do the existing arm_b/c/d/e dW
  files SHARE seeds or targets? They do — several are seed-MEANS and reuse target IDs across arms — so
  **effective-n is BELOW the file count** (pseudo-replication). Compute effective-n first; it CAPS what
  an existing-data pre-look can conclude and directly motivates the purpose-built zoo (Q1).
- **Baselines are explicit:** Facet C beats the FITTED nuisance model; clusters beat the shuffle null;
  Facet L beats the shuffled-combo. Report recovery-MINUS-baseline with the CI on the difference.
- **Multiple factors/arms tested ⇒ report the comparison count**; no cherry-picking the one that separated.
- **Weakest-attacker footer on every recovery number.**

---

## 7. Open-design-question resolutions

- **Q1 — existing population vs purpose-built zoo? → PURPOSE-BUILT SMALL FACTORIAL ZOO.**
  {seeds/inits} × {lr} × {activation} × {composition}, with **ΔW, B, AND A saved for every cell**
  (one extra save-hook line so both clustering methods run off the SAME population). **Reason (state
  it):** the crux sweep varies activation/lr but saves only reconstructions, NOT ΔW; the arms SAVE ΔW
  but FIX activation/lr — so **NEITHER existing population can attribute the clustering to a factor.**
  The factorial zoo is the ONE clean design; it is also what Facet A's nuisance fit and the §5 contrast
  require. **E2-on-existing-data drops to at most a cheap exploratory PRE-LOOK** (bounded by the
  effective-n cap, §6), never the headline.
- **Q2 — headline featurization? → F-c (two-sided Bures–Wasserstein) HEADLINE, F-a (SVD/principal
  angles) as a corroborating gauge-invariant second view, F-b (canonicalized raw B,A) as CONTRAST only.**
  F-c/F-a need no gauge fix and no code change; F-b needs both the factor-save hook and the canonicalizer.
- **Q3 — does composition-recovery fold into recover-N? → YES: ONE structural-leakage FAMILY, three
  readouts.** recover-N (count) · recover-composition (class-set / imbalance label) · recover-shared-
  transform (perturbation param). All three gated IDENTICALLY: nuisance-matched / structure-blind
  baseline + label-permutation null + bootstrap-CI>0 over independent adapters, all weakest-attacker
  scoped. Report them together as the adapter-space structural-leakage readouts, adjacent to (not
  replacing) the per-image results.

---

## 8. Compute / sequencing

1. **Existing-data PRE-LOOK (cheap, exploratory only):** F-a + F-c distances over arm_b/c/d/e ΔW,
   effective-n computed FIRST, clusters vs shuffle null. CPU analysis — but STILL awaits approval.
   Purpose: does ANY composition signal survive the effective-n cap? Not a headline.
2. **GO/NO-GO** on the pre-look + effective-n.
3. **If GO → build the FACTORIAL ZOO** (NEW GPU compute): {seeds/inits}×{lr}×{activation}×{composition},
   save ΔW+B+A per cell. Then Facet A (fit nuisance model), Facet C (composition beats fitted residual),
   §5 two-clustering contrast, all under §6 gates.
4. **Facet L** on available pure/mixed pairs; build arm-F mixed adapters only if promising.
5. **F-b path (optional, later):** the canonicalizer for within-(B,A) distances — only if the §5 contrast
   needs more than the nuisance-anchored associations.

**Data-need summary.**
| Facet / step | Runs on existing data? | New compute / code |
|---|---|---|
| Pre-look (F-a/F-c clusters, effective-n) | YES (CPU) | none |
| Facet A — nuisance fit (init/lr/activation/wd) | NO | **factorial zoo** (varies + saves the factors) |
| Facet C — composition residual vs fitted model | pre-look only | factorial zoo for a clean CI |
| §5 two-clustering (ΔW vs raw B,A) | NO | **save-hook: persist B, A, AND ΔW per cell** |
| Facet L — linear-combo / laziness | PARTIAL (pure exist) | arm-F mixed adapters may need building |
| F-b canonicalized distances | NO | factor-save hook + arXiv:2410.04207 canonicalizer |

---

## COMPUTE GATE (read before running anything)

This document is **planning only**. No experiment, zoo build, re-save, or save-hook commit is authorized
by its existence. **Every step — including the CPU-only pre-look — awaits the user's DIRECT in-session
approval; no relay-firing.** When approved, standard program discipline applies: stage-0 smoke on WEXAC,
`python -u`, rsync `experiments/` first, save ΔW+B+A + visual artifacts, OBSERVE-not-conclude captions,
weakest-attacker footer on every recovery number.

---

## 9. Optional enrichments (yoado-2a co-review — non-blocking; folded post-audit-pass)

These sharpen the plan without changing the spine; each names the section it refines.

1. **RANK as an axis (refines §7 zoo / §3).** The gauge group is GL(r) and the top-p subspace dim scales
   with r, so composition-leakage (Facet C residual) may vary with rank. Add r as an OPTIONAL 5th zoo axis
   (or a small separate rank-sweep) — ties directly to the r≥N NTK argument (r≥N ≈ full-FT). Cost: zoo
   size. **Minimum:** FIX and STATE r=8 as the studied rank if not swept.
2. **PER-LAYER atlas (refines §3 / §7 save-hook).** The arm ΔW is layer-0 only (1000×784); the valley
   showed ‖Δμ‖ L0>L1>L2 (instance signal concentrated in L0), so **L0 stays the PRIMARY** — but the zoo
   save-hook should persist ALL layers' ΔW+B+A so a per-layer atlas is available. Open question it buys:
   does composition live in a DIFFERENT layer than the recipe?
3. **CRUX signed prior for Facet A (cross-track prediction).** The free-c crux result (kinked
   relu/leaky_relu/selu leak MORE / sit in a different NTK regime) predicts kinked activations sit APART
   in the activation-nuisance structure. **Checkable:** if the activation clusters do NOT separate
   kinked-vs-smooth, that is TENSION with the crux — a genuine cross-track consistency test.
4. **R2F two-sidedness = the mechanistic WHY for F-b (refines §3 F-b, §8 step 5).** ∇_A = Bᵀ∇W and
   ∇_B = ∇W·Aᵀ, so A and B are output-/input-side gradient measurements — even CANONICALIZED, (B,A) MIGHT
   carry two-sided structure that the product ΔW collapses. This is the mechanistic motivation for the
   later canonicalizer effort, beyond "keeps the gauge."
5. **SEED-MEAN double-edge (sharpens §3 / §6 / §8 pre-look).** The existing arm ΔW are seed-MEANS: they
   have ALREADY averaged out the seed nuisance, so **the pre-look is SEED-BLIND — it cannot study the seed
   factor at all**, and a seed-mean is not a real single adapter (may wash or artificially clean instance
   signal). Make the pre-look's seed-blindness EXPLICIT; it reinforces why the zoo must save per-seed ΔW.

**Status unchanged: PLANNING ONLY.** These enrichments add no authorization; all compute still awaits the
user's direct in-session go.

---

## 10. Audit revisions (yoado-bd full science audit — SUPERSEDE the body where they conflict)

Three load-bearing (#1, #2 fix BEFORE any build; #3 baked into the zoo design) + two readout/wording.

**1. CROSS-FIT the nuisance-fit / composition-test — Facet C is otherwise in-sample & optimistic
(SUPERSEDES §4 Facet C).** Because the E1-fitted nuisance model IS the E4 baseline (one pipeline), if
Facet A's {init,lr,activation,wd} model is FIT on the same adapters Facet C is TESTED on, the residual is
in-sample: the fit can ABSORB composition variance (whenever init/composition are even slightly
correlated in the realized zoo) or leave an optimistically-clean residual, and the bootstrap-over-adapters
CI does NOT capture this (it resamples the same fit). **FIX (mandatory, not optional): cross-fit** — fit
the nuisance model on one adapter split, test composition-recovery on the HELD-OUT residual, K-fold /
rotate; the bootstrap CI resamples WITHIN the cross-fit. This is the same discipline that killed the
winner's-curse in the whitened-metric round; the one-pipeline design makes it MANDATORY here. Without it,
"composition beats nuisance" can be an in-sample artifact.

**2. §5 (B,A) init-contrast uses RAW, UNCANONICALIZED (B,A) — the OPPOSITE of §3-F-b (CLARIFIES §3/§5,
resolves an internal inconsistency).** §5's contrast NEEDS the init-pinned r×r frame INTACT — that frame
IS the signal that makes raw (B,A) light up on init. Canonicalizing (arXiv:2410.04207) fixes the GL(r)
gauge → REMOVES the init-frame → (B,A) would no longer cluster by init → the §5 contrast DIES. So they are
DIFFERENT objects for DIFFERENT questions: **§5's contrast distance on (B,A) is deliberately
RAW/uncanonicalized** (exposes init); **F-b's canonicalized distance (§3, Q2) is a SEPARATE optional
analysis** ("is there (B,A) structure BEYOND the gauge?"). The §5 contrast requires the OPPOSITE of
canonicalization — no reader should think it needs F-b's canonicalizer.

**3. The factorial zoo must be BALANCED + MULTI-SEED-PER-CELL, or the decomposition has no power (REFINES
§7).** Factorial is necessary, not sufficient: (a) **BALANCED / orthogonal cells** — every composition
crossed EQUALLY with every init/lr/activation, so the Facet-A-vs-Facet-C variance attribution is
design-clean and ORDER-INDEPENDENT (nested variance is order-dependent under imbalance); (b) **MULTIPLE
seeds per (activation, lr, composition) cell** — single-seed-per-cell CANNOT separate composition from
seed noise, and the bootstrap-over-adapters CI can exclude 0 only with within-cell replication. **Specify
seeds-per-cell** (that replication is what buys power) and **state the expected effective-n per factor
level up front** so a "likely INDETERMINATE at this scale" verdict is set honestly before the run, not
after.

**4. SCOPE the severity of composition-leakage (wording, alongside the weakest-attacker footer).**
Composition-recovery = set-level METADATA (N, class balance, presence of a shared transform), NOT
per-image content. Keep "read the private makeup off the weights" from being heard as "read the private
IMAGES" — it is a DISTINCT, WEAKER channel than image reconstruction. State it as such wherever the
recovery claim appears.

**5. Facet L — report the c* coefficients, not just the residual fraction (cheap interpretability).**
Uniform c* ⇒ concept-blend; sparse c* ⇒ one-image-dominant mixture. Free on top of the residual test.

**Status unchanged: PLANNING ONLY — these revisions add no authorization; all compute awaits the user's
direct in-session go.**

---

## 11. Audit revisions — §9 enrichments pass (yoado-bd; SUPERSEDE §9 and refine §10 #2)

**1. DROP the canonicalized-F-b path entirely — it chases structure the gauge theorem says isn't there
(SUPERSEDES §9 enrichment 4, §3-F-b canonicalizer, §8 step 5, and the F-b half of Q2).** §2 already states
ΔW=BA is the COMPLETE gauge-invariant. Therefore a canonicalized (B,A) is just ΔW re-expressed
(B=UΣ^½, A=Σ^½Vᵀ): it carries ZERO gauge-invariant information beyond ΔW, by definition. The two-sided
structure R2F points at (∇_A=Bᵀ∇W input side, ∇_B=∇W·Aᵀ output side) is ALREADY IN ΔW's SVD: C_out=ΔWΔWᵀ
(=U, output side) and C_in=ΔWᵀΔW (=V, input side) — which is exactly **F-c, already the headline metric.**
ΔW does NOT "collapse" the two sides; its SVD retains both. What ΔW discards is ONLY the r×r internal
frame = pure gauge = no information. **FIX:** R2F two-sidedness correctly motivates the two-sided F-c
(C_in AND C_out) — which the plan already has — NOT a separate canonicalized F-b. Drop the F-b
canonicalizer from §8 step 5, Q2, and the enrichment. **This also RESOLVES §10 #2 cleanly: there is no
"canonicalized-F-b question" at all — canonicalized(B,A) ≡ ΔW.** The ONLY legitimate raw-(B,A) use is the
§5 UNCANONICALIZED init-contrast (which measures NUISANCE signal — the init-pinned frame — not
gauge-invariant information). Net: simpler AND correct — one fewer featurization, one fewer code
dependency (the canonicalizer is gone).

**2. Enrichment 3's crux prior is ORACLE-PROVISIONAL — not a live consistency check yet (SUPERSEDES §9
enrichment 3's framing).** The free-c crux "kinked leak more / different NTK regime" is NOT settled: the
free-c wc-ladder (job 392821) has not landed and the current read is the oracle first-pass, which MAY
FLIP. So if the atlas activation-clusters do NOT separate kinked-vs-smooth, one cannot distinguish
"tension with the crux" from "the crux kink-effect was an oracle artifact" — two unsettled results
propping each other up (circular). **FIX:** frame enrichment 3 as "a cross-track consistency check to run
ONCE the crux free-c ranking is settled," NOT a live signed prior now.

**3. Seed-mean pre-look signal is an UPPER BOUND, not evidence (SHARPENS §9 enrichment 5 / §6).** The
existing arm ΔW are seed-MEANS: seed-averaging removed the very noise that would blur composition clusters,
so a pre-look on seed-mean ΔW can OVERSTATE composition separation. **FIX:** state explicitly that any
pre-look composition signal is an UPPER BOUND (noise pre-removed), not evidence of leakage — only the
per-seed zoo gives the honest read. (Stacks with §6's effective-n cap: the pre-look is both seed-blind AND
optimistically clean.)

**Minor — rank axis (§9 enrichment 1):** adding r multiplies zoo cells and COMPOUNDS the
seeds-per-cell power problem (§10 #3), so **"fix r=8 and state it" is the right default at this scale**;
a rank-sweep only if the base decomposition shows power to spare. Per-layer (enrichment 2) — no issue.

**Status unchanged: PLANNING ONLY — no authorization added; all compute awaits the user's direct go.**

---

## 12. Cross-fit split recipe (yoado-bd, for the eventual build) — with ONE open nesting question

Concretizes §10 #1 (mandatory cross-fit of the Facet-A nuisance-fit / Facet-C composition-test).

- **Split unit = one independent adapter** (composition × seed × knob-cell). NEVER rows within a ΔW.
- **K-fold cross-fit:** fit the {init,lr,activation,wd} nuisance model on K−1 folds, score
  composition-recovery on the held-out fold's RESIDUAL, aggregate across the K held-out folds. **STRATIFY
  folds** so every fold spans all nuisance levels AND all composition levels (a fold missing an activation
  cannot fit that activation's contribution). **K:** 5-fold if ≥~5 adapters per nuisance-cell; else
  leave-one-adapter-out (LOO) — state that LOO CIs are high-variance.
- **Label-permutation null:** shuffle the composition labels and re-run the IDENTICAL cross-fit+CI
  pipeline, so the null inherits the same structure. **Headline = recovery MINUS null, CI on the
  DIFFERENCE, claim stands iff CI lower bound > 0.**
- **Honest expectation (state up front):** K-fold × bootstrap × permutation is data-hungry; at a handful
  of adapters per cell the CI will be WIDE → likely INDETERMINATE. That is the EXPECTED small-N outcome,
  not a disappointment.

**RESOLVED — Facet-C composition-recovery inference, FINAL (yoado-bd + yoado-2a fully converged).**
Both naive nestings and the naive denominator are OUT; this is the single agreed recipe.

1. **STATISTIC — cross-fitted HELD-OUT ACCURACY DIFFERENCE.** acc(classifier on nuisance+ΔW features) −
   acc(classifier on nuisance-only features), predicting the composition label, each adapter scored by a
   model NOT trained on it. Per-adapter score s_i = 1[full correct on i] − 1[nuisance correct on i]; point
   estimate = mean(s_i). (Framing it as a mean of per-adapter scores is what makes the analytic CI valid.)
2. **PRIMARY CI — double-ML INFLUENCE-FUNCTION CI, CLUSTER-ROBUST over EFFECTIVE units (not raw adapter
   count).** Adapters sharing a (composition × lr × activation) cell and differing only by seed are
   near-replicates → cluster the sd on the independent unit (§6's effective-n analysis identifies it —
   typically the composition-cell) with n_effective = number of clusters G. **CI = mean(s_i) ±
   t_{G−1, .975} · sd_cluster / √G** — use the **t-quantile with (G−1) df, NOT z**, because with few
   clusters (G ~ 4–8) the normal is anti-conservative. No nesting → no train/test leak; cluster-robust +
   n_eff → no pseudo-replication.
3. **OPTIONAL CHECK — WITHOUT-replacement m-out-of-n SUBSAMPLING** (train/test stay DISJOINT adapters →
   kills the with-replacement duplicate-leak), each subsample re-running the full cross-fit, stratified by
   nuisance-cell. NOT with-replacement bootstrap-outer.
4. **CLUSTER-ASSOCIATION stats (adjusted-Rand / silhouette-by-label — Facet A + §5 anchoring) are NOT
   per-adapter means** → keep the **LABEL-PERMUTATION NULL** (+ subsampling for stability) as their
   inferential tool, NOT the influence-function CI. Two tools, matched to statistic type.
5. **SMALL-N honesty:** at a handful of clusters both the DML asymptotics and the t-with-few-df are
   approximate → CI wide, INDETERMINATE the EXPECTED honest outcome, not a defect. Unit throughout = the
   independent adapter / cluster, never ΔW rows. For tiny G (≤5 clusters), the **wild cluster
   bootstrap** (Cameron–Gelbach–Miller) is the standard less-conservative-yet-valid upgrade over the
   t_{G−1} approximation — named as an OPTIONAL future upgrade only; **t_{G−1} stays the default**,
   since for an exploratory plan a wide / INDETERMINATE CI is the honest expected read anyway.

**OUT (all three rejected, with reason):** with-replacement bootstrap-outer (duplicate-adapter fit/test
split → leak), fixed-split bootstrap-inner (conditions on estimated nuisance + partition → under-counts),
naive sd/√n_adapters (within-cell correlation → pseudo-replication).

**Status unchanged: PLANNING ONLY — compute-gated on the user's direct go.**
