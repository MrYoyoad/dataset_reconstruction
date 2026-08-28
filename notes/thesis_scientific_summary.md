# Thesis — Consolidated Scientific Summary

*Reconstructing / detecting private training data from LoRA‑fine‑tuned model weights.*
*Prepared 2026‑08‑28 (rev 2026‑08‑29). An organized record of what has been OBSERVED so far, with figures, examples, caveats, and open questions. Numbers cite job IDs; every headline carries its limitation.*

**How to read this — stance (per the user's 2026‑08‑29 directive).** This is **early exploratory research**: the results below are **observations, not thesis‑level conclusions** — hold them loosely. Each carries open questions (CIs, confounds, alternatives); a metric "pass" means "no blocker to observing more," not "settled." **Crucially, the leakage/reconstruction numbers here characterize only the WEAKEST attacker** — prior‑free, recipe‑blind, adapter‑only, per‑image. Stronger attackers go *past* them: generative priors (Direction‑3), known‑recipe inversion (e.g. the recent CNN‑inversion work), structural leakage (class / shared‑perturbation / N‑structure), and simply a better decoder. So "the adapter doesn't reconstruct" means "this weakest attacker doesn't" — a **lower bound on leakage, not an upper bound on what an attacker could do.**

---

## 0. State of the project in one paragraph

We ask how much a released fine‑tuned adapter leaks about its private training set, and whether that leakage can be turned back into the images. Two observations so far (held loosely): **(1) the leakage GEOMETRY looks real and high** — an attacker can recover a large fraction of the private‑data directions from the adapter, and, per image, *how much* an image leaks appears predictable from the public base model alone (its base‑gradient‑norm g₀); **(2) but end‑to‑end PIXEL reconstruction by the weakest (prior‑free, adapter‑only) attacker does not yet beat a trivial baseline** — the information is present, that decoder cannot yet convert it to images (a *stronger* attacker — priors, known recipe — may). A counter‑intuitive sub‑result: **multi‑class training does not amplify leakage — it leaks *fewer* recoverable directions** than binary, an effect that lives at low LoRA rank and vanishes as the adapter approaches full fine‑tuning. The active frontier extends this to the full‑weight regime (the "valley‑width" comparison) and to a per‑image privacy predictor.

---

## 1. The question and the two axes

**Setup.** A public base model θ₀ is fine‑tuned with LoRA (adapter ΔW = BA) on a small private set D. The attacker sees the released weights. We measure leakage two complementary ways:

- **Identifiability / Jacobian axis** — the data‑latent Jacobian J = ∂(adapter)/∂(data), giving `r_J` (how many private directions the adapter encodes) and `q_eff(ε)` (how many an attacker recovers against the training‑noise floor). *Upper bound on what is recoverable.*
- **Dataset‑sensitivity axis** — a whitened (Mahalanobis) sensitivity d² of the adapter to a one‑image change, normalized by the reseed‑noise floor (the differential‑privacy notion, measured on parameters). *Per‑record detectability.*

Both feed the thesis end‑goal (Haim et al. lineage): **reconstruction** of the private images. The gap between "detectable" and "reconstructable" is itself a measured quantity here.

---

## 2. Headline scientific findings

### 2.1 Leakage is high; multi‑class leaks FEWER directions (not amplified)

At convergence both binary (BCE) and 10‑class (CE) bases leak a large fraction of all private directions. The surprise: **10‑class recovers FEWER** at every attacker budget ε — the opposite of the intuition that more classes leak more. Observed consistently, including a fully‑converged N=10 lock (q_eff 36 < 59 @ ε=1; job 484948) — a robust *observation* across the configs run so far, not yet a settled conclusion. The earlier claim of a **"2× multi‑class amplification" is RETRACTED** — it was a low‑lr underfit + undersampling artifact; at healthy lr both bases fill the signal subspace from the first training steps.

![Left: identifiability — multi-class (orange) leaks fewer recoverable directions than binary (blue) at every attacker budget ε. Right: reconstruction — every decoded cell is below the mean-image baseline (0/40).](figures/combined/leakage_identifiability_plus_reconstruction.png)

### 2.2 The reversal is a LOW‑RANK effect that vanishes at full fine‑tuning

Sweeping LoRA rank r ∈ {2,4,8,16,32} (job 581629): the multi‑class "leaks‑fewer" gap **closes monotonically 23 → 13 → 0** across the converged ranks r=8/16/32. Binary q_eff is flat (~58); the 10‑class value climbs to meet it as r → N and beyond (Jang 2024: r ≳ N ≈ full fine‑tuning). So the effect is genuinely *low‑rank‑specific*, not a general property of cross‑entropy; and the noise‑coupling mechanism that explains it at r=8 *decouples* by r=16. Low‑rank measurability bracket: r=1 is the only true capacity floor (never memorizes, job 635386); r=2/4 are underfit at the matched recipe and quarantined.

![Rank sweep: the multi-class reversal gap closes 23→13→0 across converged ranks r=8/16/32 (the reversal is low-rank, gone by full fine-tuning). Convergence-gate panel quarantines the underfit r=2/4.](figures/rank_sweep/rank_sweep_headline.png)

### 2.3 Reconstruction: geometry leaks, pixels do NOT (the honest correction)

On the like‑for‑like RAW‑SSIM comparison, **0 of 40 decoded adapter‑only arms beat the trivial mean‑image baseline** (audit‑verified across 20 result files). The number that had circulated — "recognizable digits, ssim_norm 0.57–0.61" — used the mean/std‑MATCHED `ssim_norm` (which inflates raw by ~0.1–0.3) compared against a RAW baseline: apples‑to‑oranges. Even the **TRUE‑ΔW oracle** (known‑recipe upper bound) fails the baseline on the small nets, clearing it in only 5/12 cells. **For the weakest (prior‑free, recipe‑blind, adapter‑only) attacker, pixel reconstruction is an OPEN LIMITATION, not a result** — the geometry (§2.1, §2.4) is where the leakage currently shows up. This 0/40 is a **lower bound**: it says this weakest attacker fails, NOT that the images are unrecoverable — generative priors, known‑recipe inversion, or a stronger decoder could exceed it, and testing that is exactly the open direction.

![Every decoded (adapter-only) reconstruction is BELOW its mean-image baseline (0/40 on the raw comparison); the oracle clears it only on 5/12 easy cells.](figures/combined/reconstruction_vs_baseline.png)

### 2.4 Per‑image leakage is predicted by the base‑gradient‑norm g₀

Which images leak most is predictable **from the public base model alone** — no adapter access. ρ(sensitivity, g₀) = **+0.857** (n=12, job 260171). The end‑to‑end privacy chain closes: ρ(leave‑one‑out memorization, sensitivity) = **+0.88** (job 272309) and ρ(memorization, g₀) = +0.80. This also *explains* the composition effects (the 3.3× class asymmetry reduces to class‑1 having 2.4× larger base gradients).

**Honest caveat (do not overstate):** at scale the predictor is weaker — ρ = +0.777 (n=24, job 272504), graded **INDETERMINATE** (bootstrap CI half‑width 0.189 > the 0.15 target; the low‑g₀ tercile sign‑flips), so g₀ is a strong predictor for *high‑leakage* images but not across the whole range. And there is a live **counterexample**: injected out‑of‑distribution USPS digits have higher g₀ yet leak *less* (n=2 — likely a smoothness / high‑frequency‑energy confound, to be arbitrated at scale).

![Margin-at-scale: per-image sensitivity vs base-gradient-norm g₀.](figures/margin_at_scale/margin_at_scale_headline.png)

### 2.5 Composition arms — what the adapter records

- **Duplication** imprint is **sub‑linear** (β ≈ 0.24 on d², rank‑invariant) — approaching the max‑margin duplication‑invariant limit but not reaching it at finite training.
- **Rank sets absolute leakage** — r=32 ≈ 15× the d² of r=8 (dimension‑confounded, reported per‑direction).
- **Class asymmetry (3.3×) is intrinsic class identity, not rarity** — it survives balancing and inverts cleanly under role‑swap; per‑image *context* rarity is weak (~1.1×).
- **The adapter records the CONCEPT, not the instance** — on the similarity distance‑dial, near‑duplicate swaps are nearly invisible (sensitivity 0.03–0.07) while cross‑digit swaps are large; the d=0 identity rung is calibrated (sensitivity 0, p=1.000).
- **Transfers to a real ViT** — a single private image is detectable in a `vit_tiny` LoRA adapter at p=0.002 (jobs 247474 / 256540).

**Effect of fine‑tuning on classification (from the rigor logs).** Memorizing the N private images via LoRA *harms* held‑out classification, and the harm is strongly asymmetric with the task: **MNIST 10‑class ≈ unchanged** (held‑acc 0.96 → 0.94–0.96, Δ ≈ −0.01 to −0.02), **MNIST binary moderate** (0.91 → 0.80–0.82, Δ ≈ −0.09 to −0.11), and **Fashion 10‑class large‑to‑catastrophic** (0.85–0.90 → 0.67–0.10, up to Δ ≈ −0.75, i.e. collapse toward chance). So there is a genuine memorization↔utility tradeoff, worst exactly where memorization is hardest (Fashion multi‑class). Measured across the rank‑sweep + Round‑B rigor runs (base θ₀ → T=1000 held‑acc; MNIST/Fashion × binary/10‑class × r∈{2..32}).

![Similarity distance-dial: sensitivity vs swap distance — near-duplicates nearly invisible (concept, not instance); d=0 calibrated to zero.](figures/similarity_ladder/similarity_ladder.png)

![Validation gate: high-sensitivity images have higher leave-one-out memorization scores (ρ=+0.88) — the sensitivity→memorization link that licenses calling this "leakage".](figures/h_spotcheck/h_spotcheck_scatter.png)

### 2.6 The activation crux (the supervisor's top ask) — a clean dissociation

Two links in the chain behave oppositely: **smoothness → linearization fidelity HOLDS** (softplus is the best linearizer; the softplus_b(β) knob is monotonic ~12× over β, job 911475), but **linearization → leakage magnitude is REFUTED** (Spearman ≈ +0.03 on MNIST; it even inverts on flowers). Under the realistic free‑coefficient (ReLU) attacker the ranking flips: **silu/gelu leak MOST** — the oracle "softplus wins" was a matched‑extraction artifact. The 152‑config rescore (job 389926, full smoothness spectrum) sharpens this: there is **no clean smoothness law in either direction** — it is a **two‑cluster effect**. relu / leaky_relu / selu leak ~4× the rest (ctrl_margin_norm ≈ 0.43 vs 0.10) with the lowest NTK survival, but **hardswish (also kinked) does not** (0.11), and *within* the smooth‑only set (n=9) the trend actually flips (+0.58). So a single Spearman (−0.38 across the spectrum) misleads; the honest headline is "kinked relu/leaky/selu leak most; smoothness does not positively predict leakage." *(Load‑bearing caveat — the first‑pass is provisional on TWO axes: (i) every config is T=1 and weight‑change‑unmatched, so −0.38 is wc‑provisional; (ii) it was run in ORACLE mode (known‑coefficient upper bound), and this project has shown the free‑coefficient realistic attack can FLIP the activation ranking (on flowers32 the oracle softplus‑favouring order reverses under free‑c). So the two‑cluster read is an UPPER BOUND, not the realistic ranking. The user‑approved closure — a free‑coefficient wc‑ladder {0.005 NTK, 0.03, 0.1, 0.3} with exact per‑activation wc‑matching + feature‑stability‑vs‑T + flowers band — is running now to produce the settled realistic ranking.)*

![The activation crux dissociates: smoothness sets linearization fidelity but NOT leakage magnitude.](figures/crux/softplusb_linearization_vs_leakage.png)

### 2.7 Direct inversion and the gradient bridge

- **Direct weight inversion** (θ_T = F(θ₀, x̂), autograd through unrolled training) recovers small‑N images (SSIM 0.57 at N=4; bit‑exact map at T=1) but hits the **superposition wall** (N=10 → 0.27, N=20 → 0.15). Headline novelty was ceded to the concurrent SimuDy paper (Gal's pointer).
- **Gradient‑bridge decoder** reaches **0.951 cosine** to the true gradient (job 956994) and leaks end‑to‑end for shallow nets with abundant proxy data — but the pixel result is the 0/40 of §2.3 (high input dimension and depth both break it; N>2 collapses both tracks).

![Gradient-bridge reconstruction example grid (MNIST N=2): true vs decoded — recognizable structure but below the pixel baseline (§2.3).](figures/gradient_bridge/phase2_e2e_mnist_N2_softplus.png)

---

## 3. Figure & example inventory (where everything is)

| Program | Figures | Example grids |
|---|---|---|
| Identifiability / rank | `figures/combined/leakage_identifiability_plus_reconstruction.png`; `figures/rank_sweep/{headline,eps,spectrum_r8}.png`; `figures/jacobian_spectrum/` | — |
| Reconstruction | `figures/combined/reconstruction_vs_baseline.png` | `figures/gradient_bridge/phase2_e2e_*_N{2,4,10}_{gelu,softplus}.png`; `figures/direct_inversion/di_grid_*` |
| Dataset‑sensitivity | `figures/margin_at_scale/`, `figures/similarity_ladder/`, `figures/h_spotcheck/`, `figures/retrieval/`, `figures/fullft_valley/calibration_bracket.png` | — |
| Activation crux | `figures/crux/{softplusb_linearization,softplusb_linearization_vs_leakage,activation_matched_wc,anchor_two_curve_summary}.png` | — |
| Prose write‑ups | `notes/leakage_story_consolidated.{md,pdf}` (leakage), `notes/dataset_sensitivity_program_plan.md` (v3), `notes/whitened_sensitivity_metric.md`, `notes/dataset_sensitivity_arm_b_result.md`, `notes/crux_activation_analysis.md` | |

Pending (not yet produced — jobs not run): full‑FT valley main figures, Arm‑G Jacobian figures, the Phase‑2 adapter atlas, the crux T‑sweep of feature‑stability.

---

## 4. Why the numbers are trustworthy (metric‑hygiene, load‑bearing)

These are the guardrails that were violated once and then fixed — they are why the headlines above survive scrutiny:

- **Always gate on the trivial baseline.** A normalized metric (`ssim_norm`) hid a below‑baseline reconstruction (§2.3). A result at or below the dataset‑mean predictor carries no instance‑specific information, however good the raw number looks.
- **Leakage = r_J / hard‑rank, never eff_rank.** `eff_rank` reads *backwards* as the spectrum concentrates; "ReLU leaks more via eff_rank" is not a leakage claim.
- **The whitened sensitivity needed a 3‑way disjoint cross‑fit.** A winner's‑curse denominator (the subspace and the noise estimated from the same seeds) faked a "sharpens with N" result; after the fix it is flat in N. Report the rank‑based permutation p‑value, never the floor‑dependent absolute d².
- **K‑non‑convergence is a sufficient artifact disqualifier.** A real quantity stabilizes with more samples; the retracted "sharpens with N" doubled at 2× samples.
- **Convergence controls are load‑bearing regardless of the result's direction.** The multi‑class arc flipped (2× → retracted → reversal) each time an underfit arm was cleaned up.

---

## 5. Coverage of what was asked (nothing silently dropped)

| Ask | Status | Evidence / caveat |
|---|---|---|
| Rank sweep, powers of two + "low rank actually fails" | **DONE** | job 581629; r=1 capacity floor (635386); Fashion 10‑class bounded‑out → MNIST carries it |
| Make the base model better / >89% | **DONE (binary)** | binary 88 → 96.7% (245178); Fashion 10‑class genuinely *harder* (94.9→87.7%) |
| Effect of fine‑tuning on classification accuracy | **DONE (from rigor logs)** | memorization↔utility tradeoff, asymmetric: MNIST 10‑class ~unchanged (Δ≈−0.02), binary moderate (Δ≈−0.10), Fashion 10‑class catastrophic (up to Δ≈−0.75) — §2.5 |
| Multi‑class (>2) + re‑run across datasets | **DONE (MNIST+Fashion)** | CIFAR deferred; `multiclass_replication_plan_DRAFT.md` never finalized |
| Consolidate the leakage story | **DONE** | `leakage_story_consolidated.{md,pdf}` + canonical figure; STATUS contradictions now fixed |
| Supervisor top ask: activation × anchor × linearization crux | **IN PROGRESS** | 152‑config rescore done (389926): two‑cluster (relu/leaky/selu leak most, NOT hardswish; within‑smooth trend flips +0.58) — no clean smoothness law; wc‑provisional. Matched‑wc LR‑band re‑sweep + feature‑stability‑vs‑T + flowers band now running (user‑approved closure) |
| Direct weight inversion axis | **Phase 0 DONE** | SSIM 0.57 @N=4 (500913); superposition wall; SimuDy reframe email to Gal SENT ~2026‑08‑21 |
| Gradient bridge decoder | **Phases 1–2 DONE** | 0.951 cosine (956994); pixel end‑result = 0/40 (§2.3) |

---

## 6. Honest limitations & open loops (the things to close)

1. **Pixel reconstruction does not beat baseline (0/40)** — the headline gap. The rigorous next step is the J‑composed Fisher bridge (adapter‑space Fisher → image‑space via the data‑latent Jacobian → Cramér‑Rao/Fano floor) to tell whether the pipeline is information‑limited or decoder‑limited.
2. **The activation crux (supervisor's top ask) — actively closing.** The 152‑config rescore (389926) delivered the first‑pass ranking (two‑cluster: kinked relu/leaky/selu leak most; no clean smoothness law; wc‑provisional). The user‑approved closure — matched‑wc LR‑band re‑sweep + feature‑stability‑vs‑T + flowers band — is running now; that turns the wc‑provisional first‑pass into the settled leakage ranking. (The rest of the program remains GELU‑only.)
3. *(Resolved — the SimuDy reframe email to Gal was sent ~2026‑08‑21; the direct‑inversion axis framing is settled on the supervisor side. Next supervision meeting: ~2026‑08‑31.)*
4. **The full‑FT‑vs‑LoRA "valley" comparison is built + audit‑passed but not yet run** (stage‑0 re‑running after a calibration‑ordering fix, job 375314); arms F/G unrun.
5. **The g₀ predictor is INDETERMINATE at scale** — declare a single canonical ρ (n=12 0.857 vs n=24 0.777) and resolve the USPS OOD counterexample with margin‑at‑scale.
6. **Record hygiene** — the active dataset‑sensitivity code + results are largely uncommitted (untracked package `__init__.py`); STATUS.md carries an old dead to‑do list; CLAUDE.md points at a superseded plan; the handover baton points at the wrong (Jacobian) thread. (STATUS's internal reconstruction contradiction is now fixed.)

---

## 7. What's next (priority order)

1. **Run the full‑FT valley wave** once stage‑0 is green — the headline "does the parameterization narrow the valley" comparison, with the B1 (dimension‑invariance) and B2 (SGD‑noise) gates before any headline read.
2. **Close the activation crux** — analyze job 857271; run the feature‑stability T‑sweep — so the smoothness study the supervisor asked for is actually finished, not GELU‑only.
3. **Build the J‑composed Fisher bridge** — turn the 0/40 into an interpretable "how far from the information floor is the decoder?"
4. **Resolve the g₀ predictor at scale** (canonical ρ + the USPS arbiter) and run the dataset‑sensitivity make‑or‑break gates (the full H validation, S2 twin‑shielding).
5. **Commit the active program + refresh the record** (handover, CLAUDE.md pointer, prune STATUS).
