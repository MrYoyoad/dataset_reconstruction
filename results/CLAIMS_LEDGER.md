# Claims ledger

**Owner:** GM lane (narrative gate + second auditor seat). Created 2026-09-07 under §12 of the 7 Sept brief.

**The rule this file exists to enforce:** *every sentence that reaches Gal or the co-authors traces to a line
below.* If a sentence has no line, it does not go out. If a line's **NOT shown** column covers what the sentence
implies, the sentence is wrong even when its number is right.

**Columns.** `text` — the claim in the language of §2. `measured on` — the cell, with `q` and `N` side by side.
`holds under` — the conditions, precision included. `NOT shown` — what a reader would wrongly infer.
`register` — `derived` / `read-rows` / `read-function` / `read-prose`; **read-prose never counts as a PASS**.

**Status.** `SETTLED` two independent PASSes · `HELD` awaiting a second PASS · `WITHDRAWN` kept in place with its
reason, never deleted.

---

## A. Structure of the release

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| A1 | The trajectory closes on two batch-sized coefficient matrices, so the release is a deterministic function of the data and the seed's reading of the private span. | forward check, 54 rows | `B0 = 0`; SGD-class; one adapted layer; fixed inputs; FP64 | survival under Adam, augmentation, or a moving input | 487882 | derived + read-rows | SETTLED |
| A2 | The seed enters only through its reading of the private span, so the unknown part is rank x span-dimension, not rank x feature-dimension. | same | as A1 | that the complement carries *no information* — it carries no *constraint* | 487882 | derived | SETTLED |
| A3 | The certificate annihilates every recorded representation, with no recipe, no labels and no seed. | cells where `q = N` | `q = N`; SGD-class; `B0 = 0`; non-shared module | exactness when `q < N` — see A4 | 701679 | derived + read-rows | SETTLED |
| A4 | Certificate exactness is conditional: at `q = N` the residual sits near 1e-16; under partial recording the worst representation climbs toward order one while the median stays small. | 26-logit head, 20 images | FP64; recorded counts 19 to 14 | that the loss is uniform — it concentrates in a handful | 725918 | read-rows | SETTLED |

## B. What the release records

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| B1 | What leaks is what the model had to learn: a representation enters at the scale of that example's accumulated error, so a model that already fits its data records nothing. | 40 batches x 4 encoders | FP64; output layer | that this bounds a *stronger* attacker; it bounds this channel | 631392, 627166 | read-rows | SETTLED |
| B2 | The release determines `q`, never the number of photographs. | derivation + measurement | all cells | that `N` is recoverable — it is not | 709507 | derived | SETTLED |

## C. Counting, and its one-sidedness

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| C1 | The recipe-free channel is **closed** where the recorded span fills the adapter: margin = min(r,d) - min(q*p, d), evaluable from architecture and batch size before any release exists. | ViT-B/16, DINO ViT-S/16, ResNet-18/50 | real photographs, pretrained, ranks 8 to 64 | **the converse.** ONE-SIDED: sound when it says closed, **silent when it says open** | 273322, 279182, 296789 | read-rows | SETTLED |
| C2 | The capacity condition is **necessary and not sufficient**: where the map from chart coordinates to the adapted input is affine, every blend of the private representations is an exact solution at any chart dimension. | pixel-layer cell, linear chart | affine chart-to-input map | that a larger `k` or a better solver helps — it cannot | approver R1 | derived + read-rows | SETTLED |
| C3 | A raw equation count is **never** evidence of identifiability; only the rank of the stacked Jacobian on the chart answers it. | — | — | that N*sum(s) bounds anything | — | derived, rule 2 | SETTLED |
| C4 | The capacity line for the reduced channel is measured **sharp to one unit of k** on synthetic and MNIST releases. | N=8 recovered at 27, alias at 28; N=14 at 21/22; N=4 at 31/32 | fp64; reduced `B_T` channel; those releases only | that it is established **for the full factor pair** — there it is a *candidate*, not a result | 467914, 481079, 593146 | read-rows | SETTLED |
| C5 | The single-layer tangent rank is **proved**; the stacked-Jacobian rank **across layers is open**, and it is the quantity the count actually needs. | — | — | that the count is unsupported — the single-layer half is a theorem; only the cross-layer stacking is open | E4b when it runs | derived | SETTLED |

## D. Routes

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| D1 | Certificate and linearised representer share a zero set **where every representation is recorded**; replay is a third, strictly stronger object. | derivation | full recording | that the equivalence holds at partial recording — it does not | approver R12 corrected | derived | SETTLED |
| D2 | Identifiability is a property of the release, the chart **and the route**: on one release the certificate recovers 0 of 60 while **19 of 60** replay starts recover every image **at the 1e-2 landing criterion** (at a **2.2e-15** worst-image bar the count is 18; at a strict **2.0e-15** bar it is **16**; best single image in the best start 6.6e-16). | affine cell, `starts=60`, `q=N=8` | same release, same chart, matched starts; **tolerance named with its count** | that either route dominates generally; **that a typical start recovers** — `replay_worst_err_median` is 1.17, so the 19 are the starts where *every* image clears the bar | 331384 | read-rows | SETTLED |
| D3 | Any statement of the form "cannot identify by **either** route" is **false**. | as D2 | — | — | 331384 | derived from D2 | SETTLED |

**Field hazard on D2 — the one that produced the error.** `replay_best_err_min` is the **best single image in the
best start**, not a per-start worst case. Counting against it gives 18 at 2.0e-15; counting against
`worst_image_err`, which is what "recovers every image" means, gives **16** at that bar. Verified at the 60
per-start rows (GM, `read-rows`): worst-image ≤2.0e-15 → 16, ≤2.2e-15 → 18, ≤1e-2 → 19. **Any count of starts that
recovered everything reads `worst_image_err`.**

**And the general lesson, from the lane that made the original slip.** The phrase that propagated was the *weakest*
form of a true claim, because it was the one a reader could disprove by opening the file. **A claim stated tighter
than the data supports does not read as more confident; it reads as unchecked**, and it puts the reader in the file
hunting the discrepancy instead of in the argument. The honest bound here is barely looser and survives inspection.

**Quotation hazard on D2, and it is the live one.** The airtight form is the **negative**: *identifiability is not
determined by the release and the chart alone.* The inclusion `{truth} ⊆ {replay=0} ⊆ {certificate=0}` is **proved**;
the measurement is that inclusion being **strict**. The positive form — "a property of all three" — follows, but it
is the form that survives being quoted out of context, and section headings and glossary lines are exactly what get
quoted. **Where a bare positive appears without the negative beside it, cite this line rather than the source.**

## E. Attacker-side instruments

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| E1 | Below the certificate's line the residual **selects the landings**: the fraction reaching each landed image's own floor equals the fraction landing on a private image, with no false certification in any cell. | digits k=6; letters k=16/32 | FP64; below the line; **per-image floor** (at_floor indexes by the landed image, not the cell minimum) | that this is an *attacker* number — the floor comes from a from-truth solve | 706721, 760909 | read-function + read-rows | SETTLED |
| E2 | The residual remains a sound witness in ordinary single precision: 0.324 certified against 0.324 landed over 3,000 starts, no disagreement either way. | letters k=32, FP32-trained | **experimenter arm**, start_attacker_buildable false; rows read `q = 11` for `N = 8` | that an attacker achieves 0.324 — they cannot compute a from-truth floor | 304540, 319712, 335734 | read-rows | SETTLED |
| E3 | An attacker ranking blind by final objective separates perfectly: precision 1.000 against a disjoint-release null of 0.000, base rate 0.324. | same cell | attacker-available information only | breadth — see E4 | 304540 | read-rows | HELD, one PASS |
| E4 | **Breadth frontier.** With the window scaled to the budget, distinct images climb 1, 2, 5, 7 and reach 8 only where precision falls to 0.648 — the last image is bought by accepting a third of the list is wrong. | letters cell | — | that more starts buy the last image; **a fixed-k distinct-image count is not an attacker statistic** and *falls* as starts grow, 5 at 300 and 2 at 3000 | 304540 | read-rows | HELD, one PASS |
| E5 | Against a shadow-model membership attack the certificate matches on separation and wins only on assumptions: no shadows, no recipe, no distributional assumption. | ViT-B/16 head, 128 shadows | frozen backbone | that it detects more | 287241 | read-rows | SETTLED |

## G. Chart fidelity — the oracle ladder (28 cells, complete; read in 2026-09-07)

Approver read both examples off the logs; **two pre-registered predictions held** (the public-chart arm and the
wrong-release arm were both called at zero before the rows existed).

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| G1 | Exact landing survives a chart error up to a bracket and then stops: **between 1.24% and 1.86%** on the MLP example, **between 0.45% and 0.90%** on the CNN one. | motorcycle/MLP 212/180/71/0 landings at measured error 0.0000/0.0062/0.0124/0.0186; keyboard/CNN 224/115/0 at 0.0000/0.0045/0.0090; every higher cell 0 | fp64; oracle chart built around the privates | **a single threshold.** The two examples **DISAGREE by about a factor of two and must never be averaged** — they differ in backbone *and* class simultaneously, so the difference is attributable to neither | 435271-435322 | read-rows | SETTLED |
| G2 | The public PCA chart returns **zero** landings in both examples, at its own measured error of 0.3176 and 0.2432 — **predicted before the rows**. | same | same | that a public chart fails for lack of starts; it fails on fidelity | 435271-435322 | read-rows | SETTLED |
| G3 | The quotable consequence, **with both ends**: the public chart is **twenty to thirty-five times coarser** than the gate. | G1 and G2 | same | a midpoint, a single factor, or an average of the two examples | 435271-435322 | derived from G1, G2 | SETTLED |
| G4 | Negative control: the wrong-release arm returns zero landings in both examples. | same | same | — | 435271-435322 | read-rows | SETTLED |

**Axis warning, binding on every figure and sentence from this group.** The perturbation *parameter* is **not** the
measured chart error — their ratio falls from 0.62 to 0.40 across the range. Any plot or claim uses the **measured**
axis. A figure drawn against the parameter would compress the very quantity the group exists to report.

---

## H. Charts in feature space (E4a, 2026-09-07)

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| H1 | **The shared-concept chart is worse, not better.** Identity is 15–16 directions; pose, lighting, crop and background are 205–208. **At matched fidelity** the shared-concept parametrisation costs 15 + 8×205 = **1655** unknowns against 8×205 = **1640** per-image. And it generalises: `k_shared + N·k_nuisance` exceeds `N·k_nuisance` for **any** positive concept axis, so shared-concept can only pay when the nuisance axis is small — precisely the condition the measurement refutes. | DINO ViT-B/16, CLIP ViT-L/14 | both dimensions from the **same 95% cumulative-variance rule** in the same script — **precondition verified, PASSES** | that the margin is large; it is ~1%, and the force is in the generalisation, not the gap | E4a (674521, 674524) | read-rows | HELD, one PASS |
| H2 | **Moving to a foundation-model embedding does not dissolve the chart problem.** Target-conditioned at k=32: pixels **0.2432–0.3176**, DINO **0.4091**, CLIP **0.2634**. Nothing gets dramatically cheaper, and the best cell anywhere remains far from the gate. | pixels = oracle ladder's public-PCA rows, **a chart fitted on public images OF the added class** (ladder docstring line 16) — i.e. **target-conditioned**; embeddings = E4a **target** arm | **every number target-conditioned**, same k, same hold-out | that pixels beat embeddings **generally** — **CLIP target (0.2634) sits INSIDE the pixel range**, so the claim is false for that backbone; and DINO is worse by ~1.3–1.7×, **not** "roughly doubles" | 435271-435322 (pixels), 674521/674524 (embeddings) | read-function + read-rows | HELD, one PASS |
| H3 | The affine-hull degeneracy **reproduces in feature space** at every cell of both backbones and both regimes: blends sit about **twice as close** to the public chart as the privates themselves. So it is a property of what public charts represent well, **not a pixel artefact**. | same | — | that it is caused by the pixel parametrisation | E4a | read-rows | HELD, one PASS |

**H2 FAILED its second audit, 2026-09-07, and the reason is the one an expert would test first.** As first
ledgered it set a **target-conditioned pixel** chart (0.2432–0.3176) against a **universal embedding** chart
(DINO 0.5991) and attributed the difference to the *space*. Conditioning is worth a lot at every k, so most of that
gap was the **regime**, not pixels-versus-embeddings. Target against target the gap is 1.3–1.7×, and **CLIP sits
inside the pixel range**, making the original claim false for one of its two backbones.

**The rule that catches it, and it is new: a comparison carries the CONSTRUCTION of both sides, not just their
values.** This was not a bad number. It was two good numbers that do not belong in the same sentence. Every number
in a comparison names its regime **inside the claim**, since the regime is the field that went wrong.

**Provenance failure, corrected 2026-09-07, and it is the third of its kind today.** H2 was first ledgered with a
pixel residual of **0.25**. That number occurs in exactly one place in this repository: the **docstring of the E4a
script**. There is no job behind it. **A number sourced only to our own prose — including prose inside code — is not
a measurement, and quoting it back as support is circular.** Replaced with the ladder's measured public-PCA rows.

**H1 is a plan change, not a table row.** §6.2 proposed shared-concept as the regime that *reduces* the unknown
count, illustrated at concept 32 against nuisance 8, and said explicitly that E4a would measure whether reality has
that structure. It does not — **the structure is inverted**. This *removes* a regime rather than adding one.

**BLOCKED FROM TRAVEL — "the best public feature chart is 16 to 44 times too coarse."** Arithmetically right,
but it divides a **feature-space** residual (DINO/CLIP) by a gate measured in **pixel space** on the CIFAR releases
with a different search. The gate is a property of a release, not a constant, and its transfer to feature space is
**unmeasured**. It goes out only with the transferability named open in the same sentence, or not at all. **Use H2
instead** — like-for-like, needs no conditional, and answers Gal's realism objection without anything to take apart.

**NOT A RESULT — the nonlinear-chart arm.** It is **non-monotone in k**, which proves an optimisation failure.
It must never be cited about nonlinear charts in **either** direction.

---

## M. Multilayer certificate — Gal's own main question (7 Sept runs, **UNAUDITED**)

> **STATUS FOR EVERY LINE BELOW: UNAUDITED, ZERO PASSES.** `experiments/multilayer_cert/RESULTS.md` is dated
> 7 Sept 17:01 and contains **no occurrence of the string AUDIT** — not a missing second PASS, no audit log at
> all. It has stood complete and uncertified for ten days, and it is the document answering the extension
> question Gal named as his main one. **Nothing here may reach him or the co-authors until two independent
> PASSes exist**, and by this project's standing rule an audit that exists only in messages does not exist.
> Rows: `survival_692603.jsonl` (216 rows, 54 configs × 4 layers), `theory_checks_{674726,683234,686846,688036}`.

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| M1 | **The certificate is exact at depth, at any drift** — not perturbative. Residual at machine precision across four decades of drift (0.2% to 923%). | 110 rows where `rank B_T = N'`; 13 deep rows above 100% drift | **`rank B_T = N'`**; FP64; random MLP | that it survives *without* the rank hypothesis — see M3 | 692603, 688036 | read-prose (GM) | **UNAUDITED** |
| M2 | **What depth costs is rank, with a hard lifetime.** The training span inflates maximally, `N' = N·T`, so `rank C = min(r, n_l) − N·T` and the certificate is **empty** once `T ≥ min(r,n_l)/N`. The death is discontinuous. | `N' = N·T` in 116/129 deep rows, `≤` in all; 12 rows at `rank C = 0` | same | that small drift is the governing condition — **it is small training span**: few steps, few examples, or drift in few directions | 692603, 688036 | read-prose (GM) | **UNAUDITED** |
| M3 | **Where the rank hypothesis fails the certificate is contaminated, not weakened** — median residual 2.3e-4, max 0.27, no small parameter. | 62 rows, last layer, width below `N'` | same | **and this is the sting: `rank B_T = N` is NOT an attacker-side certificate of exactness at depth**, because the attacker cannot observe `N'` | 692603 | read-prose (GM) | **UNAUDITED** |
| M4 | **Depth adds information additively** up to two ceilings: stacked Jacobian rank 9, 18, 20, 20 for one to four layers, matching `min(k₁, Σ q_l)` exactly. **Two layers close a k=20 chart that one layer cannot.** | `k = 20`, four layers, `r − N = 9` per layer | same | — **but this is the direct answer to the "~200 equations" concern**, and it is a *rank*, not a raw count | 692603 | read-prose (GM) | **UNAUDITED** |
| M5 | **A prediction of ours was refuted.** Tying adapter initialisations across layers does **not** collapse additivity — shared-seed and independent both give 9, 18, 20, 20 — so a shared `A₀` is **not a defence**. T5's defence claim is downgraded to a conjecture about tying the whole adapter. | same | same | — | 692603 | read-prose (GM) | **UNAUDITED** |

**Two things the write-up itself flags, and they belong in any relay.** Everything is a **random FP64 MLP**; the
span-inflation law `N' = N·T` is the claim most likely to change on a trained network with structured features,
and the lifetime bound in M2 rests on it. And where `N' > N` the certificate constrains the **span**, not the
individual images, so per-image norms are not the legitimate metric — sections reporting residuals and ranks are
unaffected, but any reconstruction work must adopt principal angles from the start.

**Audit hazards to carry, from CLAUDE.md's own ground rules for this track.** The load-bearing check is the
exactness one (T2) and its tolerance must **not** be loosened on a FAIL. Report **three** outcomes, not two — a
cell that tested nothing is distinct from a pass and a fail, and a **vacuous** deep test is exactly what an eager
reader turns into a positive. Rank thresholds need an **absolute** floor: a relative-only test calls a numerically
zero matrix full rank, which reports a closed channel as open.

---

## F. Withdrawn, kept in place with the reason

| # | text | why withdrawn | replaced by | date |
|---|---|---|---|---|
| F1 | "An adapter on the input layer read through a linear chart cannot identify two or more images, **by either route**." | Replay escapes the degeneracy — measured. | D2, D3 | 2026-09-06 |
| F2 | The same statement **as a defence**. | Real mathematics, vacuous as protection: **the remedy belongs to the attacker**, who picks the chart. | C1 as one-sided; and: no function of the release alone can certify a release is safe | 2026-09-06 |
| F3 | "Self-verification does not survive realistic arithmetic, about 3% certifiable against 29.7% landing." | Artefact of scoring against a historical **absolute** cut no FP32 release can reach. | E2 | 2026-09-06 |
| F4 | "The collapsed attractor sits at the certificate floor." | The *ideal* blend does; the point the solver reached is orders above it. | C2 | 2026-09-06 |
| F5 | A fixed-k distinct-image breadth count. | The window fills with the easiest image; the count falls as starts grow. | E4 | 2026-09-07 |
| F6 | "A linear chart beats a learned one." | Three seeds: learned median 151/400 against linear 171 from **one** seed; one learned seed reaches 176. | *Over-training the chart hurts* — 640 epochs worst in all three seeds | 2026-09-07 |

---

## Standing prohibitions, enforced here

Never in Gal-facing or co-author-facing text: reduced-precision results unless asked; SimuDy's memory table as our
speedup; "recovers scale"; "exact acceptance test"; "works for linear heads"; "N from the release";
"feasible/infeasible"; a raw equation count as evidence of identifiability.

**The one most likely to slip** is the last, because the worked numbers in the brief are raw counts and read
persuasively. Any sentence quoting them must carry the chain **raw rows, then per-example restrictions, then rank
on the chart, then remaining joint degrees of freedom**, or it does not go out.

## Open, and blocking nothing

- `q` versus `N` is now the project's notation. Every earlier claim written with `N` must be re-read for whether it
  meant the span dimension or the photograph count; E2's cell is the live example, reading `q = 11` for `N = 8`.
- Whether a recovered representation is recognisable as an image is unmeasured, and it is the load-bearing gap.
