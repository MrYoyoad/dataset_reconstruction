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

## §0. QUESTION INDEX — read this before commissioning anything

**Why this exists.** On 2026-09-17, five experiments were commissioned whose answers were already in this
repository, one of them in this file. The ledger is organised by **claim** — *what have we said* — but a lane about
to spend compute asks a **question** — *what do we already know about X*. This index is the second lookup. **Add a
line here whenever you add a row below.** See LESSONS_LEARNED, "the bottleneck is not measurement".

| live question | rows / sources that already bear on it | state |
|---|---|---|
| Can a chart be both narrow enough to be identifiable and faithful enough to contain the data? | `STATUS.md` top section (two-wall table, both releases); G1–G4 | **Answered for public PCA: no width works, and the fidelity wall alone carries it.** Open for other chart *families* |
| How coarse is a public chart against the landing gate? | **G3** (k=32, oracle ladder) and `STATUS.md` two-wall table (k=66 and k=384). **Ruled 2026-09-17 (55f1929): NO SCALAR IS CANONICAL** — see §0.1 | **Never quote a single ratio.** The quantity is chart error *at a named width* over the gate *as a bracket* |
| Does a foundation-model embedding dissolve the chart problem? | H1–H3 | Answered: no. Shared-concept chart is *worse* |
| Does depth add information additively? | **M4** (9,18,20,20 — rank-preserving regime only), **job 688036** (**NOT** 692603); **M6** real-encoder (job **354535**, attested); **k-sweep** (job **355531**); `STATUS.md` T5.2 section; audit F11–F14 | **T5.2 FALSE as stated** — refuted as an EXACT rank on the SYNTHETIC net (F11, integer ranks, real gap). **At REAL scale T5.2 overpredicts the USABLE rank ~2×** (M6: effective rank ~51% of ambient 784 vs T5.2's near-full 96%, rung-independent, PASS #1, 2nd audit pending) — but the real-net spectrum has no fp64 gap, so this is an EFFECTIVE-rank statement, NOT an exact-rank refutation (6e's exact-vs-effective distinction). The corrected law's **VALUE is unconfirmed** — matches neither law (`matches_corrected` reads 1e-10=402 not the 418 quoted from 1e-12: rung-selection), ladder monotone-unconverged (~430–443, above corrected's 417); still a candidate as a *proof*. **k-sweep (355531): depth BUYS chart-constraint width additively — the cap RISES with depth, `cap(L)=min(L(r−N)+(m−1), nesting ceiling 417)`, reducing at L=1 to the capacity line `k<m+r−N` (DERIVED from the N(m−1+r−N) variety, confirmed sharp-in-k at r=16 jobs 467914/469120; r never varied there, so r=108 is an r-axis extrapolation, not measured — 6e's 110 is that L=1 value r-extrapolated). `k*=417` is where the ceiling binds and the two laws diverge — a threshold in k AND L (k=512 flips at L=5). Window OPEN: identifiability reaches the ceiling with depth; whether fidelity suffices at that k needs a landing gate measured ON MNIST — the CIFAR gate must NOT be applied (cross-construction). Earlier "depth free at buildable width / windows disjoint" WITHDRAWN 2026-09-17. ⚠ **The k\*/1.8× RANK numbers are PENDING a gap/tau→0 check (2026-09-18):** on a real deep φ the spectrum may have no gap, making them crossings toward the ambient dimension, not integer ranks (T5.4's effective-rank caveat, now on the real encoder). T5.2's refutation is restated against the saturation curve (effective rank ~51% of ambient vs T5.2's near-full 96%) and survives; the exact k\* may not. Clean-FP64 A100 355778 settles it.** M4's sweep is rank-preserving and discriminates neither |
| **Which model can discriminate the two additivity laws?** | Profile arithmetic, 2026-09-17 (`83`, pre-submission) | **The 3-layer MNIST MLP provably CANNOT** — profile 784/784/690, a drop of ~94 arriving too late, so `q_1 ≈ k_1` and both laws saturate at `k_1` immediately. **Use the 15-layer deep encoder** (`mnist_mlp_d15w1000.pth`, profile 784→692→219→187→138→104→85): adapting through the 692→219 cliff gives corrected ≈319 against T5.2's 692. **Pixel-space, no chart — a THEORY test of the rank law, not an attack configuration** (`k_1 = 692` is ~10× the measured identifiability cap) |
| How many adapted layers buy a chart of width k? | `STATUS.md` T5.2 section; **§19a** / `STATUS.md:323` ladder, jobs 218345/218346 | Upper bound only. **Precondition (`rank M_l` per layer) already measured and unfavourable** |
| Is a shared/tied `A_0` across layers a defence? | **M5**; audit F11 counterexample | Answered: **no**, three ways. **But job 674726 asserts the opposite from byte-identical data with `passed: true` — mark superseded in place, do not cite** |
| Does the certificate survive when a layer's inputs drift? (Gal's main question) | M1–M3 | **ZERO PASSES.** M4/M5 have PASS #1; M1–M3 have none. Nothing here reaches Gal yet |
| Can the attacker tell when the certificate is dead, or contaminated? | M2, M3; audit F6 | Rank death **is** attacker-visible; contamination is **not**. Lifetime result is **defender-facing** |
| Does a chart restore identifiability? | N1–N4 | Yes — **measured with the release's own ORACLE chart**. Establishes the mechanism a chart must supply, not that any buildable chart supplies it |
| Are identifiability and recoverability the same thing? | N2; seed-known LM arm (in flight) | **Provisionally no** — nullity 0 with zero landings. Two separable jobs for a chart |
| **Does the certificate/identifiability framework survive depth?** | **P1–P4** (E1B depth sweep, 2026-09-18); `STATUS.md` top section | **The law is exact wherever evaluable; depth destroys the MEASUREMENT between 8 and 12.** Say it in that form — stated loosely it reads as a refutation. One config, 7 points, PASS #1 only |
| **What are the strongest recipe-free recovery cells, with controls?** | **E8** (EMNIST new class, 8/8), **E9** (CIFAR hidden layer 253/400 8/8, head 171/400 8/8, wrong-release 0/400, raw privates 0/400), E7 (digits k=6, chart off → 0) | All on-chart. **Read E9's NOT-shown before quoting: `k32_onchart/` is the vacuous replica, not the cell** |
| **Does a CNN change the depth picture?** | **Q3, Q4** (job 355907) | **At zero drift depth is moot on a CNN: one conv layer's certificate, applied at every position, pins any chart up to pixel space.** Under drift the conv widths cap `rank B_T` and the dense count grows far below `N·T` |
| **What is the MNIST landing gate?** | **Q1** (letters, 3-layer strong MLP), **Q2** (digits on d15: the release records nothing) | Letters: cliff between 0.0069 and 0.0139, no all-8 end. **The depth-window encoder still has no gate** — `d15_letter_a` running |
| **Can a pretrained decoder be the chart?** | **Q5** (SD VAE, smoke); full 356034–356051 | On 32×32 CIFAR the decoder's own autoencoding error is 3–4.5× the gate: CEILING-BOUND before any chart is built |
| **Are the base models fully trained?** | **Q0** | Measured; three used checkpoints pass, the d15 and conv nets do not (twins exist, not substituted) |
| **Does class composition (one added class vs two) matter?** | WP2 cells 355926–355983 (running); earlier mixed a+t 302280 = 8/8 | Pending |
| **Can the chart be bootstrapped from the recovery?** | WP4 355987 (MNIST, done, unread) / 355988 (CIFAR, running); **Q6** | Pending; the reference point is settled (Q6) |
| What does an attacker actually recover, and how many starts does it take? | D section; E1–E5; **E6** (yield vs start budget) | Both measured; see D for the route split |

### §0.1 The public-chart shortfall — why no single number is canonical

Two of our own records disagreed, in **two** places. (1) **Different widths**: G3 reads the ladder's public row at
`k=32`; the table reads `k=66` (the identifiability cap) and `k=384`. Chart error falls with width, so these are not
the same quantity. (2) **The gate is a BRACKET, not a point** — the ladder observes landings at one width and none at
the next, so the gate is an interval: **0.0124–0.0186** (motorcycle/MLP), **0.0045–0.0090** (keyboard/CNN). **G3
divides by the bracket midpoint** (an undisclosed step: `0.3176/0.0155 = 20.5`, `0.2432/0.00675 = 36.0` — the
"20–35×"); the table divides by the **near end** only.

| chart width | motorcycle / MLP | keyboard / CNN | overall |
|---|---|---|---|
| k = 32 | 17.1–25.6× | 27.0–54.0× | **17–54×** |
| k = 66 (the cap) | 9.9–14.9× | 22.9–45.7× | **10–46×** |
| k = 384 (far past the cap) | 5.9–8.8× | 13.2–26.4× | **6–26×** |

> **The claim that survives every convention, and the one to quote:** at the **most generous reading available
> anywhere in this grid** — widest chart, far end of the bracket — the shortfall is still **5.9×**, and it **never
> approaches 1 at any width under any convention**. That is robust to whichever number a reader prefers, which is
> worth more than any individual figure.

**Index rule (2026-09-17):** point at **NOT-shown** columns as well as findings. Half of this evening's corrections
were about what a row does *not* establish rather than what it does.

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

| E6 | **Fashion-MNIST yield is 6 of 8 at 400 starts** (job 556643). The widely-quoted **4 of 8 is the same cell at 150 starts** (job 473802, `landed 9/150`) — one cell read at two budgets, **not** a wrong number and its correction. | Fashion-MNIST cell, start budgets 150 and 400 | as the cell | **that either figure is a property of the release.** A landed count without its start budget is **not a quantity** — yield rises with budget, so any count quoted bare understates or overstates by the budget's ratio. The SUPERSEDED verdict on 4/8 does **not** mean the archive was wrong | 473802, 556643 | read-rows (c4) | **SETTLED** |

| E7 | **The recipe-free certificate route recovers private images from random starts, and the chart is what makes it solvable.** Below the certificate line at `k=6`, `r=16`, `N=8`, `n'=7`: **51%** of 2000 random starts land on a recorded private image and **7 of the 8** distinct images are found, with `recipe_used=false`, `labels_used=false`, `oracle=[]`. The **same cell with the chart off returns 0.0 on every metric** (`truth_on_chart=false`), at all of k=6/8/10 and in both image sets. | `step67_cert_below_706721.jsonl`, 12 cells | FP64; below the line; truth on the chart | **that an attacker measures this** — landing and floor are scored against ground truth; the attacker-side statement is E3. That all 8 are found — image 3 is missed in the confident set. Yield falls with k (0.51 at k=6, 0.18 at k=8, 0.13 at k=10) | 706721 | read-rows (GM) | **HELD, one PASS** |

| E8 | **A new class is recovered recipe-free from random starts: EMNIST letter 'a' added as an 11th class to a digit model, `r=64`, `k=32`, `N=8`: 38.4% of 500 random starts land on a recorded image and all 8 of 8 are found** (`recorded_images_found=[0..7]`, `argmin_err` 8.9e-13, 0 degenerate starts). At `k=16`: 82.4% land, 7 of 8 found. | `step80_trainprec_letters_760909.jsonl`, part B, fp64-trained release | FP64; below the line (`n'=8`, line 56); `new_pca` chart; truth on the chart | that a **public** chart of the new class does this — `new_pca` is built from the added class. Reduced-precision releases are a separate thread and are not quoted here | 760909 | read-rows (GM) | **HELD, one PASS** |
| E9 | **Colour images, adapter behind a nonlinearity, wrong-release control at zero.** CIFAR-10, public PCA chart `k=32`, `r=64`, `N=8`, 400 random starts: hidden layer **253/400 land, 8/8 found** (residual median 2.8e-14); classifier head **171/400, 8/8**; **wrong-release control 0/400, 0/8**; **raw (off-chart) privates 0/400, 0/8**. Landings recomputed from `X_found` against `H_proj` at `rel<1e-2` and match each job's own `result.landed`. | `experiments/cifar/charts/{L2_pca_lm_onchart_k32, L3_pca_lm_onchart_k32_zero_row, L3_pca_lm_onchart_k32_WRONGRELEASE_control, L3_pca_lm_raw_k32}/release_and_search.pt` | FP64; on-chart privates; base acc 0.5435; `T=200` | **that raw photographs are recovered — they are not**: the same head cell with raw privates lands 0/400, so every recovered image is a chart projection. **`experiments/cifar/k32_onchart/` is the vacuous-criterion replica (RESULT.md §5) and must not be read as this cell** — it gives 14 landed / 1 of 8 against pixels | 272363–272380 (layer 2), 279960 (head), control per RESULT.md §5 | read-rows (GM) | **HELD, one PASS** |

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

> **UPDATED 2026-09-17.** **M4 and M5 now carry PASS #1** (`read-rows`, `notes/m4_additivity_verification_2026-09-17.md`):
> their job id is **688036**, not 692603, and M4 **discriminates neither additivity law** — the path is
> rank-preserving (`k_1=20`, `q_l=[9,9,9,9]`, widths 30), so the corrected law provably collapses to the stated one
> there. **M1–M3 remain at ZERO PASSES.** And the **stated rank law `r − N'` was REFUTED and corrected today** to
> `(min(r,n_l) − N')_+`: the de-rigged check `T2_rank_C_matches_AS_WRITTEN_r_minus_Nprime` **FAILS** at job **351745**
> while the corrected form passes. **Do not cite "the rank law" to 688036/692603 — cite the corrected law to 351745.**
> A second PASS is still required on every line before any of this reaches Gal.
>
> **STATUS FOR EVERY LINE BELOW: UNAUDITED unless marked otherwise.** `experiments/multilayer_cert/RESULTS.md` is dated
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
| M6 | **Real-encoder discrimination: the corrected law (F11) is confirmed and T5.2 is refuted.** On the real 15-layer MNIST encoder, adapting from layer 3 through the encoder's 784→687→**217** rank cliff, the stacked certificate rank saturates at **418 = corrected `min_j(d_j + Σ_{l<j} q_l)` = 417** (±1), **1.8× below** T5.2's `min(k₁,Σq_l)` = 756; layers 5–8 add nothing (nesting ceiling `d_3 + q_1 + q_2 = 217 + 200`). The rank-preserving control saturates at both laws' common value. | `mnist_mlp_d15w1000.pth`, pixel arm, `r=108`, `N=8`, first adapted layer 3 | zero-drift certs; FP64; **THEORY test, NOT an attack** — the pixel `k₁` is ~10× the k≤66 cap, and at k≤66 both laws coincide (66) | that any buildable chart (k≤66) shows it — it does **not**; and the exact 418 elbow — rungs below 1e-10 undercount (402/403) and are **provisional** pending the clean-FP64 A100 run (354537) | 354535 (attested: `script_sha dd81201f5399`) | **read-rows (83)** | **PASS #1** (2nd audit pending) |

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

## N. The chart restores identifiability — measured as a nullity, not argued (E1B, 2026-09-17)

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| N1 | **Constraining the features to a k=12 chart takes the seed-free nullity from 232 to 0.** The chart restores identifiability by removing exactly the feature directions that trade against the unknown seed. | affine two-routes release; nullity of the residual Jacobian **at the truth** | fp64; that release and chart | that a solver *finds* it — this is identifiability, not recovery | 350928, 350940 | read-rows | HELD, one PASS |
| N2 | **The decision line is PARTIAL, not yes/no.** Invertible with a known seed (nullity 0 at 512 unknowns); **not** invertible with a free seed (nullity 232); a chart restores it. | same | same | a bare "the dynamics determine the features" — the seed is what breaks it | 350928, 350940 | read-rows | HELD, one PASS |
| N3 | **The entire ambiguity is a seed-against-features trade.** Of the 232-dimensional family, all 232 directions move the features and **none** moves them with the seed held fixed. | same | same | that the family is a property of the features alone | 350928, 350940 | read-rows | HELD, one PASS |
| N4 | **Reducing out the seed changes nothing.** Nullity 232 both ways — the reduction removed 1023 unknowns and exactly 1023 of the rank. | same | same | that the reduced parametrisation is weaker or stronger; it is neither | 350928, 350940 | read-rows | HELD, one PASS |

**N5 — the methodological rule, and it is the strongest thing here.** An approver lane derived **by counting** that
the seed-free arm had a **32**-dimensional solution family. The measured nullity is **232**. Counting was wrong by
a factor of seven, and the correction cost one small job.

> **An equation count is a hypothesis about identifiability, never a measurement of it.** Where the map is
> differentiable, the nullity of its Jacobian at the truth is cheap, exact, and owes nothing to a solver. Compute
> it instead of arguing.

This is ledger line **C3** — "a raw equation count is never evidence of identifiability" — now demonstrated rather
than asserted, and it is the answer to the "~200 equations" concern in the form Gal can check.

**Consequence for sequencing, and it should be settled before either is pursued.** Two threads are currently
supplying *more equations*: the multilayer additivity of M4, and Gal's own perturbed-inputs suggestion. **If a
chart already takes the seed-free nullity to zero in the configuration the attack actually runs in, both answer a
question that no longer binds.** Sequence them against N1 rather than running them in parallel.

---

## P. Depth and the limit of measurability (E1B depth sweep, 2026-09-18)

> Read from the rows by the GM lane on 2026-09-18: `results/e1b/depth_vs_measurability_matched.jsonl`
> (7 cells, budget scaled with depth) and `results/e1b/depth_vs_measurability.jsonl` (7 cells, fixed budget).
> One configuration throughout: `N=8, T=20, r=12, k=24, m=11, width=256`, 576 unknowns against 516 equations.
> **PASS #1 only — a second independent read is required before any of this reaches Gal.**

| # | text | measured on | holds under | NOT shown | job ids | register | status |
|---|---|---|---|---|---|---|---|
| P1 | **The law is exact at every depth where it can be evaluated.** Nullity is **80 against a predicted 80** at depths 2, 3, 4, 6 and 8, with no drift. | matched arm, 5 cells | fp64; that one configuration; lightly-trained nets (train acc 0.368 to 0.535) | that it was checked at depth > 8 — **it cannot be**, see P2. Seven points, one configuration, **no fitted form** | matched arm rows | read-rows (GM) | **HELD, one PASS** |
| P2 | **What depth destroys is the measurement, not the law.** The gap at the cut collapses from **4.4e+07 at depth 8 to 1.48 at depth 12** and 1.26 at depth 16, the rank ladder spreads from 31 to 67 to 91, and the nullity stops being a property of the Jacobian (83, then 144, against the predicted 80). | matched arm, depths 8/12/16 | same | **that the law FAILS at depth.** This is measurability in fp64, **not** exact-arithmetic rank — in exact arithmetic the deep rank may be unchanged | matched arm rows | read-rows (GM) | **HELD, one PASS** |
| P3 | **The trainedness confound is controlled at the boundary**: train accuracy is 0.5346 at depth 8 against 0.5286 at depth 12, so the collapse is not deeper-nets-are-less-trained. | matched arm, depths 8 and 12 | same | that trainedness is controlled **across the whole sweep** — it is not; accuracy falls to 0.400 by depth 16 | matched arm rows | read-rows (GM) | **HELD, one PASS** |
| P4 | **The `rank_exists` flag fires only in the matched arm.** In the fixed-budget arm it is `false` at **every** depth including 2, while the nullity there still reads 80 = predicted. | both arms, 14 cells | same | that the two arms disagree on the *nullity* — they agree at depths 2 to 8. The disagreement is in the **verdict flag**, so any "rank exists" statement must name the matched arm | both files | read-rows (GM) | **HELD, one PASS** |

**Why this group matters for sequencing.** It lands directly on the meeting's agreed step #1. The framework stays
correct and stops being *applicable* — which is more useful than either "it holds" or "it fails", and it is the
kind of statement that has to be said in exactly that form or it will be read as a refutation.

---

## Q. The 18 Sept package — CNN rank law, class composition, decoder chart, bootstrap chart, MNIST gate (2026-09-18)

Plan + audit: `notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md`. Every row below is a builder's own read of its
rows (**PASS #1 at most**); a second, independent read is pending on all of them. WP2 (58 cells), WP3 (18 cells) and
WP4-CIFAR are still running and have no rows here yet.

| id | text | measured on | holds under | NOT shown | jobs | register | status |
|---|---|---|---|---|---|---|---|
| Q0 | **Base training is a measured gate.** `mnist_mlp_strong` (99.83 % train / CE 6.3e-3), the CIFAR CNN (99.83 % / 9.8e-3) and the over-trained CIFAR MLP (100 % / 6.8e-4) pass train acc ≥ 99.5 % ∧ CE ≤ 1e-2; `mnist_mlp_d15w1000` (98.69 % / 4.9e-2), `mnist_conv_deep` (99.66 % / 1.03e-2), `mnist_conv` fail. Fully-trained twins exist and are NOT substituted this round. | all 8 checkpoints, full splits, FP64, each file's own loader | the gate as stated | that a failing base invalidates earlier rows — the depth-window rows stay on the original d15 by design | 355833; twins 355840/355869/355870, re-gate 355842/355873 | read-rows (WP0) | HELD |
| Q1 | **The MNIST letters landing gate**: recovery survives a measured chart error of **0.0069** (7/8 found, 151/400) and is gone at **0.0139** (0/8, and at every larger error). **No all-8 end exists** — one letter is never found even in the exact chart (ε = 0: 7/8, 127/400). Public PCA 0.3123 → 0/8. Wrong-release control 0/8, truth residual 4.9e-1. | `mnist_mlp_strong`, EMNIST a, N=8 (idx 323 693 173 92 129 696 298 767), r=64, T=400, k=32, raw privates, 400 LM starts, 12-ε ladder | this release; nothing between 0.0069 and 0.0139 was measured | a gate for the DEPTH WINDOW — that is digits on the 15-layer net; quoting Q1 against the k-sweep's MNIST fidelity ladder is the cross-construction error | 355845–355858, generator 356068 | read-rows (WP5); 2nd read PASS (rows L29–42) | **SETTLED** (2 reads) |
| Q2 | **A confident batch on a fully-confident base records nothing, and then the certificate is degenerate.** Head adapter on `mnist_mlp_d15w1000` trained on the k-sweep's eight test digits: `‖B_T‖_F = 0.37`, `σ₈/σ₁ = 8.7e-11`, softmax residual at `W₀` ≤ 1e-3 on 5/8 (≤ 2e-3 on 6/8); **0/400 in every cell including ε = 0**, and the **wrong-release control reaches the same ~1e-17 objective**. Read literally: not an information failure, a release with no recording. | d15, N=8 digits (idx 723 923 2619 3739 5981 4186 6644 913), r=64, T=400, k=32, 400 starts | known classes, base at 98.69 % train | a gate for the d15 encoder — none exists on this release; `d15_letter_a` (new class) is the follow-up | 355883–355896 | read-rows (WP5); 2nd read PARTIAL→fixed (5/8) | HELD (2nd read applied) |
| Q10 | **Depth kills the SEARCH, not the information: on the 15-layer encoder a recording release with an exact certificate is unreachable from random starts, even with the truth's own chart.** `d15_letter_a`: rank `B_T` = 8, `‖B_T‖_F` = 1.267, `σ₈/σ₁` = 1.6e-5, certificate residual at the true letters 1.7e-14, rank C = 56, truth floor 2.8e-28 — and **0/400 landings at ε = 0** (best image error 0.51), every start stalling ≥ 1.6e-17 (5.6e10 × the floor). Wrong-release control 5.5e13 above its own floor and ~500× above the true release's stalls, so the landscape is release-shaped (unlike Q2, where control and truth coincided). No landing gate is readable on this encoder at any ε. | `mnist_mlp_d15w1000` (Q0 FAIL, used unchanged), EMNIST a as an 11th class, same 8 letters as Q1, r=64, T=400, k=32, raw privates, 400 LM starts, ALL 14 cells | LM from random starts; FP64 | an information or identifiability failure — the certificate is exact here; a gate for the depth window; any other solver or initialiser | 356075–356088 (complete), smoke 356072, generator 356517 | read-rows (WP5) | HELD |
| Q3 | **On a CNN one conv layer pins the whole chart at zero drift, so depth is moot.** A conv certificate acts at every spatial position: measured `q₂ = k` at every k ≤ 784 (24 certificate rows × 49 positions), `q₃ = k`; corrected law = T5.2 in all 96 zero-drift cells (first=1 depths 1–6, first=3 depths 1–4, first=5 depths 1–2; **pre-registered VACUOUS**), stacked rank at k from the fp16 rung down (the bf16 rung reads 739–767 at k=784); the dense-only control saturates at the bottleneck width 128 with a real gap (5.7–7.5e7 at L=2, 0.9–2.1e8 at L=1). The dense-style count `q_l = rank C` (audit's estimate) is refuted by the factor `P_l`. | bottleneck conv net (1→64→128→8→256, dense 1024→1000, head; Q0 PASS), r=256, N=8, first ∈ {1,3,5}, maxL 6, k ∈ {16…784}, seed 1 | zero drift, FP64, abs floor 1e-10 on certificate rank | a solve or an attack (algebraic rank at the truth, `claim_class` on every row); the corrected law confirmed or refuted on a CNN — no separating regime exists on any conv spec with `rankC·P_l ≥ k` at the first live conv; conv-4 `q = 96 < 128` unexplained; single seed/net/r | 355907 (`377d841-dirty`, script e4792c7e2149) | read-rows (WP1); 2nd read PARTIAL→fixed (96 cells, bf16 rung, gap range) | HELD (2nd read applied) |
| Q4 | **Under drift on the CNN the dense count grows far below `N·T` and width caps the rest.** Dense `N'(T) = 7/15/18/21/26` at T = 1/5/20/100/400 (M2's `N·T` would be 8…3200); conv 4 saturates at its patch dimension 72; conv 2 / conv 3 `rank B_T` are width-capped at 128 / 8, the 8-channel bottleneck killing the drifted certificate (residual 0.87). The first-adapted conv keeps `N' = 9` at all T (`|C_T − C_zd|/|A₀| ≈ 6e-14`). In the SEPARATE dense-first configuration (first=5: dense+head only, inputs frozen) dense `N' = 7` at T ≤ 20 because one digit's base softmax residual is 1.4e-13 (7 images imprint above 1e-12), 8 from T = 100. | same net; the `N'(T)` series from the all-six-modules arm (first=1), the `N'=7` sentence from the dense-first arm (first=5), lr 0.01, no divergence, loss 9.8e-2 → 2e-5 | one lr, one seed; `N'` at tolerance 1e-12 (at 1e-10 the T=400 dense count reads 21, not 26) | that M2's `N' = N·T` is wrong in general — M2 is synthetic/dense; this is one real CNN cell and is evidence against `N·T` as the governing rate there | 355907 | read-rows (WP1); 2nd read PARTIAL→fixed (configuration split, tolerance) | HELD (2nd read applied) |
| Q5 | **A pretrained SD VAE cannot be a chart for 32×32 CIFAR on fidelity alone**: autoencoding ceiling on the ladder's eight motorcycles **0.055** at 8× (0.078 at 4×, 0.21 at 2×) against the gate bracket 0.0124–0.0186 → **CEILING-BOUND** (pre-registered). Attacker-anchored local latent chart 0.36 at k=16 (pixel PCA 0.34); oracle latent anchor 0.045; global latent PCA worse than pixel PCA. | `sd-vae-ft-mse` (diffusers 0.32.2), motorcycle set (ladder idx 23 32 89 56 41 26 34 61), k=16, K=64, 200 Adam steps, FP32 | exact replicate/block-mean resize (floor 0.0000); w=0 at the oracle anchor reproduces the ceiling exactly | any other decoder; the remaining local-chart rows (K/k grid, 356034–356098 running). **Interim, all three sets: ceiling 0.055 / 0.024 / 0.021 (motorcycle / keyboard / letters-a at 8×) vs brackets 0.0124–0.0186 / 0.0045–0.0090 / 0.0069–0.0139 → CEILING-BOUND on every set; global latent PCA < pixel PCA at every k; letters-a attacker-anchored local chart 0.170 at k=32 beats pixel PCA 0.235 — first chart to beat PCA here — at 12× the gate's high end.** | smoke 355910; interim 356034–356098 | read-rows (WP3) | HELD |
| Q6 | **For raw privates the truth's pixel projection is NOT the chart's argmin of the certificate objective**: objective at the recovery is 0.003–0.57× its value at the projection on 8/8 letters; an oracle start converges to a chart point 10–80× below the projection's objective. Every off-chart cell must reference the oracle-start chart optimum, not the projection. | MNIST letters-a release, generic EMNIST PCA-32 chart, 10 starts (smoke) | nonlinear φ (the strong MLP) | a landing statement — smoke only; full rows 355987/355988 | 355880, 355915 | read-rows (WP4, smoke) | HELD |
| Q8 | **Bootstrapping the chart from the recovery STALLS (MNIST letters).** Variant A (class recognition → class chart): top-1 7/8, calibration on projections 90 % (86.9 % for `a`); class chart optimum-to-truth 0.401 vs generic 0.433 vs wrong-class 0.501, yet recovery-to-truth 0.758 vs 0.604 — one-mode collapse inside the class chart (7/8 candidates the same upright A). Variant B (local chart on the recovery's 200 nearest public images, 4 rounds): per-slot chart fidelity 0.27–0.31 (< per-class PCA 0.40, < random-anchor control), but slots collapse onto 2/8 truths and nothing improves after round 1; per-truth recovery 0.84 → 0.82. | `mnist_mlp_strong`, EMNIST a (same 8 as Q1), r=64, T=400, k=32, raw privates, 200 starts (8 local), generic EMNIST PCA-32 round 0 | Q6's reference (oracle-start chart optimum); no void fired | any landing vs the raw truth (0/8 everywhere, by construction at chart error ≥ 0.25); CIFAR (355988 running, variant A will be VOID by calibration); other K, k, round-0 charts | 355987 | read-rows (WP4) | HELD |
| Q9 | **The certificate is composition-blind; the linearised route is not** (INTERIM). Pooled PCA k=16, 200 starts: certificate 8/8 for eight a's, eight t's, 4a+4t on two rows, 4a+4t on one row, landed 157–191/200, flat over T = 1…400; k=48 collapses alike in all compositions (basin, not composition). At T=1 `C` is label-independent by `row(B_1) = col(A_0 H D^T) = col(A_0 H)` (rank D = N): two-row and one-row mixed land the identical 157/200. NTK lora:varpro: a → 0,0,1,8,1 /8 at T = 1,5,20,100,400; t → 8,8,2 at 1,5,20; mixed → 8,0,·,·,8; CIFAR mixed (over-trained MLP) → 5,5,8 at 1,5,20; every miss has residual ≥ 1e-3 over a ≤ 1e-15 floor (search failure, never alias). | `mnist_mlp_strong` (Q0 PASS), EMNIST a/t, N=8, r=64, on-chart privates, seed 1; CIFAR over-trained MLP, motorcycle+bottle | pooled `pca` only so far | per-class vs pooled charts, AE, k=32/48 for t and same-row, all CNN cells, CIFAR T ≥ 100 — running | 355926–355983; matched earlier rows 308862/308863 (same config to every printed digit at T=5/20) | read-rows (WP2, interim) | HELD |
| Q7 | **Two-walls (e1b C7) recomputed on the gate's own images — conclusion stands, motorcycle shortfall larger.** `two_walls.py:58` selects with `np.random.RandomState(seed)` (idx 80 84 33 81 93 17 36 82), the ladder with a torch generator at `seed+7` (idx 23 32 89 56 41 26 34 61); C7's stored means reproduce to 4 decimals on its own set, so the difference is the images. On the ladder's images, pixel-PCA mean error / shortfall vs gate low / high: motorcycle k=66 **0.2702 → 21.8× / 14.5×** (C7: 0.1845, 14.9×), k=384 **0.1596 → 12.9× / 8.6×** (C7: 0.1094); keyboard k=66 **0.2107 → 46.8× / 23.4×** (C7: 0.2058, 45.7×), k=384 **0.1222 → 27.2× / 13.6×**. No public-PCA width satisfies both walls on the gate's images at either end of either bracket. | ladder index sets, FP64 closed-form PCA on the class's public train pool, C7's convention | — | that the two_walls numbers were wrong on their own images (they reproduce); e1b/RESULT.md C7 still quotes the other set and should be updated by lane 6e | 356106 (`results/decoder_chart/c7_index_sets_356106.jsonl`) | read-rows (WP3) | HELD (recomputed; flag resolved) |

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
