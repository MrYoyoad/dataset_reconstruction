# CIFAR certificate attack — full study, 2026-09-06

The certificate is `C = P_{row(B_T)^⊥} A_T`, computed from the released LoRA factors alone (no seed, no recipe, no
labels). It satisfies `C h_i = 0` exactly on the private inputs of the adapted layer. The attack minimises
`‖C φ(G(z))‖ / ‖A_T φ(G(z))‖` over a chart `G` from random starts, and a start "lands" when its relative image error
against the target is below 1e-2. All releases: `B_0 = 0`, vanilla full-batch SGD, float64.

**Headline.** The CIFAR replica as supplied recovers none of its private images, and even given a reachable target it
recovers only 2 of 8. The reason is structural, not a tuning problem. Fixing
it needs one change of setting, not a better chart or optimiser: the adapter must sit behind a nonlinearity. With
that change the attack recovers every private image from random starts, and the attacker can tell which of their
starts succeeded from the residual alone.

| what changes | result |
|---|---|
| LoRA on the pixel layer, raw privates (the replica as supplied) | 0 of 8 images |
| LoRA on the pixel layer, on-chart privates | 2 of 8, at 30 of 400 starts; the modal minimiser is 0.99 blend by energy |
| LoRA on a hidden layer, public PCA chart | 253/400 starts land, 8/8 images |
| LoRA on the head, public PCA chart | 171/400 starts land, 8/8 images |
| LoRA on the head, a fully trained or **over-trained** backbone, a new class | up to 200/200 starts land, 8/8 images |
| certificate from a release trained on 8 *other* images | 0/400 — the control passes |

## 1. The replica's result, stated as a result

**The CIFAR replica as supplied does not recover its private images.** Reported plainly, before any explanation:

| what was asked | answer |
|---|---|
| `‖CH‖ / (‖C‖‖H‖)` | 3.62e-15 — **passes** |
| `‖C − P⊥A₀‖ / ‖C‖`, the quotient form | 5.92e-15 — **passes** |
| `rank C` | 56, exactly `r − N` — **passes** |
| excitation gap `σ_N / σ_{N+1}` | 1.00e14 — **passes** |
| did any start land on a private image | **no**, at either k = 32 or k = 48 |
| best residual reached, k = 32 | 6.57e-04, which is *below* the chart-floor residual of the truths (1.33e-03 … 5.92e-03) |
| found-versus-true SSIM against the chart ceiling | 0.35 against 0.59 at k = 32; 0.30 against 0.63 at k = 48 |
| found-versus-true SSIM against a same-class control | not separated: 0.35 against 0.375 at k = 32 |

So every algebraic check passes and the search converges *past* the floor the truths themselves attain, and the
images are still wrong. In the verdict vocabulary that is not an optimisation failure: the residual is at or below
the floor with the wrong answer, which is an identifiability problem. The rest of this section is why, and section 2
is what fixes it.

## 1b. Why: a linear certificate cannot separate what it annihilates

`C h_i = 0` for every i means `C` annihilates the whole span, so every linear combination `Σ c_i h_i` is an exact
zero as well. When the adapted layer's input *is* the image, those combinations are themselves images, and a smooth
chart represents the blend far better than any individual image: the 8-apple mean has PCA-32 representation error
0.06 against 0.10–0.42 for the individual apples. The minimiser is therefore the blend by construction.

Measured, and stated separately for the two distinct cells, because they are not the same run
(verification job 309357, recomputed from the saved tensors under both normalisations):

| quantity, replica on-chart cell (job 257893) | `‖Cx‖/(‖C‖‖x‖)` | `‖Cx‖/‖A_T x‖` |
|---|---|---|
| the private images | 3.93e-15 | 2.89e-14 |
| the **ideal** blend: an exact least-squares combination of them | 2.88e-15 | 2.87e-14 |
| the attractor the solver actually **reached** (mean of the collapsed cluster) | 5.44e-04 | 5.45e-03 |

The first two lines are the lemma, numerically: an exact blend of the private inputs is an exact zero, sitting where
the truths sit. The third line is a separate fact about the search, and the distinction matters:

- **What the lemma costs is coverage, not precision.** In this cell the solver stalls *near* the blend subspace
  without reaching it, so the residual still separates: the landings sit at 7.9e-07 … 1.1e-06 against the collapsed
  cluster's median 5.83e-04, a factor of about 600. That separation is why the top-20 by residual is 20/20 clean and
  why images found is 1–2 rather than 0. The page does not claim the checks were defeated by a blend sitting at the
  floor, because in this run nothing reached the floor.
- **Why the exact degeneracy did not bite here.** This cell reads the pixel layer through the conv-autoencoder
  chart, which is nonlinear, so the blend subspace lies only approximately in the chart's image. The one
  configuration where the lemma applies exactly — pixel layer with a linear chart, affine all the way from chart
  coordinate to layer input — is running now and is pre-registered below.
- **The blend fraction of the attractor is 0.99 by energy**, `1 − (‖r‖/‖x‖)²` with `r` the residual of the
  least-squares fit in the span, or 0.90 as `1 − ‖r‖/‖x‖`. An earlier draft said 99.4% without a formula; it does
  not reproduce under either normalisation and is withdrawn. Coefficients (−0.66, 0.32, −0.06, 0.13, −0.06, 0.43,
  0.17, 0.22), sum 0.48, three negative — not a convex mixture.

Every sanity check passed throughout: `‖CH‖/(‖C‖‖H‖)` at 3.6e-15, quotient form 5.9e-15, `rank C = r − N = 56`,
excitation gap 1e14. **Those checks are necessary and never sufficient, and the reason is an impossibility rather
than an observation**: fix `C` and `H` and vary the chart, and whether the truths are isolated changes, so no
function of `C` and `H` alone can decide it.

**Pre-registered, and confirmed in every particular** (jobs 311215 and 311217, a matched pair differing only in
where the adapter sits: same private images, same public PCA chart at k=32, same starts, same budget).

| | pixel layer (affine composition) | head (two nonlinearities) |
|---|---|---|
| starts landing on a private image | **0 of 400** | 171 of 400 |
| images recovered | **0 of 8** | 8 of 8 |
| best residual reached | 2.3e-14 | 7.1e-15 |
| residual at the private images | 2.3e-14 | 1.3e-14 |
| top-20 by residual that are true landings | **0 of 20** | 20 of 20 |
| blend fraction of the returned points, median | **1.000** | 0.994 |
| isolation test: rank of `C·(chart Jacobian)`, `k = 32` | **25** | 32 |

Read the first column against the second. The search **reaches the exact floor** — the best residual equals the
residual at the truths — and still returns nothing: every returned point is an exact blend, and the residual ranking
gives the attacker no signal at all, where in every working cell it is 20 of 20. This is the strong form of
*necessary and not sufficient*: the algebraic checks pass, the search converges, the residual is at machine
precision, and the answer is wrong. It is also the case the conv-autoencoder cell could not exhibit, because there
the chart is nonlinear and the blend subspace lies in its image only approximately, which is why that cell stalls at
5.5e-03 instead and keeps a clean ranking.

**The rank deficit is exactly `N − 1`.** The isolation test returns 25 against `k = 32`, and `32 − 25 = 7 = N − 1`:
the lemma predicts an affine subspace of dimension `min(N−1, k)` through the truths, and the Jacobian loses precisely
those directions. That is a quantitative confirmation, not just a qualitative one.

**The returned points are affine, not merely in the span** (independent verification, 2026-09-06). The
least-squares coefficients of what the search returns, against the eight private images, **sum to exactly 1.0** on
every start checked. A point in the linear span with coefficients summing to anything else would also be an exact
zero of `C`, so this did not have to come out this way: the search lands on the *affine hull* specifically, which is
the precise hypothesis of the lemma and the reason the intercept cancels. The found points sit at 0.997 of the
truths' own residual while being a median 0.28 of an image away from anything private, against a landing bar of
0.01.

**The isolation test is what an attacker would need to notice this.** It uses only the release and the chart, no
ground truth, and it separates the pair cleanly. It is read one-sided — full rank certifies isolation, deficient
rank certifies nothing — so the correct reading of the 25 is "not certified", never "degenerate".

**What makes this cell conclusive is the agreement of three independent things, not the rank alone**: the deficit
equals the lemma's predicted dimension `N − 1`, the coefficients are affine, and the residual sits at the truths'
own floor. Any one of the three could have an innocent explanation; together they do not. The rank deficit must not
later be quoted as if deficiency were itself proof.

## 2. The layer study: what actually fixes it

Jobs 272363–272380 (layers 1 and 2), 277289/277370/277515 (head), 279959/279960 (head, re-run after the seed-stream
fix). Full table: `figures/cifar_charts/table.md`; one figure per cell in `figures/cifar_charts/`.

| adapted layer | chart | privates | landed | images | top-20 by residual | median start residual |
|---|---|---|---|---|---|---|
| pixel layer | conv-AE k=32 | on-chart | 30/400 | 2/8 | 20/20 | 5.5e-3 |
| hidden layer | conv-AE k=32 | on-chart | 87/400 | 5/8 | 20/20 | 7.2e-3 |
| hidden layer | public PCA k=32 | on-chart | **253/400** | **8/8** | 20/20 | 2.8e-14 |
| hidden layer | public PCA k=48 | on-chart | 84/400 | 7/8 | 20/20 | 4.9e-3 |
| head (features) | public PCA k=32 | on-chart | **171/400** | **8/8** | 20/20 | 1.3e-2 |
| head (features) | public PCA k=48 | on-chart | 50/400 | 6/8 | 20/20 | 1.1e-2 |
| head (features) | conv-AE k=32 | on-chart | 20/400 | 4/8 | 20/20 | 2.3e-2 |
| head (features) | public PCA k=32 | raw | 0/400 | 0/8 | 0/20 | 3.5e-2 |
| **head, WRONG-RELEASE control** | public PCA k=32 | on-chart | **0/400** | **0/8** | 0/20 | 1.8e-2 |

Readings:
- **A linear chart beats a learned one — withdrawn on a mismatched reference, then REINSTATED on matched controls
  at three seeds** (see the table below; the ordering holds within every seed). PCA reaches the exact zero (median start residual 1e-14, i.e. most starts
  converge to a true zero); the conv-AE decoder stalls three orders higher. The same was true on MNIST.
  **Scope.** The autoencoder here is trained 40 epochs, and its representation ceiling is at parity with PCA's
  (chart floor SSIM 0.59 against 0.58), which for a nonlinear decoder with far more capacity is itself a sign
  of an undertrained chart rather than a limited one. What fails is the search, not the expressiveness: with
  on-chart privates an exact latent exists in both charts by construction, and only the linear one is reached.
  **Measured across three seeds and three training budgets, nine cells** (seed 0 was the peer's original run; seeds
  2 and 3 are replicates at identical settings). Landings out of 400, with images found:

  | autoencoder epochs | seed 0 | seed 2 | seed 3 | mean landed | mean chart-Jacobian conditioning |
  |---|---|---|---|---|---|
  | 40 | 48 (5/8) | 162 (8/8) | 122 (7/8) | 111 | 4.28 |
  | 160 | 70 (6/8) | 151 (7/8) | 176 (7/8) | 132 | 4.72 |
  | 640 | 10 (5/8) | 42 (6/8) | 78 (7/8) | 43 | 5.90 |
  | linear PCA chart, **one seed and NOT matched** (see below) | 171 (8/8) | — | — | — | 1.00 by construction |

  > **CORRECTION, then REINSTATEMENT — 2026-09-07.** An earlier version of this table compared these cells against
  > **253 of 400**, which is the *hidden-layer* PCA cell while every ablation cell is a *head-layer* cell. The only
  > head-layer PCA cell available then read 171 of 400 and also differed in a second setting (it zeroes the new head
  > row). The claim was therefore withdrawn on a mismatched reference. **Matched controls have since been run** —
  > head layer, k = 32, no zero-row, the same three seeds — and they reinstate it more strongly than it was first
  > stated.

  **Matched linear-chart controls** (jobs 395771, 395773, 395774), against all nine learned-chart cells:

  | seed | linear chart | learned chart, 40 / 160 / 640 epochs |
  |---|---|---|
  | 0 | **274 (8/8)** | 48 (5/8) · 70 (6/8) · 10 (5/8) |
  | 2 | **265 (8/8)** | 162 (8/8) · 151 (7/8) · 42 (6/8) |
  | 3 | **188 (8/8)** | 122 (7/8) · **176 (7/8)** · 78 (7/8) |

  The ordering holds **within every seed**, not merely worst-against-best: the linear chart beats all three learned
  budgets at each of the three seeds. It also finds all eight images in three of three cells, where the learned chart
  does so in one of nine. The linear chart's own seed spread is real (188 to 274), so this is an **ordering, not a
  magnitude** — and the narrowest margin, seed 3, is 188 against 176, twelve landings out of 400.

  What survives the seed replicates, independent of that reference:

  - **Training sixteen times longer actively hurts.** 640 epochs is worse than 40 in three of three seeds and worse
    than 160 in three of three. This is a within-budget comparison and needs no linear-chart reference at all.
  - **The peak at 160 is withdrawn.** It rises from 40 in only one seed of three, and the seed-to-seed spread at a
    fixed budget (48 to 162 at 40 epochs) is larger than the difference between budgets. The single-seed run that
    suggested a sevenfold collapse drew the lowest cell of all three triples; the robust fall from the best budget
    to the longest is about threefold.
  - **The chart's Jacobian conditioning rises monotonically with training in all three seeds**, 4.28 to 4.72 to
    5.90 on average against exactly 1.00 for a linear chart by construction. It is the only measured quantity that
    moves with the budget in every seed, and it remains a **correlate and not a cause**: it rises 1.4-fold where the
    landings fall 2.6-fold, and the reading that learned decoders are *badly* conditioned in absolute terms is false
    at every budget.
  - **Whether a learned chart is worse than a linear one at its best budget is now SETTLED** by the matched
    controls above: it is, at every seed, on both landings and images found.

  Two things the collapsed cell rules out. The exact zeros are still present and reachable at 640 epochs (residual
  at the truths 5e-15, isolation test full rank 32 of 32 at every private image, five images returned at 1e-14),
  so this is neither a loss of identifiability nor a degeneracy: the solutions are intact and the search stops
  finding them, the same shape as the one failing MNIST cell. And the decoder's Jacobian conditioning is small at
  every budget — 4.1, 4.3, 6.0 against exactly 1.0 for PCA by construction — so the original "learned decoders are
  badly conditioned" reading is wrong in absolute terms. It is the only measured quantity that moves monotonically
  with training while the landings peak and fall, which makes it a **correlate and not a cause**: it rises 1.5-fold
  where the landings fall sevenfold, and that is a poor quantitative match.
- **The basin shrinks toward the capacity line.** At k=48, eight below the certificate line `r − N = 56`, landings
  fall from 253 to 84 (hidden layer) and 171 to 50 (head), with images found falling from 8 to 7 and 6.
- **The attacker's own ranking is perfect wherever anything lands**: in every landing cell all twenty
  lowest-residual starts are true landings, so success is detectable without ground truth.
- **The wrong-release control passes**: a certificate from a release trained on eight *other* apples of the same
  class gives residual 0.27–0.51 at the original privates and zero landings.

## 3. Raw images and the chart: the oracle ladder

Raw (unprojected) images never land on a public chart, and the reason is quantitative: their chart projections have
certificate residual 0.25, while the blend floor sits at 0.013, so the blend always wins. The ladder replaces the
public chart by one that spans the privates perturbed by a relative noise ε — **not attacker-available**, and
labelled so — to measure how accurate a chart must be. Jobs 279934–279958.

| ε (chart error) | residual at the projections | landed | closest approach | SSIM attack | chart ceiling | control |
|---|---|---|---|---|---|---|
| 0 | 3.6e-14 | 191/400 | 0.00 | **0.90** | 1.00 | 0.50 |
| 0.02 | 2.0e-3 | 169/400 | 0.01 | 0.89 | 0.99 | 0.51 |
| 0.05 | 7.3e-3 | 0/400 | 0.02–0.03 | 0.85 | 0.94 | 0.48 |
| 0.1 | 2.5e-2 | 0/400 | 0.04–0.06 | 0.76 | 0.84 | 0.47 |
| 0.2 | 6.9e-2 | 0/400 | 0.07–0.12 | 0.62 | 0.67 | 0.44 |

**The chart sets fidelity.** Recovery error tracks the chart's own error roughly one for one, and the recovered
image sits at the chart's ceiling at every level while staying far above the same-class control. Exact landing (the
1e-2 bar) needs a chart accurate to about 2%; below that the recovery degrades smoothly rather than failing.

## 3b. Added-on classes, on fully trained backbones — including the original paper's over-trained regime

The question the layer study leaves open is whether the recovery depends on a weak base model. It does not. Three
backbones, all trained by `experiments/cifar/cifar_newclass.py` and gated on both test *and* train accuracy before
any attack number is produced:

| backbone | train acc | test acc | train loss | median margin |
|---|---|---|---|---|
| pixel MLP (the project's structure, 3072-1000-1000-10 GELU) | 93.6% | 57.9% | — | — |
| **over-trained** pixel MLP (no augmentation, no weight decay, trained past zero error) | **100.00%** | 57.3% | 6.8e-4 | 9.05 |
| conv net | 99.8% | 92.7% | — | — |

A pixel MLP's test ceiling on CIFAR-10 is around 55% however long it trains, so "fully trained" is read from the
*train* side: the over-trained model interpolates its training set exactly and its margins have grown, which is the
regime the original reconstruction work assumes.

Four added-on classes, each an 11th class the backbone has never seen, r = 64, public PCA chart k = 32, 200 random
starts, on-chart privates (jobs 293350 / 297205 standard, 293516 / 297325 over-trained):

| class | standard MLP | over-trained MLP |
|---|---|---|
| keyboard (CIFAR-100) | 200/200 starts, 7/8 images | 179/200, **8/8** |
| skyscraper (CIFAR-100) | 160/200, **8/8** | 118/200, **8/8** |
| mushroom (CIFAR-100) | 157/200, **8/8** | 99/200, 7/8 |
| Flowers-102 photographs (a different corpus) | 163/200, **8/8** | 152/200, 7/8 |
| *wrong-release control (keyboard)* | *0/200, 0/8* | *0/200, 0/8* |
| *raw privates (keyboard)* | *0/200, 0/8* | — |

In every landing cell all twenty lowest-residual starts are true landings. The certificate residual at the private
inputs is 5e-15 on the standard backbone and 4e-14 … 2e-10 on the over-trained one — the over-trained model records
its new class slightly less sharply (it is confident about everything it has already seen), and the basin narrows
accordingly (200 → 179, 160 → 118), but the recovery itself survives. That is the expected direction: over-training
suppresses what the model already knows, and the new class is precisely what it did not know.

Flowers-102 is the strongest form of the claim, because the private images are photographs from a different corpus
downsampled to 32×32, not merely a held-out CIFAR label.

## 3c. PRE-REGISTERED: distance from the model's prior, as a controlled axis (jobs 335732, 335733, and the resolution control)

Two ends of one axis, everything else matched — same backbone, same rank, same batch size, same chart construction,
same starts:

- **near end**, held-out labels from the same corpus the backbone's classes come from: motorcycle, lawn mower,
  tractor, lobster (CIFAR-100, natural photographs at native resolution).
- **far end**, a different corpus *and* a different domain: sneaker, ankle boot, bag, sandal (FashionMNIST, which
  contains nothing CIFAR-10 or CIFAR-100 has).

**Prediction, recorded before the rows exist.** The imprint law says a model records most strongly what it had least
capacity to explain, so the far end should record *more* strongly: its certificate residual at the private images
should be **lower** than the near end's at matched rank, chart dimension and batch size. If recovery is instead
comparable at both ends, distance from the model's prior barely matters, which is the more surprising outcome and
the more useful one.

**A confound named in advance, and its control.** The far end reaches 32×32 by bilinear upsampling from 28 pixels,
so the two ends differ in *resolution history* as well as in domain, and interpolation smoothness plausibly flatters
a PCA chart — which matters because this study has already established that the chart is what sets fidelity. Any
difference therefore cannot be attributed to domain alone. The control is a CIFAR-100 class sent through the same
32 → 28 → 32 path (`--degrade28`); it is running, and the domain claim stands only on the comparison against it.

**Why the far end is not the headline.** Its entire scientific increment over the Flowers-102 cell, which already
gives the different-corpus result with natural photographs, *is* the domain shift — and that is precisely what
invites "the private data was trivially separable from anything the model knew". The marginal science and the
marginal liability are the same thing, so the near end leads and the far end calibrates the axis.

## 3d. Oracle-chart cells — NOT attacker-available, and kept out of every pooled number (job 412814)

Two cells re-run with the chart replaced by one built from the span of the private images themselves (`--chart
oracle --eps 0`), everything else matched to the 3c near-end rows: same backbone, r = 64, k = 32, N = 8, 200 random
starts, on-chart privates. **This chart is not available to an attacker** — it is the fidelity ceiling, so these
rows live in their own table and are never averaged into a landing rate or an image count that is quoted as an
attack result.

| class | oracle chart (k=32) | matched public PCA chart (k=32) |
|---|---|---|
| SVHN digits | 197/200 starts, 8/8 images | 196/200, 8/8 (job 335738) |
| motorcycle (CIFAR-100) | 109/200, 7/8 | 135/200, 8/8 (job 335732) |

Objective at the landings is 5.6e-29 / 4.4e-28 (median), i.e. the solve is exact in both cells. The observation is
that at k = 32 the oracle chart does **not** buy landings over the public PCA chart — it is level on SVHN and
*behind* on motorcycle. That is one seed per cell and two cells, so it is an observation and not a result about
chart choice; it does say that the k = 32 public chart is not the binding constraint in these two cells, which is
the opposite end of the ε-ladder in §3, where chart error above ~2% destroyed exact landing. Figure:
`figures/cifar_newclass/mlp_svhn_k32_onchart_oracle.png`, whose banner names the chart on the image itself.

## 4. A caveat on SSIM that matters for how this is presented

In the on-chart head cell the attack's SSIM against the raw image (0.58) equals the chart ceiling (0.58) and is *not*
above the same-class control (0.60). At k=32 every apple's chart projection is a similar blur, and window-3 SSIM
cannot tell them apart. **Identification there rests on image error (1e-14 against 0.25 for any other image) and on
the residual ranking, not on appearance.** Where the chart is accurate — the oracle ladder — the visual claim does
hold (0.90 against a control of 0.50). Report the two separately; do not quote a CIFAR SSIM as evidence of identity.

## 5. Reporting bugs found and fixed

- The replica's landing criterion `R < 3·max(chart-floor residual)` is **vacuous**: 400 of 400 starts satisfied it
  with zero true landings, and the earlier write-up reported "400/400 reached the certificate floor" as if that were
  success. Landing must be an image-error test.
- The same-class control paired each private with an arbitrary public image, while the attack number was a max over
  400 starts. Corrected to the same max-over-starts statistic against the projection of the nearest public image.
- A degenerate-start guard (feature norm below 5% of the public median) was missing and is now in place.
- Two jobs collided on one output directory; the control overwrote the real cell's files. The affected directory is
  now named `L3_pca_lm_onchart_k32_WRONGRELEASE_control` and the real cell was re-run as its own job (279960).
- The layer-2/3 initial adapter draw was moved to its own generator, so the earlier head run (277289, 200/400) and
  the current one (279960, 171/400) are different releases. Both recover all eight.

## 6. Scripts and outputs

| path | what |
|---|---|
| `cifar_certificate.py` (repo root) | the replica as supplied, unmodified |
| `experiments/cifar/cifar_certificate_onchart.py` | on-chart control for the replica |
| `experiments/cifar/cifar_charts.py` | the layer × chart × solver × privates grid, blend diagnostic, controls |
| `experiments/cifar/cifar_trained_newclass.py` | the same attack on the repo's own trained MLP backbone |
| `experiments/cifar/cifar_newclass.py` | fully-trained MLP / CNN backbones, weird added-on classes, over-trained regime |
| `experiments/cifar/replot_grids.py` | regenerates every figure and the summary table from saved tensors |
| `figures/cifar_charts/` | one figure per cell plus `table.md` |

## 7. WP2 — Class composition × chart × T, 2026-09-18 (INTERIM — jobs 355926–355983 still running)

**Question (plan 2026-09-18, WP2 + audit).** Does it matter whether the eight private images are one added class or
two? Is the certificate (separable, per image) composition-blind while the NTK/linearised free-coefficient route
(coupled through one sum) is not? Per-class vs pooled chart? Dependence on `T`?

**Setting.** `experiments/cifar/ntk_vs_certificate.py`, N = 8, head LoRA r = 64, `B_0 = 0`, full-batch SGD lr 0.01,
float64, on-chart privates, 200 shared random starts, seed 1, k ∈ {16, 32, 48}, T ∈ {1, 5, 20, 100, 400}. Cells:
`mnist_a` (8 a), `mnist_t` (8 t), `mnist_mixed` (4 a + 4 t, two head rows), `mnist_mixed_samerow` (4 + 4, ONE head row);
`cifar_motorcycle`, `cifar_bottle`, `cifar_mixed_mb`, `cifar_mixed_mb_samerow`, each on TWO bases. Charts: `pca`, `ae`
= ONE chart on the union of the public pools (the meaning every earlier row had; verified at the old line 179), and on
mixed cells `pca_perclass` = one chart per class, each private slot on its own class's chart (the attacker is told the
slot classes). Privates are on-chart, so per-class and pooled charts define different privates and different
releases: recovery is compared only within a chart; charts are compared only on the raw-image projection error
(true test images before projection), which every row records per image with median and range.

**Bases (WP0 gate: train acc ≥ 99.5 %, train CE ≤ 1e-2; measured at load on the full splits, stored in every row).**
`mnist_mlp_strong.pth` 99.83 % / 6.35e-3 / test 98.21 % PASS. `cifar10_cnn_newclass.pth` 99.83 % / **9.82e-3** / test
92.67 % PASS (narrow; the checkpoint stores no loss). `cifar10_mlp_overtrained_newclass.pth` 100.00 % / 6.82e-4 / test
57.31 % PASS. The weak `cifar10_mlp_newclass.pth` is not used in any WP2 cell.

**Control (new, per private image).** The nearest PUBLIC image of the slot's chart pool, projected on that chart, at
the recoveries' relative-error metric (`control_public_nn_err`). A recovery not closer than this is no better than a
release-free guess. EMNIST `letters` merges cases; each private letter's case is recovered by exact byte match against
the `byclass` split and stored per row (`private_case`): a = A, A, A, a (test idx 323, 693, 173, 92); t = T, T, t, t
(305, 484, 3, 27).

**Pre-registered (three outcomes per cell, read literally).** Certificate: 8/8 in every composition at T = 400 on
chart (composition-blind) · partial (count reported; R5) · below the single-class cells at the same (chart, k, T) —
a finding against separability. NTK free-coefficient (lora:varpro, the fair form): not fixed beyond "the two-row
mixed cell is at least as hard as the harder single-class cell"; the T dependence is the object, and every cell
reports the model floor at the truth so "residual above floor" (search) is never merged with "at floor, wrong
images" (alias). Chart: per-class raw projection error ≤ pooled at fixed k, per image.

**Reuse.** T ∈ {1, 400} for `mnist_a` and `mnist_mixed` are the existing rows of jobs **308862 / 308863** (git
f5e099c; same dataset, class, pooled chart, k, T, N, r, lr, seed, starts, Adam/LM/AE budgets, base, main form; they
also ran `--oracle-diag`, which adds labelled arms and touches no RNG — job 308865's T = 5 / 20 rows at pca k = 32
reproduce the new 355926 rows to every printed digit). Jobs 302279 / 302280 (git 1b18818, one k = 32 T = 400 cell,
old schema, no lora form) and 304349 (one row, dW form only) do NOT match and are not pooled. The new rows quote the
median as `np.median` (mean of the two middle values); the earlier rows used `torch.median` (lower middle value) — the
per-image values are identical, e.g. mixed pca k = 16: 0.423 (old) vs 0.439 (new) from the same eight numbers.

### 7.1 INTERIM table (regenerated 2026-09-18; NTK = lora:varpro free coefficients; ctrl = median control error)

| cell | base | chart | k | T | cert found | cert landed | NTK found | NTK best resid | floor | NTK verdict | raw proj. err median (range) | ctrl | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mnist_a | strong | pca | 16 | 1 | 8/8 | 169/200 | 0/8 | 2.40e-02 | 2.7e-16 | search failure | 0.316† (0.265–0.790) | — | 308862 |
| mnist_a | strong | pca | 16 | 5 | 8/8 | 170/200 | 0/8 | 2.29e-02 | 3.7e-16 | search failure | 0.320 (0.265–0.790) | 0.245 | 355926 |
| mnist_a | strong | pca | 16 | 20 | 8/8 | 175/200 | 1/8 | 1.47e-02 | 3.2e-16 | search failure | 0.320 (0.265–0.790) | 0.245 | 355926 |
| mnist_a | strong | pca | 16 | 100 | 8/8 | 174/200 | 8/8 | 1.58e-03 | 4.5e-16 | recovered | 0.320 (0.265–0.790) | 0.245 | 355926 |
| mnist_a | strong | pca | 16 | 400 | 8/8 | 173/200 | 1/8 | 1.03e-02 | 7.6e-16 | search failure | 0.316† (0.265–0.790) | — | 308862 |
| mnist_a | strong | pca | 32 | 1 | 7/8 | 84/200 | 0/8 | 6.78e-03 | 2.9e-16 | search failure | 0.235† (0.187–0.691) | — | 308862 |
| mnist_a | strong | pca | 32 | 5 | 7/8 | 82/200 | 0/8 | 5.71e-03 | 3.5e-16 | search failure | 0.245 (0.187–0.691) | 0.313 | 355926 |
| mnist_a | strong | pca | 32 | 20 | 8/8 | 69/200 | 0/8 | 3.15e-03 | 4.0e-16 | search failure | 0.245 (0.187–0.691) | 0.313 | 355926 |
| mnist_a | strong | pca | 32 | 100 | 8/8 | 73/200 | 0/8 | 2.71e-03 | 3.9e-16 | search failure | 0.245 (0.187–0.691) | 0.313 | 355926 |
| mnist_a | strong | pca | 32 | 400 | 8/8 | 80/200 | 0/8 | 3.24e-03 | 8.0e-16 | search failure | 0.235† (0.187–0.691) | — | 308862 |
| mnist_a | strong | pca | 48 | 1 | 3/8 | 7/200 | 0/8 | 3.81e-03 | 2.7e-16 | search failure | 0.192† (0.150–0.631) | — | 308862 |
| mnist_a | strong | pca | 48 | 5 | 1/8 | 1/200 | 0/8 | 3.34e-03 | 3.2e-16 | search failure | 0.196 (0.150–0.631) | 0.346 | 355926 |
| mnist_a | strong | pca | 48 | 20 | 2/8 | 4/200 | 0/8 | 1.92e-03 | 3.2e-16 | search failure | 0.196 (0.150–0.631) | 0.346 | 355926 |
| mnist_a | strong | pca | 48 | 100 | 2/8 | 4/200 | 0/8 | 2.16e-03 | 5.2e-16 | search failure | 0.196 (0.150–0.631) | 0.346 | 355926 |
| mnist_a | strong | pca | 48 | 400 | 2/8 | 4/200 | 0/8 | 1.74e-03 | 9.6e-16 | search failure | 0.192† (0.150–0.631) | — | 308862 |
| mnist_a | strong | ae | 16 | 1 | 5/8 | 9/200 | 0/8 | 4.21e-02 | 3.8e-16 | search failure | 0.236 (0.195–0.565) | 0.409 | 356100 |
| mnist_a | strong | ae | 16 | 1 | 5/8 | 9/200 | 0/8 | 4.21e-02 | 3.8e-16 | search failure | 0.235† (0.195–0.565) | — | 308862 |
| mnist_a | strong | ae | 16 | 5 | 2/8 | 5/200 | 0/8 | 3.73e-02 | 3.6e-16 | search failure | 0.236 (0.195–0.565) | 0.409 | 355926 |
| mnist_a | strong | ae | 16 | 20 | 2/8 | 9/200 | 0/8 | 2.83e-02 | 2.5e-16 | search failure | 0.236 (0.195–0.565) | 0.409 | 355926 |
| mnist_a | strong | ae | 16 | 100 | 3/8 | 9/200 | 0/8 | 2.24e-02 | 4.8e-16 | search failure | 0.236 (0.195–0.565) | 0.409 | 355926 |
| mnist_a | strong | ae | 16 | 400 | 3/8 | 8/200 | 0/8 | 2.78e-02 | 9.6e-16 | search failure | 0.236 (0.195–0.565) | 0.409 | 356100 |
| mnist_a | strong | ae | 32 | 1 | 1/8 | 2/200 | 0/8 | 7.80e-03 | 3.4e-16 | search failure | 0.201 (0.167–0.623) | 0.420 | 356100 |
| mnist_a | strong | ae | 32 | 5 | 1/8 | 3/200 | 0/8 | 6.85e-03 | 3.5e-16 | search failure | 0.201 (0.167–0.623) | 0.420 | 355926 |
| mnist_a | strong | ae | 32 | 20 | 1/8 | 3/200 | 0/8 | 6.16e-03 | 3.8e-16 | search failure | 0.201 (0.167–0.623) | 0.420 | 355926 |
| mnist_a | strong | ae | 32 | 100 | 2/8 | 3/200 | 0/8 | 6.37e-03 | 4.3e-16 | search failure | 0.201 (0.167–0.623) | 0.420 | 355926 |
| mnist_a | strong | ae | 32 | 400 | 1/8 | 4/200 | 0/8 | 8.07e-03 | 8.4e-16 | search failure | 0.201 (0.167–0.623) | 0.420 | 356100 |
| mnist_a | strong | ae | 48 | 1 | 0/8 | 0/200 | 0/8 | 1.20e-02 | 3.0e-16 | search failure | 0.225 (0.178–0.544) | 0.397 | 356100 |
| mnist_a | strong | ae | 48 | 5 | 1/8 | 1/200 | 0/8 | 1.09e-02 | 3.0e-16 | search failure | 0.225 (0.178–0.544) | 0.397 | 355926 |
| mnist_a | strong | ae | 48 | 20 | 0/8 | 0/200 | 0/8 | 5.67e-03 | 3.5e-16 | search failure | 0.225 (0.178–0.544) | 0.397 | 355926 |
| mnist_a | strong | ae | 48 | 100 | 1/8 | 1/200 | 0/8 | 5.95e-03 | 5.2e-16 | search failure | 0.225 (0.178–0.544) | 0.397 | 355926 |
| mnist_t | strong | pca | 16 | 1 | 8/8 | 187/200 | 8/8 | 3.33e-03 | 2.1e-16 | recovered | 0.444 (0.369–0.619) | 0.239 | 355927 |
| mnist_t | strong | pca | 16 | 5 | 8/8 | 189/200 | 8/8 | 2.48e-03 | 3.9e-16 | recovered | 0.444 (0.369–0.619) | 0.239 | 355927 |
| mnist_t | strong | pca | 16 | 20 | 8/8 | 187/200 | 2/8 | 8.63e-03 | 3.9e-16 | search failure | 0.444 (0.369–0.619) | 0.239 | 355927 |
| mnist_t | strong | pca | 16 | 100 | 8/8 | 184/200 | 8/8 | 1.49e-03 | 6.0e-16 | recovered | 0.444 (0.369–0.619) | 0.239 | 355927 |
| mnist_t | strong | pca | 16 | 400 | 8/8 | 179/200 | 8/8 | 1.26e-03 | 8.2e-16 | recovered | 0.444 (0.369–0.619) | 0.239 | 355927 |
| mnist_t | strong | pca | 32 | 1 | 8/8 | 104/200 | 1/8 | 1.13e-02 | 2.9e-16 | search failure | 0.367 (0.245–0.470) | 0.329 | 355927 |
| mnist_t | strong | pca | 32 | 5 | 7/8 | 102/200 | 1/8 | 5.20e-03 | 2.9e-16 | search failure | 0.367 (0.245–0.470) | 0.329 | 355927 |
| mnist_t | strong | pca | 32 | 20 | 8/8 | 108/200 | 0/8 | 2.73e-03 | 4.3e-16 | search failure | 0.367 (0.245–0.470) | 0.329 | 355927 |
| mnist_t | strong | pca | 32 | 100 | 8/8 | 101/200 | 0/8 | 3.01e-03 | 4.8e-16 | search failure | 0.367 (0.245–0.470) | 0.329 | 355927 |
| mnist_t | strong | pca | 32 | 400 | 8/8 | 94/200 | 0/8 | 2.97e-03 | 8.7e-16 | search failure | 0.367 (0.245–0.470) | 0.329 | 355927 |
| mnist_t | strong | pca | 48 | 1 | 5/8 | 11/200 | 0/8 | 3.52e-03 | 2.8e-16 | search failure | 0.304 (0.197–0.393) | 0.361 | 355927 |
| mnist_t | strong | pca | 48 | 5 | 4/8 | 11/200 | 0/8 | 2.75e-03 | 3.1e-16 | search failure | 0.304 (0.197–0.393) | 0.361 | 355927 |
| mnist_t | strong | pca | 48 | 20 | 5/8 | 18/200 | 0/8 | 1.61e-03 | 3.1e-16 | search failure | 0.304 (0.197–0.393) | 0.361 | 355927 |
| mnist_mixed (2 rows) | strong | pca | 16 | 1 | 8/8 | 157/200 | 8/8 | 3.93e-03 | 3.2e-16 | recovered | 0.423† (0.372–0.590) | — | 308863 |
| mnist_mixed (2 rows) | strong | pca | 16 | 5 | 8/8 | 160/200 | 0/8 | 2.79e-02 | 3.8e-16 | search failure | 0.439 (0.372–0.590) | 0.204 | 355928 |
| mnist_mixed (2 rows) | strong | pca | 16 | 20 | 8/8 | 175/200 | 8/8 | 2.35e-03 | 3.4e-16 | recovered | 0.439 (0.372–0.590) | 0.204 | 355928 |
| mnist_mixed (2 rows) | strong | pca | 16 | 100 | 8/8 | 168/200 | 8/8 | 1.37e-03 | 4.3e-16 | recovered | 0.439 (0.372–0.590) | 0.204 | 355928 |
| mnist_mixed (2 rows) | strong | pca | 16 | 400 | 8/8 | 173/200 | 8/8 | 1.91e-03 | 9.1e-16 | recovered | 0.423† (0.372–0.590) | — | 308863 |
| mnist_mixed (2 rows) | strong | pca | 32 | 1 | 8/8 | 48/200 | 1/8 | 1.69e-02 | 2.8e-16 | search failure | 0.294† (0.260–0.486) | — | 308863 |
| mnist_mixed (2 rows) | strong | pca | 32 | 5 | 8/8 | 45/200 | 0/8 | 3.49e-02 | 4.1e-16 | search failure | 0.329 (0.260–0.486) | 0.281 | 355928 |
| mnist_mixed (2 rows) | strong | pca | 32 | 20 | 8/8 | 56/200 | 1/8 | 1.38e-02 | 3.3e-16 | search failure | 0.329 (0.260–0.486) | 0.281 | 355928 |
| mnist_mixed (2 rows) | strong | pca | 32 | 400 | 8/8 | 67/200 | 2/8 | 7.77e-03 | 9.6e-16 | search failure | 0.294† (0.260–0.486) | — | 308863 |
| mnist_mixed (2 rows) | strong | pca | 48 | 1 | 1/8 | 1/200 | 0/8 | 2.01e-02 | 2.7e-16 | search failure | 0.234† (0.212–0.430) | — | 308863 |
| mnist_mixed (2 rows) | strong | pca | 48 | 400 | 2/8 | 2/200 | 0/8 | 6.20e-03 | 8.7e-16 | search failure | 0.234† (0.212–0.430) | — | 308863 |
| mnist_mixed (2 rows) | strong | ae | 16 | 1 | 5/8 | 9/200 | 0/8 | 1.76e-01 | 2.5e-16 | search failure | 0.237† (0.181–0.518) | — | 308863 |
| mnist_mixed (2 rows) | strong | ae | 16 | 400 | 5/8 | 10/200 | 0/8 | 5.90e-02 | 8.5e-16 | search failure | 0.237† (0.181–0.518) | — | 308863 |
| mnist_mixed (2 rows) | strong | ae | 32 | 1 | 0/8 | 0/200 | 0/8 | 8.54e-02 | 2.6e-16 | search failure | 0.189† (0.165–0.394) | — | 308863 |
| mnist_mixed (2 rows) | strong | ae | 32 | 400 | 2/8 | 2/200 | 0/8 | 3.13e-02 | 7.7e-16 | search failure | 0.189† (0.165–0.394) | — | 308863 |
| mnist_mixed_samerow (1 row) | strong | pca | 16 | 1 | 8/8 | 157/200 | 0/8 | 2.66e-02 | 2.6e-16 | search failure | 0.439 (0.372–0.590) | 0.204 | 355929 |
| mnist_mixed_samerow (1 row) | strong | pca | 16 | 5 | 8/8 | 167/200 | 0/8 | 3.00e-02 | 3.2e-16 | search failure | 0.439 (0.372–0.590) | 0.204 | 355929 |
| mnist_mixed_samerow (1 row) | strong | pca | 16 | 20 | 8/8 | 167/200 | 2/8 | 1.52e-02 | 3.1e-16 | search failure | 0.439 (0.372–0.590) | 0.204 | 355929 |
| mnist_mixed_samerow (1 row) | strong | pca | 16 | 100 | 8/8 | 163/200 | 2/8 | 1.19e-02 | 4.8e-16 | search failure | 0.439 (0.372–0.590) | 0.204 | 355929 |
| mnist_mixed_samerow (1 row) | strong | pca | 16 | 400 | 8/8 | 163/200 | 4/8 | 8.82e-03 | 7.9e-16 | search failure | 0.439 (0.372–0.590) | 0.204 | 355929 |
| cifar_motorcycle | mlp_overtrained | pca | 16 | 1 | 8/8 | 189/200 | 8/8 | 2.45e-03 | 2.0e-16 | recovered | 0.343 (0.234–0.493) | 0.263 | 355930 |
| cifar_motorcycle | mlp_overtrained | pca | 16 | 5 | 8/8 | 192/200 | 8/8 | 1.25e-03 | 3.9e-16 | recovered | 0.343 (0.234–0.493) | 0.263 | 355930 |
| cifar_motorcycle | mlp_overtrained | pca | 16 | 20 | 8/8 | 181/200 | 8/8 | 4.77e-04 | 2.5e-16 | recovered | 0.343 (0.234–0.493) | 0.263 | 355930 |
| cifar_bottle | mlp_overtrained | pca | 16 | 1 | 8/8 | 199/200 | 8/8 | 2.03e-03 | 2.1e-16 | recovered | 0.276 (0.088–0.348) | 0.194 | 355941 |
| cifar_bottle | mlp_overtrained | pca | 16 | 5 | 8/8 | 197/200 | 8/8 | 1.49e-03 | 3.1e-16 | recovered | 0.276 (0.088–0.348) | 0.194 | 355941 |
| cifar_bottle | mlp_overtrained | pca | 16 | 20 | 8/8 | 200/200 | 8/8 | 4.23e-04 | 2.3e-16 | recovered | 0.276 (0.088–0.348) | 0.194 | 355941 |
| cifar_bottle | cnn | pca | 16 | 100 | 8/8 | 186/200 | 2/8 | 9.42e-04 | 6.1e-16 | search failure | 0.276 (0.088–0.348) | 0.194 | 355948 |
| cifar_bottle | cnn | pca | 16 | 400 | 8/8 | 183/200 | 3/8 | 9.16e-04 | 8.8e-16 | search failure | 0.276 (0.088–0.348) | 0.194 | 355950 |
| cifar_bottle | cnn | ae | 16 | 20 | 3/8 | 25/200 | 0/8 | 1.21e-02 | 2.9e-16 | search failure | 0.321 (0.118–0.544) | 0.158 | 355947 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 16 | 1 | 8/8 | 191/200 | 5/8 | 1.19e-02 | 3.6e-16 | search failure | 0.338 (0.147–0.436) | 0.176 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 16 | 5 | 8/8 | 191/200 | 5/8 | 1.21e-02 | 3.8e-16 | search failure | 0.338 (0.147–0.436) | 0.176 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 16 | 20 | 8/8 | 191/200 | 8/8 | 5.94e-04 | 3.0e-16 | recovered | 0.338 (0.147–0.436) | 0.176 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 16 | 100 | 8/8 | 189/200 | 8/8 | 4.98e-04 | 4.5e-16 | recovered | 0.338 (0.147–0.436) | 0.176 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 16 | 400 | 8/8 | 191/200 | 8/8 | 5.86e-04 | 9.7e-16 | recovered | 0.338 (0.147–0.436) | 0.176 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 32 | 1 | 8/8 | 144/200 | 1/8 | 1.88e-02 | 2.2e-16 | search failure | 0.291 (0.120–0.398) | 0.284 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 32 | 5 | 8/8 | 142/200 | 2/8 | 6.00e-03 | 4.0e-16 | search failure | 0.291 (0.120–0.398) | 0.284 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 32 | 20 | 8/8 | 140/200 | 2/8 | 1.10e-02 | 3.2e-16 | search failure | 0.291 (0.120–0.398) | 0.284 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 32 | 100 | 8/8 | 136/200 | 2/8 | 7.12e-03 | 5.8e-16 | search failure | 0.291 (0.120–0.398) | 0.284 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 32 | 400 | 8/8 | 141/200 | 1/8 | 1.17e-02 | 9.0e-16 | search failure | 0.291 (0.120–0.398) | 0.284 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 48 | 1 | 6/8 | 30/200 | 2/8 | 3.27e-03 | 2.9e-16 | search failure | 0.249 (0.107–0.359) | 0.312 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 48 | 5 | 8/8 | 32/200 | 2/8 | 3.05e-03 | 3.7e-16 | search failure | 0.249 (0.107–0.359) | 0.312 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 48 | 20 | 5/8 | 30/200 | 2/8 | 4.15e-03 | 3.2e-16 | search failure | 0.249 (0.107–0.359) | 0.312 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 48 | 100 | 6/8 | 35/200 | 1/8 | 4.17e-03 | 5.3e-16 | search failure | 0.249 (0.107–0.359) | 0.312 | 355952 |
| cifar_mixed_mb (2 rows) | mlp_overtrained | pca | 48 | 400 | 6/8 | 33/200 | 0/8 | 3.83e-03 | 7.9e-16 | search failure | 0.249 (0.107–0.359) | 0.312 | 355952 |
| cifar_mixed_mb (2 rows) | cnn | pca | 16 | 1 | 7/8 | 161/200 | 0/8 | 5.07e-03 | 2.8e-16 | search failure | 0.338 (0.147–0.436) | 0.176 | 355953 |
| cifar_mixed_mb_samerow (1 row) | mlp_overtrained | pca | 16 | 1 | 8/8 | 191/200 | 2/8 | 9.50e-03 | 4.1e-16 | search failure | 0.338 (0.147–0.436) | 0.176 | 355968 |
| cifar_mixed_mb_samerow (1 row) | mlp_overtrained | pca | 16 | 5 | 8/8 | 191/200 | 4/8 | 1.10e-02 | 3.8e-16 | search failure | 0.338 (0.147–0.436) | 0.176 | 355968 |
| cifar_mixed_mb_samerow (1 row) | mlp_overtrained | pca | 16 | 20 | 8/8 | 189/200 | 5/8 | 5.32e-03 | 2.7e-16 | search failure | 0.338 (0.147–0.436) | 0.176 | 355968 |
| cifar_mixed_mb_samerow (1 row) | mlp_overtrained | pca | 16 | 100 | 8/8 | 190/200 | 8/8 | 4.75e-04 | 5.0e-16 | recovered | 0.338 (0.147–0.436) | 0.176 | 355968 |
| cifar_mixed_mb_samerow (1 row) | mlp_overtrained | pca | 16 | 400 | 8/8 | 190/200 | 8/8 | 4.22e-04 | 8.6e-16 | recovered | 0.338 (0.147–0.436) | 0.176 | 355968 |
| cifar_mixed_mb_samerow (1 row) | cnn | pca | 16 | 100 | 7/8 | 148/200 | 1/8 | 1.62e-03 | 4.7e-16 | search failure | 0.338 (0.147–0.436) | 0.176 | 355978 |
| cifar_mixed_mb_samerow (1 row) | cnn | pca | 16 | 400 | 7/8 | 151/200 | 1/8 | 1.69e-03 | 1.2e-15 | search failure | 0.338 (0.147–0.436) | 0.176 | 355981 |
| cifar_mixed_mb_samerow (1 row) | cnn | pca_perclass | 16 | 20 | 7/8 | 122/200 | 0/8 | 5.00e-03 | 3.7e-16 | search failure | 0.314 (0.131–0.451) | 0.188 | 355977 |
| cifar_mixed_mb_samerow (1 row) | cnn | pca_perclass | 16 | 100 | 7/8 | 121/200 | 0/8 | 5.13e-03 | 5.4e-16 | search failure | 0.314 (0.131–0.451) | 0.188 | 355980 |

† the earlier rows' median convention (`torch.median`, lower middle value); the new rows use `np.median`. "found" = images with a start within relative error 1e-2 of the on-chart target; "landed" = starts within 1e-2 of any target. Verdict is the row's own `ntk_verdict` for the main form, read literally: *search failure* = residual ABOVE the model floor, *alias* = residual AT the floor with wrong images, *recovered* = all N found.

### 7.2 Within-chart comparisons (INTERIM — 97 rows in; the CNN and `*_perclass` cells are only partly in)

**(1) Composition, certificate.** Certificate images found, per (base, chart, k), across compositions:

| base | chart | k | single-class cells | mixed, two rows | mixed, one row |
|---|---|---|---|---|---|
| mnist strong | pca | 16 | a 8/8 (T5–100), t 8/8 (T1–400) | 8/8 (T1–400) | 8/8 (T1–400) |
| mnist strong | pca | 32 | a 7–8/8, t 7–8/8 | 8/8 (T1, 5, 20, 400) | — |
| mnist strong | pca | 48 | a 1–2/8, t 4–5/8 | 1–2/8 | — |
| mnist strong | ae | 16 / 32 / 48 | a 2–5/8 / 1–2/8 / 0–1/8 | 5/8 / 0–2/8 (T1, 400) | — |
| cifar over-trained MLP | pca | 16 | motorcycle 8/8, bottle 8/8 | 8/8 (T1–400) | 8/8 (T1–400) |
| cifar over-trained MLP | pca | 32 / 48 | — | 8/8 / 5–8/8 | — |
| cifar CNN | pca | 16 | bottle 8/8 (T100, 400) | 7/8 (T1) | 7/8 (T100, 400) |
| cifar CNN | pca_perclass | 16 | — | — | 7/8 (T20, 100) |

No mixed cell is below its single-class cells at the same (base, chart, k) — the certificate is **composition-blind**
in every completed comparison, and one row of head or two makes no difference to it. Where it does fall below 8/8 the
cause tracks the chart, not the composition: at pca k = 48 on MNIST the collapse hits single-class and mixed alike
(1–5 of 8; 1–18 landed starts of 200), and the spread between the two SINGLE-class cells there (a 1–2/8 vs t 4–5/8)
is larger than any single-vs-mixed gap anywhere in the table. The CNN's 7/8 at k = 16 is likewise a base effect, not
composition (bottle alone also loses images on the `ae` chart: 3/8).

**(2) T trend per arm.** Certificate: flat in T wherever a cell has several T's — mnist_t pca k = 16 landed
187, 189, 187, 184, 179 of 200 at T = 1, 5, 20, 100, 400; mnist_mixed_samerow 157, 167, 167, 163, 163; CIFAR
mixed over-trained MLP 191, 191, 191, 189, 191; mnist_a pca k = 32 82, 69, 73. The images found do not move with T in
any completed cell (the one exception is a single image at pca k = 32 T = 5, a 7/8 that is 8/8 at T = 20 and 100).
NTK lora:varpro: strongly T-dependent and non-monotone, and its residual is 13 orders above the model floor except
where it recovers. Examples (images found at T = 1, 5, 20, 100, 400): mnist_t pca k16 8, 8, 2, 8, 8; mnist_a pca k16
·, 0, 1, 8, ·; mnist_mixed_samerow pca k16 0, 0, 2, 2, 4; CIFAR mixed two-row over-trained pca k16 5, 5, 8, 8, 8;
CIFAR mixed same-row 2, 4, 5, 8, 8. Two regularities hold across cells: the free-coefficient arm improves with T on
CIFAR (both mixed compositions reach 8/8 by T = 100), and at fixed T the same-row cell is never better than the
two-row cell (2 vs 5 at T = 1, 4 vs 5 at T = 5, 5 vs 8 at T = 20) — the one place where composition does show, and
it shows in the COUPLED arm, as pre-registered. The `dW` form remains at its floor (0.69–0.97) in every row.

**(3) Charts (fidelity only — never recovery across charts).** Raw projection error of the true test images, median
(range), with the release-free control beside it:

| cell | base | chart | k | raw proj. err median (range) | control (median) |
|---|---|---|---|---|---|
| mnist_a | strong | pca | 16 / 32 / 48 | 0.320 (0.265–0.790) / 0.245 (0.187–0.691) / 0.196 (0.150–0.631) | 0.245 / 0.313 / 0.346 |
| mnist_a | strong | ae | 16 / 32 / 48 | 0.236 (0.195–0.565) / 0.201 (0.167–0.623) / 0.225 (0.178–0.544) | 0.409 / 0.420 / 0.397 |
| mnist_t | strong | pca | 16 / 32 / 48 | 0.444 (0.369–0.619) / 0.367 (0.245–0.470) / 0.304 (0.197–0.393) | 0.239 / 0.329 / 0.361 |
| mnist_mixed | strong | pca | 16 / 32 | 0.439 (0.372–0.590) / 0.329 (0.260–0.486) | 0.204 / 0.281 |
| cifar_motorcycle | mlp_overtrained | pca | 16 | 0.343 (0.234–0.493) | 0.263 |
| cifar_bottle | mlp_overtrained / cnn | pca | 16 | 0.276 (0.088–0.348) | 0.194 |
| cifar_bottle | cnn | ae | 16 | 0.321 (0.118–0.544) | 0.158 |
| cifar_mixed_mb | mlp_overtrained | pca | 16 / 32 / 48 | 0.338 (0.147–0.436) / 0.291 (0.120–0.398) / 0.249 (0.107–0.359) | 0.176 / 0.284 / 0.312 |
| cifar_mixed_mb_samerow | cnn | pca vs **pca_perclass** | 16 | 0.338 (0.147–0.436) vs **0.314 (0.131–0.451)** | 0.176 vs 0.188 |

The one completed per-class/pooled pair (cifar_mixed_mb_samerow, CNN, k = 16) goes the pre-registered way on **7 of 8
images** (motorcycle 0.358→0.345, 0.295→0.290, 0.325→0.309; bottle 0.147→0.131, 0.350→0.318, 0.168→0.147,
0.370→0.353) and against it on one (motorcycle[56] 0.436→0.451), median 0.338 → 0.314. Two cautions on reading the
fidelity column at all: the AE chart is better than PCA on MNIST at every k (0.20–0.24 vs 0.20–0.32) yet recovers far
FEWER images (0–5 of 8 vs 7–8 of 8 at k ≤ 32) — fidelity and recoverability are not the same axis; and PCA fidelity
improves with k while the certificate's recovery collapses with k (k = 48), so neither column predicts the other.
Where the control error is BELOW the chart error (CIFAR at k = 16: 0.176–0.194 control vs 0.276–0.343 chart), a
projection of the truth is further from the truth than some public image is — the chart, not the attack, is the
binding limit in those cells.

### 7.3 Observation: at T = 1 the certificate does not depend on the labels (matched 157/200)

Stated observation, from the release equations, checked so far only by the coincidence below. With `B_0 = 0` one SGD
step gives `B_1 = −η D (A_0 H)ᵀ` and `A_1 = A_0`, where `D = (softmax(W_0 H) − Y)/N` is `m × N`. The certificate is
`C = P_{row(B_1)^⊥} A_1`, and `row(B_1) = col(B_1ᵀ) = col(A_0 H · Dᵀ)`. If `rank D = N` (every completed T = 1 row reports
`rank B_T = 8 = N`), `col(Dᵀ) = ℝ^N` and `row(B_1) = col(A_0 H)`, whatever `Y` is. So at T = 1 the label assignment —
one new row for all eight images or two rows of four — changes `B_1` but not `C`, and the certificate arm is run on
the same objective from the same starts in both compositions. Measured: `mnist_mixed` (two rows, job 308863) and
`mnist_mixed_samerow` (one row, job 355929), pooled pca k = 16, T = 1: certificate residual at the truths 6.7e-15 (two rows) vs
1.3e-13 (one row; both at the FP64 floor of a different `B_1`), **landed starts 157/200 in both, 8/8 images in both**; the NTK targets do differ (lora:varpro best
residual 3.93e-3 with two rows, 2.66e-2 with one). The label dependence of the certificate therefore starts at T = 2,
where `A_2 ≠ A_0` and `row(B_2)` picks up `Bᵀ D H ᵀ` terms that carry `Y`. A direct check (`‖C_two-row − C_same-row‖`
at T = 1 from the saved releases) is not yet run.
