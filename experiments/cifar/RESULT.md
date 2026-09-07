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
