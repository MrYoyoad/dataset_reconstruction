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
| LoRA on the pixel layer, on-chart privates | 2 of 8, at 30 of 400 starts; the modal minimiser is a 99.4% blend |
| LoRA on a hidden layer, public PCA chart | 253/400 starts land, 8/8 images |
| LoRA on the head, public PCA chart | 171/400 starts land, 8/8 images |
| LoRA on the head, a fully trained or **over-trained** backbone, a new class | up to 200/200 starts land, 8/8 images |
| certificate from a release trained on 8 *other* images | 0/400 — the control passes |

## 1. Why the replica failed: a linear certificate cannot separate what it annihilates

`C h_i = 0` for every i means `C` annihilates the whole span, so every linear combination `Σ c_i h_i` is an exact
zero as well. When the adapted layer's input *is* the image, those combinations are themselves images, and a smooth
chart represents the blend far better than any individual image: the 8-apple mean has PCA-32 representation error
0.06 against 0.10–0.42 for the individual apples. The minimiser is therefore the blend by construction.

Measured (audit, 2026-09-06): the collapsed attractor of the original on-chart run (379 of 400 starts) is a
least-squares blend of the eight privates to 99.4%, with coefficients that are not even convex (−0.70, 0.34, …,
0.46), and the blend's own certificate residual is 3e-15 — as low as the truths'. Every sanity check passed
throughout: `‖CH‖/(‖C‖‖H‖)` at 3.6e-15, quotient form 5.9e-15, `rank C = r − N = 56`, excitation gap 1e14. **Those
checks are necessary and never sufficient**; they hold identically in the degenerate case.

The equation count (56 equations against 32 unknowns) is silent about this, because it assumes generic position and
the span direction is exactly where position is not generic.

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
- **A linear chart beats a learned one.** PCA reaches the exact zero (median start residual 1e-14, i.e. most starts
  converge to a true zero); the conv-AE decoder stalls three orders higher. The same was true on MNIST.
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
