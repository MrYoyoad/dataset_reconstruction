# Figure provenance for the 2026-09-15 figure pack

Three provenance documents from the archive, concatenated unedited, plus the corrections found when the
panels were traced back to repository rows on 2026-09-17. Figures live in `figures/gal_2026-09/`,
editable sources in `scripts/figpack_2026_09_15/`.

## Corrections found while tracing (2026-09-17, this repo)

1. **The Fashion-MNIST "4 of 8" is a superseded number.** It is the 150-start cell, job 473802
   (`results/cifar_newclass/sharp2_473802.jsonl`, `landed 9/150`, `images_found 4`,
   `landings_per_image [0,1,2,0,0,2,4,0]` — the landed columns are exactly the four the panel shows).
   The repo's current figure for the same cell, `figures/cifar_newclass/cnn_fashion_bag_k64_onchart.png`,
   is a 400-start re-run: **23/400 landed, 6 of 8 images** (job 556643). Quote 6/8 with the start count.
2. **The MNIST "private-built reference" strip pairs each digit with a byte-identical copy of itself**
   (`three_charts_row_1.png` == `row_4.png` == `three_charts_oracle_reference.png`, same sha256). That is
   consistent with the 5.7e-14 error of an oracle chart built from the private digits, but the panel gives a
   reader no cue that the chart is **not attacker-available**. The repo's later figure
   (`figures/exact_inversion/chart_dependence_k17.png`) says so on its face; use that one.
3. **"Glyph correlations ≈ 0.95–0.97"** appears in the provenance prose with no script, row or log behind it
   anywhere in the archive or the repo. **UNTRACED** — do not repeat it.

---

## Part 1 — SOURCES.md (LoRA_visual_figures, verbatim)

# Figure sources and scope

These notes stay outside the figure-only PDF. The PDF contains diagrams, image comparisons, short labels and equations for oral presentation. No new reconstruction experiment was run for this figure set.

## Files

1. `01_one_layer`: a public image family precedes the frozen feature map and one LoRA-adapted linear layer. The pixel grid is a schematic, not experimental evidence. Green denotes searched image-family coordinates; blue denotes the frozen model or compatible subspace; orange denotes the adapter or certificate residual.
2. `02_certificate_subspaces`: the fixed-feature conservation law and the released-factor certificate. The drawing is a low-dimensional schematic.
3. `03_reconstruction_results`: supplied letter, CIFAR-100 motorcycle and Fashion-MNIST bag results, with actual original-image references.
4. `04_ntk_partial_information`: an exact one-step identity and the ambiguity introduced by treating coefficients as free variables. The vectors are schematic, not measured reconstructions.
5. `05_image_family`: an exactly computable polynomial toy and its normalized certificate score. This is an analytic illustration, not an experimental performance plot.
6. `06_multilayer`: the user's preferred architecture figure, with its conditional multilayer-certificate scope retained.
7. `07_mnist_coverage`: a separate supplied replay diagnostic, not a certificate-only result or an established comparison across one fixed release.
8. `08_local_families`: a proposal for searching small patches of a richer public decoder. It is not a new reconstruction result.

The PDF order and PNG filenames match. Each figure also has a vector PDF. The `source` directory contains editable TeX, unchanged source figure assets and a build script. Run `python3 build.py` inside `source` with PyMuPDF and a TeX installation providing pdfLaTeX, TikZ, PGFPlots and Latin Modern. No input dataset or private training examples are synthesized by that script.

## Reconstruction provenance

The original/PCA distinction is intentional. These on-family experiments trained on public-PCA projections, so the reported target-recovery counts refer to those projected inputs, not exact recovery of the unprojected photographs.

- **Letters:** the original glyphs come from the explicitly labelled `private (raw)` panel on page 3 of `LoRA_meeting_board_guide(1).pdf`, also present on page 6 of `meeting_told_slowly.pdf`. Four A columns are matched to the PCA/NTK/certificate panel on page 4 of `figures_for_gal(1).pdf`. The fit panel reports an eight-example A/T experiment; only the four original-matched A columns are displayed here. Matching was checked at figure level, not through recovered dataset indices. It does not establish that the earlier raw-reference panel and later comparison used the same release. Source display polarity is preserved. The historical NTK row is retained as an observed output; it is not a corrected, controlled four-arm NTK benchmark, and its failure does not prove failure of all NTK inversion methods.
- **Motorcycles:** eight original/output pairs from the supplied CIFAR-100 motorcycle figure. The reported eight target matches are to the PCA-projected training inputs. These are not MNIST images.
- **Bags:** the actual Fashion-MNIST originals and corresponding outputs come from the `private (raw)` and reconstruction rows on page 3 of `figures_for_gal(1).pdf`. Four source columns marked as landed (2, 3, 6, 7) and two approximate outputs (4, 5) are shown. The reported total remains four of eight target matches. A blank source candidate was not replaced with an invented output. Pairing outputs with targets uses ground truth for evaluation, not a demonstrated attacker-only selection rule.
- **MNIST:** six of eight digits from the supplied public-PCA, warped-family and private-built-reference replay panels. Each row refers to its own represented targets. The record does not establish a common release across rows; the private-built family is not an attacker-available public prior.

Image assets were reused without generating, enhancing or substituting experimental content. Originals are never relabelled PCA projections. A copy of the detailed existing provenance note is included as `ground_truth_provenance.md`.

## Mathematical scope

### Preserved subspace

Let `H` contain the fixed private feature vectors, with rank `q`, and let the adapter have `A` of shape `r x n` and `B` of shape `m x r`. Assume simultaneous plain scalar-step SGD on `W0 + BA`, zero-initialized `B`, no weight decay or other added update, a generic initial `A0`, and `q < r <= n`. Let `S = col(A0 H)` and let `P0` project orthogonally onto its complement. Then `B_t P0 = 0` and `P0 A_t = P0 A0` at every finite step. Under the additional excitation condition `rank(B_T) = q`, the release identifies `S = row(B_T)`, and `C = P0 A_T` annihilates every private feature with rank `r-q`. Full row rank of `A_T` is not required. Observing `rank(B_T)` alone does not establish that excitation equals the unobserved private-feature rank. AdamW generally breaks this exact invariant.

In the left drawing, the component of `A_t h` perpendicular to `S` is unchanged. In the right drawing, the projection removes the component in `S`; private features have no remaining component. This is a projection in adapter-coordinate space, not an assertion that an input-space complement of `S` exists.

### One-step NTK

`D0` contains the derivatives of the loss with respect to the private examples' initial logits; any batch averaging can be absorbed in `D0` or the step size. At one simultaneous update with `B0=0`, `A1=A0` and `B1=-eta D0 (A0 H)^T` exactly. Thus linearization error is not the explanation at one step, and `A0` is actually available as `A1` in this setting.

Writing `B1 = Lambda V^T` with free `Lambda` makes the equation invariant to `V -> V M` and `Lambda -> Lambda M^{-T}`. This algebraic ambiguity becomes an image ambiguity only if the alternative features are realizable by admissible images. The true coefficients depend on the candidate data and labels and are not arbitrary. Label, coefficient, nonlinear-head or dynamical constraints may therefore add information. The rank bound shown is for the scalar initial tangent kernel of the adapted linear head, per output, not every neural tangent kernel. It does not imply that all finite-rank-kernel inverse problems are unsolvable.

The supplied NTK output in the result board is not offered as empirical proof of this specific mechanism. Separately, the correct one-step merged update includes the sketch `A0^T A0`; dropping that factor is a model misspecification, not an intrinsic NTK failure.

### Polynomial family and score

The toy uses `psi(t)=(1,t,t^2)` on `[-2,2]`, private parameters `-1,+1`, `A0=I`, and `C=u u^T` with `u=(-1,0,1)/sqrt(2)`. It is an exact one-step realization when the recorded span is the span of those two private features. The left drawing is the affine slice with first coordinate equal to one. Compatibility is the line with third coordinate equal to one; intersecting it with the parabola leaves only the two private points.

The score is `||C psi(t)|| / ||psi(t)|| = |t^2-1| / (sqrt(2)*sqrt(1+t^2+t^4))`. The green intervals use epsilon 1/4. The orange line is the actual minimum score outside those intervals; the grey threshold lies below it. The displayed implication follows directly from that definition. Here the toy has `k=p=1`; it illustrates exact identification in a particular polynomial family, not the generic strict `k<p` theorem.

For a fixed public family, the equation budget is `p=r-q`. The supplied chart theorem addresses generic exclusion of extra solutions outside the private feature span when `k<p`, with the stated regularity and independence conditions. To identify only the private images also requires ruling out other family points already inside that span. Local stability further requires tangent separation and a nondegenerate local derivative. The numerical inequality alone is not a global identification or optimization guarantee.

### Several layers

The base-model plug-in shown is conditional: all private trajectory features and base-model private features at layer `ell` must lie in one subspace `U_ell`, and the released `B` must excite its seeded image `A_{ell,0} U_ell`. Plain-SGD/zero-`B` assumptions remain in force. The rank of that seeded image need not be silently identified with the dimension of `U_ell`. The single-layer reconstruction panels are not presented as evidence that this condition holds automatically in a deep model.

### Richer local families

In `g_j(z)=G(w_j+U_j z)`, `G` is a public decoder, `w_j` is a patch centre, and `U_j` has `k` search directions in an ambient latent space of dimension `D`. The `k<r-q` condition is a proposed local equation budget, not a sufficient guarantee by itself. Fixed-public-family hypotheses, coverage, global span separation, conditioning and search still matter. Selecting from fixed public patches differs from learning unrestricted new patch directions from the private release; the latter needs a separate analysis. A small number of controls in each optimization step does not by itself bound the full set of reachable images.

## Proof and literature record

The certificate/deep-extension statements follow the supplied `framework_rev12.tex` Section 4 and Rev 11.2 plan Section 3.5.1. The family conditions follow the supplied `charts.pdf` / `chart_inversion_theory.tex`; multilayer qualifications are also recorded in `multilayer_lora_theory.pdf`. These notes distinguish conditional mathematical statements, existing experiments and proposals rather than making a new exhaustive novelty claim.

- Haim et al., [Reconstructing Training Data from Trained Neural Networks](https://arxiv.org/abs/2206.07758): the earlier reconstruction line referenced in the architecture diagram.
- Oz et al., [Reconstructing Training Data From Real World Models Trained with Transfer Learning](https://arxiv.org/abs/2407.15845): embedding-space reconstruction followed by image decoding; not a pixel-only baseline.
- Loo et al., [Understanding Reconstruction Attacks with the Neural Tangent Kernel and Dataset Distillation](https://arxiv.org/abs/2302.01428): the kernel-based comparison behind the NTK discussion; its recovery assumptions must not be replaced by a blanket claim about any finite model.
- [GIAS](https://arxiv.org/abs/2110.14962) and [GIFD](https://arxiv.org/abs/2308.04699): adjacent generator-based inversion work. Using a generator inside an inversion objective is not claimed as a new idea by itself.

Replay remains a comparator or a possible downstream beneficiary of better candidate selection, not the proposed central contribution or the direction of the local-family diagram.

---

## Part 2 — ground_truth_provenance.md (verbatim; identical in three archive locations)

# Original-image correction

The meeting PDFs distinguish original ground truth from the PCA projections actually used as training inputs. Reconstruction counts in the on-chart experiments still refer to recovery of those projected inputs.

## A images

- Raw originals: `LoRA_meeting_board_guide(1).pdf`, page 3, embedded 672-by-243-pixel figure. Its first row is explicitly labelled `private (raw)`.
- The same embedded figure also occurs in `meeting_told_slowly.pdf`, page 6.
- That source figure contains eight A/a images. The first four correspond to the A columns in the separate mixed A/T comparison in `figures_for_gal(1).pdf`, page 4.
- Correspondence was checked visually and by comparing the displayed PCA glyphs after accounting for opposite display polarity, image scaling, and crop. Each of the first four source PCA glyphs has its strongest match in the same-numbered mixed-panel column; normalized glyph correlations are approximately 0.95–0.97. This is a figure-level correspondence check, not recovery of dataset indices or a claim that the two panels came from the same release.
- The PDFs display only these four matched A columns, as original / PCA training target / NTK output / certificate output. The PCA, NTK and certificate rows come from the mixed A/T panel, not from the earlier A-only run. The reported experiment has eight examples; four are shown. The original top PCA row was a valid training-target reference, not raw ground truth.
- Raw source display polarity is retained. No glyph is regenerated, enhanced, or replaced with a similar public example.

Assets: `letters_raw_source_panel.png` preserves the whole earlier embedded source; `letters_a_raw_gt.pdf` is the original-image row; `letters_a_pca.pdf`, `letters_a_ntk.pdf`, and `letters_a_certificate.pdf` are corresponding fitted-panel crops. `letters_original.png` retains the full eight-column A/T comparison.

`audit/make_letter_reference_panel.py` produces `letters_four_rows.pdf` from the unchanged row assets. Every column refers to the same matched example. Comparing original to PCA shows representation loss; comparing the output rows to PCA shows inversion error. The opposite original/output polarity is retained from the sources.

## Fashion-MNIST: target matches and approximate reconstructions

`figures_for_gal(1).pdf`, page 3, contains the true Fashion-MNIST bag row, explicitly labelled `private (raw)`. The new `fashion_original.pdf` preserves that entire source page. No projection is used as an original.

- Reported target matches: source columns **2, 3, 6 and 7**, the four columns marked `landed` in the source. All four appear in both documents.
- Approximate reconstructions: source columns **4 and 5**, with their genuine closest-search outputs. These also retain recognizable bag structure. They appear alongside the four matches in the brief and as Figure 5 in the companion.
- The main brief overview therefore shows six of the eight examples. Its label `Target match` means the projected training input met the source's recovery criterion. `Approximate` credits partial reconstruction without changing the reported total of 4/8 target matches.
- The fourth unrecovered source example (column 8) has no displayed candidate; the blank source cell is never presented as a synthesized reconstruction.
- Matching outputs to targets is an evaluation procedure using ground truth, not a demonstrated attacker-only selection rule.

`audit/make_fashion_comparisons.py` recomposes the pairs directly from the supplied PDF page. Its crop coordinates refer to the embedded 1754×1240 raster: 133-pixel square image tiles; raw row starts at y=410.5, output row at y=791.5. Image content, aspect ratio and display polarity are preserved. The script creates the four-match row/grid, six-example overview and two-approximation panel.

The older `fashion_mnist_gt.pdf` and `bags_row_*.pdf` contain source columns 2–5 and remain archived. The current PDFs use the newer, explicitly selected panels above.

## Other figures

- **Motorcycles:** all eight raw originals from the supplied figure remain immediately beside their outputs. Training used public-PCA projections; all eight projected targets are reported recovered.
- **MNIST:** each original digit is paired with each chart's corresponding replay output. Six of eight digits occur in the source panel. Chart-reference rows are separately labelled in the companion and never called ground truth. The record does not establish a common release across the charts.
- **Apples and keyboards:** original assets and paired comparisons are preserved in the archive but removed from the meeting PDFs. The apples' same-release chart comparison remains a source claim; its removal is a presentation decision.

No generated, enhanced or online look-alike image was substituted for experimental ground truth. Figure-level source matching is the strongest provenance available where original dataset indices are absent.

---

## Part 3 — figure_roles.md (meeting_handoff/audit, verbatim)

# Figure roles for the meeting documents

This private record explains why each figure is present. Captions in the brief state the observation and the scope without repeating the editorial rationale.

| Figure | Question it answers | Role and permitted claim | Location |
|---|---|---|---|
| Small-input architecture | Where does the low-dimensional input enter? | k controls pass through a public decoder into d pixel values, with k much smaller than d. The full-network drawing shows a frozen base with optional adapted branches, then a label. It marks this work's small-input search and the Haim/Oz search locations. The caption distinguishes head-only image experiments from the multilayer extension. Reconstruction changes the decoder inputs and uses adapter-derived equations; the label is the original task, not the attack objective. | Page 1 of both |
| Letters, four rows | What was recovered, and why retain the NTK comparison? | Separate actual original, PCA training target, NTK output and certificate output. Show four verified A originals from an eight-letter run reporting all eight projected targets recovered. Both methods use PCA. The original comparison was useful; the earlier revision's use of PCA as raw truth was the error. | Brief 1; companion 2 |
| CIFAR-100 motorcycles | Does the mechanism work beyond letters? | All eight actual originals beside outputs; all eight public-PCA training targets reported recovered. The next question is how to retain more original detail. | Brief 2; companion 2 |
| Fashion-MNIST bags | What useful recovery exists even in an incomplete run? | Four reported target matches plus two recognizable approximations, each paired with its actual original. All eight targets lie in the family. The incomplete result is not explained solely by coverage. | Brief 2; companion 2 |
| MNIST, three families | How can the family limit returned detail? | Six actual originals paired with reconstructions under public PCA, a warped family, and a private-built reference. This is the supplied replay diagnostic. Each family reconstructs its own represented inputs; one common release is not established. The reference uses private data. | Page 3 of both |

## Ground truth and correspondence

- The first letters row contains the actual source originals. PCA is separately labelled as the training target. No missing T originals are invented; source polarity is retained.
- Other figures put actual originals on the left of each pair. Target-match counts concern projected training inputs, while these originals show overall fidelity.
- Fashion target matches are source columns 2, 3, 6 and 7. Approximate pairs are columns 4 and 5. Both PDFs now show them together; the old separate approximate panel is archived.
- No source image payload was edited in this rewrite. The new architecture is a vector diagram, not an experimental result.

## What is omitted, and why

Apples originally test family mismatch while holding a raw-photo release fixed. That result is retained in the archive, but its current visual does not help this meeting's simpler narrative. MNIST provides an easier coverage illustration; it does not inherit the apples' same-release control.

Keyboards add another example without a distinct question after letters, motorcycles and Fashion. Their source figure and reported count remain archived.

Drift and capacity curves remain out of both PDFs at the user's request. The companion explains their conditional implications in words. Redundant family-reference rows and the second Fashion display are also archived rather than repeated.

## Shared order

Architecture -> letters -> motorcycles -> Fashion-MNIST -> MNIST. The deep certificate is a reference note, not another experimental plot. Every figure either explains the idea, demonstrates recovery, or motivates the next research question.

## Preferred supplied design

The architecture is extracted as vectors from lora_brief_v2.pdf, page 1. The stronger visual explanation is retained. Two labels now say that exact equations need the certificate conditions and that branches are on adapted layers. The supplied prose and theorem claims were not imported. Prior diagram sources are archived separately.
