# Results bundle 2026-09-06 — index and the deck rules that exist nowhere else

Imported verbatim from `results_2026-09-06.zip` inside `Thesis_files_2026-09-17.zip` on 2026-09-17.
114 of that bundle's 155 files are already byte-identical to files in this repo (rows under
`results/cifar_newclass/`, `results/record_strength/`, figures under `figures/cifar_charts/`,
`figures/cifar_study/`, `figures/ntk_vs_cert/`, `figures/record_strength/`, and
`notes/ntk_vs_certificate_comparison.md`). The two files below were **not** in the repo in any form:
the experiment index, and the slide rules including four claims withdrawn in the same night.
Both are reproduced unedited; where they disagree with a later measurement the later one wins and the
difference is noted in `notes/artifact_index_2026-09-17.md`.

---

## Part 1 — INDEX.md (verbatim)

# Results bundle — 2026-09-06

Two studies, both run on the WEXAC GPU cluster. Every number here comes from a job whose id is given; nothing is
carried over from an earlier write-up without being re-measured.

## 1. Record strength vs recovery  (`record_strength/RESULT.md`)

Does an example's recovery from the certificate search depend on how strongly the adapter recorded it, or is
recording a threshold? Answer: a threshold in exact arithmetic, over eleven orders of magnitude of record strength;
a graded effect appears only when the release itself is trained in reduced precision.

Reproduction gate: every landing count reproduces the original jobs (760909, 764976, 706721) exactly.

Figures: `figures/record_strength/`
  recovery_error_vs_sigma.png    the threshold — two bands, nothing between
  basin_vs_sigma.png             how many random starts find each example
  sigma_vs_u.png                 record strength against the accumulated softmax error
  letters_precision_overlay.png  the same eight letters from releases trained in fp64 / fp32 / bf16 / fp16
Tables: `tables/record_strength_table.md` (per example), `tables/sigma_decomposition.json`
Jobs: 255095 (letters), 255098 (confident digits), 255107 (plots), 279342 (decomposition + replot)

## 2. The CIFAR certificate study  (`cifar/RESULT.md`)

The supplied CIFAR replica does not recover its private images. The cause is structural: with the adapter on the
pixel layer the certificate is linear in the image, so every blend of the private images is an exact zero and the
search returns a blend. Moving the adapter behind a nonlinearity fixes it, and then random starts recover every
private image, with the attacker able to tell which starts succeeded from the residual alone.

Figures:
  cifar_replica/            the replica as supplied (jobs 252897, 252898) and its on-chart control (257893, 257895)
  cifar_charts/             one panel per cell of the layer x chart x solver x privates grid, plus table.md
  cifar_newclass/           weird added-on classes on fully trained backbones (MLP, CNN, over-trained)
  mnist_reference/          the MNIST cells this replicates: letters (760909) and the k=6 digits (706721)

## 3. THE COMPARISON TO THE NTK-REGIME ROUTE  (`EQUIVALENCE_linearised_vs_certificate.md`)

**If you are looking for the comparison to the NTK regime, this is it.** It is filed under "equivalence" rather than
"comparison" because the result changed: the two routes turn out to have the same zero set, so a comparison of which
recovers more would have been measuring solvers rather than information. The rename was our decision and the
document opens by withdrawing the earlier comparison framing.

## 3b. What that document says

`EQUIVALENCE_linearised_vs_certificate.md`. This is NOT a comparison of which route recovers more, and an earlier
draft that was has been withdrawn. Two results: the LoRA-aware linearised model is never mis-specified, at any step
count, as a corollary of the closure lemma; and the two routes have the SAME ZERO SET, the certificate being the
per-candidate form of the condition and the linearised representer its joint form plus an independence clause. So
the equations cannot be the difference between them, and every measured gap is solver and search arity. The document
also records the two solver corrections found in review, one of which was a handicap that favoured the certificate.

## 4. THE TWO-TYPES-OF-IMAGES FIGURES

Three different pairings could be meant by this, so all three are named here and each is one search away.

**(i) Two kinds of image in ONE private batch** — the mixed-class cells, which is what was asked for in the session:
CIFAR keyboards *and* apples fine-tuned together with a new output row each, and MNIST letters *a* and *t* together.
Files: `figures/ntk_vs_cert/cifar_keyboard+apple_*.png` and `figures/ntk_vs_cert/mnist_letter_a+letter_t_*.png`.
What the pair shows: the certificate recovers 8 of 8 from a heterogeneous private batch, so mixing two unrelated
added classes does not degrade it.

**(ii) The raw private image against its chart projection** — every panel in `figures/cifar_newclass/` and
`figures/cifar_charts/` has three rows: the raw private image, its chart projection, and what the attack returned.
What the pair shows: the attack returns the chart's projection of the private image, not the image itself, which is
the fidelity caveat. It is also why identification rests on image error rather than on a similarity score.

**(iii) Held-out classes against a different corpus** — `figures/cifar_newclass/mlp_keyboard_*`,
`mlp_skyscraper_*`, `mlp_mushroom_*` are held-out CIFAR-100 classes; `mlp_flowers102_*` are photographs from a
different dataset entirely. What the pair shows: the recovery does not depend on the private images coming from the
same corpus as anything the model saw.

## 5. THE STUDY FIGURES  (`figures/cifar_study/`)

  F1_blend_degeneracy.png     the certificate residual along the affine hull of the private images: FLAT at machine
                              precision for an affine composition, which is the lemma with nothing left to argue;
                              a bowl touching the floor only at the truths once a nonlinearity is in the way
  F2_coverage_vs_precision.png  the pixel-layer cell fails at coverage, not precision
  F3_equivalence.png          the certificate residual against the linearised fit's floor: they vanish together
  F4_verdict_axis.png         alias against search failure on one axis, with the project's existing threshold
  F5_solver_handicap.png      a handicap in our OWN comparison, found and removed
  F6_isolation_test.png       the isolation test, read one-sided: full rank certifies, deficiency certifies nothing

## 6. Scripts

`scripts/` holds every script used, including the job submitters. `cifar_certificate.py` is the file as supplied,
unmodified; everything else is new and lives under `experiments/`.

---

## Part 2 — READ_ME_FIRST_deck_v6_update.md (verbatim)

# READ ME FIRST — instruction for the next session

**Go back to the v6 deck and update it against this bundle.** Do not start a new deck. v6 is the baseline; its
slide figures are in `figures/v6/` and the generators in `scripts/figures/`. Rebuild from the spec, not by
hand-editing a .pptx — see the round-trip note in CLAUDE.md.

Every scientific ruling behind what follows is in `notes/math_rulings_2026-09-06.md`, R1–R20, committed. Read it
before writing a claim; several statements below replace earlier ones that were withdrawn during the night.

## What changed since v6, in the order it should reach a slide

1. **A new route that v6 predates entirely.** The certificate is computed from the released adapter factors alone —
   no recipe, no labels, no seed, no batch size — and the private inputs lie in its kernel.
2. **It works on colour images.** With the adapter behind a nonlinearity, random starts recover every private image
   on CIFAR, and the wrong-release control recovers nothing. Numbers in `cifar/RESULT.md`.
3. **A batch holding two different kinds of image recovers completely**, on both CIFAR and MNIST, by both routes.
4. **We found a lemma — check it out, it is the best new theory in the bundle.** Every blend of the private images
   is an *exact* zero of the certificate, because the private span sits inside its kernel by construction. So when
   the map from chart coordinates to the adapted layer's input is affine, the chart contains those blends and the
   private images are not isolated solutions at any chart dimension. The counting condition therefore needs a second
   hypothesis beside it: **the chart must meet the private span only at the private points.** Proved, then
   demonstrated — a pixel-layer cell with a linear chart recovers 0 of 8 while every sanity check passes to machine
   precision, the found points are blends at fraction 1.000, and the residual ranking that works everywhere else
   fails there too. `F1_blend_degeneracy.png` is the one-panel version.
5. **We proved the linearised route and the certificate are the same object** where every image is recorded: same
   zero set, the certificate being the per-candidate form and the representer the joint form. They are not
   competitors. `EQUIVALENCE_linearised_vs_certificate.md`.
6. **But the degeneracy is route-specific, and this is the sharpest result of the night.** On one release and one
   chart, the recipe-free certificate lands **0 of 60** while replaying the training recovers **all eight images
   from 19 of 60 starts**, with zero aliases. So identifiability is a property of the release, the chart **and the
   route**. Any statement of the form "cannot identify by either route" is false — a cell falsified it within an
   hour of it nearly being published.
7. **Recovering and being able to tell you recovered are different**, and the attacker has three self-checks that
   need no ground truth: the residual ranking, the captured-directions count, and a stop signal that certifies its
   own precondition from the attacker's own output.
8. **Breadth has a frontier.** The last private image is not bought with more starts; it arrives only where
   precision falls to about two thirds. Coverage is a coupon-collector cost set by the rarest image.

## What must NOT go on a slide

- Any claim that one route beats the other. It is an equivalence; every measured gap is solver and search arity.
- The counting rule as a safety property. It is **one-sided** — sound when it says the channel is closed, silent
  when it says open — and the remedy for the closed case belongs to the **attacker**, who picks the chart. Geometry,
  not a defence.
- A fitted curve or a law from a handful of points. Measured, not modelled.
- Any similarity score without its control in the same row.
- A breadth figure at a fixed window size. It measures window against skew, not reach.
- Two coverage budgets sharing a name. The hardest single image and all of them are different numbers.

## Claims WITHDRAWN during the night — do not reinstate

- **"A linear chart beats a learned one."** The comparator was a different adapter position, then a different
  setting, and had one seed against the learned chart's three. Withdrawn outright at source. Matched controls were
  running when this bundle was built.
- **"More training makes the chart worse."** Too strong, and narrowed rather than withdrawn. What holds, paired by
  seed and three times for three, is that **the largest budget is worse than both smaller ones**. Between the two
  smaller budgets it is two seeds up and one down, i.e. no separation. A threshold at the top and a monotone drift
  are different phenomena and only the first is measured.
- **"Further from the model's prior therefore recovers better."** Refuted: the off-corpus class records *weaker* on
  both recording measures and recovers better anyway. Third independent demonstration that recording measures do
  not order recovery.
- **A defender-facing form of the counting rule.** Withdrawn; see above.

## Still open — do not present as settled

- One cell of four fails, and it is now fully diagnosed as a genuine local minimum with a **rotated** span: no
  private image is absent and none is captured, five of eight directions explained. Five candidate mechanisms were
  eliminated by measurement.
- Whether the largest-budget effect and the blend degeneracy are the same. Hypothesis only. The test must be a
  RATIO — blend reachability against individual-image reachability, both in feature space — because a uniformly
  better chart reaches everything better and that predicts the attack improving, not degrading. Decoder
  conditioning is already excluded: it degrades most between the two budgets where the attack IMPROVES.
- Distance-from-prior is unmeasured in either direction until classes are matched on chart fit.

## Where things are

`INDEX.md` explains every experiment, names each figure and gives the job behind each number. Study write-ups are
`record_strength/RESULT.md` and `cifar/RESULT.md`. Scripts under `scripts/`.

## The convolutional backbone — read before putting any conv cell on a slide

Six cells, job 293438. Both controls pass.

| added class | landed / starts | images found |
|---|---|---|
| keyboard | 52 / 200 | 6 of 8 |
| skyscraper | 35 / 200 | 6 of 8 |
| mushroom | 23 / 200 | 5 of 8 |
| **Flowers-102 (a different corpus)** | **66 / 200** | **8 of 8** |
| *wrong-release control* | *0 / 200* | *0 of 8* |
| *raw privates* | *0 / 200* | *0 of 8* |

The attack **degrades but does not fail**: landings fall four to sevenfold against the plain network and coverage
from eight images to five or six. Flowers is the only conv cell recovering everything; prefer it if one cell must
carry the arm.

**It is NOT the architecture, and say this first.** The recorded rank and the certificate's margin are identical in
every cell of all three arms. Nor could they differ: the adapter sits on the **head**, whose input is a single
pooled vector per image whatever the backbone does, so the counting rule that penalises many positions per image
does not apply. **The experiment that would test the architecture is an adapter inside the convolutional stack, and
it has not been run.** That converts an apparent weakness into a named next experiment.

**What moves is how strongly the model recorded the new class** — the residual at the true images rises two to
three orders — and the over-trained plain network shows the same direction with architecture held fixed.

**But recording strength does not account for the size of the drop.** Two columns were checked and neither orders
the landings: the over-trained cells have worse residuals yet land four times more often, and the spectrum gap
fails within an arm as well as across it. So the feature map's effect on the search contributes too, and **the
split is not measured**. Do not present either as the explanation.

**Precision survives.** The lowest-residual starts are still true landings, so the conv backbone costs coverage,
not trustworthiness — the same shape as the pixel-layer cell.

### Which conv figure to put on a slide, if any

**Use `figures/cifar_newclass/cnn_flowers102_k32_onchart.png`, and only that one.** It is the single convolutional
cell that recovers all eight images, so its grid reads as eight recoveries with no gaps. The recovered images in it
are exact, not approximate.

**Do not use the keyboard, skyscraper or mushroom conv figures as visual evidence.** They find six, six and five of
eight, so a grid of eight tiles shows two or three failures. That reads to an audience as an attack that half
works, which is the wrong impression: every image it does recover, it recovers exactly. If you want to show the
degradation honestly, show it as the numbers in the table above and use the flowers grid for the picture.

**And pair it with a plain-network cell rather than showing it alone**, so the drop is visible as a comparison
rather than presented as the attack's normal performance.
