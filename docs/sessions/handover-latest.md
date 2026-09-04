# Handover — 2026-09-04 14:14

## State
Branch `step1-activation-rescore-retrieval`. Executor session of the exact-inversion thread (now **yoado-41**; the
lanes were renamed by a restart — write-up is **yoado-81** (`notes/exact_channel_rev10.tex`, 36 pp), genuineness
audit **yoado-7e**, theory/source-verify **yoado-c9**, claims audit **yoado-b9**, deck/supervision **yoado-cd**,
explainer artifacts **yoado-21**). All of yesterday's jobs have finished and are written up; four are in flight, three
of them a knee-curve experiment that is the live question.

## The live question (Step 26, RESULTS)
The matched-arithmetic attacker recovers the private letters from a bf16-TRAINED adapter, and the k = 32 fidelity
reversal (bf16 4.6% better than fp16 7.5%, inverting the k = 16 order) was traced to **over-descent**: stopping fp16
at bf16's residual gives 4.72% against bf16's 4.56% (job 85300). Generalised as *the image finishes long before the
residual does* — a lower floor is a liability, the stopping rule is the **knee**, not the floor. Three sweeps now
measure the knee and test the mechanism: **86888** (fp16, stops 0.08 → 0.0019), **87369** (bf16, 0.08 → 0.0124),
**88743** (fp32, 0.08 → 1e-6). Pre-registered in RESULTS: U-shape with a minimum ~4.6% near residual ~0.011;
coincidence of all three on the shared **left arm** (residual above the knee); divergence of the right arms by
format; and **fp32 has no right arm** — a monotone descent to ~1e-6. An fp32 upturn would falsify the
distance-to-minimiser reading. Also still running: **650890** (most-leaking) and **658575** (weak model's letter ladder).

## Settled since the last note (all in RESULTS, STATUS, LESSONS)
- **The lead result** is the new class: letters at r = 64, k = 32, trained in fp32 — all eight recovered from random
  starts, recipe-free, tight tolerance. Step 23 (confident batch) is retitled *the exact-arithmetic existence corner*
  with both disqualifiers inline (frozen adapter `A_T = A_0`, feedback 4e-19; release 7.6e-18 with 2e-29 imprints).
- **Matched-arithmetic recovery from half-precision training:** bf16-trained 4.6% at k = 32 and 3.0% at k = 16,
  fp16-trained 7.5% / 0.97%, fp32-trained 6e-6 / 3e-7. Half-precision training is not protection.
- **A rank read off a low-precision release is not physical:** the cap-violating direction is the all-ones vector
  (σ tracks the softmax column-sum error), removing it leaves 10 not 8, and **no release-agnostic tolerance
  recovers the true rank** — 10ε under-counts even in FP64 (reads 3), tight tolerances over-count (11).
- Withdrawn on evidence: "fp16 stalled"; "the bf16 cells sit below a widened line"; "records what it gets wrong"
  (it is the margin-order law); "approximately and rarely" for the above-line cells (the argmin is a post-hoc
  nearest match, 113% from its intended target, zero landings in 5,000); the dynamic-range reading of the reversal.
- Closures: control ladder complete; wide-head ladder 18/15/13/12/9 of 20 at k = 8…40; subset selection rule holds
  while the omitted imprint exceeds the achievable residual; negative controls pass; OOD grids confirm the
  margin-order law across three encoders; the 3,000-iteration chart reruns rank nothing (only PCA converges).

## Gotchas
- Compute nodes cannot see the session scratchpad — inline diagnostics in the bsub heredoc.
- Every figure has had **one reader**; yoado-81's image reads time out. A third check is needed before any figure
  goes to the supervisor (STATUS carries this).
- `train_precision.py` is the executor's own module (matched route, landscape, knee sweeps); `certificate.py`,
  `subset_and_ood.py`, `lora_exact_inversion.py`, `vae_chart.py` are shared — do not edit while multi-invocation
  jobs hold them.

## State
Branch `step1-activation-rescore-retrieval`, HEAD `9edcbd2`. Executor session of the exact-inversion thread
(write-up: yoado-ed owns `notes/exact_channel_rev10.tex`; auditor: yoado-6e). Every result up to the start-scale
cell is written into `experiments/exact_inversion/RESULTS.md` (Steps 18–23 + the closures under "The start-scale
cell") and STATUS.md; ten WEXAC jobs are still running with Monitor watches; a small diagnostic (748065, queue
`short`) is recomputing the release for the cell where the ladder job 721391 tripped the imprint-sum assertion.

## Done this session
- **Imprint law** (Step 18): `B_T = Σ_i C_i`, `‖C_i‖ ∝` the accumulated softmax residual of image i (Kendall 28/28
  on the strong model; rank `B_T` = number of images above the floor in all 40 batches); recording is decided by
  margin order within the batch (9/9). A 98% model records nothing of confident digits (imprints ≤ 1e-24,
  rank 3) and everything of a new class (flowers on CIFAR, letter 'a' on MNIST; rank 8, aligned, cosine ~.5).
  Withdrawn en route: feature-Gram coupling (QR-basis artefact), label-multiset causation (confounded draws),
  "richer chart ⇒ worse conditioned", "foreign sets drawn better" (resolution), and a false "0/500 basin
  collapse" (a raw cell).
- **Certificate** (Step 22): `C = P_{row(B_T)⊥}A_T`, `Ch_i ≈ 0` exactly for recorded images (1e-16…1e-8 vs 0.1–1
  invisible) — recipe-free, label-free. Certificate-only inversion from RANDOM public-scale starts, on-chart,
  below `k < r − N′`: 51% of 2,000 starts land exactly on private images at k=6 (18% at k=8), floor fraction =
  recorded fraction (kernel count measured), argmin reliable below the line, spurious zeros dense at/above.
  Objective must be `‖Cφ‖/‖A_Tφ‖` (a constant-normalised form let a blank image win; audit catch).
- **Rank is a leakage dial three ways** (fixed-k arm, job 721391): at k=8 the basin is 16.6 → 74.4 → 96.4% at
  r = 16/32/64 (1/17/49 below the line) — distance below the line governs, rank buys reachability as well as
  budget; fidelity axis standalone (`chart_fidelity.py`, 2,000 held-out digits): class survival .52/.68/.88/.97 and
  instance survival .04/.11/.57/.94 at k = 6/8/16/32.
- **Membership needs no fidelity**: a recovered projection identifies its source among 10k projected candidates
  with certainty at every k, robust to 3% coordinate noise at k ≥ 8.
- **Second certificate cap `N′ ≤ m − 1`** (softmax simplex): twenty recorded digits on a 10-class head → rank 9,
  certificate dead for all; the SAME MLP with a padded 26-logit head → rank 19 (20 at tolerance 1e-14, rank C 44),
  residuals to 1e-8…4e-3. The twentieth image is collinearity (σ 6e-14 vs imprint 3e-7), not precision.
- **Precision**: dynamic range decides, not mantissa — FP16 underflows the small imprints, bfloat16 keeps them
  coarsely (most revealing low-precision format); structural closure only when many examples are comparably
  recorded. **(R5)** only `lr/N` enters the recurrence — N is not identifiable, only N′ (measured 6/6).
- Subset ("find some") test: recipe error found (batch size is part of the recipe) and fixed (`lr·N′/N`);
  corrected first row: predicted floor = residual at the recorded truth (1.1e-16 both).

## Update (after the note was written)
Job 728592 landed and is the thread's HEADLINE (RESULTS Step 23, commit 19170d3): r=64, k=32, 500 random starts →
66% land on a private digit, all eight found, argmin correct, chart instance-identifying (.94); reproduced in
721391. Also in: wide-head twenty-image cell 18 of 20 from 10,000 starts (725918); per-rank sweep table; k=10
bracket. Next step 1 below is DONE; continue from step 2.

## Write-up lane (yoado-ed's state, folded in at their request)
`notes/exact_channel_rev10.tex` (33 pp; builds via `bash scripts/rev10_figs/build_pdf.sh`; the PDF is gitignored, the
.tex is not; the Rev 9 source is Mac-only, so a MERGE NOTE at the top maps every Rev 9 number and the user merges on
Overleaf). Current as of their HEAD `99e9c37`:
- Headline = the NEW-CLASS cell (letters, r=64, k=32, trained in FP32, all eight from random starts, instance-level);
  the confident-batch cell demoted to "the extreme" with its 7.6e-18 release and frozen adapter stated; the measured
  section and the pitch both lead with the letters.
- Certificate block ordered mechanism → limits (N′ ≤ m−1, precision) → from-nothing → basin → three disclosures →
  synthesis (moved last so no claim precedes its own qualification).
- Carried with scope: ‖B_T‖ collapse along k; storage precision by the gap, not a fixed threshold; the FP16 zero
  scoped to this cell's norm; basin governed mostly by distance with k costing ~2× per 32 unknowns; the start-scale
  falsifier narrowing the feature-norm claim to one scale; R5 (batch size not identifiable), R6 (subset chosen by
  the level of the floor). The half-precision block ends with the three attackers on one release.
- NOT yet in the .tex (waiting on rows): the fp16/fp32/k=32 matched-arithmetic rows (782682); the k=58/60 FP64
  control (753886); the control ladder's completion (749362 — done, RESULTS has it); both ladder figures. **Three
  figure reads timed out on their side — no figure since the k=6 panel has been checked by yoado-ed personally.**
- Their lane's traps: a patch script that applies several edits and writes once discards all of them if a later
  anchor misses — verify against the file's bytes; never grep a pattern that spans a LaTeX line wrap or contains `$`.
- **Figures have had one reader.** yoado-ed's image reads timed out four times; nothing since the k=6 panel was
  verified by them. Before a figure goes to Gal: a third check against the rows (STATUS carries the same caveat).

## Next step(s)
1. ~~**Read job 728592**~~ (done — see Update) (`step76_r64k32_728592.jsonl`): r=64, k=32, confident on-chart — the cell combining budget
   (line 56, 24 below), basin (saturating with distance) and fidelity (instance id .94 at k=32). If random
   starts land on private images there, it is the thread's headline: instance-identifying images from random
   starts with no recipe. Send the row (not a reading) to yoado-ed.
2. Read the rest as they land, against RESULTS' pre-registrations: 725918 (wide-head twenty-image landings;
   extension pinned by the boundary image — read the nineteen), 721391 (per-rank sweeps), 706721 (k=10 bracket:
   floor and recorded fractions must come APART; hard1_diff), 706597 (subset controls: one-swapped and
   confident-only must NOT reach the recorded floor), 650890 (one-image attack: argmin label over 10 must be 0;
   k=24/25/26 line), 652786 (negative controls: random-encoder false-positive trigger must fail), 644064 (OOD
   inversion grids), 656205 (flowers mixed 1/4/7), 658575 (letters on mid/weak — predicted: ladder FLATTENS for a
   new class), 614344 (98% sweep — alias form?), 624463/624465 (3000-iter chart reruns, β family; fidelity
   ranking embargoed until then).
3. When 624463/624465/614344 finish (they hold `lora_exact_inversion.py`, `vae_chart.py`, `conditional_charts.py`,
   `trained_backbone.py`): the deferred edits in one commit — `jac_cond_truth` field per cell script, β kwarg
   fold-back into `vae_chart.train_vae`, off-chart best-point residual, `median_gap_to_chart`, per-iteration
   trajectory trace in `invert_lm`, certificate block appended to the full residual (report with/without),
   subset oracle flag. Run `basin_predictors.py` over all ladder cells (pool per-cell taus, never raw pairs).
4. If wanted: the 26-class head as the basin-with-power cell (built: `models/exact_inversion/mnist_mlp_m26_strong.pth`).

## Open threads / gotchas
- Running jobs (all watched): 728592, 725918, 721391, 706721, 706597, 650890, 652786, 644064, 656205, 658575,
  624463, 624465, 614344. Several share GPU nodes and are slow (3000-iteration LM cells take hours).
- Launch new modules with `python -u -m experiments.exact_inversion.<mod>` (a file path dies on import).
- Part B of `certificate.py` is meaningful ON-CHART only (`--settings on`); raw truths are not on the chart.
- `invert_lm`'s `restarts` = number of ATTEMPTS (0 runs nothing).
- `subset_and_ood.py` and `most_leaking.py` write rows as produced; older runs wrote only at the end.
- Every recovery cell except the certificate arm starts NEAR THE TRUTH (identifiability test, flagged per row).
- Read `setting` and `_truth` fields before any number; check the residual at the truth before reading a solve.
- The auditors' rules that bit today: grep for retraction survivors; matched-nuisance controls for cross-set
  claims; per-image quantities must be basis-independent; a count needs its gap; tolerance is the attacker's knob.

## Pointers
- RESULTS: `experiments/exact_inversion/RESULTS.md` (Steps 18–22 + corrections); STATUS.md top; LESSONS top.
- Scripts: `certificate.py`, `tolerance_sweep.py`, `precision_check.py`, `chart_fidelity.py`, `basin_predictors.py`,
  `margin_check.py` (imprints, traced release), `subset_and_ood.py`, `most_leaking.py`, `new_class.py`,
  `train_strong_backbone.py --n-out 26`, `batch_scale_check.py`, `blur_control.py`.
- Figures: `figures/exact_inversion/certificate_recovery_k6_706721.png`, `newclass_recoveries_preview.png`.
- Submit pattern: `bsub -q long-gpu -gpu "num=1" -R "rusage[mem=8192] select[ngpus>0]" -J <name> -o scripts/wexac_logs/<name>_%J.out -e ... <<'EOF' ... set +u; source .../conda.sh; conda activate .../rec; cd /home/projects/galvardi/yoado; python -u -m ... EOF`
- Siblings: yoado-ed (write-up, uds:/run/user/50309/cc-socks/4170091.sock), yoado-6e (auditor, ...4170067.sock).
