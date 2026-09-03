# Handover — 2026-09-03 22:39
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

## Update 2 (after the headline)
- Start-scale cell (737516), subset one-swapped control (706597), the 98% sweep closure (614344: no alias form),
  the k=6 replication on hard1_diff (706721) and wide-head k=16 (725918: 15/20) are in RESULTS/STATUS (commit
  9edcbd2) and were sent to yoado-ed with the ladder-figure path.
- Ladder job 721391 died on `subset_and_ood.release_and_imprints`'s assertion (`||sum_i C_i - B_T||/||B_T|| < 1e-10`)
  at `mnist_control r=64` on the cell AFTER k=16 (ks were 16 24 32 40 48 56; the on-chart batch depends on k, so the
  release differs per k). All confident rows and mnist_control k=8 (r=16/32/64) and k=16 (r=64) are on disk and
  unaffected (the assertion is at release time). Resolved: FP64 roundoff on a release that shrinks 4 orders at k>=24 (control digits become confident on-chart); assertion floored (bc0f907); the five control cells resubmitted as job 749362 (`step69_cert_rank_ctrl_749362.jsonl`). Diagnostic job 748065 printed ||B_T||, the absolute and relative
  mismatch and the imprints at k=16..56 (`scripts/wexac_logs/imprint_chk_748065.out`). Suspected: roundoff on a tiny
  B_T (well-classified projected digits) — if so, relax the assertion to an absolute floor and log it; do NOT edit
  `subset_and_ood.py` while 706597/644064 (its main module) or 706721/725918 (import it, multi-invocation job
  scripts) are running.
- Diagnostic scripts must be inlined in the bsub heredoc: compute nodes cannot see the session scratchpad (/tmp is
  node-local); job 747682 died on "No such file".

- Auditor ask (yoado-6e): when 749362's rows land, read ‖B_T‖ and rank from THAT job's saved release tensors under `results/exact_inversion/step69_ctrl_749362_r64/` (not from the CPU diagnostic) before writing the collapse-along-k finding beside its Part B rows.

## Update 3 (the release-vs-k finding and the precision programme)
- **Job 752500 (RESULTS "The release against k"):** the confident batch's release collapses monotonically with the
  chart — ‖B_T‖ 0.47 (k=8) → 0.08 (16) → 1.6e-13 (24) → **7.6e-18 (32, the HEADLINE cell)** → 2.8e-24 (56), tracking
  the projections' margins 11 → 60; the control plateaus at 1e-6…1e-4 (margins 11–15). No "window"; the rank dip at
  k=16 is a relative artefact. The headline number now carries "release norm 7.6e-18" (STATUS scope caveat).
- **Mechanism, corrected before any row (RESULTS Step 24 continued):** off-class softmax entries exp(−margin) survive
  to a format's subnormal floor (fp32/bf16: margins ~100; fp16: ~16.6); unit roundoff only zeroes the own-class
  entry; the release never feeds back (1e-18 vs logits 40, below the ulp even in FP64) so A_T = A0 and B_T is the
  one-step gradient × T. Predicted: headline survives fp32/bf16 TRAINING, erased by fp16's range; fp16 STORAGE zeroes
  the whole file (0 found); spectrum (primary predictor) and imprints agree in this cell: fp32 5, tf32 2, bf16 2.
- **Jobs in flight for this:** 753371 (headline cell from a quantised release, 9 cells; the fp64 cell first),
  753886 (bf16 vs fp64 at k=58/60 — relabelled "is a bf16 release attackable where the chart is faithful"; the
  "widen the line" trade hypothesis was withdrawn: on-chart the fidelity is the training k), 760909 (letters 'a'
  as an 11th class, loop run in fp64/fp32/bf16/fp16, search from each — the decisive arm), 760912 (confident k=32,
  control k=32, confident k=8 in the four formats), 749362 (control ladder k=24…56). Read rows against the
  pre-registrations; the letters' projected margins at t=1 are read from the fp64 row first.
- `train_precision.py` is new (no shared module edited — 753371/753886 are multi-invocation jobs holding
  certificate.py; do not edit certificate.py / lora_exact_inversion.py / subset_and_ood.py until they finish).
- **Step 24 results so far (753371):** fp64 8/8 (reproduces the headline); fp32 and tf32 at a TIGHT tolerance 5/8
  (the three weakest directions destroyed, residual 0.9 at their truths); the "noise-matched" tolerance loses images
  (use the noise rank); fp16 file underflows to exactly zero, 0/8; bf16 pending. Lesson: the quantised spectrum's floor
  is ~3 orders below the unit roundoff — predict from the measured spectrum. k=10 bracket (706721) closed: floor and
  recorded fractions separate 6x AT the line. Flowers corrected design (656205): identifiable near the truth,
  search failure from random starts; mixed batches rerunning (762253, save-path bug fixed a18893a).
- **Step 25 (760909/760912/763805, RESULTS):** the LEAD result — letters as a new class, all 8 from random starts at
  FP64 (k=32, release 1.0, adapter moving 43%); fp32 training: 8/8 at k=16, 6/8 at k=32 (10ε tol; tight-tol job
  764976 pending — if 8/8 the sentence for Gal is unqualified); bf16/fp16 TRAINING keep the norm, degrade directions
  2–25%, certificate finds 0. Headline cell: fp32 exact, mismatch 6e-32 (signal). Landing error ≈ 5× residual at truth.
  Figure: figures/exact_inversion/letters_recovery_k32_760909.png.
- **Step 26 (771329, pre-registered):** recipe route (FP64 simulator, near start) vs the bf16/fp16-trained letter
  releases — decides whether "recorded but not certificate-recoverable" is an extraction gap or a demonstrated
  non-protection. 764976 (tight-tolerance certificate search from fp32/bf16-trained releases) still running; 763805
  (random head row) done: reproduces the zero row. Script freeze: certificate.py / lora_exact_inversion.py /
  subset_and_ood.py / vae_chart.py until 753886 and 749362 finish (multi-invocation); train_precision.py is safe to edit
  between its jobs (single invocation each).
- **CLOSED (764976):** letters k=32 from the fp32-TRAINED release at the tight tolerance — ALL 8 from random starts
  (32.8%); the unqualified sentence holds at the instance-identifying chart. **Step 26 (771329) k=16 so far:** recipe
  route recovers fp32-trained letters to 2e-6; on bf16-trained it lands on an ALIAS (residual 240× below the truth's
  floor, wrong images) — recorded, recoverable by neither route we ran, an open; fp16 and k=32 rows pending.
- **779207 (Step 26 addendum):** matched-arithmetic landscape of the bf16/fp16/fp32 letter releases (response to
  perturbations; residual along start→truth). Pre-registered: bf16 map responds 1e-2…1e-1 to a 1e-6 perturbation →
  no differentiable matched solver exists (verification oracle only); falsifier: linear response → run a matched LM.
- **Landscape (779207/779969) FALSIFIED the needle prediction toward the attacker:** the bf16 training map is smooth
  (response 4.4e-6 to δ=1e-6; rounding floor 2–4e-3; monotone start→truth at every window). The pre-committed
  matched-arithmetic recipe route is running (782682: `matched_lm`, Z-parametrised A₀ candidate, FP64 Jacobian
  surrogate, bf16 residual). Pre-registered: recovery to ≲5e-2 → "not protection" demonstrated; falsifier: alias
  persists (A₀ floor). Control ladder complete; ladder figure has the control cells. bf16 k=58: 0 of 5,000, near-miss 7.8%.
- **Step 26 RESULT (782682, bf16-trained letters k=16, matched-arithmetic route): recovered to 3% median image
  error (max 6.9%), raw error at the chart floor; residual 0.014 vs the truth's A₀ floor 0.023; Z error 0.13 —
  not protection, DEMONSTRATED. fp16/fp32 and k=32 rows pending.**

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
