# Handover — 2026-09-03 20:52
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
