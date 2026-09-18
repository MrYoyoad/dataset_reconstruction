# Bootstrap / iterative chart — WP4 (plan 2026-09-18), round2 TEST 8, first run

Code: `experiments/bootstrap_chart/bootstrap.py` · runner: `scripts/run_bootstrap_chart_wexac.sh {smoke|full}` ·
rows: `results/bootstrap_chart/rounds_<jobid>.jsonl` (one row per release × variant × round × arm) · tensors:
`results/bootstrap_chart/<release>_<variant>_round<t>_<arm>_<jobid>.pth` · grids: `figures/bootstrap_chart/*.png`.
Every number below is provisional until the row is read by a second session.

## Question
Can the chart be improved from what the attack recovers, without private data touching the chart fit? Two routes:
**A** recognise the class of the round-0 recovery and refit the chart on that class's public pool; **B** refit a LOCAL chart
around the recovery (K = 200 nearest public images of the recognised class).

## Setting (both releases, privates RAW)
| | MNIST | CIFAR |
|---|---|---|
| backbone | `mnist_mlp_strong.pth` (784-1000-1000-10 GELU); base gate PASS in job 355833 (99.83 % train, CE 6.3e-3); re-measured in-job on every row | `cifar10_cnn_newclass.pth` (CNN, 256-dim penultimate); train / test / CE measured in-job |
| added class | EMNIST letter `a` (test split), the letters cell's eight (`randperm(seed+7)[:8]`) | CIFAR-100 `motorcycle` (test split), the ladder's eight photographs |
| adapter | head LoRA r = 64, B0 = 0, T = 400 SGD steps, lr 0.01, A0 = randn/√n (seed+7), 11th head row zero | same |
| solver | `certificate.lm_cert` (LM on ‖C φ(ψ(w))‖/‖A_T φ(ψ(w))‖), 300 iters, k = 32 | same |
| round-0 chart | PCA-32 of ALL 124,800 EMNIST-letters train images (26 classes) | PCA-32 of ALL 50,000 CIFAR-100 train images (100 classes) |
| recogniser | 784-512-512-26 MLP trained in-job on the public pool (validation tenth held out) | CNN(m=100) trained in-job on CIFAR-100 train (validation tenth held out) |
| starts | 200 per global chart; variant B: 1 warm + 8 random per slot | 200 per global chart; variant B: 1 warm + 2 random per slot (CNN ≈ 25 s/start) |

Both recognisers are trained with 50 % chart-projected augmentation and **calibrated on PCA-32 projections of held-out
public images, per class** (audit 2026-09-18); no ImageNet name map is used (CIFAR-100 motorcycle has no ImageNet counterpart).

## Metrics per round, per image (audit 2026-09-18)
`chart_err_truth` — projection error of the TRUE image in that round's chart (fidelity) · `err_proj` — recovery-to-PROJECTION
(did the solver reach the chart's best) · `err` — recovery-to-TRUTH (did the chart improve) · `objective` — certificate
objective at the recovery · `floor` — objective at the truth's projection into that round's chart · flags: `reached_projection`
(err_proj < 1e-2), `landed` (err < 1e-2; dead by construction while chart error ≈ 0.3, reported anyway), `at_floor`
(objective ≤ floor), `alias_flag` (at the floor but not at the projection — information problem), `solver_short` (above the
floor and not at the projection — optimisation problem) · recognised `top1` / `top2` and the true class's rank in the
round-level class ordering. Attacker-side selection (nearest of the N best-objective de-duplicated candidates) and
oracle-side selection (best start of any kind, `best_any_*`) are recorded separately and never mixed.

## Void conditions (pre-registered)
* `void_round0` (as run, after the reference change below): round-0 median `err_opt` > 1e-2 AND the objective at the
  recovery is above the oracle-start optimum on more than N/2 images — the solver fell short of the chart's best; nothing
  to bootstrap from → VOID. (The audit's first form, median `err_proj` > 1e-2, is superseded: see "Reference change".)
* `void_variant_A`: fewer than 5 of 8 round-0 candidates top-1 correct, OR the recogniser's calibration accuracy on
  projections of the true class ≤ 5/8 → variant A is VOID (not null).
* A round whose control could not have disagreed (identical chart → identical deterministic run) is marked `identical_to`.

## Pre-registered outcomes
* **IMPROVES** — `chart_err_truth` and `err` fall over rounds AND beat the wrong-class (A) / random-anchor (B) control.
* **REFIT-ONLY** — the control improves as much → the gain is chart narrowing / locality, not what was recovered.
* **STALLS** — no change after round 1.
* **ALIAS** — any recovery at the floor with `err_proj` not small is flagged and never counted as reaching the chart.
* Variant A's ceiling is the per-class public PCA fidelity (C7 measured 0.18 at k = 66 on CIFAR): its value is reaching that
  WITHOUT the class label. Only variant B can go below per-class PCA.

## Controls (same job, same starts, same row format)
A: second-ranked class chart (wrong-class refit) · true-class chart (oracle label, NOT attacker-available).
B: random-anchor (neighbours of a random public image of the recognised class; the decisive control) · oracle truth-anchor
(neighbours of the truth itself, NOT attacker-available).

## Smoke run — job 355880 (MNIST only, 10 starts, 1-epoch recogniser, 1 round of each variant; 2 min on short-gpu)
Purpose: exercise every code path. The numbers are 10-start numbers and are NOT the result.
* Base gate measured in-job: 99.83 % train / 98.21 % test / CE 6.35e-3 → PASS (matches job 355833).
* Release: rank B_T = 8, rank C = 56, certificate line k < 56 (k = 32 below); objective at the raw truths ≤ 1.4e-22.
* Round 0 (generic PCA-32 of 124,800 letters, explained variance 0.769): chart error of the truths median 0.286 (range
  0.21–0.71); recovery-to-truth median 0.737; **recovery-to-projection median 0.708, reached 0/8 → `void_round0` fires**.
  The objective at every recovery (3e-4 … 4e-3) is 3–500× BELOW the floor at the truth's projection (5e-3 … 4e-2): the
  solver did reach and pass the chart's projection of the truth — the projection is simply not the chart's argmin of
  the certificate objective when the privates are raw. Every recovery is therefore `at_floor` + `alias_flag`
  (an equally-or-better-scoring chart point that is a different image), `solver_short` = 0. Read literally: an
  information problem in the generic chart, not an optimisation failure. Whether the 200-start run changes this is
  what the full job measures.
* Recognition (1-epoch MLP, calibration on projections 72.2 % overall, 55.1 % for `a`): round-level class ordering
  `a, d, r, u, f` (true class rank 1), but only 3/8 candidates top-1 correct → `void_variant_A` fires at this
  classifier strength.
* Variant A round 1: recognised `a` chart → err-to-truth median 0.596 (chart err 0.245); wrong-class `d` → 0.799
  (chart err 0.342); oracle = recognised (identical chart, marked `identical_to`).
* Variant B round 1 (1 warm + 2 random starts): recovery-anchor 0.797, random-anchor 0.756, oracle-anchor 0.712.
* Cost: 0.5–2.1 s per LM start on the MLP (the wrong-class chart runs to the 300-iter cap more often).

## Reference change after the smoke (coordinator, 2026-09-18 ~01:15)
The smoke showed the projection of the truth is NOT the chart's argmin of the certificate objective for raw privates
(evidence, job 355880 round 0, per image: objective at the recovery 3.1e-4, 5.0e-4, 3.8e-3, 4.4e-4, 4.4e-4, 2.9e-3, 3.8e-3,
4.4e-4 versus objective at the truth's projection 7.2e-3, 1.26e-2, 1.23e-2, 2.04e-2, 1.14e-2, 4.14e-2, 2.79e-2, 5.3e-3 —
ratio `objective_over_floor` 0.003–0.57, below 1 on 8/8; same pattern on all A/B smoke rows). So a projection-based
void rule would void a solver that beat its own reference. Fix applied to the REFERENCE, not the rule: per round and per
image the same LM solver is run from the ORACLE START `w0 = coords(projection of the truth)` (NOT attacker-available) to
give `x*_chart`, the chart optimum nearest the truth. New per-recovery columns: `err_opt` = err(recovery, x*_chart)
[solver metric], `err_opt_truth` = err(x*_chart, truth) [chart fidelity — the number a better chart must reduce],
`err` = err(recovery, truth), `objective` vs `objective_opt`. Flags: `reached_opt` (err_opt < 1e-2), `alias_in_chart`
(objective at or below the oracle-start optimum but far from it — an alias within the chart, NOT void),
`solver_short_opt` (above it and far from it). **Void round 0 = median err_opt > 1e-2 AND objective above the
oracle-start optimum on more than N/2 images.** Projection-based columns are kept. Variant B runs one oracle start per
slot (for the slot's nearest truth); a truth no slot points at carries `opt_available=false` in the per-truth view.
Jobs 355903 / 355904 (started 00:49) were killed ~25 min in and resubmitted with this in.

### Re-smoke with the oracle-start reference — job 355915 (10 starts; crashed at the first variant-B row on a duplicate
row key, fixed; round 0 and variant A completed)
* x*_chart from the oracle start, round 0: objective 2.6e-4 … 1.1e-3 per image versus 5.3e-3 … 4.1e-2 at the projection
  (x*_chart is 10–80× better than the projection on every image); err(x*_chart, truth) median 0.433 (range 0.37–1.24:
  one oracle start wandered far from its truth while lowering the objective).
* Round 0 recoveries (10 starts): err(recovery, x*_chart) median 0.718, reached 0/8; objective ABOVE the oracle-start
  optimum on 7/8 → `solver_short_opt` 7, `alias_in_chart` 1 → **void_round0 fires at 10 starts** (a solver statement at
  this budget; the 200-start run decides).
* Variant A round 1, recognised `a` chart: err-to-x*chart 0.640 (x*chart-to-truth 0.401), reached 1/8, alias-in-chart 3,
  solver-short 4; wrong-class `d`: 0.752 (x*chart-to-truth 0.696), solver-short 7.

## Full run — jobs 355987 (`bsc_mnist`) and 355988 (`bsc_cifar`), long-gpu, submitted 2026-09-18 ~01:35
(post-fix smoke: job 355986). Superseded: 355903 / 355904 (killed at ~25 min, pre-reference-change) and 355916 / 355917
(killed before start, carried the duplicate-key bug); no rows from any of them are kept.
Commands (from `scripts/run_bootstrap_chart_wexac.sh full`):
`python -u -m experiments.bootstrap_chart.bootstrap --releases mnist --starts 200 --local-starts 8 --a-rounds 2 --b-rounds 4 --K 200`
`python -u -m experiments.bootstrap_chart.bootstrap --releases cifar --starts 200 --local-starts 2 --a-rounds 2 --b-rounds 4 --K 200`
Rows → `results/bootstrap_chart/rounds_355987.jsonl`, `rounds_355988.jsonl`. `mnist_mlp_strong_full.pth` did not exist at
submission and per the 2026-09-18 audit would not have been used anyway.
### MNIST — job 355987 (DONE 2026-09-18; 16 rows; 200 starts per global chart, 1 warm + 8 random per slot in B)
Base gate measured in-job 99.83 % / 98.21 % / CE 6.35e-3 PASS. Release rank B_T 8, rank C 56, k = 32 < 56. Recogniser
(8 epochs): validation 90.7 % raw, **90.0 % on PCA-32 projections, 86.9 % for class `a`**.

**Void decisions.** `void_round0 = False`: median err(recovery, x*_chart) 0.553 > 1e-2, but the objective at the recovery
is above the oracle-start optimum on only **1/8** images (objective 3e-4 at the matched candidates vs x*_chart 3e-4 … 1.1e-3)
— 2/8 reached x*_chart exactly, 5/8 are aliases within the chart, 1/8 solver-short. `void_variant_A = False`: round-0
top-1 correct **7/8** (≥ 5) and calibration on projections of `a` 0.869 > 0.625; round-level class ordering `a, n, h, g, r`
(true class rank 1). Every "landed vs truth" count is 0/8 in every row (chart error 0.25–0.45 of the truths).

Medians over the 8 truths (per-truth view). `x*→truth` = err(x*_chart, truth), the chart's fidelity at its own optimum.

| variant | round | arm (chart) | chart err | err(rec, x*) | x*→truth | err(rec, truth) | reached x* / landed | alias-in-chart | solver-short | top-1 on cands (calib. proj.) |
|---|---|---|---|---|---|---|---|---|---|---|
| round 0 | 0 | generic PCA-32, 124,800 letters | 0.286 | 0.553 | 0.433 | 0.604 | 2 / 0 | 5 | 1 | 63 % (`a` 86.9 %) |
| A | 1 | recognised `a` (4,800) | 0.245 | 0.730 | 0.401 | 0.758 | 2 / 0 | 5 | 1 | 88 % |
| A | 1 | wrong-class `n` (control) | 0.306 | 0.824 | 0.501 | 0.831 | 1 / 0 | 7 | 0 | 38 % |
| A | 1 | oracle class `a` | = recognised (identical chart, `identical_to`) | | | | | | | |
| A | 2 | — | re-recognition on round-1 candidates: `a` again → STABLE, stopped | | | | | | | |
| B | 1 | recovery-anchor | 0.281 | 0.000 | 0.266 | 0.838 | 2 / 0 | 0 | 0 | 100 % |
| B | 1 | random-anchor (control) | 0.454 | 0.195 | 0.727 | 0.830 | 2 / 0 | 2 | 0 | 38 % |
| B | 1 | oracle truth-anchor | 0.249 | 0.000 | 0.316 | 0.711 | 3 / 0 | 0 | 0 | 100 % |
| B | 2 | recovery-anchor | 0.299 | 0.000 | 0.301 | 0.809 | 2 / 0 | 0 | 0 | 100 % |
| B | 2 | random-anchor | 0.339 | 0.000 | 0.396 | 0.614 | 4 / 0 | 0 | 1 | 88 % |
| B | 2 | oracle truth-anchor | 0.249 | 0.000 | 0.316 | 0.711 | 3 / 0 | 0 | 0 | 100 % |
| B | 3 | recovery-anchor | 0.291 | 0.000 | 0.257 | 0.812 | 2 / 0 | 0 | 0 | 100 % |
| B | 3 | random-anchor | 0.329 | 0.000 | 0.498 | 0.671 | 3 / 0 | 0 | 1 | 88 % |
| B | 3 | oracle truth-anchor | 0.249 | 0.000 | 0.316 | 0.711 | 3 / 0 | 0 | 0 | 100 % |
| B | 4 | recovery-anchor | 0.315 | 0.000 | 0.307 | 0.816 | 2 / 0 | 0 | 0 | 100 % |
| B | 4 | random-anchor | 0.314 | 0.000 | 0.433 | 0.550 | 3 / 0 | 1 | 1 | 88 % |
| B | 4 | oracle truth-anchor | 0.249 | 0.000 | 0.316 | 0.711 | 3 / 0 | 0 | 0 | 100 % |

In variant B the optimum-based columns are per SLOT (one oracle start per slot, for the slot's nearest truth); the
per-truth `err(rec, truth)` medians are dominated by **slot collapse**: the 8 recovery-anchored slots point at only
2 truths in every round (`nearest_truth` = [7,0,7,0,7,0,0,0]), the oracle-anchored at 3, the random-anchored at 4–5.
Per slot, recovery-anchored err(candidate, its truth) is 0.26–0.50 in round 1 and 0.28–0.39 in round 4; the
random-anchor control's per-slot fidelity is 0.36–1.07 (r1) and 0.28–0.61 (r4).

**MNIST verdicts (pre-registered categories, read literally).**
* **Variant A — STALLS.** Recognition works (7/8, class rank 1, well above the 5/8 bar) and the recognised-class chart is
  more faithful than the generic one at its own optimum (x*→truth 0.401 vs 0.433; chart err 0.245 vs 0.286) and than the
  wrong-class control (0.501) — but the recovery error to the truth does not fall (0.758 vs 0.604 at round 0) and
  err(rec, x*) rises (0.730 vs 0.553): inside the `a` chart the certificate objective is minimised by ONE mode (grid: 7 of
  8 candidates are the same upright "A"), 5/8 aliases within the chart. Round 2 stable. Not IMPROVES (recovery error did
  not fall); not REFIT-ONLY (the wrong-class control is worse on every column).
* **Variant B — STALLS per truth; a per-slot fidelity gain that the control does not match.** Per-truth median recovery
  error is flat over rounds 1–4 (0.84 → 0.82) and is BEATEN by the random-anchor control (0.83 → 0.55), purely because the
  random anchors cover 4–5 truths while the recovery anchors collapse onto 2. Per slot, the recovery-anchored local chart
  reaches its own optimum exactly (err(rec, x*) = 0 in 27/32 slot-rounds) at x*→truth median 0.27–0.31 — below the
  per-class PCA fidelity 0.40 and below the random-anchor control's 0.40–0.73 — so locality anchored on the recovery does
  buy fidelity for the truths it points at; it does not improve after round 1 and it does not reach the other 6 truths.
  ALIAS is not the verdict (alias-in-chart 0 in the recovery-anchored rows); no round is void.
* Two observations for the write-up: (i) the projection is not the chart's argmin (round 0: x*_chart objective 10–80×
  below the projection's on every image); (ii) the oracle-anchored arm returns the identical numbers in rounds 1–4 — the
  warm start from the previous candidate is the best start every round, so the chain converges in one round.

### CIFAR — job 355988 (running)
_(to be filled in when it finishes)_

## Verdict
MNIST: variant A STALLS, variant B STALLS per truth (per-slot fidelity gain 0.27–0.31 vs 0.40 class PCA, coverage 2/8).
CIFAR: pending.

## Deviations from the plan
* Variant B on the CNN uses 2 random starts + 1 warm start per slot per round (not 200) because a CNN LM start costs ≈ 25 s
  (ladder job 435321: 400 starts in 9,799 s); the global charts keep 200 starts.
* A0 is drawn as in the ladder / `ntk_vs_certificate.py` (`randn(r, n, seed+7)/√n`), not from `new_class.py`'s shared
  generator stream (which depends on that script's loop order); the privates are the same eight in both.
* The recogniser's chart-projected augmentation and per-class calibration follow the 2026-09-18 audit, not the original WP4 text.

## P8 — Moving between two charts: the Jacobian at the hand-over (jobs 366170 MNIST, 366171 CIFAR; 2026-09-18)

*Plan `notes/plan_2026-09-18_multilayer_parameter_program.md` P8 + audit item 10. Code
`experiments/bootstrap_chart/handover_jacobian.py`, runner `scripts/run_chart_program_wexac.sh p8`, rows
`results/bootstrap_chart/handover_{366170,366171}.jsonl`, builder read-out `HANDOVER_NOTES.md`. Analysis on the saved
355987 / 355988 releases and charts (fwd_check exact; variant-B local charts rebuilt from the saved anchors reproduce the
saved rows). Reference is `x*_chart`, never the truth's projection. Single read, provisional.*

For each round pair (t → t+1), each of the eight truths, `J = C · J_φ · V` at `x*_chart` under `V_t`, `V_{t+1}` and the
orthonormalised union `[V_t V_{t+1}]` (rank at 1e-10 with the absolute floor; medians over 8):

| release | chain | rank `V_t` / `V_{t+1}` / union | cond `V_t` / `V_{t+1}` / union | gap (union) | min principal angle | alias flags /8 |
|---|---|---|---|---|---|---|
| CIFAR 355988 | round0 → A1 recognised / wrong / oracle | 32 / 32 / **56** | 36 / 37 / 183–226 | 2e13 | 1.6–2.7° | **7–8** |
| CIFAR | A1 → A2 recognised | 32 / 32 / 56 | 33 / 33 / 199 | 1e13 | 3.6° | 7 |
| MNIST 355987 | round0 → A1 (three arms) | 32 / 32 / 56 | 65 / 49–54 / 365–376 | 1e13 | 1.6–2.0° | 5–7 |
| MNIST | B recovery-anchored, rounds 0→1→2→3→4 | 32 / 32 / 56 | 56–63 / 54–59 / 412–558 | 8e12 | 7.6° → 3.0° → 1.1° → 1.1° | **0, 1, 2, 2** |
| MNIST | B random-anchor | 32 / 32 / 56 | 55–76 / 48–70 / 344–593 | 1e13 | 6–11° | 4, 0, 3, 1 |
| MNIST | B oracle-anchor, rounds 1→2→3→4 | 32 / 32 / **32** (chart stationary) | 62 / 62 / 62 | ∞ | 0° | 0 |

**Readings.** (1) **Two different 32-charts span 64 > 56 = rank C, and the union Jacobian has rank exactly 56 with a
gap of 1e13**: the hand-over opens an 8-dimensional exact null space by construction — every cell, both releases. The
union is also ×5–8 worse conditioned than either chart. This is a measurement (the round-1 chart is built after seeing
`C`; audit 10c), not an identifiability theorem. (2) **On CIFAR global charts the coverage reading does not hold**: 7–8
of 8 recoveries carry the alias flag (objective at the floor, recovery ≠ `x*_chart`), so there the recoveries are
equal-scoring different chart points. On MNIST the recovery-anchored chain has **0 alias flags in round 1** (coverage
reading survives there) and 1–2 in later rounds (weakened, not falsified). (3) The recovery-anchored chart chain
converges: the smallest principal angle between successive charts falls 7.6° → 1.1° with up to four shared directions
by round 3; the oracle-anchored chain is exactly stationary from round 2. (4) Carrying `x*(t)` into chart t+1 costs
×1–3 in objective on the recovery chain vs ×2–10 on class charts; `err(x*_t, x*_{t+1}) ≈ 0.8` on CIFAR.

**NOT shown.** CIFAR variant-B pairs (bootstrap had not reached them); no test of identifiability on the union;
single seed; the alias flag is the pre-registered falsifier of the coverage reading only (audit 10b).
