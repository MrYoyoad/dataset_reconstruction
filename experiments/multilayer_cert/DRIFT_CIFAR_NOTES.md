# Drift harness on the trained CIFAR-10 MLPs — backlog E.1 (builder notes, 2026-09-19)

Backlog item 1 of `notes/meeting_2026-09-15_decisions_and_backlog.md` §E.1 ("multilayer / changing-input reconstruction
on a trained backbone"): the LoRA adapter INSIDE a trained CIFAR-10 backbone with the layer below also adapted (so the
target layer's input drifts), raw private images from a NEW CIFAR-100 class, certificate chart search from random
starts, wrong-release control. The coordinator writes RESULTS.md / STATUS / the ledger; nothing here is a claim.

## What was added (only `drift_cert.py` and `run_drift_cert_wexac.sh` were edited; MNIST defaults unchanged)

`experiments/multilayer_cert/drift_cert.py`
- **Loader** for the `cifar_newclass.MLP` state_dict (`l1.weight (1000,3072)`, `l1.bias`, `l2.weight (1000,1000)`,
  `head.weight (10,1000)`) into the deep_stack convention `Ws=[l1,l2,head], b1=l1.bias` — GELU after layers 1 and 2,
  none after the head, bias on layer 1 only: read off `cifar_newclass.MLP.phi = gelu(l2(gelu(l1(x))))`, `head` bias-free.
  `meta.layout = "cifar_mlp"`, with the checkpoint's own `test_acc / train_acc / train_loss / overtrain / epochs`.
  Base-gate record (`experiments/exact_inversion/BASE_TRAINING_GATE.md`, job 355833): `cifar10_mlp_overtrained_newclass`
  100.000 % train / 57.31 % test / CE 6.8e-4 → WP0 **PASS**; `cifar10_mlp_newclass` 93.64 % train / 57.9 % test →
  WP0 not met, used as-is because it is the ladder's `mlp_motorcycle` base (gate (0.50, 0.90) in `ladder_cell.EXAMPLES`).
- `--dataset {emnist,cifar100}` (default emnist = the old path), `--class-name` (default `a` / `motorcycle`),
  `--cifar-root` (default `data`, = `perceptual_id.CIFAR_ROOT`). CIFAR: loader gate = CIFAR-10 test accuracy on 2000
  images > 0.5 (cifar_newclass's mlp gate; MNIST keeps > 0.9); privates = the CIFAR-100 class TEST split at
  `randperm(seed+7)[:N]` (torch Generator — the join key of `ladder_cell.py` / `cifar_newclass.py`; **not** the numpy
  `RandomState(seed)` of `two_walls.py`, LESSONS 2026-09-18 "two scripts, two RNGs"); chart = PCA-k of the class TRAIN
  split; head extended by a zero row, m = 11, every private labelled 10 (`cifar_newclass.py` W0 / y). Tier 2 key
  `('cifar100', <class>)`, shape (3,32,32); `perceptual_id.feature_extractor` already handles the checkpoint name.
- `grid_png(..., shape=)` draws RGB for 3-channel images (MNIST grids identical).
- `--wrong-release` (both datasets): the release trains on `randperm(seed+7)[N:2N]` (cifar_newclass `sel[N:]`,
  ladder_cell `perm[N:2N]`; on-chart projected under `--private onchart`) and the certificate searches for the TRUE
  eight. Drift / N' / rank B_T are measured on the release's own trajectory (`H0s` of the training images);
  `res_truth`, `objective_at_raw_truth`, landings and tier 2 are against the true eight (`H0s_true`). New fields:
  `wrong_release`, `wrong_release_idx`, `solves.*.objective_at_release_train`, `release_floor_objective`,
  `certificate_nondegenerate` (= min non-degenerate start objective > 1e4 × the release's own floor, ladder_cell's
  assertion). The per-image alias floor uses the release's own floor there (the truth's objective is O(1) by
  construction, so "100 × the objective at the truth" would call every start an alias). Tag suffix `_wrong`.
- New row fields on every row (MNIST too, values only): `dataset`, `class_name`, `image_shape`, `wrong_release`,
  `wrong_release_idx` (None), and the three `objective_at_release_train*` solve fields (== the truth's on a normal cell).
  No number on the MNIST path changes (same code path, same RNG draws: the wrong-release slice is only taken when set).
- Known, NOT changed here: the cell-level `contaminated` flag still overwrites the per-image verdict
  (LESSONS 2026-09-18); the coordinator's `relabel_drift_verdicts.py` repairs it read-side and applies unchanged to
  these rows.

`scripts/run_drift_cert_wexac.sh`
- `submit cifar_smoke` (short-gpu, -W 0:40): over-trained MLP control (`--adapt 2 --target 2`, T 1/20, 20 starts,
  both certificates), drift (`--adapt 1 2 --target 2`, r 16/64), head target no-solve, the 60-epoch checkpoint through
  the gate, and a 5-start wrong-release cell.
- `submit cifar` (long-gpu, -W 24:00, `-gpu num=1:gmem=20G`, `rusage[mem=24576]`, `set +u`, `python -u`): for
  PRIVATE ∈ {raw, onchart} × seed {1,2,3} × model {overtrained, newclass} × target {2, 3}: control (adapter on the
  target only, 15 cells T {1,5,20,100,400} × lr {0.003,0.01,0.03}) + drift (`--adapt tgt-1 tgt`, r_lower {4,16,64},
  45 cells); r = 64, k = 32, CERT=both, STARTS=200 → 24 jobs of 60 cells. Plus per PRIVATE × model × target one
  wrong-release job (control + drift r_lower 16, T grid, lr 0.01, seed 1) → 8 jobs. Outputs
  `results/multilayer_cert/drift_cert_cifar_{model}_tgt{t}_s{seed}_{private}_{job}.jsonl` and
  `drift_cert_cifar_wrong_{model}_tgt{t}_{private}_{job}.jsonl`, tensors `results/multilayer_cert/drift_cert/*_{job}.pth`,
  grids `figures/multilayer_cert/drift_cert/cifar10_mlp*_{job}.png`; git hash (+dirty), script sha, cmd, host on every row.
- Timing basis: the strong-MLP p3 job (60 cells, 200 starts, both certificates) took 3.4 h (366214); the CIFAR MLP
  costs 0.74 s/start on the ladder (295 s / 400 starts, `mlp_motorcycle` pca row) — same order, so ≈ 4–6 h per job.

## Pre-registered outcomes for the CIFAR cells (written BEFORE the full `cifar` submit; smoke 369540 pending)

Layer widths: n_1 = 3072 (input), n_2 = 1000, n_3 = 1000 (head input), m = 11, r = 64, N = 8, k = 32.

1. **Zero-drift controls (`--adapt t --target t`).** Both models, both targets, every (T, lr): `rho_full` and
   `rho_trunc` at the FP64 floor (≲ 1e-14), `N' = rank B_T = 8`, `rank_C_full = 56 = r − N` with a clean gap,
   `contaminated = False`. Target 3 control = the `cifar_newclass` head cell: with on-chart privates expect ≥ 7/8
   `recovered` at T = 400, lr 0.01 (the ladder oracle-eps0 row found 7/8 at 200 starts; nc rows 8/8 on the 60-epoch
   MLP); with raw privates tier 1 cannot land (PCA-32 projection error of motorcycles ≈ 0.32) → `chart-limited` where
   x*_chart is reached, `err_opt_truth` ≈ the chart error. Target 2 control (hidden layer, frozen input): same
   arithmetic, the smoke's MNIST analogue reached x*_chart on 6/8; no prior on the CIFAR landing rate.
2. **Drift, target 2 (`--adapt 1 2`).** Rule A: `rho_full` stays at the floor wherever `rank B_T = N'`, and
   `N' = 8 + drift_rank` grows ≈ 8 per step through the GELU (MNIST smoke: N' = 150 at T = 20), so
   `rank_C_full = 64 − N'` falls below k = 32 once N' > 32, i.e. by T ≈ 4–5 at lr 0.01, and `B_T` (1000 × 64) runs out
   of rank at N' > 64 → `contaminated` for T ≥ 20 at every lr and r_lower. Prediction: at T = 1 all three r_lower arms
   behave as the control (no drift yet: reps[0] = H^0); at T = 5 the outcome depends on `rank_C_full` vs k, not on δ;
   at T ≥ 20 the full-certificate solve is vacuous (rank < k → aliases / no isolated x*_chart) and the truncated
   certificate carries `K_l · δ_perp` with K_l measured per net (no band). r_lower orders the drift magnitude
   (δ_perp grows with r_lower) but NOT the rank loss, which is set by T. A landing rate that tracks δ while
   `rank_C_full ≥ k` refutes Rule A. Under drift, expect the truncated certificate to out-recover the full one
   wherever `rank_C_full ≈ k` (LESSONS 2026-09-18 "giving up exactness to keep rank").
3. **Drift, target 3 (`--adapt 2 3`, the head).** `rank B_T ≤ m − 1 = 10 < N + drift_rank` for T > 1 →
   **pre-registered contaminated** at every T > 1 (as the strong MNIST head; the harness prints the line). At T = 1
   it equals the control.
4. **Wrong-release controls.** 0 `recovered`, 0 tier-1 landings on the true eight at every T; tier 2 at chance
   (top-1 ≈ 1/100 per image; the "found (nearest start)" arm is oracle-selected and may score above chance on SSIM —
   read the attacker arm). A recovery here is a harness failure, not a result. **Amended after smoke 369540, before
   the full submit:** the ladder's floor assertion `certificate_nondegenerate` (min start objective ≫ 1e4 × the
   release's own floor) holds for RAW privates only. With ON-CHART privates the release's own eight are in the chart,
   so random starts reach the floor by landing on THOSE images (smoke: 5/5 starts on 3/8 of the wrong eight, 0/5 on
   the true eight) — that is the certificate doing its job on the release it was given, not a degeneracy. The row
   now records `landed_on_release_train` / `images_of_release_train_found`, sets `certificate_nondegenerate = None`
   on on-chart cells, and issues the verdict `control (wrong release): not recovered` (never `chart-limited`, since
   x*_chart from the truth's coordinates rolls to a zero of another release). Predictions: raw cells
   `certificate_nondegenerate = True` (smoke: min objective 1.6e-3 vs floor 2.5e-28); on-chart cells
   `landed_on_release_train > 0` at zero drift and `landed = 0` everywhere.
5. **Over-trained vs 60-epoch base.** No pre-registered difference in the certificate arithmetic (the drift and rank
   laws do not read the base's margins); the recording strength (`B_T_fro`, softmax residual) may differ — recorded,
   not gated on.

Three outcomes per cell (never merged): `recovered` / `chart-limited`, `optimisation failure (residual not zero)`,
`alias (residual zero, wrong image)`; `contaminated` is the row condition. Three seeds before any number enters the
ledger; join a measured rank to its own row's `expect_rank_C`.

## Smoke read-out — jobs 369540 (A40, first pass) and 369553 (H100, after the wrong-release relabel)

Rows `results/multilayer_cert/drift_cert_cifar_smoke_{369540,369553}.jsonl` (7 / 8 rows; script sha `c409d1467c26` /
`5e91f3956c09`, git `fa4763f-dirty`), tensors `results/multilayer_cert/drift_cert/cifar10_mlp*_{job}.pth`, colour grids
`figures/multilayer_cert/drift_cert/cifar10_mlp*_{job}.png` (rendered RGB, checked). Loader gate: over-trained 57.25 %,
60-epoch 59.05 % on 2000 CIFAR-10 test images (checkpoints record 57.31 / 57.90 %) — PASS. Join key
`private_join_idx = [23, 32, 89, 56, 41, 26, 34, 61]`, wrong-release `[64, 22, 18, 11, 4, 86, 59, 94]` (seed 1; the
torch-Generator eight of ladder_cell / cifar_newclass). PCA-32 explained variance 0.751; chart error of the raw
motorcycles 0.25–0.30. 20 starts, on-chart privates, both certificates. Wall time 2.5 min / 1 min per job.

| cell | adapt→target | T | N' | rank B_T | rank C (expect) | rho_full | rho_trunc | landed / reached x*_chart | tier-2 attacker top-1 (ssim/feat) | verdicts |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 2→2 | 1 | 8 | 8 | 56 (56) | 4.2e-16 | 4.2e-16 | 19/20, 7/8 images (369553 on H100: 5/8) | 7/7 | recovered 7, opt-failure 1 |
| control | 2→2 | 20 | 8 | 8 | 56 (56) | 9.8e-16 | 9.8e-16 | 19/20, 7/8 (H100: 20/20, 6/8) | 7/7 | recovered 7, opt-failure 1 |
| drift r16 | 1,2→2 | 1 | 8 | 8 | 56 (56) | 5.8e-16 | 5.8e-16 | 16/20, 7/8 | 7/8 | recovered 7 |
| drift r16 | 1,2→2 | 20 | **160** | 64 | **0** (0) | 6.8e-2 | 1.2e-3 | full 0/20; trunc reached x*_chart 19/20 (8/8), eps_land 2.5e-2, err_opt_truth 0.02–0.075 | full 3/1; trunc 8/8 | contaminated (row flag); trunc would relabel chart-limited 8 |
| head | 2,3→3 | 20 | 160 | 10 | 54 (0) | 2.0e-3 | 3.2e-3 | no solve | – | contaminated, as pre-registered |
| 60-epoch loader | 2→2 | 1 | 8 | 8 | 56 (56) | 4.5e-16 | – | no solve | – | – |
| wrong-release, onchart | 2→2 | 20 | 8 | 8 | 56 | 8.2e-16 | – | true eight 0/5; on the release's own images 5/5 (3/8 of them) | 2/0 (found arm), attacker 2/0 | control: not recovered 8 |
| wrong-release, raw | 2→2 | 20 | 8 | 8 | 56 | 1.1e-15 | – | true eight 0/5; own images 0/5; min objective 1.6e-3 vs floor 2.5e-28 → NON-DEGENERATE | 0/0 | control: not recovered 8 |

Observations only:
- The zero-drift control on the trained CIFAR MLP sits at the FP64 floor (rho ≤ 1e-15, per-image residual at the
  truth ≤ 2e-14, rank C = 56 = r − N with a clean gap) and lands 19–20 of 20 starts on 5–7 of the 8 on-chart privates
  with tier-2 top-1 on every landed image: the head-adapter `cifar_newclass` result reproduces on a HIDDEN layer with
  a frozen input.
- With layer 1 adapted (r_lower 16, lr 0.01), 20 steps put layer 2's input span at N' = 160 = 8·T (δ_⊥ 0.154, i.e.
  essentially all of the drift is orthogonal, exactly the MNIST smoke's pattern), `B_T` (1000 × 64) saturates at rank
  64 = r and the full certificate is annihilated (rank 0, rho 6.8e-2): contaminated, as Rule A predicts once N' > r.
  The truncated certificate (top-8 directions) still drives 19/20 starts to within 2–7.5 % of the truth with tier-2
  8/8 on the attacker's own candidates — the "give up exactness to keep rank" reading, on a trained backbone.
- Same seed, same starts, different GPU (A40 vs H100): 19/20 landings both times but on 7 vs 5 distinct images —
  the LM's basin choice is hardware-sensitive at the 1e-16 level; landing COUNTS per image are not reproducible across
  GPU models, only the floor / rank / rho quantities are. Three seeds remain the unit.
- The head-target drift cell measures rank C = 54 against `expect_rank_C = 0`: `expect_rank_C = max(0, min(r, n_in) −
  N')` assumes rank B_T = N', but the 11-row head caps rank B_T at 10, so C keeps 64 − 10 = 54 directions that annihilate
  nothing (rho 2e-3). Read `contaminated`, not `expect_rank_C`, on head rows.

## Full-stage job ids — submitted 2026-09-19 03:27 after the smoke read-out (`bash scripts/run_drift_cert_wexac.sh submit cifar`, CERT=both STARTS=200)

long-gpu, -W 24:00 (wrong-release -W 8:00); 11 started immediately, 21 pending behind the per-user GPU cap (55 running
under `yoado` at submit time, all lanes). Job name → id (`bjobs -w`; outputs `results/multilayer_cert/drift_cert_cifar_*_<id>.jsonl`):

| private | seed | overtrained t2 | overtrained t3 | newclass t2 | newclass t3 |
|---|---|---|---|---|---|
| raw | 1 | 369565 | 369566 | 369567 | 369568 |
| raw | 2 | 369569 | 369570 | 369571 | 369572 |
| raw | 3 | 369573 | 369574 | 369575 | 369576 |
| onchart | 1 | 369581 | 369582 | 369583 | 369584 |
| onchart | 2 | 369585 | 369586 | 369587 | 369588 |
| onchart | 3 | 369589 | 369590 | 369591 | 369592 |

Wrong-release controls (seed 1, control + drift r_lower 16, T grid, lr 0.01): raw — overtrained t2 369577, t3 369578,
newclass t2 369579, t3 369580; onchart — overtrained t2 369593, t3 369594, newclass t2 369595, t3 369596.

Expected wall time per 60-cell job ≈ 4–6 h (0.7–1 s per LM start × 200 starts × 2 certificates × 60 cells + 8
oracle starts per cell); the head-target jobs are faster (C = 0 or contaminated cells converge in one LM step).
