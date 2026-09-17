# WP3 — Pretrained decoder as chart: fidelity check (no inversion)

Plan: `notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md` §WP3 + Audit 2026-09-18. Code:
`experiments/decoder_chart/fidelity.py`, runner `scripts/run_decoder_chart_wexac.sh`. Rows:
`results/decoder_chart/fidelity_<jobid>.jsonl`; tensors `fidelity_<set>_<jobid>.pth`; grids `figures/decoder_chart/`.
All numbers here are **provisional** until the row is read by a second session. Nothing below reads a release: the
base checkpoints only fix WHICH eight images are private (images-only), and the gates are the oracle ladder's.

**Question.** Can `stabilityai/sd-vae-ft-mse` (KL-VAE decoder `D`, latent `4 × s/8 × s/8`), used as a local chart
`G(w) = D(a + U w)`, contain the eight private images to within the landing gate at a width `k ≤ 66`
(identifiability cap on the CIFAR releases, `m + r − N − 1 = 66`)? Every chart measured so far was PCA-like and
lost to the gate by 15× / 46× at `k = 66` (e1b C7).

**Setup.** diffusers 0.32.2 in `.conda/extra_pkgs` (NOT the shared `rec` env; PYTHONPATH in the job script only),
torch 2.4.1+cu121, FP32, `local_files_only=True`, `HF_HUB_OFFLINE=1`. Images: the ladder's eight motorcycles (MLP)
and keyboards (CNN) — `torch.Generator().manual_seed(seed+7)`, `randperm` over the CIFAR-100 class TEST split,
seed 1 — and the letters cell's eight EMNIST `a` (same generator over the EMNIST `a` test split). Public pool =
the class TRAIN split. Resize path in every decoder arm: up = bilinear (`align_corners=False`), grey → 3 channels,
`[0,1]→[−1,1]`; down = area (adaptive average pooling) to native, channels averaged for grey. Error =
`‖x̂ − x‖/‖x‖` in NATIVE space (3×32×32 / 1×28×28), unclamped (clamped-to-[0,1] value recorded beside it),
median + range over the 8.

**Gates are brackets** (audit; ledger §0.1, A22): MLP-motorcycle **0.0124–0.0186**, CNN-keyboard **0.0045–0.0090**
(last all-8 landing – first failure). Every row is compared at both ends. MNIST: no gate yet (WP5) — no shortfall quoted.

## Pre-registered outcomes (fixed before the first row)

- **CEILING-BOUND**: the autoencoding ceiling `‖D(E(x)) − x‖/‖x‖` (measurement 1, reported FIRST) is itself above
  the gate bracket's high end at every scale → no chart built on this decoder can pass; the result is about the
  decoder, not the chart.
- **PASS**: some attacker-available arm (global latent PCA, or local chart with the `proxy_nn` anchor) at `k ≤ 66`
  has median error ≤ the bracket's LOW end. (Median inside the bracket = reported as "inside", not a pass.)
- **FAIL**: no attacker-available arm is below the bracket's HIGH end at any `k ≤ 128`.
- Solver rule (audit item 3): the local-chart number is solver-bounded. At the oracle anchor `a = E(x)`, `w = 0`
  must reproduce the ceiling (same resize path, `w0_reproduces_ceiling`); a fit whose best-of-restarts FINAL error
  ends above the ceiling is a **solver failure** row, never a chart number. Any anchor whose fit ends above its own
  `w = 0` value is **solver-limited**. Best-of-restarts and `w = 0` are reported side by side.
- Not attacker-available (controls only): `truth_nn` (neighbours of `E(x)`), `truth_latent` (`a = E(x)`).

## Smoke (job 355910, `mlp_motorcycle`, k 16, K 64, 200 Adam steps, factors x2/x4/x8, up = nearest)

Earlier smoke attempts: 355876 (bilinear up: resize floor alone 0.061, 5x the gate -> replaced by exact replication;
then a latent-shape bug), 355897 (FP64 default dtype leaked from imported modules), 355900 (shared decode graph freed
by the first chunk's backward). All three are harness bugs, no science read from them.

| measurement | scale | k / K | anchor | median | range | vs bracket 0.0124-0.0186 |
|---|---|---|---|---|---|---|
| resize floor | 64 / 128 / 256 | | | 0.0000 | 0.0000-0.0000 | exact (replication + block mean) |
| AE ceiling D(E(x)) | 64 | | | 0.2145 | 0.1263-0.2737 | 17.3x low / 11.5x high |
| AE ceiling D(E(x)) | 128 | | | 0.0777 | 0.0566-0.1052 | 6.3x / 4.2x |
| AE ceiling D(E(x)) | 256 | | | **0.0554** | 0.0194-0.0803 | **4.5x / 3.0x -> CEILING-BOUND on this set** |
| pixel PCA (C7 convention) | native | 16 | | 0.3413 | 0.2343-0.4930 | 27.5x / 18.3x |
| global latent PCA | 64 / 128 / 256 | 16 | | 0.3982 / 0.3642 / 0.3492 | | worse than pixel PCA at the same k |
| local chart | 256 | 16 / 64 | proxy_nn (attacker) | 0.3585 (w=0: 0.4487) | 0.2459-0.4850 | 28.9x / 19.3x; worse than pixel PCA |
| local chart | 256 | 16 / 64 | truth_nn (NOT available) | 0.3513 (w=0: 0.4496) | 0.2488-0.5140 | 28.3x / 18.9x |
| local chart | 256 | 16 / 64 | truth_latent (NOT available) | w=0 = 0.05538 = ceiling (diff 0.0); fit 0.0454 | 0.0188-0.0670 | solver check OK; 3.7x / 2.4x |

Timing: 4.18 s per Adam step (8 images x 3 restarts, 256^2, shared A40) -> 200 steps = 14 min per combination; the
trace is flat from step ~80. C7 cross-check: the same pixel-PCA computation on the two_walls.py index set
`[80, 84, 33, 81, 93, 17, 36, 82]` gives mean 0.2456 at k = 16, identical to C7's stored row -- the convention matches.
NOTE: C7's index set is NOT the ladder's (`np.random.RandomState(1)` vs `torch.randperm(seed+7)`); on the ladder's
eight motorcycles `[23, 32, 89, 56, 41, 26, 34, 61]`, the images the gates were measured on, pixel PCA at k = 16
is 0.341 median, not 0.246.

## Full run — 18 jobs, one per (image set, anchor, K), submitted 2026-09-18 01:36, 400 Adam steps, factor x8 local chart

| image set | proxy_nn K64 / K256 | truth_nn K64 / K256 | truth_latent K64 / K256 |
|---|---|---|---|
| mlp_motorcycle | 356034 / 356035 | 356036 / 356037 | 356038 / 356039 |
| cnn_keyboard | 356040 / 356041 | 356042 / 356043 | 356044 / 356045 |
| mnist_letter_a | 356046 / 356047 | 356048 / 356049 | 356050 / 356051 |

Rows: `results/decoder_chart/fidelity_<anchor>_K<K>_<jobid>.jsonl` (measurements 0/1/2/4 are recomputed identically in
every job, so they appear 6x per image set — same numbers); tensors and grids carry the same tag. Command per job:
`python -u -m experiments.decoder_chart.fidelity --image-sets <set> --anchors <anchor> --Ks <K> --steps 400 --tag <anchor>_K<K>`.

### 1. Autoencoding ceiling (reported first) — `‖D(E(x)) − x‖/‖x‖`, native space

| image set | resize floor (64/128/256) | ceiling @64 | ceiling @128 | ceiling @256 | bracket | verdict |
|---|---|---|---|---|---|---|
| mlp_motorcycle | | | | | 0.0124–0.0186 | |
| cnn_keyboard | | | | | 0.0045–0.0090 | |
| mnist_letter_a | | | | | none (WP5) | — |

### 2. Global latent PCA chart (attacker-available; on-chart recovery convention) vs 4. pixel PCA at the same `k`

| image set | k | pixel PCA median (range) | latent PCA @64 | @128 | @256 | vs bracket |
|---|---|---|---|---|---|---|
| mlp_motorcycle | 16 / 32 / 66 / 128 | | | | | |
| cnn_keyboard | 16 / 32 / 66 / 128 | | | | | |
| mnist_letter_a | 16 / 32 / 66 / 128 | | | | | |

(Cross-check: the same pixel-PCA computation on the two_walls.py index set reproduces C7's 0.1845 / 0.2058 at k = 66? ___)

### 3. Local patch chart `min_w ‖D(a + U w) − x‖/‖x‖` (Adam, FP32, 2000 steps, 3 restarts; scale ___)

| image set | K | k (k_eff) | anchor | attacker-available | w = 0 median | best-of-restarts median (range) | solver flag | vs bracket |
|---|---|---|---|---|---|---|---|---|
| | 64 / 256 | 16 / 32 / 66 / 128 | proxy_nn | yes | | | | |
| | 64 / 256 | 16 / 32 / 66 / 128 | truth_nn | NO | | | | |
| | 64 / 256 | 16 / 32 / 66 / 128 | truth_latent | NO (ceiling check) | | | | |

### Verdict per image set

| image set | ceiling vs bracket | best attacker-available arm at k ≤ 66 | outcome (PASS / FAIL / CEILING-BOUND) |
|---|---|---|---|
| mlp_motorcycle | | | |
| cnn_keyboard | | | |
| mnist_letter_a | | no bracket: fidelity numbers only | — |

## Deviations from the plan as written

- **Adam steps 400, not 2000; jobs split by (anchor, K).** Smoke 355910 measured 4.18 s per step (8 images x 3
  restarts at 256^2 on a shared A40), so 2000 steps x 24 combinations = ~56 h per image set against a 6 h queue limit.
  The smoke trace is flat from step ~80 (0.3585 at steps 80..200) under the same cosine schedule shape; `err_trajmin`
  is recorded beside the final value so a non-monotone trajectory is visible. 3 restarts, FP32, lr 0.1 as planned.
- **Decoder input scale is a multiple of the native side, not 256 for all sets:** 64/128/256 for CIFAR (32 x 2/4/8),
  56/112/224 for EMNIST (28 x 2/4/8). Reason: an exact resize round trip (replication up, block mean down) needs an
  integer factor; bilinear-to-256 had a floor of 0.061 on its own (job 355876), 5x the gate, which would have been read
  as a decoder number. The VAE is fully convolutional; 224 gives a 28 x 28 latent grid.
- **Global latent PCA (measurement 2) is the decoded projection of E(x)** (on-chart recovery convention, mirroring pixel
  PCA), not an Adam fit; the Adam fit is measurement 3 only.
- **Three anchors, not two:** `proxy_nn` (attacker), `truth_nn` (neighbours of the truth; NOT available), `truth_latent`
  (a = E(x); NOT available, the audit's solver check). Neighbours are found in latent space for all three; the query
  point is the only difference.
- **k_eff = min(k, K−1):** with K = 64 the chart has at most 63 directions, so the k = 66 and k = 128 rows at K = 64 are
  the full affine span of the 64 neighbours; recorded as `k_eff` in the row.
- **Pixel PCA in FP32** (the whole arm is FP32 by plan); the C7 cross-check row reproduces C7's FP64 mean to 4 decimals.
- diffusers 0.36.0 (pip's default for this env) needs peft >= 0.17 and refuses to import beside rec's peft 0.7.1;
  **diffusers 0.32.2** is the version used, in `.conda/extra_pkgs` (no other package was needed there).
