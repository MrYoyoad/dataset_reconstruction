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
(last all-8 landing – first failure), EMNIST-a **0.0069–0.0139** (WP5, jobs 355845–355858: recovery at 0.0069, gone at 0.0139). Every row is compared at both ends.

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

### INTERIM (2026-09-18 02:10) — closed-form rows complete for all three sets; local chart partial

First submission 356034–356051 (plain `num=1`): **9 of 18 died of CUDA OOM on shared A40s** (356037, 356039, 356040,
356042, 356043, 356044, 356047, 356050, 356051 — other users' processes held 13–20 GB of 44 GB; the fit needs ~12 GB).
Their fragments held only the closed-form rows (identical in every job) and were deleted; the nine cells were
resubmitted with `gmem=20G` as 356090 (motorcycle truth_nn K256), 356091 (motorcycle truth_latent K256), 356092
(keyboard proxy_nn K64), 356093 (keyboard truth_nn K64), 356094 (keyboard truth_nn K256), 356095 (keyboard
truth_latent K64), 356096 (letters proxy_nn K256), 356097 (letters truth_latent K64), 356098 (letters truth_latent
K256). The nine survivors of the first submission (356034, 356035, 356036, 356038, 356041, 356045, 356046, 356048,
356049) are running.

#### 1. Autoencoding ceiling (reported first) — `‖D(E(x)) − x‖/‖x‖`, native space, median (range over 8)

| image set | floor ×2/×4/×8 | ceiling ×2 | ceiling ×4 | ceiling ×8 | bracket | ×8 vs low / high | verdict |
|---|---|---|---|---|---|---|---|
| mlp_motorcycle | 0 / 0 / 0 | 0.2145 (0.126–0.274) | 0.0777 (0.057–0.105) | **0.0554** (0.019–0.080) | 0.0124–0.0186 | 4.5× / 3.0× | **CEILING-BOUND** (0 of 8 below the high end) |
| cnn_keyboard | 0 / 0 / 0 | 0.1812 (0.058–0.292) | 0.0456 (0.022–0.087) | **0.0237** (0.009–0.138) | 0.0045–0.0090 | 5.3× / 2.6× | **CEILING-BOUND** (1 of 8 at 0.0089, i.e. at the high end; median 2.6× above it) |
| mnist_letter_a | 0 / 0 / 0 | 0.1309 (0.103–0.297) | 0.0552 (0.042–0.122) | **0.0213** (0.015–0.048) | 0.0069–0.0139 | 3.1× / 1.5× | **CEILING-BOUND** (0 of 8 below the high end) |

Pre-registered rule: the ceiling exceeds the bracket's high end on every set, so no chart built on `D(E(·))` passes;
the result is about the decoder. Caveat recorded, not a verdict change: the oracle-anchor Adam fit (below) goes UNDER
`D(E(x))` — the encoder's latent is not the decoder's best latent — to 0.0430 at k = 32 on motorcycle, still 2.3× the
high end; the decoder's true floor is bounded above by those `truth_latent` rows, not by `D(E(x))`.

#### 2 + 4. Pixel PCA (C7 convention) vs global latent PCA (decoded projection), median over the ladder's 8 images

| image set | k | pixel PCA | latent PCA ×2 | ×4 | ×8 | best vs bracket low / high |
|---|---|---|---|---|---|---|
| mlp_motorcycle | 16 | 0.3413 | 0.3982 | 0.3642 | 0.3492 | 27.5× / 18.3× |
| | 32 | 0.2983 | 0.3729 | 0.3410 | 0.3198 | 24.1× / 16.0× |
| | 66 | 0.2549 | 0.3330 | 0.3201 | 0.3062 | 20.6× / 13.7× |
| | 128 | 0.2180 | 0.2967 | 0.3018 | 0.2826 | 17.6× / 11.7× |
| cnn_keyboard | 16 | 0.2858 | 0.2993 | 0.2950 | 0.2924 | 63.5× / 31.8× |
| | 32 | 0.2539 | 0.2957 | 0.2937 | 0.2804 | 56.4× / 28.2× |
| | 66 | 0.2009 | 0.2757 | 0.2748 | 0.2394 | 44.6× / 22.3× |
| | 128 | 0.1681 | 0.2458 | 0.2465 | 0.2218 | 37.4× / 18.7× |
| mnist_letter_a | 16 | 0.3155 | 0.3236 | 0.3466 | 0.3304 | 45.7× / 22.7× |
| | 32 | 0.2346 | 0.2948 | 0.2769 | 0.2696 | 34.0× / 16.9× |
| | 66 | 0.1659 | 0.1874 | 0.2123 | 0.2080 | 24.0× / 11.9× |
| | 128 | 0.1101 | 0.1375 | 0.1677 | 0.1835 | 16.0× / 7.9× |

Global latent PCA is worse than pixel PCA at every k on every set. C7 cross-check (two_walls index set, mean):
motorcycle 0.2456 / 0.2151 / 0.1845 / 0.1566, keyboard 0.2840 / 0.2573 / 0.2058 / 0.1704 at k = 16 / 32 / 66 / 128 —
C7's stored rows to 4 decimals. On the ladder's own images the motorcycle numbers are higher (0.2549 at k = 66, not
0.1845): C7 and the ladder used different index sets.

#### 3. Local chart (Adam 400 steps, 3 restarts, FP32, ×8 scale) — rows landed so far

| image set | K | k (k_eff) | anchor | avail. | w = 0 | best-of-restarts (range) | traj. min | solver flag | vs low / high |
|---|---|---|---|---|---|---|---|---|---|
| mlp_motorcycle | 64 | 16 | truth_latent | NO | 0.0554 (= ceiling, diff 0.0) | 0.0454 (0.019–0.067) | 0.0454 | ok | 3.7× / 2.4× |
| mlp_motorcycle | 64 | 32 | truth_latent | NO | 0.0554 (= ceiling) | 0.0430 (0.018–0.061) | 0.0430 | ok | 3.5× / 2.3× |
| mlp_motorcycle | 64 | 16 | truth_nn | NO | 0.4496 | 0.3513 (0.249–0.514) | 0.3513 | ok | 28.3× / 18.9× |
| mnist_letter_a | 64 | 16 | proxy_nn | yes | 0.4044 | 0.2555 (0.218–0.456) | 0.2555 | ok | 37.0× / 18.4× |
| mnist_letter_a | 64 | 32 | proxy_nn | yes | 0.3716 | 0.1699 (0.140–0.378) | 0.1699 | ok | 24.6× / 12.2× |
| mnist_letter_a | 256 | 16 | truth_nn | NO | 0.4171 | 0.2999 (0.264–0.566) | 0.2999 | ok | 43.5× / 21.6× |

No solver-failure or solver-limited row so far (every fit ends at its trajectory minimum, below its w = 0 value). The
attacker-available local chart on letters (0.2555 / 0.1699 at k = 16 / 32) is BETTER than pixel PCA at the same k
(0.3155 / 0.2346) — the first chart to beat PCA in this repo — but still 12× the bracket's high end at k = 32.
On motorcycle (smoke 355910, K = 64, k = 16) it was worse than pixel PCA (0.3585 vs 0.3413). Keyboard local rows: none yet.

#### Verdict per image set (INTERIM)

| image set | ceiling vs bracket | best attacker-available arm at k ≤ 66 so far | outcome |
|---|---|---|---|
| mlp_motorcycle | 0.0554 vs 0.0124–0.0186 | pixel PCA 0.2549 at k = 66 (local chart rows pending) | **CEILING-BOUND** |
| cnn_keyboard | 0.0237 vs 0.0045–0.0090 | pixel PCA 0.2009 at k = 66 (local chart rows pending) | **CEILING-BOUND** |
| mnist_letter_a | 0.0213 vs 0.0069–0.0139 | local chart proxy_nn K = 64 k = 32: 0.1699 | **CEILING-BOUND** |

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
