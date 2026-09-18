# WP3 — Pretrained decoder as chart: fidelity check (no inversion)

Plan: `notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md` §WP3 + Audit 2026-09-18. Code:
`experiments/decoder_chart/fidelity.py`, runner `scripts/run_decoder_chart_wexac.sh`. Rows:
`results/decoder_chart/fidelity_<anchor>_K<K>_<jobid>.jsonl`; tensors `fidelity_<set>_<jobid>.pth`; grids `figures/decoder_chart/`.
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
the class TRAIN split. Resize path in every decoder arm: up = exact pixel replication at an integer factor
(×2/×4/×8 of the native side: 64/128/256 CIFAR, 56/112/224 EMNIST), grey → 3 channels, `[0,1]→[−1,1]`; down = block
mean (area) to native, channels averaged for grey. `down(up(x)) = x` exactly, so the resize floor is 0. Error =
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

### FINAL (2026-09-18) — all 18 cells complete, 72 local-chart rows

Submission history: 356034–356051 (plain `num=1`) lost 12 of 18 cells to CUDA OOM on shared A40s (co-tenants at
13–25 GB of 44 GB; our fit needs ~16–24 GB at chunk 8). Resubmitted with `gmem=20G` (356090–356098) and, where that
still lost, with `--chunk 4` (356101–356103, 356113, 356115, 356116, 356472). Every cell that produced no local rows
had its fragment deleted, so the rows below come from one successful run each. Completing job ids: motorcycle
356101/356102 (proxy_nn K64/K256), 356036/356090 (truth_nn), 356038/356091 (truth_latent); keyboard 356113/356041,
356115/356116, 356095/356103; letters 356046/356096, 356472/356049, 356097/356098.

#### 1. Autoencoding ceiling (reported first) — `‖D(E(x)) − x‖/‖x‖`, native space, median (range over 8)

| image set | floor ×2/×4/×8 | ceiling ×2 | ceiling ×4 | ceiling ×8 | bracket | ×8 vs low / high | verdict |
|---|---|---|---|---|---|---|---|
| mlp_motorcycle | 0 / 0 / 0 | 0.2145 (0.126–0.274) | 0.0777 (0.057–0.105) | **0.0554** (0.019–0.080) | 0.0124–0.0186 | 4.5× / 3.0× | **CEILING-BOUND** (0 of 8 below the high end) |
| cnn_keyboard | 0 / 0 / 0 | 0.1812 (0.058–0.292) | 0.0456 (0.022–0.087) | **0.0237** (0.009–0.138) | 0.0045–0.0090 | 5.3× / 2.6× | **CEILING-BOUND** (1 of 8 at 0.0089, i.e. at the high end; median 2.6× above it) |
| mnist_letter_a | 0 / 0 / 0 | 0.1309 (0.103–0.297) | 0.0552 (0.042–0.122) | **0.0213** (0.015–0.048) | 0.0069–0.0139 | 3.1× / 1.5× | **CEILING-BOUND** (0 of 8 below the high end) |

Pre-registered rule: the ceiling exceeds the bracket's high end on every set, so no chart built on `D(E(·))` passes;
the result is about the decoder. Caveat recorded, not a verdict change: the oracle-anchor Adam fit (below) goes UNDER
`D(E(x))` — the encoder's latent is not the decoder's best latent — to 0.0430 at k = 32 on motorcycle, still 2.3× the
high end; the decoder's true floor is bounded above by those `truth_latent` rows, not by `D(E(x))`. **On letters this
matters**: the oracle-anchor optimum at K = 256, k = 128 is 0.0092 (range 0.0079–0.0174), INSIDE the 0.0069–0.0139
bracket (jobs 356097/356098). So on the grey 28 × 28 set the decoder itself is NOT the obstruction — its latent
family does contain the letters to bracket precision — while `D(E(x))` (0.0213) is; the CEILING-BOUND verdict is
therefore a statement about the pre-registered ceiling definition, and the binding constraints on letters are the
anchor (that row uses `a = E(x)`, not attacker-available; the attacker's `proxy_nn` anchor reaches 0.0549 at the same
K, k) and the width (k = 128 is above the cap). On the two CIFAR sets the oracle-anchor optimum stays above the
bracket at every k (motorcycle 0.0371 at K = 256, k = 128; keyboard 0.0207 — vs high ends 0.0186 / 0.0090), so there the decoder
is the obstruction as pre-registered.

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

#### 3. Local chart `min_w ‖D(a + U w) − x‖/‖x‖` — Adam 400 steps, 3 restarts, FP32, ×8 scale — ALL 72 rows

| image set | K | k (k_eff) | anchor | attacker-avail. | w = 0 | best-of-restarts (range) | traj. min | solver | vs low / high |
|---|---|---|---|---|---|---|---|---|---|
| mlp_motorcycle | 64 | 16 | proxy_nn | yes | 0.4487 | 0.3585 (0.246–0.485) | 0.3585 | ok | 28.9× / 19.3× |
| mlp_motorcycle | 64 | 32 | proxy_nn | yes | 0.4484 | 0.3217 (0.217–0.470) | 0.3217 | ok | 25.9× / 17.3× |
| mlp_motorcycle | 64 | 66 (64) | proxy_nn | yes | 0.4459 | 0.2941 (0.192–0.415) | 0.2941 | ok | 23.7× / 15.8× |
| mlp_motorcycle | 64 | 128 (64) | proxy_nn | yes | 0.4533 | 0.2951 (0.192–0.423) | 0.2951 | ok | 23.8× / 15.9× |
| mlp_motorcycle | 256 | 16 | proxy_nn | yes | 0.4711 | 0.3321 (0.241–0.427) | 0.3321 | ok | 26.8× / 17.9× |
| mlp_motorcycle | 256 | 32 | proxy_nn | yes | 0.4707 | 0.2833 (0.215–0.412) | 0.2833 | ok | 22.8× / 15.2× |
| mlp_motorcycle | 256 | 66 | proxy_nn | yes | 0.4706 | 0.2550 (0.184–0.358) | 0.2550 | ok | 20.6× / 13.7× |
| mlp_motorcycle | 256 | 128 | proxy_nn | yes | 0.4713 | 0.2324 (0.156–0.285) | 0.2324 | ok | 18.7× / 12.5× |
| mlp_motorcycle | 64 | 16 | truth_nn | NO | 0.4496 | 0.3513 (0.249–0.514) | 0.3513 | ok | 28.3× / 18.9× |
| mlp_motorcycle | 64 | 32 | truth_nn | NO | 0.4496 | 0.3123 (0.215–0.472) | 0.3123 | ok | 25.2× / 16.8× |
| mlp_motorcycle | 64 | 66 (64) | truth_nn | NO | 0.4496 | 0.2916 (0.191–0.431) | 0.2916 | ok | 23.5× / 15.7× |
| mlp_motorcycle | 64 | 128 (64) | truth_nn | NO | 0.4496 | 0.2916 (0.191–0.431) | 0.2916 | ok | 23.5× / 15.7× |
| mlp_motorcycle | 256 | 16 | truth_nn | NO | 0.4718 | 0.3292 (0.239–0.429) | 0.3292 | ok | 26.6× / 17.7× |
| mlp_motorcycle | 256 | 32 | truth_nn | NO | 0.4718 | 0.2881 (0.215–0.405) | 0.2881 | ok | 23.2× / 15.5× |
| mlp_motorcycle | 256 | 66 | truth_nn | NO | 0.4718 | 0.2560 (0.181–0.363) | 0.2560 | ok | 20.6× / 13.8× |
| mlp_motorcycle | 256 | 128 | truth_nn | NO | 0.4718 | 0.2251 (0.159–0.286) | 0.2251 | ok | 18.2× / 12.1× |
| mlp_motorcycle | 64 | 16 | truth_latent | NO | 0.0554 (= ceiling) | 0.0454 (0.019–0.067) | 0.0454 | ok | 3.7× / 2.4× |
| mlp_motorcycle | 64 | 32 | truth_latent | NO | 0.0554 (= ceiling) | 0.0430 (0.018–0.061) | 0.0430 | ok | 3.5× / 2.3× |
| mlp_motorcycle | 64 | 66 (64) | truth_latent | NO | 0.0554 (= ceiling) | 0.0411 (0.017–0.054) | 0.0411 | ok | 3.3× / 2.2× |
| mlp_motorcycle | 64 | 128 (64) | truth_latent | NO | 0.0554 (= ceiling) | 0.0411 (0.017–0.054) | 0.0411 | ok | 3.3× / 2.2× |
| mlp_motorcycle | 256 | 16 | truth_latent | NO | 0.0554 (= ceiling) | 0.0471 (0.019–0.067) | 0.0471 | ok | 3.8× / 2.5× |
| mlp_motorcycle | 256 | 32 | truth_latent | NO | 0.0554 (= ceiling) | 0.0431 (0.018–0.063) | 0.0431 | ok | 3.5× / 2.3× |
| mlp_motorcycle | 256 | 66 | truth_latent | NO | 0.0554 (= ceiling) | 0.0412 (0.017–0.053) | 0.0412 | ok | 3.3× / 2.2× |
| mlp_motorcycle | 256 | 128 | truth_latent | NO | 0.0554 (= ceiling) | 0.0371 (0.016–0.045) | 0.0371 | ok | 3.0× / 2.0× |
| cnn_keyboard | 64 | 16 | proxy_nn | yes | 0.2963 | 0.2407 (0.098–0.431) | 0.2407 | ok | 53.5× / 26.7× |
| cnn_keyboard | 64 | 32 | proxy_nn | yes | 0.2964 | 0.2221 (0.088–0.403) | 0.2221 | ok | 49.4× / 24.7× |
| cnn_keyboard | 64 | 66 (64) | proxy_nn | yes | 0.2983 | 0.2101 (0.079–0.346) | 0.2101 | ok | 46.7× / 23.3× |
| cnn_keyboard | 64 | 128 (64) | proxy_nn | yes | 0.2977 | 0.2088 (0.080–0.348) | 0.2088 | ok | 46.4× / 23.2× |
| cnn_keyboard | 256 | 16 | proxy_nn | yes | 0.3294 | 0.2882 (0.095–0.402) | 0.2882 | ok | 64.0× / 32.0× |
| cnn_keyboard | 256 | 32 | proxy_nn | yes | 0.3308 | 0.2512 (0.088–0.357) | 0.2512 | ok | 55.8× / 27.9× |
| cnn_keyboard | 256 | 66 | proxy_nn | yes | 0.3309 | 0.2125 (0.079–0.319) | 0.2125 | ok | 47.2× / 23.6× |
| cnn_keyboard | 256 | 128 | proxy_nn | yes | 0.3326 | 0.1846 (0.066–0.270) | 0.1846 | ok | 41.0× / 20.5× |
| cnn_keyboard | 64 | 16 | truth_nn | NO | 0.2989 | 0.2384 (0.098–0.402) | 0.2384 | ok | 53.0× / 26.5× |
| cnn_keyboard | 64 | 32 | truth_nn | NO | 0.2989 | 0.2250 (0.087–0.368) | 0.2250 | ok | 50.0× / 25.0× |
| cnn_keyboard | 64 | 66 (64) | truth_nn | NO | 0.2989 | 0.2086 (0.080–0.347) | 0.2086 | ok | 46.4× / 23.2× |
| cnn_keyboard | 64 | 128 (64) | truth_nn | NO | 0.2989 | 0.2086 (0.080–0.347) | 0.2086 | ok | 46.4× / 23.2× |
| cnn_keyboard | 256 | 16 | truth_nn | NO | 0.3335 | 0.2893 (0.094–0.401) | 0.2893 | ok | 64.3× / 32.1× |
| cnn_keyboard | 256 | 32 | truth_nn | NO | 0.3335 | 0.2541 (0.088–0.359) | 0.2541 | ok | 56.5× / 28.2× |
| cnn_keyboard | 256 | 66 | truth_nn | NO | 0.3335 | 0.2132 (0.074–0.326) | 0.2132 | ok | 47.4× / 23.7× |
| cnn_keyboard | 256 | 128 | truth_nn | NO | 0.3335 | 0.1843 (0.066–0.274) | 0.1843 | ok | 40.9× / 20.5× |
| cnn_keyboard | 64 | 16 | truth_latent | NO | 0.0237 (= ceiling) | 0.0221 (0.008–0.063) | 0.0221 | ok | 4.9× / 2.5× |
| cnn_keyboard | 64 | 32 | truth_latent | NO | 0.0237 (= ceiling) | 0.0219 (0.008–0.060) | 0.0219 | ok | 4.9× / 2.4× |
| cnn_keyboard | 64 | 66 (64) | truth_latent | NO | 0.0237 (= ceiling) | 0.0214 (0.008–0.051) | 0.0214 | ok | 4.8× / 2.4× |
| cnn_keyboard | 64 | 128 (64) | truth_latent | NO | 0.0237 (= ceiling) | 0.0214 (0.008–0.051) | 0.0214 | ok | 4.8× / 2.4× |
| cnn_keyboard | 256 | 16 | truth_latent | NO | 0.0237 (= ceiling) | 0.0221 (0.008–0.076) | 0.0221 | ok | 4.9× / 2.5× |
| cnn_keyboard | 256 | 32 | truth_latent | NO | 0.0237 (= ceiling) | 0.0219 (0.008–0.068) | 0.0219 | ok | 4.9× / 2.4× |
| cnn_keyboard | 256 | 66 | truth_latent | NO | 0.0237 (= ceiling) | 0.0216 (0.008–0.058) | 0.0216 | ok | 4.8× / 2.4× |
| cnn_keyboard | 256 | 128 | truth_latent | NO | 0.0237 (= ceiling) | 0.0207 (0.008–0.047) | 0.0207 | ok | 4.6× / 2.3× |
| mnist_letter_a | 64 | 16 | proxy_nn | yes | 0.4044 | 0.2555 (0.218–0.456) | 0.2555 | ok | 37.0× / 18.4× |
| mnist_letter_a | 64 | 32 | proxy_nn | yes | 0.3716 | 0.1699 (0.140–0.378) | 0.1699 | ok | 24.6× / 12.2× |
| mnist_letter_a | 64 | 66 (64) | proxy_nn | yes | 0.3700 | 0.1141 (0.090–0.229) | 0.1141 | ok | 16.5× / 8.2× |
| mnist_letter_a | 64 | 128 (64) | proxy_nn | yes | 0.3684 | 0.1202 (0.086–0.233) | 0.1202 | ok | 17.4× / 8.7× |
| mnist_letter_a | 256 | 16 | proxy_nn | yes | 0.4276 | 0.2519 (0.222–0.505) | 0.2519 | ok | 36.5× / 18.1× |
| mnist_letter_a | 256 | 32 | proxy_nn | yes | 0.4194 | 0.1459 (0.137–0.370) | 0.1459 | ok | 21.2× / 10.5× |
| mnist_letter_a | 256 | 66 | proxy_nn | yes | 0.4171 | 0.1022 (0.086–0.221) | 0.1022 | ok | 14.8× / 7.4× |
| mnist_letter_a | 256 | 128 | proxy_nn | yes | 0.4138 | 0.0549 (0.043–0.117) | 0.0549 | ok | 8.0× / 3.9× |
| mnist_letter_a | 64 | 16 | truth_nn | NO | 0.3597 | 0.2630 (0.199–0.494) | 0.2630 | ok | 38.1× / 18.9× |
| mnist_letter_a | 64 | 32 | truth_nn | NO | 0.3597 | 0.1860 (0.140–0.410) | 0.1860 | ok | 27.0× / 13.4× |
| mnist_letter_a | 64 | 66 (64) | truth_nn | NO | 0.3597 | 0.1167 (0.096–0.262) | 0.1167 | ok | 16.9× / 8.4× |
| mnist_letter_a | 64 | 128 (64) | truth_nn | NO | 0.3597 | 0.1167 (0.096–0.262) | 0.1167 | ok | 16.9× / 8.4× |
| mnist_letter_a | 256 | 16 | truth_nn | NO | 0.4171 | 0.2999 (0.264–0.566) | 0.2999 | ok | 43.5× / 21.6× |
| mnist_letter_a | 256 | 32 | truth_nn | NO | 0.4171 | 0.1660 (0.144–0.403) | 0.1660 | ok | 24.1× / 11.9× |
| mnist_letter_a | 256 | 66 | truth_nn | NO | 0.4171 | 0.0999 (0.077–0.245) | 0.0999 | ok | 14.5× / 7.2× |
| mnist_letter_a | 256 | 128 | truth_nn | NO | 0.4171 | 0.0539 (0.046–0.104) | 0.0539 | ok | 7.8× / 3.9× |
| mnist_letter_a | 64 | 16 | truth_latent | NO | 0.0213 (= ceiling) | 0.0191 (0.014–0.040) | 0.0191 | ok | 2.8× / 1.4× |
| mnist_letter_a | 64 | 32 | truth_latent | NO | 0.0213 (= ceiling) | 0.0179 (0.012–0.029) | 0.0179 | ok | 2.6× / 1.3× |
| mnist_letter_a | 64 | 66 (64) | truth_latent | NO | 0.0213 (= ceiling) | 0.0163 (0.011–0.023) | 0.0163 | ok | 2.4× / 1.2× |
| mnist_letter_a | 64 | 128 (64) | truth_latent | NO | 0.0213 (= ceiling) | 0.0163 (0.011–0.023) | 0.0163 | ok | 2.4× / 1.2× |
| mnist_letter_a | 256 | 16 | truth_latent | NO | 0.0213 (= ceiling) | 0.0188 (0.014–0.043) | 0.0188 | ok | 2.7× / 1.4× |
| mnist_letter_a | 256 | 32 | truth_latent | NO | 0.0213 (= ceiling) | 0.0176 (0.013–0.026) | 0.0176 | ok | 2.6× / 1.3× |
| mnist_letter_a | 256 | 66 | truth_latent | NO | 0.0213 (= ceiling) | 0.0150 (0.011–0.021) | 0.0150 | ok | 2.2× / 1.1× |
| mnist_letter_a | 256 | 128 | truth_latent | NO | 0.0213 (= ceiling) | 0.0092 (0.008–0.017) | 0.0092 | ok | 1.3× / 0.7× |

**Solver audit: 0 failures in 72 rows.** Every fit ends at its trajectory minimum and at or below its own `w = 0`
value; at the oracle anchor `w = 0` reproduces the autoencoding ceiling on all 24 rows (max |diff| 0.0 except
1.5e-8 on keyboard K = 256). So no chart number here is solver-bounded in the audit's sense.

Three readings across the 72 rows. (i) **The attacker's anchor costs nothing**: `proxy_nn` (neighbours of the
pixel-PCA recovery) matches `truth_nn` (neighbours of the truth) everywhere — 0.2941 vs 0.2916 (motorcycle K=64,
k=66), 0.2101 vs 0.2086 (keyboard K=64), 0.1022 vs 0.0998-equivalent on letters — so knowing the private image does
not help pick the neighbourhood; only the oracle LATENT (`truth_latent`) helps. (ii) **K = 256 beats K = 64 only
where k can exceed 63**, since `k_eff = min(k, K−1)` caps the K = 64 charts. (iii) **The decoder chart beats pixel
PCA only on letters**: on CIFAR it is level (motorcycle 0.2550 vs 0.2549 at k = 66; keyboard 0.2101 vs 0.2009),
on letters it is better at every k (0.1022 vs 0.1659 at k = 66; 0.0549 vs 0.1101 at k = 128).

#### C7 recomputed on the gate's images (ledger Q7) — job 356106, `experiments/decoder_chart/c7_index_sets.py`, FP64, CPU

C7 (job 350993, `two_walls.py`) drew its eight privates with `np.random.RandomState(1)`; the oracle ladder, which
measured the gate brackets, drew them with `torch.Generator().manual_seed(seed+7)` + `randperm`. Different images.
Same chart (public PCA of the added class, train split), same convention (`‖(x−μ) − VVᵀ(x−μ)‖/‖x‖`), both index sets
in one job; rows in `results/decoder_chart/c7_index_sets_356106.jsonl`. two_walls' stored means are reproduced to 4
decimals, so the difference below is the images, not the computation.

| release | k | ladder images (gate's) mean / median (range) | two_walls images mean / median (range) | ladder mean vs bracket low / high |
|---|---|---|---|---|
| motorcycle / MLP (0.0124–0.0186) | 16 | 0.3531 / 0.3413 (0.234–0.493) | 0.2456 / 0.2233 (0.190–0.362) | 28.5× / 19.0× |
| | 32 | 0.3176 / 0.2983 (0.224–0.432) | 0.2151 / 0.1930 (0.154–0.326) | 25.6× / 17.1× |
| | **66** | **0.2702** / 0.2549 (0.190–0.382) | 0.1845 / 0.1698 (0.128–0.300) | **21.8× / 14.5×** |
| | 128 | 0.2282 / 0.2180 (0.166–0.306) | 0.1566 / 0.1416 (0.100–0.261) | 18.4× / 12.3× |
| | **384** | **0.1596** / 0.1502 (0.112–0.216) | 0.1094 / 0.0994 (0.072–0.185) | **12.9× / 8.6×** |
| keyboard / CNN (0.0045–0.0090) | 16 | 0.2676 / 0.2858 (0.096–0.409) | 0.2840 / 0.1945 (0.092–0.652) | 59.5× / 29.7× |
| | 32 | 0.2432 / 0.2539 (0.088–0.379) | 0.2573 / 0.1794 (0.083–0.576) | 54.0× / 27.0× |
| | **66** | **0.2107** / 0.2009 (0.080–0.328) | 0.2058 / 0.1573 (0.072–0.394) | **46.8× / 23.4×** |
| | 128 | 0.1767 / 0.1681 (0.067–0.274) | 0.1704 / 0.1328 (0.061–0.283) | 39.3× / 19.6× |
| | **384** | **0.1222** / 0.1132 (0.050–0.179) | 0.1191 / 0.0944 (0.045–0.201) | **27.2× / 13.6×** |

Shortfall on the gate's own images (mean, as C7 quoted): motorcycle **21.8× the low end / 14.5× the high end at
k = 66** and 12.9× / 8.6× at k = 384 (C7 said 14.9× at 66 and ~9× at 384 against the point gate 0.0124, on the other
images); keyboard **46.8× / 23.4× at k = 66** and 27.2× / 13.6× at k = 384 (C7: 45.7× and ~26×). C7's conclusion
(no public-PCA width satisfies both walls) holds on the gate's images at both ends of both brackets; on motorcycle the
ladder's images are harder for PCA than C7's by ~1.45× at every k, on keyboard the two sets agree within 3%.

#### Verdict per image set (FINAL, all 18 cells)

| image set | ceiling (×8) vs bracket | best attacker-available at k ≤ 66 | best attacker-available at any k ≤ 128 | oracle-anchor floor | outcome |
|---|---|---|---|---|---|
| mlp_motorcycle | 0.0554 vs 0.0124–0.0186 | 0.2550 (K=256, k=66) — 20.6× / 13.7× | 0.2324 (K=256, k=128) | 0.0371 | **CEILING-BOUND** |
| cnn_keyboard | 0.0237 vs 0.0045–0.0090 | 0.2101 (K=64, k=66) — 46.7× / 23.3× | 0.1846 (K=256, k=128) | 0.0207 | **CEILING-BOUND** |
| mnist_letter_a | 0.0213 vs 0.0069–0.0139 | 0.1022 (K=256, k=66) — 14.8× / 7.4× | 0.0549 (K=256, k=128) | **0.0092 (inside bracket)** | **CEILING-BOUND** (see caveat) |

The pre-registered PASS condition (an attacker-available arm at `k ≤ 66` below the bracket's low end) is not met on
any set, and FAIL's condition (nothing below the high end at any `k ≤ 128`) holds on all three; the ceiling row,
reported first as required, already decided it. The caveat on letters is recorded above and matters for what comes
next: the decoder's latent family DOES contain the eight letters to 0.0092 — inside the 0.0069–0.0139 bracket — so
on that set the obstruction is the attacker's anchor and the width `k = 128 > 66`, not the decoder. On the two CIFAR
sets the decoder itself is the wall (oracle floors 0.0371 and 0.0207, both above the brackets' high ends).

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
- **CUDA OOM, not a science deviation, but it shaped the run:** short-gpu A40s are shared and a plain `num=1`
  reservation does not reserve GPU memory. 12 of the first 18 cells died. Fix: `-gpu "num=1:gmem=20G"` plus
  `--chunk 4` (decode 4 images at a time instead of 8) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  No chunk-4 job failed. A cell that OOM'd wrote only the closed-form rows; those fragments were deleted so no row
  in the tables comes from a partial run.
- **The pre-registered CEILING-BOUND definition uses `D(E(x))`, which is not the decoder's floor.** The oracle-anchor
  fit reaches below `D(E(x))` on every set (letters 0.0092 vs 0.0213), because the encoder's posterior mean is not
  the decoder's best latent. The verdicts are reported on the pre-registered definition, with the oracle floor
  stated beside each one; on letters the two disagree about what the obstruction is, and that is said explicitly.
- **The letters set has no identifiability cap of its own recorded here.** `k ≤ 66` is the CIFAR releases' cap
  (`m + r − N − 1` at m = 11, r = 64, N = 8); the letters release shares those shapes, so the same arithmetic gives
  66, but no measurement in this job tests it — the `k = 128` letters rows are quoted as being above the cap on that
  arithmetic alone.
