# Plan 2026-09-18 — CNN rank law, class composition, pretrained-decoder chart, bootstrap chart, MNIST gate

Requested by Yoad on 2026-09-18 (chat): (1) run the multilayer certificate rank check on a CNN, with a T sweep;
(2) split the added classes: A's and T's as two separate classes on MNIST, motorcycles and bottles on CIFAR, and
compare same-class against mixed-class private batches, across charts and T's; (3) a pretrained decoder (unCLIP-style)
as the chart, fidelity first; (4) the iterative/bootstrap chart (round2 TEST 8), never run; (5) the MNIST landing gate,
the one measurement blocking the depth-window question. Five work packages, built by sibling sessions in parallel,
audited before submission (this file is the audit's object). Every cell runs on WEXAC via `bsub`; nothing runs on a
login node. FP64 wherever a rank or a certificate residual is read; the decoder fidelity arm is FP32 and says so.

Conventions that bind every package (from `notes/APPROVER_HANDOVER.md`, `results/CLAIMS_LEDGER.md` §0.1,
LESSONS 2026-09-17): pre-register outcomes before the first row; three outcomes, not two (pass / fail / vacuous);
every number carries the configuration it was measured in (space, release, depth, start family, solver) and is
never placed in one expression with a number from another configuration; a control that could have disagreed in
the same job and row format; residual-zero-wrong-image (alias) is reported separately from residual-not-zero
(solver); consistency with the release is not evidence of correctness, only distance to ground truth is.

## WP1 — Multilayer rank law on a CNN, plus the T arm

**Question.** Does the corrected depth law `stacked rank = min_j(d_j + Σ_{l<j} q_l)` (refuting T5.2's `min(k_1, Σ q_l)`)
hold when the frozen encoder is convolutional, where a layer's certificate acts on patch vectors shared across
positions? And how does the certificate at drifting layers degrade with the number of LoRA steps T?

**Harness.** New `experiments/multilayer_cert/conv_encoder_ranklaw.py`, ported from `real_encoder_ranklaw.py` (same
row format, same tolerance ladder, same gap diagnostic, same pre-registration) onto `models/exact_inversion/mnist_conv_deep.pth`
(spec `deep` in `conv_certificate.py`: conv 1→64→128→256→256, stride 2, GELU, linear head; 98.6% test).
Per adapted conv layer `l`: input object is the patch matrix `P_l(x) = unfold(h_l)` of shape `(d_l, P_l)`,
`d_l = C_in·3·3`; recorded span `U_l` = orthonormal basis of the `N·P_l` patch vectors of the truths; zero-drift
certificate `C_l = P_{col(A_{l,0} U_l)^⊥} A_{l,0}` with `A_{l,0} ∈ R^{r×d_l}`; stacked objective
`g(v) = concat_l vec(C_l P_l(ψ(v)))`. The head is a dense layer and is treated as in the MLP harness.
`d_l` here is the rank of the Jacobian of `vec(P_l)` w.r.t. the chart coordinate; `q_l = rank(C_l ⊗ I) J_l`.
Chart: pixel (`k = 784`) and PCA `k ∈ {16, 32, 66, 128, 256, 384, 512}` on 50 000 MNIST train images (existing `PCAChart`).
Configs: `first ∈ {1, 2}`, `maxL = 5` (four convs + head). `N = 8`, `r = 64` (so `r − N > 0` at every dense-like count;
record the conv-vacuous condition `N·P_l ≥ d_l` per layer explicitly as in `conv_certificate.py`).

**T arm** (`--T-arm 1 5 20 100 400`, `lr = 0.01`, full batch, `B_0 = 0`, FP64): train LoRA on all adapted layers with
`conv_certificate.run_training`, then compute the DRIFTED certificate from the release, `C_l = P_{row(B_{l,T})^⊥} A_{l,T}`,
and record per layer: `rank B_{l,T}` (the recorded count `N'_l(T)`), `rank C_l = r − N'_l(T)`, the residual
`‖C_l P_l(truth)‖ / ‖A_{l,T} P_l(truth)‖` (zero at the first adapted layer, whose input never moves; the number at deeper
layers is the drift cost), and the stacked rank ladder with the drifted certificates.

**Pre-registered.** (a) DISCRIMINATION: in a config with `corrected_pred < t52_pred`, measured rank at `corrected_pred`
and strictly below `k_1`. FALSIFIED: measured at `t52_pred`. CONTROL in the same job: a shallow-first config with
`corrected_pred == t52_pred` saturating at that common value. VACUOUS: no config has `corrected_pred < t52_pred`
(a legitimate result; report it). CONV-VACUOUS per layer: `rank C_l = 0` where `N·P_l ≥ d_l` — recorded, not hidden.
(b) T arm, two live hypotheses stated before the run: `N'_l(T)` grows like `N·T` until it saturates at `min(r, d_l)`
(ledger M2, synthetic) versus plateaus far below (the r=256 MLP plateau, RESULTS.md Step ~4186); the row decides.
The first adapted layer's `N'` must be exactly `N` at every T (frozen-input invariance) — a violation is a harness bug.
(c) No gap at the cut → report the ladder and the spectrum, never an integer.

**Resources.** `long-gpu`, A100 exclusive (`-gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4"`, LESSONS: FP64 on
the shared queue is slow and noisy), 32 GB, ~6 h. Output `results/multilayer_cert/conv_ranklaw_<jobid>.jsonl`,
write-up section in `experiments/multilayer_cert/RESULTS.md` (§6, "CNN").

## WP2 — Added-class composition × chart × T (MNIST letters, CIFAR-100 classes)

**Question.** Does it matter whether the eight private images are one added class or two? Is the certificate
(separable, per image) indifferent to batch composition while the NTK/linearised route (coupled through one sum) is
not? Does a per-class chart beat a pooled chart for the mixed batch? How does each depend on T?

**Harness.** `experiments/cifar/ntk_vs_certificate.py` (existing; on-chart privates; certificate + NTK arms; charts
`pca`, `ae`; `--Ts`; two added classes get two head rows). Additions: (i) `--newclass cifar100:bottle`,
`--newclass cifar100:motorcycle` (verify both names exist in the CIFAR-100 fine-label list); (ii) a `--same-row`
flag: two classes' images but ONE new head row (mixed content, single label), separating "two labels" from "two contents";
(iii) for mixed cells, chart variants `pca` (per-class PCA on each image's own class, as now — verify) and `pca_pooled`
(one PCA on the union of both classes' public images). Cells, all `N = 8`, `r = 64`, `k ∈ {16, 32, 48}`, `Ts = 1 5 20 100 400`,
200 starts, seed 1:

| cell | private batch | head rows |
|---|---|---|
| mnist_a | 8 a | 1 |
| mnist_t | 8 t | 1 |
| mnist_mixed | 4 a + 4 t | 2 |
| mnist_mixed_samerow | 4 a + 4 t | 1 |
| cifar_motorcycle | 8 motorcycle | 1 |
| cifar_bottle | 8 bottle | 1 |
| cifar_mixed_mb | 4 motorcycle + 4 bottle | 2 |
| cifar_mixed_mb_samerow | 4 + 4 | 1 |

EMNIST `letters` merges cases: the `a` class holds both a and A, the `t` class both t and T. Record per private image
which case it is (the join key is the index list printed by the harness) so an "A's" panel is labelled honestly.

**Pre-registered.** Certificate: images found = 8/8 in every composition at `T = 400` on-chart (composition-blind);
if a mixed cell drops below its single-class cells, that is a finding against separability. NTK free-coefficient:
prediction NOT fixed in advance beyond "the two-row mixed cell is at least as hard as the harder single-class cell";
the T dependence is the object. Chart: at fixed `k`, per-class PCA fidelity ≥ pooled PCA fidelity for each image
(pooled spends directions on the other class); report the fidelity table beside the recovery table so the two are
never conflated. Every cell reports the model floor at the truth.

**Resources.** One `long-gpu` job per cell via the existing `submit_ntk_vs_cert.sh` (extend the case list), 24 GB,
~3–8 h each. Outputs `results/ntk_vs_cert/sweep_<cell>_<jobid>.jsonl`, saved tensors + grids under the harness's
`--save-dir` / `--fig-dir` (verify it writes both; if not, add it: truth / recovery / control per image, best and worst).

## WP3 — Pretrained decoder as chart: fidelity check first (no inversion)

**Question.** Can a pretrained image decoder, used as a local chart `G(w) = D(a + U w)`, contain the private images
to within the landing gate at a width below the identifiability cap? This is the untried slot: every measured chart
so far is public PCA, warped PCA, class-local PCA, self-trained VAE/AE, or feature-space PCA, and PCA won every time.

**Decoder.** `stabilityai/sd-vae-ft-mse` (the KL-VAE decoder that the unCLIP / SD pipelines decode through), via
`diffusers` installed into the `rec` env with `pip` (record the versions). Download on the login node (HF reachable),
cache under `~/.cache/huggingface`; compute nodes may have no network, so the job must fail loudly if the weights
are missing rather than try to download.

**Images.** The same private images as the releases we would attack: the eight CIFAR-100 motorcycles (MLP release)
and eight keyboards (CNN release) at the oracle ladder's seed indices (`experiments/oracle_ladder/ladder_cell.py`),
plus the eight EMNIST `a` letters of the letters release (`new_class.py --domain mnist --letter a`, seed 1).
Public pools: the class's public train images (as the PCA chart uses).

**Measurements** (relative pixel error, `‖x̂ − x‖/‖x‖`, median and range over the 8, in the native 32×32 / 28×28 space,
with the resize path to the decoder's 256×256 input stated and applied identically to every arm):
1. Autoencoding ceiling `‖D(E(x)) − x‖/‖x‖` at input scales 64/128/256. Everything below is bounded by this.
2. Global latent PCA chart: top-`k` PCA of the public pool's latents `E(x_pub)`, decode; `k ∈ {16, 32, 66, 128}`.
3. Local patch chart: anchor `a` = mean latent of the `K` nearest public images to `x` in latent space (`K ∈ {64, 256}`),
   `U` = top-`k` PCA directions of those `K` latents, fidelity `min_w ‖D(a + U w) − x‖/‖x‖` by Adam in `w` (FP32,
   2000 steps, 3 restarts, report the best). Oracle-anchor control (`a = E(x)`) marked NOT attacker-available.
   Attacker version: anchor chosen by nearest neighbour to the PCA-chart recovery, not to `x` — report both.
4. Pixel PCA at the same `k` on the same images, in the same job, as the baseline PCA wins against.
Compare against the measured gates (0.0124 MLP-motorcycle, 0.0045 CNN-keyboard; MNIST gate from WP5 when it exists;
until then no MNIST shortfall number is quoted).

**Pre-registered.** PASS: some attacker-available arm at `k ≤ 66` lands below the gate. FAIL: no arm below the gate at
any `k ≤ 128`. CEILING-BOUND: the autoencoding ceiling itself is above the gate, in which case no chart built on this
decoder can pass and the result is about the decoder, not the chart. The ceiling row must be reported first.

**Resources.** `short-gpu`, one job, FP32 (fidelity numbers, no rank read). Outputs `results/decoder_chart/fidelity_<jobid>.jsonl`,
grid `figures/decoder_chart/`, write-up `experiments/decoder_chart/RESULT.md`.

## WP4 — Bootstrap / iterative chart (round2 TEST 8), first run

**Question.** Can the chart be improved from what the attack recovers, without private data touching the chart fit?

**Why a linear global refit cannot work, and what "refit" therefore means.** In a global PCA chart the recovery is
exactly the projection of the truth onto the chart, so refitting a global PCA on recovery + public data cannot leave
the original span. The bootstrap that can move is the LOCAL chart: neighbours of the recovery in the public pool
define a new anchor and new directions (the "local families" schematic, `figures/gal_2026-09/08_local_families.png`,
proposal only until now).

**Setting.** Below the certificate line, where aliases are excluded by the residual guard: MNIST strong MLP,
EMNIST `a`, `N = 8`, `r = 64`, `T = 400`, `k = 32`, privates RAW (not on-chart), so there is chart error to remove.
Round 0: global public PCA-32 of EMNIST-a train; certificate LM attack from 200 random starts (the `new_class.py`
cell family or the certificate arm of `ntk_vs_certificate.py`; reuse, do not rewrite). Round `t+1`, per image:
anchor = mean of the `K = 200` nearest public images to `x̂_i^{(t)}`, directions = top-32 PCA of those neighbours;
attack again from `x̂_i^{(t)}` and from fresh random starts. Four rounds.

**Metrics per round and per image.** Chart error of the TRUE image in that round's chart (fidelity); recovery error
vs the truth; certificate residual at the recovery; landed (`err < 1e-2`). **Controls in the same job:** (i) neighbours
of a RANDOM public image instead of the recovery (the decisive control); (ii) neighbours of the truth (oracle bound,
not attacker-available); (iii) round 0 re-run with the same starts (no-refit baseline).

**Void condition.** Round-0 median recovery error above 0.5 → nothing to bootstrap from; report VOID, not null.

**Pre-registered.** IMPROVES: fidelity and recovery error fall monotonically over rounds and beat control (i).
REFIT-ONLY: control (i) improves as much → the gain is from locality, not from what was recovered. STALLS: no change
after round 1. ALIAS: any recovery whose residual sits above the floor while its image error is small is flagged,
never counted as a landing.

**Resources.** `long-gpu`, one job, FP64, ~6 h. Outputs `results/bootstrap_chart/rounds_<jobid>.jsonl`, per-round
tensors `.pth`, grid per round (truth / recovery / control), `experiments/bootstrap_chart/RESULT.md`.

## WP5 — The MNIST landing gate

**Question.** At what measured chart error does recovery stop on the MNIST letters release? Without it the depth
result (identifiability reaching `k ≈ 417` with 8 adapted layers, chart error 0.0385 at `k = 384`) cannot be turned
into a window statement, and quoting the CIFAR gate against the MNIST ladder is the cross-construction error struck
twice on 2026-09-17.

**Harness.** Extend `experiments/oracle_ladder/ladder_cell.py` with an example `mlp_letter_a`: `mnist_mlp_strong.pth`,
EMNIST `a`, `N = 8`, `r = 64`, `T = 400`, `k = 32`, privates RAW, same ε ladder
`0 0.01 0.02 0.03 0.05 0.075 0.10 0.15 0.20 0.30 0.40 0.60`, same 400 starts, same LM solver, same landing bar
(`err < 1e-2`), plus the `pca` cell and the wrong-release control, exactly as for the two CIFAR examples. Extend
`submit_ladder.sh` with the new example; extend `make_ladder.py` so the MNIST table and figure are generated by the same
generator (its off-by-one bug is fixed; do not reintroduce it).

**Pre-registered.** The gate is the largest measured projection error at which all 8 are found, and separately the
smallest at which 0 are found (report both ends; never a single number). If MNIST's gate is far above CIFAR's, say so
as a difference in backbone and data, not as a property of "the gate".

**Resources.** 14 `short-gpu` jobs (12 ε + pca + wrong release), 24 GB, ~1 h each. Output rows into
`results/oracle_ladder/rows.jsonl` (same file, same schema, example key distinguishes), regenerated `RESULT.md`.

## Coordination

- Builders touch only their own package's files. Shared files (`STATUS.md`, `LESSONS_LEARNED.md`, `CLAIMS_LEDGER.md`)
  are edited by the coordinating session after results land.
- Each builder: write code, submit a SHORT-queue smoke cell (tiny config, minutes), read its log, fix, then submit the
  full job(s); report job ids, the exact command lines, and what the smoke run printed. No local runs of any kind.
- Base models are the trained ones (MNIST strong 98%, CIFAR CNN 93% test, MNIST conv deep 98.6%); the CIFAR MLP is a
  weak 58% encoder and any cell on it says so in its row.
- Numbers from these runs are provisional until the row is read by a second session.
