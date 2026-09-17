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

## WP0 — Base models must be fully trained (gate on every cell)

Yoad (2026-09-18): "make sure base fully trained". Every cell in WP1–WP5 reports, in its row, the frozen base's
train accuracy, test accuracy and final train loss, read from the checkpoint at load time (not copied from a note).
A base is FULLY TRAINED for this purpose when its train loss has collapsed (train accuracy ≥ 99.5%, train loss ≤ 1e-2)
— the Haim et al. regime the repo already implements as `--overtrain` in `experiments/cifar/cifar_newclass.py`
(no augmentation, no weight decay, train past zero error; `cifar10_mlp_overtrained_newclass.pth` is 100.00% train).
Before any package submits its full job, the builder checks its checkpoint(s) against this gate:

| checkpoint | used by | known state | action |
|---|---|---|---|
| `mnist_mlp_strong.pth` | WP2 mnist, WP4, WP5 | 98.24% test; train acc / loss NOT recorded | measure train acc + loss; if below the gate, train `mnist_mlp_strong_full.pth` to the gate with `train_strong_backbone.py` (add `--target-train-acc 0.995`, no augmentation) and use it, with test acc recorded |
| `mnist_conv_deep.pth` | WP1 | 98.63% test; train state NOT recorded | same: measure; over-train a `_full` twin if needed (`conv_certificate.train_backbone`, more epochs) |
| `cifar10_cnn_newclass.pth` | WP2 cifar (CNN arm), WP5 keyboard gate | 99.8% train / 92.7% test | passes |
| `cifar10_mlp_overtrained_newclass.pth` | WP2 cifar (MLP arm) | 100.00% train / 57.3% test, loss 6.8e-4 | passes; the plain `cifar10_mlp.pth` (58% / weak) is NOT used in any new cell |
| `cifar10_mlp.pth` | WP3 image selection only (same 8 images as the ladder) | weak | images only, no release read from it |

WP2's CIFAR cells therefore run on the CNN backbone AND the over-trained MLP, as two arms, both recorded. If a new
`_full` twin is trained, the letters cells that already exist on the 98% model are NOT overwritten: the twin gets a
new checkpoint name and the row records which one it ran on, so the earlier 8/8 letters result stays comparable.

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

## WP4 — Bootstrap / iterative chart (round2 TEST 8), first run — two variants

**Question.** Can the chart be improved from what the attack recovers, without private data touching the chart fit?

**Variant A — class-recognition bootstrap (Yoad, 2026-09-18: "refit C1 on the general class we saw; like a bike
generally recognised; change C1 to be the chart of bike").** Round 0 uses a GENERIC public chart the attacker can
build without knowing the added class: PCA-`k` of a broad public pool (CIFAR: all CIFAR-100 train images; MNIST:
all EMNIST letters). Recover. Then RECOGNISE the class of the recovery with a public classifier that never saw the
private images (CIFAR: an ImageNet-pretrained ResNet-50 from `~/.cache/torch/hub`, top-1 mapped to the CIFAR-100
class vocabulary by name, or a CIFAR-100 classifier trained on the public pool; MNIST: a 26-way EMNIST-letters
classifier trained on the public pool). Round 1 chart = PCA-`k` of THAT class's public pool (the stand-in for
"fetch bike images online": the public class pool plays the role of the web search; recorded as such). Recover
again. Round 2 (optional): re-recognise on the round-1 recovery; stop when the class is stable.
Controls in the same job: (i) round 1 with the chart of the SECOND-ranked class (wrong-class refit) — if it improves
as much, the gain is chart narrowing, not recognition; (ii) round 1 with the TRUE class chart handed over (oracle
class label, not attacker-available) — the ceiling of this variant; (iii) recognition accuracy on the round-0
recoveries, per image, so a wrong class is a recorded event rather than a silent one.

**Variant B — local-neighbour bootstrap.** In a global PCA chart the recovery is exactly the projection of the truth,
so a global refit on recovery + public data cannot leave the original span; what can move is the LOCAL chart
(the "local families" schematic, `figures/gal_2026-09/08_local_families.png`, proposal only until now). Round `t+1`,
per image: anchor = mean of the `K = 200` nearest public images (of the recognised class) to `x̂_i^{(t)}`,
directions = top-`k` PCA of those neighbours; attack again from `x̂_i^{(t)}` and from fresh random starts. Four rounds.
Control: neighbours of a RANDOM public image instead of the recovery (the decisive control); oracle: neighbours of
the truth.

**Setting.** Below the certificate line, where aliases are excluded by the residual guard. Two releases: MNIST strong
MLP (or its `_full` twin per WP0), EMNIST `a`, `N = 8`, `r = 64`, `T = 400`, `k = 32`, privates RAW (not on-chart);
and CIFAR CNN, motorcycle (the ladder's eight photographs), same adapter. Certificate LM attack from 200 random
starts at every round (reuse the `new_class.py` / `ntk_vs_certificate.py` certificate arm; do not rewrite the solver).

**Metrics per round and per image.** Chart error of the TRUE image in that round's chart (fidelity); recovery error
vs the truth; certificate residual at the recovery; landed (`err < 1e-2`); recognised class and its rank.
Round-0 baseline re-run with the same starts (no-refit control).

**Void condition.** Round-0 median recovery error above 0.5, or recognition at chance on the round-0 recoveries →
nothing to bootstrap from; report VOID, not null.

**Pre-registered.** IMPROVES: fidelity and recovery error fall over rounds and beat the wrong-class / random-anchor
control. REFIT-ONLY: the control improves as much → the gain is narrowing/locality, not what was recovered.
STALLS: no change after round 1. ALIAS: any recovery whose residual sits above the floor while its image error is
small is flagged, never counted as a landing. Note that variant A's round-1 chart for a correctly recognised class
IS the per-class public PCA that C7 measured (fidelity 0.18 at k=66 on CIFAR): variant A can therefore at best reach
the known per-class PCA fidelity; its value is showing the attacker gets there WITHOUT knowing the class. Variant B
is the one that can go below per-class PCA.

**Resources.** `long-gpu`, one job per release, FP64, ~6 h. Outputs `results/bootstrap_chart/rounds_<jobid>.jsonl`,
per-round tensors `.pth`, grid per round (truth / recovery / controls), `experiments/bootstrap_chart/RESULT.md`.

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

## Audit 2026-09-18 (sibling auditor) — fixes applied before submission

The audit is the authority over the sections above wherever they conflict; builders received these as instructions.

**WP0.** `mnist_mlp_d15w1000.pth` added to the gate table (it is the encoder the depth window lives on). Every
checkpoint reports loss as well as accuracy. **No package substitutes a `_full` twin this round**: all cells run on
the original checkpoints so no two packages sit on different releases; twins, if trained, are a separate later arm.
The conv-deep checkpoint's recorded accuracy is 98.77% (`step93_convdeep_205887.jsonl`), not 98.6%.

**WP1 — the deep spec is vacuous by arithmetic; replaced.** Layer widths along the deep spec are
784 → 12544 → 6272 → 4096 → 1024 → 10; every chart `k ≤ 784` is below every intermediate width, so `d_j = min(k, 784)`
at all layers, both laws coincide provably (rank-preserving case, T5 §137–142), and no `first`/`maxL` discriminates.
At `r = 64` conv layers 1–3 are certificate-vacuous (job 205887: patch-span ranks 9/232/117/32, `rank C = 0/0/0/32/56`),
so the T arm would be head-only. Replacement: a **bottlenecked conv spec** with a mid-stack contraction below `k`
followed by at least two adapted layers, and `r = 256` so conv certificates are non-empty:
`1→64 (s2, 14×14) → 128 (s2, 7×7) → 8 (s2, 4×4; width 128 < k) → 256 (s2, 2×2) → dense 1024→1000 → head 10`,
trained to the WP0 gate by `conv_certificate.train_backbone` (new `SPECS["bottleneck"]`, checkpoint
`mnist_conv_bottleneck.pth`). Pre-submission arithmetic at `r = 256`, `first = 1`: `q_1 = 0` (d=9), `q_2 ≈ 256−232 = 24`,
`q_3 ≈ 256−128 = 128`, `q_4 ≈ 72−32 = 40`, `q_dense = 248`, `q_head ≤ 9`; corrected at `j = conv4` is
`128 + 0 + 24 + 128 = 280` against T5.2 `min(784, 449) = 449` for `k ≥ 280` — discriminating; the shallow-first
control (`first = 5`, dense+head only) coincides. The measured patch-span ranks replace these estimates in the row.
Corrections to the definitions: conv vacuity is `patch_span_rank ≥ min(r, d_l)` (not `N·P_l ≥ d_l`); certificate rank
is `min(r, d_l) − N'_l`; the "`N' = N` at the first adapted layer" invariant holds only for DENSE layers (a conv first
layer records its patch span, 9 or 64, never 8) and is asserted for the dense layers only; `d_l` in the row means
`rank M_l` (T5's symbol) and the patch dimension is written `p_l = C_in·9`. `sigma0` is set per layer as
`1/sqrt(p_l)`, and the run is labelled as such. Hypothesis (b) of the T arm is restated: at conv layers `B_{l,1}`
already sums `N·P_l` rank-one terms at `T = 1`, so the `N·T` growth law is a dense-layer statement and the conv rows
test only whether `N'_l(T)` moves at all with `T`. Resources: rows `Σ r·P_l ≈ 68k × k = 784` in FP64 is ~0.4 GB, so a
plain `long-gpu` GPU with 32 GB is enough; A100 is not required.

**WP2.** The harness's existing `pca` chart POOLS both classes' public images (`ntk_vs_certificate.py:179`); per-class
PCA is the addition, named `pca_perclass`. Because privates are on-chart, per-class and pooled charts define
different private images and hence different releases: recovery is compared only within a chart, and charts are
compared only on the RAW-image projection error measured before projection (a table in the same job). Cells
already run with this exact configuration — `mnist_a` and `mnist_mixed` at `T ∈ {1, 400}`, jobs 302279 / 302280 /
304349 — are reused, and only `T ∈ {5, 20, 100}` is added for them. `build_cifar` hard-codes the MLP; the CNN arm
exists only once a backbone flag is in and its smoke log shows the CNN loaded. Pre-registration has three outcomes
per cell: 8/8, partial (count reported, R5), below the single-class cells.

**WP3.** Gates are brackets, never points: MLP-motorcycle 0.0124–0.0186, CNN-keyboard 0.0045–0.0090 (ledger §0.1, A22).
The Adam-fitted local-chart fidelity is solver-bounded while pixel PCA is closed-form: the oracle anchor with `w = 0`
must reproduce the autoencoding ceiling exactly, else the row is a solver failure and says so. `diffusers` is NOT
installed into the shared `rec` env (other lanes run on it): it goes to a separate `--target` directory added to
`PYTHONPATH` by the job script only.

**WP4.** On CIFAR the raw privates sit at chart error ≥ 0.32 at `k = 32`, so "landed vs truth" is dead by construction;
the per-round floor is the certificate residual at the TRUTH'S PROJECTION into that round's chart, and two errors
are reported per round: recovery-to-projection (did the solver reach the chart's best) and recovery-to-truth (did the
chart improve). Void: round-0 recovery-to-projection above 1e-2 (solver did not reach the chart). Recognition is by an
in-job classifier trained on the public pool only (no ImageNet name map: CIFAR-100 "motorcycle" has no ImageNet
counterpart), calibrated on PCA-`k` projections of held-out public images of the class; VOID unless at least 5 of 8
round-0 recoveries are top-1 correct AND the calibration accuracy is above that.

**WP5 — the plan's own cross-construction error.** The 0.0385-at-`k = 384` fidelity ladder is MNIST DIGITS on the
15-LAYER encoder; the letters gate is EMNIST on the 3-layer strong MLP, a different question. Two arms, both run,
neither quoted against the other: (a) `mlp_letter_a` as written (the gate for the letters cells and WP4's void
condition); (b) `d15_digits`: head adapter (`r = 64`, `T = 400`) on `mnist_mlp_d15w1000.pth`, privates = the eight
test digits at the k-sweep's join-key indices (seed+7), chart pool = the k-sweep's 50 000 train digits, same ε ladder.
Arm (b) shares encoder, images and chart pool with the depth window; it differs in adapter placement (head only,
since no multi-layer attack harness exists) and in being a confident batch (known classes), and the row reports
`rank B_T` and the recording strength so a weak recording is visible. It is the closest attainable gate for the
window, and the write-up says exactly that.
