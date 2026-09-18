# P2(ii) — zero-drift conv rank law on ResNet-18 / CIFAR-10 (builder notes, 2026-09-18)

Plan: `notes/plan_2026-09-18_multilayer_parameter_program.md` package P2, Audit item 7. Second real architecture for
the conv rank law of `RESULTS.md` §6-CNN. **Theory rows at the truth, zero drift: no LoRA training, no solve, no
attack.** Numbers here are from the smoke only and are provisional until the `full` stage runs and a second session
reads the rows. Nothing in this file is a claim; claims go to `results/CLAIMS_LEDGER.md` by the coordinator.

## Files

| file | role |
|---|---|
| `experiments/exact_inversion/train_resnet_backbone.py` | trainer to the base gate (FP32); `build_resnet18_cifar()` / `load_resnet18()` are THE constructor/loader every reader imports |
| `experiments/exact_inversion/base_training_gate.py` | +8 lines: family `cifar_resnet` (stem `cifar10_resnet18`), additive — no other family's path touched |
| `experiments/multilayer_cert/resnet_ranklaw.py` | FP64 harness; pre-registration in the docstring |
| `scripts/run_resnet_ranklaw_wexac.sh` | `submit train\|smoke\|full`; pre-submission arithmetic in the header |
| `models/exact_inversion/cifar10_resnet18.pth` | the base (state_dict + train/test acc/CE, epochs_run, rule_met, recipe, git, history) |
| `results/multilayer_cert/resnet_ranklaw_smoke_<job>.jsonl` / `resnet_ranklaw_<job>.jsonl` | rows: PATCH_SPAN / CONVLAYER / RANKLAW |

## Design decisions (and why)

1. **Which stage.** The brief's "stage 2: 128 channels, 8×8 positions, p_l = 128·9 = 1152, P_l = 64" is torchvision
   `layer3` with the 3×3 stride-1 stem (layer1 64@32², layer2 128@16², layer3 256@8², layer4 512@4²): the stage whose
   INPUT is the 128-channel map and whose convs run at 8×8. Only `layer3.0.conv1` (stride 2) has p_l = 1152; the
   other three convs have p_l = 256·9 = 2304. P_l = 64 and N·P_l = 512 for all four. `--stage` selects another stage.
2. **Certificate on conv INPUT patches; skip path outside it.** conv1's input is the block input, conv2's is
   `relu(bn1(conv1(·)))`. The identity / 1×1-downsample path adds into the block OUTPUT only, so it never enters an
   adapted conv's input; it enters the chart Jacobian of the deeper convs like every other frozen op.
3. **BatchNorm eval = per-channel affine**, written explicitly (`bn_st`, `conv_bn`) so `torch.func` traces plain
   ops; checked at load against the module forward (`max abs diff` printed, asserted < 1e-9 in FP64).
4. **ReLU kept** (torchvision's), `inplace=False` at construction (same map). Piecewise linear ⇒ the chart Jacobian
   is exact a.e. No double-backward is needed (zero drift, no unrolled training).
5. **d_l from the representation Jacobian**, not the patch Jacobian: unfold (k=3, pad 1, stride 1 or stride 2 on
   an even side) reads every input element at least once, so it is an injective linear map and the ranks coincide;
   `--verify-patch-rank` measures both at k ≤ 128 and records `d_patch_equals_d_rep`.
6. **Row-space-reduced certificate in the stack.** `U_c^T C_l` (rank C_l rows) replaces `C_l` (r rows): `C_l =
   U_c (U_c^T C_l)` up to the part below the `1e-10·σ_max(A_0)` floor, and an orthonormal left factor leaves singular
   values unchanged, so ranks / gaps / cond are those of the full stack at a fraction of the rows.
7. **Pre-registered live/vacuous set** (rule of §6-CNN, vacuous iff N'_l ≥ min(r, p_l)): N'_l ≤ 512 < p_l for every
   conv, so the p_l-saturation death cannot occur; the r-side vacuity can — r = 64 and 256 vacuous unless the span is
   degenerate (N'_l < r), r = 512 live iff the 512 patch vectors are linearly dependent, r = 1024 live by arithmetic.
   The smoke therefore carries `--r 64 512 1024` (64 = the brief's smoke rank; 512/1024 exercise the stacked code
   path so a smoke cannot pass by testing nothing — ground rule 2).
8. **Public pool = CIFAR-100 train, all 100 classes** (50 000 images) for the PCA charts; private = the ladder's
   eight CIFAR-100 motorcycles (motorcycle TEST split, `randperm(seed+7)[:8]`, `ladder_cell.py`'s join key).
9. **Inputs are raw [0, 1] pixels, no normalisation** (as `cifar_newclass.py`), so the pixel chart is the raw
   pixel space and the BN stem absorbs the scale.

## Training / gate — job 366181 (short-gpu, hgn18, 1994 s total: 1608 s train + 363 s FP64 gate)

`train_resnet_backbone.py --seed 1 --epochs 120 --max-epochs 200 --target-train-acc 0.995 --min-train-loss 1e-2`
(SGD m 0.9 wd 5e-4, lr 0.1 cosine→1e-4 over 120 ep, bs 128, crop+flip; ≈15 s/epoch). **Gate met at epoch 98**
(un-augmented train split, measured in the job every epoch); the checkpoint's own numbers: train 99.806 % / CE
8.040e-3, test 93.81 % / CE 0.2327, `rule_met: true`, git ddac8c6.

Independent gate row (`base_training_gate.py --ckpts models/exact_inversion/cifar10_resnet18.pth`, family
`cifar_resnet`, FP64, BN eval, appended to `results/base_training_gate.jsonl`):

| checkpoint | train acc | test acc | train CE | margin>0 | recorded test acc matches | gate |
|---|---|---|---|---|---|---|
| `cifar10_resnet18.pth` | 99.806 % | 93.81 % | 8.038e-3 | 99.806 % | true | **PASS** |

The harness re-measured the same at load (99.806 % / 8.038e-3 / 93.81 %, `fully_trained_gate: true`). Note for the
next reader: `recorded_of` in the gate script stringifies the checkpoint's `history` list into `ckpt_recorded`
(one ~20 KB string on the row); harmless, but a future trainer may prefer not to store the history in the dict.

## Smoke — job 366250 (short-gpu, 372 s wall; rows `results/multilayer_cert/resnet_ranklaw_smoke_366250.jsonl`)

`--r 64 512 1024 --ks 32 128 --maxL 2 --verify-patch-rank`. Functional forward vs module forward: 5.3e-15.

| conv (torchvision) | input | p_l | P_l | N·P_l | N'_l | rank C_l at r=64 / 512 / 1024 | residual at truth (median) |
|---|---|---|---|---|---|---|---|
| 1 `layer3.0.conv1` s2 | 128@16² | 1152 | 64 | 512 | **512** (σ_512/σ_1 1.5e-3, next 0) | 0 / 0 / 512 | 9.5e-16 / 1.7e-15 / 1.8e-15 |
| 2 `layer3.0.conv2` | 256@8² | 2304 | 64 | 512 | **512** (6.3e-3) | 0 / 0 / 512 | 9.2e-16 / 1.6e-15 / 1.9e-15 |
| 3 `layer3.1.conv1` | 256@8² | 2304 | 64 | 512 | **512** (1.1e-2) | 0 / 0 / 512 | 9.4e-16 / 1.6e-15 / 1.9e-15 |
| 4 `layer3.1.conv2` | 256@8² | 2304 | 64 | 512 | **512** (1.6e-2) | 0 / 0 / 512 | 9.7e-16 / 1.6e-15 / 1.9e-15 |

**The 512 base patch vectors are linearly independent at every conv** (smallest kept singular ratio 1.5e-3–1.6e-2,
nothing near the 1e-10 cut), so N'_l = N·P_l = 512 exactly and, by the pre-registered rule, **every conv is
conv-vacuous at r = 64, 256 and 512** — the brief's whole r-set. The certificate rank formula min(r, p_l) − N'_l is
matched by the measured rank at every (r, conv). r = 1024 is live with rank C_l = 512 (32 768 conditions/image).

Stacked ladder (prefix over block 1), r = 1024: k = 32 → 32 at all 12 rungs (L = 1: 32 768 rows, cond@1e-10 10.2;
L = 2: 65 536 rows, 10.5); k = 128 → 128 at all rungs (cond 21.4 / 22.8). d_j = k at every conv and image
(`d_patch_equals_d_rep` true), q_l = k, T5.2 = corrected = k = ambient ⇒ `gap_at_corrected: null`,
`rank_test_outcome: no_gap_vacuous` — the CNN section's picture (one live conv pins the chart). r = 64 / 512 rows:
0 rows, `no_gap_vacuous` (nothing stacked).

**Consequence for `full`.** As specified (r 64 256 512) the full stage tests nothing at N = 8: every row would be
`no_gap_vacuous` with 0 rows. A live cell needs r > N·P_l (r = 1024 is the smallest power of two; rank C = r − 512)
or fewer patch vectors (N = 4 at r = 512 gives rank C = 256 only if the 256 patches are independent, which the
smoke's spectra say they are). Not submitted; coordinator's call.

## Open

- The brief's r-set {64, 256, 512} sits at or below N·P_l = 512: whether any of it is live is decided by the measured
  N'_l (smoke), not by arithmetic. If N'_l = 512 everywhere, r = 1024 is the smallest live rank at N = 8 (or N = 4 at
  r = 512) and `full` should add it.
- Single seed, single batch, single trained net; zero drift only (no T arm — the brief did not ask for one; the CNN
  harness's `run_training6` has no ResNet port).
