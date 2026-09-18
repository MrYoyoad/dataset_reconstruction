# WP0 — base-training gate (2026-09-18)

Plan: `notes/plan_2026-09-18_cnn_ranklaw_newclass_charts.md` §WP0. Rule: a frozen base is FULLY TRAINED when
**train accuracy ≥ 0.995 AND mean train cross-entropy ≤ 1e-2**, measured from the checkpoint on the FULL train split
(FP64, each checkpoint loaded by the loader that wrote it), never copied from a note.

Script: `experiments/exact_inversion/base_training_gate.py` · runner: `scripts/run_base_training_gate_wexac.sh
{gate [ckpt ...]|mlp_full|conv_full}` · rows: `results/base_training_gate.jsonl` (one JSON line per checkpoint,
appended; fields `train_acc, test_acc, train_loss, test_loss, train_margin_pos_frac, train_margin_median/min,
ckpt_recorded` (what the dict itself stored), `recorded_test_acc_matches, fully_trained`, git, host, cmd).

## Gate table — job 355833 (short-gpu, hgn23, 141 s, git f5c9874)

| checkpoint (`models/exact_inversion/`) | loader | train acc | test acc | train CE | margin>0 | recorded in ckpt | gate |
|---|---|---|---|---|---|---|---|
| `mnist_mlp_strong.pth` | `state_dict` 784-1000-1000-10 (train_strong_backbone) | 99.827% | 98.21% | 6.35e-3 | 99.827% | epoch 30, test_acc 0.9821 | **PASS** |
| `mnist_mlp_m26_strong.pth` | same, 26-wide head | 99.855% | 98.28% | 5.17e-3 | 99.855% | epoch 30, test_acc 0.9828 | **PASS** |
| `mnist_mlp_d15w1000.pth` | `deep_stack.load_deep` (depth 15) | 98.690% | 97.38% | 4.92e-2 | 98.690% | test_acc 0.9738, 40 epochs | FAIL |
| `mnist_conv_deep.pth` | `conv_certificate` spec `deep` | 99.660% | 98.77% | **1.031e-2** | 99.660% | test_acc 0.9877 | FAIL (loss only) |
| `mnist_conv.pth` | `conv_certificate` spec `shallow` | 99.302% | 98.63% | 2.24e-2 | 99.302% | test_acc 0.9863 | FAIL |
| `cifar10_cnn_newclass.pth` | `cifar_newclass.CNN` (eval) | 99.828% | 92.67% | 9.82e-3 | 99.828% | train 0.9983 / test 0.9267, loss 9.8e-3, 40 ep | **PASS** |
| `cifar10_mlp_overtrained_newclass.pth` | `cifar_newclass.MLP` | 100.000% | 57.31% | 6.82e-4 | 100.000% | train 1.0 / test 0.5731, loss 6.8e-4, overtrain, 200 ep | **PASS** |
| `cifar10_mlp.pth` | `state_dict` 3072-1000-1000-10 (train_cifar_backbone) | 74.890% | 53.19% | 7.17e-1 | 74.890% | epoch 5, test_acc 0.5319 | FAIL |

Every checkpoint's recorded `test_acc` agrees with the re-measured value to < 5e-4 (`recorded_test_acc_matches: true`
on all eight rows), so the loaders are the right ones. Train-side numbers for the five MNIST checkpoints did not
exist before this job. `cifar10_cnn_newclass` passes by 1.8% on the loss (9.82e-3), with BatchNorm in eval mode and
the un-augmented train split.

## Twins

Brief: a `_full` twin for every failing checkpoint that WP1/WP2/WP4/WP5 use.

- **`mnist_mlp_strong.pth` PASSES** (99.83% / 6.3e-3) → **no twin**. WP2-mnist, WP4, WP5 use the original as-is.
  `train_strong_backbone.py` nonetheless gained the stopping rule (`--target-train-acc`, `--min-train-loss`,
  `--max-epochs`, `--init-from`, `--out`) and the runner has an `mlp_full` stage, so a twin is one command away;
  the default code path (no rule) is unchanged except that the final checkpoint now also records train acc / loss.
- **`mnist_conv_deep.pth` FAILS on loss alone** (1.031e-2) → **`models/exact_inversion/mnist_conv_deep_full.pth`**,
  job 355840 (long-gpu, hgn55, git 7b5fb44; first attempt 355838 died at import on a repeated keyword, fixed):
  `python -u -m experiments.exact_inversion.train_conv_backbone --spec deep --init-from models/exact_inversion/mnist_conv_deep.pth
  --out models/exact_inversion/mnist_conv_deep_full.pth --target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300 --seed 1`.
  Continuation of the original (warm start), Adam 1e-3, bs 128, no augmentation, no weight decay, ReduceLROnPlateau
  (×0.5, patience 5, floor 1e-5 — never triggered). Epoch 0 (= the original) reproduced the gate's 99.660% / 1.031e-2
  exactly; the rule was met after **1 epoch**: train 99.812%, CE 5.074e-3, test 98.84%. Same checkpoint format
  (`Wms/bs/Whead/bhead/test_acc/git`) plus `train_acc, train_loss, train_margin_pos_frac, test_loss, epochs_run,
  init_from, stopping_rule, ...`; the original was not modified (mtime Sep 4 19:59).
  Re-gate: job 355842 — see the row below.
- `mnist_mlp_d15w1000.pth` and `mnist_conv.pth` fail but no new package reads a release from them (the plan's table
  does not list them); `cifar10_mlp.pth` is WP3 image-selection only. Not twinned. Anyone who wants them to the gate:
  `train_deep_backbone.py` has no rule yet; `conv_full` can be pointed at `--spec shallow`.

## Re-gate of the twin — job 355842

| checkpoint | train acc | test acc | train CE | margin>0 | gate |
|---|---|---|---|---|---|
| `mnist_conv_deep_full.pth` | 99.812% | 98.84% | 5.074e-3 | 99.812% | **PASS** (short-gpu, hgn55; row appended to the jsonl) |

## Audit additions (2026-09-18, plan "Audit" section)

### Bottleneck conv spec — `models/exact_inversion/mnist_conv_bottleneck.pth` (job 355869, from scratch)

`conv_certificate.SPECS["bottleneck"] = [(1,64,3,2,1), (64,128,3,2,1), (128,8,3,2,1), (8,256,3,2,1)]` followed by a
frozen DENSE hidden layer 1024 -> 1000 (GELU) before the 10-way head (`conv_certificate.DENSE_HIDDEN["bottleneck"] = 1000`;
`conv_forward(..., Wd=, bd=)`, `run_training(..., Wd=, bd=)`, `train_backbone(..., dense_hidden=)` — all keyword-only,
default None, so the shallow/deep specs and every positional call are byte-identical). The dense layer has NO LoRA slot:
`As[-1]` still acts on the head's input, which is now the dense layer's output (1000-dim), like the MLP harness's head.
Checkpoint: `Wms/bs/Whead/bhead` + `Wd/bd` + `dense_hidden, spec, test_acc, train_acc, train_loss, train_margin_pos_frac,
stopping_rule, ...`; `conv_certificate.py --spec bottleneck --ckpt <path>` loads it (main() reads `Wd/bd` when present).

Per-layer patch-span arithmetic (28x28 input, k=3, stride 2, pad 1; sides 28 -> 14 -> 7 -> 4 -> 2; N = 8):

| layer | C_in -> C_out | d_l = C_in*9 | P_l | N*P_l (N=8) | N*P_l >= d_l ? |
|---|---|---|---|---|---|
| conv1 | 1 -> 64 | 9 | 196 | 1568 | yes (saturates: patch span can fill d) |
| conv2 | 64 -> 128 | 576 | 49 | 392 | no |
| conv3 | 128 -> 8 | 1152 | 16 | 128 | no |
| conv4 | 8 -> 256 | **72** | 4 | 32 | no (32 < 72, but only by 40) |
| dense | 1024 -> 1000 (frozen, no slot) | 1024 | 1 | 8 | no |
| head | 1000 -> 10 | 1000 | 1 | 8 | no |

The bottleneck is conv4's input: d_4 = 72 patch dims (8 channels x 9), after conv3 with d_3 = 1152.  The flatten
after conv4 is 256 x 2 x 2 = 1024, which the dense layer maps to 1000.

### Depth-window twin — `models/exact_inversion/mnist_mlp_d15w1000_full.pth` (job 355870)

`train_deep_backbone.py` gained the same flags as `train_strong_backbone.py` (`--init-from`, `--target-train-acc`,
`--min-train-loss`, `--max-epochs`, `--out`, plateau lr halving); depth/width are read from the warm-start checkpoint
and the output is written with `deep_stack.save_deep` (`Ws / b1 / test_acc / depth / width / residual` + train acc /
loss / margin fraction / stopping rule), so `deep_stack.load_deep` and every consumer of `mnist_mlp_d15w1000.pth`
load it unchanged. Continuation of the original (Adam 3e-4, bs 128, no augmentation, no weight decay). Nobody
substitutes it this round; it is for a later re-run of the depth window.

| checkpoint | train acc | test acc | train CE | margin>0 | gate |
|---|---|---|---|---|---|
| `mnist_conv_bottleneck.pth` (355869: from scratch, Adam 1e-3, bs 128, seed 1, rule met at epoch 9, 13 s) | 99.742% | 98.63% | 9.102e-3 | 99.742% | **PASS** (re-gate 355873) |
| `mnist_mlp_d15w1000_full.pth` (355870: continuation, Adam 3e-4 -> 1.5e-4 after one plateau halving, rule met at epoch 25, 48 s) | 99.775% | 98.26% | 9.186e-3 | 99.775% | **PASS** (re-gate 355873) |

Both pass the loss gate by < 10% (9.1e-3 and 9.2e-3 against 1e-2): the rule stops at the FIRST epoch that clears it,
so these are "just past the line", not deep in the interpolation regime.  The gate script's own re-measurement
matches each trainer's recorded numbers to all printed digits.  Originals untouched (mtimes Sep 4).

Jobs: gate 355833 · conv twin 355838 (crashed at import) -> 355840 · re-gate 355842 · bottleneck 355869 ·
d15 twin 355870 · re-gate 355873.  All rows are in `results/base_training_gate.jsonl` (11 lines: 8 + 1 + 2).

## What the other packages should use

| package | checkpoint |
|---|---|
| WP1 (conv rank law, T arm) | `models/exact_inversion/mnist_conv_deep_full.pth` (record it in every row; the original's rows stay comparable) |
| WP2 mnist, WP4, WP5 | `models/exact_inversion/mnist_mlp_strong.pth` (passes; nothing to switch) |
| WP2 cifar CNN arm, WP5 keyboard gate | `models/exact_inversion/cifar10_cnn_newclass.pth` |
| WP2 cifar MLP arm | `models/exact_inversion/cifar10_mlp_overtrained_newclass.pth` |
| WP3 image selection | `models/exact_inversion/cifar10_mlp.pth` (images only; it fails the gate and must not supply a release) |

Every cell should read `train_acc / test_acc / train_loss` from the checkpoint dict at load time where the dict has
them (`cifar10_*_newclass`, `mnist_conv_deep_full`) and otherwise from `results/base_training_gate.jsonl`.

Code touched (not committed): `base_training_gate.py` (new), `train_conv_backbone.py` (new),
`scripts/run_base_training_gate_wexac.sh` (new), `train_strong_backbone.py` (flags), `conv_certificate.train_backbone`
(keyword-only `init / target_train_acc / min_train_loss / plateau_patience / return_stats`; positional use byte-identical).

## Activation twins — P5 step 1 (2026-09-18, plan `notes/plan_2026-09-18_multilayer_parameter_program.md`)

Trained from scratch with `--act {relu,tanh}` (trainers now record `act` in the checkpoint; GELU originals carry no key
and every loader must default to gelu), same seed / optimiser / batch as the GELU originals, rule
`--target-train-acc 0.995 --min-train-loss 1e-2 --max-epochs 300`. Training jobs 366155–366158; gate re-measure
(FP64) jobs 366204 / 366207 — duplicate rows from the two gate jobs were removed, one row per checkpoint remains.

| checkpoint | act | epochs to rule | train acc | test acc | train CE | gate |
|---|---|---|---|---|---|---|
| `mnist_mlp_strong_relu.pth` | relu | 8 | 99.773% | 98.36% | 7.05e-3 | **PASS** |
| `mnist_mlp_strong_tanh.pth` | tanh | 13 | 99.710% | 97.71% | 9.22e-3 | **PASS** (loss line by < 8%) |
| `mnist_mlp_d15w1000_relu_full.pth` | relu | 31 | 99.830% | 97.82% | 6.12e-3 | **PASS** |
| `mnist_mlp_d15w1000_tanh_full.pth` | tanh | 66 | 99.832% | 97.81% | 9.52e-3 | **PASS** (loss line by < 5%) |

Step 2 (pending): `deep_stack.inputs_of/forward_deep/run_training_deep` and the `state_dict` loaders in
`trained_backbone.py` / `multilayer_lora.py` hard-code GELU; they must dispatch on `ck.get("act", "gelu")` before any
rank-law or drift row is read from these twins. Until then no harness may load them.
