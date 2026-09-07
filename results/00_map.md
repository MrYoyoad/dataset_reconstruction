# Orientation map — what exists on disk, before the first job of the 7 Sept brief

Written by the executor session before any compute, per §12.0. Paths, job ids, and which release is which. **Where
the disk contradicts the brief, the disk wins and the discrepancy is logged here.**

## 1. Core code (located, not invented)

| what | path |
|---|---|
| replay / exact inversion (SGD and Adam modes, `train_release`, `simulate_sgd_reduced`, `invert_lm`) | `experiments/exact_inversion/lora_exact_inversion.py` |
| certificate `C = Pi A_T`, and `lm_cert` (the certificate-route solver) | `experiments/exact_inversion/certificate.py` |
| the 19/60 replay cell, both routes on one release | `experiments/exact_inversion/affine_chart_two_routes.py` |
| head-adapter cells on real backbones (CIFAR-10 MLP / CNN, added classes) | `experiments/cifar/cifar_newclass.py` |
| chart x layer x solver grid, isolation test, blend diagnostic | `experiments/cifar/cifar_charts.py` |
| linearised-representer vs certificate head-to-head, variable projection | `experiments/cifar/ntk_vs_certificate.py` |
| oracle-chart ladder (E8-adjacent, running) | `experiments/oracle_ladder/ladder_cell.py` |

## 2. The releases

| release | where | shape | fp | q vs N | use |
|---|---|---|---|---|---|
| **affine two-routes** — the 19/60 replay result | row `results/exact_inversion/affine_two_routes_331384.jsonl`; generator `affine_chart_two_routes.py` | k=12, N=8, r=24, m=20, d=P=64, T=400, lr=0.05, seed=1 | **fp64** | rank C = 16, r = 24 -> **q = 8 = N** | E1B-tiny (§12 names it) |
| head-adapter cells on real backbones | `results/cifar_newclass/*.pth` (35 cells) | N=8, r=64, k=32/48, T=400, d=1000 (MLP) or 256 (CNN) | fp64 | q = N = 8 (rank B_T = 8 in every row) | the cells where Phi0 is a real network |
| oracle ladder, raw privates | `results/oracle_ladder/` (28 cells, **complete**) | N=8, r=64, k=32, T=400 | fp64 | reported per cell | E8-adjacent |
| precision-varied letter releases | `results/exact_inversion/step80_760909/` | r=64, k=32, N=8 | fp64/fp32/bf16/fp16 | q = 8 | precision caveats (rule 9: not Gal-facing) |

`results/exact_inversion/` holds 115 `.pth` releases; the fp64 lab releases with known seeds are there
(`spectrum_sgd_*`, `step11_sgd_*`, ...).

## 3. DISCREPANCIES — the disk wins, logged per §12.0

**D1. The release §12 names for E1B makes three of its own day-one measurements vacuous.** In
`affine_chart_two_routes.py` the world is `phi = identity`: the adapted layer *is* the input layer. Consequences,
all structural rather than incidental:

- **The manifold test is vacuous.** A5 asks for `min_x ||Phi0(x) - h_hat||` beside the chart residual. With
  `Phi0 = id`, `range(Phi0) = R^d`, so *every* candidate is a real feature and the residual is identically zero.
  The zero-set question of §9 (`Z_image <-> Z_feature ∩ range(Phi0)^N`) cannot be asked on this release at all.
- **The image-space comparison is vacuous.** §12 asks for feature replay against image replay on matched starts.
  With `Phi0 = id` the two are the same computation, so there is no cost saving to measure and no basin difference.
- **`k = 128` exceeds `d = 64`.** The chart-distance measure at k in {32, 64, 128} degenerates: at k >= 64 a PCA
  chart of a 64-dimensional space is the whole space and the residual is zero by construction.

**Resolution, and it changes no instruction:** this release still answers the question E1B exists to answer — *can
replay recover H\* when H is free* — and it is the release §12 names, is fp64, and satisfies A1's `q = N`
precondition. It is run as specified. The three vacuous measurements are reported as vacuous with the reason,
**not** as zeros, and the manifold and cost questions are additionally run on a **real-backbone** release from
`results/cifar_newclass/` where `Phi0` is an actual frozen network (d = 1000 for the MLP, 256 for the CNN). That
addition is flagged as an addition, not as the named cell.

**D2. E5 is already done, and the brief tags it `new`.** Job 331384 matches its spec clause for clause: linear
chart, `k = 12` below both capacity lines and above `N-1`, certificate landings **0 of 60** with every returned
point an exact affine combination (coefficient sum 1.000000000 at every one of 60 starts, max deviation 8.9e-16), isolation rank 5 = `k - (N-1)`,
and replay on the same release and the same starts recovering all eight from **19 of 60** with **zero aliases**.
Already noted in the plan audit as A8. Retag `done`; do not re-run.

**D3. The "20 attacker-buildable starts" are referenced in `experiments/exact_inversion/constrained_replay.py`,
`multilayer_budget.py`, `train_precision.py` and `RESULTS.md`**, not in a single artefact file. E2 will have to
regenerate them from the constrained-replay path rather than load them; noted so it is not discovered mid-experiment.

## 4. Day-one status

| item | status |
|---|---|
| 00_map.md (this file) | done, before the first job |
| E1B-tiny | pre-check first (A2 scale symmetry, no solve), then the two arms |
| E4a | DINO-ViT-B/16 and CLIP ViT-L/14 embeddings, public subjects disjoint from privates |
| E6 request | drafted, **not sent** |
| oracle ladder (E8-adjacent) | **all 28 cells COMPLETE**, jobs 435271-435322; rows in `results/oracle_ladder/rows.jsonl` |

## 5. Standing constraints carried from the brief

Rule 1: report **q and N side by side**, always; `q = r - rank C` is a span dimension and never an image count.
Rule 2: never present a raw equation count as identifiability. Rule 10 language, and A3's required continuation —
"not ruled out by count" is followed in the same sentence by *identifiability additionally requires the chart to
meet span(H) only at the training points (Lemma 15); the count does not test that*. Rule 9: precision results stay
out of Gal-facing material. Rule 13: self-trained releases only.
