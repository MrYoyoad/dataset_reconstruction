# Perceptual identification tier (`experiments/utils/perceptual_id.py`)

Two tiers of "recovered":

| tier | criterion | where it comes from |
|---|---|---|
| **1. exact landing** | relative pixel error `‖x̂ − x‖ / ‖x‖ < 1e-2` | the ladder / bootstrap / WP2 landing gate (`LAND`, `RECOVER_TOL`) |
| **2. line-up identification** | the recovery picks the private image out of a line-up of 100 same-class public images (rank 1 = top-1, rank ≤ 5 = top-5) | this module |

Tier 2 asks the question a human would ask of a reconstruction: is this identifiably *the* private image, or only a
generic member of its class? It is strictly weaker than tier 1 (a landed image is rank 1 trivially) and it is a
*lower bound on identifiability*, not the reconstruction limit: a recovery ranked outside the top-5 by SSIM may still be
identified by a stronger judge.

## Line-up protocol

* **Candidates**: the truth plus `N_DECOYS = 99` public images of the same class, drawn as the first 99 of
  `numpy.random.default_rng(DECOY_SEED=0).permutation(pool)`; the same 99 for every image of that class in every cell.
* **Pools are the PUBLIC train splits only** — EMNIST letters (`new_class.load_emnist_letters`, train), MNIST digits
  (`trained_backbone.read_idx`, train, per label), CIFAR-100 classes (`cifar_newclass.load_cifar100_class`, train). The
  privates come from the test splits in every source, so a decoy is never the private image.
* **Scores of a recovery against each candidate**: SSIM (`kornia.metrics.ssim`, window 3, images clamped to [0,1] —
  the `common_utils/image.py` convention), pixel L2, and L2 / cosine in the frozen public base's penultimate features
  (`TrainedBackbone.phi` for `mnist_mlp_strong*`, `deep_stack.inputs_of(.)[-1]` for `mnist_mlp_d15w1000`, `CNN.phi`
  (256-d) / `MLP.phi` (1000-d) for the CIFAR bases; chosen by the checkpoint path stored in the cell).
* **Rank** of the truth = 1 + number of decoys scoring strictly better; `top1_*`, `top5_*` flags per score.
* **Control** (the existing convention): the nearest public image of the class to the truth (pixel L2, train split).
  `ssim_control = SSIM(recovery, control)`; `ssim_truth_vs_control = SSIM(truth, control)` is what a look-alike scores.
* **Exact-landing flag** recomputed against the same reference image and carried in every row.

## Sources and what is scored

Formats were read off the savers, not assumed. Arms with `attacker_output = False` are references, not attacks.

| source | files | arms | reference |
|---|---|---|---|
| `oracle_ladder` | `results/oracle_ladder/*.pth` (`ladder_cell.py`) | `cert_best` (attack); `chart_projection` (ceiling) | raw truth |
| `ntk_vs_cert` | `results/ntk_vs_cert/<tag>_<chart>_k<k>_T<T>.pth` (`ntk_vs_certificate.py`; aggregates with `panels` are unpacked) | `cert_best`, `ntk_best` (attacks), each scored **twice**: vs the raw truth and vs the on-chart target; `control_public_nn`, `chart_projection` (refs) | raw / on-chart |
| `bootstrap_chart` | `results/bootstrap_chart/<domain>_<variant>_round<t>_<arm>_<job>.pth` (`bootstrap.py`) | round 0: `matched_candidate` (attacker selection), `best_any_start` (oracle selection, ref); A: `matched_candidate`; B: `matched_slot`; `chart_optimum_oracle_start`, `chart_projection` (refs) | raw truth |
| `decoder_chart` | `results/decoder_chart/fidelity_*.pth` (`fidelity.py`, `(N, D)` rows, fp32) | every `x_*` set: `pixpca_k*`, `glob_s*_k*`, `local_K*_k*_proxy_nn` (attacker-available charts); `ae_*`, `*_truth_nn`, `*_truth_latent` (refs, anchored on the private image) | raw truth |

For WP2 the on-chart target line-up keeps the raw public decoys; only the reference image changes (the chart's
projection of the truth). The chart's own projection is scored as `chart_projection` in every source so that "the chart
already lost the identity" is separable from "the attack lost it".

## Outputs

* `results/perceptual_id/<source>_<jobid>.jsonl` — one row per (cell, arm, target, image): the tier fields above,
  the cell's `meta` (chart, eps / k / T, measured chart error, landings recorded by the producer), the checkpoint.
* `results/perceptual_id/summary_<jobid>.md` / `.json` — per (cell, arm, target): n, exact landed, SSIM top-1, SSIM top-5,
  L2 top-1, feature top-1 / top-5, median SSIM to truth, median SSIM to control, chart error; a **Sources seen** block
  listing which sources were absent and which producer jobs (`ol_*`, `nvc_*`, `bsc_*`, `dc_*`) were still running,
  so the sweep is rerun later with the same command; and per ladder example a table of identification vs measured chart
  error with the "0 exact landings but still SSIM top-1" line read off automatically.
* `figures/perceptual_id/<source>_<cell>_<arm>_<target>.png` (with `--figures`) — per image: truth | recovery |
  control | the three decoys most similar to the recovery, with SSIM and rank.

## Running

```bash
bash scripts/run_perceptual_id_wexac.sh smoke   # letters ladder: mlp_letter_a eps 0 / 0.03 / 0.10 + pca (queue short, CPU)
bash scripts/run_perceptual_id_wexac.sh full    # everything on disk now; rerun later with the same command
python -m experiments.utils.perceptual_id results/oracle_ladder/mlp_letter_a_eps0.03.pth --figure   # one cell (inside a job)
```

## Reading the numbers

* `exact landed` counts tier 1; `SSIM top-1` counts tier 2 by SSIM; the gap between them is the regime where the
  recovery is no longer pixel-exact but still identifies the private image.
* A `chart_projection` row with SSIM top-1 below 8/8 means the chart itself does not contain an identifiable version
  of the truth at that error — the attack cannot do better than that in that chart.
* Median SSIM to truth vs median SSIM to control: if they coincide the recovery is a class look-alike; the line-up
  rank is the sharper statement of the same thing.
* The decoy line-up is a fixed random sample of the public class; a near-duplicate of the truth in the public pool would
  make the truth's rank 2 without any failure of the attack. `ssim_best_decoy` and `control_pool_index` are stored so such
  cases can be inspected.

## Known skips

* `results/ntk_vs_cert/{cifar_apple,cifar_keyboard,cifar_keyboard+apple,mnist_letter_a,mnist_letter_a+letter_t}_k32_N8_r64.pth`
  and the `*_N1_r64.pth` files (2026-09-06) were written by `experiments/cifar/cifar_certificate_onchart.py`
  (keys `A0, A_T, B_T, C, meta, results, x_chart, x_raw`), not by `ntk_vs_certificate.py`; they carry no per-image
  recovery tensor in the WP2 layout and are logged as SKIP with their keys.
* `results/bootstrap_chart/clf_*.pth` are the public classifier caches (no images) and are excluded by default.

## Caveat read off the first sweep (job 356492)

On the **oracle ladder** the chart is spanned by perturbed copies of the private images, so its projection of the
truth keeps the identity by construction at every eps: `chart_projection` is SSIM top-1 8/8 up to eps 0.6 on
`mlp_letter_a` / `mlp_motorcycle`, and the **wrong-release controls** (oracle chart, release trained on eight OTHER
images) still reach 4/8 (letters), 5/8 (keyboard), 6-7/8 (d15) SSIM top-1 with 0 landings. Tier 2 on an oracle cell
therefore measures "the attack reached the chart's identifiable point", not attacker skill; the attacker-relevant
tier-2 numbers are the `pca` ladder cells, the WP2 (`ntk_vs_cert`) cells, the bootstrap rounds and the
attacker-available decoder charts. The feature-space rank is the more discriminating judge on the wrong-release
controls (feat top-1 0-1/8 where SSIM top-1 is 4-7/8) because the recovered image is a same-class look-alike in pixels
but not in the base model's features.
