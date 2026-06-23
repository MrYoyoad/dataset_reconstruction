# SSIM convention used in v6 figures

All SSIM values reported in the v6 deck use the **same canonical setting** used
throughout the project, defined in `dataset_reconstruction/common_utils/image.py`:

```python
from kornia import metrics

def get_ssim_pairs_kornia(x, y):
    return metrics.ssim(x, y, window_size=3).reshape(x.shape[0], -1).mean(dim=1)
```

i.e. **`kornia.metrics.ssim(x, y, window_size=3)`**, then mean over the per-pixel
SSIM map.

## Why `window_size=3` (and not 11 or some other default)?

This is the SSIM variant that all stored `.pth` metrics were computed with, so
v6 figures stay consistent with:

- Numbers in `slide 22` (Backup — headline numbers) of the deck
- The brief's reference values (e.g. *face1 ≈ 0.52*, *slide-5 Δ ≈ +0.06*)
- The original Sprint 1 figure captions

Switching to `window_size=11` (a more common default in some libraries) gives
**systematically lower SSIMs** — face1 drops from 0.52 → 0.42 on the same
reconstruction. The bigger window smooths over local structure, penalises
noisy-but-locally-similar reconstructions, and is less informative for
small-image cases like MNIST 28×28.

`window_size=3` is the **"good but realistic" setting** for this project:
it captures local pixel agreement, doesn't punish noise away from edges,
and matches the project's existing reference numbers.

## What the SSIM number means here

- Input range: tensors are clamped to `[0, 1]` before SSIM (we add back the
  dataset mean if it was subtracted).
- SSIM is computed *per pixel*, then averaged → one scalar per image pair.
- Range: 0 (no similarity) to 1 (identical). Values around 0.3 are the
  "recognizability gate" used in the deck (`SSIM = 0.30` dashed grey line on
  any quality axis).

## ⚠ Bug #2: face / ViT pipeline (slide 13)

Face tensors are stored in **ImageNet-normalized space** (range ~[-2.1, 2.3]),
not [0, 1]. The naïve `tensor.clamp(0, 1)` zeros most of the pixel content
before SSIM and gives artificially low SSIM.

**Correct pipeline (matches `experiments/phase0_vit_inversion.py::compute_metrics`):**
```python
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
x_pixel = (x * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
ssim = kornia.metrics.ssim(x_recon_pixel, x_target_pixel, window_size=3)
```

With this fix:
- face1: **0.5217** (matches stored 0.5217)
- face2: **0.6663** (matches stored 0.6663)
- face3: **0.5815** (matches stored 0.5815)

The earlier v6 values (0.50 / 0.43 / 0.56) were depressed because of the
clamp-without-denorm bug. The bug-fix script is `scripts/figures/v6_faces_fix.py`.

## ⚠ Bug #1: MNIST pipeline (slides 5, 7, 10)

The project helper `compute_ssim(x_recon, x_target, ds_mean)` **adds
`ds_mean` to both arguments unconditionally** and then clamps. But ground-truth
tensors (e.g. `x_train`, `x_ctrl`) are usually already stored in `[0, 1]`.
Adding `ds_mean` pushes them above 1, then `.clamp(0, 1)` damages the image
and depresses SSIM by ~0.10–0.17.

**Fix (used by `scripts/figures/v6_refix.py`):** add `ds_mean` only to
tensors that look mean-subtracted (range allows negatives or max < 0.5).
Reconstructions need `ds_mean` added; ground truth and same-class controls
do not.

After this fix, the project's stored `.pth` metrics (which were computed at
run time with the right convention) match the fresh recompute within ~0.005.

## Where v6 numbers come from

- v6 figures use `scripts/figures/v6_refix.py` for slides 5, 7, 10, which
  applies the fix above and produces SSIMs that agree with the stored
  values to within ~0.005.
- The 50-seed multi-seed CSV (`results/multiseed_freec_vs_oracle_*.csv`) was
  generated at run time using the buggy `compute_ssim` path, so its values
  are systematically depressed by ~0.10. The relative story (recon mean
  above both floors) is preserved because all 50 seeds + the controls in v6
  share the same pipeline.

## Negative controls (slides 5 and 10)

Two control concepts are used in v6:

1. **Same-class control** — mean SSIM of the LoRA reconstruction against
   20 random *same-class* test-set instances (e.g. 20 other digit-5s for a
   Sample-1-is-5 case). This is the "class-identity floor": how close does
   our recon look to *any* digit-5 by chance? Used as the dashed orange line
   on slide 10.

2. **Cross-class baseline** — mean SSIM of the LoRA reconstruction against
   20 random *different-class* instances. The "random-image floor".
   Used as the dashed red line on slide 10.

Both are averaged over 20 samples (`np.random.RandomState(2026)`) to avoid
single-instance lucky-look noise. Before this stabilization, slide 5's
Sample 2 control happened to land at 0.789 because the chosen random "0"
looked nearly identical to the GT "0" by coincidence; now it averages
to ~0.29 across 20 different "0"s.

## Script that produces all this

`scripts/figures/make_v6_figures.py` — main figure pipeline.
`scripts/figures/v6_controls.py` — recomputes the negative controls (n=20)
and re-renders slides 5 and 10.

Both are deterministic given the same input tensors + seed=2026.
