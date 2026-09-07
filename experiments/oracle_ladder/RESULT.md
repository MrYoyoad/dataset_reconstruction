# How accurate must the chart be? The oracle ladder on the two meeting examples

Two examples, chosen so every figure in the talk is about the same objects: **eight CIFAR-100 motorcycles added to the CIFAR-10 MLP**, and **eight CIFAR-100 keyboards added to the CIFAR-10 CNN**. Head adapter r=64, T=400 SGD steps, k=32, 400 random starts, the same Levenberg-Marquardt solver and the same landing bar (relative image error below 1e-2) in every cell, with nothing tuned between cells.

**The privates are the RAW photographs**, not their chart projections: the adapter is fine-tuned on the photographs and the attack targets the photographs.

## Two things to read before the tables

**`eps` is not the chart's error.** The ladder perturbs each private photograph by relative noise `eps` and then spans the perturbed vectors, so the true photographs do not lie in the resulting chart. What belongs on the axis is the **measured relative projection error of the true photographs** onto each chart, reported beside `eps` in every row, and it is what both figures are drawn against. At `eps = 0` the chart spans the photographs exactly, so that error is **zero by construction** and that cell is effectively on-chart even though the privates are raw — it anchors the axis rather than being a result.

**Only one point on these axes is an attack.** Every oracle chart is built from the private photographs themselves and is **not attacker-available at any `eps`, including 0**; they measure a fidelity ceiling. The public PCA row is the only attacker-available cell, and it is marked on both figures rather than only in a caption. No median or total anywhere pools the two.

**The releases are new, and the private images are not.** These cells are fine-tuned on the raw photographs, while the existing motorcycle and keyboard cells elsewhere in this study are fine-tuned on chart projections. Same eight photographs, same seed, **different release** — the earlier one could not be reused because it was trained on projections. A slide may place a ladder panel beside an on-chart panel, but a reader must not read a difference between them as an effect of the chart alone.

**The two examples differ in backbone as well as class** (MLP against CNN), so if they disagree about where the gate sits, the disagreement is **not attributable to the class** and they are reported separately rather than averaged.

**These tables were regenerated after a bug in this generator, and the numbers moved.** The cell measurements (jobs in `results/oracle_ladder/rows.jsonl`, 28 cells, one job each) were never re-run and are unchanged; what was wrong was the prose the generator derived from them. It computed the first chart that *leaves* the SSIM ceiling and then printed it as the last chart that *tracks* it — off by one row on both examples — and, where no chart recovered all eight, it still placed the public chart '**below** the last error at which every photograph is recovered', a comparison against a threshold that does not exist on that example, in the direction that would suggest the public chart ought to work. Both are fixed and the no-threshold case now says so explicitly. Any earlier copy of this file carrying those two sentences should be discarded rather than reconciled.


## mlp motorcycle  (13 charts)

| chart | eps | measured projection error of the photographs (mean, range) | landings / 400 | images found | residual at the photographs (max) | median start residual | SSIM attack | SSIM ceiling | SSIM control | top-20 all landings |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle (not attacker-available) | 0 | 0.0000  (0.0000–0.0000) | 212/400 | 8/8 | 2.8e-14 | 4.9e-28 | 1.00 | 1.00 | 0.31 | yes |
| oracle (not attacker-available) | 0.01 | 0.0062  (0.0039–0.0107) | 180/400 | 7/8 | 2.8e-14 | 1.2e-07 | 1.00 | 1.00 | 0.31 | yes |
| oracle (not attacker-available) | 0.02 | 0.0124  (0.0079–0.0214) | 71/400 | 3/8 | 2.8e-14 | 4.7e-07 | 0.99 | 0.99 | 0.31 | yes |
| oracle (not attacker-available) | 0.03 | 0.0186  (0.0118–0.0319) | 0/400 | 0/8 | 2.8e-14 | 1.1e-06 | 0.98 | 0.98 | 0.31 | no (0/20) |
| oracle (not attacker-available) | 0.05 | 0.0309  (0.0196–0.0529) | 0/400 | 0/8 | 2.8e-14 | 2.9e-06 | 0.96 | 0.96 | 0.30 | no (0/20) |
| oracle (not attacker-available) | 0.075 | 0.0460  (0.0293–0.0784) | 0/400 | 0/8 | 2.8e-14 | 6.5e-06 | 0.92 | 0.92 | 0.29 | no (0/20) |
| oracle (not attacker-available) | 0.1 | 0.0608  (0.0389–0.1030) | 0/400 | 0/8 | 2.8e-14 | 1.1e-05 | 0.87 | 0.88 | 0.29 | no (0/20) |
| oracle (not attacker-available) | 0.15 | 0.0891  (0.0574–0.1486) | 0/400 | 0/8 | 2.8e-14 | 5.2e-05 | 0.79 | 0.80 | 0.28 | no (0/20) |
| oracle (not attacker-available) | 0.2 | 0.1151  (0.0750–0.1888) | 0/400 | 0/8 | 2.7e-14 | 9.1e-05 | 0.71 | 0.74 | 0.26 | no (0/20) |
| oracle (not attacker-available) | 0.3 | 0.1598  (0.1067–0.2528) | 0/400 | 0/8 | 2.7e-14 | 2.0e-04 | 0.59 | 0.63 | 0.25 | no (0/20) |
| oracle (not attacker-available) | 0.4 | 0.1949  (0.1332–0.2981) | 0/400 | 0/8 | 2.8e-14 | 6.6e-04 | 0.50 | 0.56 | 0.18 | no (0/20) |
| oracle (not attacker-available) | 0.6 | 0.2428  (0.1717–0.3525) | 0/400 | 0/8 | 2.8e-14 | 7.8e-04 | 0.37 | 0.46 | 0.21 | no (0/20) |
| **public PCA (ATTACKER-AVAILABLE)** | — | 0.3176  (0.2241–0.4318) | 0/400 | 0/8 | 2.8e-14 | 2.3e-03 | 0.19 | 0.31 | 0.34 | no (0/20) |
| *wrong-release control, oracle eps 0* | 0 | 0.0000 | **0/400** | **0/8** | 6.4e-01 | 2.0e-03 | 0.17 | 1.00 | 0.28 | no |


- **Where exact landing stops.** All 8 photographs are still recovered at a measured projection error of 0.0000 and none at 0.0186.
- **Where the returned image leaves the ceiling.** The attack tracks the chart's own ceiling to within 0.05 SSIM up to a projection error of 0.1598, and falls below it at 0.1949 (0.50 against a ceiling of 0.56).
- **Where the real public chart sits.** Its projection error is 0.3176, above the last error at which every photograph is recovered (0.0000), and it returns 0 of 400 landings and 0 of 8 images.



![ladder](../../figures/oracle_ladder/ladder_mlp_motorcycle.png)
![curve](../../figures/oracle_ladder/curve_mlp_motorcycle.png)


## cnn keyboard  (13 charts)

| chart | eps | measured projection error of the photographs (mean, range) | landings / 400 | images found | residual at the photographs (max) | median start residual | SSIM attack | SSIM ceiling | SSIM control | top-20 all landings |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle (not attacker-available) | 0 | 0.0000  (0.0000–0.0000) | 224/400 | 7/8 | 5.3e-15 | 1.2e-29 | 0.92 | 1.00 | 0.43 | yes |
| oracle (not attacker-available) | 0.01 | 0.0045  (0.0019–0.0088) | 115/400 | 1/8 | 5.3e-15 | 5.5e-06 | 0.93 | 1.00 | 0.41 | no (0/20) |
| oracle (not attacker-available) | 0.02 | 0.0090  (0.0037–0.0176) | 0/400 | 0/8 | 5.3e-15 | 3.0e-05 | 0.81 | 1.00 | 0.41 | no (0/20) |
| oracle (not attacker-available) | 0.03 | 0.0135  (0.0056–0.0263) | 0/400 | 0/8 | 5.3e-15 | 9.4e-05 | 0.80 | 0.99 | 0.42 | no (0/20) |
| oracle (not attacker-available) | 0.05 | 0.0224  (0.0093–0.0436) | 0/400 | 0/8 | 5.3e-15 | 4.1e-04 | 0.76 | 0.98 | 0.38 | no (0/20) |
| oracle (not attacker-available) | 0.075 | 0.0334  (0.0139–0.0646) | 0/400 | 0/8 | 5.3e-15 | 2.5e-03 | 0.71 | 0.95 | 0.36 | no (0/20) |
| oracle (not attacker-available) | 0.1 | 0.0441  (0.0183–0.0848) | 0/400 | 0/8 | 5.3e-15 | 2.9e-03 | 0.59 | 0.92 | 0.37 | no (0/20) |
| oracle (not attacker-available) | 0.15 | 0.0646  (0.0268–0.1218) | 0/400 | 0/8 | 5.3e-15 | 4.4e-03 | 0.44 | 0.87 | 0.38 | no (0/20) |
| oracle (not attacker-available) | 0.2 | 0.0834  (0.0346–0.1536) | 0/400 | 0/8 | 5.3e-15 | 7.0e-03 | 0.35 | 0.81 | 0.33 | no (0/20) |
| oracle (not attacker-available) | 0.3 | 0.1158  (0.0477–0.2022) | 0/400 | 0/8 | 5.0e-15 | 1.1e-02 | 0.25 | 0.72 | 0.33 | no (0/20) |
| oracle (not attacker-available) | 0.4 | 0.1417  (0.0576–0.2344) | 0/400 | 0/8 | 5.3e-15 | 1.4e-02 | 0.26 | 0.65 | 0.30 | no (0/20) |
| oracle (not attacker-available) | 0.6 | 0.1781  (0.0704–0.2698) | 0/400 | 0/8 | 5.3e-15 | 1.5e-02 | 0.12 | 0.55 | 0.20 | no (0/20) |
| **public PCA (ATTACKER-AVAILABLE)** | — | 0.2432  (0.0883–0.3788) | 0/400 | 0/8 | 5.3e-15 | 1.5e-02 | 0.13 | 0.31 | 0.29 | no (0/20) |
| *wrong-release control, oracle eps 0* | 0 | 0.0000 | **0/400** | **0/8** | 5.0e-01 | 6.8e-03 | 0.32 | 1.00 | 0.38 | no |


- **Where exact landing stops.** No oracle cell on this ladder recovered all 8; the most recovered anywhere is 7 of 8, at projection error 0.0000. This example therefore has no all-8 gate to locate, and none is quoted for it.
- **Where the returned image leaves the ceiling.** There is no tracking region on this example: the attack is already more than 0.05 SSIM below the chart's own ceiling at the lowest-error cell of the ladder (0.92 against 1.00 at projection error 0.0000), so the gap at that end is the search and not the chart.
- **Where the real public chart sits.** Its projection error is 0.2432. No oracle cell here recovered all 8, so there is no all-8 threshold to place it against, and above the largest error at which any start landed at all (0.0045). What is measured is that it returns 0 of 400 landings and 0 of 8 images.



![ladder](../../figures/oracle_ladder/ladder_cnn_keyboard.png)
![curve](../../figures/oracle_ladder/curve_cnn_keyboard.png)


## The two examples do not agree, and are not averaged

**The gate sits at a different place on each.** On the MLP/motorcycle release some start still lands at a measured projection error of 0.0124 and no start lands at 0.0186. On the CNN/keyboard release the last landing is at 0.0045 and none survives to 0.0090 — and even the exactly-spanning chart returns 7 of 8 rather than 8. The last-landing error differs by a factor of 2.8 between the two.

**What that difference cannot be attributed to.** The two cells differ in backbone (MLP against CNN) *and* in class (motorcycle against keyboard) at the same time, so this ladder cannot say which of the two moves the gate. It says only that the gate is not one number across releases. Nothing here is averaged over the two, and a single ladder figure should not be shown as though it were the ladder.

**Both pre-registered predictions held.** The attacker-available public PCA chart returns 0 of 400 on the motorcycle release and 0 of 400 on the keyboard release, at projection errors 0.3176 and 0.2432. The wrong-release control returns nothing even at an exactly spanning chart, which is what says the ladder is measuring the release and not the chart's ability to hold the images.

**`eps` is not proportional to the error it induces.** On the motorcycle ladder the measured projection error is 0.62 of `eps` at `eps`=0.01 and 0.40 of it at `eps`=0.6: the perturbation saturates, so `eps` must not be read as an error axis anywhere.

