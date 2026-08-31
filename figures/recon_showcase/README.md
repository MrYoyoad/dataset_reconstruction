# Free-coefficient reconstruction showcase — LoRA vs full fine-tune (2026-08-31)

Realistic attack only: **free coefficients** (the attacker does not know the per-image loss coefficients);
oracle-coefficient runs appear nowhere. Non-trivial training length: T ∈ {5, 10, 20} SGD steps (T=1 panels are
kept as `freec_T1_*` for reference only and are NOT part of the deliverable).

Setup: N = 2 private images, one fine-tuned MLP (Haim-style), seed 42. MNIST: leaky-ReLU fine-tune (the leaking,
kinked activation), extraction with L-BFGS; LoRA = rank-r adapter on the first layer; "full fine-tune" = all first-layer
weights. Flowers-102 (32 px, RGB): ReLU-extraction recipe (relu_alpha 10000, SGD). Per (dataset, T, rank) the best
learning-rate cell of the sweep is shown, chosen by: (1) raw SSIM must beat the dataset-mean image baseline, then
(2) largest control-margin. Every sweep cell with all stored metrics: `../../results/recon_showcase_sweep.csv`.

Metrics (experiments/metrics.py, experiments/recompute_metrics.py):
- **ssim** — raw SSIM (window 3) between reconstruction and its private image.
- **ssim_norm** — SSIM after matching the reconstruction's mean/std to the target (structure only).
- **margin** = ssim_norm(recon vs private image) − ssim_norm(recon vs a same-class, different-sample control image):
  the project's clip-robust leakage proxy. Margin ≈ 0 means the reconstruction is no closer to the private image than
  to any image of that class.
- **baseline gate** — raw SSIM vs the trivial predictor (dataset-mean image); a cell at/below it carries no
  instance-specific information. Verdict printed in each figure's footer.
- **clip** — fraction of reconstruction pixels outside [0,1] before clamping (~40% on MNIST; benign, see LESSONS_LEARNED).
- Under each tile: raw ssim / ssim_norm of that reconstruction vs that tile's image (control row: the reconstruction
  scored against the control image — the margin's reference).

Flowers T=5/10/20 grids are near-identical: on flowers the free-c recovery is nearly T-independent in this range (kept as the T-robustness evidence).
Honesty notes: N=2 makes the dataset-mean baseline nearly an image itself (esp. flowers, baseline 0.65), so flowers
cells that miss it by 0.01 are shown with the failure printed rather than hidden. All numbers bound this attacker
(free-coefficient NTK extraction from the released first-layer weight change), not the reconstruction limit.
Jobs: 323866 / 323867 / 336206 / 341742 (scripts/run_freec_showcase_T*_wexac.sh). Builder: scripts/deck/make_recon_showcase.py.
