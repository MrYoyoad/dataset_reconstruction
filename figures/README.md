# figures/

Plots and visualizations, organized by phase/sprint.

```
figures/
├── thesis_overview_april2026.pdf       Top-level thesis overview diagram
├── phase0/                              Active Phase-0 ViT gradient inversion
│   ├── phase0_full_r8_n1{,_loss,_progress}.png
│   ├── phase0_lora_r8_n1.png
│   ├── phase0_d1_comparison.png        Optimizer × TV-weight grid (4 cells)
│   ├── phase0_d1_cossim_overlay.png    Cosine-sim trajectories
│   ├── d1_panels/                       Per-config detail/loss panels
│   ├── d2_sweep/                        D2 targeted sweep (40 configs)
│   │   ├── phase0_d2_heatmap.png       SSIM grid: tv × lr, panels for 10K/30K iters
│   │   ├── phase0_d2_top_comparison_by_tv.png   GT + best reconstruction at each TV level
│   │   ├── phase0_d2_cossim_overlay_by_tv.png   cos_sim & total-loss curves, one per TV
│   │   ├── d2_<idx>_<config>_recon.png Per-config reconstructions (40 files)
│   │   ├── d2_<idx>_<config>_loss.png  Per-config loss curves (40 files)
│   │   └── snapshots_<config>/          Per-config optimization snapshots
│   └── snapshots/                       Phase-0 single-image optimization snapshots
├── sprint1/                             Sprint 1 NTK reconstruction archive
│   ├── experiment_b_grid_{oracle,free,r32}.png
│   ├── multi_seed_analysis.png         200-seed signal distribution
│   ├── rank_sweep_sprint1.png          SSIM vs LoRA rank
│   ├── sprint1_summary.png
│   ├── t_sweep_examples.pdf            Best recon per T
│   └── phase0_r8_n1.png                (superseded; kept for history)
├── training_dynamics/                   Early parameter-trajectory plots
│   └── parameters_as_function_of_epoch{,_full_fine_tune_comparison,_with_sweet_spot}.png
└── free_c_all_seeds/                    Free-c LoRA per-seed visualizations
    └── seed_{0,1,2,3,42}.png
```

## Where new figures land

| Generator | Output directory |
|-----------|------------------|
| `phase0_vit_inversion.py` (default modes) | `phase0/` |
| `phase0_vit_inversion.py --mode d1` / `phase0_d1_compare.py` | `phase0/` |
| `phase0_vit_inversion.py --mode d2` / `phase0_d2_compare.py` | `phase0/d2_sweep/` |
| `plotting.py` (Sprint 1 NTK) | root `FIGURES_DIR` (callers pass explicit `save_dir`) |
| `gen_t_sweep_examples.py` | root `FIGURES_DIR` |

## Regenerating figures

```bash
# Phase 0 single-image runs
python -m experiments.phase0_vit_inversion --device cuda --mode both

# Phase 0 D1 (optimizer × TV-weight comparison)
python -m experiments.phase0_d1_compare

# Phase 0 D2 sweep comparison
python -m experiments.phase0_d2_compare

# Sprint 1 reconstruction grids
python -m experiments.plotting --input results/exp_b_T1_r8_free_s42_a10000.pth

# T-sweep PDF
python -m experiments.gen_t_sweep_examples --device cuda
```

## Style conventions

See [style_guide/plots.md](../style_guide/plots.md) for DPI, palette, project conventions; [style_guide/guardrails.md](../style_guide/guardrails.md) §T5 for plot font minimums.
