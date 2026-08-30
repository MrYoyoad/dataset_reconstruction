# Reconstructing Private Data from LoRA Adapters

**MSc Thesis, Weizmann Institute of Science**
Advisor: [Gal Vardi](https://scholar.google.co.il/citations?user=LVk3xE4AAAAJ&hl=en)

> Extending Haim et al. (NeurIPS 2022) — *"Reconstructing Training Data from Trained Neural Networks"* — to the era of Foundation Models and Parameter-Efficient Fine-Tuning (PEFT).

**Status:** Active research. Current front: the dataset-sensitivity program + the full-FT-vs-LoRA "valley" comparison; supervisor meeting deck built for 2026-08-31.

> **Posture — observe, don't conclude.** Every leakage number in this work is a **lower bound on the weakest attacker** (prior-free, adapter-only, per-image). It bounds what such an attacker gets; it never bounds what a stronger attacker could recover.

---

## Abstract

LoRA adapters are widely published on platforms like HuggingFace and CivitAI, yet they are a gate-weighted recording of the per-image gradients of their private fine-tuning data. This thesis asks two complementary questions and builds an instrument for each:

1. **What does the adapter *record*?** We treat fine-tuning as a deterministic map `(A_T, B_T) = F(z)` from private latents to the released adapter, and build an **attack-independent identifiability ruler** — the whitened end-to-end Jacobian `J = ∂(adapter)/∂(data)`. From it we read the hard rank `r_J` (how many private directions are recorded at all), the whitened sensitivity `d²` (= 2·KL = optimal-detector SNR², the best any attacker can do), and `q_eff(ε)` (how many directions clear the training-noise floor). This measures leakage *before* any reconstruction, and every number is calibrated against the training-randomness null.

2. **Can it be turned back into pixels?** Reconstruction has two regimes: the **full-gradient ceiling** (known-recipe upper bound) already returns recognizable images across MNIST / Fashion / CIFAR-10 / Flowers and structure on ViT-B/16 faces; **robust adapter-only pixel inversion is the open milestone** — the information is present (q_eff is high), so this is an *extraction* gap, not a missing-information one.

Every negative reconstruction is placed in one of **three worlds** — (A) identifiability wall, (B) extraction-limited, (C) prior hallucination — so each experiment states which reality it rules in or out.

---

## Research Questions

- **What decides whether private images survive fine-tuning into the adapter?** The rank and conditioning of the gate-weighted update `Ω = G·Xᵀ`, `G = D_v·M·D_c` — the activation enters *only* through the gate `M_ki = σ′(⟨w_k, x_i⟩)`.
- **Which images leak, and can an attacker predict it in advance?** Per-image leakage is predicted from the public base model alone by the base-gradient-norm `g₀` (ρ = +0.857 at n=12; +0.777 at n=24, graded indeterminate).
- **Does the dataset *around* an image change its leakage?** Largely no — dilution is flat in N, duplication sub-linear, context rarity ≈ nothing; what matters is the image itself (class identity, base gradient).
- **Does full fine-tuning remember more sharply than LoRA?** It records ~5× more signal per image but at about the same resolution (target-dependent, an approximate wash).
- **Does the adapter betray *what* it was trained on?** Which digit-subset was present is recoverable from ΔW above a recipe-aware baseline (cross-fitted); which specific exemplar is the open instance-level question.

---

## The instrument and the seven experiments

One object — the gate-weighted update — generates every experiment; each is read through the same whitened ruler.

| # | Experiment | What it probes | Status |
|---|------------|----------------|--------|
| E1 | Controlled secret | per-direction recovery crosses 1 exactly at `ε·ν_i ≈ 1` | toy confirmed, scale-up open |
| E2 | (N, r, L) phase diagram | leakage boundary is spectral, not a rank count; multi-class "leaks fewer" is a low-rank effect (gap 23→13→0 at r=8/16/32) | rank slice done |
| E3 | Activation crux (advisor's top ask) | activation enters only via σ′; kinked leaks ~5× smooth, yet smooth linearizes best — a clean dissociation | MNIST done |
| E4 | Who leaks — the g₀ predictor | per-image leakage predictable from the public model | strong at n=12, indeterminate at n=24 |
| E5 | Full-FT vs LoRA — the valley | more signal, ~same resolution (geomean ratio 1.02, target-dependent) | n=6, exploratory |
| E6 | Composition atlas | which digit-subset is recoverable from ΔW above the recipe baseline (content-level) | positive (scoped); instance-level open |
| E7 | Robust adapter-only inversion | turn presence into pixels | the open milestone (World B) |

---

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/MrYoyoad/dataset_reconstruction.git yoado
cd yoado
conda env create -f dataset_reconstruction/environment_macos.yaml
conda activate rec

# The identifiability ruler (whitened secret-swap sensitivity)
python -m experiments.dataset_sensitivity.whitened_metric --help

# The reconstruction attack (free-coefficient NTK, targets ΔW)
python -m experiments.run_experiment_b \
  --rank 8 --n_steps 1 --free_coefficients \
  --consistency_weight 1.0 --optimizer sgd
```

All serious compute runs on the WEXAC GPU cluster — see [scripts/README.md](scripts/README.md). Results save to `results/` as CSV + `.pth` tensors (the bulk tensors are git-ignored; commit the code, docs, and curated figures).

---

## Key Results (honest, current)

| Finding | Value | Job |
|---------|-------|-----|
| **Full-gradient ceiling** (known-recipe upper bound) | SSIM up to ~0.99 (MNIST/Fashion/CIFAR/Flowers); ViT faces return structure | 956994 et al. |
| **Direct weight inversion** | recognizable at N=4 (SSIM ~0.57), superposes by N=10 (~0.27) | 500913 / 887704 |
| **Activation crux** | kinked ≈ 5× smooth (control-margin 0.47 vs 0.09); Spearman(feature-stability, leakage) ≈ 0 | 392821 / 390026 |
| **g₀ predictor** | ρ(sensitivity, g₀) = +0.857 (n=12) / +0.777 (n=24, CI [0.53, 0.91]) | 260171 / 272504 |
| **Full-FT vs LoRA valley** | ~5× more signal per image, valley-width ratio geomean 1.02 / median 0.86 (target-dependent) | 695782 |
| **Composition atlas** | ΔW clusters by composition (ARI +1.00); cross-fit recovery +0.989 above recipe baseline (which digit-subset) | 838868 |
| **Rank-sweep reversal** | multi-class "leaks fewer" gap 23→13→0 at r=8/16/32 — a low-rank effect | 581629 |

**Method note (retraction, kept honest):** an earlier framing reported adapter-only reconstruction at "SSIM ~0.557 proving leakage." That used a mean/std-matched SSIM (`ssim_norm`) that inflates the absolute and never beat the trivial mean-image baseline — it is **retracted**. The defensible reconstruction claims are the full-gradient *ceiling* (which works) and the identifiability ruler (which shows the information is present); robust *adapter-only* pixel inversion remains open. See the deck appendix "what we retracted" and [notes/thesis_note_v2.md](notes/thesis_note_v2.md).

---

## Directory Structure

```
yoado/
├── README.md                      <- This file
├── STATUS.md                      <- Progress, landed results, pending tasks (start here for "what's the state")
├── LESSONS_LEARNED.md             <- Running log of insights and pitfalls
├── STYLE_GUIDE.md  style_guide/   <- Formatting rules for docs, slides (pptx.md), LaTeX, plots + visual guardrails
│
├── experiments/                   <- Thesis experiment code
│   ├── run_experiment_b.py        <- Free-coefficient NTK reconstruction (targets ΔW)
│   ├── ntk_extraction.py          <- Core reconstruction algorithm
│   ├── direct_inversion.py        <- Direct weight inversion (autograd through unrolled SGD)
│   ├── ntk_verification.py        <- Feature-stability / function-space linearization checks
│   ├── phase0_vit_inversion.py    <- ViT-B/16 gradient inversion + face-structure prior
│   ├── gradient_bridge/           <- LoRA -> full-gradient decoder (GB-Phase 1)
│   ├── dataset_sensitivity/       <- The identifiability ruler + the sensitivity program:
│   │   ├── whitened_metric.py     <-   3-way cross-fit whitened secret-swap sensitivity d²
│   │   ├── jacobian_spectrum.py   <-   J = ∂(adapter)/∂(data); r_J, q_eff on col(J)
│   │   ├── arm_b_dilution.py …    <-   arms B (dilution) / C (class imbalance) / D (context) / E (duplication)
│   │   ├── margin_vs_sensitivity.py<-  the g₀ who-leaks predictor
│   │   ├── fullft_valley.py       <-   full-FT vs LoRA valley-width comparison
│   │   ├── atlas_zoo.py / atlas_analyze.py <- composition atlas (which-digit; --same_digits = instance-level)
│   │   └── atlas_ecosystem.py / eco_*.py   <- ecosystem common-mode-subtraction prototype
│   └── tests/                     <- pytest suite
│
├── scripts/                       <- WEXAC (LSF bsub) job submission scripts
│   └── deck/                      <- Modular python-pptx generator for the supervisor deck
├── notes/                         <- Plans + theoretical analyses (thesis_note_v2.md, thesis_scientific_summary.md, …)
├── docs/                          <- Guides, session handovers, audit reports
├── results/                       <- CSV metrics (tracked) + .pth tensors (git-ignored, large)
├── figures/                       <- Plots and visualizations
├── papers/                        <- Reference PDFs
│
└── dataset_reconstruction/        <- Original Haim et al. codebase (separate git)
    ├── Main.py                    <- Base code entry point (train / reconstruct)
    ├── extraction.py              <- KKT loss optimization
    └── CreateModel.py             <- MLP with ModifiedReLU
```

---

## Installation

### Local Development (Mac)

```bash
cd dataset_reconstruction
conda env create -f environment_macos.yaml   # Apple Silicon (MPS backend)
conda activate rec
```

### WEXAC GPU Cluster

All serious experiments run on WEXAC (NVIDIA L40S/A100, CUDA 12.x); the `rec` env there runs PyTorch 2.4.1+cu121, timm 0.9.12, peft 0.7.1.

```bash
cd dataset_reconstruction && ./wexac_connect.sh shell   # interactive GPU shell
bsub < scripts/run_exp_b_gpu.sh                          # or submit a batch job
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [STATUS.md](STATUS.md) | Current state: landed results, pending tasks, known issues |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md) | Running log of insights and pitfalls |
| [notes/thesis_note_v2.md](notes/thesis_note_v2.md) | The mechanism, the ruler, and where each experiment stands |
| [notes/thesis_scientific_summary.md](notes/thesis_scientific_summary.md) | Consolidated science summary |
| [notes/next_experiment_plan.md](notes/next_experiment_plan.md) | Single source of actionable to-do |
| [STYLE_GUIDE.md](STYLE_GUIDE.md) / [style_guide/](style_guide/) | Formatting rules + visual guardrails for docs, slides, plots |

---

## Citation

This thesis builds on:

```bib
@inproceedings{haim2022reconstructing,
  author = {Haim, Niv and Vardi, Gal and Yehudai, Gilad and Shamir, Ohad and Irani, Michal},
  booktitle = {Advances in Neural Information Processing Systems},
  title = {Reconstructing Training Data From Trained Neural Networks},
  volume = {35},
  pages = {22911--22924},
  year = {2022}
}
```

Key external anchors: Jang et al. (ICML 2024, LoRA NTK, r ≳ √N); Putterman/Lim et al. (ICLR 2025, Learning on LoRAs); Tian et al. (ICLR 2025, SimuDy).

---

## License

Research use only. Based on the [Haim et al. implementation](https://github.com/nivha/dataset_reconstruction).
