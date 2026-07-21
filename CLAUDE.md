# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository is a **thesis project** extending the work of Haim et al. (NeurIPS 2022, Oral) — *"Reconstructing Training Data from Trained Neural Networks"* — into the era of Foundation Models and Parameter-Efficient Fine-Tuning (PEFT).

### Base Implementation (dataset_reconstruction/)

The `dataset_reconstruction/` subdirectory contains the original PyTorch implementation. It reconstructs training samples from a trained neural network's weights by optimizing inputs that satisfy the KKT optimality conditions of the implicit max-margin problem. All commands below should be run from that directory.

### Thesis Research Direction

The thesis focuses on the **image domain**, extending the Haim et al. reconstruction paradigm to Foundation Models and PEFT:

> **Primary axis (added 2026-05-14, after the first supervision meeting): Direct Weight Inversion.**
> Treat fine-tuning as a deterministic differentiable map `θ_T = F(θ₀, {x_i})` and recover the private
> samples by minimizing `‖θ_T − F(θ₀, {x̂_i})‖²` (autograd backprops through F to the candidate data).
> This is **complementary to — not a replacement for — the Gradient Bridge** below: direct inversion is
> the leakage *upper bound* under best-case (known-recipe) knowledge; the Gradient Bridge characterizes
> how leakage *degrades* under realistic, weaker assumptions. Status: **proposed** (generalizes the
> Approach-G / S3.4 sketch). Actionable plan: [notes/experiment_plan.md](notes/experiment_plan.md);
> rationale + attack taxonomy: [notes/unified_direction_analysis.md](notes/unified_direction_analysis.md)
> ("Direct Weight Inversion — New Primary Axis"); full record: [notes/thesis_update_briefing.md](notes/thesis_update_briefing.md).

The three PEFT-reconstruction directions (the Gradient Bridge axis + its theoretical underpinnings):

1. **LoRA Reconstruction via the "Gradient Bridge"**: LoRA adapters (A, B matrices) are structured, compressed recordings of cumulative training gradients. The LoRA update ΔW = BA is a low-rank projection of the full gradient that can be inverted. A learned **Gradient Decoder** (trained on proxy data) approximates the inverse projection, recovering full-dimensional gradients from low-rank adapters, which are then fed into gradient inversion pipelines (GradInversion for vision).

2. **LoRA in the NTK Regime**: When LoRA rank r is sufficiently large (r ≳ N, where N = number of data points), LoRA optimization in the NTK regime converges to the same global minimum as full fine-tuning (Jang et al., 2024). This means LoRA weights BA encode the same support vector geometry as full weights — the KKT stationarity condition adapts to a coupled system where B^T(Σ λ_i y_i ∇_W Φ(θ; x_i)) = 0 and the analogous condition for A.

3. **Generative Priors (Diffusion-Based)**: When LoRA rank is very low, a hybrid loss combines KKT residuals with Score Distillation Sampling (SDS) from a frozen diffusion model to hallucinate missing high-frequency details. Targets face reconstruction from Stable Diffusion / ViT LoRA adapters.

### Key Theoretical Foundations

- **Implicit bias of GD**: For homogeneous networks trained with BCE, gradient flow converges to KKT conditions of the max-margin problem: θ* ∝ Σ λ_i y_i ∇_θ Φ(θ*; x_i). Weights encode "support vectors" — the hardest training examples.
- **LoRA as gradient projection**: ∇_A L = B^T ∇_W L and ∇_B L = (∇_W L)A^T. The adapter weights are not arbitrary compression artifacts; they are high-fidelity measurements of the principal components of the full gradient.
- **Information density argument**: While LoRA adapters have fewer parameters than raw pixel count (e.g., 294K LoRA params vs. N×150K pixels for ViT-B/16), natural images have far lower intrinsic dimensionality than their pixel count — MNIST digits ~10-20 dims, faces ~100-1000 dims, not 150K. In terms of intrinsic degrees of freedom, the system is well-constrained even with LoRA-only for realistic N. With the full model gradient (86M params), the system is massively overdetermined (>100×) even in raw pixel space. The Gradient Bridge decoder recovers this full gradient from LoRA, further strengthening the attack.
- **R2F (Recover-to-Forget)**: Demonstrates that full-model gradients can be reconstructed from LoRA updates using a decoder trained on a proxy model — proof that the LoRA-to-gradient bridge is learnable. See dedicated section below.

### Recover-to-Forget (R2F) — Core Reference

**Paper:** "Recover-to-Forget: Gradient Reconstruction from LoRA for Efficient LLM Unlearning" (Liu et al., Dec 2025). This is the single most important external reference for the thesis. The paper's original purpose is machine unlearning; our thesis pivots the same mechanism to **attack** (reconstruct) private training data.

**Central Idea:** LoRA updates ΔW = BA are low-rank projections of the full gradient ∇_W L. R2F trains a small "Gradient Decoder" network f_φ that learns the inverse map: given a low-rank adapter update, output the approximate full-rank gradient. Once full gradients are recovered, they can be used downstream (for unlearning in their case, for gradient inversion in ours).

**Gradient Decoder Architecture (Section 3 of the paper):**
- Input: the flattened LoRA product BA (or equivalently the separate A, B matrices) for a given layer.
- Output: the predicted full-rank gradient ∇_W L for that same layer.
- Network: a small MLP (or U-Net for structured gradients). Trained per-layer — one decoder per weight matrix in the base model.
- Training data: generated from a **proxy dataset** (public data, not the private fine-tuning data). For each proxy batch, compute a single LoRA training step and record the pair (BA, ∇_W L). Repeat ~50k times to build the training set.
- Loss: cosine similarity between predicted and true gradient (the paper also explores MSE; cosine similarity preserves direction, which is what inversion algorithms care about most).

**Key Design Choices We Inherit:**
1. **Proxy data need not match the private distribution** — the decoder learns the geometric relationship between the low-rank subspace and the full gradient space, not the data semantics. CIFAR-100 can proxy for faces; WikiText can proxy for clinical notes.
2. **Single-step vs. multi-step updates** — the decoder is trained on single-step LoRA updates. For adapters trained over many steps, the paper accumulates or averages the decoded gradients. We may need to handle this for our attack scenario.
3. **Rank sensitivity** — decoder accuracy degrades as LoRA rank r decreases (less information retained). The paper reports strong results for r ≥ 4. Our thesis should ablate over r to characterize the privacy-utility tradeoff.
4. **Per-layer decoders** — each transformer layer gets its own decoder. The dimensionality of the gradient varies by layer (e.g., query/key/value projections in attention, up/down projections in MLP blocks).

**How We Diverge from R2F:**
- R2F uses decoded gradients for **gradient ascent** (unlearning). We feed them into **gradient inversion** algorithms (e.g., Inverting Gradients / GradInversion) to reconstruct the actual training inputs.
- R2F targets LLMs (text). Our thesis targets the **image domain** (ViT / ResNet LoRA adapters fine-tuned on images).
- We add **generative priors** (SDS from frozen diffusion models) to compensate for approximation noise in the decoded gradient — R2F does not need this because unlearning is more tolerant of noisy gradients than pixel-level reconstruction.

### Key Reference Papers (in papers/)

- `papers/THE_PAPER.pdf` — Haim et al. (NeurIPS 2022), the foundational reconstruction paper
- `papers/NEWER_2025_Paper.pdf` — Recent follow-up work
- `papers/Gradient Bridge_ PEFT Privacy Attack.pdf` — The Gradient Bridge attack formulation (LoRA → Gradient Decoder → Gradient Inversion)
- `papers/Thesis Ideas_ LoRA, NTK, Reconstruction.pdf` — Full thesis prospectus covering all three directions with theoretical analysis

### Critical External References (not in papers/ — download needed)

- **Gradient Inversion on PEFT (Sami et al., CVPR 2025)** — Shows PEFT dimensionality reduction *focuses* gradient info, making inversion *easier* than full FT. Recovers from batches up to N=128. Directly validates thesis direction. [arXiv:2506.04453]
- **Cocktail Party Attack (Kariyappa et al., ICML 2023)** — ICA on FC layer gradient rows separates N sources from averaged gradient. Scales to N=1024. Key for N>2 reconstruction. [GitHub: facebookresearch/cocktail_party_attack]
- **SPEAR (NeurIPS 2024)** — Exact batch recovery via SVD + ReLU sparsity filtering, N≤25 on FC+ReLU. [openreview.net/forum?id=lPDxPVS6ix]
- **ReCIT (2025)** — Reconstruct private data from PEFT gradients. [arXiv:2504.20570]
- **ARES (2025)** — Sparse recovery gradient inversion, scales to N=384. [arXiv:2603.17623]

### Superposition Problem for N>1 Reconstruction

When reconstructing N≥2 images, the NTK loss has a mixing symmetry: any linear recombination of per-sample gradients that sums to the same total gives the same loss. Reconstructions appear as superpositions (blends) of all training images. Key decomposition approaches:
1. **Cross-gradient orthogonality penalty**: penalize cos_sim between per-sample gradients
2. **Label-based grouping**: cᵢ signs differ by class in binary classification — separate first
3. **ICA on weight gradient matrix**: each FC layer row is a linear mixture of N sources; FastICA separates them (Cocktail Party Attack). Scales to N ≤ layer width.
4. **Sequential peeling**: reconstruct one image at a time from the residual, then joint refinement
5. **Existing code**: `get_diversity_penalty()` in `ntk_extraction.py` is implemented but not wired in

## Environment Setup

```bash
cd dataset_reconstruction
conda env create -f environment_macos.yaml   # Apple Silicon (MPS backend)
conda activate rec
```

Key dependencies: Python 3.8, PyTorch 2.2.2, TorchVision 0.17.2, Kornia 0.7.0, wandb.

A `settings.py` must exist in `dataset_reconstruction/` defining `datasets_dir`, `results_base_dir`, and `models_dir` paths. One already exists with relative paths (`./data/`, `./runs/`, `./models/`).

## WEXAC GPU Access

Scripts in `dataset_reconstruction/` automate GPU allocation on the Weizmann WEXAC cluster (LSF scheduler). Requires Weizmann VPN and `wexac` configured in `~/.ssh/config`.

**Jupyter mode (default)** — launches JupyterLab on a GPU node with SSH tunnels:
```bash
cd dataset_reconstruction
./wexac_connect.sh            # or: ./wexac_connect.sh jupyter
```

**Shell mode** — interactive bash shell on a GPU node for running experiments directly:
```bash
./wexac_connect.sh shell
```
This allocates a GPU via `bsub -q interactive-gpu`, activates the `rec` conda env, and drops you into an SSH session on the GPU node. Run training/reconstruction commands directly from there.

**Disconnect** — kills GPU jobs, tunnels, and temp files:
```bash
./wexac_disconnect.sh
```

Configuration (top of `wexac_connect.sh`): GPU queue, memory, conda env path, Jupyter token, ports.

## Running the Code

Everything runs through `Main.py` with `--run_mode=train` or `--run_mode=reconstruct`.

**Train a model:**
```bash
python Main.py --run_mode=train --problem=cifar10_vehicles_animals --proj_name=cifar10_vehicles_animals \
  --data_per_class_train=250 --model_hidden_list=[1000,1000] --model_init_list=[0.0001,0.0001] \
  --train_epochs=1000000 --train_lr=0.01 --train_evaluate_rate=1000
```

**Reconstruct from a trained model:**
```bash
python Main.py --run_mode=reconstruct --problem=cifar10_vehicles_animals \
  --pretrained_model_path=weights-cifar10_vehicles_animals_d250_....pth \
  [extraction hyperparameters]
```

Pre-built command lines with tested hyperparameters are in `command_line_args/`. Use W&B sweeps for hyperparameter search on new problems.

**Analysis notebooks:** `reconstruction_cifar10.ipynb` and `reconstruction_mnist.ipynb` analyze pre-computed reconstructions.

### Phase 0 Face-Structure Prior

`experiments/phase0_vit_inversion.py` supports a semantic face prior on top of the cos_sim + TV objective. Backbone: kornia.contrib.FaceDetector (YuNet, in env). Three loss terms — top-1 detection confidence, 5-pt landmark layout (eyes < nose < mouth, eye-spacing, nose alignment), and bbox horizontal symmetry. With `--face_weight 0` the legacy code path is byte-equivalent.

| Flag | Default | Purpose |
|------|---------|---------|
| `--cos_weight` | 1.0 | Multiplier on `-cos_sim` (was implicit). |
| `--face_weight` | 0.0 | Master weight on face prior. 0 disables it. |
| `--face_layout_weight` | 1.0 | α on landmark-layout penalty inside the face term. |
| `--face_sym_weight` | 0.5 | β on bbox horizontal symmetry. |
| `--face_warmup_iters` | 5000 | Iters of pure-TV before the face term engages (detectors do not fire on noise). |
| `--face_ramp_iters` | 2000 | Linear 0→1 ramp duration after warmup. |
| `--face_model` | `auto` | Backend: `auto` / `kornia` / `kornia_yunet`. Reserved for future face_alignment/mediapipe. |

Face-prior ablation sweep (9 arms, `face1.jpg`, all parallel bsub jobs):
```bash
./scripts/run_phase0_face_prior_sweep.sh         # submit
python -m experiments.analyze_face_prior_sweep  # build figures + CSV after all complete
```
Outputs: `figures/phase0/face_prior/face_prior_*.png` and `results/phase0_face_prior_sweep_<ts>.csv`.

## Architecture

### Pipeline Flow

```
Main.py (entry point)
  ├── GetParams.py        → argparse (~40 parameters)
  ├── CreateData.py        → dispatches to problem-specific loader
  ├── CreateModel.py       → builds MLP with ModifiedReLU activations
  ├── Train mode:          → SGD on BCEWithLogitsLoss, saves weights to .pth
  └── Reconstruct mode:
      ├── extraction.py    → KKT loss optimization (core algorithm)
      └── evaluations.py   → NCC distance, SSIM, nearest-neighbor matching
```

### Core Algorithm (extraction.py)

The reconstruction optimizes two sets of variables:
- **x** (reconstructed inputs) — initialized randomly, optimized with momentum SGD
- **λ** (Lagrange multipliers) — one per sample, optimized separately

The loss has two components:
1. **KKT loss**: `||∇L(x,λ) - w||²` — the trained weights should equal the gradient of the loss at reconstructed points
2. **Constraint/verification loss**: bounds enforcement (x ∈ [-1,1], λ ≥ 0.05)

### Model (CreateModel.py)

`NeuralNetwork` is a configurable MLP (e.g., D-1000-1000-1) using `ModifiedRelu` — a custom activation with sigmoid-modulated gradients for smoother optimization during extraction.

### Adding New Problems

Add a Python file under `problems/` that provides data loading logic and model parameters. Existing examples: `cifar10_vehicles_animals.py` (binary CIFAR-10), `mnist_odd_even.py` (binary MNIST), `simple_2d.py` (toy 2D).

### Key Directories

- `models/` — pre-trained weight files (.pth)
- `reconstructions/` — pre-computed reconstruction outputs
- `common_utils/` — shared utilities (dataset loading, SSIM via Kornia, image processing)
- `data/` — downloaded datasets (MNIST, CIFAR-10)
- `runs/` — training/extraction output (created at runtime)

## Thesis Directory Structure

```
/home/projects/galvardi/yoado/     ← WEXAC home dir = top-level git repo
├── .gitignore
├── CLAUDE.md                      ← this file
├── STATUS.md                      ← project status, what's done/pending, known issues
├── LESSONS_LEARNED.md             ← running log of insights and pitfalls
├── STYLE_GUIDE.md                 ← formatting rules for docs, slides, LaTeX, plots
├── papers/                        ← reference PDFs (most need syncing from Mac)
├── figures/                       ← graphs, plots, visualizations
├── results/                       ← experimental outputs and metrics (.csv, .pth)
├── notes/                         ← planning docs, theoretical analyses (.tex)
├── scripts/                       ← WEXAC job submission scripts (.sh)
│   └── wexac_logs/                ← WEXAC job stdout/stderr logs
├── experiments/                   ← new experiment code (LoRA bridge, NTK, SDS, etc.)
│   └── tests/                     ← pytest test suite
└── dataset_reconstruction/        ← original Haim et al. codebase (has its own .git)
```

### Mac ↔ WEXAC Sync

The project lives in two places:
- **Mac (local dev):** `~/Documents/Weizmann/Thesis/`
- **WEXAC (GPU compute):** `/home/projects/galvardi/yoado/`

Sync code to WEXAC before experiments:
```bash
rsync -avz --exclude='__pycache__' experiments/ wexac:~/experiments/
rsync -avz papers/ wexac:~/papers/
```

## Key Documents

- [STATUS.md](STATUS.md) — current project status: what's done, what's not started, known issues, pending tasks
- [LESSONS_LEARNED.md](LESSONS_LEARNED.md) — running log of insights, pitfalls, and things to remember
- [notes/experiment_plan.md](notes/experiment_plan.md) — **single source of actionable to-do** (the three meeting additions + DI-Phase 0…3 direct-inversion sequence + GB-Phase 0…2 Gradient Bridge track). Start here for "what to do next."
- [notes/thesis_update_briefing.md](notes/thesis_update_briefing.md) — canonical post-meeting briefing (2026-05-14): direct weight inversion, the three additions, honesty conventions
- [notes/unified_direction_analysis.md](notes/unified_direction_analysis.md) — direction reconciliation + "Direct Weight Inversion — New Primary Axis" section
- [notes/reconstruction_approaches.tex](notes/reconstruction_approaches.tex) — catalog of reconstruction approaches and next steps (March 2026); Approach G is the precursor to direct weight inversion
- [notes/GRADIENT_BRIDGE_PLAN.md](notes/GRADIENT_BRIDGE_PLAN.md) — Gradient Bridge reading syllabus + decoder roadmap (GB-Phase 0 → 1 → 2). Background only; actionable to-do now lives in experiment_plan.md
- [STYLE_GUIDE.md](STYLE_GUIDE.md) — formatting rules for Word docs, PPTX, LaTeX, and plots

### Always Do After Analysis

- After completing any research, analysis, or investigation, **always** update all relevant files (docs, summaries, data files, markdown) with findings — keep updates succinct and factual.
- After making significant or multi-file changes, **always** git commit with a clear message describing what changed and why.

### Presentation & Document Rules

- When the user gives feedback, remarks, or requests about presentation slides, **always** log the remark in `docs/presentation-remarks-log.md` (create if needed) in addition to executing the requested changes.
- **Before creating or modifying a docx/pptx/LaTeX generator**, read `STYLE_GUIDE.md` first. Do NOT rely on memory or guessing — always read the file to get exact details.
- When writing or modifying a docx report generator, follow the style guide's docx conventions (header layout, cover page, TOC, page breaks, logo paths).
- **Quick markdown → PDF on WEXAC:** use Python `fpdf2` + the system DejaVu fonts. `tectonic`/`pandoc`/`pdflatex`/`xelatex` are all unavailable (glibc too old for the tectonic binary). Formal thesis LaTeX still compiles via Overleaf. See LESSONS_LEARNED.md "Markdown → PDF on WEXAC" and the `reference_pdf_generation_method` memory for the recipe + gotchas.

### Data Freshness Rules (Critical)

These rules prevent stale numbers from appearing in documents and presentations:

1. **Single source of truth**: All canonical numbers (metrics, thresholds, counts) must be defined in ONE place (this CLAUDE.md or a dedicated data file). Every document/slide references this source.
2. **Grep after changes**: After changing ANY number or metric, `grep -rn "OLD_VALUE"` across all generator files, markdown, and LaTeX source.
3. **Regenerate plots**: After modifying data, always re-run the relevant plot generators AND regenerate any presentations/documents that embed those plots.
4. **Speaker notes**: When slide content changes, review and update speaker notes for that slide too.

### Document Audit Process

When auditing or revising presentations/documents, follow this order:
1. **Numbers first**: Grep for all instances of stale values across ALL files
2. **Examples second**: Cross-reference every example against source data
3. **New content third**: Add new slides, sections, or examples
4. **Cross-format sync**: Ensure PPTX, Beamer, and docx all match
5. **Polish last**: Speaker notes, transitions, naming consistency
6. **Final audit**: Even after all phases, grep for known stale patterns — the first sweep ALWAYS misses some

## Session Handover

A `/handover` skill (`.claude/skills/handover/SKILL.md`) carries a baton between sessions.

- `/handover save` writes a concise "where work stands" note to `docs/sessions/handover-latest.md` and appends a timestamped copy to `docs/sessions/handover-log.md`.
- A `SessionStart` hook in `.claude/settings.json` prints that note ("📋 Handover note: …") at the top of every new session, so the next session sees it automatically (the hook stays silent until the first save).
- `/handover resume` reads the latest note, **verifies it against the live repo** (referenced files/branches/runs may have moved or finished), and proposes the next step.

This is a short-horizon baton ("what I was mid-way through") — distinct from STATUS.md / LESSONS_LEARNED.md, which are the durable project record. Run `/handover save` before ending a session that leaves something in flight.

## Compute Rules

**ALL experiments must run on the WEXAC GPU cluster, NOT on the local MacBook (MPS).** MPS is only for light local dev and debugging. Any real training, reconstruction, or serious compute must be run on WEXAC (NVIDIA L40S, CUDA 12.6). Use `wexac_connect.sh shell` to get a GPU node. When writing or modifying experiment scripts, always assume CUDA — never write MPS-specific code for experiments.

### WEXAC Compatibility

**WEXAC `rec` env runs PyTorch 2.4.1+cu121** with timm 0.9.12, peft 0.7.1, torchvision 0.12.0, and kornia 0.7.0. Use `weights_only=False` in `torch.load()` to suppress FutureWarnings.
- When submitting experiments to WEXAC, always `rsync` the latest code first: `rsync -avz --exclude='__pycache__' experiments/ wexac:~/experiments/`

## Experiment Output Rules

**Every experiment run must save visual examples, not just numbers.** When running any reconstruction experiment (single config or sweep):

1. **Save image tensors**: Always persist `x_train`, `x_recon_full`, `x_recon_lora`, `x_ctrl`, and `ds_mean` to a `.pth` file so results can be visualized later without re-running extraction.
2. **Save per-config results in sweeps**: For sweeps, save a `.pth` per configuration (e.g., `results/sweep_<name>/T{T}_r{rank}.pth`) containing both metrics and image tensors.
3. **Generate visual grids**: After a sweep completes, generate a PNG or PDF grid showing the best and worst reconstructions (by SSIM). Include ground truth, reconstruction, and control side-by-side so quality is immediately visible.
4. **Include both good and bad examples**: Don't cherry-pick — always show the best *and* worst results from each run. Bad examples are as informative as good ones for understanding failure modes.

This prevents the situation where you have SSIM numbers in a CSV but have to re-run hours of compute just to see what the reconstructions actually look like.

## Code Hygiene Rules

1. **Keep the project clean.** No orphaned files, no dead code left behind, no clutter in the repo root. Every file should have a clear purpose and live in the right directory.
2. **Temporary files go in `/tmp/`.** Any throwaway scripts, scratch notebooks, one-off checks, or debugging artifacts must be written to `/tmp/` (not the project tree) so they never appear in git.
3. **Don't repeat yourself.** If a code snippet is used more than once, extract it into a function — in the same file if it's local, or into a shared utility file (e.g., under `common_utils/` or `experiments/utils/`) if it's cross-cutting.

## Git & Documentation Rules

**Before every commit/push, update project documentation:**

1. **STATUS.md** — Update with what was done: new features, bug fixes, experiments added, scripts changed. Keep the "What's Done" and "What's Pending" sections current.
2. **LESSONS_LEARNED.md** — Log any new insights, pitfalls discovered, or design decisions made during the work.
3. **CLAUDE.md** — If new scripts, tools, workflows, or configuration were added, document them in the relevant section here.

This ensures the project state is always accurately reflected in docs, not just in git history.

### Document big results and lessons — always

**IMPORTANT: this rule is non-optional.** Whenever any of the following happen, update the relevant doc(s) **in the same turn** (don't defer, don't wait to be asked, don't assume the commit message is sufficient):

- **Big research result** (gate crossed, sweep finished, surprising finding, dead end) → write it up in **STATUS.md** with the headline number, the config that produced it, and the next step.
- **Bug discovered and fixed** → log it in **LESSONS_LEARNED.md** with: (a) what the bug was, (b) how it presented (wrong number, silent failure, etc.), (c) the root cause, (d) the fix. The fix being in git history is not enough — the *insight* needs to be retrievable without a git archaeology session.
- **Pitfall, gotcha, or non-obvious design decision** (chosen approach + why, alternatives ruled out, environmental quirk) → also LESSONS_LEARNED.md.
- **New workflow, script, or config** that future-you would otherwise have to rediscover → CLAUDE.md.

Skip only for trivial mechanical work (typo fix, doc reflow, file rename with no behavior change). For anything that produced a new number, a new failure mode, or a new way of doing things, document it. **If you're about to commit and you haven't checked whether STATUS.md / LESSONS_LEARNED.md needs an update, you are not yet ready to commit.**

## Git

The entire thesis directory (`/home/projects/galvardi/yoado/`) is a git repo, initialized 2026-03-19. The `dataset_reconstruction/` subdirectory has its own separate `.git` (origin: `https://github.com/MrYoyoad/Data_Reconstruciton_server.git`).

### Top-Level Remotes

- `myfork`: `https://github.com/MrYoyoad/dataset_reconstruction.git` (personal fork — primary push target)
- `origin`: `git@github.com:ai-hub-weizmann/dataset_reconstruction.git` (Weizmann fork)
- `upstream`: `https://github.com/nivha/dataset_reconstruction.git` (original Haim et al. repo)

### dataset_reconstruction/ Remote (separate git)

- `origin`: `https://github.com/MrYoyoad/Data_Reconstruciton_server.git`
