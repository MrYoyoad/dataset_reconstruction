# Lessons Learned

Running log of insights, pitfalls, and things to remember as the thesis progresses.

---

## Base Reconstruction (Haim et al.)

### Setup & Environment
- Apple Silicon (MPS backend) works but watch for dtype mismatches — MPS doesn't support all float64 ops.
- The `settings.py` file with relative paths (`./data/`, `./runs/`, `./models/`) keeps things portable.
- **Primary compute is WEXAC cluster**, not the MacBook. GPU: NVIDIA L40S (46 GB VRAM), CUDA 12.6. Connect via `wexac_connect.sh` (requires Weizmann VPN). Conda env on cluster: `/home/projects/galvardi/yoado/.conda/envs/rec`.
- The L40S easily handles all planned experiments: ViT LoRA fine-tuning (~2-3 GB), gradient decoder training (~50k pairs), gradient inversion (~4-8 GB), and even Stable Diffusion for SDS priors (~8-12 GB). **Compute is not a bottleneck.**

### Training
- Models need to train to near-stationarity (very long — 1M epochs) for the KKT conditions to hold. Don't cut training short.
- `ModifiedReLU` is critical — standard ReLU gives much worse reconstruction because the smooth gradients matter during extraction.
- BCE loss (not cross-entropy) is required for the implicit bias / max-margin convergence theory to apply.

### Reconstruction
- KKT loss optimization is sensitive to initialization — random restarts help.
- Lambda (Lagrange multiplier) optimization needs a separate, typically smaller learning rate.
- The number of reconstructed samples should match the actual training set size for best results.

---

## LoRA / Gradient Bridge

### Key Realizations
- The Gradient Bridge is theoretically sound and all building blocks exist independently (R2F for decoding, Inverting Gradients for inversion). But **"building blocks exist" ≠ "easy to do"** — the real research risks are empirical, not computational:
  1. **Decoder accuracy for pixel-level reconstruction**: R2F proved the decoder works for unlearning (tolerant of noisy gradients). Nobody has shown it works for pixel-level image reconstruction, which is far more sensitive to gradient noise. Even 0.9 cosine similarity may not be enough.
  2. **Multi-step accumulation**: The decoder is trained on single-step LoRA updates. Real adapters train for thousands of steps. How to handle accumulated updates is an open question.
  3. **Error compounding**: LoRA approximation × decoder approximation × inversion approximation — each stage is "pretty good" but errors multiply through the pipeline.
- These are answerable by running experiments, and we have the compute (L40S) to run them fast.
- The correct strategy is to de-risk in order: (1) Sprint 1 compose-and-reconstruct, (2) Phase 0 "cheating" with perfect gradients to find the ceiling, (3) only then attempt the decoder. If Phase 0 fails, the decoder won't save it.

### What Worked
-

### What Didn't Work
-

### Pitfalls to Avoid
-

---

## NTK Regime & High-Rank LoRA

### Key Realizations
- **Start from pre-trained weights, not random init.** The original Experiment B design used random init with `init_scale=0.0001`, which put ALL pre-activations within ±0.002 of the ReLU kink. After 1 SGD step, ~half the neurons flipped on↔off, destroying NTK feature stability (cosine similarity 0.39 instead of >0.99). The NTK approximation was completely invalid, and all reconstructions were noise (SSIM ~0.35).
  - **Root cause**: `init_scale=0.0001` was inherited from the convergence/max-margin pipeline (Experiment A / Haim et al.), where tiny init helps the implicit bias theory. But for NTK, you need pre-activations AWAY from the ReLU kink. The pre-trained model (`weights-mnist_odd_even_d250`) has layer 0 std=0.004 (40× larger), giving pre-activation std=0.25 — well away from zero.
  - **Fix**: Load pre-trained weights as θ₀ and fine-tune on **held-out data** (MNIST test set, not the 250/class training data). With pre-trained weights, feature stability = 1.000 at T=1.
  - **The correct attack scenario**: Pre-trained model + LoRA fine-tuning on private data → attacker reconstructs private data from ΔW = BA. This matches real-world LoRA usage.
- **Fine-tuning data must not overlap with pre-training data.** The pre-trained model was trained on first 250 even + first 250 odd digits (sequential, no shuffle from MNIST train set). If you fine-tune on samples already in the training set, the gradient is essentially zero (coefficients ≈ 1e-18) because the model already converged on them. Use MNIST test set or clearly held-out indices.
- **The pre-training data selection is deterministic**: `shuffle=False, start=0, end=50000` in `mnist_odd_even.py`, so the exact 250/class samples are reproducible.

### What Worked
-

### What Didn't Work
- Random init with init_scale=0.0001 for NTK experiments → all pre-activations at ReLU kink → destroyed NTK feature stability → noisy reconstructions

---

## Discrete Sequence Reconstruction (LLMs)

*(Fill in as you go)*

### Key Realizations
-

### What Worked
-

### What Didn't Work
-

---

## Diffusion Priors / SDS

*(Fill in as you go)*

### Key Realizations
-

### What Worked
-

### What Didn't Work
-

---

## General Research Process
- **"The math says it works" vs. "you can make it work"** are different claims. Optimistic theoretical analyses (like the Gradient Bridge feasibility argument) are correct about the information being there, but gloss over engineering gaps and empirical unknowns. Calibrate accordingly: the idea is sound, but the execution is the hard part — which is exactly what makes it a thesis.
- **De-risk before building**: always run the cheapest experiment that could falsify your approach before investing weeks in the full pipeline.
