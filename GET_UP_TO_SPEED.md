# GET UP TO SPEED IN 30 MINUTES

**You've been away since late March 2026. Here's the current state.**

---

## STATUS AT A GLANCE (April 7, 2026)

| Component | Status | Result |
|-----------|--------|--------|
| Base Reconstruction (Haim et al.) | ✓ Complete | MNIST SSIM 0.85-0.95 |
| Experiment A (KKT-LoRA) | ✗ Closed | Structural failure (W₀ confound) |
| Experiment B (NTK-LoRA) | ✓ Complete | SSIM 0.557±0.034 (free-c, 50-seed) |
| Free Coefficients | ✓ Complete | Honest attack works (no oracle) |
| Multi-Step (T=1-100) | ✓ Complete | LeakyReLU stable through T=100 |
| Phase 0 (ViT gate) | 🔧 Debugging | 3 bugs fixed, resubmitting |
| Phase 1 (Decoder) | ⏳ Pending | Blocked on Phase 0 |

---

## THREE COMPETING DIRECTIONS

### 1. NTK-LoRA (Primary, 70-80% confidence)
- **Idea:** Reconstruct from ΔW = θ_T - θ₀ using NTK approximation
- **Why:** Empirically proven on MNIST, no oracle access needed, scales to T=100
- **Challenge:** 190× pixel increase (MNIST→ViT), NTK validity unknown
- **Timeline:** 6-8 weeks (Phase 0 gate 1w, ViT scaling 5-7w)
- **Contribution:** First NTK attack on vision foundation models

### 2. KKT-LoRA (Parallel, 50-60% confidence)
- **Idea:** Prove phase transition r* = Nd/(d_in+d_out) formula
- **Why:** Theoretically elegant, adds rigor to thesis
- **Challenge:** Experiment A shows KKT fails on pre-trained models, formula may not be sharp
- **Timeline:** 4-6 weeks (parallel to NTK)
- **Contribution:** Theoretical guarantees for few-shot reconstruction

### 3. Gradient Bridge (Ambitious, 30-40% confidence)
- **Idea:** Decoder (R2F) inverts LoRA→full gradient, feed to inversion
- **Why:** Handles multi-step better, most sophisticated
- **Challenge:** Error cascade, nobody studied 0.9-cosine-similarity inversion, Phase 0 dependency
- **Timeline:** 10-12 weeks (requires Phase 0 success)
- **Contribution:** Very high impact if successful, risky

**Decision point:** Phase 0 result (this week) determines path forward.

---

## CRITICAL BOTTLENECK: PHASE 0

**What:** Validate exact ViT-B/16 gradients can invert to images.

**Status:** Debugging (3 bugs fixed)
- Bug 1: Non-differentiable cosine similarity (torch.autograd.grad fix)
- Bug 2: Per-tensor averaging (use global flattened cosine sim)
- Bug 3: LoRA-only gradients (enable requires_grad on all params)

**Submit this week:** `bash scripts/run_phase0_fixed_wexac.sh`

**Expected:** If SSIM > 0.3 → all three directions viable. If < 0.3 → pivot to KKT theory.

---

## WHAT WORKED (KEY RESULTS)

### Free-Coefficient Extraction
- Full model: **SSIM=0.997** (matches oracle, no cheating)
- LoRA r=8: **SSIM=0.557±0.034** (50-seed, recognizable)
- Beats oracle (46/50): Consistency penalty provides implicit regularization

### Multi-Step Robustness (LeakyReLU)
- T=1: SSIM 0.557
- T=10: SSIM 0.572
- T=100: SSIM 0.78-0.80 (stable all the way)

### Control Analysis
- Same-digit images: SSIM 0.394-0.426
- Attack images: SSIM 0.557
- **Gap proves instance-specific leakage, not class recovery**

---

## WHAT FAILED (AND WHY)

### Experiment A: KKT-LoRA (Compose+Reconstruct)
- **Why:** W = W₀ + BA satisfies KKT over ~502 total samples (500 pre-train + 2 fine-tune)
- **Effect:** KKT loss = ||W₀||² (unexplainable pre-training residual)
- **Lesson:** 2 images can't explain 502-sample information
- **Thesis value:** Negative result motivates ΔW/NTK approach ✓

### Seed=42 Outlier
- **Seed 42 SSIM:** 0.83 (top 11% of 200-seed sweep)
- **Median SSIM:** 0.55
- **Lesson:** Don't cherry-pick seeds; use 50-seed statistics

---

## KEY INSIGHTS (READ THESE)

1. **Pre-trained θ₀ is essential** — Random init breaks NTK (0.4 vs 0.99 feature stability)
2. **LoRA subspace isolates fine-tuning** — ΔW cancels W₀, enabling reconstruction
3. **Activation function > optimizer** — ReLU+L-BFGS (0.744) >> ModifiedReLU (0.18)
4. **Consistency penalty breaks non-uniqueness** — α=1 optimal, α=0 fails (sign-flip)
5. **Few-shot is the attack sweet spot** — N << parameters, T << convergence time
6. **Oracle access not required** — Free-coefficient extraction works honestly
7. **LeakyReLU enables multi-step** — Stable through T=100, ReLU NaN's at T≥50

---

## IMMEDIATE NEXT STEPS

### This Week
1. Submit Phase 0 (4 hours) → await result
2. Extend `run_experiment_b.py` for ViT (2 days)
3. Read Geiping et al. gradient inversion paper (1 day)
4. Update thesis prospectus (1 day)

### Week 2-3 (Phase 0 dependent)
- **If SSIM > 0.3:** ViT scaling (NTK primary track)
- **If SSIM < 0.3:** KKT theory (pivot track)

### Decision Points
- Week 1: Phase 0 result (fork)
- Week 3: ViT NTK T=1 (go/no-go)
- Week 5: Multi-step (proceed to Gradient Bridge?)
- Week 8: Thesis narrative finalized

---

## WHERE TO READ (IN ORDER)

### For Deep Understanding
1. **CLAUDE.md** (10 min) — Full project context, R2F mechanism, architecture
2. **LESSONS_LEARNED.md** (20 min) — Comprehensive running log of all insights
3. **STATUS.md** (15 min) — Current status, pending tasks, git state
4. **notes/unified_direction_analysis.md** (10 min) — Three approaches reconciled

### For Theory
1. **notes/R2F_Guide.pdf** (15 min) — Gradient decoder architecture
2. **notes/Inversion_Feasibility_Analysis.pdf** (10 min) — Error cascade analysis
3. **papers/THE_PAPER.pdf** (2-3 hours) — Haim et al. foundational work
4. **papers/27_Recover_to_Forget_Gradient_.pdf** (2 hours) — R2F mechanism

### For Code
1. **experiments/run_experiment_b.py** — NTK extraction entry point
2. **experiments/ntk_extraction.py** — Reconstruction algorithm
3. **experiments/phase0_vit_inversion.py** — ViT gradient inversion
4. **experiments/configs.py** — Constants and device detection

---

## KEY FILES & COMMANDS

### Submit Phase 0 to WEXAC
```bash
ssh wexac  # Requires Weizmann VPN
cd /home/projects/galvardi/yoado
bash scripts/run_phase0_fixed_wexac.sh
```

### Monitor Jobs
```bash
bjobs  # List active LSF jobs
bpeek <job_id>  # Tail job output
```

### Local Testing
```bash
cd experiments
python run_experiment_b.py --rank 8 --n_steps 1 --free_coefficients \
  --consistency_weight 1.0 --optimizer sgd
```

### Key Papers
- **THE_PAPER.pdf** — Haim et al. (foundational)
- **27_Recover_to_Forget_Gradient_.pdf** — R2F (decoder)
- **Gradient Bridge_ PEFT Privacy Attack.pdf** — Our formulation

---

## NUMBERS TO REMEMBER

| Metric | Value | Context |
|--------|-------|---------|
| Full model SSIM (T=1, honest) | 0.997 | Near-perfect (oracle quality) |
| LoRA r=8 SSIM (50-seed avg) | 0.557±0.034 | Recognizable digits |
| LoRA r=32 SSIM (best) | 0.680 | Moderate reconstruction |
| Control SSIM (same digit) | 0.394-0.426 | Large gap → real leakage |
| Multi-step stable (T=100, LeakyReLU) | 0.78-0.80 | Through typical fine-tuning |
| Seeds with strong signal | 11% (22/200) | Median SSIM ≈ 0.55 |
| MNIST pixel space | 784 | 28×28 image |
| ViT pixel space | 150,528 | 224×224×3 image |
| Search space increase | 190× | Major scaling challenge |

---

## THE THREAT MODEL

**When is LoRA reconstruction most dangerous?**

- **Few-shot fine-tuning:** N=5-50 samples (adapters far exceed parameter count)
- **Few gradient steps:** T=1-100 (typical fine-tuning duration)
- **Realistic scenarios:**
  - CivitAI face LoRAs (~5-20 selfies)
  - Medical imaging adapters (~50 scans)
  - Internal LLM fine-tuning (custom docs)
- **Why vulnerable:**
  - All samples become support vectors (overdetermined)
  - LoRA encodes full gradient in low-rank subspace
  - Published adapters are public attack surface

---

## 10 THINGS TO REMEMBER

1. **ΔW, not W** — Targeting weight *change* (not composed weights) solves pre-training confound
2. **Free coefficients work** — Consistency penalty removes oracle requirement
3. **Seed matters** — 11% of seeds strong, 82.5% weak; use 50-seed stats
4. **LeakyReLU stable** — Critical for multi-step; ReLU NaN's at T≥50
5. **Phase 0 gates everything** — ViT gradient inversion must work first
6. **NTK is the backup** — If gradient inversion fails, pivot to KKT theory
7. **Activation > optimizer** — Choice of ReLU/LeakyReLU matters more than SGD/L-BFGS
8. **Few-shot is realistic** — Attack most potent for N << parameters
9. **Save tensors, not metrics** — Visual examples essential for failure analysis
10. **Gradient flow is tricky** — Use `create_graph=True` for cosine similarity

---

## NEXT IMMEDIATE ACTION

**THIS WEEK:**
1. Review Phase 0 implementation
2. Submit to WEXAC
3. Read phase 0 bugs to understand ViT inversion challenges
4. Prepare ViT pipeline while waiting for results

**By week 2:** Phase 0 result determines thesis direction.

---

*For questions, see CLAUDE.md (project overview), LESSONS_LEARNED.md (detailed insights), STATUS.md (task tracking).*

