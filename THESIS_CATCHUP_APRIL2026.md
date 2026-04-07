# THESIS CATCH-UP GUIDE: APRIL 2026

## EXECUTIVE SUMMARY

You are building attacks to reconstruct private training data from LoRA adapters. You've proven the concept on MNIST (SSIM 0.83) with honest free-coefficient extraction that doesn't cheat on oracle knowledge. Now scaling to ViT on real images via the Gradient Bridge.

**Status:** Sprint 2 complete. Phase 0 (ViT gradient inversion gate) underway with 3 bugs fixed.

**Timeline to MVP:** 12 weeks (Phases 0-2 + thesis writing)

---

## 1. THE BIG PICTURE

**Thesis Question:** Can LoRA adapters leak training data?

**Why:** LoRA adapters are published (HuggingFace, CivitAI), encode training data info, few-shot fine-tuning on sensitive data is common.

**Your Contribution:**
- Extend Haim et al. (NeurIPS 2022, KKT reconstruction) to few-shot PEFT regime
- Prove NTK-based attack works honestly (no oracle, multi-step validated)
- Develop Gradient Bridge (decoder maps LoRA → gradient → inversion)

---

## 2. THEORY: KKT + NTK + LoRA

**KKT (Haim et al.):** W* ∝ Σ λ_i y_i ∇_W Φ(W*; x_i). Weights encode training data geometry.

**NTK:** For short training: ΔW = -η Σ c_i ∇_θ f(θ_0; x_i). Reconstruction minimizes ||ΔW + η Σ c_i ∇_θ f||².

**LoRA:** W_ft = W_0 + BA (low-rank). Key: ΔW cancels W_0, isolating fine-tuning signal.

**Few-shot overdetermined:** N=5 images × 150K pixels << r × (d_in + d_out) LoRA params. Reconstruction feasible.

---

## 3. SPRINT 1: PROOF OF CONCEPT

**Experiment A (KKT+Compose):** FAILED. W satisfies KKT over 502 samples; 2 images can't explain. Root cause: structural.

**Experiment B (NTK):** SUCCESS. Full SSIM=0.9999, LoRA r=8 SSIM=0.797, r=32 SSIM=0.826. (Oracle coefficients.)

---

## 4. SPRINT 2: HONEST ATTACK & SCALING

**Free Coefficients:** Add consistency penalty α|c_i - (σ(f(θ_0;x_i))-y_i)/N|². SGD + α=1 matches oracle (SSIM 0.997).

**Multi-Step:** LeakyReLU scales to T=100. T=1: SSIM 0.78, T=100: SSIM 0.73.

**Multi-Seed (50 seeds):** Mean SSIM 0.558±0.034 (control 0.408±0.025). Seed=42 was outlier (0.830).

---

## 5. THE GRADIENT BRIDGE

**Three stages:**
1. Train decoder f_φ: LoRA BA → full gradient ∇_W L (on proxy public data)
2. Decode: predicted ∇_W L
3. Invert: feed to gradient inversion (Geiping et al.) to recover x

**Reference:** Recover-to-Forget (Liu et al., Dec 2025). Proxy data doesn't need to match private distribution.

**Phase 0 (Gate):** Can ViT gradient inversion reconstruct from exact gradients? If yes, proceed to Phase 1 (decoder). If no, inversion broken.

**Status:** 3 bugs fixed. Ready for WEXAC.

---

## 6. KEY INSIGHTS

**Few-Shot Sweet Spot:** N=5-50, T=1-100. Overdetermined. All support vectors. Realistic threat.

**Optimizer+Activation:** SGD + LeakyReLU wins. Interaction matters more than either alone.

**Consistency Penalty:** α=0 non-unique (signs flip). α=1 unique, oracle-matched.

**Seed Variance:** 11% strong signal, 82.5% weak. Not every run vulnerable.

---

## 7. ROADMAP

**Phase 0 (3-5 days, now):** ViT gradient inversion gate.

**Phase 1 (Weeks 3-5):** Train decoder on 50k proxy pairs. Target >0.9 cosine sim.

**Phase 2 (Weeks 6-8):** End-to-end attack. Success: full SSIM>0.9, r=8 SSIM>0.6.

**Sprint 3:** Real data, larger models, longer training.

---

## 8. DOCUMENTS TO READ

1. CLAUDE.md (full context + R2F mechanism)
2. STATUS.md (sprint breakdown)
3. LESSONS_LEARNED.md (detailed insights)
4. papers/THE_PAPER.pdf (Haim et al.)
5. notes/R2F_Guide.tex (decoder mechanism)

---

## 9. NEXT STEPS

**Today:** Read CLAUDE.md + LESSONS_LEARNED.md (2 hours).
**Tomorrow:** Review Phase 0 code. Check WEXAC access.
**This week:** Run Phase 0 on WEXAC (3-5 days).

---

**Total reading time:** 1-2 hours for full document. Start with CLAUDE.md.

