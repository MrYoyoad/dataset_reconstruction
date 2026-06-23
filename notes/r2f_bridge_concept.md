# R2F Bridge — slide-ready conceptual content

**Use this as the source text for the "Gradient Bridge: from unlearning to reconstruction" slide (currently slide 3 of the deck).**

Reference: Liu et al., *"Recover-to-Forget: Gradient Reconstruction from LoRA for Efficient LLM Unlearning"*, Dec 2025. arXiv id: pending in `papers/`.

---

## Slide title

**Bridge: from unlearning to reconstruction**

Subtitle (optional): *"Same machinery R2F uses for forgetting, we use for recovering."*

---

## The setup in one paragraph

R2F is a **defensive** technique. Given a LoRA-fine-tuned model and a request to "forget" specific training samples, R2F trains a small neural network — a **gradient decoder** `f_φ` — that maps the LoRA adapter `(A, B)` back to an approximation of the full-rank gradient `∇_W L`. The defender then does **gradient ascent** on the decoded gradient to push the model away from the data it should forget.

We **flip the same pipeline 180°**: take the decoded gradient and feed it into a gradient-inversion attack to **recover** the fine-tuning data instead of forgetting it.

---

## The pipeline (single diagram)

```
        ┌──────────────────────────────────────────────┐
        │  LoRA adapter (A, B)  — published artifact   │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  Gradient decoder  f_φ                        │
        │  • Trained on proxy data                      │
        │  • Per-layer MLP                              │
        │  • Cosine-similarity loss                     │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │  ∇_W L̂   (approximate full-rank gradient)    │
        └──────────────────────┬───────────────────────┘
              R2F branch ──────┼──────  Our branch
              (defense)        │        (attack)
                               │
        ┌──────────────────────┴───────────────────────┐
        │                                              │
        ▼                                              ▼
   Gradient ASCENT                              Gradient INVERSION
   to forget x                                  to RECOVER x
   (R2F's purpose)                              (our purpose)
```

The decoder is the **same network**, trained the same way. The only thing that differs is what we do with `∇_W L̂` after we have it.

---

## How the decoder is trained (R2F's recipe — we inherit)

1. **Proxy dataset.** Public data, doesn't need to match the private fine-tuning distribution (e.g., CIFAR-100 as a proxy for face data).
2. **Generate training pairs.** For each proxy batch:
   - Do **one LoRA fine-tuning step** on a fresh base model.
   - Record the resulting `(A, B)` adapter — input to the decoder.
   - Compute the actual full-rank gradient `∇_W L` for that batch — target for the decoder.
3. **Repeat ~50,000 times** → 50K (input, target) pairs.
4. **Train one decoder per layer** (per-layer because different layers have different gradient dimensionalities and structures).
5. **Loss: cosine similarity** between predicted gradient and true gradient — preserves direction, which is what downstream gradient inversion cares about.
6. **Result:** a learned approximate inverse of the LoRA projection `∇_W L ↦ (B^T ∇_W L, ∇_W L A^T)`.

---

## Why this matters mathematically (one-line version)

LoRA stores a rank-`r` projection of the gradient:

> `∇_A L = B^T ∇_W L`     and     `∇_B L = (∇_W L) A^T`

So the LoRA update `BA` is **not** an arbitrary compression — it's a **structured measurement** of the principal components of the gradient. The decoder learns the **structure of the inverse projection** from proxy data alone, without seeing any private data.

---

## What R2F proves (and what we inherit)

R2F demonstrates that **the LoRA-to-gradient bridge is learnable** — i.e., recovering full-rank gradient information from a LoRA adapter is *practically feasible*, not just theoretically possible. They show this works for **rank ≥ 4** with high cosine similarity to the true gradient.

This is the **single load-bearing claim** for our thesis. If the decoder is learnable, then the rest of our attack pipeline (gradient inversion on decoded gradients) is just a downstream consumer.

---

## How our flip differs (in one slide block)

| Dimension | R2F (defense) | Ours (attack) |
|---|---|---|
| **Purpose** | Forget fine-tuning data | Recover fine-tuning data |
| **Post-decoder step** | Gradient ascent on the model | Gradient inversion in pixel space |
| **Domain** | LLMs (text) | ViT / vision |
| **Tolerance to noise** | High — ascent works with imperfect gradients | Lower — pixel-level recovery is sensitive |
| **Extra machinery** | Just the decoder | Decoder + (optionally) diffusion priors / SDS to compensate for decoder noise |

---

## The two open questions (slide bullet form)

1. **Does the decoder generalize from proxy to victim distribution?** R2F tested this on text; we need to validate for vision (e.g., CIFAR-100 → face data). If proxy/victim shift breaks the decoder, the whole pipeline degrades.
2. **How much decoder noise can gradient inversion tolerate before reconstruction collapses?** This is Theory Q-A on slide 18 — the inverse-problem stability question. The decoder gives an *approximate* gradient; we need a Lipschitz characterization of the gradient → image map.

---

## One-sentence elevator line

> **"R2F proves the LoRA-to-gradient bridge is learnable for defense; we run the same bridge backwards to make it an attack."**

---

## Citations to footnote

- Liu et al., 2025 — *Recover-to-Forget* (R2F): the source paper for the bridge concept.
- Sami et al., CVPR 2025 — gradient inversion on PEFT, validates that PEFT dimensionality reduction makes inversion *easier* not harder.
- Geiping et al., 2020 — *Inverting Gradients*: the downstream gradient-inversion engine we feed the decoded gradient into.
- (Optional) Smorodinsky-Vardi-Safran 2024 — *Provable Privacy Attacks*: the theoretical KKT-based attack from Vardi's group; R2F + our flip is the *practical* / *PEFT-aware* extension.

---

## Suggested slide layout

- **Top half:** the pipeline diagram (ASCII version above translates well to PowerPoint shapes: 3 boxes with two divergent arrows at the bottom).
- **Bottom-left quadrant:** the elevator sentence in italic.
- **Bottom-right quadrant:** the 2-row table (R2F vs Ours), keep just 3 rows max for legibility on the slide.
- **Speaker notes:** include the "how the decoder is trained" recipe and the two open questions — those are talking-track only, not on-slide.
