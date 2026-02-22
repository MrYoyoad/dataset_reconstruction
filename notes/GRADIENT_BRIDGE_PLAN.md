# The LoRA-Gradient Bridge: Concrete Thesis Plan

**Timeline:** 8–12 week sprint to Minimum Viable Paper (MVP)

---

## Part 1: The Syllabus (Reading & Watching)

You need to understand three areas: (1) LoRA internals, (2) The "Decoder" mechanism (mapping low-rank to full-rank), and (3) Gradient Inversion (the actual attack).

### Module A: The "Bridge" Mechanics (LoRA & Decoders)

**Goal:** Understand how LoRA parameters ($A, B$) relate to the full model gradient $\nabla W$.

| Resource | Why |
|----------|-----|
| **"Recover-to-Forget: Gradient Reconstruction from LoRA for Efficient LLM Unlearning"** (Liu et al., Dec 2025) | The blueprint. They invented the "Gradient Decoder" to unlearn data. Your thesis pivots this mechanism to *attack* data. Read Section 3 carefully — how they train the decoder on proxy data. |
| **"LoRA-GA: Low-Rank Adaptation with Gradient Approximation"** (Wang et al., NeurIPS 2024) | Theoretical backbone. Mathematically proves that LoRA updates can approximate full-rank gradients. The citation you need to justify why your attack is theoretically possible. |
| **"LoRA: Low-Rank Adaptation of Large Language Models — Explained visually + PyTorch code from scratch"** by Umar Jamil (YouTube) | You need to understand the exact matrix dimensions in the backward pass. This video shows the code. |

### Module B: The Attack Engine (Gradient Inversion)

**Goal:** Understand how to turn a gradient $\nabla W$ back into an image or sentence.

| Resource | Why |
|----------|-----|
| **"Inverting Gradients — How easy is it to break privacy in federated learning?"** (Geiping et al., NeurIPS 2020) | The standard "optimization-based" attack. You will use this algorithm for the Vision part of your thesis. |
| **"DAGER: Exact Gradient Inversion for Large Language Models"** (Petrov et al., NeurIPS 2024) | SOTA for text. Inverting text is harder than images because tokens are discrete. Code is available — understand *why* it works rather than reimplementing from scratch. |
| **"Deep Leakage from Gradients"** (MIT Han Lab, video) | A quick visual demo of the concept. |

### Module C: Threat Modeling & Differentiation

- **"Building Gradient Bridges: Label Leakage from Restricted Gradient Sharing"** (Zhang et al., 2024) — Read the abstract to differentiate your work. They attack *labels* in Federated Learning. You are attacking *input data* from released adapters. Title your thesis **"The LoRA-Gradient Bridge"** to avoid naming collisions.
- **"Reconstructing Training Data from Trained Neural Networks"** (Haim et al., 2022) — Understand the goal of "weight-to-data" inversion.
- **"DSiRe"** (Salama et al., 2024) — Learn how to extract dataset size ($N$) from the LoRA spectrum.

### First 7-Day Reading Priority

1. Recover-to-Forget (Liu et al., 2025) — decoder architecture
2. Inverting Gradients (Geiping et al., 2020) — attack loop
3. Haim et al. (2022) — weight-to-data inversion goal
4. DSiRe (Salama et al., 2024) — dataset size from LoRA spectrum

---

## Part 2: The Coding Roadmap

**Do not spend weeks reading before you code. Start coding Day 1.** The theory will make more sense once you see the tensors.

### When to Start Coding?

**Immediately.** Start reading the Recover-to-Forget paper today. Tomorrow, open a Jupyter notebook and try to manually calculate the gradient of a matrix product $W + BA$. If you can derive $\nabla B$ and $\nabla A$ in PyTorch, you are ready to start Phase 0.

**Your First Technical Challenge:** Derive the gradient of the LoRA matrices in a notebook. If you can calculate $\nabla A$ and $\nabla B$ relative to a full gradient $\nabla W$, you have successfully mapped out the bridge.

---

### Phase 0: The "Perfect Signal" Scaffold (Weeks 1–2)

**Goal:** A script that overfits a LoRA adapter on 1 image/sentence and tries to recover it using "perfect" (cheating) gradients. Before you can reconstruct data from a decoded approximation, you must prove you can reconstruct it from the exact high-dimensional signal.

**Setup:**
- Clone `huggingface/peft` and a gradient inversion library (e.g., `JonasGeiping/breaching` or a simple DLG implementation).
- Load a small model (e.g., ViT, ResNet-18, small ConvNet, or tiny transformer).

**Tasks:**
1. Fine-tune a LoRA adapter ($r=8$) on a single "private" image $x$.
2. **The Cheat:** During the backward pass, manually save the actual full-rank gradient $\nabla_W \mathcal{L}$ (which you usually wouldn't have).
3. **Attack:** Feed that "perfect" gradient into a DLG / Inverting Gradients optimizer.

**Success Metric:** If the reconstructed image matches the original, your inversion engine is ready. Now you just need to bridge the gap from LoRA to that gradient.

---

### Phase 1: The Bridge — Training the R2F Decoder (Weeks 3–5)

**Goal:** Train the "Decoder" network $f_\phi$ that guesses full gradients from LoRA weights.

**Dataset:** Use a public proxy dataset (e.g., CIFAR-100 if your target is faces, or WikiText if your target is clinical notes, or ImageNet).

**Training Procedure:**
1. Simulate 50,000 single-step LoRA updates. For each step, record the pair:
   - $\Delta W_{\text{LoRA}} = BA$ (the low-rank update)
   - $\nabla_W \mathcal{L}$ (the true full gradient)
2. Train a small network (MLP or U-Net — "The Decoder") that takes $\Delta W_{\text{LoRA}}$ (flattened) and predicts $\nabla_W \mathcal{L}$.
3. **Loss:** Cosine Similarity between predicted gradient and true gradient (or MSE).

**Refinement:** Incorporate the DSiRe method to ensure your system can automatically estimate the number of samples ($N$) by analyzing the singular value spectrum of the adapter.

**Milestone:** Achieve high cosine similarity (>0.9) on held-out proxy data.

---

### Phase 2: Crossing the Bridge — End-to-End Attack (Weeks 6–8)

**Goal:** End-to-end reconstruction. Connect the decoder to the inversion engine.

**Procedure:**
1. **Freeze** the Decoder from Phase 1.
2. **Victim:** Fine-tune a new LoRA adapter on a secret image/sentence (*not* in the proxy set).
3. **Attack Pipeline:**
   - Take Victim Adapter $(A, B)$ $\to$ Gradient Decoder $\to$ Approximate Gradient $\hat{G}_{\text{full}}$
   - Feed $\hat{G}_{\text{full}}$ $\to$ Inversion Algorithm (Phase 0 code) $\to$ Reconstructed Image/Sentence
4. **Result:** A reconstructed image or sentence.

**Enhancement — Strong Priors (thesis-grade contribution):** Use Score Distillation Sampling (SDS) from a frozen Stable Diffusion model, or GUIDE, to clean up noise in the reconstruction. This will make your work stand out.

---

## Part 3: Key Thesis "Gotcha"

**Reviewers will ask:** *"How does this differ from just inverting the LoRA weights directly using Haim et al.?"*

**Your Answer:** Haim et al. requires the network to be homogeneous and fully trained to stationarity. The Gradient Bridge works on *any* LoRA adapter, even partially trained ones, because it exploits the mechanical relationship between low-rank updates and full gradients, rather than asymptotic convergence properties. This makes it a more practical, "black-box" style attack.

---

## Summary Timeline

| Weeks | Phase | Deliverable |
|-------|-------|-------------|
| 1–2 | Phase 0: Perfect Signal Scaffold | Script that reconstructs an image from a "cheating" full gradient |
| 3–5 | Phase 1: The Bridge (Decoder) | Trained decoder achieving >0.9 cosine similarity on proxy data |
| 6–8 | Phase 2: End-to-End Attack | Reconstructed images/sentences from LoRA adapters on unseen data |
| 9–12 | Writing & Ablations | Paper draft, ablation studies, strong-prior enhancements |
