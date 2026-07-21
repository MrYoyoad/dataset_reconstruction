# Related work: SimuDy (ICLR 2025) vs. the Direct-Weight-Inversion axis

**Logged:** 2026-06-29 (resolves the Part D novelty search for `θ_T = F(θ₀, x_i)` matching).
**Provenance:** Gal sent the paper himself — *"It seems that the following paper already showed an idea
that we discussed."* So the scooped idea is the direct simulate-dynamics primitive brainstormed in
supervision; the analysis below is the case for what of that discussion remains open (PEFT/LoRA, theory).

**Paper:** Tian, Liu, He, He, Huang, Yang, Huang (SJTU), *"Simulating Training Dynamics to
Reconstruct Training Data from Deep Neural Networks"*, ICLR 2025.
OpenReview `ZJftXKy12x`; code `github.com/BlueBlood6/SimuDy`.

## What it does (one paragraph)
SimuDy reconstructs training data by **simulating the whole training run with a dummy dataset and
optimizing the dummy data so the simulated final weights match the real final weights.** Outer loss
`min_x̂ d(θ_f − θ₀, θ̂_f(x̂) − θ₀)` with `d` = **cosine similarity** (+ TV prior), where `θ̂_f(x̂)` is
produced by **unrolling T epochs of mini-batch SGD** through the dummy images (gradient clipping for
stability). This is **exactly our Direct-Weight-Inversion outer loss** `argmin‖θ_T − F(θ₀,x̂)‖²` /
Approach-G "differentiable unrolling" / S3.4 — the full-unroll (no-linearization) version of it.

## Threat model / setting (what makes it the *best-case* regime)
- Assumes attacker has **both θ₀ and θ_f** (the "fine-tuning scheme", following Loo et al. 2024).
- **Full fine-tuning of all weights** — **no LoRA, no PEFT, no adapters** (terms never appear).
- Known dataset size + resolution; recipe (η, |B|, T) known, or **grid-searched** from early-loss
  (robust to wrong batch-size guess) — i.e. Regime A (known/recoverable recipe).
- Models: **ResNet-18** (main), ResNet-50, SVHN; **ViT** (Appendix C.6, **only 10 ImageNet images**,
  needs a position-embedding loss term); TinyBERT/CoLA (NLP, batch of 4).

## Results
- MLP/100 imgs SSIM 0.337 (vs Loo 0.138); ResNet/50 SSIM **0.198** (vs Loo 0.077, Buzaglo 0.030);
  ResNet/120 SSIM ~0.12 (~80 of 120 recovered). Binary MLP/20: 0.538.
- **Cost is the wall:** full computation graph kept in memory → **22 GB / 15 h for 120 CIFAR-32²
  images on a ResNet-18** (their Table 3). Stated primary limitation: degrades with dataset size,
  bounded by GPU memory. ViT shown at N=10 only.

## Does it undercut us? No — but it *takes the full-FT direct-inversion novelty.* Why we survive:
1. **PEFT/LoRA untouched.** Our differentiator (row 3: leak only the **adapter** (A,B), not full
   weights; Gradient Bridge decoder; LoRA-as-compressed-gradient) is not addressed at all.
2. **Best-case regime only.** SimuDy lives entirely in known-θ₀ + known/grid-searchable recipe +
   full weights. Our framing already calls direct inversion the *upper bound under best-case
   knowledge*; the contribution is **leakage degradation under weaker assumptions** (adapter-only,
   no recipe) — SimuDy says nothing here.
3. **No theory.** Purely empirical (even their "linearity metric" proves nothing). Our Gal-shaped
   core — identifiability/stability of `R: g→x̂` (Q-A), pretrain/finetune overlap (Q-B), the
   anchor-α linearization-vs-contamination tradeoff — is open. SimuDy actually *supplies phenomena*
   for it (Fig 8 background-vs-object = a Q-B identifiability effect; dummy-size expt = identifiability probe).
4. **It's the expensive full-unroll; our NTK-anchor track is the cheap approximation.** Their own
   limitation (memory/graph) is exactly what LoRA low-dimensionality + linearization addresses.

## Plan implications (reframe, don't abandon)
- **Demote** "direct weight inversion of full fine-tuning" as a *headline novelty* — it's published.
  **Re-center** novelty on: (i) PEFT/LoRA-only leakage + Gradient Bridge; (ii) weaker-knowledge
  regimes; (iii) identifiability/stability theory + anchor-α tradeoff; (iv) NTK-linearized efficient
  inversion as the tractable counterpart to SimuDy's full unroll.
- **Cite SimuDy as the closest prior work and a feasibility de-risker** (full unroll demonstrably
  works on ResNet/ViT) and as a **baseline to compare against** in the LoRA regime.
- Note: **Gal Vardi co-authored both Haim et al. 2022 and Buzaglo et al. 2024**, which SimuDy
  benchmarks against and beats — he is personally close to this and likely wants both "are we
  scooped?" and "what's our response to it beating our line?" answered.

## Decision brief
Feasibility, paper-worthiness, the "burden" question, and the fail-fast gated plan (B1→B5) live in
[simudy_decision_brief.md](simudy_decision_brief.md) (written as the 1→N reconstruction chain).
