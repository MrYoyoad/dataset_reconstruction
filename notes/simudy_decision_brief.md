# SimuDy Decision Brief — does the ICLR'25 scoop kill the direction?

**Date:** 2026-06-29 · **Trigger:** Gal sent SimuDy — *"It seems that the following paper already
showed an idea that we discussed."* · **Companion:** [related_work_simudy.md](related_work_simudy.md)
(paper teardown) · **Resolves:** Part D novelty search in [experiment_plan.md](experiment_plan.md).

Written as the dependency chain 1→N (read top-down; each step assumes the one above).

---

1. **Provenance.** The "idea we discussed" in supervision = direct weight inversion (simulate the
   fine-tune, match final weights). SimuDy (Tian et al., ICLR 2025, OpenReview `ZJftXKy12x`) published
   exactly that primitive. Gal flagged it himself, neutrally — routine "we got scooped on a sub-idea,"
   not "the thesis is dead."

2. **What SimuDy is.** Init dummy data x̂ from noise → **unroll T epochs of mini-batch SGD** through x̂
   → optimize x̂ so simulated final weights match real ones. Loss `−cos(θ_f−θ₀, θ̂_f−θ₀) + α·TV(x)`
   (+ gradient clipping in the unroll). This is *literally* our `argmin‖θ_T − F(θ₀,x̂)‖²` /
   Approach-G / S3.4 — the full-unroll, no-linearization version.

3. **The collision.** It is the published, **full-fine-tuning** realization of our DI primary axis.
   So the headline "direct inversion of fine-tuning" novelty is **taken**. Concede this fast.

4. **Verdict (undermine / strengthen / pivot).**
   - *Undermines* only the empirical-attack-as-headline framing (the weakest-to-defend part).
   - *Strengthens* the thesis if it's theory-led: free baseline + public code + de-risked premise
     (weight inversion provably works, even on a ViT) + free phenomena to theorize about.
   - *Forces a significant pivot* of emphasis: headline moves from "build the attack" (scooped,
     incremental) → "**identifiability theory + LoRA-only regime**" (open, Gal-shaped). Not a topic change.

5. **The reconstruction chain (to learn {xᵢ} from an adapter A,B):**
   1. **Weight signal** — from (A,B) get ΔW=BA, with base θ₀ get θ_f. *(True LoRA-only / rank
      bottleneck needs a decoder (A,B)→full gradient = the Gradient Bridge.)*
   2. **Identifiability** — prove θ_f (or ΔW) *uniquely* determines {xᵢ}; no superposition degeneracy.
   3. **Model the dynamics** — characterize F (multi-step SGD/Adam) well enough to differentiate through.
   4. **Invert** — minimize ‖θ_f−F(θ₀,x̂)‖ with reachable-optimum ⟺ true data (loss↓ ⟺ quality↑).
   5. **Decouple N** — separate each xᵢ from the cumulative gradient sum at realistic N.
   6. **Scale to huge models** — do 3–5 without the unroll graph blowing up memory.
   7. **Realism + verdict** — robust to unknown recipe/order/optimizer; x̂ᵢ are instance-level private
      images at real fidelity (not 0.2-SSIM blends) → "leakage."

6. **What SimuDy proves.** Step 3 ✔ (core contribution; full-unroll beats linear-dynamics shortcut;
   M_lin=0.60 ResNet vs 0.95 wide-MLP). Step 4 ✔ empirically (loss↔SSIM, Fig.2). Step 5 ✔ only at
   small N (~80/120). Step 1 ✔ only the easy half (full ΔW known). **Conditions:** full FT, small
   models (ResNet-18; ViT only **N=10**), **N≤~120**, known/grid-searchable recipe, full weight
   access, SSIM≈0.2 (memorization evidence, *not* a privacy attack).

7. **What SimuDy misses.** Step 1 LoRA-only: **untouched.** Step 2 identifiability: *asserted not
   proven* — they even exhibit degeneracies (background lost Fig.8, blended frogs Fig.9). Step 5 at
   scale: degrades with N. Step 6 huge models: memory-bound (**22 GB / 15 h for 120 CIFAR-32² on
   ResNet-18**). Step 7: no Adam/scheduler/order ablation; framed as interpretability, not leakage.
   → **Answer to "do they prove leakage from huge-model fine-tune?": No.** Existence at small scale,
   not a scalable privacy attack. The open links (1,2,5,6,7) ARE the thesis.

8. **Feasibility.** Full ambition ("huge models, full unroll") = **not feasible** for one student on
   an L40S (SimuDy already maxes a 4090 at 120 tiny imgs). Right target is the *opposite of brute
   force*: NTK-linearization + anchor-α + low-rank = **memory-tractable** inversion SimuDy can't do.
   Reachable ceiling = **real ViT-B/L & SD-scale LoRA adapters**, not LLM-7B (still scary/relevant —
   that's what people upload). Feasibility hinges on two cheap gates (step 9).

9. **The bottleneck-gated plan (fail-fast: cheapest killer first).**
   1. **B1 — recover xᵢ from (A,B) alone at all?** (the premise). Small model, small N, invert from
      adapter only. Partial positive evidence already: NTK-LoRA r=8 oracle SSIM **0.80**, free-c
      **0.62**, multi-step stable to T=100; control 0.58–0.69. Yellow flag: free-c stubborn at
      r=16/32 (~0.42). Go/no-go: beat same-class control. *Fail ⇒ empirical thesis dead, pivot to
      theory-only.* Days.
   2. **B2 — can linearized/anchor inversion replace full unroll?** (the scale enabler). SimuDy
      full-unroll vs NTK-anchor on identical ResNet-18/CIFAR; compare SSIM **and** memory; run the
      α two-curve (lin-error vs α, SSIM vs α). Yellow flag: M_lin=0.60 says linearization is crude →
      anchor-α is the fix being tested. *Fail ⇒ can't reach ViT-scale, fall back to small-scale+theory.* ~1 wk.
   3. **B1+B2 are the whole bet.** Both pass ⇒ thesis + likely paper. Below is upside.
   4. **B3 — N-scaling** (SSIM-vs-N, adapter-only). Tailwind: Sami et al. CVPR'25 — PEFT *focuses*
      gradient info, inversion *easier*, N up to 128. Tools: diversity penalty, ICA/cocktail-party.
   5. **B4 — model scale** small→ViT-B (already gated)→larger, using B2's linearized inversion.
   6. **B5 — realism** (Adam, unknown order, weight decay) after B1–B4.
   7. **Theory track (parallel, Gal-led):** identifiability for B1/B3, anchor tradeoff for B2 — the
      durable, un-scoopable contribution; feasible *as analysis* even if scales stall.

10. **Paper-worthy? Conditionally yes.** Hot subarea (Sami'25, ReCIT'25, ARES'25); "can your
    published LoRA adapter leak your private fine-tune images?" is clean and scary. LoRA-only +
    identifiability theorem is a genuine delta vs full-FT-no-theory SimuDy. **Condition:** at least one
    of (i) adapter-only recovery at a setting/scale SimuDy can't, or (ii) a real identifiability
    theorem, must land. "SimuDy-on-LoRA, small N, no theory" alone = weak increment.

11. **"Burden we fill for them"?** Yes — the *hard half*, not cleanup. SimuDy explicitly punts on
    memory/scale and has zero theory. Size of the gap = both the value and the risk: open because hard,
    not guaranteed to close. Real burden = real contribution = real risk, equal measure.

12. **Decision / next step.** Commit to the **gated** version: "**when** does a LoRA adapter leak its
    fine-tune data + a method that reaches ViT-scale without unrolling," gated on **B1+B2** (weeks of
    cheap work). Pass ⇒ strong thesis, paper, SimuDy as cited baseline-to-beat. Fail ⇒ pivot to
    theory-only (most Gal-shaped outcome anyway). *Do not* defend direct-full-FT-inversion as novel.

13. **Reply to Gal (drafted, short, constructive):** confirm it's the same primitive (SimuDy) →
    note it stops where our discussion went further (no LoRA/PEFT, no theory, brute-force unroll
    22GB/15h, ViT only N=10) → propose: cite as baseline, re-center on adapter-only recovery +
    identifiability/anchor theory → "two weeks to settle B1+B2, then the LoRA-leakage paper is real or
    we have the theory either way."
