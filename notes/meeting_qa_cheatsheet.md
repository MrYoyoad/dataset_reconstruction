# Q&A Cheatsheet — Supervisor Meeting (Thursday 2026-05-14)

In-pocket reference. Updated 2026-05-13 late evening — **deck final at v18 (29 slides, all visible)**. File: [figures/supervisor_meeting_2026_05_14_v18.pptx](figures/supervisor_meeting_2026_05_14_v18.pptx) (lives under `figures/`, not `notes/`).

**Headline v12 → v18 changes:**
- NEW slide 3: **Attack taxonomy** — 4-row table of what the attacker sees (∇L / ΔW / (A,B) / W₀+BA), each mapped to evidence in the deck and to a status (done / proposed / failed). This is now the structural framing for the whole talk — Q9 below covers it.
- **Part III divider now exists** (slide 20) — the previously-missing divider was added. Parts are now cleanly I / II / III (was I / II / IV). No more "where's Part III?" risk.
- Slide 13 retitled and math refined: "Batches and epochs: **same inverse problem with sample-weighted residuals**". The c′_i derivation is now the full sum c′_i = Σ_t 𝟙[i ∈ ℬ_t] · c_i(t), with the simple form c′ ≈ n · c presented as a "stable-residual" special case.
- Slide 18 retitled: "Multiple images: no collapse, **but imperfect identity separation**" — sharpens the caveat.
- Slide 29 (last) is now a visible NTK-plot backup (was a hidden composite-of-7+8 in v12).

---

## ⏰ Thursday morning — 45-min protocol

Strict time-box. Don't overshoot any block.

| Block | min | What |
|---|---|---|
| 1. Tier-1 reading | 15 | Skim [notes/tier1_paper_notes.md](notes/tier1_paper_notes.md) (digest of SVS 2025 + Gronich-Vardi). Focus on **homogeneity assumption** + **Thm 4.2 (MIA)** in SVS. **Do NOT open the PDFs themselves** — the digest is enough. |
| 2. Live Q&A drill | 20 | Open chat, type **"run live Q&A drill"**. Claude poses the 6 drill questions below at meeting pace. You answer **out loud** (or quick keyboard). Claude flags weak/missing pieces in ≤2 sentences each. |
| 3. Deck walkthrough | 10 | Open v18 PPTX. Click through all 29 slides at meeting pace. Do **not** edit anything — note issues verbally only. Confirm: slide 3 (attack taxonomy) is the framing beat, slide 6 (SSIM tutorial) is "point and move on", slide 9 (instance leakage) is the new must-deliver beat. |

---

## 🎯 Live Q&A Drill — 6 fast questions (20 min total, ~3 min each)

Highest-probability supervisor questions. Drill protocol: Claude asks one at a time, you answer in 1–2 sentences plus the killer line. Claude gives one piece of feedback per answer, then moves on. **No long discussions during the drill** — that defeats the time-box.

| # | Drill question | Look in cheatsheet section |
|---|---|---|
| **D1** | *"Walk me through why composed-weight KKT fails — in one minute."* | Q1 |
| **D2** | *"Why does inverting $\Delta W$ avoid that problem?"* | Q2 |
| **D3** | *"What exactly does the attacker know? Including labels."* | Q3 + Q3b |
| **D4** | *"Why is LoRA $r{=}8$ stuck at ~0.56 mean SSIM if full FT gives 0.997 — is it information or optimization?"* | Q4 |
| **D5** | *"How does this connect to my SVS paper? What's the obstruction PEFT introduces?"* | Q5 |
| **D6** | *"Three directions — which one do *you* want to push, and why?"* | Q5 ("Three directions…") |

**Reserve questions if time:**
- *"You showed N=3 — doesn't that just blend into the centroid?"* → Q6
- *"Isn't slide 9 just class memorization?"* → Q4b
- *"How does the R2F decoder actually get trained?"* → Q7
- *"You said TV is the lever, not the gradient — why are you sure?"* → Q8
- *"Walk me through the attack-taxonomy table on slide 3."* → Q9

**Drill grading rubric (for Claude, internal):**
- Killer line delivered + headline number cited = **green**.
- Right concept but fuzzy on number or framing = **yellow** (give one sharpening sentence).
- Missing or wrong = **red** (give the killer line back, move on).

Target after drill: ≥4 green, ≤1 red.

---

## Deck navigation (slide → what to lead with)

**Current deck: 29 slides, all visible.** Parts I / II / III (clean — no more missing-divider risk).

| # | Slide title | What to say | Theory link |
|---|---|---|---|
| 1 | Title | Soft supervision ask + Oct deadline (~30s) | — |
| 2 | Where we are in 30 seconds | 4-quadrant flyover, pause for interrupts | — |
| **3** | **Attack taxonomy: what does the attacker see? (NEW)** | ⭐ Framing beat. 4 rows: ∇L (done, ViT), ΔW (done, NTK MNIST), (A,B) (proposed, R2F bridge), W₀+BA (failed, structural). "The thesis is row 3 — the LoRA adapter setting." | Frames the whole talk |
| 4 | Bridge: from unlearning to reconstruction (R2F) | "Same machinery R2F uses for forgetting, we use for reconstruction." | Direction 1 setup |
| 5 | R2F algorithm: how f_φ is trained | 4-step recipe; emphasize *proxy ≠ victim data is OK*. Don't dwell — flag the two open questions at the bottom. | Direction 1 mechanics |
| 6 | How to read SSIM (tutorial) | **Don't dwell.** Point at the 0.30 gate, mention "kornia window=3", move on. | — |
| 7 | ▷ Part I — NTK reconstruction on MNIST | Pause beat. | — |
| 8 | NTK on MNIST — recognizable reconstruction | Credit him out loud. Full FT 0.990/0.998, LoRA r=8 0.686/0.496. 47× compression → ~0.3 SSIM drop, still well above gate. | Single-step NTK |
| **9** | **Instance-level leakage: not class memorization** | ⭐ Must deliver. "Δ = +0.22 / +0.18 over same-class control — real per-instance leak, not 'recovered a digit-5'." | Defuses class-memorization objection |
| 10 | NTK survives multi-step fine-tuning | "Smooth activations preserve the linearization through T=100." | Multi-step NTK |
| 11 | NTK reconstruction at T = 10 steps | Full FT decays 1.00 → 0.80; LoRA r=8 nearly flat 0.80 → 0.77 (side panel). S1=0.73, S2=0.75 at T=10. | Pixel-space decay |
| 12 | How well does the NTK assumption hold? | Feature stability quantified. ReLU breaks ≥T=50; LeakyReLU stable to T=100. LoRA tracks full FT within 0.01–0.03. | ⭐ **Q-A motivator** |
| **13** | **Batches and epochs: sample-weighted residuals (RETITLED)** | Full math: c′_i = Σ_t 𝟙[i ∈ ℬ_t] · c_i(t). "Under stable residuals, c′ ≈ n · c — same inverse problem, sample weights." | Extends NTK to stochastic FT |
| 14 | Multi-seed validation | "50 random inits. Recon mean 0.56. 100% beat same-class (0.38) and cross-class (0.39)." | Reliability |
| 15 | ▷ Part II — Scaling to ViT-B/16 | Pause beat. | — |
| 16 | ViT gradient inversion: gate crossed | Gate ≈ 0.3, best ≈ 0.55, median 0.35–0.45. cos_sim ≈ 0.95 across all configs — **the lever is the prior, not gradient match**. | Scaling NTK→ViT |
| 17 | A face is recoverable from a ViT gradient | ⭐ Visual hook. face1=0.52, face2=0.67, face3=0.58 — all clear the gate. Real OOD portraits. | Practical attack |
| 18 | Multiple images: no collapse, but imperfect identity separation | N=3 SSIM 0.60/0.67/0.71, mean 0.66. **But:** best-match is [1,1,2] not [0,1,2] — partial centroid attraction. (Cross-matrix on slide 28.) | ⭐ **Q on N>1 separability** |
| 19 | Why not just extend Haim et al.? | ⭐ KKT equation re-rendered. "Two-image extraction cannot explain weights encoding ~110 support vectors." | Motivates Direction 1 |
| 20 | ▷ Part III — Open theoretical questions | "Two questions for you." | — |
| 21 | Q-A: When does noise destroy reconstruction? | Lipschitz framing of ℛ: g → x̂. Sets up next two slides. | ⭐ **Q-A formal** |
| 22 | Q-A evidence: cos_sim saturates | 28 D2 configs. cos 0.94–0.96 saturated. SSIM 0.18–0.55. ρ(cos, SSIM)=+0.43 weak. **TV is the lever.** | Empirical motivator |
| 23 | Q-A evidence: 4 corner configs | Punchline of slide 22. cos≈0.94 across all 4; SSIM 0.10→0.55. | Drives Q-A home |
| 24 | Q-B: Pre-train / fine-tune overlap | Feature-map injectivity hook: φ(x) = ∇Φ(θ₀; x). | ⭐ **Q-B formal** |
| 25 | What we built + what landed | "Built end-to-end from scratch." Mention overnight: 5-arm face-prior ablation, multi-seed face1, chroma-TV. | Work shown |
| 26 | Three directions on independent axes | Information / Dynamics / Prior. **★ slide marks Gradient Bridge as my pick** — ask him to overrule or confirm. | Roadmap |
| 27 | Backup — headline numbers | Only show if he asks for specifics. | — |
| 28 | Backup — N=3 cross-matrix | Pull up if he probes superposition. SSIM(recon_i, GT_j) diagonal does NOT dominate — [1,1,2] match. | Q on N>1 separability |
| 29 | Backup — NTK plot: feature stability + quality vs T | Pull up if he wants to dwell on the T sweep itself. Duplicates slide 11 figure but standalone. | — |

---

## Morning checklist (review when you wake up)

1. ✅ **Deck final at v18** — [figures/supervisor_meeting_2026_05_14_v18.pptx](figures/supervisor_meeting_2026_05_14_v18.pptx). **29 slides, all visible**. Parts I / II / III labelled (clean). Don't edit.
2. ✅ **Figures bug-fixed** — kornia SSIM window=3, ImageNet denorm, ds_mean. All embedded in v18.
3. ✅ **Tier-1 paper notes ready:** [notes/tier1_paper_notes.md](notes/tier1_paper_notes.md) — SVS 2025 + Gronich-Vardi. Skim THIS, not the PDFs.
4. ✅ **Use the correct SVS PDF:** [papers/Smorodinsky_Vardi_Safran_2025_Provable_Privacy_Attacks_2410.07632.pdf](papers/Smorodinsky_Vardi_Safran_2025_Provable_Privacy_Attacks_2410.07632.pdf) (the old `..._2024_...` file was misnamed and contained an unrelated paper — renamed `MISNAMED__do_not_use__...`).
5. ⏰ **Stick to the 45-min protocol above.** No deck edits Thursday morning.
6. 🆕 **Memorize the v18 additions:**
   - **Slide 3 attack taxonomy:** 4 rows = ∇L (done) / ΔW (done) / (A,B) (proposed) / W₀+BA (failed). The thesis is row 3.
   - **Slide 9 instance lift:** Δ = **+0.220 / +0.182** over same-class control (S1 / S2)
   - **Slide 11 T=10 decay:** full FT 1.00 → 0.80, LoRA r=8 nearly flat 0.80 → 0.77
   - **Slide 13 sample-weighted residuals:** c′_i = Σ_t 𝟙[i ∈ ℬ_t] · c_i(t); under stable c, reduces to n_i · c_i
   - **Slide 22 Q-A spread:** cos_sim 0.94–0.96 saturated, SSIM 0.18–0.55, ρ = +0.43

---

## Q1 — Why does composed-weight KKT fail? (slide 19)

> *"It's a 3-part answer — numerical, practical, structural."*

**Numerical.** Set $N=2$ in the extraction → KKT residual bottoms at $\|W_0\|^2$ because the composed weights satisfy KKT over ~110 effective support vectors. Two-image extraction can't account for that floor.

**Practical.** Setting $N=110+$ would in principle close the residual, but Haim et al. needed 1M epochs at $N=500$ on MNIST. We don't have that compute headroom on LoRA-scale models.

**Structural (the one for him).** Even if a large-N extraction worked perfectly, you can't tell *which 2 of 110* are the fine-tuning targets. The composition $W = W_0 + BA$ destroys the fine-tuning/pre-training partition. To filter, you'd need access to the pre-training data. **That's why we pivoted to inverting $\Delta W$ directly — it preserves the partition by cancelling $W_0$.**

**Confirmed empirically:** Sprint 2c Track A, 48 configs, KKT loss stuck at 330–350 regardless of $N$.

**Killer line:**
> *"Composed-weight KKT mixes fine-tuning and pre-training support vectors — you can recover ~110 candidates in principle but you can't tell which 2 are the fine-tuning targets without the pre-training data. $\Delta W$ preserves the partition by cancelling $W_0$. The attack-taxonomy table on slide 3 captures this — row 4 is structurally failed; rows 2–3 are where the leverage is."*

---

## Q2 — Why does NTK on $\Delta W$ avoid the same problem? (slides 8–14)

> *"$\Delta W = -\eta \sum_i c_i \nabla\Phi(\theta_0; x_i)$, sum over the fine-tuning batch only."*

**The math (write on his whiteboard):**

```
Linearization:        Φ(θ₀ + Δθ; x) ≈ Φ(θ₀; x) + ⟨∇Φ(θ₀; x), Δθ⟩
Feature stability:    ∇Φ(θ₀ + Δθ; x) ≈ ∇Φ(θ₀; x)
One SGD step:         Δθ = -η · Σᵢ cᵢ · ∇Φ(θ₀; xᵢ)    [sum over fine-tuning batch ONLY]
Inverse problem:      argmin_{x̂, ĉ}  ‖ Δθ + η Σᵢ ĉᵢ ∇Φ(θ₀; x̂ᵢ) ‖²
```

**Multi-step extension (slide 13 — refined in v18):**
```
Δθ ≈ -η Σᵢ c′ᵢ · ∇Φ(θ₀; xᵢ)
   c′ᵢ = Σ_t 𝟙[xᵢ ∈ ℬ_t] · cᵢ(t)         [full form: sample-weighted residual]
   Under stable residuals (cᵢ(t) ≈ cᵢ):   c′ᵢ ≈ nᵢ · cᵢ
   nᵢ = #{t : xᵢ ∈ ℬ_t},  𝔼[nᵢ] = T·|ℬ|/N    (uniform sampling)
```
**The inverse problem is structurally identical** — just sample weights instead of a single residual. Attack pipeline unchanged.

**Feature stability (measured, slide 12):** ReLU breaks at $T \geq 50$ (Jacobian discontinuous at kinks); LeakyReLU / ModifiedReLU stable through $T=100$. SSIM holds ≈ 0.8 at T=100. LoRA r=8 tracks full FT within 0.01–0.03 SSIM.

**Sprint results (slides 8, 11, 27):**
- Full FT MNIST: SSIM = **0.990 / 0.998** (individual); 0.997 multi-seed mean.
- LoRA r=8: **0.686 / 0.496** individual; **0.557** mean of 50 seeds.
- At T=10: S1=0.73, S2=0.75 (still well above gate).

**Killer line:**
> *"$\Delta W$ is literally the gradient at $\theta_0$ evaluated on the fine-tuning batch. The pre-training data lives inside $\theta_0$ and is invariant when we subtract. Multi-step doesn't break this — it just turns single-residual c_i into sample-weighted c′_i, same inverse problem."*

---

## Q3 — Threat model (anchored to slide 3 attack taxonomy)

**Attacker sees:** $\theta_0$ (public HuggingFace base), either $\theta_1$ (full FT) or LoRA $(A, B)$ — distinguishable by file format. Architecture + training hyperparameters $(\eta, T, \text{batch})$ known or efficiently swept.

**Attacker does NOT see:** the fine-tuning images $\{x_i\}$, the number $N$, the coefficients $c_i$, the specific label assignments.

**Goal:** recover at least one $\hat{x}_i$ that's identification-faithful (face recognition matches, license plate readable, etc.). Pixel-faithful is the upper bound; identification alone is the privacy violation.

**Label space:** public (binary vs K-class is on the model card). Label *assignments* are recovered jointly with $\hat{x}_i$ via free-c — see Q3b below.

**Killer line:**
> *"Attacker has the weights, not the data. Sees $\theta_0$ + ($\theta_1$ or LoRA $A, B$). $\eta, T$, batch size known or swept. Labels: space known, assignments jointly recovered. Goal: identification-faithful $\hat{x}$, not necessarily pixel-perfect. Slide 3 makes the four possible attack objects explicit."*

### Q3b — What about $y_i$? Are labels known?

**Short version:** *"Space known, assignments jointly recovered. We don't assume oracle label access."*

Three regimes:

**(1) Sprint 1/2 (NTK on MNIST MLP, binary BCE).** $c_i = \tfrac{1}{N}(\sigma(f(\theta_0; x_i)) - y_i)$. For binary $y_i \in \{0, 1\}$, the sign of $c_i$ encodes the label. **Free-c extraction picks up the label from $\text{sign}(\hat{c}_i)$ as a byproduct — labels are jointly recovered, not oracle.**

**(2) Phase 0 (gradient inversion on ViT-B/16).** The observed gradient $g_{\text{obs}} = \nabla L_{CE}(\theta_0; x, y_{\text{target}})$ depends on the target label. The inversion objective evaluates $\nabla L(\theta_0; x, y_{\text{target}})$ at trial $x$, so $y_{\text{target}}$ must be supplied. **Attacker supplies $y_{\text{target}}$** — either reads model's prediction on similar public inputs, or brute-forces over $K$ classes (cheap when $K \leq 10$).

**(3) Realistic LoRA attack.** Class space is public (model card). Specific per-sample labels are not. Binary case: free-c recovers labels alongside images. Multi-class case: generalize $c_i$ to softmax + cross-entropy; not yet tested empirically.

**Edge case to flag if pushed:** class imbalance. If 19 of 20 fine-tuning samples are class $+1$, $\text{sign}(\hat{c}_i)$ trivially leaks labels but minority-class recovery degrades because the gradient signal is dominated by the majority direction. Not a thesis-killer but a known degradation regime.

**Sentence for the meeting:**
> *"Labels are jointly recovered as an additional discrete variable in the free-c optimization — inferred from $\text{sign}(\hat{c}_i)$ in the binary case. The label space is public from the model card; assignments are not assumed known. In Phase 0 gradient inversion, the attacker supplies a target class — either the model's prediction on similar public inputs, or brute-forced over $K$. We don't assume oracle label access."*

---

## Q4 — Why is LoRA SSIM lower than full FT? (slides 8, 27)

> *"At $r \geq 8$ on MNIST, it's optimization-limited, not information-theoretic."*

**v18 numbers to have ready:**
- Full FT MNIST: SSIM **0.990 / 0.998** (two examples, slide 8); **0.997** multi-seed mean.
- LoRA r=8 individual: **0.686 / 0.496** (S1 / S2, slide 8) — wide per-instance variance.
- LoRA r=8 multi-seed mean: **0.557** (n=50, slide 27).
- LoRA r=32 free-c: **0.680** (slide 27).
- 47× parameter compression → ~0.3 SSIM drop, still above the 0.30 gate.

**The argument:**
- LoRA gradient: $\nabla_A L = B^T \nabla_W L$, $\nabla_B L = (\nabla_W L) A^T$.
- At $r=8$ on a 1000-d MLP layer: ~$1.6 \times 10^4$ scalars in the projected gradient (~38K total params).
- MNIST intrinsic dim: ~10–20.
- → System is **overdetermined by 3 orders of magnitude** in intrinsic-dim terms. Information is there.

**The empirical signature: rank-sweep plateau.**

| $r$ | SSIM (SGD+LeakyReLU) |
|-----|----|
| 4 | 0.509 |
| 8 | 0.617 (single seed) / 0.557 (mean of 50) |
| 16 | 0.624 |
| 32 | 0.680 |
| 64 | 0.635 |

If the bottleneck were info-theoretic, SSIM should climb monotonically with $r$ toward 0.997. It doesn't — plateaus at 0.6–0.68, slightly regresses at $r=64$. **That's an optimization-limited regime.** Adding rank doesn't help; we're stuck in a non-global local minimum of the inverse problem.

**Where info-theoretic regime starts:** below $r \approx 2$–$4$, the rank-$r$ projection compresses enough that distinct images map to similar gradients — invertibility breaks. For high-intrinsic-dim data (faces, ~100–1000 dims), threshold rank is much higher.

**Bonus connection for him:** the optimization-limited regime is exactly where image priors (TV, LPIPS, SDS) bite — they reshape the loss landscape and lift the optimizer out of bad local minima. Conceptual hook for the diffusion-prior direction. **This is what slides 22–23 demonstrate empirically on the ViT side: cos_sim saturates 0.94–0.96, but SSIM ranges 0.18–0.55 depending on the TV weight.**

**Killer line:**
> *"r=8 on MNIST: optimization-limited. ~16K LoRA gradient scalars for ~10–20 intrinsic dims = overdetermined by 3 orders of magnitude. Multi-seed mean SSIM 0.557, plateau at r=8→64 (0.62→0.68) is the signature. Adding rank doesn't help, so info is there but we can't extract it. Info-theoretic regime only kicks in below r≈4. Slides 22–23 are the ViT-side analog of the same story."*

---

## Q4b — Why is instance-level leakage on slide 9 a big deal?

> *"Because 'recovered a digit-5' is class memorization, but 'recovered THIS digit-5 better than 20 random other 5s' is per-instance privacy leakage."*

**Setup:** for each LoRA reconstruction, compare SSIM(recon, true GT) vs mean SSIM(recon, 20 random same-class GTs).

**Result:** Δ = **+0.220 / +0.182** (S1 / S2). Both positive → reconstructions match their own GT more strongly than the class average → instance-specific leak.

**Why this matters:** distinguishes a real privacy attack from a model that just "knows what 5s look like." A regulator / privacy auditor will ask exactly this question.

**Killer line:**
> *"The reconstruction matches THIS 5 by Δ = 0.22 SSIM more than the average of 20 random 5s. That's not class memorization — that's a per-instance privacy leak."*

---

## Q5 — Connection to SVS 2024 + direction (slide 26)

> ⚠️ **NOTE:** The local `papers/` had the SVS 2024 PDF *misnamed*. Use [papers/Smorodinsky_Vardi_Safran_2025_Provable_Privacy_Attacks_2410.07632.pdf](papers/Smorodinsky_Vardi_Safran_2025_Provable_Privacy_Attacks_2410.07632.pdf) and the auto-digest in [notes/tier1_paper_notes.md](notes/tier1_paper_notes.md).

> *"Honest: I've skimmed your paper, my best guess at the connection is X — tell me where I'm wrong."*

**Best-guess connection:**
> *"Your provable reconstruction guarantees rely on homogeneity. PEFT breaks homogeneity — $W_0$ is fixed during fine-tuning, only $\Delta W = BA$ is trained. That's exactly why composed-weight KKT fails for me. The $\Delta W$ / NTK approach restores something like homogeneity in the linearized regime — but I'd want your read on whether there's a precise 'restricted homogeneity' condition that PEFT satisfies, under which an SVS-style theorem still holds."*

**Three directions on independent axes** (slide 26 — note the ★ on Gradient Bridge):

1. **Information** — Gradient Bridge: R2F-style decoder $f_\phi : (A, B) \to \nabla_W L$, then feed into Phase 0 inversion. End-to-end attack. **★ My pick. ~2–3 wk cost.** Risk: decoder must generalize off proxy data.
2. **Dynamics** — NTK extension: differentiable unrolling or higher-order corrections for $T > 1$. ~1 wk cost. Low risk — machinery already built. Theory payoff, smaller new result.
3. **Prior** — Diffusion / SDS for very-low-rank LoRA. ~3–4 wk cost. GPU-heavy; prior may dominate signal. Strongest visual results if it works.

**Separately, the theory question:** *"When does an approximate gradient still support reconstruction?"* — grounded in slides 22–23's 3× SSIM spread at saturated cos_sim. Could anchor a thesis chapter regardless of which direction is picked.

**Pre-answer on NTK→ReLU:**
> *"ReLU breakdown at $T>1$ is structural — Jacobian discontinuous at kinks. LeakyReLU was a stepping stone, won't carry into transformers (they use GELU). Phase 0 / Gradient Bridge sidesteps NTK linearization entirely. So the NTK→ReLU question is moot if we lead with Direction 1. It matters only if Direction 2 becomes its own chapter."*

**Killer line:**
> *"Honest about not having read SVS 2024 in depth — best-guess connection: their guarantees assume homogeneity, PEFT breaks it, $\Delta W$ restores it in the linearized regime. Three directions on independent axes (Information / Dynamics / Prior), Gradient Bridge is my pick, awaiting your prioritization."*

---

## Q6 — "Doesn't N>1 cause superposition collapse?" (slides 18, 28)

**Short answer:** *"Empirically, partial — and the cross-matrix on slide 28 makes it precise."*

**What we measured (N=3 same-person joint inversion, face1+2+3):**

- Per-image SSIM: recon[0]=0.60, recon[1]=0.67, recon[2]=0.71, mean **0.662** (vs face1 N=1 multi-seed mean ≈ 0.56).
- **But:** best-match for recon[i] is NOT always GT[i]. Best-match assignment is **[1, 1, 2]** instead of [0, 1, 2]. Recon P1 matches GT P2 (0.71) more than GT P1 (0.60).
- The 3 reconstructions are more similar to each other than the 3 GTs are.

**Interpretation.** Joint inversion has an attractor at the centroid of the target set. For binary classes the sign of $c_i$ separates them; for same-class targets it does not.

**Ideas to separate** (code-ready or a few hours of work):

1. **Diversity penalty (CODE READY).** `get_diversity_penalty(x, min_dist=0.5)` in `experiments/ntk_extraction.py` line 439. Adapted from Haim et al. Sigmoid-smooth repulsive penalty when reconstructed-image pairs get too close. **Implemented but not yet wired into the inversion loop.** Quick win for a follow-up.
2. **Sequential peeling.** Reconstruct image 1 → fix it → reconstruct image 2 from the residual gradient → iterate. Bypasses the joint-optimization centroid trap.
3. **ICA on FC layer gradients (Cocktail Party Attack, ICML 2023).** Each FC row is a linear mixture of N sources. FastICA separates them up to N ≤ layer width. Scales to N=1024 for fully connected. Open for ViT.
4. **Class-based grouping (binary case).** Use sign of $\hat{c}_i$ to assign labels; reconstruct each class block separately.

**Killer line:**
> *"Partial collapse, yes — the centroid attractor. Mean per-image SSIM 0.66 clears the gate, but the cross-matrix shows recon-to-GT misalignment. The v18 slide title is honest: 'no collapse, but imperfect identity separation.' Diversity penalty is already in the codebase, not yet wired in. Cleaner long-term fix is ICA on FC gradients (Cocktail Party Attack); ICA scales to N=1024 in the FC case, the ViT case is open."*

---

## Q7 — "The R2F decoder — how does it actually train?" (slide 5)

If he probes after slide 5. Walk the 4-step recipe:

1. **Proxy data** — public dataset; need not match victim (CIFAR-100 proxies for faces).
2. **Per batch** — one LoRA fine-tuning step on a fresh base model. Record (A, B) as input, true $\nabla_W L$ as target.
3. **Repeat ~50K times** → 50K (adapter, gradient) pairs.
4. **Train one small MLP per layer.** Loss = cosine similarity (direction is what downstream inversion needs).

**Two open questions** (also on the slide):
1. Does $f_\phi$ generalize from proxy to victim data?
2. How much decoder noise does inversion tolerate? ← exactly the Q-A question on slide 21.

**Compute estimate:** ~50K LoRA steps × per-layer decoder × N layers. For ViT-B/16 (12 transformer blocks, 4 LoRA targets each = 48 layers): on a single L40S, ~2–3 wk wall-clock. **This is the slide-26 Direction-1 cost estimate.**

**Killer line:**
> *"Proxy data, 50K LoRA steps, one MLP per layer, cosine-similarity loss. R2F already shows the inverse is learnable for unlearning; the open question is whether the learnt decoder is faithful enough that downstream gradient inversion still recovers pixels."*

---

## Q8 — "TV is the lever, not the gradient — explain." (slides 22, 23)

If he asks why we are confident in this framing:

**Setup:** 28 D2 ViT configs on face1, varying (TV weight, frequency-domain weight, learning rate). All configs have access to the *same* gradient signal.

**Observation:**
- cos_sim(decoded gradient, true gradient) is in **[0.94, 0.96]** across all 28 — saturates.
- SSIM(recon, GT) varies **0.18 → 0.55** — wide.
- Pearson ρ(cos_sim, SSIM) = **+0.43** (weak).

**Interpretation:** the gradient-match metric (cos_sim) is nearly *constant* across the sweep, yet reconstruction quality varies 3× SSIM. The thing that *does* track SSIM is the TV penalty (middle panel of slide 22). **Therefore: the inversion is prior-limited, not gradient-limited.** This is the empirical motivator for the Lipschitz framing on slide 21.

**Bridge to Direction 3 (diffusion prior):** if a hand-tuned TV penalty buys you 0.18 → 0.55 SSIM at constant gradient, a learned generative prior (SDS / DDIM) should do much better. That's the Direction-3 thesis bet.

**Killer line:**
> *"28 configs, gradient match saturated at cos≈0.95, SSIM ranges 3×. The lever is the prior, not the gradient. That's exactly the regime where a learned image prior — diffusion / SDS — should buy us another big jump."*

---

## Q9 — "Walk me through the attack-taxonomy table on slide 3." (NEW in v18)

If he asks why this slide was added — the framing matters.

**The 4 rows:**

| Object attacker has | Evidence in this deck | Status |
|---|---|---|
| **Full gradient ∇L** (federated / leak setting) | ViT gradient inversion on Flowers + faces (Part II). Best face SSIM 0.52 / 0.67 / 0.58. | ✓ done |
| **Fine-tuning update ΔW** (full FT or LoRA, decomposed) | NTK reconstruction on MNIST (Part I). Full FT SSIM 0.99; LoRA r=8 SSIM 0.50–0.69. | ✓ done |
| **LoRA adapter (A, B)** (the actual PEFT artifact) | R2F bridge: train decoder f_φ : (A, B) → ∇L (slide 5). | → proposed |
| **Composed weights W₀ + BA** (final fine-tuned model) | Haim-style KKT extraction (slide 19). | ✗ failed (structural mismatch) |

**Why this slide matters:** it makes the threat model explicit *before* he sees results, and it positions every later slide on this taxonomy. Row 3 — the LoRA adapter — is **what the thesis is actually about**. Rows 1, 2, 4 are the surrounding context.

**Conversational use:**
- If he asks "why is the realistic threat the LoRA adapter, not the full gradient?" → row 3, the actual PEFT artifact is what gets published / shared.
- If he asks "what about model extraction from W₀+BA?" → row 4, structurally failed, see slide 19 + Q1.
- If he asks "what's the path from row 3 to row 1?" → that's the R2F bridge (slides 4–5), the Gradient Bridge direction (slide 26).

**Killer line:**
> *"Four possible attack objects — different inverse problems, different identifiability. The thesis lives in row 3, the LoRA adapter. We have evidence for rows 1 and 2 already; row 4 fails structurally; row 3 is the open and most realistic threat — and the R2F bridge is the path to attacking it."*

---

## Reserve theory questions (deploy if he engages deeply)

- **A. Noise tolerance.** When is gradient-inversion map $\mathcal{R}: g \mapsto \hat{x}$ Lipschitz? — **now anchored to slides 21–23 empirics**.
- **B. Distribution overlap.** When $\mathcal{D}_{pre} \cap \mathcal{D}_{ft} \neq \emptyset$, does $\nabla\Phi(\theta_0; x_{ft})$ encode $x_{ft}$ absolutely or only $x_{ft} - \mu_{pre}$? — slide 24.
- **C. PEFT extension of SVS 2024.** "Restricted homogeneity" condition that PEFT satisfies, under which SVS-style theorems hold? — **the killer move if he asks about SVS**.
- **D. Composition-vs-partition.** When does $W = W_0 + BA$ preserve the fine-tuning/pre-training partition in the KKT framework?
- **E. NTK feature stability across architectures.** Bounds on $\cos(\nabla\Phi(\theta_0), \nabla\Phi(\theta_T))$ as a function of $T$, $\eta$, activation Lipschitz? — slide 12.
- **F. Phase transition.** Threshold rank $r^*$ where LoRA inversion becomes info-theoretically impossible? Compressed-sensing analog.
- **G. Gradient projection invertibility.** When is $(B^T \nabla L, \nabla L \cdot A^T)$ jointly invertible to $\nabla_W L$? — formal version of the Q7 / slide 5 decoder.
- **H. Proxy-data generalization for R2F decoder.** Distribution-shift theory for the decoder. — open question on slide 4.

---

## Headline numbers (have these cold)

| Result | Number | Where in deck |
|---|---|---|
| NTK MNIST full FT | SSIM **0.990 / 0.998** | slide 8 (individual S1/S2) |
| NTK MNIST full FT (multi-seed mean) | SSIM **0.997** | slide 27 |
| NTK MNIST LoRA r=8 individual | SSIM **0.686 / 0.496** | slide 8 |
| NTK MNIST LoRA r=8 (mean of 50) | SSIM **0.557** | slide 27 |
| NTK MNIST LoRA r=32 free-c | SSIM **0.680** | slide 27 |
| Instance-level lift (LoRA vs same-class control) | Δ = **+0.220 / +0.182** | slide 9 |
| NTK at T=10 | SSIM **0.73 / 0.75** | slide 11 |
| Feature-stability decay (full FT, T=1→100) | SSIM **1.00 → 0.80** | slide 12 |
| Feature-stability decay (LoRA r=8, T=1→100) | SSIM **0.80 → 0.77** (nearly flat) | slide 12 |
| Multi-seed face1 mean | SSIM **0.56**, 100% beat ctrl 0.38 / 0.39 | slide 14 |
| ViT-B/16 Flowers102 D2 best | SSIM **0.548**, PSNR **15.1 dB** | slides 16 / 27 |
| ViT face1 / face2 / face3 | SSIM **0.52 / 0.67 / 0.58** | slide 17 |
| ViT N=3 same-person joint | mean SSIM **0.662**, best-match [1,1,2] | slides 18 / 28 |
| KKT compose residual floor | loss **330 – 350** (48 configs, all N) | slide 27 |
| Pre-training effective support vectors | **~110** of 500 | slide 19 |
| Q-A spread (D2, 28 configs) | cos_sim **0.94–0.96**, SSIM **0.18–0.55**, ρ=**+0.43** | slide 22 |
| D1 cos_sim spread at fixed gradient | cos_sim **0.887–0.934** | slide 27 |
| ViT-B/16 param count | **86×10⁶** | slide 15 |
| LoRA r=8 param count (MLP) | ~**1.6×10⁴** scalars/layer (~38K total, 47× compression) | slide 8 |

---

## Hard deadlines

- **2026-10-01** — supervisor must be locked in (university deadline)
- **2026-10-31** — thesis proposal must be submitted

Backward: supervision formalized by ~early July leaves ~3 months for proposal drafting with him.

---

## Conversational tactics

- **Slide 3 (attack taxonomy) is your framing beat** — spend a real moment on it before R2F. Use it to position every later result. If he interrupts on row 3, that's actually the best place for him to go.
- **Lead with NTK on slide 8** (after the R2F bridge slides 4–5 and SSIM intro slide 6) — credits him for the suggestion explicitly.
- **Don't dwell on slide 6 (SSIM tutorial)** — point at the 0.30 gate, mention "kornia, window=3", move on. It's there for him to consult, not for you to teach.
- **Slide 9 (instance leakage) is the slide that defuses "isn't this just class memorization?"** — make sure he sees the +0.22 Δ.
- **Slide 23 is the punchline of slide 22** — show 22 to motivate the question, 23 to drive it home.
- **Don't apologize for the gap** once it's acknowledged in the opening (one mention only).
- **Don't read the math at him on slide 19** — say the sentence: *"two-image extraction cannot explain weights that encode ~110."*
- **If he asks about lr/TV/batch/optimizer details** — redirect to the more interesting question. Don't go down the implementation rabbit hole.
- **If he asks about Gronich-Vardi 2026 → optimizer-norm connection** — acknowledge it, note he said ℓ∞ is impractical, don't pursue.
- **Exception clause** (if he opens cool/distant) — invert: bring up supervision earlier, treat as relationship-repair-first.
