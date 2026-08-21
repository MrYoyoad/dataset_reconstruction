# MineGrad vs. the Gradient Bridge — teardown, mapping, and re-plan

**Paper:** *MineGrad: Gradient Inversion Attacks on LoRA Fine-Tuning*, Hasin Us Sami\*, Swapneel
Sen\*, Başak Güler. AISTATS 2026 (PMLR vol. 300). arXiv:2608.01521. Code:
`github.com/info-ucr/MineGrad`. PDF archived at
`papers/Sami_Sen_Guler_2026_MineGrad_LoRA_Gradient_Inversion_AISTATS.pdf`, text at
`papers/MineGrad_2026_fulltext.txt`.

**Author lineage / disambiguation.** MineGrad is the **LoRA-specific successor** to the paper
already in `papers/` — *Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning* (Sami,
Sen, Roy-Chowdhury, Krishnamurthy, Güler; arXiv 2506.04453; code **PEFTLeak**), which attacks
generic *adapter* modules that are followed by a nonlinear activation. Our CLAUDE.md mislabels
2506.04453 as "CVPR 2025 / honest, easier-than-FT"; it is in fact a **malicious-server** attack.
MineGrad drops two authors, targets **LoRA** specifically, and — this is its headline — removes the
requirement of a post-adapter activation and the DAGER token≤rank constraint.

Written 2026-08-21 in response to a request to rethink the gradient-bridge direction against MineGrad.

---

## 1. The MineGrad mechanism, algebraically

**Threat model (stronger than ours).** Malicious FL server. It (i) ships a **poisoned frozen
pretrained model** once before training, and (ii) ships **poisoned global LoRA matrices** `A,B` each
round. The victim runs *ordinary* LoRA fine-tuning on private data and returns `∂L/∂{A,B}`. The
server inverts those gradients analytically. No optimization, no proxy training, one shot, one round.

Notation: token `n` has (frozen) word embedding `x⁽ⁿ⁾∈ℝ^D`, position encoding `e⁽ⁿ⁾∈ℝ^D`, combined
`y⁽ⁿ⁾ = x⁽ⁿ⁾ + e⁽ⁿ⁾`. `D` embed dim, `H` heads, `D̄=D/H`, `N` tokens, `r` LoRA rank, `S` encoders.

### The five engineered tricks

**Trick 1 — position encodings become orthogonal 2-hot fingerprints (poisoned frozen model), Eq 4.**
The server overwrites `e⁽ⁿ⁾` to `+c₁` at coord `2n+1`, `−c₁` at `2n+2` (head 1), and small `±c₂` at
the analogous coords of heads 2..H, everything else 0; `c₁∼10²`, `c₁≫c₂`. This makes positions
mutually near-orthogonal *within each head*: `(e⁽ⁱ⁾)ₕᵀ(e⁽ⁱ⁾)ₕ ≫ (e⁽ⁱ⁾)ₕᵀ(e⁽ʲ⁾)ₕ`, `i≠j` (Eq 5).
Because real word embeddings are small (`x∈[−1,1]`), `e` dominates the statistics of `y`
(mean ≈0, std ≈ constant σ). *Code:* `Design_Model_LORA.py::Pos_Encoding.tampering` writes exactly
this 2-hot-per-position pattern into `embedding.position_embeddings`.

**Trick 2 — identity-mapping self-attention (poisoned frozen model + zeroed QKV-LoRA), Eqs 18–22.**
Set `W_Q=W_K=W_V=I_{D×D}`, biases 0, and the Q/K/V LoRA to 0. Then attention logits are governed by
the orthogonal position fingerprints, so `softmax(QKᵀ/√D̄) ≈ I_{(N+1)×(N+1)}`: **each token attends
only to itself**. With `V=I`, the MSA output is the identity map of the inputs. *Code:*
`attack_parameter` sets `Q_head = 10⁵·I` per head (the huge scale forces softmax→hard delta),
`K_head=V_head=I`. The output projection `W_O` is set to `0`, so the only thing acting on the
(preserved) token stream is the trainable `ΔW_O = B_O A_O`.

**Trick 3 — the LoRA `A` matrix is a coordinate-selector (poisoned LoRA init), Eq 9.**
`A_{O,t}∈ℝ^{r×D}` is set so `A_{O,t}[p,q]=1` iff `p=n, q=2·T_t(n)+2` — row `n` of `A` reads out
coordinate `2·T_t(n)+2`, the fingerprint coordinate of the token sitting at target position `T_t(n)`.
`B_{O,t}=0`. So `(A_{O,t} y⁽ⁱ⁾)_n = y⁽ⁱ⁾_{2T_t(n)+2}`, which is large (`≈ −c₁`) **only when
`i=T_t(n)`** and ≈0 for every other token. `A` collapses the whole sequence into an `r`-vector whose
`n`-th entry is dominated by the single token at position `T_t(n)`.

**Trick 4 — the `B` gradient reads out one token per column, Eqs 10, 14, 17.**
Because `B_O=0`, the forward pass is undisturbed (`BA=0`, residual passes `y` through). But the
gradient of the `n`-th **column** of `B` is
```
∂L/∂B_{O,t}[:,n] ≈ (z₁ − l₁) · c₃ · c₁²/((N+1)σ²D̄) · y^{(T_t(n))}          (Eq 17)
```
i.e. **proportional to the embedding of the token at target position `T_t(n)`, isolated.** The
isolation is the product `y⁽ⁱ⁾_{2T_t(n)+1}·y⁽ⁱ⁾_{2T_t(n)+2} = (+c₁)(−c₁) = −c₁²`, huge and
single-signed for the target token, negligible for the rest (Eq 14). Divide by the known scalar,
subtract the known `e^{(T_t(n))}` → recover `x^{(T_t(n))}`. *Code:* `reconstruction.py::recover1`
takes `weight_grad[0][:,t-1]` (column `t-1` of `∂B`), divides by `weight`(=c₃) · `div_factor` ·
`new_scaling`, subtracts the position embedding, clamps to `[−1,1]`. For text, `recover2` then
snaps to the nearest word by cosine over the vocab table.

**Trick 5 — last encoder averages all tokens into the class token, Eqs 13–16, 23.**
Encoder `S` sets attention to `(1/(N+1))·1·1ᵀ`, so the class-token embedding `a⁽⁰⁾` becomes the mean
of all token embeddings. Only the class token feeds the classifier, so the classification loss
depends on **every** token — making every `B`-column gradient non-zero. `W_CLS` (Eq 15) is set so
logit 1 is driven by the fingerprint coords, `softmax → z₁≈1`, giving the attacker a **known**
`∂L/∂q`. If the sample truly belongs to class 1 the gradient vanishes (`l₁=1`); the server just
targets a different class next round.

### Where `r` appears, and exactly why the attack is NOT limited to `r` tokens

- Per **single** matrix `B`, only `r` columns exist ⇒ **≤ r tokens per module** ("Key intuition",
  L385: *"Since B has rank r, at most r tokens can be revealed per encoder"*).
- **The bottleneck is broken by using many independent modules, each programmed to expose a
  *disjoint* set of ≤ r tokens.** Target encoders `1..S−1` each own a token set `T_t`; the identity
  forward pass (Trick 2) preserves the *same* undistorted embeddings all the way up, so every encoder
  re-measures the same tokens through a *different* coordinate-selector `A`. Both the `W_O` **and**
  `W_V` LoRA modules are usable per encoder (L491–493). So

  **recoverable tokens ≈ (#usable modules) × r ≈ 2·(S−1)·r**, *not* `r`.

  For a 12-layer transformer, `r=4` → ~`2·11·4 = 88` token-slots. Fig 6 is the money figure:
  **multi-encoder `r=2` matches single-encoder `r=16`.** Rank per matrix is not the leakage bound;
  **number of independent, coordinated low-rank views is.**

**This is the one sentence to carry away:** MineGrad does not make a rank-`r` matrix leak more than
`r` numbers. It manufactures `L` independent rank-`r` measurement operators (`L ≈ 2(S−1)`), each a
different coordinate-selective projection, and *tiles* the token set across them — the extra
information comes from **coordinated multi-module measurement of a forward pass engineered to keep
every sample individually resolvable.**

### Extra-information audit (the user's explicit checklist)

| Source of extra information | Used by MineGrad? | How |
|---|---|---|
| Multiple LoRA layers/encoders | **Yes — central** | disjoint token set `T_t` per encoder; tiles N over ~`(S−1)` blocks |
| Multiple `A`/`B` factors | **Yes, asymmetrically** | data is read from `∂B`; `A` is the fixed coordinate-selector that puts it there |
| Different modules (Q/K/V/O) | **Yes** | `W_O` and `W_V` are the two readout channels; Q/K only shape the identity attention |
| Specially selected rows/cols | **Yes** | `A[n, 2T_t(n)+2]=1` selects the fingerprint coord; column `n` of `∂B` is the readout |
| Multiple independent measurements | **Yes** | = the multi-module tiling above |
| Poisoned frozen weights | **Yes — essential** | orthogonal position fingerprints + identity attention + zeroed MLP/`W_O` |
| Engineered activation patterns | **Yes** | huge-`c₁` fingerprints make the softmax a hard delta and the `y_{2n+1}y_{2n+2}` selector huge |

### Vision, batches, defenses (from experiments)

- **Vision (ViT):** patches are the "tokens"; there is no vocabulary, so the recovered patch
  embedding is (a known linear map of) the pixels — invert the frozen patch projection to get pixels.
  Reported **LPIPS 0.20±0.04 (CIFAR-10), 0.21±0.06 (CIFAR-100)** over 100 images (Table 4) — i.e.
  recognizable but **not** pixel-perfect, and modest compared to their near-perfect text BLEU/ROUGE≈1.
- **Batch of M sequences (App D):** each `∂B` column now recovers the **average** `g=(1/M)Σ x⁽ᵐ⁾` of
  the M target-position embeddings. They de-mix it by a **top-M cosine search over the discrete
  vocabulary** (`E[gᵀv]=1/(3M)` if `v` in batch, 0 otherwise; boosted by a malicious high-variance
  word-embedding layer, App E.1). Batch 64 → up to 52% tokens (Table 3); seq-len 32 batch → ~80%.
  **This de-mixing has no vision analog** — a continuous average of M patches cannot be snapped to a
  codebook. MineGrad's batch trick is fundamentally discrete-token. **Their superposition solution
  does not transfer to images.**
- **Defenses (App E.2):** survives secure aggregation via *model inconsistency* (send different
  poisoned models to different users — inapplicable to a single honest release); degrades under
  gradient noise / clipping / pruning like any analytical attack.
- **vs DAGER (Petrov 2024):** DAGER tests whether vocab embeddings lie in the column space of the
  rank-deficient `∂W`, needs `N ≤ min(D,r)` and an exhaustive vocab search (no vision). MineGrad holds
  at `r=4 < seq-len 16` where DAGER collapses (Fig 9).

**Hyperparameters to steal for a baseline:** `c₁=c₃=10²`, `c₂=3`, RoBERTa-base, `r=4`, seq-len 16,
4 encoders → 16 tokens; ViT on CIFAR-10/100; metrics BLEU/ROUGE-L (text), **LPIPS** (vision).

---

## 2. Mapping MineGrad onto the gradient-bridge setting

Grounded in the actual bridge code (`experiments/gradient_bridge/`): the testbed is the Haim
**784-1000-1000-1 MLP**, and "LoRA" is an *analytically simulated single-step* rank-`r` projection
`A₁ = −lr·B₀ᵀ∇_W L` (fixed random `B₀`, `A=0` init) — **not** a trained `peft` adapter on a
transformer.

| Component | MineGrad | Gradient-bridge attack (as built) |
|---|---|---|
| Base model | Poisoned RoBERTa/BERT/ViT (foundation transformer) | Honest, frozen, max-margin **MLP 784-1000-1000-1** (toy) |
| Attacker control | **Poisons frozen weights + every-round global A,B** | **None** — observes an ordinary release; controls only the decoder it trains offline |
| Observed object | Victim's `∂L/∂{A,B}` for the poisoned modules, one round | Simulated single-step `∂L/∂A` (and `∂L/∂B` two-sided) = rank-`r` projection of `∇_W L` |
| LoRA information | Real transformer LoRA on Q/K/V/O per encoder | One analytic rank-`r` view per layer; `B₀` random, `A=0` init, T=1 |
| Source of extra constraints | Engineered orthogonality + identity forward pass + `L≈2(S−1)` coordinated modules | The **natural image manifold** (learned proxy prior in the decoder) + θ₀ structure in the inverter |
| Reconstruction method | **Closed-form** division + vocab/patch de-projection | **Learned** per-layer cosine-loss MLP decoder → model-based `run_ntk_extraction` inverter |
| Requires malicious modification? | **Yes — the whole attack is the modification** | **No** |
| Single step vs final adapter | Single round, single step | Single step (T=1) only; multi-step not simulated |
| Vision applicability | Yes — ViT patches, CIFAR, LPIPS 0.20 | Yes — but only pixel-space MLP, N=2, SSIM_norm≈0.6 (oracle upper bound) |
| Rank bottleneck | Sidestepped by `L≈2(S−1)` modules, each disjoint `≤r` | Binds hard: one rank-`r` view; leakage `≤ min(rank(M), r, N)` |

**Which mechanisms transfer, and which are malice-only:**

- **Malice-only (do NOT transfer):** (i) orthogonal position fingerprints (Trick 1) — the attacker
  rewrites the frozen embeddings; (ii) identity-mapping attention via `Q=10⁵·I` (Trick 2); (iii) the
  coordinate-selector `A` init (Trick 3); (iv) the class-token averaging + `W_CLS` logit steering
  (Trick 5); (v) the malicious high-variance word-embedding layer (App E.1). Every one of these is a
  *chosen* parameterization. Strip them and the analytical readout `∂L/∂B[:,n] ∝ y^{(token)}`
  collapses. **The honest bridge cannot use any of the analytical machinery.**
- **Transferable as *ideas* (not code):**
  - **The multi-module tiling principle** (the rank-breaker). It is architecture-level, not
    malice-level: *L independent low-rank views can carry `L·r` independent measurements even when
    each view is rank `r`.* In the honest setting the views are the LoRA modules an ordinary release
    actually exposes (Q,K,V,O × every layer). This is directly testable (Experiment A below) and is
    the single most important thing to take from MineGrad.
  - **Data lives on the input/`A` side; the `B`-gradient carries only `r`-dim projections of it.**
    MineGrad has to engineer *both* sides (`A` selector + `B=0` readout); that the data must be
    routed through `A` is a structural fact the honest attack inherits (Experiment B).
  - **The forward pass must preserve per-sample resolvability.** MineGrad *forces* this (identity
    attention). The honest question is whether a real pretrained forward pass keeps samples resolvable
    — precisely the feature ceiling `rank(M)` in the existing theory.
  - **Batch = averaging; de-mixing needs a codebook.** MineGrad de-mixes batch averages by snapping
    to the vocabulary. Vision has no codebook — which is *exactly* the superposition wall the bridge
    hits at N≈10. A generative prior is the vision replacement for the vocabulary (see §5).

---

## 3. The deepest conceptual connection

**The user's hypothesis, restated:** MineGrad builds a LoRA measurement operator that is injective on
the private-data-induced gradient set; the bridge asks whether an *ordinary* pretrained model already
induces an approximately injective LoRA observation on the manifold of naturally occurring gradients.

**Verdict: the frame is correct, but sharpen it in two ways.**

**(a) The right object is the composite observation map restricted to the data manifold, and its
Jacobian's singular values — not "recovering the full gradient."** Write
```
X  --F-->  g_full = ∇_W L(X)  --P_LoRA-->  y = B₀ᵀ g_full        (single-step LoRA view)
```
`P_LoRA` is globally many-to-one (`ℝ^{m×d} → ℝ^{r×d}`), so `g_full` is *not* recoverable in general —
correct. But recovery of `X` needs only that the **composite** `Y := P_LoRA∘F` be injective on the
image manifold `ℳ^N`, i.e. that the restricted Jacobian
```
DY(X)|_{T_X ℳ^N} = P_LoRA · DF(X)|_{T_X ℳ^N}
```
have **full column rank** (locally), with a **smallest singular value bounded away from 0** (global,
quantitative). The bridge is a *learned manifold-constrained left-inverse* of `Y`: among all
gradients consistent with the observation `y`, it returns the one lying on `F(ℳ^N)`. **It does not
manufacture rank.** It exploits that `F(ℳ^N)` is low-dimensional and *transverse* to `ker(P_LoRA)`,
so the constrained inverse is well-posed even though the unconstrained one is not. That is the
information-theoretically coherent statement, and it is standard **manifold-embedding / restricted-
isometry** territory: a generic rank-`ρ` projection embeds a `d_ℳ`-dimensional manifold iff
`ρ·(measurement width) ≳ d_ℳ` with conditioning set by the manifold's reach vs. `ker(P_LoRA)`.
MineGrad *guarantees* transversality + conditioning by construction; the honest bridge inherits
whatever the pretrained `F` and random `B₀` give.

**(b) Distinguish "recover `g_full`" from "recover `X`" — they are NOT the same, and the bridge has
been optimizing the wrong one.** For the **input layer** of the MLP, `∇_{W₀}L = g_err·xᵀ` (rank-1).
The single-step `A`-gradient is
```
∂L/∂A₀  =  B₀ᵀ ∇_{W₀}L  =  (B₀ᵀ g_err)·xᵀ   (N=1)   =  U Xᵀ   (general N),   U := B₀ᵀ G ∈ ℝ^{r×N}
```
where `G` is the (output-side) gate/error matrix of the existing theory (`Ω = G Xᵀ`). So:

- For **N=1**, `∂L/∂A₀` is rank-1 with **row space exactly `span(x)`** — `x` is the right singular
  vector, present *analytically, no decoder, no rank penalty.* The reported single-sample "input
  decode 0.637" is the cosine of the **full `m×d` gradient**, which additionally needs the
  **output-side factor `g_err`** — seen only through its `r`-dim projection `B₀ᵀg_err`. **`g_err` is
  exactly the part inversion does not need for a two-layer net.** The bridge has been scoring itself
  on recovering the inversion-irrelevant factor.
- For **N>1**, `∂L/∂A₀ = U Xᵀ` is the *same* "gates × data" factorization as `Ω = G Xᵀ`, with the
  gate matrix **projected by the LoRA map**: `U = B₀ᵀ G`, so `rank(U) ≤ min(rank(G), r) =
  min(rank(M), r)`. Identifiability of `{x_i}` needs `rank(U) = N`, i.e. `min(rank(M), r) ≥ N`. **This
  re-derives the existing `leakage ≤ min(rank(M), r, N)` ceiling cleanly, straight from the LoRA
  A-gradient — no separate "full gradient" object required.**

**Consequence:** the correct bridge objective is not "decode `∇_W L`" but "decode the
**inversion-relevant sufficient statistic**," which for a shallow net is the **row factor `X`**
itself. Reframing this (i) explains why the 0.997 hidden-layer decode was useless, (ii) predicts that
single-sided input-layer `x`-cosine may already be far above 0.637, and (iii) makes the target of the
whole pipeline `X`, collapsing the "LoRA→gradient→inversion" three-stage story into "LoRA→`X`"
whenever the inversion-relevant statistic is linear in the observation.

---

## 4. Redesigned experiments (5 highest-information, all runnable in the existing analytic sim)

The bridge's analytic simulator makes every MineGrad-inspired control a few lines: stacking `L`
independent `B₀` projections *is* the multi-module attack; swapping which factor is read *is* the A/B
study. No transformer or `peft` needed for the mechanism experiments.

**A. LoRA rank vs. number of independent LoRA views (the MineGrad rank-breaker).**
Fix a total measurement budget `R_tot = L·r`. Compare, at matched `R_tot`:
`(L=1, r=32)` vs `(L=8, r=4)` vs `(L=32, r=1)`, where each "view" is an independent random `B₀ₗ`
projection of the *same* `∇_W L`, and the decoder/inverter sees all `L`. **Hypothesis (from MineGrad
Fig 6):** many-thin beats few-thick, because independent projections tile the `N·(gate rank)`
signal while one thick projection wastes rank on the intra-view null space. **Metric:** control
margin and `x`-cosine vs `N`, at fixed `R_tot`. If many-thin wins, the honest analog of MineGrad's
"multi-encoder `r=2` = single-encoder `r=16`" holds, and the privacy takeaway "small `r` is safe"
is false for multi-module releases. *This is the single most important experiment.* Then repeat with
the views being the **real** modules a release exposes (Q,K,V,O × layers) rather than i.i.d. `B₀`, to
check whether real modules are as "independent" as random ones (they may be correlated → less than
`L·r` effective).

**B. `A` vs `B` gradients — measure `x`-cosine, not full-gradient cosine.**
Re-score the existing single-sided/two-sided runs with the **row-factor (`x`) cosine** and the
**column-factor (`g_err`) cosine** *separately*. Prediction from §3(b): single-sided input-layer
already recovers `x` far better than the 0.637 full-gradient number suggests; the "two-sided rescue
to 0.912" is largely recovering `g_err`, which inversion discards. Arms: only `∂A`, only `∂B`, both;
input layer vs hidden layer; `A=0` init (only `∂B` informative at T=1) vs `B=0` init (only `∂A`
informative) vs small-both. **This directly tests "does mediocre global gradient reconstruction still
give excellent image inversion?" (user Q4C) — and their own data likely already contains the answer.**

**C. Bridge-accuracy vs inversion-accuracy decoupling.**
Plot `‖ĝ_full − g_full‖` (and its per-layer, per-factor breakdown) against final reconstruction
`x`-quality across the activation × rank × N grid. **Identify which components of `g_full` are
inversion-relevant** by ablation: corrupt only the row factor / only the column factor / only the
hidden layer / only the input layer, and watch SSIM. (The existing "corrupt hidden→ssim 0.98,
corrupt input→ssim 0.52" result is the first data point — generalize it into a *sensitivity map*.)
Deliverable: a statement of the form "inversion depends on `g_full` only through `X = row-factor of
the input-layer gradient`," making the decoder target explicit and minimal.

**D. Destroy cross-layer / cross-module correspondence (the coordination control).**
For a fixed set of samples, shuffle the `L` LoRA-view observations *across independently drawn
samples* before feeding the decoder (preserve each view's marginal statistics, destroy the joint).
If performance collapses, the bridge is exploiting **coordinated, complementary measurements** (the
MineGrad mechanism); if it survives, it is mostly a **generic image prior** hallucinating from one
view (closer to Yao 2024). **This cleanly separates "information from coordination" from "prior
hallucination" — the user's central worry.**

**E. Information vs. prior — the two-dataset stress test.**
Build two proxy/private pairs whose LoRA observations are deliberately near-collinear (e.g. two
images with matched low-frequency content, differing only in high-frequency detail the LoRA view
under-measures). Ask whether the bridge tracks the *true* private sample or emits the prior's most
plausible completion. **Metric:** does `x̂` correlate with the true `x` beyond what the shared
low-frequency component explains? If not, the bridge is a prior, not an attack. (This is the honest
version of MineGrad's malicious high-variance-embedding trick, which *guarantees* separation — we are
measuring whether nature provides it for free.)

---

## 5. Revisiting the rank/identifiability theory

**Do not rescue `d·ρ ≳ Nk` as a standalone formula.** It is a *capacity* count and it already failed
its sharpest test (the flowers high-`k` prediction came out backwards: downsampled natural images are
*lower* effective-rank than MNIST). Replace the two separate ceilings with **one object**: the
restricted composite Jacobian and its singular spectrum.

**The unified statement.** Let `x_i = g(z_i)` be a generative parameterization (`z_i∈ℝ^k` the latent,
`g` the manifold chart — the SDS/diffusion decoder in practice). The end-to-end observation is
`Y(z_1,…,z_N) = P_LoRA(F(g(z_1),…,g(z_N)))`. Local identifiability ⇔
```
J := ∂ vec(Y) / ∂(z_1,…,z_N) ∈ ℝ^{(rd)×(Nk)}   has full column rank Nk,
```
and *quality* is governed by its **smallest singular value `σ_min(J)`** and **conditioning
`σ_max/σ_min`**, not by exact rank. This single Jacobian **subsumes both existing ceilings**:
- the **feature ceiling** `rank(M)` = whether `DF` separates samples (upstream factor of `J`);
- the **capacity ceiling** `ρ(m+d) ≳ Nk` = whether `P_LoRA` preserves the `Nk` manifold directions
  (the `rd ≥ Nk` shape condition for `J` to *possibly* be full-rank).
Both are necessary conditions for `σ_min(J) > 0`; `J` is the object that says whether they *jointly*
clear, and by how much.

**Why this is cleaner than an effective gate-rank `ρ_eff`.**
- It is **directly computable** at the anchor with one forward + a handful of JVPs — no 50k-epoch
  extraction — and it is a *number per configuration* (`σ_min`, condition number), so it can be
  **correlated against actual SSIM** across the whole activation × rank × module × N × dataset grid.
  That correlation is the falsifiable theory.
- It **explains each knob** through one mechanism: **activation** reshapes `DF` (the gate range → the
  feature block of `J`); **rank/module coverage** is exactly the row-dimension `rd` and the row-space
  of `P_LoRA` (Experiment A moves `rd` without moving `r`); **anchor `α`** slides pre-activations,
  moving `σ'` into/out of the sample band (re-conditioning `DF` — the observed GELU-rescue); **`N`**
  grows the column count `Nk` toward the row budget `rd` (the observed N-collapse is `σ_min(J)→0` as
  `Nk↑→rd`); **image complexity** is the local `k` = `dim T_{z}ℳ` (bigger `k` fills the budget
  faster — but must be measured *locally on the manifold*, not by `eff_rank(X)`, which is why the
  downsampled-flowers proxy failed).
- **Singular values, not exact rank**, because every real result is graded (SSIM 0.3–0.9), never
  binary. `σ_min(J)/σ_max(J)` is the natural predictor of "recognizable but blurry."

**The MineGrad lesson for the theory:** MineGrad is the explicit demonstration that `σ_min(J)` is
**not a property of `r` alone** — the malicious construction drives `σ_min(J)` to its maximum at
`r=2` by choosing `F` and `P_LoRA`. The honest question is precisely *how large `σ_min(J)` is for
natural `F` and random `B₀`*, and whether stacking real modules (raising `rd`) lifts it. That is a
measurement, and Experiment A + the Jacobian diagnostic make it one.

**A generative prior enters the theory honestly here.** SDS/diffusion is **not** a hard `k`-dim
manifold constraint (the user is right to reject that). In the Jacobian frame it does two concrete
things: (i) it supplies the chart `g` so the object of recovery is `z∈ℝ^k` not `x∈ℝ^d` (shrinking the
column dimension `Nk ≪ Nd`, relaxing the *shape* condition); (ii) as a soft prior it **re-conditions**
the inverse — raising the effective `σ_min` on the manifold-tangent directions while leaving the
off-manifold null space unconstrained. It is the vision replacement for MineGrad's vocabulary
codebook: the thing that de-mixes a superposed average by snapping to plausible completions. Its
*limit* is that it cannot separate two private samples that share the low-frequency content the LoRA
view actually measured (Experiment E) — it will hallucinate a plausible, not the true, high-frequency
detail. That is the precise boundary of the whole direction.

---

## 6. What to steal from MineGrad

**Reproduce (adopt directly):**
- **Vision evaluation protocol:** ViT + CIFAR-10/100, **LPIPS** as the headline metric (add to the
  bridge, which currently reports only SSIM/SSIM_norm — LPIPS is the field-standard and lets us
  compare numbers to MineGrad's 0.20 and to Yao 2024).
- **The `r=4`, seq-len/patch-count, 100-sample evaluation scale** as a baseline grid.
- **DAGER as the analytical baseline** to plot against (it is the honest-ish low-rank column-space
  test; shows exactly where "small `r`" defeats a *non-learned* attack — the gap the bridge must beat).

**Adapt (port the idea, not the code):**
- **Multi-module tiling → Experiment A.** Their strongest structural insight, reusable without malice.
- **The A/B asymmetry → Experiment B.** Their `A`-selector/`B`-readout split is the malicious version
  of "data lives on the input side."
- **Batch-averaging analysis (App D)** → the theory of *why* the bridge hits a superposition wall at
  N≈10, and why a codebook/prior is required to pass it.

**Ignore (malice-dependent, irrelevant to a passive attacker):**
- Poisoned position encodings, identity attention, coordinate-selector `A` init, class-token
  averaging, `W_CLS` logit steering, malicious high-variance embeddings, model-inconsistency
  secure-aggregation bypass. All are *chosen parameterizations*; none exist in an honest release.

---

## 7. The related-work chain (closest competitors first)

MineGrad's own citations sort into two families. **Every analytical LoRA/PEFT attack is malicious.**
The passive/honest lineage is the optimization/learned family — that is where the real competition is.

| Paper | Attacker knowledge | Control | Gradients or final adapter | Vision? | Batch? | Malicious init/model? | What the bridge adds beyond it |
|---|---|---|---|---|---|---|---|
| **Yao 2024**, *Risks When Sharing LoRA Fine-Tuned Diffusion Weights* (arXiv 2409.08482) — **CLOSEST** | Final LoRA weights only | **None (passive)** | **Final adapter** | **Yes (diffusion, faces)** | identity-level | **No** | Bridge: honest **discriminative** base (not a shared diffusion LoRA); grounded in gradient-inversion/KKT identifiability, not a black-box VAE; **no in-domain public data** (proxy only) |
| **SimuDy** (Tian et al., ICLR 2025) | Recipe + θ₀,θ_T | None (passive) | Final weights | Yes | small N | No | Bridge: LoRA-only observation + identifiability theory (SimuDy already owns direct-weight-inversion — see `simudy_decision_brief.md`) |
| **DAGER** (Petrov 2024, arXiv 2405.15586) | Gradients, honest | None | Per-step gradient | **No (needs vocab)** | up to `r` tokens | No | Bridge: vision, learned prior beats the `N≤min(D,r)` and vocab-search limits |
| **MineGrad** (this paper) | Full server control | **Poison model+LoRA** | One-round gradient | Yes | 52% @ batch 64 | **Yes** | Bridge: **honest** — MineGrad is the malicious upper bound, not a competitor for the passive claim |
| **Sami 2025 / PEFTLeak** (2506.04453) | Full server control | Poison adapter | Gradient | Yes | large | **Yes** | same as MineGrad |
| **Feng & Tramèr 2024**, *Privacy Backdoors* (ICML) | Corrupts pretrained model | **Poison θ₀** | FFT gradient | Yes | yes | **Yes** | Bridge: no poisoning; LoRA not FFT |
| **Fowl 2022/2023** (Robbing the Fed / Decepticons) | Malicious server | Poison model | Gradient | Yes/text | very large | **Yes** | Bridge: honest |
| DLG/IG/GradInversion (Zhu'19, Geiping'20, Yin'21) | Gradients, honest | None | Per-step gradient | Yes | small | No | Bridge: LoRA-only (far less signal), + identifiability characterization |

**Read immediately, in order:** (1) **Yao 2024, arXiv 2409.08482** — the passive learned-inversion-
from-LoRA-weights paper; it is the direct novelty threat and must be positioned against. (2) DAGER
(2405.15586) for the analytical low-rank baseline. (3) confirm SimuDy positioning (already done in
`simudy_decision_brief.md`).

---

## 8. Bottom line

1. **What MineGrad teaches (mechanism).** The per-matrix rank-`r` bound is real but *irrelevant* to
   total leakage: with `L` coordinated low-rank modules you get `≈ L·r` independent measurements, and
   a forward pass that keeps samples resolvable lets each module tile a disjoint slice of the data.
   Leakage scales with **number of independent, complementary LoRA views**, not with `r`. The data is
   read out of the **`B`-gradient** but only because the **`A`-side routes it there** — the
   input/`A` side is where samples physically live.
2. **What MineGrad does NOT establish for us.** Nothing about the honest setting. Every bit of its
   power is a *chosen* parameterization (orthogonal fingerprints, identity attention, selector-`A`,
   logit steering). Its vision result (LPIPS 0.20) and its batch trick (vocabulary de-mixing) both
   depend on the malicious construction / a discrete codebook. It is the **leakage upper bound under
   full server control**, i.e. exactly the "best-case knowledge" endpoint the thesis already frames
   direct-inversion as — *not* a passive-attack result.
3. **Strongest connection.** The bridge works iff the honest composite map `Y = P_LoRA∘F` is
   approximately injective (well-conditioned restricted Jacobian `DY|_{T_Xℳ}`) on the natural-gradient
   manifold. The bridge is a **learned manifold-constrained inverse**; it does **not** recover lost
   rank — it uses the prior that gradients come from real data to pick the one preimage on the
   manifold. MineGrad *guarantees* that conditioning; the bridge *measures* whether nature provides it.
4. **The 3–5 experiments:** A (rank vs. #views, matched `L·r` budget) · B (`x`-cosine vs
   `g_err`-cosine for A/B channels) · C (bridge-accuracy vs inversion-accuracy sensitivity map) ·
   D (shuffle cross-module correspondence: coordination vs prior) · E (two-collinear-datasets:
   information vs hallucination). A and D are the decisive ones.
5. **Theory change:** retire the standalone `d·ρ ≳ Nk` and `ρ_eff`; adopt the **restricted composite
   Jacobian `J = ∂vec(Y)/∂z` and its `σ_min` / conditioning** as the single predictor, computed
   cheaply at the anchor and correlated against SSIM across the grid. It subsumes both ceilings and
   explains all six knobs. **Reframe the decoder target from `∇_W L` to the inversion-relevant
   sufficient statistic (`X` itself for a shallow net).**
6. **Read now:** Yao 2024 (2409.08482) — closest passive competitor — then DAGER (2405.15586).
7. **Novelty statement (conditional on A/D succeeding):** *"We show that ordinary (non-malicious)
   LoRA fine-tuning of a pretrained model leaks individual training images to a passive attacker who
   never modifies the model or the adapter initialization, and we characterize exactly when: leakage
   is governed by the smallest singular value of the LoRA-restricted gradient Jacobian on the image
   manifold, which grows with the number of independent adapter modules (not the per-module rank) and
   collapses as the batch fills the measurement budget. This isolates the privacy cost of LoRA
   release itself — the honest lower bound complementary to MineGrad's malicious upper bound."*
8. **What MineGrad makes less novel / less plausible.** (a) "LoRA gradient inversion works for vision"
   is now published (MineGrad malicious; Yao passive-diffusion) — the bridge cannot claim vision-LoRA
   leakage *per se* as novel; it must claim the **honest, discriminative, ordinary-init** setting +
   the **identifiability characterization**. (b) MineGrad's `L·r` mechanism shows the interesting
   science is **multi-module coverage**, not single-adapter rank — a bridge framed around one adapter
   is attacking the wrong variable. (c) The reframing in §3(b) shows the current "decode the full
   gradient" objective (and its 0.997 hidden-layer headline) is **measuring the wrong quantity**; the
   honest numbers are the **free-coefficient, `x`-cosine, multi-module** ones, which are not yet run.
