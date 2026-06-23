# Tier-1 Reading Notes (Thursday AM — 15 min budget)

## ⏰ How to use this in 15 minutes

**Do NOT open the PDFs.** This digest IS the reading.

| Minutes | Read | Why |
|---|---|---|
| **0–10** | Paper 1 (CORRECTED) — SVS 2025: Headline → Threat model → **Homogeneity assumption (verbatim box)** → Main theorems (skim) → **"Connection to Yoad's work — for Gal"** (memorize the framing). | This is the paper he's most likely to ask about (it's his). |
| **10–15** | Paper 2 — Gronich-Vardi: Headline → Optimizer-norm table → "Why this matters / caveats". Skip the assumptions block. | Context only. He may name-drop it; ℓ∞ direction is OFF per his prior guidance. |

**One sentence each, after 15 min:**
- SVS 2025: *"Provable privacy attacks (MIA in high $d$, reconstruction in $d{=}1$) under homogeneity-of-$\theta$ via Lyu-Li KKT. PEFT breaks homogeneity because $W_0$ is fixed — $\Delta W$ in the NTK regime restores something like local homogeneity in $\Delta W$."*
- Gronich-Vardi: *"Adam/Muon/Signum direction-converge to KKT of the $\|\cdot\|$-margin problem under the optimizer's aligned norm. Smooth homogeneous nets only — ReLU fails M1."*

**If pushed beyond your reading:** *"I've skimmed the digest, not the full proofs. Best-guess connection is X — tell me where I'm wrong."* That's the move.

---



## Paper 1: Smorodinsky-Vardi-Safran 2024 — Provable Privacy Attacks

### Status: FILE MISMATCH — cannot summarize

The file `/home/projects/galvardi/yoado/papers/Smorodinsky_Vardi_Safran_2024_Provable_Privacy_Attacks.pdf` is **misnamed**. Its actual contents:

> *"How Many Images Does It Take? Estimating Imitation Thresholds in Text-to-Image Models"* — Verma, Rassin, Das, Bhatt, Seshadri, Shah, Bilmes, Hajishirzi, Elazar (TMLR 12/2025, arXiv:2410.15002v2)

An unrelated TMLR paper on text-to-image imitation thresholds (200–700 images). **Not** the Smorodinsky-Vardi-Safran provable-privacy paper. None of the other 23 PDFs in `papers/` match either.

**Action before Thursday:** re-download the actual paper (likely arXiv version of *"Provable Privacy Attacks on Trained Shallow Neural Networks"* — Smorodinsky/Vardi/Safran, NeurIPS or COLT 2024). I did **not** fabricate theorem statements per instructions. Re-run this skill once the correct PDF is in place.

---

## Paper 1 (CORRECTED): Smorodinsky-Vardi-Safran 2024/2025 — Provable Privacy Attacks on Trained Shallow Neural Networks

arXiv:2410.07632v2 (Feb 2025). PDF: `papers/Smorodinsky_Vardi_Safran_2025_Provable_Privacy_Attacks_2410.07632.pdf`.

### Headline
First **provably correct** privacy attacks from the implicit bias of 2-layer ReLU nets: (i) in 1-D, $\ge 1/4$ of an algorithmically-built finite candidate set are training points; (ii) in high-$d$, **membership inference** succeeds w.p. $1-o_d(1)$ via $|\Phi(\theta;x)|$ vs. margin $m$.

### Threat model
White-box on $\theta$ (no labels, no $\lambda_i$, possibly no $m$). All attacks rest on **Asm 2.1**: $\theta$ satisfies KKT of $\min_\theta\tfrac12\|\theta\|^2$ s.t. $y_i\Phi(\theta;x_i)\ge 1$ — i.e. $\theta=\sum_i\lambda_i y_i\nabla_\theta\Phi(\theta;x_i)$, $\lambda_i\ge 0$, complementary slackness. **Reconstruction** (Sec. 3, $d=1$): finite set $S\subset\mathbb{R}$ guaranteed to contain training points. **MIA** (Sec. 4, large $d$): decide if $x\in$ train; works **black-box** (only $\Phi(\theta;\cdot)$ queries — Remark 4.1).

### Homogeneity assumption (verbatim — critical for PEFT angle)
> "A network $\Phi(\theta;x)$ is called **homogeneous** if there exists $c>0$ such that for every $b>0$, $\theta$ and $x$, $\Phi(b\theta;x)=b^c\Phi(\theta;x)$."

Chain: Lyu-Li/Ji-Telgarsky (Thm 2.1) needs homogeneity in $\theta$ $\Rightarrow$ GF $\to$ KKT direction $\Rightarrow$ Asm 2.1 $\Rightarrow$ all attacks. The paper's 2-layer ReLU has hidden biases but **no second-layer bias** $\Rightarrow$ $\Phi(b\theta;x)=b^2\Phi(\theta;x)$. Any deviation that kills positive homogeneity in $\theta$ breaks the chain.

### Main theorems
**Univariate ($d=1$, Sec. 3):**
- **Thm 3.2**: For two adjacent breakpoint intervals where $\Phi$ is *not* constant on the margin, the combined interval contains a training point on the margin; $\le 4$ margin-points per such interval.
- **Thm 3.3**: For three adjacent intervals alternating on/off margin (one always-active neuron, $\theta$ a local optimum), $\ge 1$ interior breakpoint $-b_i/w_i$ is a training point.
- **Thm 3.4** (main): Algorithm 1 yields finite $S$ with $\ge 1/4$ training points. Needs (a) one neuron active on all data, (b) $\theta$ a local optimum.

**High-dimensional (Sec. 4):**
- **Asm 4.1** (near-orthogonality): $\Pr[n|x_i^\top x_j|\le o(d)]\ge 1-\tau/n^2$, $\Pr[\|x\|^2\ge\Omega(d)]\ge 1-\tau/n$. Holds for uniform on $\sqrt d\,S^{d-1}$, $\mathcal{N}(\mu,I)$ with $\|\mu\|^2=o(d)$, bounded-mean GMMs.
- **Thm 4.2** (MIA tool): w.p. $\ge 1-2\tau$, $x\in$ train $\Rightarrow |\Phi(\theta;x)|=m$; w.p. $\ge 1-4\tau$, $x\sim D \Rightarrow |\Phi(\theta;x)|=O(nm\delta/\Delta)=o_d(m)$, with $\delta=\max_{i\ne j}|x_i^\top x_j|$, $\Delta=\min_i\|x_i\|^2$.
- **Cors. 4.3–4.5**: MIA when (4.3) $m$ known; (4.4) one leaked point fixes $m$; (4.5) $m$ merely bounded.

### Univariate vs. high-dim
Reconstruction (Thms 3.2–3.4) lives **only in $d=1$**. MIA (Thm 4.2) needs large $d$. Sec. 5 experiments show MIA holds empirically at moderate $d$ where Asm 4.1 is violated; high-$d$ reconstruction is open.

### Connection to Yoad's work — for Gal
"Your Thm 4.2 gives a clean MIA guarantee under Asm 2.1, which rests on homogeneity in $\theta$ via Lyu-Li. LoRA breaks this: only $A,B$ train while $W_0$ is frozen, so $\Phi(W_0+BA;x)$ is *not* homogeneous in $(A,B)$ when $W_0\ne 0$. In the NTK-linearized regime $\Delta W=BA$ acts on $\phi(x)=\nabla_W\Phi(W_0;x)$, restoring a kind of homogeneity in $\Delta W$. I'd love your read on whether there's a precise 'restricted homogeneity' or local-around-$W_0$ KKT condition under which an SVS-style theorem still holds for PEFT — and whether a Thm 4.2-style $|\Phi-\Phi_{W_0}|\approx m$ test is the right empirical handle."

---

## Paper 2: Gronich-Vardi 2026 — Implicit Bias of Adam and Muon on Smooth Homogeneous Networks

### Headline
Extends the Lyu-Li / Tsilivis et al. *gradient-flow KKT* implicit-bias story from GD to **momentum-based optimizers (Adam, Muon, Signum, Muon-Signum, Muon-Adam)** on **smooth homogeneous** networks, via an "approximate steepest descent" framework: each optimizer direction-converges to a KKT point of the max-margin problem under *the norm it is geometrically aligned with*.

### Main result (Theorem 3.3, near-verbatim)
> **Theorem 3.3.** *Let $\theta_t$ be a trajectory of normalized or unnormalized momentum steepest descent w.r.t. norm $\|\cdot\|$. Under Assumptions (M1), (M2), (LR-MSD), (T1), (T2), the limit point $\bar\theta$ of $\theta_t/\|\theta_t\|$ is the direction of a KKT point of Problem (11) with the norm $\|\cdot\|$.*

Problem (11) is the $\|\cdot\|$-margin program $\min_{\theta} \tfrac12\|\theta\|^2$ s.t. $y_i f(x_i;\theta)\ge 1\ \forall i$.

**Assumptions:**
- **(M1)** $f$ smooth in $\theta$ ($C^1$). Pure ReLU fails this; squared-ReLU, quadratic, GELU satisfy it. Section 4 weakens to M1-Weak (locally Lipschitz + Whitney $C^1$-stratifiable) under extra condition T3.
- **(M2)** $L$-homogeneous: $f(x;\alpha\theta)=\alpha^L f(x;\theta)$.
- **(LR-MSD)** $\int_0^\infty \eta(t)\,dt=\infty$ and $\eta(t)\le o(t^{1/L-1})$ — decaying schedule.
- **(T1)** $\|\theta_t\|\ge N_{\min}$ eventually (mild).
- **(T2)** Directional convergence: $\theta_t/\|\theta_t\| \to \bar\theta$ with $\gamma(\bar\theta)>0$. *Stronger than Lyu-Li — needed because momentum is history-preserving.*

Adam (Theorem 3.6) adds one technical init assumption (A1) to keep $v_t[j]>0$ without the stability constant.

### Approximate steepest descent framework
Def 5.1: an arc is an **approximate steepest descent** trajectory w.r.t. $\|\cdot\|$ if its update direction is (up to lower-order error) a unit dual-norm element aligned with the negative gradient. Adam/Muon/Signum/MomentumGD all fit this *asymptotically* once $\eta(t)$ decays. A single master theorem (C.17) then yields KKT convergence for the whole family.

### Optimizer-norm table

| Optimizer | Margin norm in KKT result | Thm |
|-----------|--------------------------|-----|
| GD / NGD / MomentumGD | $\ell_2$ | 3.3 |
| Signum | $\ell_\infty$ | 3.3 / Cor 3.5 |
| **Adam** (no $\varepsilon$, $c_1\ge c_2$) | **$\ell_\infty$** | **3.6** |
| Muon (per-matrix) | $\|\cdot\|_{\mathrm{msp}}=\max_k\|W_k\|_{\mathrm{sp}}$ | Cor 3.4 |
| Muon-Signum | $\max\{\|(W_{1..K})\|_{\mathrm{msp}},\|u\|_\infty\}$ | Cor 3.5 |
| Muon-Adam | $\max\{(\eta_0^A/\eta_0^M)\|(W_{1..K})\|_{\mathrm{msp}},\|u\|_\infty\}$ | 3.7 |

### Why this matters for our work
Most real fine-tuning uses **Adam**, not SGD — Theorem 3.6 says the implicit-bias / KKT story *still works*, but the residual is for the $\ell_\infty$-margin problem, not $\ell_2$. Useful context for any extraction objective derived from KKT stationarity under Adam.

Caveats: (a) still requires homogeneity in $\theta$, which LoRA breaks (only $A,B$ trained, $W_0$ fixed → $f$ not homogeneous in $(A,B)$ when $W_0\ne 0$); (b) smooth-activation requirement is satisfied by ViT/GELU but only weakly by ReLU.

**Per prior Vardi guidance: do NOT pitch follow-up on the $\ell_\infty$ direction.** Read for context only — it is the visible novelty but Vardi has signalled it is not the angle he wants pursued.
