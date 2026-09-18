# Plan — multilayer parameter program (Yoad, 2026-09-18)

Status: **DRAFT, not audited, no job submitted.** Convention: plan note → audit sibling → parallel builders (smoke,
then full bsub). Source of the questions: session of 2026-09-18, after `notes/open_questions_2026-09-18_yoad.md`
(what is run / not run). Every package below states what the theory already predicts, so a run is a test and not a
look. Observe, don't conclude: PASS on a check is not a proof; verdicts keep the three outcomes.

Common ground for every package: FP64; `B_0 = 0`; one release per cell trained on the same eight images as the
18 Sept package (`randperm(seed+7)[:8]`, the k-sweep join key); three seeds where a number is quoted (all real-net
multilayer rows to date are single-seed); rank = ladder + gap, never an integer without a gap; **new row fields**
`cond_at_1e10 = σ₁/σ_c` and `cond_at_corrected` emitted by every harness (today they are derivable but not stored);
`--layers` list flag on both rank-law harnesses (today: contiguous prefix only). Nothing is edited under a running
job (365681 must finish first).

Metrics per cell, always the same five: stacked rank at 1e-10 · `gap_at_corrected` · `cond_at_1e10` · usable rank at
fp16 · certificate residual at the truth. Attack cells add tier-1 landing rate and tier-2 line-up identification.

---

## P1. Rank r for the whole network

**Question.** How do rank, gap and conditioning move with the adapter rank when every adapted layer has the same r?
**Predicts.** `q_l = min(r − N', d_l)`, so `cap(L) = min(L·(r − N) + (m − 1), nesting ceiling)`; the ceiling itself
scales with r through `q_l`. The capacity line `k < m + r − N` is measured only at r = 16 (jobs 467914/469120).
**Cells.** `r ∈ {8, 16, 32, 64, 108, 256}` × `first ∈ {1, 3}` × `L ∈ {1, 2, 4, 8}` × `k ∈ {32, 128, 384, 784}` on
the d15 MLP **twin** (`mnist_mlp_d15w1000_full`, PASSES the gate) and on the bottleneck CNN. Harness:
`real_encoder_ranklaw.py --r` becomes a list (the CNN harness already takes one).
**Pre-registered.** (a) rank tracks `cap(L)` at every r within ±1 where a gap exists; (b) cond at the effective
rank rises with r at fixed L (more rows of C, deeper into J_φ's spectrum); (c) at deployed r ≤ 64 with N = 8 the
per-layer budget `r − N` is ≤ 56, so the chart width a stack can pin is `L·56` capped — state the L at which k = 384 is
first pinned, per r. Outcome that would change the picture: the gap surviving at deployed r where it did not at 108.

## P2. Other architectures

**Question.** Does the picture (conv pins everything at every position; dense stacks lose the gap with frozen depth)
hold beyond one MLP and one bottleneck CNN?
**Cells.** (i) plain conv net without the 8-channel bottleneck (`mnist_conv_deep_full`, PASSES); (ii) ResNet-18 on
CIFAR-10 with LoRA on the conv layers of one stage, BN in eval mode (affine, so the certificate applies unchanged);
(iii) a small ViT (patch 4, 6 blocks) trained on CIFAR-10 to the gate, LoRA on q/v projections — the input to the
adapted linear map is the token stream, so `N'` counts **tokens × images**, the conv-position analogue; the T5.2
per-layer budget under token sharing is `min(rank C_l · tokens, d_l)`, pre-register it as in the CNN header.
**Predicts.** Attention adapters behave like conv (many positions per image → one layer pins the chart) unless the
token count exceeds `r`, at which point the certificate is conv-vacuous (`N' ≥ min(r, p_l)`) — the same death the
CNN's conv 1 showed. State the token budget per r before running.
**Not in scope.** Diffusion, text.

## P3. Drift: measurement and a predictive rule

**Question (Gal's).** If the layer below the adapted one is not frozen, how much does its output move, is that
measurable, and is there a threshold on the drift below which the certificate works and above which it does not?
**What theory already says.** Two different objects, two different rules:
- Full certificate: exact at any drift *magnitude*; what it loses is rank, `rank C = min(r, n_l) − N'_l`, with
  `N'_l = N + rank(orthogonal drift)`. **Rule A (rank):** the certificate can pin a k-chart iff `min(r, n_l) − N'_l ≥ k`.
  Attacker-side proxy: `rank B_T ≤ N'_l` is observable; the rule is testable from the release when `rank B_T = N'_l`.
- Truncated certificate (rank `r − N` bought back): error `ρ ≈ K_l · δ_perp`, `K_l ≈ 0.08` measured on one synthetic
  net, slope 1.0004. **Rule B (magnitude):** landing survives iff `K_l · δ_perp < ε_land`, where `ε_land` is the
  measured landing floor (≈ 5× certificate residual, project memory). Only `δ_perp` counts; in-span drift is free.
**Cells.** Real net, adapter at layer `l ∈ {2, 4}` of the strong MNIST MLP (PASSES), with layer `l − 1` **also
adapted** (that is how the input drifts in practice) at `r_{l-1} ∈ {4, 16, 64}`, `T ∈ {1, 5, 20, 100, 400}`,
`lr ∈ {0.003, 0.01, 0.03}`. Record per cell `δ`, `δ_perp`, `drift_rank`, `N'_l`, `rank B_T`, both certificates'
residual at the truth, `K_l` fitted, then a **solve** (LM, 400 starts, k = 32, tier 1 + tier 2).
**Pre-registered.** Rule A: landing rate drops to the wrong-release floor exactly where `min(r, n_l) − N'_l < k`,
independent of `δ`. Rule B: for the truncated certificate, landing rate vs `K_l · δ_perp / ε_land` is a single
step curve across (T, lr, r_{l-1}); the threshold sits within a factor 3 of 1. A monotone curve in `δ` but not in
`δ_perp` refutes the T3 axis. Also record whether `K_l` is stable across nets (it is measured on one).

## P4. Layer subsets for the Jacobian

**Cells.** On the d15 twin and the CNN, `--layers` patterns at k = 384 and 784: prefix (today), suffix (last L),
middle block, alternating (one yes one no), random half (3 draws), single deep layer. Same five metrics.
**Predicts (corrected law, nesting).** `rank = min_j (d_j + Σ_{l<j} q_l)` over the **chosen** layers in network
order: the shallowest chosen layer's `d_j` caps everything, so skipping shallow layers costs the cap and adding deep
ones adds only what fits inside it. Alternating should match the prefix of the same shallowest layer up to `Σq`.
Conditioning: pre-register that a subset skipping the first adapted layer has strictly worse `cond_at_1e10` than
the prefix of equal L (deeper J_φ spectrum). Failure of either is a finding against the nesting reading.

## P5. Activation

**Cells.** Retrain the strong MNIST MLP and the bottleneck CNN to the gate with ReLU and tanh (GELU is the only
activation in every multilayer row). Same rank-law sweep at r = 64, first ∈ {1, 3}, L ≤ 4, k ∈ {128, 384, 784}.
**Predicts.** Certificate exactness is activation-independent (linear in the features). The **conditioning** of
`J = C · J_φ · V` is not: ReLU's piecewise-linear J_φ is expected to hold the gap deeper (no smooth decay), tanh
to lose it sooner (saturation kills rows). State this before running; the opposite ordering is informative.

## P6. Momentum and weight decay

**What theory already says (technical record A2).** Momentum with zero initial buffers is covered: every update
remains in the same subspaces, so the certificate is exact. Scalar weight decay is covered **only while the preserved
seed scale `(1 − ηλ)^T` is nonzero at release**; Counterexample 5.1 shows `C = 0` with `rank B_T = q` still true on an
open set — the boundary is sharp. Adam is out (already measured: `C ≡ 0`).
**Cells.** Head and layer-2 adapters on the strong MNIST MLP, `momentum ∈ {0, 0.9}`, `weight_decay ∈ {0, 1e-4, 1e-3,
1e-2}` (applied to A and B), `T ∈ {100, 400}`, r = 64, N = 8, plus the P3 drift cell at the best (T, lr).
**Pre-registered.** Momentum: residual at truth at the FP64 floor, rank/gap/cond unchanged within seed spread.
Weight decay: residual at the floor while `(1 − ηλ)^T ‖A_0‖ ≫ ‖A_T − (1−ηλ)^T A_0‖`, then a collapse of `rank C`
toward 0 as the seed scale dies; report the λT at which `cond_at_1e10` first exceeds 1e8 — that number is the
practical "how much weight decay survives" rule. Under drift (P3 cell) pre-register that momentum raises `N'`
faster per step (the buffer carries old-span directions) — this is a **conjecture**, mark it so.

## P7. Charts

**Cells.** Same release, same eight images, charts: PCA-k, per-class PCA, autoencoder, local PCA (bootstrap round-1),
SD-VAE latent (fidelity known, no inversion yet), at k ∈ {32, 128}. Record `J = C·J_φ·V` rank/gap/cond and the chart
Jacobian conditioning `V` on its own (the learned-chart correlate measured in the CIFAR layer study, 4.3 → 5.9).
**Predicts.** With a linear chart `cond(J)` is set by `C·J_φ` restricted to `col(V)`; a nonlinear chart multiplies by
`cond(J_ψ)`. The prediction to test: landing rate orders the charts by `cond_at_1e10`, not by fidelity (fidelity and
recoverability already shown to be different axes, cifar RESULT §7.2).

## P8. Moving between two charts

**Question.** In the iterative scheme (round-0 global chart → round-1 local chart), how do the rank and the
Jacobian change at the hand-over, and does the hand-over lose or gain identifiability?
**Cells.** Bootstrap releases (355987 MNIST, 355988 CIFAR): at each round compute, at the round's recovery and at
the truth's projection, `J` under chart t, under chart t+1, and under the **union** chart `[V_t V_{t+1}]`; record
the principal angles between `col(V_t)` and `col(V_{t+1})`, the rank/gap/cond of each, and the certificate
objective before and after the switch.
**Predicts.** `rank(J_{union}) ≤ rank(J_t) + rank(J_{t+1}) − overlap`; if the union chart exceeds `r − N` in width
the certificate becomes alias-prone (`k < r − q` violated) — the observed **slot collapse** (bootstrap RESULT) is
pre-registered as an alias event at the hand-over, testable by the objective at the truth's projection staying at
the floor while the recovery moves. If the union rank stays below `r − N` and the collapse persists, it is a basin
event instead. Either way the number to report is the width at which the switch first aliases.

---

## Order and cost

P4 and P1 are cheap (rank-law harness, CPU/one GPU, hours) and unblock the others: run first. P3 is the one Gal
asked for and needs the solve: second. P6 rides on P3's harness. P2, P5 need new checkpoints to the gate (WP0
runner exists for MLP/conv; ResNet/ViT trainers do not). P7 and P8 reuse saved releases.

Ground rules carried over: never quote a rank without its gap; never edit a harness under a running job; twins,
not the failing d15 original; three seeds before a number enters the ledger; the solve's verdict keeps
`recovered` / `optimisation failure` / `alias` apart.
