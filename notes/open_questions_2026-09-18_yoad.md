# Yoad's open questions, 2026-09-18 — what is run, what is not

Raised in session on 2026-09-18 after the 18 Sept package. Each line: the question, what the rows already say, and
whether it is RUN / RUN-NEGATIVE / NOT RUN. Sources are the RESULT(S).md of each track and the job rows; nothing here
is a new measurement except §3, which is derived from saved spectra.

## 1. Multilayer / architectures / parameters

| question | state | where |
|---|---|---|
| Multilayer certificate | RUN as algebra at the truth (rank, residual, gap); **NOT RUN as a solve** (M4 not started) | `experiments/multilayer_cert/RESULTS.md` §7 (iv) |
| Different architectures | Attacks: synthetic toy, MNIST MLP 3-layer + d15, CIFAR CNN + over-trained MLP. Theory-only: random MLPs, bottleneck CNN, ViT-B/16, DINO, ResNet, CLIP, SD-VAE chart. No transformer / diffusion attack | `notes/research_overview_2026-09-17.md`, `experiments/exact_inversion/BASE_TRAINING_GATE.md` |
| Base fully trained? | Gated (job 355833). PASS: mnist_mlp_strong, m26, cifar CNN, cifar over-trained MLP. FAIL: d15, both plain convs, weak CIFAR MLP. Twins exist, not substituted | `BASE_TRAINING_GATE.md` |
| LoRA on new class vs same class | New class everywhere; same-class new examples once (d15, records nothing, 355883–355896); mixed two-class WP2: certificate composition-blind, NTK route not | `experiments/cifar/RESULT.md` §7 |
| Different initial guesses | Certificate route: random starts only (200–2000). Replay route: near (oracle) / random / span / cert / spananchor, 1 of 20 seeds. Job 351007: a start needs cosine ≈ 0.5 with the private representation | `experiments/exact_inversion/RESULTS.md` Step 2b, STATUS 351007 |
| Which layers: beginning / middle / end / all | Only CONTIGUOUS prefixes from `first_adapted` (first ∈ {1,3,4,8,12} MLP, {1,3,5} CNN). Alternating, skip, mixed, sampled fraction: **NOT RUN** | `real_encoder_ranklaw.py:155` |

## 2. Rank and stability of the stacked J

- Rank: no spectral gap at any deployed adaptation depth on the d15 MLP; only raw-input adaptation has a clean rank
  (355835). CNN: conv certificates pin the chart from the first non-vacuous conv, dense+head arm has a real gap (355907).
- `condition_number` is null on every multilayer_cert row: reserved for a 6e join that never happened, and the authors
  treated a scalar as ill-posed without a rank. The spectrum IS saved (`spectrum_window_img0`, image 0 only).
- **Job 355778 (A100 clean-FP64 gap check) EXITED 2026-09-18 14:20 with no rows and no log.** STATUS.md and
  CLAIMS_LEDGER.md still say "pending / settles it". Rerun 365681 on a shared A40, k = 784 cells not yet written at
  the time of this note. Anything conditioned on 355778 is unverified.

## 3. Derived: effective condition number σ₁/σ_c of the stacked certificate Jacobian (c = rank at the 1e-10 rung)

Computed 2026-09-18 from `spectrum_window_img0` of jobs 365681 (d15 MLP, r = 108, k = 784), 355835 (same-net
control) and 355907 (bottleneck CNN, r = 256). Single image, single seed, N = 8. No new compute.

| net | first adapted | L = 1 | L = 2 | L = 4 | L = 8 |
|---|---|---|---|---|---|
| d15 MLP, k = 784 | 1 | 2.1 | 12 | 5.8e3 | 8.2e9 |
| d15 MLP, k = 784 | 3 | 1.3e2 | 3.1e3 | 1.0e10 | 7.9e9 |
| d15 MLP, k = 784 | 4 | 1.5e3 | 8.0e4 | 6.7e9 | – |
| d15 MLP, k = 784 | 8 | 2.6e15 | 1.6e15 | 1.4e15 | – |
| d15 MLP, k = 784 | 12 | 2.5e9 | 2.5e9 | 2.7e9 | – |
| CNN, k = 128 | 1 / 3 | 6–7 | 3–30 | 7–31 | – |
| CNN, k = 784 | 1 / 3 | 1.4e2 | 5e1–9e5 | 5e1–8e2 | – |
| CNN, dense+head only | 5 | 2.4e7 | 3.6e7 | – | – |

Reading: on the MLP each stacked layer costs one to two orders of conditioning, and a deep frozen path (first ≥ 8) puts
J at the FP64 floor already at L = 1; at fp16 the usable rank is cut near cond ≈ 2e3. On the CNN the stacked J stays
in the tens because conv certificates act at every position. Older run 355781 stored a windowed spectrum and does not
yield the scalar. To emit it on future rows: add `cond_at_1e10` / `cond_at_corrected` in the RANKLAW block of
`real_encoder_ranklaw.py` / `conv_encoder_ranklaw.py` **after 365681 finishes** (never edit under a running job).

## 4. Literature's best per-example optimiser?

No. The per-image solve is Levenberg–Marquardt (jacfwd Jacobian, FP64, 12 backtracks, λ/3 on accept, ×4 on reject),
random multi-start, no prior. Haim KKT: tried and dropped (Experiment A). Geiping cos+TV+signAdam: Phase 0 ViT only.
GradInversion, Cocktail-Party ICA, SPEAR, ARES, DAGER, R2F decoder, Yao 2024, SDS/diffusion prior: never run.
`get_diversity_penalty` IS wired in `ntk_extraction.py` behind `diversity_weight` (default 0); the CLAUDE.md line
"not wired in" is stale. Failures are attributed to the basin, not information (P3; Q10 d15 letters 0/400 at ε = 0).

## Candidate experiments (not planned, not costed)

1. M4 proper: certificate-only solve with 1/2/4/8 stacked layers on a real chart, tier-1 and tier-2 metrics.
2. Non-contiguous layer sets: alternating, middle-only, end-only, random fraction; needs a `--layers` list flag.
3. Emit the condition number per row; sweep it against landing rate to test "conditioning, not rank, is the wall".
4. One literature prior in the LM solve (TV, or a learned chart as initialiser) against the same 400 starts.
5. Base-gate twins substituted for d15 (Q2, Q10, M6, k-sweep re-run on `mnist_mlp_d15w1000_full`).
