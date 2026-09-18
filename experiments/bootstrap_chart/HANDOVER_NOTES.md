# Chart program — P8 (chart hand-over) and P7 (charts on one release), 2026-09-18

Code: `experiments/bootstrap_chart/handover_jacobian.py` (P8), `experiments/cifar/chart_conditioning.py` (P7),
runner `scripts/run_chart_program_wexac.sh {p8 <mnist|cifar> [b-arms]|p7 [charts]}`. Analysis on SAVED artifacts only
(bootstrap tensors 355987 / 355988, the CIFAR CNN motorcycle head release); no new attack runs, no new training except the
P7 autoencoder chart (WP2 did not save its instance). FP64. Every number below is a first read by the builder and is
provisional until a second session reads the rows.

Rows: `results/bootstrap_chart/handover_366170.jsonl` (MNIST), `handover_366171.jsonl` (CIFAR);
`results/cifar/chart_conditioning_366191.jsonl` (pca + pca_perclass marker), `_366192.jsonl` (ae), `_366193.jsonl` (local).
Logs: `scripts/wexac_logs/chartprog_<jobid>.{out,err}`.

Pre-registration (in the docstrings): reference point is `x*_chart` (oracle-start LM optimum, 10a), never the truth's
projection; the slot collapse is read as coverage with the alias flag as the falsifier, counted (10b); the round-1 chart
is built after seeing C, so the union-chart rank / cond is a measurement, not a theorem test, and a union wider than
r − N' = 56 is alias-prone by construction, reported but not tested for identifiability (10c). P7: k ∈ {16, 32, 48}
only (item 9); the landing rate is predicted to order charts by `cond_at_rank`, not by projection error.

Definitions. `J = C · J_φ(x) · V` with C = P_{row(B_T)^⊥} A_T (64 × n, rank 56), J_φ the Jacobian of the penultimate
features at x, V the chart basis (orthonormal for PCA / local; `J_ψ(w*)` for the autoencoder, plus its orthonormalised
copy). Rank at the 1e-10 rung with the absolute floor σ₁(A_T J_φ V) (`multilayer_cert.common.numrank`); gap =
σ_rank / σ_rank+1 (∞ when the rank equals the smaller dimension: nothing below it); cond = σ₁ / σ_rank. Union chart =
orthonormal basis of [V_t V_t+1] at the 1e-10 rung. fwd_check = rebuilt release vs the saved A_T, B_T, C.

## P8 — CIFAR (job 366171, 92 s wall on one GPU; fwd_check exact: max |Δ| = 0.0 on A_T, B_T, C)

Source: 355988 rows round 0, A1 (recognised `clock`, wrong-class `lamp`, oracle `motorcycle`), A2 (recognised `skunk`,
wrong-class `lamp` again). **Variant B had not started when the job ran** (355988 still running): no B pairs for CIFAR.

| pair (t → t+1) | union width | principal angles min / median (deg), #<1° | rank at x*(t+1): V_t / V_t+1 / U | gap at 56 (U) | cond at x*(t+1): V_t / V_t+1 / U | obj x*(t) in chart t+1 / x*(t+1) in chart t | alias / reached / short (of 8) |
|---|---|---|---|---|---|---|---|
| round0 → A1 recognised (clock) | 64 (> 56) | 1.7 / 19.1, 0 | 32 / 32 / 56 | 1.6e13 | 34 / 33 / 228 | 9.1e-02 / 4.0e-01 | 7 / 0 / 1 |
| A1 → A2 recognised (skunk) | 64 (> 56) | 3.6 / 28.3, 0 | 32 / 32 / 56 | 1.3e13 | 33 / 30 / 199 | 3.0e-01 / 2.1e-01 | 7 / 0 / 1 |
| round0 → A1 wrong-class (lamp) | 64 (> 56) | 1.6 / 17.7, 0 | 32 / 32 / 56 | 1.0e13 | 36 / 31 / 221 | 1.3e-01 / 4.0e-01 | 7 / 0 / 1 |
| A1 → A2 wrong-class (lamp = lamp) | 32 | 0.0 / 0.0, 32 | 32 / 32 / 32 | — | 31 / 31 / 31 | 1.1e-02 / 1.1e-02 | 7 / 0 / 1 |
| round0 → A1 oracle (motorcycle) | 64 (> 56) | 2.7 / 22.9, 0 | 32 / 32 / 56 | 1.2e13 | 36 / 37 / 197 | 7.8e-02 / 4.2e-01 | 8 / 0 / 0 |

Round 0 in its own chart: alias flags 8/8 (the saved row also says 8). Reading: every single chart is full column rank
(32) with cond ≈ 30–37 at both optima; every union of two different PCA-32 charts has width 64 > 56 and J_union has
rank exactly 56 = rank C with a 1e13 gap (σ₅₇/σ₁ ≈ 2e-16): an 8-dimensional exact null direction set in the union
coordinates, alias-prone by construction as pre-registered; cond of the union rises to ≈ 200 (×6). The two charts of a
pair share no direction (smallest principal angle 1.6–3.6°, none below 1°). x*(t) carried into chart t+1 loses its
objective by 1–2 orders (9e-2 → the new chart's own optimum 1e-2); err(x*_t, x*_t+1) ≈ 0.8 — the two optima are
different images. Alias flags 7–8 of 8 on every CIFAR global round (rows agree): in CIFAR the coverage reading does NOT
hold for the global charts — the recoveries are equally-scoring different chart points (information, not solver).

## P8 — MNIST (job 366170, 628 s wall on 4 CPU cores; fwd_check 1e-16 on A_T/B_T, 1.7e-12 on C: pass)

Source: 355987 rows round 0, A1 (recognised `a`, wrong-class `n`, oracle = recognised), B1–B4 × {recovery, random-anchor,
oracle-anchor}. The variant-B local charts were rebuilt from the saved anchors (nearest-truth index 8/8 identical to the
saved rows on every slot-round, chart error |saved − rebuilt| ≤ 6e-16, x*_chart objective relative drift ≤ 4e-10 vs the
saved `objective_opt`; per-slot alias / reached counts identical to the saved rows on all 12 slot-rounds).

| pair (t → t+1) | union width | angles min / median (deg), #<1° | rank at x*(t+1): V_t / V_t+1 / U | gap at 56 (U, med) | cond at x*(t+1): V_t / V_t+1 / U | obj x*(t) in chart t+1 / x*(t+1) in chart t | alias / reached / short (of 8) |
|---|---|---|---|---|---|---|---|
| round0 → A1 recognised (a) | 64 (> 56) | 1.6 / 13.4, 0 | 32 / 32 / 56 | 1.1e13 | 60 / 55 / 365 | 2.0e-02 / 9.5e-03 | 5 / 2 / 1 |
| round0 → A1 wrong-class (n) | 64 (> 56) | 2.0 / 13.3, 0 | 32 / 32 / 56 | 9.9e12 | 52 / 56 / 440 | 2.1e-02 / 2.4e-02 | 7 / 1 / 0 |
| round0 → A1 oracle (= recognised) | identical to the recognised row | | | | | | |
| round0 → B1 recovery | 64 (> 56) | 7.6 / 34.7, 0 | 32 / 32 / 56 | 6.0e12 | 60 / 60 / 556 | 7.7e-03 / 6.6e-03 | 0 / 8 / 0 |
| B1 → B2 recovery | 64 (> 56) | 3.0 / 17.6, 0 | 32 / 32 / 56 | 9.3e12 | 60 / 63 / 408 | 5.6e-03 / 2.7e-03 | 1 / 7 / 0 |
| B2 → B3 recovery | 64 (> 56) | 1.1 / 7.7, 0–4 | 32 / 32 / 56 | 7.6e12 | 58 / 58 / 512 | 2.5e-03 / 3.2e-03 | 2 / 6 / 0 |
| B3 → B4 recovery | 64 (> 56) | 1.1 / 8.1, 0–2 | 32 / 32 / 56 | 7.8e12 | 62 / 60 / 489 | 2.7e-03 / 2.6e-03 | 2 / 6 / 0 |
| round0 → B1 random-anchor | 64 (> 56) | 6.2 / 33.9, 0 | 32 / 32 / 56 | 8.9e12 | 54 / 76 / 456 | 1.6e-02 / 2.1e-02 | 4 / 4 / 0 |
| B1 → B2 random-anchor | 64 (> 56) | 10.3 / 38.3, 0 | 32 / 32 / 56 | 1.4e13 | 56 / 55 / 331 | 3.4e-02 / 2.9e-02 | 0 / 6 / 2 |
| B2 → B3 random-anchor | 64 (> 56) | 10.5 / 36.4, 0 | 32 / 32 / 56 | 9.2e12 | 61 / 56 / 323 | 2.1e-02 / 1.9e-02 | 3 / 3 / 2 |
| B3 → B4 random-anchor | 64 (> 56) | 10.6 / 39.4, 0 | 32 / 32 / 56 | 7.1e12 | 61 / 63 / 433 | 1.8e-02 / 3.2e-02 | 1 / 5 / 2 |
| round0 → B1 oracle-anchor | 64 (> 56) | 7.6 / 36.7, 0 | 32 / 32 / 56 | 8.3e12 | 58 / 62 / 536 | 8.9e-03 / 6.0e-03 | 0 / 8 / 0 |
| B1 → B2 oracle-anchor | 32 (7 slots) / 64 (1) | 0.0 / 0.0 | 32 / 32 / 32–56 | — | 62 / 62 / 62 | 1.1e-04 / 1.1e-04 | 0 / 8 / 0 |
| B2 → B3, B3 → B4 oracle-anchor | 32 | 0.0 / 0.0, 32 | 32 / 32 / 32 | — | 62 / 62 / 62 | 5.8e-05 / 5.8e-05 | 0 / 8 / 0 |

Round 0 in its own chart: alias flags 5/8 (row says 5). Reading. (i) Same structure as CIFAR: every chart full column
rank 32, cond ≈ 52–76 (the MNIST MLP has cond ≈ 2× the CIFAR CNN), union width 64 > 56 with rank(J_union) = 56 = rank C
exactly (gap ≈ 1e13) and cond ≈ 330–560 (×6–9). (ii) The recovery-anchored local charts CONVERGE: the smallest
principal angle between consecutive slot charts falls 7.6° → 3.0° → 1.1° → 1.1°, with up to 4 shared directions (< 1°)
by round 3; the oracle-anchored chain is exactly stationary from round 2 (angles 0, width 32, x* unchanged) — the
"warm start is the best start" observation of the bootstrap RESULT, seen in the chart itself. (iii) Objective hand-over:
x*(t) carried into the next recovery-anchored chart keeps an objective within ×1–3 of the new optimum (5.6e-3 vs 2.7e-3),
against a ×2–10 loss for the class charts; on the recovery chain the two optima are 0.27–0.45 apart in image space and
x*(t+1) is 0.27–0.29 from its truth (the per-slot fidelity gain of the RESULT). (iv) Alias falsifier for the coverage
reading (10b): recovery-anchored slot-rounds 0 + 1 + 2 + 2 = **5 alias flags in 32 slot-rounds** (0 in round 1, the
rounds where the collapse is first measured), oracle-anchored 0/32, random-anchor 8/32, global charts 5–7/8. The coverage
reading survives its falsifier in round 1 and is weakened, not falsified, in rounds 2–4 (slots 1, 4, 6, 7 — the slots
whose x* drifted). Union-chart numbers are measurements under 10c; identifiability was not tested there.

## P7 — CIFAR CNN motorcycle head release (jobs 366191 pca / 366192 ae / 366193 local; 982 / 1167 / 980 s on one GPU each)

Release rebuilt on the RAW privates, fwd_check vs 355988 exact (0.0), rank B_T 8, rank C 56, objective at the raw
truths ≤ 1e-27. `pca_perclass` = `pca` on this single-class cell (marker row, not recomputed; WP2 skips it too).
SD-VAE: no saved latent chart at k ≤ 48 (results/decoder_chart holds fidelity rows only, `release_read: False`): skipped.
WP2 join (cifar / cnn / motorcycle / T = 400): only pca-k16 (105/200 landed, 8/8 images; 355939) and ae-k16 (27/200,
3/8; 355940) exist — jobs 355939/355940 stopped inside the k = 32 cell, so k = 32 / 48 have no landing rate on this
release. `raw` rows = x*_chart from the oracle-start LM on the raw release (300 iters, the bootstrap's reference);
`onchart` rows = the WP2 construction (privates = chart projections, release retrained, x* = truth; objective ≤ 4e-27;
the `ae` instance is retrained, so its onchart release is WP2's recipe but not WP2's decoder).

| chart | k | release | rank | cond@rank med / max | cond(J_orth) | cond(J_ψ) | proj err med | err(x*, truth) | obj x* med | WP2 landing | images |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pca | 16 | raw | 16 | 9.6 / 14.5 | 9.6 | 1 | 0.343 | 0.998 | 1.8e-02 | 0.525 | 8/8 |
| pca | 16 | onchart | 16 | 12.1 / 15.3 | 12.1 | 1 | 0.343 | 0 | 1e-28 | 0.525 | 8/8 |
| pca | 32 | raw | 32 | 36.7 / 45.5 | 36.7 | 1 | 0.299 | 0.869 | 5.7e-03 | — | — |
| pca | 32 | onchart | 32 | 35.4 / 44.5 | 35.4 | 1 | 0.299 | 0 | 7e-29 | — | — |
| pca | 48 | raw | 48 | 153 / 189 | 153 | 1 | 0.274 | 0.737 | 1.6e-03 | — | — |
| pca | 48 | onchart | 48 | 105 / 185 | 105 | 1 | 0.274 | 0 | 6e-29 | — | — |
| ae | 16 | raw | 16 | 26.3 / 33.8 | 10.1 | 12.9 | 0.388 | 0.775 | 1.8e-02 | 0.135 | 3/8 |
| ae | 16 | onchart | 16 | 16.8 / 21.2 | 11.4 | 15.3 | 0.388 | 0 | 4e-29 | 0.135 (approx.) | 3/8 |
| ae | 32 | raw | 32 | 72.3 / 114 | 35.5 | 21.4 | 0.398 | 0.856 | 7.0e-03 | — | — |
| ae | 32 | onchart | 32 | 51.6 / 68.9 | 30.2 | 22.0 | 0.398 | 0 | 3e-27 | — | — |
| ae | 48 | raw | 48 | 358 / 583 | 131 | 34.4 | 0.395 | 0.861 | 4.5e-03 | — | — |
| ae | 48 | onchart | 48 | 234 / 349 | 115 | 36.2 | 0.395 | 0 | 3e-28 | — | — |
| local (K=200, truth-anchored) | 16 | raw | 16 | 10.2 / 11.8 | 10.2 | 1 | 0.343 | 0.906 | 1.6e-02 | — | — |
| local | 16 | onchart | 16 | 9.1 / 12.8 | 9.1 | 1 | 0.343 | 0 | 1e-27 | — | — |
| local | 32 | raw | 32 | 29.2 / 41.1 | 29.2 | 1 | 0.309 | 0.805 | 6.0e-03 | — | — |
| local | 32 | onchart | 32 | 33.5 / 39.3 | 33.5 | 1 | 0.309 | 0 | 4e-27 | — | — |
| local | 48 | raw | 48 | 120 / 166 | 120 | 1 | 0.296 | 0.768 | 2.5e-03 | — | — |
| local | 48 | onchart | 48 | 147 / 297 | 147 | 1 | 0.296 | 0 | 1e-29 | — | — |

Gap is ∞ in every cell (rank = k = the smaller dimension; k < 56 so no null direction). Orderings (pre-registered):
* by cond (best first), raw: pca-16 (9.6), local-16 (10.2), ae-16 (26), local-32 (29), pca-32 (37), ae-32 (72), local-48
  (120), pca-48 (153), ae-48 (358) — cond grows ≈ ×4 per +16 in k within every family, and the AE adds its own
  cond(J_ψ) = 13 → 34 on top (its orthonormalised J_orth sits at the PCA value).
* by projection error (best first), raw: pca-48 (0.274), local-48 (0.296), pca-32 (0.299), local-32 (0.309), pca-16 =
  local-16 (0.343), ae-16 (0.388), ae-48 (0.395), ae-32 (0.398) — the REVERSE k-order of the cond list within a family.
* by WP2 landing rate: pca-16 (0.525) > ae-16 (0.135). On the two joinable cells cond and projection error BOTH order the
  same way (pca-16 better on both), so the head-to-head is a TIE between the two predictors on this release: the
  pre-registered separation (cond vs fidelity) is not decided here. Across k the two predictors point in opposite
  directions; the k = 32 / 48 landing cells on THIS release are what would decide it and they do not exist. Adjacent
  evidence only (different class / base, same recipe): the bottle-CNN T = 400 rows land 183/200 → 87/200 (pca k16 → 32)
  and 31 → 5 (ae), i.e. the landing rate falls with k as cond does and against projection error.
* x*_chart on the raw release is far from the truth (0.74–1.0) with objective 1e-3 … 2e-2: on RAW privates no chart point
  is at the floor, and the k-trend (objective falls ×10 from k = 16 to 48 while cond rises ×16) is the fidelity / conditioning
  trade-off itself.

## What could not be computed as specified
* CIFAR P8 variant-B pairs: the bootstrap CIFAR job (355988) had not reached variant B when this ran.
* P7 landing rates at k = 32, 48 (any chart) and for the local chart: no WP2 cell exists on this release (355939 / 355940
  ended inside k = 32; the local chart was never a WP2 chart). `pca_perclass` is the same chart as `pca` here.
* SD-VAE: no saved latent chart at k ≤ 48.
* The `ae` join is approximate (retrained decoder). The `onchart` ae release is therefore NOT WP2's release.
