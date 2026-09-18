# Drift harness (P3 / P6) — design notes and smoke read-out

Builder notes for `experiments/multilayer_cert/drift_cert.py` + `scripts/run_drift_cert_wexac.sh` (2026-09-18).
Plan: `notes/plan_2026-09-18_multilayer_parameter_program.md` P3, P6 and its "Audit 2026-09-18" items 1–4, which
supersede the package text. The coordinator writes RESULTS.md; nothing here is a claim.

## What the harness is

The audit's missing object: a checkpoint-loaded MLP (d15 twin via `load_deep`; the strong 784-1000-1000-10 via its
own `state_dict` loader) with LoRA on a **chosen subset** of layers (`--adapt`, an adapter mask over
`deep_stack.inputs_of` — `None` = frozen), trained by plain SGD / momentum (zero buffers) / scalar weight decay
(`P ← (1−ηλ)P − η v`, `v ← μ v + ∇`), FP64. At the truth it records, per adapted layer, survival.py's quantities
with the same definitions (`layer_stats`, ported line by line from `survival.measure`): `delta`, `delta_perp`,
`delta_final`, `drift_rank`, `N_prime` (span over `t < T`), `rank_B_T`, `gap`, `rank_C_full` (absolute floor
`ref=‖A_T‖_F`), `rho_full`, `rho_trunc`, `beta`, the divergence gate (rel drift > 1e3 or β > 1e6 or non-finite);
plus the P3/P6 fields: `contaminated = rank_B_T < N_prime`, `K_l = rho_trunc/delta_perp`, `seed_scale = (1−ηλ)^T`,
`rank_C_at_seed_floor` (σ(C) > 1e-10·seed_scale·‖A_0‖) beside `rank_C_at_AT_floor` (= `rank_C_full`), the full σ
ladders of `B_T` and `C_full`, and the per-image residual at the truth through the base map.

The solve is the ladder's (`certificate.lm_cert`, same random starts `Generator(seed+31)`, same landing criterion
1e-2, same degenerate guard 0.05 on the feature norm), objective `C_l φ_l(ψ(w))/‖A_l φ_l(ψ(w))‖` with `φ_l` the
**public base network's** input to layer `l` (the `t = 0` point of the training span, annihilated exactly by the
full certificate by Prop. A). `--stack-below` concatenates the lower adapted layer's certificate; `--solve-cert`
chooses full / trunc / both. References: the raw truth (tier 1) and `x*_chart` from an oracle-started LM (audit
item 10a). Tier 2 reuses `experiments.utils.perceptual_id.score_image` (truth + 99 public decoys, SSIM and base-model
feature ranks) on two arms: `found_best` (start nearest each truth, the ladder's arm) and `attacker` (best-objective,
non-degenerate, de-duplicated at 0.05, ≤ N candidates, each truth scored against its nearest candidate — coverage).

## Decisions the spec left open (say so, do not work around)

1. **Verdict set has a fifth value, `chart-limited`.** With raw privates on a PCA-32 chart the truth is not in the
   chart (proj err ≈ 0.3 on the ladder), so tier-1 landing on the truth is impossible by construction and `x*_chart`
   is the reference (audit 10a). A start that reaches `x*_chart` while `x*_chart` is > 1e-2 from the truth is neither
   an optimisation failure nor an alias; labelling it as either would be wrong. It is reported as `chart-limited`
   with `err_opt_truth` on the row. The four spec values keep their meaning; `recovered` also fires when `x*_chart`
   is within 1e-2 of the truth and a start reaches it.
2. **Alias** (verdict) = a non-degenerate start at the truth's floor (objective ≤ max(1e-20, 100 × objective at the
   raw truth)) that is at no image's `x*_chart` and on no truth. Bootstrap's weaker `alias_in_chart` (objective ≤ the
   chart optimum, elsewhere) is kept as a field only — it fired at residual 0.09 on the smoke's raw PCA chart.
3. **`eps_land`** = max √objective over starts that reached some `x*_chart` (every cell records it; the control cell
   `is_control=True` is the one Rule B uses; the join is by (model, target, T, lr, seed) — the coordinator's).
4. **`N_prime`** uses `span_of`'s relative 1e-10 (survival's definition, kept verbatim as instructed) and the row
   carries the σ ladder around the cut so an absolute-floor reading can be recomputed. `rank_C_full` uses the absolute
   floor (`numrank(ref=‖A_T‖_F)`) as in survival.
5. **A0 draw**: one `Generator(seed+7)` over the adapted layers in network order (so layer 2's A0 differs between
   `--adapt 2` and `--adapt 1 2`; the images are the same — the join key is the image indices, printed on the row).
6. **Head**: extended by a zero row (m = 11), all privates labelled with the new class — ladder `d15_letter_a` /
   `mlp_letter_a` verbatim. The gradient reaches the adapted hidden layers through the ten trained rows.
7. **Rule B's landing curve** needs `--solve-cert trunc` (or `both`); the P3 runner solves with `full` by default
   (`CERT=both` doubles the solve time). `K_l` itself needs no solve and is on every row.
8. **P6 λ grid**: `ηλT ∈ {0.01, 0.1, 1, 3, 10, 20}` → `λ = ηλT/(ηT)` derived in-job, plus `λ = 1/η` (the corner,
   seed_scale = 0 exactly) and `λ = 0`. `P6_LR` / `P6_RL` are placeholders the coordinator fills from P3.
9. **Loader gate**: test accuracy on 2000 MNIST digits must exceed 90% or the job exits 3 (catches a mis-assembled
   `state_dict` loader before any number exists).

## Smoke read-out — job 366163 (short-gpu after `bmod` off the saturated long-gpu per-user cap; A40; 146 s wall)

Rows: `results/multilayer_cert/drift_cert_smoke_366163.jsonl` (7 rows, script sha `9ad37f23917b`, git `8f0dfcf-dirty`);
tensors `results/multilayer_cert/drift_cert/*_366163.pth`; grids `figures/multilayer_cert/drift_cert/*_366163.png`.
Twin gate 97.55 % / strong gate 97.85 % on 2000 test digits (both PASS). 20 starts, k = 32, PCA chart (explained 0.805).

| cell | adapt→target | T | N' | rank B_T | rank C | rho_full | rho_trunc | K_l | reached x*_chart | tier-2 SSIM top-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| control | 2→2 | 1 | 8 | 8 | 56 | 4.5e-16 | 4.5e-16 | – | 12/20 (6/8 images) | 5/8 |
| control | 2→2 | 20 | 8 | 8 | 56 | 4.9e-16 | 4.9e-16 | – | 13/20 (6/8) | 5/8 |
| drift r16 | 1,2→2 | 1 | 8 | 8 | 56 | 3.8e-16 | 3.8e-16 | – | 13/20 (7/8) | 6/8 |
| drift r16 | 1,2→2 | 20 | **150** | 51 | 13 | 5.4e-7 | 2.5e-3 | 0.053 | 0/20 | 0/8 (contaminated) |
| strong r16 | 1,2→2 | 20 | 88 | 40 | 24 | 9.1e-8 | 7.2e-4 | 0.0089 | no solve | – |
| strong head | 2,3→3 | 20 | 116 | 10 | 54 | 1.1e-4 | 1.9e-4 | 0.044 | no solve | – (pre-reg. contaminated: yes) |
| options m0.9 wd1e-3 stack both | 1,2→2 | 20 | 160 | 64 | 0 | 7.7e-2 | 1.3e-2 | 0.0087 | full 0/5; trunc 5/5 (3/8) | trunc 3/8 |

Read-out, observations only:
- **Zero-drift control is at the FP64 floor**: rho_full 4.5e-16 / 4.9e-16, per-image residual at the truth ≤ 1.6e-14,
  rank C = 56 = r − N with a clean gap (σ₅₆ = 0.77, σ₅₇ = 1.8e-14). The harness reproduces the ladder's head-adapter
  numbers on a hidden layer with a frozen input.
- **Drift at T = 20, lr 0.01, r_lower 16** (twin, layer 2's input): δ = 4.96e-2, δ_perp = 4.69e-2 (almost all of the
  drift is orthogonal), drift_rank 152, **N' = 150 ≈ 8·(T−1)**: through the GELU every step adds a fresh 8-dim slice to
  the input span, so N' outruns r = 64 after ~7 steps and the full certificate is `contaminated` (rank B_T 51 < N',
  rank C 13 < k). rho_full there is 5.4e-7, not at the floor, as the contaminated branch predicts (no small parameter).
  The full-certificate solve is then vacuous (13 equations, 32 unknowns: x*_chart is not isolated; 0/20 reached it).
- **Rule B numbers exist on every row**: K_l = 0.053 (twin layer 2), 0.0089 (strong layer 2), 0.044 (strong head).
- **Tier 1 vs raw is 0 in every cell**, as on the ladder's PCA-32 cells (proj err ≈ 0.3): the reference that moves is
  x*_chart (reached by 12–13/20 starts at zero drift; 6–7/8 images) and tier 2 (SSIM top-1 5–6/8, feature top-1 0/8).
- **eps_land as defined (max residual at a reached x*_chart) = 0.18–0.19 at zero drift with raw privates**: that is the
  chart error's residual, not a landing floor, so Rule B's comparison K_l·δ_perp < eps_land is dominated by the chart
  there. Added after the smoke (not run): `--private onchart` trains the release on the projections so the truth is in
  the chart; `PRIVATE=onchart` on the runner. The p3 read-out for Rule B should use that arm.
- **Verdict fix after the smoke (not run)**: the first version labelled `alias (residual zero, ...)` from bootstrap's
  `alias_in_chart` (objective ≤ chart optimum, elsewhere), which fired at residual 0.09 on the raw chart. The verdict
  now requires the truth's floor (objective ≤ max(1e-20, 100·objective at the raw truth)); `alias_in_chart` and
  `n_starts_below_opt_elsewhere` stay as fields. Rows of 366163 carry the OLD label logic (sha 9ad37f23917b); the
  release, the drift numbers and every other field are unchanged by the patch (current sha 4751f212c763).
- Momentum 0.9 + wd 1e-3 at lr 0.01, T = 20: δ = 1.6 (not gated: rel drift < 1e3, β finite), rank B_T = 64 = r,
  rank C = 0 — the certificate is empty; the truncated one still reaches x*_chart 5/5 (3/8 images). The `--stack-below`
  path and `--solve-cert both` run.
- cusolver emitted its "SVD failed to converge, using fallback" warning inside `common.span_of` on the 150-column
  drift stack; the fallback result is what the row carries.

## Cost (from the smoke)
Twin target 2: 0.4 s/start (A40) → 400 starts ≈ 160 s; + 8 oracle starts + tier 2 + training ≈ 3 min per cell with
`CERT=full`; contaminated cells solve in seconds. Target 4 ≈ 1.5×, target 8 ≈ 3× (deeper `phi_l` in the Jacobian).
P3 = 15 jobs × 60 cells: twin t2 ≈ 3 h, t4 ≈ 4.5 h, t8 ≈ 9 h, strong t2/t3 ≈ 3 h each → ≈ 68 GPU-h, ≈ 9 h wall if
the jobs run in parallel (the long-gpu per-user cap of 70 GPUs was saturated by other lanes at smoke time — the
smoke ran on short-gpu). `CERT=both` doubles the solve part. P6 = 2 jobs × 48 cells ≈ 2.5 h each.
