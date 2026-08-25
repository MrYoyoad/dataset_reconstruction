# Multi-class replication of all leakage knobs — EXECUTABLE PLAN (draft by yoado-dc-executor session, for yoado-19 audit)

**Goal:** a clean **binary vs 10-class** comparison of every leakage knob on the *honest* base models
(`weights-<ds>10_<act>.pth` multi-class; `weights-<ds>_<act>.pth` binary). So far most knobs were
measured on binary only. Fill the coverage gaps, on one metric law, cheapest-signal-first.

## 0. Metric law (NON-NEGOTIABLE — from yoado-8a audit; put at top of every table)
- **Leakage = `r_J` (hard rank of col(J)) and `q_eff`. NEVER `eff_rank`.** eff_rank DROPS as CE/smoothness
  concentrates the spectrum even while r_J maxes out → it reads *backwards*. Report eff_rank only as a
  spectrum-shape diagnostic, clearly labelled "not leakage".
- **`r_J` is S-robust** (it's just `#singular values of J > tol`) → **cheap**: needs only enough noise
  samples to span col(J), or none (r_J is computable from J alone, no Σ). This is the cheap-signal knob.
- **`q_eff` needs S ≥ 4·Nk** or it is undersampled (the amp run's q_eff=97 was S=64 < r_J=160 → rough).
  Budget S = 4·Nk per clean-q_eff config. **Never quote a q_eff whose S < 4·Nk without an "undersampled"
  flag.**
- `q_eff|col(J)` is a conservative **LOWER BOUND** on true q_eff → phrase as "at least X directions".
- All runs: GELU (exact-J), float64, honest θ0, **never local — always bsub**. modifiedrelu = accuracy
  table only (guarded out of exact-J).

## 1. The central open question this plan must answer
At k=8, N=20 we are **already at the domain ceiling** r_J = Nk = 160. You cannot see scaling against a
ceiling. So the headline design question is:

> **DOMAIN-limited (r_J tracks Nk for all K) or MEASUREMENT-limited (r_J plateaus below Nk at a
> K-dependent value)?**

To answer it we need **headroom above the ceiling**: raise k so Nk ≫ the plausible measurement limit.
Every headline config therefore runs a **k-ladder {8, 16, 32}** (Nk = {N·8, N·16, N·32}) and asks whether
r_J tracks Nk or plateaus. This is the single most important axis; it is cheap (r_J only).

## 2. Code wiring needed BEFORE any run (small, param-only elsewhere)
- **`run_schemes`**: add `num_classes=2, classes_present=None` to its signature and pass them into its
  `_mnist_ctx(...)` call (currently omitted → binary-only). P is a loss-agnostic coordinate map, so
  `J_s = J·P` is correct once the base J is multi-class — no other change. **Verify** with the toy-AD gate
  at `num_classes=10` (Stage 0) before trusting numbers.
- **`--num_classes`, `--classes_present`** CLI args already exist and are wired to run_j0/j1/h1/rigor.
  run_schemes just needs the two forwarded (add to its CLI dispatch line too).
- **`subhead_k`** (head-width knob, §Phase 4) is ABSENT — clean slate. Only implement if we run Phase 4;
  keep it out of the critical path (Phases 1–3 need no new code beyond the run_schemes wiring).

## 3. Phases, cheapest-signal-first

### Phase 0 — gate + smoke (Stage 0 of every script; ~minutes)
`toy_ad_gate(num_classes=10)` must pass (FD < 1e-6) — abort-on-fail. One tiny end-to-end
`run_h1 --num_classes 10 --N 2 --k 8` smoke to confirm the multi-class path returns r_J.

### Phase 1 — DIRECTIONS knob (Knob 1) on 10-class, r_J only (CHEAP; ~1 GPU-h)
Multi-class currently has only `qr`. Run the full basis set param-only:
```
--h1 --num_classes 10 --dataset {mnist,fashion} --activation gelu --seed 42 \
     --h1_methods pca qr difference pca_tail residual --N {2,4} --k {8,16,32} --T 200
```
- **Report:** `r_J`, `col(J)` overlap between methods, `hard_rank`, and (labelled non-leakage) eff_rank.
- **Compare to binary** (existing H1 numbers): does the "difference tangent collapses into col(J)"
  result (binary: overlap ~0.84 N=2) hold at 10-class? Does multi-class dissolve the direction-dependence
  (CE amplification "dissolves collinearity")?
- **T = 200** (memorization; T=5 was the underfit corner — leakage grows to plateau). One-line T=50 spot
  check to confirm we're on the plateau, not climbing.

### Phase 2 — SCHEMES knob (Knob 2) on BOTH honest bases, r_J only (CHEAP; needs the wiring; ~1 GPU-h)
After the run_schemes wiring + gate:
```
--schemes --num_classes {2,10} --dataset {mnist,fashion} --activation gelu \
          --N {2,4} --k {8,16,32} --T 200 --seed 42
```
- DIFFERENT (P=I_Nk) / SAME (1_N⊗I_k) / MIXTURE (rank-deficient blend). Report `r_J`/`hard_rank(J_s)` +
  deterministic recovery per scheme, binary vs 10-class side by side.
- **Key contrast:** SAME restricts to k reinforced coords → r_J should cap at ~k·(rank of reinforcement);
  does 10-class CE change the SAME/MIXTURE ceilings vs binary?

### Phase 3 — N-SWEEP knob (Knob 3), r_J across the k-ladder (CHEAP; ~1–2 GPU-h)
```
--j1 --num_classes {2,10} --dataset mnist --activation gelu \
     --N {2,4,10,20} --k {8,16,32} --T 200 --S_list <small, r_J-adequate>  --seed 42
```
- **This is where §1 is answered.** Plot `r_J vs Nk` for each K on the k-ladder. DOMAIN-limited ⇒ r_J = Nk
  on the diagonal; MEASUREMENT-limited ⇒ r_J plateaus at a K-dependent value below Nk (and the plateau
  should be HIGHER for 10-class if "K drives leakage").
- r_J only here (small S). Pick the 2–3 most informative (N,k,K) cells for Phase 5 clean-q_eff.

### Phase 4 — HEAD-WIDTH knob (yoado-8a; SEQUENCED follow-up, needs `subhead_k`)
Isolates *measurement count* from base-θ0 AND classes-present: **fixed 10-class base**,
`classes_present=2`, CE over the first **K′ logits** (sub-head K′∈{2,3,5,10}), k=16, S≥1280.
- **Decision for yoado-dc:** this is a genuinely distinct 4th knob (it moves ONLY the observed-logit
  count, holding the base and the data fixed) → worth doing, but it needs the `subhead_k` implementation
  first. **Recommend: sequence it AFTER Phases 1–3** so the critical path needs no new code. If Phases 1–3
  already show r_J is MEASUREMENT-limited and tracks K, Phase 4 is the clean confirmation; if
  DOMAIN-limited, Phase 4 becomes the decisive disambiguator. Implement `subhead_k` then.

### Phase 5 — CLEAN q_eff on the headline cells (EXPENSIVE; S = 4·Nk; ~2–6 GPU-h)
Only after r_J has picked the interesting cells. For ~3–4 (N,k,K,dataset) cells that showed the
binary↔multiclass contrast most sharply:
```
--j1 --num_classes {2,10} --N <cell> --k <cell> --T 200 --S_list <4·Nk> --eps_list <as shipped> --seed 42
```
- Report `q_eff(ε)` binary vs 10-class at matched Nk, with the S ≥ 4·Nk adequacy flag printed. This is the
  headline "multi-class CE ~2× leakage" claim, now on clean S and the full knob set.

### Phase 6 — OVERTRAINING T-sweep (secondary cut; cheap add-on)
For 1–2 headline cells: `--Ts 50 200 1000`. **Question:** does CE-saturation at very long T *collapse*
r_J back toward the binary wall (transient leak) or hold (stable leak)? Report r_J-vs-T.

## 4. Deliverable tables (what goes to Gal)
1. **r_J vs Nk across the k-ladder**, one line per K∈{2,10} × dataset — answers domain-vs-measurement.
2. **Knob coverage matrix** binary vs 10-class: DIRECTIONS × SCHEMES × N × (r_J, q_eff), cells filled.
3. **q_eff(ε) binary vs 10-class** on the headline cells (clean S).
4. r_J-vs-T (overtraining) + the head-width K′ ladder (if Phase 4 runs).

## 5. Compute budget & ordering
- Phases 1–3 + 6 are **r_J-only → cheap** (small S, S-robust); run them first, in parallel, split ≤ a few
  configs/job to fit short-gpu runlimit (the 6-config runlimit lesson).
- Phase 5 (clean q_eff, S=4·Nk) is the cost risk — do ONLY the 3–4 cells r_J flagged. At N=20 k=32,
  Nk=640 → S=2560 samples × per-sample forward-Y; budget mem/queue accordingly (likely long-gpu or A100
  for float64).
- Every script: Stage-0 gate, `python -u`, `hname!='hgn46'` exclusion, save `.pth` + a metrics line.

## 6. Gates / honesty
- Toy-AD gate at num_classes=10 passes before any real run.
- Never report eff_rank as leakage. Flag every q_eff with S < 4·Nk.
- State leakage as "≥ X directions" (q_eff is a lower bound).
- T=200 (memorization) is the reporting regime; show it's on the plateau, not climbing.

---
**Open questions for yoado-19 audit:** (a) is the k-ladder {8,16,32} enough headroom, or do we need k=64 at
small N to guarantee Nk ≫ measurement limit? (b) Phase 4 `subhead_k` — 4th knob now or sequenced? (c) for
SCHEMES multi-class, does J_s = J·P remain valid when P mixes across images that now have DIFFERENT class
labels (the MIXTURE blend across classes)? (d) datasets: mnist+fashion enough, or add the CIFAR base for
R5? (e) is S=4·Nk the right adequacy bar for q_eff, or should it scale with the noise conditioning?
