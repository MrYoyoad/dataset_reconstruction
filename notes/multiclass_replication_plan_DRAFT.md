# Multi-class replication of all leakage knobs — EXECUTABLE PLAN (rev2)

Draft: executor session. Substance audit: yoado-35 (ex-8a). Code-state audit: yoado-19. Executes: yoado-dc.
**rev2 folds in yoado-35's substance audit + the T-sweep tightenings + yoado-19's corrections.**

**Goal:** a clean **binary vs 10-class** comparison of every leakage knob on the *honest* base models
(`weights-<ds>10_<act>.pth`; binary `weights-<ds>_<act>.pth`) — filling the binary-only coverage gaps,
on one metric law, cheapest-signal-first, **at matched CONVERGENCE not matched T**.

## 0. Metric law (NON-NEGOTIABLE — yoado-35/8a; top of every table)
- **Leakage = `r_J` (hard rank of col(J)) and `q_eff`. NEVER `eff_rank`** (it DROPS as CE/smoothness
  concentrate the spectrum even while r_J maxes → reads backwards; report only as a labelled spectrum
  diagnostic).
- **`r_J` is strictly S-INVARIANT** — `#(svals(J) > tol)` from **J alone**, computed once, no S-loop. Use
  ONE minimal S (~16, only so `q_eff_colspace` can run); print σ straddling `tol=1e-8` to make r_J
  bulletproof. This is the cheap knob → the whole plan is r_J-first, q_eff last.
- **`q_eff`**: `S≥4·Nk` is a FLOOR (tighter 4·r_J); the REAL adequacy gate is (i) convergence — q_eff
  stable across {S, 2S}; and (ii) `eff_rank(Σ_seed) ≳ r_J` (the log prints it; 247834 had 62.8 at S=64 =
  undersampled). Conditioning-aware for free. Never quote a q_eff failing both.
- `q_eff|col(J)` is a conservative **LOWER BOUND** → phrase "≥ X directions".
- All runs GELU (exact-J), float64, honest θ0, **never local — always bsub**. modifiedrelu = accuracy
  table only (out of exact-J).

## 1. The two questions this plan answers
1. **Is the "multi-class ~2× amplification" real, or a training-SPEED artifact?** (THE load-bearing risk.)
   247834 (multiclass) was UNDERFIT at T=200 (max_bce~0.017); the binary r_J=99 was T=50 @ default lr~0.01
   — also plausibly underfit. My log check shows convergence-by-fixed-T is
   NON-UNIVERSAL: binary N=4 k8 at **lr=0.1** converges by T=200 (max_bce 6.48e-4) while 247834 (Nk=160,
   **lr~0.01**) is underfit — but those points differ in BOTH Nk AND lr (10×), so this proves
   non-universality, it does NOT isolate Nk as the cause (could be the lr). Robust takeaway: check
   `max_bce<1e-3` **per (Nk, base) cell with lr held FIXED across both bases**, and compare at matched
   convergence AND matched plateau — Phase 1.
2. **DOMAIN-limited (r_J tracks Nk for all K) or MEASUREMENT-limited (r_J plateaus below Nk at a
   K-dependent value)?** At k=8/N=20 we are AT the ceiling r_J=Nk=160; binary already plateaus (r_J≈99 <
   160) ⇒ binary is measurement-limited at ~100. Need HEADROOM ⇒ adaptive **k-ladder**, **N=20 focus**.

## 2. Code wiring BEFORE any run
- **`run_schemes`**: add `num_classes=2, classes_present=None` → forward into its `_mnist_ctx(...)` +
  CLI dispatch (currently binary-only). `J_s = J·P` valid multi-class (chain rule on `a`, label-agnostic).
- **Binary byte-identical regression gate**: after the wire, `num_classes=2` must reproduce the shipped
  binary numbers (guards the baseline).
- **Value-only unroll path** for noise sampling: `estimate_sigma_seed` calls `forward_Y`
  (`create_graph=True`) S times → builds+discards a T-step 2nd-order graph S× (S=2560 at k32/N20). Add a
  `create_graph=False` / detached-SGD forward for VALUE-only samples — likely the difference between
  long-gpu-feasible and needing an A100. (For J itself keep create_graph; only the Σ samples are
  value-only.)
- **`subhead_k`** (Phase 5): ABSENT — build IN PARALLEL during Phases 1–4 (zero latency); the K′ logit
  slice must INCLUDE the `classes_present` label indices (classes_present=2 → classes {0,1}, slice
  `[0:K′]`, K′≥2).
- `--num_classes/--classes_present` CLI already wired to j0/j1/h1/rigor.

## 3. Phases — r_J-first (cheap), q_eff last (expensive)

### Phase 0 — gates (Stage 0 of every script; minutes)
`toy_ad_gate(num_classes=10)` FD<1e-6 abort-on-fail; binary-regression gate (above); one tiny
`run_h1 --num_classes 10 --N 2 --k 8` smoke returning r_J.

### Phase 1 — r_J vs T to MATCHED PLATEAU, BOTH bases (LOAD-BEARING; r_J-only → cheap; do FIRST)
Merges the confound test AND the old overtraining-collapse worry into ONE front-loaded sweep (yoado-35):
```
T-ladder {5, 50, 200, 1000, ...} × base∈{binary(K=2), 10-class(K=10)} × (mnist,fashion),
  N=4 (+N=20 spot), k=8, activation gelu, seed 42, SAME lr for both bases.
```
- Report `r_J`, `max_bce`, `memorized` per (T, base). **Stop each ladder when max_bce<1e-3 AND r_J FLAT.**
- **Report the amplification at the matched PLATEAU** (both bases converged AND r_J stopped moving), or
  explicitly bound as "at fixed T". If r_J never plateaus post-memorization → that IS the finding
  (leakage genuinely T-dependent). If it COLLAPSES from softmax saturation → the old Phase-6 worry is real.
- **Recipe control (yoado-35):** T and lr are NOT equivalent for J — higher lr = shallower unroll (less
  meta-grad chaos, coarser linearization point); higher T = deeper unroll (chaos risk). Prefer a moderate
  lr bump over very deep T; **FD-spot-check J + NaN/Inf guard at the LARGEST T** (Stage 0 gates only T=5).
  Use the SAME lr for binary and CE (different recipes would confound the comparison).
- **This decides whether the headline survives.** Everything downstream reports at the T this fixes.

### Phase 2 — DIRECTIONS knob (Knob 1) on 10-class, r_J-only (CHEAP)
```
--h1 --num_classes 10 --dataset {mnist,fashion} --h1_methods pca qr difference pca_tail residual \
     --N {2,4} --k {8,16,32} --T <Phase-1 plateau> --seed 42
```
- r_J, col(J) overlap between methods, hard_rank. Compare to binary H1 (does "difference collapses into
  col(J)" hold at 10-class; does CE dissolve the direction-dependence?).

### Phase 3 — SCHEMES knob (Knob 2), BOTH bases, r_J-only (CHEAP; needs the wire)
```
--schemes --num_classes {2,10} --dataset {mnist,fashion} --N {2,4} --k {8,16,32} --T <plateau> --seed 42
```
- DIFFERENT (P=I_Nk) / SAME (1_N⊗I_k) / MIXTURE (rank-deficient blend). **SAME ceiling: r_J(SAME) ≤ k for
  BOTH bases** (P has rank k; CE cannot raise it) — the contrast is *does CE FILL the k-subspace (r_J=k)
  while binary falls short (r_J<k)?* MIXTURE caps r_J at `r_mix·k` (=80 at N=20,k8); **verify the impl
  mixes `a`, NOT `y`** (blending labels would change J and break the identity).

### Phase 4 — N-SWEEP + the DOMAIN-vs-MEASUREMENT test (Knob 3), r_J-only (CHEAP; the §1.2 answer)
```
--j1 --num_classes {2,10} --dataset mnist --N {2,4,10,20} --k {8,16,32(,64,128 adaptive)} \
     --T <plateau> --S_list 16 --seed 42
```
- Plot **r_J vs Nk per K on the k-ladder**, **primarily at N=20** (small-N is below any plausible limit →
  uninformative). **Adaptive:** push k∈{64,128} ONLY if r_J still == Nk at k=32 (r_J is S-invariant → this
  is cheap, cost ~linear in Nk of the J-build). DOMAIN-limited ⇒ r_J on the Nk diagonal; MEASUREMENT ⇒
  plateau below Nk (higher for 10-class if K drives leakage). **Binary plateau ≈100 = the positive
  control.** If the measurement limit itself scales with k (no plateau) → "effectively domain-limited in
  the accessible regime" is the finding.

### Phase 5 — HEAD-WIDTH knob (yoado-35's; sequenced, subhead_k built in parallel)
Fixed 10-class base, `classes_present=2`, CE over first **K′ logits** (K′∈{2,3,5,10}), k=16, S≥1280.
Isolates measurement-count from base-θ0 AND classes-present. Role set by Phase 4: **confirmation** if
measurement-limited, **decisive disambiguator** if domain-limited.

### Phase 6 — CLEAN q_eff on the headline cells (EXPENSIVE; do LAST, only cells Phases 1–4 flagged)
```
--j1 --num_classes {2,10} --N <cell> --k <cell> --T <plateau> \
     --S_list <4·Nk and 8·Nk for the {S,2S} convergence check> --eps_list <shipped> --seed 42
```
- q_eff(ε) binary vs 10-class at matched Nk AND matched plateau, with BOTH adequacy gates printed
  (convergence {S,2S}; eff_rank(Σ_seed)≳r_J). Value-only sampling path (§2). This is the headline number.

## 4. Deliverables for Gal
1. **r_J vs T, both bases** (Phase 1) — the confound resolved: amplification at matched plateau.
2. **r_J vs Nk k-ladder** (Phase 4) — domain-vs-measurement, binary-plateau positive control.
3. **Knob coverage matrix** binary vs 10-class (DIRECTIONS × SCHEMES × N × r_J,q_eff) — cells filled.
4. **q_eff(ε) binary vs 10-class** on headline cells (clean S) + head-width K′ ladder.

## 5. Compute & ordering
- Phases 1–5 are **r_J-only → cheap** (single minimal S; S-invariant). Run first/parallel, split ≤ a few
  configs/job (short-gpu runlimit lesson), `hname!='hgn46'`, Stage-0 gate, save `.pth`+metrics line.
- Phase 6 (clean q_eff, S=4–8·Nk, value-only sampling) is the cost risk — ONLY the 3–4 cells r_J flagged;
  N=20 k=32 → Nk=640, S up to 5120 → long-gpu/A100, float64.
- **Ship gated on the executing user's compute-scope call** (yoado-19 correctly won't presume it).

## 6. Honesty gates
Toy-AD@10 + binary-regression pass first; never eff_rank as leakage; flag under-adequate q_eff; "≥X
directions"; report at matched convergence+plateau (not fixed T); FD-gate the deepest-T J.
