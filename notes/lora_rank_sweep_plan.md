# LoRA rank sweep — is the q_eff reversal rank-robust?

## THE EXPERIMENT (what is being tested — read first)

**In one line:** we have a CONFIRMED result — at convergence, **10-class q_eff < binary q_eff** (multi-class
leaks *less* in the noise-robust sense), via a mechanism where 10-class couples *more* seed-noise into the
signal subspace (**iso_ratio 10-class > binary**). This sweep asks: **is that reversal rank-robust across
LoRA rank r ∈ {2, 4, 8, 16, 32}** — and in particular, does it **break at r ≥ N=10** (r=16, 32), where LoRA
≈ full fine-tune (NTK regime, Jang 2024)?

**The confirmed anchor (r=8, verified from job mc_lock_484948.out; N=10 k=8 T=1000 lr=0.5, both memorized):**
| base | iso_ratio | q_eff\|col(J) at ε=1 |
|---|---|---|
| binary (nc=2) | 0.49 | **59**/80 |
| 10-class (nc=10) | 0.68 | **36**/80 |
Reversal + iso-gap both real, clearest at LOW ε (0.1/0.3/1); washes out at high ε. r_J = 80 (full) for both.

**The sweep (independent variable):**
- **r ∈ {2, 4, 8, 16, 32}** — powers of two; **r=8 is the confirmed anchor** (must reproduce).
- **Regime split (the headline axis):** **r < N** (2, 4, 8) vs **r ≥ N** (16, 32). At r ≥ N=10 LoRA
  converges to ~full fine-tune (Jang 2024) → the low-rank structure that plausibly *drives* the
  noise-coupling reversal may weaken or vanish. **Prediction to test: the reversal is an r<N (genuinely
  low-rank) phenomenon and attenuates at r≥N.**

**Fixed (the verified convergence regime):** N=10, k=8 (Nk=80), lr=0.5, T=1000, mnist gelu, both bases
(nc 2 & 10), S=320 (=4·Nk), tangent qr, seed 42. This is the clean lr≤0.6/T≤1000 differentiable island
where both bases fully memorize 10 images (1/class for 10-class).

**Measured per (r, base):** q_eff|col(J) at ε∈{0.1,0.3,1,3,10}; **iso_ratio** = tr(Σ_J)/(μ·r_J); r_J;
max_bce (convergence check). 

**The reads:**
1. **q_eff(10-class) vs q_eff(binary) across r** — does the reversal HOLD at every r, or break?
2. **iso-gap = iso(10-class) − iso(binary) across r** — is the mechanism monotone/constant/vanishing in r?
3. **r<N vs r≥N split** — does the reversal attenuate in the NTK regime (r=16,32)?
4. **ε-dependence** — the reversal lives at low ε; confirm it stays low-ε across r.

---

## Configs (exact — no new code; pure param sweep of a validated config)

For each **r ∈ {2,4,8,16,32}** × base **nc ∈ {2,10}** (10 configs), run per-config:
1. **FULL-config FD-gate** on the actual J (`[dimY(r), 80]`, dimY scales with r) — abort-on-fail.
2. **rigor** — convergence check: `--rigor --dataset mnist --activation gelu --num_classes {2,10} --N 10
   --k 8 --rank {r} --Ts 1000 --lr 0.5 --tangent qr --seed 42 --save`. Require **max_bce < 1e-3**.
3. **j1** — leakage: `--j1 --dataset mnist --activation gelu --num_classes {2,10} --N 10 --k 8 --T 1000
   --rank {r} --tangent qr --S_list 320 --shrink_list 0.01 --eps_list 0.1 0.3 1 3 10 --seed 42 --save`.

**Datasets:** mnist for the full 5-rank sweep; **fashion at r∈{8,16} only** (the r<N→r≥N crossing) as a
generality check — not a full fashion sweep. Same crossing behavior on fashion ⇒ the rank claim generalizes;
different ⇒ a dataset-specific rank-dependence worth reporting.

---

## Correctness gates (the ones that actually bite here)

0. **CODE GOTCHA — thread the sweep's r everywhere (VERIFIED against the lock script).** The lock script
   `scripts/run_mc_lock_attempt_wexac.sh` **hardcodes `rank=8`** in three places: the FD-gate
   `_mnist_ctx(..., rank=8, ...)` (line 36) and `--rank 8` in the rigor/j1 lines (54/56/65/67). The sweep
   script MUST thread the sweep `r` into ALL of them — **especially the FD-gate**, else it silently FD-gates
   a rank-8 J while the runs use the sweep r (wrong-size J validated). First fix before any run.
1. **r_J-PER-r = the MASTER VALIDITY GATE (resolves Q1, elevated from a side metric).** q_eff is "recoverable
   out of r_J directions"; the binary-vs-10class comparison is like-for-like ONLY if r_J is the same at each
   r. r_J is domain-bounded at ≤Nk=80 for ALL r (adapter dim r·1784 ≫ 80 never binds), so it *should* be 80
   — but it can drop below 80 at BOTH ends: **low r (r=2)** the factorization may lack capacity to encode all
   80 perturbation directions; **high r (r=32)** over-param can DILUTE signal so some col(J) directions fall
   below tol=1e-8. So **report r_J per r FIRST**: where r_J=80 → clean raw-count comparison; where r_J<80 →
   compare the **FRACTION q_eff/r_J** (same col(J)-dim caveat), never raw counts across different col(J) dims.
2. **Convergence-per-r (bites at LOW r):** r=2 may NOT memorize 10 images even at N=10 — and the reversal is
   a *convergence* claim, so an unconverged cell's q_eff is CONFOUNDED. Gate each config on **max_bce <
   1e-3**; flag/exclude non-converged r.
3. **FD-per-r (bites at HIGH r) — gate r=32 FIRST, do NOT change the recipe if it fails.** The chaos-onset
   island (lr≤0.6/T≤1000) was characterized at r=8; r=32's bigger 2nd-order graph could shift it. **FD-gate
   r=32 as a PRE-CHECK** before the full sweep. Pass → proceed. **FAIL → that's a finding (island is
   r-dependent) and r=32 is BOUNDED-OUT** — do NOT drop to lower lr/shorter T for r=32 only (a different
   recipe breaks cross-r comparability, the confound we've fought throughout). Report the r-range where the
   *matched* recipe is measurable. *(Nice symmetry: the measurable window is bracketed — convergence gate at
   low r, FD gate at high r.)*
4. **S adequacy — S=320 is enough at EVERY r, no bump.** `q_eff_colspace` estimates Σ_J inside the ≤80-dim
   col(J)-projection; the r-dependent dimY does NOT enter that estimate, so S≥4·r_J (≤320) covers all r. The
   full-Σ undersampling (eff_rank(Σ_seed)≈S−1) is irrelevant — we read col(J)-restricted. Run the **{S,2S}
   stability check at r=8 (anchor) AND r=32** (the stress case where col(J) has the tiniest σ, not r=16).
5. **Anchor reproduction:** r=8 MUST reproduce iso 0.49/0.68 and q_eff|col(J) ε=1 = 59/36 — if it doesn't,
   the sweep harness differs from the locked run and nothing downstream is trustworthy.
6. **Metric law:** leakage = r_J + q_eff, NEVER eff_rank; phrase as "≥ X directions" (q_eff|col(J) is a
   conservative lower bound). GELU exact-J; float64; **bsub-only, never local.**

---

## Compute

10 configs (5 r × 2 bases) × [rigor T=1000 + j1 S=320]. r=32 has the biggest adapter/J → most mem/time;
estimate walltime at r=32 first. **long-gpu** (deep T=1000 unroll). FD-gate each. Split by walltime only
(no config-count cap). r_J is S-invariant so it's cheap; the S=320 q_eff is the cost.

---

## Deliverable

- **Table:** r × {binary, 10-class} → q_eff(ε), iso_ratio, r_J, converged? (max_bce).
- **Plots:** q_eff-vs-r (both bases, at ε=1 the reversal-clearest) + iso-gap-vs-r, marking the **r=N=10
  crossing** (between r=8 and r=16). 
- **Headline read:** reversal rank-robust across r<N? does it attenuate/break at r≥N (NTK regime)?

---

## Audit resolutions (yoado-35 — all folded into gates/configs above)

- **Q1 metric-validity across r: RESOLVED VALID.** r_J is domain-bounded ≤Nk=80 for all r (adapter dim
  r·1784 ≫ 80 never binds) → col(J) is the same ≤80-dim subspace at every r → q_eff is always "recoverable
  out of ≤80." Over-param changes signal-spread / noise-coupling = exactly the phenomenon under test, not a
  confound. And the headline is a WITHIN-r binary-vs-10class GAP, so any intrinsic r-dependence is
  COMMON-MODE and cancels. Enforced by the r_J-per-r master gate (#1).
- **Q2 S=320:** adequate at every r (gate #4). **Q3 FD-island:** r=32 pre-check, bound-out not
  recipe-change (gate #3). **Q4 datasets:** mnist full sweep + fashion at the CROSSING only (see configs).

## Science framing (do NOT pre-commit the outcome)

"Attenuates at r≥N" is a **HYPOTHESIS, not an expected result.** The mechanism (CE injects more seed-noise
into col(J)) is a LOSS property (softmax-residual structure), not obviously rank-specific — so the reversal
could **attenuate** at r≥N (low-rank bottleneck forces noise into col(J); over-param lets it spread),
**persist** (fundamental CE property), or **strengthen** (bigger B0 → more init noise for CE to couple).
**All three are informative** — the result says whether the reversal is a low-rank artifact or a general CE
property. Two bonuses: (a) by Jang 2024, r≥N ≈ full-FT, so r=16/32 incidentally give the
**full-fine-tuning-regime** reversal — extends the reach beyond LoRA. (b) the anchor r=8 is NEAR the N=10
boundary (not deep low-rank), so the low-rank signature (if any) should be clearest at **r=2,4** — weight
those in the read.
