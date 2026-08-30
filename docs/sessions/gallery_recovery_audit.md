# Adversarial audit — notes/gallery_recovery_plan.md (closed-world training-set recovery)

Auditor: yoado-72, 2026-08-31, at yoado-40's request. Read-only; plan not edited. Ranked worst-first.

## Verdict in one line
The framing is honest and the gates are the right ones, but the **atom model in §2 is the wrong object for a LoRA
adapter**, and the **verify step has no demonstrated resolution for single-image errors** — fix those two before
any build, because together they can make the whole pipeline an oracle-init upper bound dressed as an attack
(exactly your Q2 fear).

## 1 · BLOCKER — the dictionary atoms live in the wrong frame for LoRA
§2 writes ΔW = BA ≈ Σ g_i x_iᵀ with atoms a(x) = g(x)·xᵀ. That is the FULL-weight one-step update. What a LoRA
adapter exposes after one step is the projected operator P_LoRA(H) = B₀B₀ᵀH + HA₀ᵀA₀ (thesis_note_v2.md:26) — and
in THIS repo LoRA inits A₀ = 0, B₀ random (STATUS.md "J1 mechanism", ~665-723: ∂L/∂A = B₀ᵀ∂L/∂W carries the data,
∂L/∂B = 0 at step 1). So the observation is H filtered through the COLUMN SPACE OF A RANDOM B₀, and after T steps
B and A co-evolve, so ΔW = B_T A_T is not Σ (atoms at θ₀) at all (that is the anchor/linearization issue you list
in §7 as a failure mode — it is not a failure mode, it is the measurement model).
Consequences: (a) correlating full-gradient atoms g(x)xᵀ against ΔW measures the wrong thing, and the "true member
ranks ~1400th" weakness may be an artifact of the frame mismatch, not of the marginal score; (b) under the
REALISTIC attacker B₀ is unknown, so no atom can be placed in the right frame analytically.
Fix: build the dictionary the way the adapter was built — the atom for gallery image x is the ADAPTER trained on
{x} (or the LoRA update after the same T steps), computed with the attacker's OWN random init, compared in the
gauge-invariant subspace distance; matching pursuit then runs over SUBSPACES (residual = the part of col/row-space
of ΔW not explained by the selected atoms), not over vectors. This is the discrete analogue of direct weight
inversion only if F is the actual LoRA training map — which is what you say in §1 but not what §2 computes.
Cheap pre-test on existing data: rank the true members of an atlas adapter with (i) full-gradient atoms and (ii)
one-image LoRA-adapter atoms under a random init; report both ranks. If (ii) ≫ (i), Q1 answers itself.

## 2 · BLOCKER — verify has no demonstrated single-image resolution
§2(c)/(d) assumes the gauge-invariant subspace distance can tell S* from S* with ONE image swapped, from a single
retrain. The evidence you cite (atlas ARI≈1.0) separates 5 compositions that differ by WHOLE digit-subsets. The
program's own single-swap result needed the WHITENED metric with K=50 seeds per side to see one swap (sensitivity
d² ≈ 20 at N=4..16; near-duplicate swaps 0.03-0.07 ≈ invisible; similarity ladder, arm B). One retrain gives one
sample from the seed cloud; if the one-swap displacement is inside that cloud in the subspace distance, the nudge
loop cannot converge on the exact set — it will random-walk among sets that verify "consistent".
Gate to add BEFORE build (CPU, existing tensors): on the atlas zoo, compute D(S*, S* reseeded) [seed noise] and
D(S*, S* with one image swapped) [signal] for the subspace distance at |S|=4. Require signal/noise ≫ 1 (say ≥3)
at ONE retrain, or budget K retrains per verify and switch verify to the whitened d² (the ruler you already
trust). This also answers Q3 more usefully than an abstract identifiability proof.

## 3 · Q3 — identifiability: yes, do a check first, and it is cheap
Two layers: (i) necessary condition rank(M) ≥ N (identifiability_rank_bound.tex) — trivially met at width 1000,
N=4; (ii) uniqueness over the GALLERY = dictionary coherence. OMP exact-recovery guarantees need mutual coherence
μ < 1/(2N−1) (Tropp 2004); MNIST same-digit atoms correlate ~0.8, so the guarantee fails by an order of magnitude
and greedy WILL substitute lookalikes. Compute μ of the (correctly framed, §1) dictionary and the Gram spectrum
of the true atoms before building; report "exact" AND "lookalike-tolerant" recovery (see §5). Plant-and-recover
is a necessary gate, not sufficient: it passes trivially when |G|=200 has no lookalikes and fails for reasons that
have nothing to do with the method when it does — run it at two coherence levels.

## 4 · Q2 — forcing the oracle/realistic distinction to surface (by design, not by vigilance)
(a) Every verify runs in three arms, always: ORACLE-init (true seed), REALISTIC (K fresh inits, report the
distribution), CROSS-RECIPE (different lr / T / activation, as B1 did). The headline is pre-registered as the
REALISTIC arm; oracle goes to an appendix table.
(b) Null-verify control: D(S*, S* reseeded) must NOT be separable from D(S*, best wrong S) under the oracle arm
only. If oracle separates and realistic does not, print "UPPER BOUND" on the figure automatically.
(c) Seed-swap sanity: run the whole pipeline against an adapter trained on S* with a DIFFERENT seed than the
attacker assumes; realistic recovery must be unchanged, oracle recovery must drop. If oracle does not drop, the
"oracle" arm is leaking the init through some other path (e.g. shared RNG state) — check for that too.
(d) Report per-image "membership contribution" from the residual under both arms; if the realistic arm's
contributions are flat while the oracle's are peaked, you are looking at the frame mismatch of §1.

## 5 · Baselines — one is missing and it is the dangerous one
The atlas already shows CONTENT-level recovery is easy (+0.989 acc-diff: which DIGITS were present). Exact-set
recovery will therefore be inflated by class composition alone. Add the class-matched random baseline: random
N-subsets with the SAME digit labels as S*. Report Jaccard / precision@N against THAT, not against unconstrained
random (whose chance rate 1/C(|G|,N) is astronomically small and proves nothing). Also report a "semantic"
recovery (right digit, wrong exemplar) column so lookalike substitutions (§3) are visible rather than hidden in
a low exact rate.

## 6 · Q1 — you are neither under- nor over-selling; you are measuring the wrong seed
The arithmetic is right ((1−0.861)·10k ≈ 1390), but the marginal score that gave AUC 0.861 (which one? — cite the
job) is not MP iteration 1 with correctly framed atoms. Re-measure the marginal rank with §1's one-image-adapter
atoms in the subspace distance before deciding the seed is weak. Prediction: it improves a lot; if it does not,
then the joint search must do the work AND §2's resolution problem bites, so the honest expectation is
partial (Jaccard) recovery, not exact.

## 7 · Smaller items
- Threat model: say explicitly that the attacker knows N (or sweep N and report recovery vs assumed N).
- Metric: "exact-set recovery rate" needs many independent (S*, seed) draws to have a CI; pre-register the count.
- Compute: MNIST-MLP LoRA retrains are seconds; the K-seed verify is affordable — budget it rather than argue
  around it.
- Scope line for every number: "closed-world (gallery of |G|), this attacker (gallery + LoRA-frame atoms), N=4,
  MNIST-MLP, realistic-init unless labelled oracle."
- Literature: your closed-world SELECTION is close to membership-inference-over-a-pool and to "dataset
  inference"; the OMP-with-verify loop resembles greedy set-membership attacks on gradients (e.g. label/batch
  recovery in iDLG-style work). Check those before claiming novelty of the loop; the LoRA-frame dictionary (§1)
  is the more defensible novel element.

## Go / no-go recommendation
No build until: (1) atoms redefined in the LoRA frame (one-image adapters, subspace MP); (2) the CPU resolution
gate of §2 passes on atlas tensors (signal/noise ≥ 3 at one retrain, or verify switched to whitened d² with K seeds);
(3) the class-matched random baseline is in the pre-registration. All three are cheap and use existing data.
