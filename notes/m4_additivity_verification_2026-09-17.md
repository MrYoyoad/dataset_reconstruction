# Verification: the multilayer additivity result (section M / T5.2) — read-rows, PASS #1

**Register: read-rows** (the job's own JSON/stdout output), NOT read-prose. Section M in the ledger stands
UNAUDITED with zero passes; nothing in `experiments/multilayer_cert/RESULTS.md` reaches Gal until two independent
PASSes exist. This is the first.

## Claim under test

`experiments/multilayer_cert/RESULTS.md` §5 and `theory/T5.md` T5.2 report stacked chart-Jacobian rank
`[9, 18, 20, 20]` for one to four adapted layers, agreeing with `min(k_1, sum_l q_l)`.

## What ran (corrected job attribution)

The `[9, 18, 20, 20]` additivity rows are **job 688036** (`theory_checks.py::check_T5`); the same **data** (the
`independent:`/`shared_seed:` stdout lines) appears in runs **674726, 683234, 686846** — all seed 0. This is
replication of the numbers, **not** of verdicts: 674726 is a known-bad job (its `T2_PropA` reports `passed: true`
with `max_rho_base: 0.00e+00` over NaN/`inf` rows, and a T5 row recorded the negation of the correct conclusion),
so its `passed`/`note` fields are untrustworthy and are not relied on here — only the measured `k1`/`q_l`/rank. They are **NOT** in the survival sweep 692603 (which has no T5 rows;
it is the M1/M2/M3 drift sweep). *(The relayed attribution of M4 to "job 692603" is incorrect; source is 688036.)*

Setup: `dims=[40,30,30,30,8]`, `k=20`, `r=12`, `N=3`, so `r-N=9`. Measured from the run's own stdout, identical
across all four job runs:

    independent: k1=20  q_l=[9, 9, 9, 9]  rank J_F(1..L)=[9, 18, 20, 20]  predicted=[9, 18, 20, 20]
    shared_seed: k1=20  q_l=[9, 9, 9, 9]  rank J_F(1..L)=[9, 18, 20, 20]  predicted=[9, 18, 20, 20]

## Found — both laws evaluated per stack depth, from the measured q_l and k1

With `q_l=[9,9,9,9]` (cumulative `[9,18,27,36]`) and `k_1 = 20`:

| stacked L | measured `rank J_F` | T5.2 `min(k_1, Σq_l)` | corrected `min_j (d_j + Σ_{l<j} q_l)` |
|---|---|---|---|
| 1 | 9  | 9  | 9  |
| 2 | 18 | 18 | 18 |
| 3 | 20 | 20 | 20 |
| 4 | 20 | 20 | 20 |

**The two laws coincide on every row, and both match the measurement.** This is not a coincidence of this cell: it
is forced whenever the frozen path is **rank-preserving** (`d_j = k_1` at every adapted layer). Proof: if `d_j = k_1`
for all `j ≤ L`, then each `j ≤ L` term is `k_1 + Σ_{l<j} q_l ≥ k_1` (minimised at `j=1`, giving `k_1`), and the
`j = L+1` term is `0 + Σ_{l≤L} q_l = Σq`. So `min_j(...) = min(k_1, Σq)`, which is T5.2 verbatim. Here the MLP has
widths `30 ≥ k = 20` and random GELU weights, so it is rank-preserving by construction.

## Verdict

**M4 confirms T5.2 in its valid (rank-preserving) regime and discriminates NEITHER law from the other.** It is a
PASS for the additivity measurement as reported, and it is **not** evidence for or against the corrected law F11
raised — the two are identical here.

## Scope caveats that must travel with this PASS

1. **The rows do not carry per-layer `d_j = rank M_l` for `l > 1`** — only `k_1 = rank M_1 = 20`. `d_j = k_1 ∀j` is
   inferred from the architecture (widths ≥ k, random), not measured in this job. A clean measured discrimination
   needs `d_j` recorded per layer.
2. **Single seed** (seed 0), replicated across four job runs but not across seeds.
3. **Random FP64 MLP.** Rank-preserving by construction; this is exactly why it cannot be generalised to real
   encoders, and exactly why the contracting evidence matters.

## Where the discrimination between the two laws actually lives

The **contracting** regime: `STATUS.md:323` / `notes/assumption_relaxation_program.md §19a`, jobs **218345 / 218346**
(2026-09-04) — the same measurement written up in two places (same setup, same `445` and `138`), so it is **one**
retrodiction, not two. There the real frozen encoder contracts rank with depth (`rank_Dphi` profile
`784,784,692,219,187,138,...`), and the depth-4 configuration delivers **445** of a **692** bound — the later
layers' encoders (220, 187, 138) too small to fill what the first left open. That is the corrected law's nesting
ceiling biting; T5.2 as stated predicts no such ceiling. Those jobs' rows carry `rank_Dphi_median` and `usable`
(the per-layer `d_j` and `q_l` analogues in pixel space); turning them into a clean per-row two-law comparison is
the next verification, and it is where a genuine discrimination can be recorded. The depth-7 row (138 delivered
against a 138 transmit cap) is capped identically under both laws and is evidence for neither.

## R2 / M5

`shared_seed` and `independent` both give `[9, 18, 20, 20]` (job 688036, all four runs): tying `A_0` across layers
does **not** collapse additivity, so a shared `A_0` is **not** a defence on its own. The T5 defence claim is already
downgraded to a conjecture about tying the *whole* adapter. No further compute needed here.

## Follow-up (same evening): the real-scale discrimination is NOT cleanly extractable from the §19a rows — stop

The discrimination between T5.2 and the corrected law lives only in a **contracting** net (`d_j < k_1`). The real
encoder ladder is such a net, and it is measured in jobs **218345 / 218346** (`STATUS.md:323` = §19a; the same
jobs, so one measurement). I attempted to turn those saved rows into a per-layer two-law comparison. The rows **do**
carry the ingredients: `ENCODER` rows give per-layer `rank_Dphi` (= `d_j = rank M_j`) and `n_prime` (= `N'_l`), and
`PIXELRANK` rows give cumulative usable across a tolerance ladder (`1e-6/1e-8/1e-10`). **But they cannot be
reconciled into a reliable comparison from the saved format:**

- The `ENCODER` rows carry **no config key** (`adapted_layers = None`), so mapping `d_j` to a starting-depth config
  requires file-order grouping, which is unreliable: one reconstructed alignment gives a measured 4-layer stack of
  **717** against the corrected law's **proven upper bound 468** — a violation that can only mean the `d_j` were
  assigned to the wrong config.
- Independently, **every** config's first-layer pixel-usable is `248 = r - N`, which is inconsistent with the small
  `rank_Dphi` (138, 104, …) the same file reports for deep layers — so the reconstructed `d_1`/`q_1` are not the
  quantities the stacked measurement actually used.
- The headline number itself **moves with the tolerance cut**: one config reads `345 / 411 / 445` at
  `1e-6 / 1e-8 / 1e-10`. An elbow that moves with the cut is a property of the cut.

**Per the standing rule, I stop rather than report a discrimination built on a guessed alignment.** Note also that
the relayed "`445 = d_2 + q_1 = 220 + q_1` at `q_1 = 225`" uses a **backed-out** `q_1`; the measured first-layer
usable is `248`, which gives the corrected upper bound `d_2 + q_1 = 220 + 248 = 468`, and the measured stack sits
**below** it (`445 @ 1e-10`, climbing from `345 @ 1e-6`) — consistent with the corrected law **as an upper bound**,
tolerance-limited, but **not** an exact numeric confirmation.

**§19a establishes NO retrodiction — numeric OR qualitative (corrected 2026-09-17, my earlier "signature" read was
itself an over-claim).** Saturation *per se* is NOT discriminating: T5.2 `= min(k_1, Σ q_l)` **also** predicts later
layers adding exactly zero once `Σ q_l ≥ k_1`. Only saturation strictly **below** `k_1` distinguishes the two laws.
The depth-7 config saturates *at* `k_1 = 138` — consistent with **both** laws. The depth-4 config is the only row
that could saturate below `k_1`, and it is exactly the row the alignment defect corrupts. And on defect (a), the
measured 717 exceeding a **proved** upper bound of 468 refutes the **alignment**, not the theorem (the bound is
proved). So §19a currently supports neither law over the other. **A clean real-scale discrimination needs an
instrumented run** that records `d_j`, `q_l`, and stacked rank **on the same row with a config key**, across a
pre-registered tolerance ladder, and is pre-declared capable of producing saturation *at* `k_1` (the
non-discriminating outcome) as well as below it. The synthetic contracting discrimination is already done (F11:
0/12 T5.2, 12/12 corrected); what is missing is the magnitude by which depth falls short on a real architecture.
