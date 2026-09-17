# Verification: the multilayer additivity result (section M / T5.2) — read-rows, PASS #1

**Register: read-rows** (the job's own JSON/stdout output), NOT read-prose. Section M in the ledger stands
UNAUDITED with zero passes; nothing in `experiments/multilayer_cert/RESULTS.md` reaches Gal until two independent
PASSes exist. This is the first.

## Claim under test

`experiments/multilayer_cert/RESULTS.md` §5 and `theory/T5.md` T5.2 report stacked chart-Jacobian rank
`[9, 18, 20, 20]` for one to four adapted layers, agreeing with `min(k_1, sum_l q_l)`.

## What ran (corrected job attribution)

The `[9, 18, 20, 20]` additivity rows are **job 688036** (`theory_checks.py::check_T5`), replicated identically in
runs **674726, 683234, 686846** — all seed 0. They are **NOT** in the survival sweep 692603 (which has no T5 rows;
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
