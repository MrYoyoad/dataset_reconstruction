# The replay fibre has a closed form, and it is the capacity line

**Approved by yoado-a8 (math/science approver), 2026-09-17.** Measurement: job `350944` (six cells) on top of
`350928` (the original four). Machinery: `experiments/e1b/fibre_dimension_check.py`, built by the executor lane
(yoado-6e); I added the dimension arguments to sweep it and have disclosed that edit to them.

## The statement

For seed-free replay against the **full** released `A_T` and `B_T`, the local dimension of the family of
`(H, seed)` reproducing the release is

```
nullity  =  max( 0 ,  N · [ d − ( (m−1) + r − N ) ] )
```

zero exactly when `d ≤ (m−1) + r − N`, i.e. when `m + r ≥ d + N + 1`. **This is the project's own capacity line
`k < m + r − N`, evaluated where the unknown is the representation itself rather than a chart coordinate.**

## Why

Two independent derivations reached it and reconcile to one formula, which is the reason it is trusted rather
than fitted.

The released `A_T` pins the seed one-for-one on the complement of the training span, so those unknowns and
their equations cancel. The approver split the seed by its `r` index (1024 pinned, 512 left); I split it by its
`d` index (1344 pinned, 192 left). **The split is a bookkeeping choice** — whatever is called pinned cancels
against its own equations — and subtracting leaves `dN − N((m−1)+r−N)` either way. Neither decomposition
contains an error; they are the same statement.

The load-bearing step is what `B_T` actually carries. It is **not** `m·r` free numbers: it lies on the rank-`N`,
zero-column-sum variety of dimension `N((m−1)+r−N)` — 280 here, not 480. The 200 equations that go missing are
exactly that difference, and they are the whole of the previously unexplained gap.

## The falsifier, fired six of six

Predictions were fixed in the job script's header before any row existed. Everything but the swept dimension is
held; fp64 throughout.

| cell | predicted | measured |
|---|---|---|
| head width 40 | 72 | **72** |
| head width 48 | 8 | **8** |
| head width 49 | 0 | **0** |
| head width 64 | 0 | **0** |
| batch 4 | 100 | **100** |
| batch 1 | 22 | **22** |

Unreduced and reduced arms agree in every cell, which confirms again that the reduced parametrisation changes
the size of the search and **not** identifiability. Singular gaps run twelve orders; at head width 49 the rank
equals the unknown count exactly and the trailing singular value is **identically zero**, so that cell is not
threshold-sensitive at all. **The one-unit discontinuity between adjacent head widths 48 and 49 is the part a
fitted curve does not produce.**

## Two corrections, recorded as corrections

**Mine.** I said a wider head *or* a smaller batch closes the deficit. The batch does not close it. `N` appears
twice — as the multiplier and inside the bracket — so shrinking it lowers the nullity (measured 100 at four, 22
at one) but the bracket at this head is `21 + N`, positive for every batch of at least one.

**The approver's.** They said `N` factors out entirely so the batch cannot help. It does not factor out, and the
batch does help; their conclusion held and their reason did not. **Only the head closes it**, at
`m ≥ d + N + 1 − r = 49`.

## The chart cells: ORACLE standing, and a confirmation rather than a discovery

Every chart cell returned nullity 0. Two labels belong on that, and without them it would be over-read.

**It is the formula's own prediction, not an independent finding.** With a chart the unknowns are `Nk`, so
`nullity = max(0, N(k − ((m−1)+r−N)))`. The bracket is 35 at the original head and larger at every wider one,
and `k = 12` sits below all of them, so zero is what the closed form says in all six cells. It is worth
reporting — a formula predicting cells it was not derived on is the point — but as confirmation.

**The chart is an oracle** (disclosed by the executor lane). The data is generated as `H = LW + b` and the search
is parametrised with that same `L` and `b`, so the truth lies in the chart by construction. An attacker does not
have it. So the correct statement is **not** "a chart closes the deficit" but **"a chart that CONTAINS the
private representations closes it"** — a mechanism a chart must supply, not a property any buildable chart has.
The composing measurement belongs in the same breath: **public charts sit at projection error 0.24–0.32 while
landings need ≤ 0.0124, so public charts do not contain the truth.**

## The useful consequence: an arithmetic ceiling on chart width

Since the chart case is `N(k − ((m−1)+r−N))`, the deficit **reopens** above `k = (m−1) + r − N` — 35 at this
configuration. That turns "try a different chart" into a constraint: **any chart wider than 35 here reopens
non-identifiability, however well it is built.** It is the same line as the glossary's `k < m + r − N`.

This also predicts the executor lane's chart-width sweep outright, including its self-check: at `k = d = 64` the
chart is the whole space and the formula returns `8 × (64 − 35) = 232`, the free-`H` value.

## Standing

E1B in the free-feature configuration is an **identifiability negative** — recovery of `H` jointly with the seed
is not unique — and the negative is now **predicted**, with a formula stating exactly what would change it. That
is stronger than a null. Two scopes: the family is a **seed-against-`H` trade** (all of it moves `H`, none of it
moves `H` with the seed held fixed, in every cell), and identifiability is not recoverability — the seed-known
arm has nullity 0 and the solver is still failing from random starts, which the executor lane is measuring
separately.

**The cell that matters next** is the same measurement with an **attacker-buildable** chart, where a different
answer is expected: a chart that does not contain the truth cannot take the nullity to zero.
