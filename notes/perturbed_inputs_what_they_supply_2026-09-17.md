# What a perturbed input can and cannot supply

**Status:** derivation, mine (yoado-b3), 2026-09-17. Requested by the GM lane as a citable note because it had
existed only in message traffic. Companion to `notes/fibre_capacity_formula_2026-09-17.md`, which supplies the
deficit this is the remedy for.

**Why it exists.** The 15 September meeting record contains the suggestion *"look for additional equations
involving perturbed inputs."* The approver's standing principle is that **a question is not a workplan**, and that
reading a method into a question is how a clause becomes a project. This is the method the question lacked,
including the half that says which version of it cannot work.

## Setup

The attacker holds the frozen base model, the recipe, and the release `(A_T, B_T)` in full. Write `sim(θ)` for
the simulated release at candidate unknowns `θ` (the representation `H`, or chart coordinates `w`, together with
the seed), so the residual is

```
ρ(θ) = sim(θ) − (A_T, B_T)        and the truth is a zero of ρ.
```

Identifiability at the truth is `nullity(Dρ) = 0`.

## Part 1 — a probe of the release supplies no rank. This is a proof, not a heuristic.

Let `F` be **any** function the attacker can evaluate on a release: project `A_T` onto a chosen direction, take a
singular value, contract it with a perturbed feature, anything. It yields a new equation

```
F(sim(θ)) = F(A_T, B_T).
```

Its differential is `D(F∘sim) = DF · D(sim) = DF · Dρ`. So the augmented Jacobian is

```
[  Dρ  ;  DF · Dρ  ]
```

whose **row space is contained in the row space of `Dρ`**. Therefore

```
rank[ Dρ ; DF·Dρ ]  =  rank(Dρ),      and the nullity is unchanged, for every F.
```

**Any equation that is a function of the simulated release adds rows and adds no rank.** Probing `A_T` in a
chosen direction is the linear case. The equation *count* rises and the solution family does not shrink by one
dimension. This is the pre-registered failure mode, and it is the version of the suggestion that would otherwise
have been costed and run.

*Corollary worth keeping:* an equation count is not an identifiability argument. Two systems with the same
information can have arbitrarily different equation counts.

## Part 2 — what can add rank, and it is exactly one thing

By Part 1, a new equation helps only if it is **not** a function of the simulated release — i.e. if it constrains
`θ` directly. The unknown that carries the deficit is the representation `H`, treated in free-feature replay as an
arbitrary `d × N` matrix. The constraint the release does not already contain is:

> **`H` is the encoder evaluated at images, not a free matrix.**

That is what a chart *is*: `H = φ(ψ(w))`, `w ∈ R^k`. And **evaluating the frozen base model at perturbed inputs is
how such a constraint is built or verified** — the attacker owns the base model and may probe it anywhere, and
probing *it* (unlike probing the release) returns genuinely new numbers, because the base model is a function they
have not yet exhausted.

So the suggestion is not a side remark. It names the only source of additional rank available.

## Part 3 — how much it supplies, in closed form

From the companion note, free-feature replay has

```
nullity = max( 0,  N · [ d − ((m−1) + r − N) ] ).
```

A chart of dimension `k` replaces the `dN` representation unknowns with `Nk`, removing `N(d − k)` of them, so

```
nullity(chart) = max( 0,  N · [ k − ((m−1) + r − N) ] ),
```

zero exactly when `k ≤ (m−1) + r − N` — the glossary's capacity line `k < m + r − N`. At the standing cell the
bracket is 35, so **the constraint closes the deficit for any chart narrower than 36 and reopens it above.**

## The limit, which must travel with the result

The measurement showing the deficit closed used a chart that **contains the private representations by
construction** (the release's own generating chart — confirmed at the code, see the companion note). An attacker
does not have that. Two consequences:

1. The honest form is two-part: **the constraint this suggestion points at removes the shortfall, and no chart we
   can currently build supplies that constraint.**
2. The one bearing measurement: public charts sit at projection error **0.24–0.32** against a landing gate of
   **≤ 0.0124** — about **19× to 26×**, between one and one and a half orders. **Not two orders.** The distinction
   decides the reading: two orders reads as closed, twenty to thirty times reads as **open**, and open is the
   accurate state. A gap must be stated at the scale that determines whether it is crossable.

A chart that fails to contain the truth is not a partial remedy — it removes the deficit **and** the solution
together, since the truth is then not in the search space at all. So "narrow the chart" is not free advice: the
width ceiling of Part 3 and the containment requirement pull in opposite directions, and that tension is the real
open problem the suggestion surfaces.

## What to say to the supervisor

*You suggested additional equations from perturbed inputs. We measured the shortfall in closed form and confirmed
it against six cells by moving head width and batch size, hitting every prediction exactly. Probing the released
adapter cannot supply anything — it is a function of numbers we already hold, so it adds equations and no rank.
What your suggestion does point at is the one constraint the release lacks, that the representation is the encoder
evaluated at an image, and that constraint removes the shortfall entirely for any chart below a width our formula
gives. What we cannot yet do is build such a chart: ours are twenty to thirty times too coarse to contain the
private representations. That gap is the open problem, and it is one to one and a half orders, not two.*
