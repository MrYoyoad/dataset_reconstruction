#!/usr/bin/env python3
"""Shared identifiability machinery: the local dimension of the family of parameters reproducing a release.

One implementation, so that two harnesses cannot disagree for untraceable reasons. Extracted from
experiments/e1b/{fibre_dimension_check, chart_capacity_sweep, basin_and_conditioning}.py, which were three copies
of the same SVD with the same conventions.

WHAT THIS MEASURES. Given a residual map `res(*params) -> vector` that is zero exactly when the candidate
reproduces the release, the local dimension of the solution family at the truth is the NULLITY of its Jacobian
there. This owes nothing to any solver: nullity 0 means the truth is locally isolated and recovery is well posed;
nullity > 0 means a family of parameters reproduces the release exactly and no solver can pick the truth out of it
without further information.

FIVE CONVENTIONS THAT MUST MATCH ACROSS HARNESSES, or two nullities are not comparable:

1. EVALUATE AT THE TRUTH. The nullity is local and at the ground-truth parameters. A nullity elsewhere measures a
   different point and is not a statement about identifiability of the truth.

2. THE RESIDUAL MUST MATCH THE FULL RELEASE, both factors, and must be BLOCK-NORMALISED -- each block divided by
   the Frobenius norm of its own target. Without normalisation the blocks enter the SVD at arbitrary relative
   scale and the rank cut is meaningless. E1B v1 matched only `A_s @ H_cand` against `A_T @ H_true`: that is
   `r*N` equations instead of `r*d`, and it reported nullity 1704 instead of 232.

3. THE RESIDUAL MUST NOT USE GROUND TRUTH IN ITS TARGET. Only released quantities. E1B v1's target was built from
   the true `H` -- the very unknown being solved for -- which made the whole run experimenter-side. If your
   residual needs the truth to evaluate its target, the nullity it yields is not about an attack.

4. RANK FLOOR. Default here is relative to the largest singular value, which is safe ONLY because convention 2
   makes the Jacobian O(1). Pass `abs_floor` whenever the matrix can legitimately VANISH -- a relative-only
   threshold calls a numerically zero matrix FULL rank, which is exactly backwards for an annihilated certificate.
   That is mandatory for any defence evaluation: merged and balanced-factorisation releases give `C = 0`
   identically, and a relative test would report a closed channel as open.

5. A CHART MUST CONTAIN THE TRUTH. If the parametrisation is a chart, the truth must be exactly representable in
   it, or "the nullity at the truth" does not exist -- there is no truth in the domain. Assert it, do not assume
   it, and remember that a chart built to contain the truth is an ORACLE chart and not attacker-available.

`gap_at_cut` is reported so the cut is inspectable rather than asserted: a cut sitting in a gap of many orders is
not threshold-sensitive, one sitting in a smooth decay is.

6. **A NULLITY IS ONLY MEANINGFUL IF THE SPECTRUM HAS A GAP.** On a deep trained network the residual Jacobian can
   decay smoothly from 1.0 to 1e-17 with no gap anywhere — measured on a 15-layer MNIST MLP, where the rank slid
   from 16014 to 16122 across ten decades of threshold (job 355750). There "rank" and "nullity" are not properties
   of the matrix, they are choices of tolerance, and reporting an integer is false precision. `rank_ladder()` below
   is the check: a real rank is FLAT across decades. Synthetic and shallow cells here show gaps of 1e9–1e12 and a
   count that does not move at all, so the contrast is unmistakable when you look — and invisible when you do not.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.func as tfn


@dataclass
class Identifiability:
    n_equations: int
    n_unknowns: int
    rank: int
    nullity: int
    gap_at_cut: float
    condition_number: float
    singular_values: list = field(repr=False, default_factory=list)

    def as_dict(self):
        d = self.__dict__.copy()
        d["singular_values"] = [float(v) for v in self.singular_values[:32]]
        d["singular_values_tail"] = [float(v) for v in self.singular_values[-32:]]
        return d


def jacobian_at(res: Callable, params: Sequence[torch.Tensor], mode: str = "rev",
                chunk_size: Optional[int] = None) -> torch.Tensor:
    """Flattened Jacobian of `res(*params)` at `params`. `rev` costs one pass per EQUATION, `fwd` one per UNKNOWN
    -- pick by whichever is smaller, since for a 2016 x 96 problem they differ by 20x.

    `chunk_size` bounds peak memory: without it the vmap materialises one full copy of the intermediate state per
    output (or per input), which on a 16000-equation problem is tens of GB. It does not change the result."""
    fn = tfn.jacrev if mode == "rev" else tfn.jacfwd
    kw = {"chunk_size": chunk_size} if chunk_size else {}
    try:
        J = fn(res, argnums=tuple(range(len(params))), **kw)(*params)
    except TypeError:                         # older torch: no chunk_size on this transform
        J = fn(res, argnums=tuple(range(len(params))))(*params)
    return torch.cat([j.reshape(-1, p.numel()) for j, p in zip(J, params)], dim=1)


def identifiability(res: Callable, params: Sequence[torch.Tensor], rtol: float = 1e-10,
                    abs_floor: Optional[float] = None, mode: str = "rev",
                    chunk_size: Optional[int] = None) -> Identifiability:
    """Nullity of the residual Jacobian at `params`. See the module docstring for the five conventions.

    res        callable taking *params and returning a 1-D residual, block-normalised, released quantities only
    params     the GROUND-TRUTH parameters (a chart's latents if the search is over a chart)
    abs_floor  absolute singular-value floor. MANDATORY wherever the Jacobian can legitimately vanish.
    """
    J = jacobian_at(res, params, mode=mode, chunk_size=chunk_size)
    sv = torch.linalg.svdvals(J)
    floor = abs_floor if abs_floor is not None else rtol * float(sv[0])
    rank = int((sv > floor).sum())
    n_in = J.shape[1]
    gap = float(sv[rank - 1] / sv[rank]) if 0 < rank < len(sv) else float("inf")
    cond = float(sv[0] / sv[rank - 1]) if rank > 0 else float("inf")
    return Identifiability(n_equations=int(J.shape[0]), n_unknowns=int(n_in), rank=rank, nullity=int(n_in - rank),
                           gap_at_cut=gap, condition_number=cond, singular_values=[float(v) for v in sv])


def rank_ladder(r: "Identifiability", thresholds=(1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16)):
    """Convention 6. Rank as a function of the relative threshold, and how far it moves.

    Returns (ladder, spread). A spread of a few units across ten decades is a real rank; a spread of dozens means
    the spectrum has no gap and the nullity should be reported as UNDEFINED, with this ladder, not as an integer."""
    sv = r.singular_values
    if not sv or sv[0] <= 0:
        return [], 0
    ladder = [(t, int(sum(1 for v in sv if v > t * sv[0]))) for t in thresholds]
    return ladder, max(x[1] for x in ladder) - min(x[1] for x in ladder)


def assert_chart_contains(chart: Callable, latents: torch.Tensor, truth: torch.Tensor, tol: float = 1e-12) -> float:
    """Convention 5. Returns the relative reconstruction error, and raises if the chart does not contain the truth."""
    rel = float(torch.linalg.norm(chart(latents) - truth) / torch.linalg.norm(truth))
    if rel > tol:
        raise AssertionError(f"chart does not contain the truth: relative error {rel:.3e} > {tol:.0e}. "
                             f"The nullity at the truth is undefined because the truth is not in the domain.")
    return rel


def block_normalised_residual(sim: Callable, targets: Sequence[torch.Tensor]) -> Callable:
    """Convention 2, as a helper: `sim(*params)` returns simulated blocks in the same order as `targets`, and each
    block is differenced against its target and divided by that target's Frobenius norm."""
    norms = [torch.linalg.norm(t) for t in targets]
    def res(*params):
        out = sim(*params)
        return torch.cat([((o - t) / n).reshape(-1) for o, t, n in zip(out, targets, norms)])
    return res
