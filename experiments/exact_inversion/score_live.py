#!/usr/bin/env python3
"""Score the live-regime rows exactly as yoado-b9 specified. Aggregation only -- the per-draw criterion is in the
run and is not recomputed here.

Three code-level points were checked against `live_regime.py` before this was written, and all three hold:
  1. BAR is the literal constant 1e-2, never a function of the row. A per-row numerical floor is ~1e-15 in FP64
     and a certificate that had gone completely vacuous would pass it; a per-row bar also moves with the cell, so
     the across-draw binomial would be over a changing criterion.
  2. Void draws leave the DENOMINATOR. A gate failure is a draw that did not test the hypothesis, not a failed
     one; counting it as a failure would bias the rate downward for a reason unrelated to specificity.
  3. `draw_pass` is a conjunction with the NULL DOMINANT. A draw whose paired null lands below the bar has shown
     the certificate annihilating an image it never saw, and no population false-positive rate redeems that.

What this script adds: the across-draw rate is computed WITHIN each margin stratum and never pooled, and the
paired null's precondition is ASSERTED rather than assumed -- draw d's member must be absent from draw d-1's
training data, which holds by construction only if the draw indices are distinct.

  python -u -m experiments.exact_inversion.score_live results/exact_inversion/step116_live_*.jsonl
"""
import glob, json, sys


def cp(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0: return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def main():
    pats = sys.argv[1:] or ["results/exact_inversion/step116_live_*.jsonl"]
    rows = [json.loads(l) for p in pats for f in glob.glob(p) for l in open(f)]
    rows = [r for r in rows if r.get("part") == "LIVE"]
    if not rows:
        print("no LIVE rows"); return
    for opt in sorted({r["optimiser"] for r in rows}):
        rs = sorted([r for r in rows if r["optimiser"] == opt], key=lambda z: z["margin_stratum"])
        draws = [r["draw"] for r in rs]
        assert len(draws) == len(set(draws)), (
            "PRECONDITION VIOLATED: draw indices repeat, so a member may appear in the previous draw's training "
            "data and the paired null is not a null.")
        print(f"\n=== optimiser = {opt}   ({len(rs)} draws, all distinct images: assertion holds) ===")
        print(f"{'stratum':>8s} {'margin':>9s} {'q_member':>10s} {'q_null':>10s} {'FPR':>8s} {'min_u q':>10s}  verdict")
        for r in rs:
            print(f"{r['margin_stratum']:8d} {r['initial_margin']:+9.3f} {r['member_residual']:10.2e} "
                  f"{r['null_same_image_untrained_release']:10.2e} {r['false_positive_rate']:8.4f} "
                  f"{r['nonmember_min']:10.2e}  {r['verdict']}")
        nv = [r for r in rs if not r["verdict"].startswith("void")]
        q = max(1, len(nv) // 4)
        print(f"\n  rate WITHIN margin strata (never pooled), Clopper-Pearson exact:")
        for i in range(0, len(nv), q):
            g = nv[i:i + q]
            k = sum(1 for r in g if r["verdict"] == "success")
            lo, hi = cp(k, len(g))
            ms = [r["initial_margin"] for r in g]
            print(f"    margin {min(ms):+7.2f} .. {max(ms):+7.2f}:  {k}/{len(g)}   95% [{lo:.3f}, {hi:.3f}]")
        print(f"  voided (left OUT of numerator and denominator): {len(rs) - len(nv)} of {len(rs)}")
        print(f"  q_member is RECORDED, NOT SCORED -- an algebraic identity at N' = 1.")
        print(f"  min_u q(u) is reported, not a criterion: a single non-member at 1e-6 inside a passing 1% budget "
              f"is the row that would say the zero set is catching natural images.")


if __name__ == "__main__":
    main()
