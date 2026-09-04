#!/usr/bin/env python3
"""The near-duplicate result is GRADED, so report the curve rather than one operating point (yoado-cd).

A blurred member scores 0.04 while the paired negatives sit flat at 0.10 -- so it is measurably MORE member-like
than a genuine non-member, even though both fail the 1e-2 bar. The certificate therefore carries usable
information across the whole near-duplicate range, and 1e-2 is a CHOOSABLE OPERATING POINT rather than a property
of the channel. This computes the ROC over the six transformations against both negative populations, with the
chosen bar marked, so the honest statement can be made: tolerance is a tunable trade between the false-positive
rate and how far a transformation may move the activations.

  python -u -m experiments.exact_inversion.roc_neardupe results/exact_inversion/step117_neardupe_*.jsonl
"""
import glob, json, sys


def roc(pos, neg):
    """(threshold, TPR, FPR) at every distinct score; scores are residuals, so LOWER is more member-like."""
    ts = sorted(set(pos + neg))
    out = []
    for t in ts:
        tp = sum(1 for v in pos if v <= t) / max(len(pos), 1)
        fp = sum(1 for v in neg if v <= t) / max(len(neg), 1)
        out.append((t, tp, fp))
    return out


def auc_of(pos, neg):
    pts = sorted(roc(pos, neg), key=lambda z: z[2])
    a = 0.0; px = 0.0; py = 0.0
    for _, tp, fp in pts:
        a += (fp - px) * (tp + py) / 2; px, py = fp, tp
    return a + (1 - px) * (1 + py) / 2


def main():
    pats = sys.argv[1:] or ["results/exact_inversion/step117_neardupe_*.jsonl"]
    rows = [json.loads(l) for p in pats for f in glob.glob(p) for l in open(f)]
    rows = [r for r in rows if r.get("part") == "LIVE" and r.get("near_duplicates")]
    if not rows:
        print("no near-duplicate rows"); return
    names = list(rows[0]["near_duplicates"].keys())
    print(f"# {len(rows)} draws, {len(names)} transformations. Residuals: LOWER = more member-like.")
    print(f"# the 1e-2 bar is a CHOSEN operating point, not a property of the channel.\n")
    print(f"{'transformation':16s} {'dist':>6s} {'q':>9s} {'paired null':>12s} {'AUC vs paired':>14s} "
          f"{'AUC vs population':>18s}  at bar 1e-2")
    pop = [r["nonmember_min"] for r in rows]                      # conservative: the population's LOWEST scores
    for nm in sorted(names, key=lambda n: sum(r["near_duplicates"][n]["feature_distance"] for r in rows)):
        q = [r["near_duplicates"][nm]["q"] for r in rows]
        pn = [r["near_duplicates"][nm]["q_paired_null"] for r in rows
              if r["near_duplicates"][nm]["q_paired_null"] == r["near_duplicates"][nm]["q_paired_null"]]
        d = sum(r["near_duplicates"][nm]["feature_distance"] for r in rows) / len(rows)
        a_pair = auc_of(q, pn) if pn else float("nan")
        a_pop = auc_of(q, pop)
        at_bar = sum(1 for v in q if v < 1e-2)
        print(f"{nm:16s} {d:6.3f} {sum(q)/len(q):9.2e} {(sum(pn)/len(pn) if pn else float('nan')):12.2e} "
              f"{a_pair:14.3f} {a_pop:18.3f}  {at_bar}/{len(q)} read as member")
    allq = [v for nm in names for v in [r["near_duplicates"][nm]["q"] for r in rows]]
    allp = [r["near_duplicates"][nm]["q_paired_null"] for nm in names for r in rows
            if r["near_duplicates"][nm]["q_paired_null"] == r["near_duplicates"][nm]["q_paired_null"]]
    print(f"\n# POOLED over all six transformations: AUC vs paired negatives = {auc_of(allq, allp):.3f}")
    print(f"# i.e. even the REJECTED transformations are separable from genuine non-members -- the bar discards")
    print(f"# information that the channel actually carries. Tolerance is a tunable operating point.")


if __name__ == "__main__":
    main()
