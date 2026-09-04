#!/usr/bin/env python3
"""Score the truncated-certificate rows under yoado-b9's PRE-FIXED rule, not under the best-looking `k`.

`truncated_certificate.py` emits the full per-`k` table and `B_T`'s spectrum, and its own printed line picks the
`k` with the best AUC. **That is a fit, not a result** -- b9's scoring spec forbids choosing `k` per module by its
outcome. This rescoring applies the rule fixed in advance instead:

  * `k` = the position of the LARGEST GAP in the log spectrum of `B_T` (data-driven, needs no private knowledge,
    and is the same rule already validated on the imprint spectra, where it sat at a clean 4.6-5.8 order break);
  * SUCCESS requires BOTH (a) perfect separation -- every member residual below every non-member residual, whose
    chance probability is 1/C(M+U, M) and must be <= 0.01, forcing M >= 5 and U >= 5 -- and (b) a gap of >= 2
    orders between the largest member residual and the smallest non-member residual;
  * VOID if fewer than 5 members or 5 non-members, or if the module's exact certificate is NOT vacuous, since the
    test's premise is that it is.

Anchors for the 2-order bar, both measured elsewhere and neither from this run: the exact certificate below its
line separates by >= 7 orders; the degraded wide-head case ran to a 20x gap and then to overlap.

  python -u -m experiments.exact_inversion.score_truncated results/exact_inversion/step110_trunc_*.jsonl
"""
import glob, json, math, sys


def choose_k(spectrum):
    """Largest gap in the LOG spectrum -- the rule, fixed in advance and identical for every module."""
    lg = [math.log10(max(v, 1e-300)) for v in spectrum]
    gaps = [(lg[i] - lg[i + 1], i + 1) for i in range(len(lg) - 1)]
    return max(gaps)[1], max(gaps)[0]


def main():
    pats = sys.argv[1:] or ["results/exact_inversion/step110_trunc_*.jsonl"]
    rows = [json.loads(l) for p in pats for f in glob.glob(p) for l in open(f)]
    rows = [r for r in rows if r.get("part") == "TRUNCATED"]
    if not rows:
        print("no TRUNCATED rows found"); return
    print(f"{'block':>5s} {'r':>4s} {'rankB':>6s} {'k(rule)':>8s} {'gap_dec':>8s} {'conds':>6s} "
          f"{'member':>10s} {'nonmember':>10s} {'orders':>7s} {'scrAUC':>7s} {'verdict':>10s}")
    for r in sorted(rows, key=lambda z: (z["block"], z["r"])):
        M, U = r["N"], r.get("n_nonmember", 64)
        k, gapdec = choose_k(r["spectrum_rel"])
        row = next((z for z in r["per_k"] if z["k"] == k), None)
        if row is None:
            k = min(max(1, k), r["r"] - 1); row = next(z for z in r["per_k"] if z["k"] == k)
        if not r.get("exact_certificate_vacuous", False):
            verdict = "VOID:alive"                       # premise of the test is that the exact certificate is 0
        elif M < 5 or U < 5:
            verdict = "VOID:n"
        elif row.get("scrambled_auc", 0.0) >= 0.9:
            verdict = "VOID:scr"                          # the matched-spectrum control separates too (7e's arm)
        else:
            orders = math.log10(row["nonmember_median"] / max(row["member_median"], 1e-300))
            perfect = (row["auc"] >= 1.0)                 # AUC 1.0 == every member below every non-member
            verdict = "SUCCESS" if (perfect and orders >= 2) else "FAIL"
        orders = math.log10(row["nonmember_median"] / max(row["member_median"], 1e-300))
        chance = 1.0 / math.comb(M + U, M)
        print(f"{r['block']:5d} {r['r']:4d} {r['rank_B_T']:6d} {k:8d} {gapdec:8.2f} {row['conditions']:6d} "
              f"{row['member_median']:10.2e} {row['nonmember_median']:10.2e} {orders:7.2f} "
              f"{row.get('scrambled_auc', float('nan')):7.3f} {verdict:>10s}"
              + ("" if verdict != "SUCCESS" else f"   (chance {chance:.1e})"))
    print("\nk is chosen by the FIXED rule (largest log-spectrum gap), never by the best AUC. "
          "The rows' own best_k field is a fit and is not used here.")


if __name__ == "__main__":
    main()
