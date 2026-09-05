#!/usr/bin/env python3
"""The DERIVED four-cell verdict, added alongside the recorded one and never replacing it.

yoado-7e's disposition, after an audit found the same cell wrong in two harnesses in opposite directions:

    at_floor AND error small  ->  recovered            (the image came out AND the release is reproduced)
    at_floor AND error large  ->  alias                (the release is reproduced by the WRONG image)
    NOT at_floor AND small    ->  unverified-recovery  (the image came out; the residual never verified it)
    NOT at_floor AND large    ->  search-failure

The three-cell schema collapses `unverified-recovery` into `search-failure` in one harness and into `recovered` in
the other. Both are wrong, in opposite directions, and the honest label is its own cell.

**Nothing here rewrites history.** The recorded verdict column is preserved exactly as written; this computes the
derived verdict from the residual and error already on disk and reports **where the two disagree**. A recorded
`recovered` whose derived value is `unverified-recovery` is a row **verified by image only** — it still counts for
"the release carries the information", but it must never have been counted as attacker-realizable. **Those
divergences are the audit flag**, and the correction pass over them belongs to the claims and genuineness lanes,
not here.

  python -u -m experiments.exact_inversion.derived_verdict results/exact_inversion/*.jsonl
"""
import glob, json, sys

RECOVER_TOL = 1e-2        # relative image error below which an image counts as recovered
RESID_ZERO = 1e-28        # absolute fallback only, for rows with no achievability floor to compare against
FLOOR_FACTOR = None       # PENDING: yoado-b9 is pinning the factor. at_floor is "within this factor of the
                          # CELL'S OWN from-truth residual", not a fixed constant -- it matters because the fp32
                          # row sits at ~5x its achievable floor, right on the recovered/unverified boundary.
                          # Until it is pinned, cells that HAVE a floor are reported at several factors rather
                          # than scored at one I invented.
FACTORS = (2.0, 5.0, 10.0, 100.0)

ERR_KEYS = ["final_err_max", "err_vs_chart_truth_MAX", "final_err_matched_max", "err_median", "err_max"]
RES_KEYS = ["residual", "cert_objective", "final_objective", "fval"]


def pick(row, keys):
    for k in keys:
        if k in row and isinstance(row[k], (int, float)):
            return float(row[k]), k
    return None, None


SYNONYM = {"optimisation failure": "search-failure", "search-failure": "search-failure",
           "recovered": "recovered", "alias": "alias", "unverified-recovery": "unverified-recovery",
           "partial": "partial", "mode collapse": "mode collapse"}


def derived(at_floor, err):
    small = err < RECOVER_TOL
    if at_floor and small: return "recovered"
    if at_floor and not small: return "alias"
    if small: return "unverified-recovery"
    return "search-failure"


def main():
    pats = sys.argv[1:] or ["results/exact_inversion/*.jsonl"]
    files = sorted({f for p in pats for f in glob.glob(p)})
    n_rows = n_scored = 0
    agree = 0
    div = {}; indeterminate = []; floorful = []
    counts = {"recovered": 0, "alias": 0, "unverified-recovery": 0, "search-failure": 0}
    # the CELL's achievability floor: the from-truth companion solve's residual, keyed by (file, set, k, r)
    floors = {}
    for f in files:
        for line in open(f):
            try: rr = json.loads(line)
            except Exception: continue
            if "floor_objective_at_truth" in rr:
                floors[(f, rr.get("set"), rr.get("k"), rr.get("r"))] = float(rr["floor_objective_at_truth"])
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_rows += 1
            rec = r.get("verdict")
            if not isinstance(rec, str): continue
            e, ek = pick(r, ERR_KEYS); v, vk = pick(r, RES_KEYS)
            if e is None or v is None: continue
            n_scored += 1
            # at_floor must come from the ROW, never from my own absolute threshold. constrained_replay's floor is
            # `fval <= 1e-28 OR fval <= 100*fwd^2` -- a RELATIVE criterion my fallback cannot see, so scoring those
            # rows against the absolute one alone manufactures divergences that are artefacts of this script. A
            # row that does not carry at_floor (or the relative reference) is INDETERMINATE and is not scored.
            if "at_floor" in r:
                af = bool(r["at_floor"])
            elif "fwd_check" in r and isinstance(r["fwd_check"], (int, float)):
                af = bool(v <= RESID_ZERO or v <= 100 * float(r["fwd_check"]) ** 2)
            elif "res_at_truth" in r and isinstance(r["res_at_truth"], (int, float)):
                af = bool(v <= RESID_ZERO or v <= 100 * float(r["res_at_truth"]) ** 2)
            else:
                indeterminate.append((f.split("/")[-1], e, v))
                continue
            fl = floors.get((f, r.get("set"), r.get("k"), r.get("r")))
            if fl is not None and fl > 0:
                floorful.append((v, fl))
            d = derived(af, e)
            counts[d] = counts.get(d, 0) + 1
            head = rec.split(" (")[0].split(" --")[0].strip().lower()
            head_norm = SYNONYM.get(head, head)
            if head_norm == d:
                agree += 1
            else:
                div.setdefault(f"recorded '{head}' -> derived '{d}'", []).append((f.split("/")[-1], e, v))
    print(f"# {len(files)} files, {n_rows} rows, {n_scored} carrying BOTH an error and a residual field")
    print(f"# recorded and derived agree on {agree}; {len(indeterminate)} INDETERMINATE (the row carries no "
          f"at_floor and no relative-floor reference, so this script cannot decide and does not guess)\n")
    if not div:
        print("no divergences: the recorded verdicts are already the four-cell function of (at_floor, error)")
        return
    print("DIVERGENCES -- these are the audit flag, not a relabelling. The correction pass belongs to the")
    print("claims and genuineness lanes; nothing here rewrites a recorded verdict.\n")
    for k, rows in sorted(div.items(), key=lambda z: -len(z[1])):
        print(f"  {len(rows):5d}  {k}")
        for fn, e, v in rows[:3]:
            print(f"           e.g. {fn}: image error {e:.3e}, residual/objective {v:.3e}")
    # THREE metrics, never collapsed into one "recovered" number (yoado-7e).
    print(f"\n# THREE counts, from the same derived column:")
    print(f"#   information-carried  = recovered + unverified-recovery   {counts['recovered'] + counts['unverified-recovery']:5d}")
    print(f"#   attacker-claimable   = recovered + alias                 {counts['recovered'] + counts['alias']:5d}")
    print(f"#   verified-true        = recovered (the intersection)      {counts['recovered']:5d}")
    print(f"#   search-failure                                           {counts['search-failure']:5d}")
    print(f"# unverified-recovery counts toward information-carried and NEVER toward an attack success rate.")
    if floorful:
        print(f"\n# {len(floorful)} rows have a cell achievability floor; at_floor by factor (b9 pins the factor):")
        for fac in FACTORS:
            n_at = sum(1 for v, fl in floorful if v <= fac * fl)
            print(f"#   within {fac:6.1f}x of the cell's from-truth residual: {n_at:5d} of {len(floorful)}")
    print(f"\nA recorded 'recovered' whose derived value is 'unverified-recovery' is verified BY IMAGE ONLY.")
    print(f"It still counts for 'the release carries the information'. It must never have been counted as")
    print(f"attacker-realizable.")


if __name__ == "__main__":
    main()
