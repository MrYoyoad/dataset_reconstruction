"""Read-side verdict repair for drift_cert rows (LESSONS 2026-09-18).

`drift_cert.py:407` writes the CELL-level hypothesis flag into every image's verdict
(`if tg["contaminated"]: verdict = "contaminated"`), which overwrites the real outcome: a cell that recovered 7 of 8
images reports `{'contaminated': 8}`.  The counts (`landed`, `images_found`, `best_err`) are unaffected, and every
field the verdict is computed from is on the row, so the outcome is recoverable without re-running anything.

This script recomputes the per-image verdict from the saved fields with the SAME rule as the harness, minus the
override, and keeps `contaminated` where it belongs: a CONDITION on the row, reported alongside the outcome.

    python -m experiments.multilayer_cert.relabel_drift_verdicts results/multilayer_cert/drift_cert_p3_*.jsonl
    python -m experiments.multilayer_cert.relabel_drift_verdicts --write-suffix _relabelled <files>

Reads only; `--write-suffix` emits a copy with `verdict_relabelled` per image and `verdicts_relabelled` per solve.
The original rows are never modified (provenance: the harness wrote them, this only reads).
"""
import argparse, glob, json, sys
from collections import Counter

LAND = 1e-2          # same tier-1 bar as the harness


def verdict_of(p):
    """The harness's rule WITHOUT the contaminated override, from fields saved on the row."""
    if bool(p.get("landed")):                                   return "recovered"
    if float(p.get("best_err", 1e9)) < LAND:                    return "recovered"
    if bool(p.get("reached_opt")):
        return "recovered" if float(p.get("err_opt_truth", 1e9)) < LAND else "chart-limited"
    if bool(p.get("alias_in_chart")) or int(p.get("n_alias_starts", 0)) > 0:
        return "alias (residual zero, wrong image)"
    return "optimisation failure (residual not zero)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--write-suffix", default=None, help="write <stem><suffix>.jsonl with the repaired labels")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    paths = [p for pat in a.files for p in sorted(glob.glob(pat))]
    if not paths:
        print("no files matched", file=sys.stderr); return 2
    grand, changed_rows = Counter(), 0
    for path in paths:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        out = []
        for r in rows:
            touched = False
            for kind, s in (r.get("solves") or {}).items():
                per = s.get("per_image") or []
                if not per: continue
                new = [verdict_of(p) for p in per]
                if any(n != p.get("verdict") for n, p in zip(new, per)): touched = True
                for n, p in zip(new, per): p["verdict_relabelled"] = n
                s["verdicts_relabelled"] = dict(Counter(new))
                grand.update(Counter(new))
                if not a.quiet and any(n != p.get("verdict") for n, p in zip(new, per)):
                    tag = f"{path.split('/')[-1]} T={r.get('T')} lr={r.get('lr')} r_lower={(r.get('r_per_layer') or ['?'])[0]} {kind}"
                    print(f"{tag:78s} was {dict(Counter(p.get('verdict') for p in per))} -> {dict(Counter(new))}"
                          f"{'   [contaminated cell]' if r.get('contaminated') else ''}")
            changed_rows += bool(touched)
            out.append(r)
        if a.write_suffix:
            dst = path[:-6] + a.write_suffix + ".jsonl"
            with open(dst, "w") as f:
                for r in out: f.write(json.dumps(r) + "\n")
            print(f"# wrote {dst}")
    print(f"\n# rows with a changed label: {changed_rows}\n# relabelled outcome totals: {dict(grand)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
