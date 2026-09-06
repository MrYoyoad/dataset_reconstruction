#!/usr/bin/env python3
"""Read the head-to-head rows and classify each linearised cell, using the project's EXISTING verdict threshold.

Every quantity here is already in the emitted rows; nothing is recomputed from tensors and no new threshold is
introduced. Per the threshold-scoping ruling (R12): report residual-over-floor as a RAW RATIO on every row, which
makes any bar irrelevant when it is 1e13; use the project's existing 1.5 for the verdict so there is one threshold
in the repo; and mark anything between the bar and ten times it UNDETERMINED rather than classifying it.

  alias           residual AT the model floor with wrong images -> an identifiability failure
  search failure  residual ABOVE the model floor                -> the solve did not converge
  UNDETERMINED    ratio in [1.5, 15]                            -> not classified

  python -m experiments.cifar.read_ntk_vs_cert [results/ntk_vs_cert/*.jsonl]
"""
import glob, json, sys

BAR, BAND = 1.5, 10.0          # 1.5 is derived_verdict.py's, pinned over the corpus; BAND*BAR is the upper edge


def verdict(found, N, res_min, floor):
    if found >= N: return "recovered", float("nan")
    ratio = res_min / max(floor, 1e-300)
    if ratio <= BAR: return "alias (residual AT the floor, wrong images)", ratio
    if ratio <= BAR * BAND: return "UNDETERMINED", ratio
    return "search failure (residual ABOVE the floor)", ratio


def main():
    pats = sys.argv[1:] or ["results/ntk_vs_cert/*.jsonl"]
    rows = [json.loads(l) for p in pats for f in sorted(glob.glob(p)) for l in open(f) if l.strip()]
    rows = [r for r in rows if r.get("part") == "ntk_vs_cert"]
    if not rows: print("# no rows yet"); return
    print(f"# {len(rows)} cells\n")
    hdr = ("| dataset | class | chart | k | T | arm | images | residual | model floor | residual/floor | verdict "
           "| certificate images | cert landed | top-20 |")
    print(hdr); print("|" + "---|" * 14)
    for r in sorted(rows, key=lambda r: (r["dataset"], r["class_name"], r["chart"], r["k"], r["T"])):
        for arm, v in sorted(r.get("ntk_by_form", {}).items()):
            fl = (v.get("model_floor_at_truth") or 0.0)
            vd, ratio = verdict(v["images_found"], r["N"], v["residual_min"], fl)
            diag = " [DIAGNOSTIC UPPER BOUND, not an attack]" if v.get("is_diagnostic_upper_bound") else ""
            print(f"| {r['dataset']} | {r['class_name']} | {r['chart']} | {r['k']} | {r['T']} | {arm}{diag} | "
                  f"{v['images_found']}/{r['N']} | {v['residual_min']:.2e} | {fl:.2e} | "
                  f"{'--' if ratio != ratio else f'{ratio:.1e}'} | {vd} | {r['cert_images_found']}/{r['N']} | "
                  f"{r['cert_landed_starts']}/{r['starts']} | {r['cert_top20_landed']}/20 |")
    print("\n# The certificate columns repeat per arm because every arm of a cell shares one release, one chart and one set of starts.")


if __name__ == "__main__":
    main()
