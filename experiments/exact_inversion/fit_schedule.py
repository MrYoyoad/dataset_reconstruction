#!/usr/bin/env python3
"""Recover the SCHEDULE PARAMETERS from probed per-step learning rates (global fit).

Why this exists as a committed script: the result was first obtained in an ad-hoc inline re-analysis, which
left no artefact — an audit correctly refused to dagger it. This is that procedure, committed, writing a
result row like every other measurement in the study.

Input: the `etas_hat` arrays already recorded by `schedule_probe.py` (the per-step learning rates the
attacker reads off by continuing training on their OWN data; recovered to ~1e-15). Those are exact; the
open question was whether the schedule PARAMETERS (base rate, schedule length, and the step count T at
which the release was taken) can be recovered from them.

A gradient fit of this lands 30-40% off — cosine fitting is multimodal, and that failure is a local
minimum, not a degeneracy. So fit GLOBALLY: eliminate the base rate by ratios, grid over (t, T_max), and
solve the base in closed form at each grid point.

  python -m experiments.exact_inversion.fit_schedule --out results/exact_inversion/step34_schedfit.jsonl
"""
import argparse, glob, json, math, socket, sys
import numpy as np

from experiments.exact_inversion.lora_exact_inversion import git_hash


def fit_cosine(e, tmax_hi=2000, t_grid=400):
    """eta(t) = base/2 (1 + cos(pi t / T_max)).  Global grid over (t, T_max); base in closed form."""
    e = np.asarray(e, dtype=np.float64); W = len(e); best = None
    for Tmax in range(50, tmax_hi + 1):
        step = max(1, Tmax // t_grid)
        for t in range(0, Tmax, step):
            pred = 0.5 * (1.0 + np.cos(np.pi * (t + np.arange(W)) / Tmax))
            d = float(pred @ pred)
            if d <= 0: continue
            base = float(pred @ e) / d
            r = float(np.linalg.norm(base * pred - e))
            if best is None or r < best[0]: best = (r, t, Tmax, base)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/exact_inversion/step30_sched2_*.jsonl",
                    help="probe output holding etas_hat (job 488314 by default)")
    ap.add_argument("--tmax-hi", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for f in sorted(glob.glob(a.glob)) for l in open(f)]
    rows = [d for d in rows if d.get("schedule") == "cosine" and len(d.get("etas_hat", [])) >= 4]
    if not rows:
        print(f"# no cosine probe rows in {a.glob}"); return
    print(f"# global schedule fit from probed per-step eta  ({len(rows)} rows)  git={git_hash()} "
          f"host={socket.gethostname()}", flush=True)
    print(f"{'win':>4} {'T true':>7} {'T hat':>7} {'Tmax true':>10} {'Tmax hat':>9} "
          f"{'base true':>10} {'base hat':>10} {'rel errs (T, Tmax, base)':>30}")
    for d in rows:
        e = d["etas_hat"]; r, t, Tmax, base = fit_cosine(e, a.tmax_hi)
        out = dict(source_glob=a.glob, probe_window=len(e), schedule="cosine",
                   T_true=d["T"], T_hat=t, T_max_true=d["T_max"], T_max_hat=Tmax,
                   base_true=d["lr_base"], base_hat=base,
                   T_rel_err=abs(t - d["T"]) / d["T"],
                   T_max_rel_err=abs(Tmax - d["T_max"]) / d["T_max"],
                   base_rel_err=abs(base - d["lr_base"]) / d["lr_base"],
                   fit_residual=r, method="global grid over (t,T_max), base in closed form",
                   git=git_hash(), host=socket.gethostname(), cmd=" ".join(sys.argv))
        print(f"{len(e):>4} {d['T']:>7} {t:>7} {d['T_max']:>10} {Tmax:>9} "
              f"{d['lr_base']:>10.6f} {base:>10.6f} "
              f"{out['T_rel_err']:>10.1e} {out['T_max_rel_err']:>9.1e} {out['base_rel_err']:>9.1e}", flush=True)
        if a.out:
            with open(a.out, "a") as f: f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
