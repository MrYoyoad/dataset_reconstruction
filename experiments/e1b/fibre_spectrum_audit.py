#!/usr/bin/env python3
"""AUDIT: is the reported NULLITY 0 real, or an artefact of a RELATIVE rank threshold?

fibre_dimension_check.py counts rank as `S > 1e-10 * S[0]`. A relative cut can place a genuinely zero
singular value just ABOVE the threshold through roundoff, OVERSTATING rank and UNDERSTATING nullity --
so it can report nullity 0 when the true nullity is positive. The 232 result was safe because it came
with a twelve-order gap; the nullity-0 cells were reported with no gap at all.

This measures, for each chart cell, the SMALLEST RETAINED singular value relative to the largest. If it
sits far above the 1e-10 cut the zero is solid; if it sits near the cut the zero is an artefact and the
cell must be recomputed against an absolute floor tied to the Jacobian's scale.

Records git (with -dirty) and its own SHA-256 in every row: a row must be self-describing when quoted.
"""
import json, hashlib, subprocess, os, torch, torch.func as tfn
from experiments.e1b.e1b_tiny import build_release, replay

torch.set_default_dtype(torch.float64)


def _prov():
    try: g = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception: g = "?"
    try: g += "-dirty" if subprocess.call(["git", "diff", "--quiet", "HEAD"]) else ""
    except Exception: pass
    try: sh = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
    except Exception: sh = "?"
    return dict(git=g, script_sha256=sh)


def spectrum(fn, args, label, cut=1e-10):
    J = tfn.jacrev(fn, argnums=tuple(range(len(args))))(*args)
    J = torch.cat([j.reshape(-1, a.numel()) for j, a in zip(J, args)], 1) if isinstance(J, tuple) \
        else J.reshape(-1, args[0].numel())
    S = torch.linalg.svdvals(J)
    s0 = float(S[0]); rel = (S / s0)
    rk = int((rel > cut).sum())
    smallest_retained = float(rel[rk - 1]) if rk else float("nan")
    first_dropped = float(rel[rk]) if rk < len(rel) else 0.0
    return dict(cell=label, shape=list(J.shape), rank=rk, nullity=int(J.shape[1] - rk),
                cut=cut, smallest_retained_rel=smallest_retained, first_dropped_rel=first_dropped,
                margin_orders=(None if smallest_retained != smallest_retained
                               else float(torch.log10(torch.tensor(smallest_retained / cut)))),
                gap_orders=(None if first_dropped <= 0 else
                            float(torch.log10(torch.tensor(smallest_retained / first_dropped)))))


def main():
    dev = "cpu"
    k, N, r, m, P, T, lr, seed = 12, 8, 24, 20, 64, 40, 0.05, 1
    R = build_release(k, N, r, m, P, T, lr, seed, dev)
    H, A0, W0, y, A_T, B_T = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"]
    L, b = R["L"], R["b"]                      # ORACLE chart: the release's own generator
    W_true = torch.linalg.lstsq(L, H - b[:, None]).solution

    def res_full(Hc, A0c):
        As, Bs = replay(Hc, A0c, W0, y, m, T, lr)
        return torch.cat([((Bs - B_T) / B_T.norm()).reshape(-1), ((As - A_T) / A_T.norm()).reshape(-1)])

    rows = []
    rows.append(spectrum(lambda W: res_full(L @ W + b[:, None], A0), (W_true.clone(),),
                         "CHART k=12, seed KNOWN, full A_T"))
    rows.append(spectrum(lambda W, A0c: res_full(L @ W + b[:, None], A0c), (W_true.clone(), A0.clone()),
                         "CHART k=12, seed FREE, full A_T"))
    rows.append(spectrum(lambda Hc, A0c: res_full(Hc, A0c), (H.clone(), A0.clone()),
                         "NO CHART, seed FREE, full A_T  (the 100/232 control)"))
    print("# Is NULLITY 0 real, or an artefact of the relative cut at 1e-10?")
    print("# %-46s %-12s %-7s %-22s %s" % ("cell", "shape", "nullity", "smallest retained/S0", "orders above cut"))
    for rw in rows:
        rw.update(_prov())
        print("  %-46s %-12s %-7d %-22.3e %s" % (rw["cell"], "x".join(map(str, rw["shape"])), rw["nullity"],
              rw["smallest_retained_rel"], ("%.1f" % rw["margin_orders"]) if rw["margin_orders"] else "-"))
    os.makedirs("results/e1b", exist_ok=True)
    with open("results/e1b/fibre_spectrum_audit.jsonl", "a") as f:
        for rw in rows: f.write(json.dumps(rw) + "\n")


if __name__ == "__main__":
    main()
