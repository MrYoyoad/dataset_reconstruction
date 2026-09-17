#!/usr/bin/env python3
"""The two walls, on the SAME axis, on the REAL releases: is there any chart width that satisfies both?

Two constraints on the chart dimension k pull in opposite directions, and until now they have only been stated in
different settings:

  IDENTIFIABILITY caps k from ABOVE. Measured exactly on the synthetic release (job 350967, 11/11 cells including
    a one-unit step): the solution family has dimension N * max(0, k - (m + r - N - 1)), so the truth stops being
    identifiable once k exceeds the bracket. That bracket is the glossary's capacity line k < m + r - N, now with
    a meaning rather than a heuristic. On the CIFAR releases (m = 11, r = 64, N = 8) the cap is k <= 66.

  FIDELITY pushes k from BELOW. The chart must CONTAIN the private data, and containment improves with k. The
    oracle ladder measured what containment is needed for the attack to land: a projection error at or below
    0.0124 (motorcycle/MLP) or 0.0045 (keyboard/CNN).

If the projection error of an ATTACKER-BUILDABLE chart has not fallen to the gate by k = 66, no chart width
satisfies both walls and the attack cannot be made to work on these releases by choosing k. That is the question,
and both numbers here are measured on the same axis, in the same space, for the same eight private photographs --
which is the point: the walls must share a row rather than be reconciled across constructions afterwards.

The chart is the one the ladder actually uses: the top-k PCA of PUBLIC train images OF THE ADDED CLASS. Public
data only, disjoint from the eight held-out privates. Not an oracle chart.

  python -m experiments.e1b.two_walls
"""
import argparse, json, os, socket, sys
import numpy as np
import torch

from experiments.cifar.cifar_newclass import load_cifar100_class
from experiments.exact_inversion.lora_exact_inversion import git_hash

torch.set_default_dtype(torch.float64)

EXAMPLES = {
    "mlp_motorcycle": dict(cls="motorcycle", gate=0.0124, note="MLP release; ladder last all-8 landing at 0.0124"),
    "cnn_keyboard":   dict(cls="keyboard",   gate=0.0045, note="CNN release; ladder last landing at 0.0045"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8); ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--m", type=int, default=11, help="head width: CIFAR-10 plus the added class")
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[4, 8, 16, 24, 32, 48, 64, 66, 80, 96, 128, 192, 256, 384, 512])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results/e1b/two_walls.jsonl")
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cap = a.m + a.r - a.N - 1
    print(f"# IDENTIFIABILITY cap: k <= m + r - N - 1 = {a.m} + {a.r} - {a.N} - 1 = {cap}")
    print(f"# (formula confirmed 11/11 on the synthetic release, job 350967, including the one-unit step)\n")

    out_rows = []
    for name, ex in EXAMPLES.items():
        pool, cname = load_cifar100_class(a.data_root, ex["cls"])
        Pub = torch.tensor(pool["train"], dtype=torch.float64)     # public: train split of the added class
        g = np.random.RandomState(a.seed)
        Pri = torch.tensor(pool["test"], dtype=torch.float64)      # privates: held-out test split
        idx = g.permutation(len(Pri))[: a.N]
        X = Pri[idx]                                              # (N, 3072)
        mu = Pub.mean(0)
        U, S, Vh = torch.linalg.svd(Pub - mu, full_matrices=False)
        print(f"## {name}  ('{cname}')   public pool {tuple(Pub.shape)}, privates {tuple(X.shape)}   "
              f"gate {ex['gate']}")
        print(f"{'k':>5}  {'proj err (mean)':>16}  {'range':>17}  {'<= gate?':>9}  {'<= ident cap?':>13}")
        rows = []
        for k in a.ks:
            if k > Vh.shape[0]: continue
            V = Vh[:k].T
            R = (X - mu) - (X - mu) @ V @ V.T
            err = (torch.linalg.norm(R, dim=1) / torch.linalg.norm(X, dim=1))
            e = float(err.mean())
            ok_fid = e <= ex["gate"]; ok_id = k <= cap
            print(f"{k:>5}  {e:>16.4f}  {float(err.min()):>7.4f}-{float(err.max()):<8.4f}  "
                  f"{'YES' if ok_fid else 'no':>9}  {'YES' if ok_id else 'NO':>13}")
            rows.append(dict(k=k, proj_err_mean=e, proj_err_min=float(err.min()), proj_err_max=float(err.max()),
                             meets_fidelity_gate=bool(ok_fid), within_identifiability_cap=bool(ok_id),
                             both=bool(ok_fid and ok_id)))
        both = [r for r in rows if r["both"]]
        at_cap = [r for r in rows if r["k"] <= cap]
        best_at_cap = min(at_cap, key=lambda r: r["proj_err_mean"]) if at_cap else None
        print(f"  -> widths satisfying BOTH walls: {len(both)}" + (f"  {[r['k'] for r in both]}" if both else ""))
        if best_at_cap:
            print(f"  -> best fidelity available under the identifiability cap (k={best_at_cap['k']}): "
                  f"{best_at_cap['proj_err_mean']:.4f}, which is {best_at_cap['proj_err_mean']/ex['gate']:.1f}x "
                  f"the gate of {ex['gate']}\n")
        out_rows.append(dict(example=name, cls=cname, gate=ex["gate"], identifiability_cap=cap,
                             m=a.m, r=a.r, N=a.N, n_public=int(Pub.shape[0]), cells=rows,
                             widths_satisfying_both=[r["k"] for r in both],
                             best_under_cap=best_at_cap, note=ex["note"]))
    verdict = all(len(r["widths_satisfying_both"]) == 0 for r in out_rows)
    print(f"# VERDICT: {'NO chart width satisfies both walls on either release.' if verdict else 'a window EXISTS -- see rows.'}")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(dict(part="two_walls", precision="fp64", chart="public PCA of the added class "
                                 "(attacker-buildable, disjoint from the privates)", examples=out_rows,
                                 no_window=bool(verdict), git=git_hash(), host=socket.gethostname(),
                                 cmd=" ".join(sys.argv))) + "\n")


if __name__ == "__main__":
    main()
