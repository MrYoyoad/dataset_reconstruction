#!/usr/bin/env python3
"""Self-test: does the shared module reproduce the committed E1B numbers exactly?

A shared tool handed to another lane must be validated against a row that is already in the repo, or it is just an
untested rewrite with more users. Targets, from jobs 350928/350940 (committed f5eb9b3):

    free H, seed known   512 unknowns, rank 512,  nullity 0
    free H, seed free   2048 unknowns, rank 1816, nullity 232
    chart k=12, free    1632 unknowns, rank 1632, nullity 0

  python -m experiments.utils.test_identifiability
"""
import torch
from experiments.e1b.e1b_tiny import build_release, replay
from experiments.utils.identifiability import identifiability, assert_chart_contains, block_normalised_residual

torch.set_default_dtype(torch.float64)

EXPECTED = {"free_H_seed_known": (512, 512, 0), "free_H_seed_free": (2048, 1816, 232), "chart_k12_seed_free": (1632, 1632, 0)}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = build_release(12, 8, 24, 20, 64, 20, 0.05, 1, dev)
    H, A0, W0, y, A_T, B_T, L, b = R["H"], R["A0"], R["W0"], R["y"], R["A_T"], R["B_T"], R["L"], R["b"]
    d, N = H.shape

    def sim(Hc, A0c):
        A_s, B_s = replay(Hc, A0c, W0, y, 20, 20, 0.05)
        return (B_s, A_s)

    res = block_normalised_residual(sim, (B_T, A_T))          # convention 2
    W_true = torch.linalg.lstsq(L, H - b[:, None]).solution
    chart = lambda W: L @ W + b[:, None]
    rel = assert_chart_contains(chart, W_true, H)             # convention 5
    print(f"# chart contains the truth to {rel:.2e}")

    cases = {
        "free_H_seed_known":   (lambda Hc: res(Hc, A0), (H,)),
        "free_H_seed_free":    (res, (H, A0)),
        "chart_k12_seed_free": (lambda W, A0c: res(chart(W), A0c), (W_true, A0)),
    }
    ok = True
    for name, (fn, params) in cases.items():
        r = identifiability(fn, params)                        # conventions 1 and 4 (relative floor is safe here)
        exp_u, exp_r, exp_n = EXPECTED[name]
        hit = (r.n_unknowns, r.rank, r.nullity) == (exp_u, exp_r, exp_n)
        ok &= hit
        print(f"{name:22s} unknowns {r.n_unknowns:5d} rank {r.rank:5d} nullity {r.nullity:4d}   "
              f"expected {exp_u}/{exp_r}/{exp_n}   {'MATCH' if hit else '*** MISMATCH ***'}   "
              f"cond {r.condition_number:.3e}  gap at cut {r.gap_at_cut:.1e}")
    print(f"\n# SELF-TEST {'PASSED' if ok else 'FAILED'}: the shared module "
          f"{'reproduces' if ok else 'does NOT reproduce'} the committed E1B rows.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
