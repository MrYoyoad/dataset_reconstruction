"""T5.2 counterexample attempt: does the NESTING of row(M_l) break generic additivity?

T5.2 claims  rank J_F = min(k_1, sum_l q_l)  with q_l = min(r_l - N, rank M_l), a.s. over independent
A_{l,0}. The upper bound is right. The EQUALITY assumes the per-layer row spaces spread independently
inside row(M_1).

But T5.1's own nesting says row(M_1) >= row(M_2) >= ... , and when q_l = rank M_l the layer's
contribution is row(M_l) EXACTLY -- so two deep layers sharing the same small row space contribute the
same subspace twice and add nothing. T5.2 would count them twice.

Predicted corrected law:   rank J_F = min_j ( d_j + sum_{l<j} q_l ),  d_j := rank M_j, d_{L+1} := 0.
"""
import numpy as np, json, sys


def run(seed, k=10, d1=10, deep=3, r=13, N=8, L=3):
    rng = np.random.default_rng(seed)
    n = 40
    # M_1 of rank d1; deeper layers pushed through a map of rank `deep` so row(M_l) nests strictly
    M1 = rng.normal(size=(n, k)); U, S, Vt = np.linalg.svd(M1, full_matrices=False)
    S[d1:] = 0; M1 = (U * S) @ Vt
    P = rng.normal(size=(n, n)); Up, Sp, Vtp = np.linalg.svd(P); Sp[deep:] = 0
    Jpsi = (Up * Sp) @ Vtp                      # rank-`deep` layer map
    Ms = [M1] + [np.linalg.matrix_power(Jpsi, i) @ M1 for i in range(1, L)]
    blocks, qs, ds = [], [], []
    for l, M in enumerate(Ms):
        A0 = rng.normal(size=(r, n))
        X = rng.normal(size=(n, N))             # the leaked span, N directions
        Q, _ = np.linalg.qr(A0 @ X)             # col(X_l) in R^r
        Ct = A0 - Q @ (Q.T @ A0)                # P_{col(X)^perp} A_0, rank r-N
        blk = Ct @ M
        blocks.append(blk)
        qs.append(np.linalg.matrix_rank(blk, tol=1e-8))
        ds.append(np.linalg.matrix_rank(M, tol=1e-8))
    J = np.vstack(blocks)
    actual = int(np.linalg.matrix_rank(J, tol=1e-8))
    k1 = ds[0]
    t52 = min(k1, sum(qs))
    corrected = min([ds[j] + sum(qs[:j]) for j in range(L)] + [sum(qs)])
    return dict(seed=seed, d=ds, q=qs, k1=k1, T52_predicts=int(t52),
                corrected_predicts=int(corrected), ACTUAL=actual,
                T52_ok=bool(t52 == actual), corrected_ok=bool(corrected == actual))

if __name__ == "__main__":
    print("# T5.2 counterexample: nested row spaces, deep layers confined to a rank-3 subspace")
    print("# %-6s %-14s %-12s %-5s %-9s %-11s %-7s %s" % ("seed","d=rank M_l","q_l","k_1","T5.2 says","corrected","ACTUAL","T5.2 correct?"))
    agree_t52 = agree_corr = 0
    for s in range(12):
        r = run(s)
        agree_t52 += r["T52_ok"]; agree_corr += r["corrected_ok"]
        print("  %-6d %-14s %-12s %-5d %-9d %-11d %-7d %s" % (
            r["seed"], r["d"], r["q"], r["k1"], r["T52_predicts"], r["corrected_predicts"],
            r["ACTUAL"], "yes" if r["T52_ok"] else "NO"))
        pass
    print("\n# T5.2 formula correct in %d/12 seeds; corrected formula correct in %d/12" % (agree_t52, agree_corr))
