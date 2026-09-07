#!/usr/bin/env python3
"""Does a relative-only rank threshold lie on this release? Checked, not assumed.

The hazard (raised by the multilayer lane after it inverted a result): `s > rtol * s[0]` calls a numerically ZERO
matrix FULL rank, because a zero matrix's own s[0] is already at rounding level so every singular value clears the
bar. It reads maximally wrong exactly when the true answer is zero. Every rank in the E1B scripts uses that form —
`matrix_rank(C, rtol=1e-10)`, `matrix_rank(H, rtol=1e-10)`, and `qb = (S_b > 1e-10*S_b[0]).sum()` on B_T — so the
question is whether any of those matrices can legitimately vanish on this release, and whether the cut sits in a
real spectral gap or in noise.

Compares each rank against an ABSOLUTE floor tied to the scale of the matrix the object was built from, and prints
the spectrum and the gap ratio at the cut so the answer is inspectable rather than asserted.

  python -m experiments.e1b.rank_threshold_check
"""
import torch
from experiments.e1b.e1b_tiny import build_release
from experiments.exact_inversion.certificate import certificate

torch.set_default_dtype(torch.float64)


def main():
    dev = torch.device("cpu")
    R = build_release(12, 8, 24, 20, 64, 400, 0.05, 1, dev)
    H, A0, B_T, A_T = R["H"], R["A0"], R["B_T"], R["A_T"]
    C, _, _ = certificate(A_T, B_T, tol=1e-12)
    print("# release: affine two-routes (331384 generator), d=64 N=8 r=24 m=20 T=400 seed=1, fp64\n")
    ok = True
    for name, M, ref_name, ref in (("B_T", B_T, "A_T", A_T), ("C", C, "A_T", A_T),
                                   ("H", H, "H", H), ("A_T", A_T, "A_T", A_T)):
        s = torch.linalg.svdvals(M)
        nrm = float(torch.linalg.norm(ref))
        rel = int((s > 1e-10 * float(s[0])).sum())
        ab = int((s > 1e-10 * nrm).sum())
        agree = rel == ab
        ok &= agree
        print(f"{name}: ||{name}|| = {float(torch.linalg.norm(M)):.4e}   s[0] = {float(s[0]):.4e}   "
              f"s[-1] = {float(s[-1]):.4e}   (absolute floor 1e-10*||{ref_name}|| = {1e-10 * nrm:.3e})")
        print(f"   rank RELATIVE (s > 1e-10*s0) = {rel}   rank ABSOLUTE (s > 1e-10*||{ref_name}||) = {ab}   "
              f"AGREE = {agree}")
        print("   spectrum: " + "  ".join(f"{float(v):.2e}" for v in s[:min(12, len(s))]))
        if 0 < rel < len(s):
            print(f"   gap at the cut: s[{rel-1}] = {float(s[rel-1]):.3e} -> s[{rel}] = {float(s[rel]):.3e}   "
                  f"ratio {float(s[rel-1] / s[rel]):.2e}")
        print()
    print(f"# VERDICT: every rank agrees under both thresholds = {ok}. "
          f"{'The relative form is not lying on this release.' if ok else 'A RANK DIFFERS -- the relative form is unsafe here.'}")
    print("# Note the hazard is real but inapplicable here: none of these matrices can vanish on this release "
          "(B_T is trained away from its zero init, H is data, A_T is the released seed-plus-update).")


if __name__ == "__main__":
    main()
